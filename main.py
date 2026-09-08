"""Sticker Plus plugin for KiraAI.

Ported from nori-core's nori_plugin_emoji to the KiraAI plugin model:

- ``send_emoji`` tool: the AI passes an emotion keyword, the plugin samples
  candidates (emotion LIKE match + random) and lets a VLM pick one by
  description; the image is returned as a ``ToolResult`` attachment which the
  AI sends via the built-in ``<file>`` tag.
- Steal hook (``@on.im_message``): market emojis / stickers sent by others
  are saved into the plugin-owned library and tagged by VLM in background.
- WebUI: management page (preview / edit / ban / upload / rescan / steal
  toggle) built as a single-file page under ``web/`` (noriflow-style).

Storage is a plugin-owned SQLite file under ``data/plugin_data/<id>/`` - the
host database is intentionally not touched, so deleting the plugin data
directory resets the library completely.

IMPORTANT: disable the built-in sticker plugin before enabling this one,
otherwise the AI sees two emoji channels.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from fastapi import File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

from core.agent.func_tool_manager import ToolResult  # re-exported by core.provider too
from core.chat import KiraMessageBatchEvent, KiraMessageEvent
from core.chat.message_elements import Image
from core.plugin import BasePlugin, PageMenu, PluginPage, on, register

from .db import EmojiDatabase
from .manager import EmojiManager
from .stealer import EmojiStealer
from .vlm import EmojiVLM

logger = logging.getLogger(__name__)

PLUGIN_ID = "kira-ai-plugin-sticker-plus"

EMOTION_KEYWORDS = (
    "胆怯、无语、调皮、开心、困惑、震惊、傲娇、害羞、温柔、委屈、"
    "期待、生气、无辜、撒娇、嫌弃、嘲讽、感谢、安慰、悲伤、欢迎"
)

USAGE_GUIDANCE = (
    "表情包使用规范（send_emoji 工具）：\n"
    "- 你可以在聊天中主动、自然地使用表情包，通常作为单独一条消息发送，或紧跟在文字之后点缀情绪\n"
    "- 保持克制：优先用文字回应，连续多轮都发表情会显得敷衍；一轮回复最多调用一次 send_emoji\n"
    "- 被@或被直接提问时，必须以文字回应为主，表情包只能作为补充\n"
    "- emotion 参数从固定情绪词表中选择最贴切当前语境的一个\n"
    f"- 可选情绪词：{EMOTION_KEYWORDS}"
)


class EmojiUpdateRequest(BaseModel):
    description: Optional[str] = None
    emotions: Optional[str] = None
    is_banned: Optional[bool] = None


class StealSettingRequest(BaseModel):
    steal_emoji: bool


class StickerPlusPlugin(BasePlugin):
    """Enhanced emoji library: VLM-selected sending + chat stealing + WebUI."""

    def __init__(self, ctx, cfg: dict):
        super().__init__(ctx, cfg)
        self._db: Optional[EmojiDatabase] = None
        self._manager: Optional[EmojiManager] = None
        self._stealer: Optional[EmojiStealer] = None

    # ------------------------------------------------------------------
    # Lifecycle (must be re-entrant: config updates re-run initialize())
    # ------------------------------------------------------------------

    async def initialize(self):
        # Tear down any previous incarnation first (hot reload safety).
        await self._shutdown_components()

        try:
            data_dir = Path(self.ctx.get_plugin_data_dir())
            emoji_dir = data_dir / "emojis"
            self._db = EmojiDatabase(data_dir / "emoji.db")
            await self._db.create_tables()

            vlm = EmojiVLM(client_resolver=self._resolve_vlm_client)
            self._manager = EmojiManager(
                db=self._db,
                vlm=vlm,
                emoji_dir=emoji_dir,
                capacity=self.plugin_cfg.get("capacity", 500),
                candidate_count=self.plugin_cfg.get("candidate_count", 9),
            )
            await self._manager.startup()

            self._stealer = EmojiStealer(
                emoji_manager=self._manager,
                max_size_mb=float(self.plugin_cfg.get("max_emoji_size_mb", 5.0)),
            )
            logger.info("Sticker Plus initialized (capacity=%d)", self._manager.capacity)
        except Exception:
            logger.exception("Sticker Plus initialization failed (plugin degrades to no-op)")
            await self._shutdown_components()

    async def terminate(self):
        await self._shutdown_components()

    async def _shutdown_components(self) -> None:
        if self._stealer is not None:
            try:
                await self._stealer.shutdown()
            except Exception:
                logger.warning("Error shutting down EmojiStealer", exc_info=True)
            self._stealer = None
        if self._manager is not None:
            try:
                await self._manager.shutdown()
            except Exception:
                logger.warning("Error shutting down EmojiManager", exc_info=True)
            self._manager = None
        if self._db is not None:
            try:
                await self._db.dispose()
            except Exception:
                logger.warning("Error disposing emoji database", exc_info=True)
            self._db = None

    def _resolve_vlm_client(self):
        """Configured model wins, else the host default VLM; None on failure."""
        model_uuid = str(self.plugin_cfg.get("vlm_model", "") or "").strip()
        try:
            if model_uuid:
                return self.ctx.get_llm_client(model_uuid=model_uuid)
            return self.ctx.provider_mgr.get_default_vlm()
        except Exception as exc:
            logger.error("Failed to resolve VLM client for emoji plugin: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Tool: send_emoji
    # ------------------------------------------------------------------

    @register.tool(
        name="send_emoji",
        description=(
            "发送一个表情包消息。根据当前聊天语境选择一个情绪关键词，"
            "表情包会由系统自动挑选并随消息发出。"
            f"emotion 可选值：{EMOTION_KEYWORDS}。"
            "保持克制地自然使用，不要每轮都调用；单独作为一条消息或在文字后点缀。"
        ),
        params={
            "type": "object",
            "properties": {
                "emotion": {
                    "type": "string",
                    "description": f"情绪关键词，从以下选择最贴切的一个：{EMOTION_KEYWORDS}",
                }
            },
            "required": ["emotion"],
        },
    )
    async def send_emoji(self, event: KiraMessageBatchEvent, *_, emotion: str):
        """Pick and return an emoji matching the emotion keyword."""
        if self._manager is None:
            return ToolResult(text="表情包库当前不可用。")

        emotion = str(emotion or "").strip()
        if not emotion:
            return ToolResult(text="缺少情绪关键词，未发送表情包。")

        recent_context = (event.message_str or "")[-500:]
        try:
            picked = await self._manager.pick_emoji(emotion, recent_context)
        except Exception as exc:
            logger.exception("send_emoji failed")
            return ToolResult(text=f"表情包选择失败：{exc}")

        if picked is None:
            return ToolResult(text="表情包库为空或暂无可用表情，未发送。")

        record, file_path = picked
        return ToolResult(
            text=(
                f"已选择表情包（描述：{record.description or '无'}；"
                f"情绪：{record.emotions or '无'}）。"
                "请用 <file type=\"image\"> 标签将它作为表情包消息发送，"
                "不要和其它标签混在同一条消息里。"
            ),
            attachments=[Image(image=str(file_path))],
        )

    # ------------------------------------------------------------------
    # Hooks
    # ------------------------------------------------------------------

    @on.llm_request()
    async def inject_emoji_guidance(self, event: KiraMessageBatchEvent, req, tag_set, *args, **kwargs):
        """Append emoji usage guidance to the tools prompt section."""
        if self._manager is None:
            return
        try:
            # Nothing selectable yet (empty library or everything pending):
            # guidance would only invite a doomed send_emoji call.
            if await self._manager.count_active() == 0:
                return
        except Exception:
            return
        for prompt in req.system_prompt:
            if prompt.name == "tools":
                prompt.content = f"{prompt.content}\n{USAGE_GUIDANCE}"
                return

    @on.im_message()
    async def steal_emoji(self, event: KiraMessageEvent, *args, **kwargs):
        """Observe-only hook: save stickers from incoming messages."""
        if self._stealer is None:
            return
        if not self.plugin_cfg.get("steal_emoji", True):
            return
        try:
            await self._stealer.handle_event(event)
        except Exception:
            logger.exception("Emoji steal hook failed")

    # ------------------------------------------------------------------
    # WebUI: management page
    # ------------------------------------------------------------------

    # The page must be mounted on a non-empty subpath, never "/":
    # - the WebUI menu store derives pageRoute from this route and falls back
    #   to "index" for empty routes, so the iframe would request
    #   /page/plugin/<id>/index which the StaticFiles mount cannot serve;
    # - a root mount is stored by Starlette Mount with the trailing slash
    #   stripped, which the host reload-cleanup prefix matching misses
    #   (stale mount shadows re-registration -> 404 after hot reload).
    # Same convention as the noriflow plugin's "/dashboard".
    @register.page(
        route="/dashboard",
        menu=PageMenu(
            label={"zh": "增强表情包", "en": "Sticker Plus"},
            icon="Picture",
            order=86,
        ),
    )
    def emoji_page(self):
        return PluginPage.from_folder("./web")

    # ------------------------------------------------------------------
    # WebUI: REST API
    # ------------------------------------------------------------------

    def _require_manager(self) -> EmojiManager:
        if self._manager is None:
            raise HTTPException(status_code=503, detail="Emoji library not initialized")
        return self._manager

    @register.api(method="GET", path="/emojis")
    async def list_emojis(
        self,
        status: str = "all",
        search: str = "",
        page: int = 1,
        page_size: int = 60,
    ):
        manager = self._require_manager()
        if status not in ("all", "active", "banned", "pending", "stolen"):
            status = "all"
        page = max(page, 1)
        page_size = min(max(page_size, 1), 200)
        items, total = await manager.list_emojis(
            status=status,
            search=search.strip(),
            offset=(page - 1) * page_size,
            limit=page_size,
        )
        return {"items": items, "total": total, "page": page, "page_size": page_size}

    @register.api(method="GET", path="/emojis/{emoji_id}/image")
    async def get_emoji_image(self, emoji_id: int):
        manager = self._require_manager()
        file_path = await manager.emoji_file(emoji_id)
        if file_path is None:
            raise HTTPException(status_code=404, detail="Emoji not found")
        return FileResponse(str(file_path))

    @register.api(method="PUT", path="/emojis/{emoji_id}")
    async def update_emoji(self, emoji_id: int, request: EmojiUpdateRequest):
        manager = self._require_manager()
        result = await manager.update_emoji(
            emoji_id,
            description=request.description,
            emotions=request.emotions,
            is_banned=request.is_banned,
        )
        if result is None:
            raise HTTPException(status_code=404, detail="Emoji not found")
        return result

    @register.api(method="DELETE", path="/emojis/{emoji_id}")
    async def delete_emoji(self, emoji_id: int):
        manager = self._require_manager()
        if not await manager.delete_emoji(emoji_id):
            raise HTTPException(status_code=404, detail="Emoji not found")
        return {"deleted": True}

    @register.api(method="PUT", path="/emojis/{emoji_id}/retag")
    async def retag_emoji(self, emoji_id: int):
        manager = self._require_manager()
        try:
            tagged = await manager.retag(emoji_id)
        except Exception as exc:
            logger.exception("Manual retag failed for #%s", emoji_id)
            raise HTTPException(status_code=500, detail=f"Retag failed: {exc}")
        if not tagged:
            raise HTTPException(status_code=404, detail="Emoji not found or file missing")
        return await manager.get_emoji(emoji_id)

    @register.api(method="POST", path="/emojis/upload")
    async def upload_emojis(self, files: list[UploadFile] = File(...)):
        manager = self._require_manager()
        added = 0
        skipped = 0
        for upload in files:
            try:
                data = await upload.read()
            except Exception as exc:
                logger.warning("Failed to read uploaded emoji: %s", exc)
                skipped += 1
                continue
            if not data:
                skipped += 1
                continue
            try:
                if await manager.add_emoji_from_bytes(data, "manual"):
                    added += 1
                else:
                    skipped += 1
            except Exception as exc:
                logger.warning("Failed to store uploaded emoji: %s", exc)
                skipped += 1
        return {"added": added, "skipped": skipped}

    @register.api(method="POST", path="/emojis/rescan")
    async def rescan(self):
        manager = self._require_manager()
        added = await manager.scan_directory()
        await manager.enforce_capacity()
        return {"added": added}

    @register.api(method="GET", path="/settings")
    async def get_settings(self):
        manager = self._require_manager()
        stats = await manager.stats()
        return {
            "steal_emoji": bool(self.plugin_cfg.get("steal_emoji", True)),
            "capacity": manager.capacity,
            "candidate_count": manager.candidate_count,
            "max_emoji_size_mb": float(self.plugin_cfg.get("max_emoji_size_mb", 5.0)),
            "vlm_model": str(self.plugin_cfg.get("vlm_model", "") or ""),
            "stats": stats,
        }

    @register.api(method="PUT", path="/settings/steal")
    async def set_steal(self, request: StealSettingRequest):
        """Toggle stealing; persists through the plugin config system.

        update_plugin_config re-initializes this plugin instance, so the
        response is built from locals only - do not touch self afterwards.
        """
        plugin_mgr = getattr(self.ctx, "plugin_mgr", None)
        if plugin_mgr is None or not plugin_mgr.has_plugin(PLUGIN_ID):
            raise HTTPException(status_code=503, detail="Plugin manager not available")
        updated = await plugin_mgr.update_plugin_config(
            PLUGIN_ID, {"steal_emoji": bool(request.steal_emoji)}
        )
        return {"steal_emoji": bool(updated.get("steal_emoji", request.steal_emoji))}
