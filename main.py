"""Sticker Plus plugin for KiraAI.

Ported from nori-core's nori_plugin_emoji to the KiraAI plugin model:

- ``<sticker_plus>情绪</sticker_plus>`` tag: the AI puts an emotion keyword in
  the tag; the plugin samples candidates (emotion LIKE match + random), lets a
  separate selection LLM pick one by description, and sends the picked sticker
  DIRECTLY via the adapter (async from the main model, so the AI's text is
  never delayed by the pick). Once the send settles, the step-result hook
  appends a ``<system_reminder>`` record (emoji id + description) to the
  assistant message, so the main AI's history contains what it just sent.
- Steal hook (``@on.im_message``): market emojis / stickers sent by others
  are saved into the plugin-owned library and tagged by VLM in background.
- WebUI: management page (preview / edit / ban / upload / rescan / steal
  toggle) built as a single-file page under ``web/`` (noriflow-style).

Storage is a plugin-owned SQLite file under ``data/plugin_data/<id>/`` - the
host database is intentionally not touched, so deleting the plugin data
directory resets the library completely.

NOTE: the built-in sticker plugin also injects a ``<sticker>`` tag; keep it
disabled so the AI sees a single emoji channel.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from fastapi import File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

from core.chat import KiraMessageBatchEvent, KiraMessageEvent, MessageChain
from core.chat.message_elements import Sticker
from core.logging_manager import get_logger
from core.plugin import BasePlugin, PageMenu, PluginPage, Priority, on, register
from core.tag import BaseTag, TagSet
from core.utils.common_utils import image_to_base64

from .db import EmojiDatabase
from .manager import EmojiManager
from .stealer import EmojiStealer
from .vlm import EmojiVLM

# Host get_logger: registers the name in the file-handler whitelist, so
# plugin lines actually reach data/log.log and the WebUI log console
# (plain logging.getLogger lines are filtered out in production).
logger = get_logger("sticker-plus", "green")

PLUGIN_ID = "kira-ai-plugin-sticker-plus"

EMOTION_KEYWORDS = (
    "胆怯、无语、调皮、开心、困惑、震惊、傲娇、害羞、温柔、委屈、"
    "期待、生气、无辜、撒娇、嫌弃、嘲讽、感谢、安慰、悲伤、欢迎"
)

# How long the step-result hook waits for the background pick-and-send task
# before giving up on attaching the record (the task itself keeps running and
# the sticker still goes out; only the history record is lost).
SEND_RECORD_WAIT_TIMEOUT = 10.0

# Pending records from turns that never reached the step-result hook (e.g. the
# event was stopped mid-send) are pruned after this many seconds.
PENDING_RECORD_MAX_AGE = 60.0

# Grace period on shutdown: in-flight background sends are awaited (not
# cancelled) for at most this long before components are torn down, so
# stickers already scheduled still go out without blocking shutdown forever.
BG_SEND_SHUTDOWN_GRACE = 15.0

STICKER_PLUS_TAG_DESCRIPTION = (
    "<sticker_plus>情绪</sticker_plus> # 发送一个表情包消息用于情绪表达，"
    "在闲聊、调侃、被调侃等场景推荐使用。填入当前语境最贴切的情绪关键词，"
    "表情包由系统自动挑选并随后发出，发送结果会以 <system_reminder> 记录附加在你的消息之后。"
    "可以和 <text> 放在同一个 <msg> 里（通常放在文字后面），也可以单独成条；"
    "同一条消息中最多使用一次。"
    f"可用的情绪关键词：{EMOTION_KEYWORDS}"
)


@dataclass
class _PendingStickerSend:
    """One scheduled background sticker send awaiting its history record."""

    done: asyncio.Event = field(default_factory=asyncio.Event)
    created_at: float = field(default_factory=time.monotonic)
    # ok=True -> sent, detail carries "编号N（描述）"; ok=False -> not sent,
    # detail carries the reason.
    ok: bool = False
    detail: str = ""

    def age_ok(self) -> bool:
        """False once the record is stale (its turn never reached the hook)."""
        return time.monotonic() - self.created_at < PENDING_RECORD_MAX_AGE

class EmojiUpdateRequest(BaseModel):
    description: Optional[str] = None
    emotions: Optional[str] = None
    is_banned: Optional[bool] = None


class StealSettingRequest(BaseModel):
    steal_emoji: bool


class StealApprovalSettingRequest(BaseModel):
    steal_require_approval: bool


class StickerPlusPlugin(BasePlugin):
    """Enhanced emoji library: VLM-selected sending + chat stealing + WebUI."""

    def __init__(self, ctx, cfg: dict):
        super().__init__(ctx, cfg)
        self._db: Optional[EmojiDatabase] = None
        self._manager: Optional[EmojiManager] = None
        self._stealer: Optional[EmojiStealer] = None
        # sid -> pending background sends awaiting their history record.
        self._pending_sends: dict[str, list[_PendingStickerSend]] = {}
        # Strong references so fire-and-forget send tasks are never GC'd.
        self._bg_sends: set[asyncio.Task] = set()

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

            vlm = EmojiVLM(
                client_resolver=self._resolve_vlm_client,
                selection_client_resolver=self._resolve_selection_client,
            )
            self._manager = EmojiManager(
                db=self._db,
                vlm=vlm,
                emoji_dir=emoji_dir,
                capacity=self.plugin_cfg.get("capacity", 500),
                candidate_count=self.plugin_cfg.get("candidate_count", 9),
                steal_require_approval=bool(
                    self.plugin_cfg.get("steal_require_approval", True)
                ),
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
        # Pending records die with this instance (hot reload boundary).
        self._pending_sends.clear()
        # Give in-flight background sends a bounded grace period instead of
        # cancelling them: a sticker already scheduled should still go out,
        # and its task keeps using the (snapshot) manager and the database
        # below, so those must not be disposed underneath it. Tasks that
        # exceed the grace period are left running (same design as above);
        # their pick/send may then fail against torn-down resources, which
        # the task's own exception handling records as a failed send.
        if self._bg_sends:
            done, still_running = await asyncio.wait(
                set(self._bg_sends), timeout=BG_SEND_SHUTDOWN_GRACE
            )
            if still_running:
                logger.warning(
                    "%d sticker_plus background send(s) still running after %.0fs grace; "
                    "continuing shutdown",
                    len(still_running), BG_SEND_SHUTDOWN_GRACE,
                )
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
        """Tagging client: needs vision. Configured model wins, else the
        host default VLM; None on failure (tagging retries later)."""
        model_uuid = str(self.plugin_cfg.get("vlm_model", "") or "").strip()
        try:
            if model_uuid:
                return self.ctx.get_llm_client(model_uuid=model_uuid)
            return self.ctx.provider_mgr.get_default_vlm()
        except Exception as exc:
            logger.error("Failed to resolve tagging VLM client for emoji plugin: %s", exc)
            return None

    def _resolve_selection_client(self):
        """Selection client: text-only choice task, no vision needed.
        Configured model wins, else the host default LLM; None on failure
        (the manager then falls back to random selection)."""
        model_uuid = str(self.plugin_cfg.get("selection_model", "") or "").strip()
        try:
            if model_uuid:
                return self.ctx.get_llm_client(model_uuid=model_uuid)
            return self.ctx.provider_mgr.get_default_llm()
        except Exception as exc:
            logger.error("Failed to resolve selection LLM client for emoji plugin: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Tag: <sticker_plus> (per-request injection, needs the batch event)
    # ------------------------------------------------------------------

    @on.llm_request(priority=Priority.SYS_HIGH - 1)
    async def inject_sticker_plus_tag(self, event: KiraMessageBatchEvent, _, tag_set: TagSet):
        """Inject the <sticker_plus> tag when the adapter carries stickers."""
        if "sticker" not in event.message_types:
            return
        tag_set.register(self._build_sticker_plus_tag(event))

    def _build_sticker_plus_tag(self, event: KiraMessageBatchEvent) -> BaseTag:
        """Build a tag instance bound to this batch.

        @register.tag handlers never see the event, so the tag is assembled
        per request (same pattern as the built-in sticker plugin) and closes
        over the sid plus the recent-chat context the selection LLM needs.
        """
        plugin = self
        # The batch event's own message_str is never populated by the host;
        # the formatted text lives on each KiraIMMessage in event.messages.
        batch_text = "\n".join(
            m.message_str for m in event.messages if m.message_str
        )
        recent_context = batch_text[-500:]

        class StickerPlusTag(BaseTag):
            name = "sticker_plus"
            description = STICKER_PLUS_TAG_DESCRIPTION

            async def handle(self, value: str, **kwargs) -> list:
                # Must never raise: a tag handle exception aborts the whole
                # reply in send_xml_messages.
                try:
                    plugin._schedule_sticker_send(
                        emotion=str(value or "").strip(),
                        recent_context=recent_context,
                        sid=event.sid,
                    )
                except Exception:
                    logger.exception("sticker_plus schedule failed")
                return []

        return StickerPlusTag()

    def _schedule_sticker_send(self, emotion: str, recent_context: str, sid: str) -> None:
        """Register a pending send and fire the background pick-and-send task."""
        if not emotion:
            logger.warning("sticker_plus tag got empty emotion, skipped")
            return
        if self._manager is None:
            logger.warning("sticker_plus skipped: emoji library not initialized")
            return
        pending = _PendingStickerSend()
        # Prune every sid's bucket, not just this one: a turn interrupted
        # before its step-result hook never pops its bucket, so without a
        # sweep the dict keys would accumulate forever.
        for b_sid, records in list(self._pending_sends.items()):
            records = [p for p in records if p.age_ok()]
            if records:
                self._pending_sends[b_sid] = records
            else:
                self._pending_sends.pop(b_sid, None)
        bucket = self._pending_sends.get(sid, [])
        bucket.append(pending)
        self._pending_sends[sid] = bucket
        task = asyncio.create_task(
            self._send_picked_sticker(emotion, recent_context, sid, pending)
        )
        self._bg_sends.add(task)
        task.add_done_callback(self._bg_sends.discard)

    async def _send_picked_sticker(
        self, emotion: str, recent_context: str, sid: str, pending: _PendingStickerSend
    ) -> None:
        """Background task: pick via the selection LLM, send directly, record.

        Runs detached from the main model's turn; the step-result hook waits
        on ``pending.done`` to attach the outcome to the assistant message.
        """
        # Snapshot the manager: a shutdown or hot reload may clear or replace
        # self._manager while this detached task is still running; the send
        # then finishes against the library instance it was scheduled with
        # (by design, in-flight sends are left alone on shutdown).
        manager = self._manager
        try:
            picked = await manager.pick_emoji(emotion, recent_context)
            if picked is None:
                pending.ok = False
                pending.detail = "表情包库中没有合适的表情"
                return
            record, file_path = picked
            sticker_b64 = await image_to_base64(str(file_path))
            chain = MessageChain([Sticker(str(record.id), sticker=sticker_b64)])
            result = await self.ctx.send_message_chain(sid, chain)
            if result is not None and not getattr(result, "ok", True):
                pending.ok = False
                pending.detail = f"发送失败：{getattr(result, 'err', '') or '未知错误'}"
                return
            desc = (record.description or "").strip() or "无描述"
            pending.ok = True
            pending.detail = f"编号{record.id}（{desc}）"
        except Exception:
            logger.exception("sticker_plus background send failed")
            pending.ok = False
            pending.detail = "发送失败：内部错误"
        finally:
            pending.done.set()

    # ------------------------------------------------------------------
    # Hooks
    # ------------------------------------------------------------------

    @on.step_result()
    async def attach_sticker_send_record(self, event: KiraMessageBatchEvent, step_result, *_):
        """Append the send record to the assistant message that used the tag.

        The host writes ``step_result.raw_output`` back into the assistant
        message and persists it, so anything appended here lands in history
        right after the AI's own <msg> block. Gated on the raw output actually
        containing the tag: this hook fires for every agent step, and
        appending to a tool-call step's empty output would overwrite the
        tool-call assistant message.
        """
        raw = getattr(step_result, "raw_output", "") or ""
        if "<sticker_plus" not in raw:
            return
        pending_list = self._pending_sends.pop(event.sid, None)
        if not pending_list:
            return
        lines = []
        # One deadline for the whole batch: per-record timeouts would multiply
        # into len(pending_list) * SEND_RECORD_WAIT_TIMEOUT in the worst case.
        deadline = asyncio.get_running_loop().time() + SEND_RECORD_WAIT_TIMEOUT
        for pending in pending_list:
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                logger.warning(
                    "sticker_plus send not settled within %.0fs; record skipped",
                    SEND_RECORD_WAIT_TIMEOUT,
                )
                continue
            try:
                # Wait on the event, not the task: on timeout the task keeps
                # running and the sticker still goes out.
                await asyncio.wait_for(pending.done.wait(), timeout=remaining)
            except asyncio.TimeoutError:
                logger.warning(
                    "sticker_plus send not settled within %.0fs; record skipped",
                    SEND_RECORD_WAIT_TIMEOUT,
                )
                continue
            if not pending.detail:
                continue
            if pending.ok:
                lines.append(
                    f"<system_reminder>已随本条消息发送表情包：{pending.detail}</system_reminder>"
                )
            else:
                lines.append(
                    f"<system_reminder>本次表情包未发送：{pending.detail}</system_reminder>"
                )
        if lines:
            step_result.raw_output = raw + "\n" + "\n".join(lines)

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
        if status not in ("all", "active", "banned", "pending", "stolen", "review"):
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
            "steal_require_approval": bool(
                self.plugin_cfg.get("steal_require_approval", True)
            ),
            "capacity": manager.capacity,
            "candidate_count": manager.candidate_count,
            "max_emoji_size_mb": float(self.plugin_cfg.get("max_emoji_size_mb", 5.0)),
            "vlm_model": str(self.plugin_cfg.get("vlm_model", "") or ""),
            "selection_model": str(self.plugin_cfg.get("selection_model", "") or ""),
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

    @register.api(method="PUT", path="/settings/steal-approval")
    async def set_steal_approval(self, request: StealApprovalSettingRequest):
        """Toggle review-before-use for stolen emojis (they then land banned).

        Same persistence pattern as set_steal: update_plugin_config
        re-initializes this plugin instance, so build the response from
        locals only - do not touch self afterwards.
        """
        plugin_mgr = getattr(self.ctx, "plugin_mgr", None)
        if plugin_mgr is None or not plugin_mgr.has_plugin(PLUGIN_ID):
            raise HTTPException(status_code=503, detail="Plugin manager not available")
        updated = await plugin_mgr.update_plugin_config(
            PLUGIN_ID,
            {"steal_require_approval": bool(request.steal_require_approval)},
        )
        return {
            "steal_require_approval": bool(
                updated.get("steal_require_approval", request.steal_require_approval)
            )
        }
