"""Integration smoke test: full plugin flow with a fake host context.

Exercises: initialize() -> intake -> VLM tagging -> send_emoji tool ->
steal hook -> terminate(), all against the real main.StickerPlusPlugin with a
minimal fake PluginContext and a scripted fake LLM client.
"""

from __future__ import annotations

import asyncio
import importlib
from types import SimpleNamespace

import pytest
import pytest_asyncio

from core.chat.message_elements import Sticker

from helpers import b64, make_png_bytes

main_mod = importlib.import_module("kira-ai-plugin-sticker-plus.main")


class PromptAwareClient:
    """Answers based on prompt content so background tasks can't race the script."""

    def __init__(self):
        self.requests = []

    async def chat(self, request):
        self.requests.append(request)
        content = request.messages[0]["content"]
        text = next(p["text"] for p in content if p["type"] == "text")
        if "表情包分析助手" in text:  # tagging prompt
            return SimpleNamespace(
                text_response='{"description": "一只小动物很开心的样子", "emotions": "开心,调皮"}'
            )
        if "从以下表情包中选择" in text:  # selection prompt
            return SimpleNamespace(
                text_response='{"emoji_index": 1, "reason": "匹配开心语境"}'
            )
        raise AssertionError(f"unexpected prompt: {text[:60]}")


class FakeCtx:
    def __init__(self, data_dir, client):
        self._data_dir = data_dir
        self._client = client
        self.provider_mgr = SimpleNamespace(
            get_default_vlm=lambda: client,
            get_default_llm=lambda: client,
        )
        self.requested_model_uuids: list[str] = []

    def get_plugin_data_dir(self):
        return str(self._data_dir)

    def get_llm_client(self, model_uuid):
        self.requested_model_uuids.append(str(model_uuid))
        return self._client


@pytest_asyncio.fixture
async def plugin(tmp_path):
    client = PromptAwareClient()
    ctx = FakeCtx(tmp_path, client)
    cfg = {"capacity": 50, "candidate_count": 9, "steal_emoji": True, "max_emoji_size_mb": 5.0}
    inst = main_mod.StickerPlusPlugin(ctx, cfg)
    await inst.initialize()
    assert inst._manager is not None, "plugin must initialize successfully"
    yield inst
    await inst.terminate()


@pytest.mark.asyncio
async def test_send_emoji_flow(plugin):
    # intake two emojis; background tagging tasks fire per add
    assert await plugin._manager.add_emoji_from_bytes(make_png_bytes(color=(7, 7, 7)), "manual")
    assert await plugin._manager.add_emoji_from_bytes(make_png_bytes(color=(8, 8, 8)), "manual")
    # let background tagging settle (serialized by the manager's tag lock)
    if plugin._manager._bg_tasks:
        await asyncio.gather(*list(plugin._manager._bg_tasks), return_exceptions=True)
    stats = await plugin._manager.stats()
    assert stats["active"] == 2

    event = SimpleNamespace(message_str="今天真开心呀，哈哈")
    result = await plugin.send_emoji(event, emotion="开心")

    assert isinstance(result, main_mod.ToolResult)
    assert result.attachments, "one image attachment must be returned"
    attachment = result.attachments[0]
    assert attachment.image_type == "path"
    import os
    assert os.path.exists(attachment.image)
    assert "已选择表情包" in result.text


@pytest.mark.asyncio
async def test_send_emoji_empty_hint(plugin):
    result = await plugin.send_emoji(SimpleNamespace(message_str="x"), emotion="")
    assert isinstance(result, main_mod.ToolResult)
    assert not result.attachments


@pytest.mark.asyncio
async def test_steal_hook_stores_sticker(plugin):
    from core.chat.message_elements import Text
    png = make_png_bytes(color=(99, 99, 99))
    event = SimpleNamespace(
        message=SimpleNamespace(
            chain=[Text("看这个"), Sticker(sticker=b64(png))],
            sender=SimpleNamespace(user_id="10001"),
            self_id="99999",
        )
    )
    await plugin.steal_emoji(event)
    # background intake: flush it deterministically via stealer shutdown path
    await plugin._stealer.shutdown()
    items, total = await plugin._manager.list_emojis()
    assert total == 1
    assert items[0]["source"] == "stolen"


@pytest.mark.asyncio
async def test_steal_hook_rejects_non_image(plugin):
    # e.g. a Telegram video sticker: extracted fine, rejected at intake
    event = SimpleNamespace(
        message=SimpleNamespace(
            chain=[Sticker(sticker=b64(b"webm-video-sticker-bytes"))],
            sender=SimpleNamespace(user_id="10001"),
            self_id="99999",
        )
    )
    await plugin.steal_emoji(event)
    await plugin._stealer.shutdown()
    _, total = await plugin._manager.list_emojis()
    assert total == 0


@pytest.mark.asyncio
async def test_steal_disabled_short_circuits_hook(plugin):
    # Toggle off -> the hook returns before any extraction/intake/tagging.
    plugin.plugin_cfg["steal_emoji"] = False
    try:
        event = SimpleNamespace(
            message=SimpleNamespace(
                chain=[Sticker(sticker=b64(make_png_bytes(color=(55, 55, 55))))],
                sender=SimpleNamespace(user_id="10001"),
                self_id="99999",
            )
        )
        await plugin.steal_emoji(event)
        await plugin._stealer.shutdown()
        _, total = await plugin._manager.list_emojis()
        assert total == 0, "nothing may be stored while stealing is disabled"
        assert not plugin._manager._bg_tasks, "no tagging may be scheduled"
    finally:
        plugin.plugin_cfg["steal_emoji"] = True


@pytest.mark.asyncio
async def test_settings_expose_split_models(plugin):
    settings = await plugin.get_settings()
    assert settings["vlm_model"] == ""
    assert settings["selection_model"] == ""


@pytest.mark.asyncio
async def test_configured_models_route_via_get_llm_client(plugin):
    plugin.plugin_cfg["vlm_model"] = "tag-uuid"
    plugin.plugin_cfg["selection_model"] = "select-uuid"
    try:
        assert plugin._resolve_vlm_client() is not None
        assert plugin._resolve_selection_client() is not None
        assert set(plugin.ctx.requested_model_uuids) == {"tag-uuid", "select-uuid"}
    finally:
        plugin.plugin_cfg.pop("vlm_model", None)
        plugin.plugin_cfg.pop("selection_model", None)
