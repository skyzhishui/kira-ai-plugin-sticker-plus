"""Integration smoke test: full plugin flow with a fake host context.

Exercises: initialize() -> intake -> VLM tagging -> <sticker_plus> tag
schedule + direct send -> step-result record injection -> steal hook ->
terminate(), all against the real main.StickerPlusPlugin with a minimal fake
PluginContext and a scripted fake LLM client.
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


SID = "qq:dm:10001"


def make_batch_event(text: str = "今天真开心呀，哈哈", sid: str = SID, message_types=None):
    """Minimal stand-in for a KiraMessageBatchEvent."""
    return SimpleNamespace(
        sid=sid,
        message_types=message_types if message_types is not None else ["text", "sticker"],
        messages=[SimpleNamespace(message_str=text)],
    )


def make_step_result(raw: str):
    """Minimal stand-in for KiraStepResult (raw_output is rewritten by the hook)."""
    return SimpleNamespace(raw_output=raw)


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
        # Records direct sends as (sid, chain); result scripted per test.
        self.sent_chains: list[tuple[str, object]] = []
        self.send_result = SimpleNamespace(ok=True, err="", message_id="msg-1")

    def get_plugin_data_dir(self):
        return str(self._data_dir)

    def get_llm_client(self, model_uuid):
        self.requested_model_uuids.append(str(model_uuid))
        return self._client

    async def send_message_chain(self, sid, chain):
        self.sent_chains.append((sid, chain))
        return self.send_result


async def add_tagged_emojis(plugin, count: int = 2) -> None:
    """Intake emojis and settle their background VLM tagging tasks."""
    for i in range(count):
        assert await plugin._manager.add_emoji_from_bytes(
            make_png_bytes(color=(7 + i, 7 + i, 7 + i)), "manual"
        )
    if plugin._manager._bg_tasks:
        await asyncio.gather(*list(plugin._manager._bg_tasks), return_exceptions=True)


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
async def test_tag_injection_gated_by_message_types(plugin):
    from core.tag import TagSet

    event = make_batch_event()
    tag_set = TagSet()
    await plugin.inject_sticker_plus_tag(event, None, tag_set)
    assert "sticker_plus" in tag_set
    tag = tag_set.get("sticker_plus")
    assert tag is not None and tag.name == "sticker_plus"

    # Adapters without sticker message types must not get the tag.
    empty_tag_set = TagSet()
    await plugin.inject_sticker_plus_tag(
        make_batch_event(message_types=["text"]), None, empty_tag_set
    )
    assert "sticker_plus" not in empty_tag_set


@pytest.mark.asyncio
async def test_sticker_plus_flow(plugin):
    await add_tagged_emojis(plugin)

    event = make_batch_event()
    tag = plugin._build_sticker_plus_tag(event)
    # The tag handle must never block the reply: returns [] immediately.
    assert await tag.handle("开心") == []

    # The step-result hook waits for the background send and appends the record.
    raw = "<msg>\n    <text>哈哈</text>\n    <sticker_plus>开心</sticker_plus>\n</msg>"
    step_result = make_step_result(raw)
    await plugin.attach_sticker_send_record(event, step_result)

    # Direct send happened through the adapter path with one Sticker element.
    assert len(plugin.ctx.sent_chains) == 1
    sent_sid, chain = plugin.ctx.sent_chains[0]
    assert sent_sid == SID
    assert len(chain) == 1 and isinstance(chain[0], Sticker)

    # History record: system_reminder appended after the AI's own message.
    assert step_result.raw_output.startswith(raw)
    assert "已随本条消息发送表情包：编号" in step_result.raw_output
    assert "<system_reminder>已随本条消息发送表情包：" in step_result.raw_output
    assert "</system_reminder>" in step_result.raw_output
    # Pending bucket drained.
    assert SID not in plugin._pending_sends

    # Regression: the selection prompt must carry the triggering batch text
    # (built from event.messages[].message_str, not the never-set batch field).
    selection_texts = [
        p["text"]
        for r in plugin.ctx._client.requests
        for p in r.messages[0]["content"]
        if p["type"] == "text" and "从以下表情包中选择" in p["text"]
    ]
    assert selection_texts, "selection prompt was not requested"
    assert "今天真开心呀，哈哈" in selection_texts[-1]


@pytest.mark.asyncio
async def test_sticker_plus_send_failure_recorded(plugin):
    await add_tagged_emojis(plugin, count=1)
    plugin.ctx.send_result = SimpleNamespace(ok=False, err="boom", message_id=None)

    event = make_batch_event()
    tag = plugin._build_sticker_plus_tag(event)
    await tag.handle("开心")

    step_result = make_step_result("<msg><sticker_plus>开心</sticker_plus></msg>")
    await plugin.attach_sticker_send_record(event, step_result)
    assert "本次表情包未发送：发送失败：boom" in step_result.raw_output


@pytest.mark.asyncio
async def test_sticker_plus_no_candidate_recorded(plugin):
    # Empty library: pick_emoji returns None -> explicit "not sent" record.
    event = make_batch_event()
    tag = plugin._build_sticker_plus_tag(event)
    await tag.handle("开心")

    step_result = make_step_result("<msg><sticker_plus>开心</sticker_plus></msg>")
    await plugin.attach_sticker_send_record(event, step_result)
    assert "本次表情包未发送：表情包库中没有合适的表情" in step_result.raw_output
    assert not plugin.ctx.sent_chains


@pytest.mark.asyncio
async def test_sticker_plus_timeout_keeps_silence(plugin, monkeypatch):
    monkeypatch.setattr(main_mod, "SEND_RECORD_WAIT_TIMEOUT", 0.05)
    release = asyncio.Event()

    async def hanging_pick(emoji_hint, recent_context=""):
        await release.wait()
        return None

    plugin._manager.pick_emoji = hanging_pick

    event = make_batch_event()
    tag = plugin._build_sticker_plus_tag(event)
    await tag.handle("开心")

    raw = "<msg><sticker_plus>开心</sticker_plus></msg>"
    step_result = make_step_result(raw)
    await plugin.attach_sticker_send_record(event, step_result)
    assert step_result.raw_output == raw, "timed-out sends must not produce a record"
    assert SID not in plugin._pending_sends

    # Let the background task finish cleanly so no task leaks past the test.
    release.set()
    if plugin._bg_sends:
        await asyncio.gather(*list(plugin._bg_sends), return_exceptions=True)


@pytest.mark.asyncio
async def test_sticker_plus_multiple_tags_multiple_records(plugin):
    await add_tagged_emojis(plugin, count=2)

    event = make_batch_event()
    tag = plugin._build_sticker_plus_tag(event)
    await tag.handle("开心")
    await tag.handle("害羞")

    step_result = make_step_result(
        "<msg><text>嗨</text><sticker_plus>开心</sticker_plus></msg>"
        "<msg><sticker_plus>害羞</sticker_plus></msg>"
    )
    await plugin.attach_sticker_send_record(event, step_result)
    assert step_result.raw_output.count("<system_reminder>已随本条消息发送表情包：") == 2
    assert len(plugin.ctx.sent_chains) == 2


@pytest.mark.asyncio
async def test_sticker_plus_gate_requires_tag_in_output(plugin):
    await add_tagged_emojis(plugin, count=1)

    event = make_batch_event()
    tag = plugin._build_sticker_plus_tag(event)
    await tag.handle("开心")

    # A step without the tag (e.g. a tool-call step) must stay untouched and
    # must not drain the pending bucket.
    step_result = make_step_result("<msg><text>普通消息</text></msg>")
    await plugin.attach_sticker_send_record(event, step_result)
    assert step_result.raw_output == "<msg><text>普通消息</text></msg>"
    assert SID in plugin._pending_sends, "pending entry must survive for the real step"

    # The real text step then drains it and appends the record.
    text_step = make_step_result("<msg><sticker_plus>开心</sticker_plus></msg>")
    await plugin.attach_sticker_send_record(event, text_step)
    assert "已随本条消息发送表情包：" in text_step.raw_output


@pytest.mark.asyncio
async def test_sticker_plus_empty_emotion_skipped(plugin):
    event = make_batch_event()
    tag = plugin._build_sticker_plus_tag(event)
    assert await tag.handle("   ") == []
    assert not plugin._pending_sends
    assert not plugin.ctx.sent_chains


@pytest.mark.asyncio
async def test_sticker_plus_library_unavailable_skipped(plugin):
    plugin._manager = None
    try:
        event = make_batch_event()
        tag = plugin._build_sticker_plus_tag(event)
        assert await tag.handle("开心") == []
        assert not plugin._pending_sends
    finally:
        await plugin.initialize()  # restore a working manager for teardown


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
    # default config: stolen emojis land banned pending manual review
    assert items[0]["is_banned"] is True
    assert items[0]["needs_review"] is True


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
    # approval toggle defaults to on (stolen emojis land disabled)
    assert settings["steal_require_approval"] is True
    assert plugin._manager._steal_require_approval is True


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
