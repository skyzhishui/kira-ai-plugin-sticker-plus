"""Integration smoke test: full plugin flow with a fake host context.

Exercises: initialize() -> intake -> VLM tagging -> <sticker_plus> tag
schedule + direct send -> step-result record injection -> steal hook ->
terminate(), all against the real main.StickerPlusPlugin with a minimal fake
PluginContext and a scripted fake LLM client.
"""

from __future__ import annotations

import asyncio
import importlib
import time
from types import SimpleNamespace

import pytest
import pytest_asyncio

from core.chat.message_elements import Sticker

from helpers import b64, make_png_bytes

main_mod = importlib.import_module("kira-ai-plugin-sticker-plus.main")


SID = "qq:dm:10001"


def make_batch_event(text: str = "今天真开心呀，哈哈", sid: str = SID, supported_elements=None):
    """Minimal stand-in for a KiraMessageBatchEvent."""
    return SimpleNamespace(
        sid=sid,
        supported_elements=supported_elements if supported_elements is not None else ["text", "sticker"],
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


@pytest_asyncio.fixture(autouse=True)
async def _clean_module_drainers():
    """Drainer tasks are module-level state; never let them leak across tests.

    A drainer left pending when a test's event loop closes would also never
    run its done-callback, permanently polluting _active_drainers. Only
    tasks on the current loop can be cancelled; ones from a closed loop are
    dead already.
    """
    yield
    loop = asyncio.get_running_loop()
    live = [
        t for t in main_mod._active_drainers
        if not t.done() and t.get_loop() is loop
    ]
    for task in live:
        task.cancel()
    if live:
        await asyncio.gather(*live, return_exceptions=True)
    main_mod._active_drainers.clear()


@pytest.mark.asyncio
async def test_tag_injection_gated_by_supported_elements(plugin):
    from core.tag import TagSet

    event = make_batch_event()
    tag_set = TagSet()
    await plugin.inject_sticker_plus_tag(event, None, tag_set)
    assert "sticker_plus" in tag_set
    tag = tag_set.get("sticker_plus")
    assert tag is not None and tag.name == "sticker_plus"

    # Adapters without sticker support must not get the tag.
    empty_tag_set = TagSet()
    await plugin.inject_sticker_plus_tag(
        make_batch_event(supported_elements=["text"]), None, empty_tag_set
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


@pytest.mark.asyncio
async def test_shutdown_cleanup_survives_cancelled_grace_wait(plugin):
    """取消 15s 宽限等待不得跳过资源清理。

    Regression (neko review): the grace wait sits in the middle of
    ``_shutdown_components()``; when the coroutine is cancelled while waiting,
    the teardown below it (stealer / manager / database) used to be skipped
    entirely - leaking exactly the handles the grace period protects. The
    teardown now lives in ``finally``, so it runs and the cancellation still
    propagates.
    """

    class _FakeStealer:
        def __init__(self):
            self.closed = False

        async def shutdown(self):
            self.closed = True

    stealer = _FakeStealer()
    plugin._stealer = stealer
    # A never-finishing background send keeps asyncio.wait inside the grace
    # window long enough to be cancelled.
    ev = asyncio.Event()
    hang = asyncio.ensure_future(ev.wait())
    plugin._bg_sends.add(hang)
    shutdown_task = asyncio.ensure_future(plugin._shutdown_components())
    await asyncio.sleep(0.05)
    shutdown_task.cancel()
    try:
        with pytest.raises(asyncio.CancelledError):
            await shutdown_task
        assert stealer.closed, "cancelling the grace wait must not skip teardown"
    finally:
        plugin._bg_sends.discard(hang)
        hang.cancel()
        await asyncio.gather(hang, return_exceptions=True)


@pytest.mark.asyncio
async def test_send_task_uses_manager_snapshotted_at_schedule_time(plugin):
    """调度时快照 manager：任务首次执行前清空 self._manager 不得影响该次发送。

    Regression (neko review): the snapshot used to be taken inside the task
    coroutine, i.e. at its first step. A hot reload / shutdown running between
    ``create_task()`` and that first step clears ``self._manager``, so the task
    then read None and raised ``AttributeError`` (recorded as a bogus
    "internal error" send failure) instead of using the library instance it
    was scheduled with.
    """
    event = make_batch_event()
    tag = plugin._build_sticker_plus_tag(event)
    saved_manager = plugin._manager
    try:
        await tag.handle("开心")  # creates the task; it has not run yet
        plugin._manager = None  # the race window: reload/shutdown in between
        if plugin._bg_sends:
            await asyncio.gather(*list(plugin._bg_sends), return_exceptions=True)
        step_result = make_step_result("<msg><sticker_plus>开心</sticker_plus></msg>")
        await plugin.attach_sticker_send_record(event, step_result)
        # Empty library -> the scheduled manager reports "no suitable emoji".
        assert "表情包库中没有合适的表情" in step_result.raw_output
        assert "内部错误" not in step_result.raw_output
    finally:
        plugin._manager = saved_manager


@pytest.mark.asyncio
async def test_sweep_keeps_slow_turns_pending_record(plugin):
    """跨会话清扫只清死记录：慢回合的记录不得被别的会话清掉。

    Regression (neko review): the sweep used PENDING_RECORD_MAX_AGE (60s), so
    a turn still running in session A (LLM + tool calls easily exceed 60s) lost
    its pending record as soon as session B scheduled a sticker - the sticker
    had gone out but the model never learned about it. The sweep now uses the
    much longer PENDING_SWEEP_MAX_AGE.
    """
    slow = main_mod._PendingStickerSend()
    slow.created_at = time.monotonic() - 70.0  # > 60s, < 300s
    plugin._pending_sends["qq:gm:slow"] = [slow]
    try:
        plugin._schedule_sticker_send("开心", "", "qq:gm:fast")
        assert "qq:gm:slow" in plugin._pending_sends, (
            "a live slow turn's record must survive another session's sweep"
        )
        assert plugin._pending_sends["qq:gm:slow"] == [slow]
    finally:
        if plugin._bg_sends:
            await asyncio.gather(*list(plugin._bg_sends), return_exceptions=True)
        plugin._pending_sends.pop("qq:gm:fast", None)
        plugin._pending_sends.pop("qq:gm:slow", None)


@pytest.mark.asyncio
async def test_hot_reload_shutdown_grace_is_short(plugin, monkeypatch):
    """热重载（设置更新）走短宽限，不把 15s 停机等待带到管理页请求上。

    Regression (neko review): update_plugin_config -> init_plugin -> terminate
    hits _shutdown_components() too, so the full shutdown grace would block the
    settings request (and the serialized queue behind it) for up to 15s.
    """
    monkeypatch.setattr(main_mod, "HOT_RELOAD_SEND_GRACE", 0.05)
    monkeypatch.setattr(main_mod, "BG_SEND_SHUTDOWN_GRACE", 0.5)
    ev = asyncio.Event()
    hang = asyncio.ensure_future(ev.wait())
    plugin._bg_sends.add(hang)
    try:
        plugin._hot_reload_depth = 1
        start = time.monotonic()
        await plugin._shutdown_components()
        elapsed = time.monotonic() - start
        assert elapsed < 0.3, f"hot reload waited {elapsed:.2f}s instead of the short grace"
    finally:
        plugin._bg_sends.discard(hang)
        hang.cancel()
        await asyncio.gather(hang, return_exceptions=True)


@pytest.mark.asyncio
async def test_hot_reload_flag_does_not_outlive_the_update(plugin):
    """_hot_reload 不得残留：更新失败、或宿主原地重初始化时都必须复位。

    Regression (neko review): the flag was never cleared, so an instance that
    stayed alive after a failed config update (or an in-place re-initialize)
    would shorten every later process shutdown to the hot-reload grace and tear
    manager/db down under in-flight sends.
    """

    class _FailingMgr:
        def has_plugin(self, plugin_id):
            return True

        async def update_plugin_config(self, plugin_id, cfg):
            raise RuntimeError("config update failed")

    plugin.ctx.plugin_mgr = _FailingMgr()
    try:
        with pytest.raises(RuntimeError):
            await plugin.set_steal(main_mod.StealSettingRequest(steal_emoji=True))
        assert plugin._hot_reload_depth == 0, "depth leaked after a failed update"

        # An in-place re-initialize inside an update keeps the depth > 0, and
        # the balanced decrement afterwards must leave no residue.
        plugin._hot_reload_depth = 1
        await plugin.initialize()
        plugin._hot_reload_depth -= 1
        assert plugin._hot_reload_depth == 0
    finally:
        del plugin.ctx.plugin_mgr


@pytest.mark.asyncio
async def test_concurrent_settings_updates_are_serialized(plugin):
    """并发设置请求：后端串行化，两个 update_plugin_config 不得交错。

    Regression (neko review): the page-side settingsQueue only orders one
    page's requests; two tabs / two clients (and the host replacing the
    instance mid-update) could still interleave terminate/init. The backend
    module-level lock must make the updates strictly sequential.
    """
    events = []
    gate = asyncio.Event()

    class _RecordingMgr:
        def has_plugin(self, plugin_id):
            return True

        async def update_plugin_config(self, plugin_id, cfg):
            events.append(("enter", plugin_id))
            await gate.wait()
            events.append(("exit", plugin_id))
            return cfg

    plugin.ctx.plugin_mgr = _RecordingMgr()
    try:
        t1 = asyncio.ensure_future(
            plugin.set_steal(main_mod.StealSettingRequest(steal_emoji=True))
        )
        t2 = asyncio.ensure_future(
            plugin.set_steal_approval(
                main_mod.StealApprovalSettingRequest(steal_require_approval=False)
            )
        )
        await asyncio.sleep(0.05)
        # The first update is parked inside the manager; the second must be
        # waiting on the lock, not running alongside it.
        assert events == [("enter", main_mod.PLUGIN_ID)], (
            "second update started before the first finished"
        )
        gate.set()
        await asyncio.gather(t1, t2)
        assert plugin._hot_reload_depth == 0
        assert events == [
            ("enter", main_mod.PLUGIN_ID),
            ("exit", main_mod.PLUGIN_ID),
            ("enter", main_mod.PLUGIN_ID),
            ("exit", main_mod.PLUGIN_ID),
        ], "updates must run strictly one after the other"
    finally:
        del plugin.ctx.plugin_mgr


@pytest.mark.asyncio
async def test_settings_updates_serialized_across_instances(plugin):
    """跨实例串行化：宿主在 update 内替换实例后，锁仍须生效。

    Regression (neko review): the host rebuilds the plugin instance inside
    update_plugin_config and rebinds the API routes to it, so an
    instance-level lock would let a request routed to the fresh instance run
    concurrently with the old instance's terminate/init. The lock is
    module-level, so two instances' updates must not overlap either.
    """
    gate = asyncio.Event()
    overlaps = []
    active = 0

    class _GatedMgr:
        def has_plugin(self, plugin_id):
            return True

        async def update_plugin_config(self, plugin_id, cfg):
            nonlocal active
            active += 1
            if active > 1:
                overlaps.append("overlap")
            try:
                await gate.wait()
                return cfg
            finally:
                active -= 1

    other = main_mod.StickerPlusPlugin(plugin.ctx, dict(plugin.plugin_cfg))
    plugin.ctx.plugin_mgr = _GatedMgr()
    try:
        # The old instance's update parks inside the manager (gate closed);
        # the "fresh" instance's update must block on the module lock.
        t1 = asyncio.ensure_future(
            plugin.set_steal(main_mod.StealSettingRequest(steal_emoji=True))
        )
        await asyncio.sleep(0.05)
        t2 = asyncio.ensure_future(
            other.set_steal_approval(
                main_mod.StealApprovalSettingRequest(steal_require_approval=False)
            )
        )
        await asyncio.sleep(0.05)
        assert overlaps == [], "updates on two instances ran concurrently"
        gate.set()
        r1, r2 = await asyncio.gather(t1, t2)
        assert r1["steal_emoji"] is True
        assert r2["steal_require_approval"] is False
        assert not main_mod._active_drainers, "no drainer expected in this test"
    finally:
        del plugin.ctx.plugin_mgr
        await other.terminate()


@pytest.mark.asyncio
async def test_hot_reload_defers_component_teardown_until_sends_settle(
    plugin, monkeypatch
):
    """热重载短宽限到期后，组件必须延迟到在途发送结束再释放。

    Regression (neko review): HOT_RELOAD_SEND_GRACE (2s) is shorter than a
    slow pick, so on the settings-update path the teardown used to dispose
    the manager/database while the send was still using them - recorded as a
    bogus "internal error" failure. Now the components are detached and
    released by a drainer only after the send settles.
    """
    monkeypatch.setattr(main_mod, "HOT_RELOAD_SEND_GRACE", 0.05)
    monkeypatch.setattr(main_mod, "BG_SEND_SHUTDOWN_GRACE", 5.0)
    pick_started = asyncio.Event()
    release_pick = asyncio.Event()
    shutdown_calls: list[float] = []
    pick_done_at: list[float] = []
    real_manager = plugin._manager

    class _SlowPickManager:
        """pick_emoji blocks until the test releases it."""

        def __getattr__(self, name):
            return getattr(real_manager, name)

        async def pick_emoji(self, emotion, recent_context):
            pick_started.set()
            await release_pick.wait()
            pick_done_at.append(time.monotonic())
            # No candidate: enough to settle the pending record as "not sent".
            return None

        async def shutdown(self):
            shutdown_calls.append(time.monotonic())
            await real_manager.shutdown()

    plugin._manager = _SlowPickManager()
    plugin._schedule_sticker_send("开心", "让我看看", "chat-1")
    assert await asyncio.wait_for(pick_started.wait(), timeout=2), "pick never started"

    plugin._hot_reload_depth = 1
    start = time.monotonic()
    await plugin._shutdown_components()
    elapsed = time.monotonic() - start
    # The settings request unblocks after the short grace, not after the send.
    assert elapsed < 2, f"shutdown blocked {elapsed:.2f}s behind a slow send"
    assert shutdown_calls == [], "manager disposed while its send was still running"

    # Let the send settle; the drainer must release the components after it.
    release_pick.set()
    drainers = set(main_mod._active_drainers)
    assert drainers, "a drainer should own the detached components"
    await asyncio.wait_for(asyncio.gather(*drainers), timeout=5)
    assert len(shutdown_calls) == 1, "detached manager released exactly once"
    assert pick_done_at, "pick must have completed"
    assert shutdown_calls[0] >= pick_done_at[0], (
        "components released before the send finished"
    )
    # The send was recorded as a graceful "no candidate", not an error.
    bucket = plugin._pending_sends.get("chat-1", [])
    assert not bucket, "pending records die at the hot-reload boundary"


@pytest.mark.asyncio
async def test_same_sid_sweep_keeps_short_ttl(plugin):
    """同 sid 清扫保持短 TTL：被打断回合的旧记录不得混进新回合的 reminder。

    Regression (neko review): the cross-session sweep used
    PENDING_SWEEP_MAX_AGE (300s) for the CURRENT sid too, so a record left
    by a turn interrupted mid-send survived for minutes and was then popped
    wholesale by the next turn's hook - the model read a stale sticker's
    outcome as if the new message had sent it.
    """
    stale = main_mod._PendingStickerSend()
    stale.created_at = time.monotonic() - 70.0  # > 60s, < 300s
    plugin._pending_sends["qq:gm:x"] = [stale]
    try:
        plugin._schedule_sticker_send("开心", "", "qq:gm:x")
        bucket = plugin._pending_sends["qq:gm:x"]
        assert stale not in bucket, (
            "a stale same-session record must be pruned by the short TTL"
        )
        assert len(bucket) == 1, "only the freshly scheduled record remains"
    finally:
        if plugin._bg_sends:
            await asyncio.gather(*list(plugin._bg_sends), return_exceptions=True)
        plugin._pending_sends.pop("qq:gm:x", None)


@pytest.mark.asyncio
async def test_shutdown_waits_for_drainers_without_own_sends(plugin, monkeypatch):
    """停机时本实例没有在途发送，也必须等在途 drainer。

    Regression (neko review): the drainer wait was nested inside
    ``if self._bg_sends:`` - with no own sends, shutdown skipped it and the
    drainer-held manager/db were released only via loop-close cancellation,
    where the release awaits are not guaranteed to run at all.
    """
    monkeypatch.setattr(main_mod, "BG_SEND_SHUTDOWN_GRACE", 2.0)
    release = asyncio.Event()
    hang = asyncio.ensure_future(release.wait())

    # Hand the plugin's real components to a drainer parked on `hang`.
    plugin._defer_component_teardown({hang})
    assert plugin._manager is None and plugin._db is None
    assert main_mod._active_drainers, "drainer must own the detached components"

    # depth == 0 -> process shutdown path; no own bg sends at all.
    assert not plugin._bg_sends
    shutdown = asyncio.ensure_future(plugin._shutdown_components())
    await asyncio.sleep(0.1)
    assert not shutdown.done(), "shutdown skipped waiting for the drainer"
    release.set()
    await asyncio.wait_for(shutdown, timeout=5)
    assert not main_mod._active_drainers, "drainer settled within the grace"
