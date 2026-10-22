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

# Cross-session sweep TTL: a pending record older than this is considered dead
# (its turn never reached the step-result hook) and is swept when any session
# schedules a new sticker. Deliberately much longer than
# PENDING_RECORD_MAX_AGE: a legitimately slow turn (LLM + tool calls easily
# exceeds 60s) must not lose its send record just because another session
# scheduled a sticker in the meantime.
PENDING_SWEEP_MAX_AGE = 300.0

# Grace period on shutdown: in-flight background sends are awaited (not
# cancelled) for at most this long before components are torn down, so
# stickers already scheduled still go out without blocking shutdown forever.
BG_SEND_SHUTDOWN_GRACE = 15.0

# Same grace for every host-management path that is NOT a process exit:
# our own settings endpoints (update_plugin_config -> init_plugin ->
# terminate on this instance) AND host-driven disable / reload / uninstall
# (set_plugin_enabled(False), prepare_plugin_reload, ... all call the same
# terminate()). Kept short on purpose: an HTTP request or management action
# waits for it, so a slow pick must not block it for the full shutdown
# grace.
#
# The trade-off this implies (a pick often needs longer than this grace) is
# handled by deferral, not by tearing resources out from under the send: when
# sends are still running when the grace expires, the components they were
# scheduled against are detached and handed to a background drainer task
# (see _drain_detached_components) that releases them once the sends settle.
# The triggering request therefore unblocks after ~this grace while the sends
# still finish against their own manager/database.
HOT_RELOAD_SEND_GRACE = 2.0

# Serializes PUT /settings/* across every instance of this plugin. The host
# replaces the plugin instance inside update_plugin_config (init_plugin builds
# a new object and rebinds the API routes to it), so an instance attribute
# would not close the race: a second request routed to the fresh instance
# would run concurrently with the first update's terminate/init. One process
# has one live instance of this plugin, so one module-level lock serializes
# all of them.
#
# The holder rebinding on loop change exists because asyncio.Lock caches the
# loop of its first acquire and raises on any other. The host runs one loop
# at a time (a new loop only appears after the old one is gone), so no holder
# of the previous lock can still be alive when the lock is rebound - the
# serialization is never weakened in practice.
class _LoopLocalLock:
    """An asyncio.Lock recreated when used from a different loop."""

    def __init__(self) -> None:
        self._lock: Optional[asyncio.Lock] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    async def __aenter__(self) -> None:
        loop = asyncio.get_running_loop()
        if self._lock is None or self._loop is not loop:
            self._lock = asyncio.Lock()
            self._loop = loop
        await self._lock.acquire()

    async def __aexit__(self, *exc) -> None:
        if self._lock is not None:
            self._lock.release()


_settings_update_lock = _LoopLocalLock()

# Drainer tasks releasing components detached at a hot-reload boundary. Tracked
# at module level because the instance that detached the components may itself
# be replaced before the drain finishes; a later process shutdown waits for
# these too, so detached sends still get their bounded grace.
_active_drainers: set[asyncio.Task] = set()

# Depth of in-flight settings updates (our own PUT /settings/* handlers),
# counted across ALL instances of this plugin: the host replaces the instance
# inside update_plugin_config, and a second request dispatched before the
# routes rebind still runs on the stale instance - its handler would raise a
# per-instance counter on an object the host no longer terminates, while the
# actually-terminated fresh instance would observe 0 and fall back to the
# long process-shutdown grace (blocking that request ~15s and force-releasing
# components under in-flight sends). Module-level, like the lock, so any
# teardown running inside an update window sees the hot-reload grace.
# A counter, not a bool: overlapping windows from stale-routed requests must
# not cancel each other's short grace.
_hot_reload_depth = 0

# Set by our ON_SHUTDOWN hook: the host fires it in lifecycle.stop() BEFORE
# plugin_manager.terminate(), which makes it the one reliable signal that a
# terminate() is a process exit rather than a plugin disable / reload /
# settings update (those all call the same instance method with no reason).
# Only a process exit may force-tear components under still-running sends;
# every host-management path defers via the drainer instead.
_process_shutting_down = False

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

    def age_ok(self, max_age: float = PENDING_RECORD_MAX_AGE) -> bool:
        """False once the record is older than ``max_age`` seconds.

        ``max_age`` defaults to PENDING_RECORD_MAX_AGE (same-sid pruning); the
        cross-session sweep passes PENDING_SWEEP_MAX_AGE so a slow but live
        turn keeps its record.
        """
        return time.monotonic() - self.created_at < max_age


async def _release_components(stealer, manager, db) -> None:
    """Shut down one (stealer, manager, db) triple, best effort.

    Used both by the synchronous teardown in ``_shutdown_components`` and by
    the detached drainer, so the disposal sequence is defined exactly once.
    Never raises: a failing component must not skip the release of the others.
    """
    if stealer is not None:
        try:
            await stealer.shutdown()
        except Exception:
            logger.warning("Error shutting down EmojiStealer", exc_info=True)
    if manager is not None:
        try:
            await manager.shutdown()
        except Exception:
            logger.warning("Error shutting down EmojiManager", exc_info=True)
    if db is not None:
        try:
            await db.dispose()
        except Exception:
            logger.warning("Error disposing emoji database", exc_info=True)


async def _drain_detached_components(stealer, manager, db, sends) -> None:
    """Wait for sends detached at a hot-reload boundary, then release their
    components.

    The hot-reload grace (HOT_RELOAD_SEND_GRACE) is deliberately shorter than
    a slow pick; rather than disposing the manager/database while such a send
    is still using them, the settings request detaches the components and
    spawns this task. It gives the stragglers the full shutdown grace to
    settle, then releases the components either way - a stuck send must not
    pin resources forever.
    """
    try:
        done, still_running = await asyncio.wait(set(sends), timeout=BG_SEND_SHUTDOWN_GRACE)
        if still_running:
            logger.warning(
                "%d sticker_plus background send(s) still running %.0fs after a "
                "hot reload; releasing their components anyway",
                len(still_running), BG_SEND_SHUTDOWN_GRACE,
            )
    finally:
        await _release_components(stealer, manager, db)


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
        # (Settings-update depth lives at module level: see _hot_reload_depth
        # there. The instance replacement inside update_plugin_config makes a
        # per-instance counter observe the wrong teardown.)
        # True while _shutdown_components() is closing this incarnation (or
        # until a successful initialize() reopens it): background sends
        # scheduled during the close would miss both the waiter snapshot and
        # the drainer handoff yet still run against components that are about
        # to be released. See _shutdown_components / _schedule_sticker_send.
        self._closing = False

    # ------------------------------------------------------------------
    # Lifecycle (must be re-entrant: config updates re-run initialize())
    # ------------------------------------------------------------------

    async def initialize(self):
        # Tear down any previous incarnation first (hot reload safety). The
        # internal teardown must NOT wait for detached drainers: they release
        # their own (previous) components on their own schedule, and blocking
        # initialization - and the settings request behind it - for up to
        # BG_SEND_SHUTDOWN_GRACE would reintroduce the very stall the drainer
        # exists to prevent.
        await self._shutdown_components(wait_detached_drainers=False)

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
            # This incarnation is fully built: accept new background sends
            # again (an in-place re-initialize kept _closing from the
            # teardown at the top of this method).
            self._closing = False
            logger.info("Sticker Plus initialized (capacity=%d)", self._manager.capacity)
        except Exception:
            logger.exception("Sticker Plus initialization failed (plugin degrades to no-op)")
            # Cleanup of THIS instance's half-built components; detached
            # drainers are unrelated and release themselves.
            await self._shutdown_components(wait_detached_drainers=False)

    async def terminate(self):
        await self._shutdown_components()

    @on.shutdown()
    async def _on_host_shutdown(self):
        """Mark this terminate sequence as a process exit.

        The host fires ON_SHUTDOWN in lifecycle.stop() BEFORE
        plugin_manager.terminate(); every other caller of terminate()
        (settings update via update_plugin_config, plugin disable / reload /
        uninstall) does not. _shutdown_components uses this flag to pick the
        full shutdown grace (force teardown allowed, process is exiting)
        over the short management grace + drainer handoff.
        """
        global _process_shutting_down
        _process_shutting_down = True

    async def _shutdown_components(
        self, wait_detached_drainers: bool = True
    ) -> None:
        # Refuse new background sends from this moment on: the host keeps
        # dispatching tag callbacks to this instance until init_plugin
        # re-registers them (which happens after terminate), so a turn can
        # still emit <sticker_plus> while the grace wait below yields
        # control. A send scheduled now would miss both the waiter snapshot
        # and the drainer handoff, yet run against components that are about
        # to be released. Cleared again by a successful initialize().
        self._closing = True
        # Pending records die with this instance (hot reload boundary).
        self._pending_sends.clear()
        # Give in-flight background sends a bounded grace period instead of
        # cancelling them: a sticker already scheduled should still go out,
        # and its task keeps using the (snapshot) manager and the database
        # below, so those must not be disposed underneath it.
        #
        # Two paths diverge here:
        # - Process shutdown (depth == 0): wait the full grace (also for
        #   drainers still releasing older components), then tear down even
        #   if sends remain - the process is exiting anyway.
        # - Hot reload / config update (depth > 0): the triggering HTTP
        #   request must not block for the full grace. If sends are still
        #   running when the short grace expires, detach the components and
        #   hand both to a background drainer (_drain_detached_components):
        #   the sends finish against the resources they were scheduled with,
        #   the request unblocks now, and the drainer releases the components
        #   afterwards (bounded by BG_SEND_SHUTDOWN_GRACE).
        #
        # The teardown lives in a finally block on purpose: when this wait is
        # cancelled (host shutdown deadline, process-level cancellation), the
        # teardown used to be skipped entirely - leaking exactly the stealer /
        # manager / database handles this grace period exists to protect.
        # Cancellation still propagates after the teardown runs.
        try:
            # Drainers whose loop is already gone can never settle - drop
            # them before building the waiter set.
            loop = asyncio.get_running_loop()
            for task in list(_active_drainers):
                if task.get_loop() is not loop:
                    _active_drainers.discard(task)
            # Only a process exit (ON_SHUTDOWN fired before terminate) may
            # force-tear components under still-running sends - every other
            # caller (settings update, host disable/reload/uninstall) is a
            # management action that must unblock fast and defer instead.
            process_shutdown = _process_shutting_down
            hot_reload = _hot_reload_depth > 0 and not process_shutdown
            waiters = set(self._bg_sends)
            if process_shutdown and wait_detached_drainers:
                # Sends detached by an earlier deferral are owned by their
                # drainers; at process shutdown they deserve the same
                # bounded grace as our own sends, even when this instance
                # has no in-flight send left of its own. Not waited for on
                # the management paths (unblocking fast is their whole
                # point) nor inside initialize()'s internal teardown (the
                # drainers release themselves; blocking re-initialization
                # would stall the settings request behind it).
                waiters |= set(_active_drainers)
            if waiters:
                grace = (
                    BG_SEND_SHUTDOWN_GRACE if process_shutdown
                    else HOT_RELOAD_SEND_GRACE
                )
                done, still_running = await asyncio.wait(waiters, timeout=grace)
                if still_running and not process_shutdown:
                    # Re-snapshot instead of intersecting with the pre-wait
                    # set: _closing refuses NEW schedules, but anything that
                    # slipped in between the snapshot and here must still be
                    # handed to the drainer (defense in depth).
                    self._defer_component_teardown(set(self._bg_sends))
                    # Components were detached; the finally below is a no-op.
                    return
                if still_running:
                    logger.warning(
                        "%d sticker_plus background task(s) still running after %.0fs grace; "
                        "continuing shutdown",
                        len(still_running), grace,
                    )
        finally:
            await _release_components(
                self._stealer, self._manager, self._db
            )
            self._stealer = None
            self._manager = None
            self._db = None

    def _defer_component_teardown(self, stragglers: set[asyncio.Task]) -> None:
        """Hand still-running sends and the components they use to a drainer.

        Called on the hot-reload path when the short grace expires: instead of
        disposing the manager/database underneath in-flight sends (which would
        turn a healthy send into a bogus "internal error" failure), the
        components are detached from this instance immediately and released by
        a background task once the sends settle.
        """
        logger.warning(
            "%d sticker_plus background send(s) still running after the %.0fs "
            "hot-reload grace; deferring component teardown until they settle",
            len(stragglers), HOT_RELOAD_SEND_GRACE,
        )
        drainer = asyncio.create_task(
            _drain_detached_components(
                self._stealer, self._manager, self._db, frozenset(stragglers)
            )
        )
        # Ownership transfer: the drainer now holds the strong references, so
        # these tasks must not be awaited/torn down again by this instance.
        self._stealer = None
        self._manager = None
        self._db = None
        for task in stragglers:
            self._bg_sends.discard(task)
        _active_drainers.add(drainer)
        drainer.add_done_callback(_active_drainers.discard)

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
        if "sticker" not in event.supported_elements:
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
        if self._closing:
            # Shutdown or hot reload already started on this instance: the
            # components this send would run against are being released.
            logger.warning(
                "sticker_plus skipped: instance shutdown already in progress"
            )
            return
        # Snapshot the manager here, at schedule time, and hand it to the
        # detached task: a hot reload or shutdown may clear self._manager
        # between create_task() and the task's first step, and reading
        # self._manager inside the task would then raise AttributeError on a
        # None (recorded as a bogus "internal error" send failure).
        manager = self._manager
        pending = _PendingStickerSend()
        # Prune every sid's bucket, not just this one: a turn interrupted
        # before its step-result hook never pops its bucket, so without a
        # sweep the dict keys would accumulate forever.
        for b_sid, records in list(self._pending_sends.items()):
            # Long TTL for OTHER sessions: a slow turn there (LLM + tool
            # calls) must survive this schedule (see PENDING_SWEEP_MAX_AGE).
            # The CURRENT sid keeps the short TTL: a record still sitting
            # here when the same session schedules again can only come from
            # a turn interrupted mid-send, and such a stale record must not
            # ride along into the next turn's reminder as if that message
            # had sent the sticker.
            ttl = PENDING_SWEEP_MAX_AGE if b_sid != sid else PENDING_RECORD_MAX_AGE
            records = [p for p in records if p.age_ok(ttl)]
            if records:
                self._pending_sends[b_sid] = records
            else:
                self._pending_sends.pop(b_sid, None)
        bucket = self._pending_sends.get(sid, [])
        bucket.append(pending)
        self._pending_sends[sid] = bucket
        task = asyncio.create_task(
            self._send_picked_sticker(manager, emotion, recent_context, sid, pending)
        )
        self._bg_sends.add(task)
        task.add_done_callback(self._bg_sends.discard)

    async def _send_picked_sticker(
        self, manager, emotion: str, recent_context: str, sid: str, pending: _PendingStickerSend
    ) -> None:
        """Background task: pick via the selection LLM, send directly, record.

        Runs detached from the main model's turn; the step-result hook waits
        on ``pending.done`` to attach the outcome to the assistant message.

        ``manager`` is snapshotted by the caller at schedule time instead of
        being read from ``self`` here: a shutdown or hot reload may clear
        ``self._manager`` while this detached task is still pending, and the
        send must finish against the library instance it was scheduled with
        (by design, in-flight sends are left alone on shutdown).
        """
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
        skipped_over_budget = 0
        for pending in pending_list:
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                skipped_over_budget += 1
                continue
            try:
                # Wait on the event, not the task: on timeout the task keeps
                # running and the sticker still goes out.
                await asyncio.wait_for(pending.done.wait(), timeout=remaining)
            except asyncio.TimeoutError:
                logger.warning(
                    "sticker_plus send not settled within the %.0fs batch budget; record skipped",
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
        if skipped_over_budget:
            # One summary line instead of one warning per remaining record.
            logger.warning(
                "sticker_plus batch budget (%.0fs) exhausted; %d pending record(s) skipped",
                SEND_RECORD_WAIT_TIMEOUT,
                skipped_over_budget,
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

        The module-level lock serializes config updates across instances:
        the page-side settingsQueue only orders requests from one page, and
        the host replaces this instance mid-update, so without a backend lock
        two clients (or two tabs) could still interleave terminate/init.
        """
        plugin_mgr = getattr(self.ctx, "plugin_mgr", None)
        if plugin_mgr is None or not plugin_mgr.has_plugin(PLUGIN_ID):
            raise HTTPException(status_code=503, detail="Plugin manager not available")
        global _hot_reload_depth
        async with _settings_update_lock:
            # Hot reload, not a process shutdown: keep the in-flight-send
            # grace short. The counter is module-level: this handler may run
            # on a stale instance while the host terminates the current one.
            _hot_reload_depth += 1
            try:
                updated = await plugin_mgr.update_plugin_config(
                    PLUGIN_ID, {"steal_emoji": bool(request.steal_emoji)}
                )
            finally:
                # Balanced even when the update raises or the host
                # re-initializes an instance in place: the depth must never
                # outlive the request, or a later real shutdown would use
                # the short grace.
                _hot_reload_depth -= 1
        return {"steal_emoji": bool(updated.get("steal_emoji", request.steal_emoji))}

    @register.api(method="PUT", path="/settings/steal-approval")
    async def set_steal_approval(self, request: StealApprovalSettingRequest):
        """Toggle review-before-use for stolen emojis (they then land banned).

        Same persistence pattern as set_steal: update_plugin_config
        re-initializes this plugin instance, so build the response from
        locals only - do not touch self afterwards. Same module-level lock,
        too: updates from any page/client/instance are strictly serialized.
        """
        plugin_mgr = getattr(self.ctx, "plugin_mgr", None)
        if plugin_mgr is None or not plugin_mgr.has_plugin(PLUGIN_ID):
            raise HTTPException(status_code=503, detail="Plugin manager not available")
        global _hot_reload_depth
        async with _settings_update_lock:
            # See set_steal: module-level depth, short grace, balanced in
            # finally.
            _hot_reload_depth += 1
            try:
                updated = await plugin_mgr.update_plugin_config(
                    PLUGIN_ID,
                    {"steal_require_approval": bool(request.steal_require_approval)},
                )
            finally:
                # Never let the depth outlive the update.
                _hot_reload_depth -= 1
        return {
            "steal_require_approval": bool(
                updated.get("steal_require_approval", request.steal_require_approval)
            )
        }
