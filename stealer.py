"""Steal emojis from incoming chat messages.

Ported from nori_plugin_emoji/stealer.py. In KiraAI the equivalent of the
upstream ``nori.receive.before_process`` hook is ``@on.im_message`` (earliest
event, raw adapter-converted elements): QQ market emojis already arrive as
``Sticker`` elements carrying base64, so the download fallback only matters
for adapters that hand over URLs.

Only ``Sticker`` elements are stolen (market emojis / stickers), never plain
``Image`` elements - that classification is done by the adapters (QQ:
sub_type=1 or "[动画表情]" summary).
"""

from __future__ import annotations

import asyncio
import base64
import logging
from typing import Optional

import httpx

logger = logging.getLogger(__name__)


class EmojiStealer:
    def __init__(self, emoji_manager, max_size_mb: float = 5.0):
        self.emoji_manager = emoji_manager
        self.max_size_bytes = int(max_size_mb * 1024 * 1024)
        self._bg_tasks: set[asyncio.Task] = set()
        # One shared client for the URL fallback path (connection reuse).
        self._client = httpx.AsyncClient(timeout=30.0, follow_redirects=True)
        self._download_semaphore = asyncio.Semaphore(4)

    async def shutdown(self) -> None:
        if self._bg_tasks:
            tasks = list(self._bg_tasks)
            done, pending = await asyncio.wait(tasks, timeout=5.0)
            for task in pending:
                task.cancel()
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)
            self._bg_tasks.clear()
        await self._client.aclose()

    # ------------------------------------------------------------------
    # Event intake (called from the plugin's @on.im_message hook)
    # ------------------------------------------------------------------

    async def handle_event(self, event) -> None:
        """Extract Sticker elements from one incoming message event.

        Never blocks the pipeline: heavy work (download/decode/store) is
        dispatched as tracked background tasks.
        """
        from core.chat.message_elements import Sticker

        message = getattr(event, "message", None)
        if message is None:
            return

        # Skip messages sent by the bot itself.
        sender_id = str(getattr(message.sender, "user_id", "") or "")
        self_id = str(getattr(message, "self_id", "") or "")
        if sender_id and sender_id == self_id:
            return

        chain = getattr(message, "chain", None) or []
        for element in chain:
            try:
                if not isinstance(element, Sticker):
                    continue
            except Exception:
                continue

            data = self._extract_bytes(element)
            if data is None:
                continue
            if len(data) > self.max_size_bytes:
                logger.debug("Steal skipped: emoji too large (%d bytes)", len(data))
                continue
            self._spawn(self._safe_add(data))

    def _extract_bytes(self, element) -> Optional[bytes]:
        """Pull image bytes out of a Sticker element (base64/data-url/url)."""
        payload = getattr(element, "sticker", None)
        if not payload:
            return None
        try:
            file_type = getattr(element, "sticker_type", None)
            if file_type == "url" and payload.startswith(("http://", "https://")):
                self._spawn(self._download_and_add(payload))
                return None
            if payload.startswith("data:"):
                payload = payload.split(",", 1)[1] if "," in payload else ""
            elif payload.startswith("base64://"):
                payload = payload.removeprefix("base64://")
            elif "," in payload and len(payload.split(",", 1)[0]) < 64:
                # data URL without the data: scheme prefix
                payload = payload.split(",", 1)[1]
            return base64.b64decode(payload, validate=False)
        except Exception as exc:
            logger.debug("Steal extract failed: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Background workers
    # ------------------------------------------------------------------

    def _spawn(self, coro) -> None:
        task = asyncio.create_task(coro)
        self._bg_tasks.add(task)
        task.add_done_callback(self._bg_tasks.discard)

    async def _download_and_add(self, url: str) -> None:
        try:
            data = await self._download_image(url)
            if data is None:
                return
            if len(data) > self.max_size_bytes:
                logger.debug("Steal skipped: downloaded emoji too large (%d bytes)", len(data))
                return
            await self._safe_add(data)
        except Exception as exc:
            logger.debug("Steal download failed: %s", exc)

    async def _download_image(self, url: str) -> Optional[bytes]:
        try:
            async with self._download_semaphore:
                response = await self._client.get(url)
                response.raise_for_status()
                return response.content
        except Exception as exc:
            logger.debug("Emoji download failed %s: %s", url[:80], exc)
            return None

    async def _safe_add(self, data: bytes) -> None:
        try:
            added = await self.emoji_manager.add_emoji_from_bytes(data, "stolen")
            if added:
                logger.info("Stole emoji into library (%d bytes)", len(data))
        except Exception as exc:
            logger.debug("Stolen emoji intake failed: %s", exc)
