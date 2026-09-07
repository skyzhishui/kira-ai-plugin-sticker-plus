"""Stealer tests: element extraction, self-message skip, size limits."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import pytest_asyncio

from core.chat.message_elements import Sticker, Text

from helpers import EmojiStealer, FakeStealManager, b64, make_png_bytes


def make_event(chain, sender_id="10001", self_id="99999"):
    return SimpleNamespace(
        message=SimpleNamespace(
            chain=chain,
            sender=SimpleNamespace(user_id=sender_id),
            self_id=self_id,
        )
    )


@pytest_asyncio.fixture
async def env():
    manager = FakeStealManager()
    stealer = EmojiStealer(emoji_manager=manager, max_size_mb=5.0)
    yield stealer, manager
    await stealer.shutdown()


@pytest.mark.asyncio
async def test_steals_sticker_base64(env):
    stealer, manager = env
    png = make_png_bytes()
    event = make_event([Text("看这个"), Sticker(sticker=b64(png))])
    await stealer.handle_event(event)
    await stealer.shutdown()
    assert manager.calls == [png]


@pytest.mark.asyncio
async def test_ignores_self_messages(env):
    stealer, manager = env
    png = make_png_bytes()
    event = make_event([Sticker(sticker=b64(png))], sender_id="99999", self_id="99999")
    await stealer.handle_event(event)
    await stealer.shutdown()
    assert manager.calls == []


@pytest.mark.asyncio
async def test_size_limit(env):
    stealer, manager = env
    small = EmojiStealer(emoji_manager=manager, max_size_mb=0.00001)
    event = make_event([Sticker(sticker=b64(make_png_bytes()))])
    await small.handle_event(event)
    await small.shutdown()
    assert manager.calls == []


@pytest.mark.asyncio
async def test_url_sticker_downloaded(env, monkeypatch):
    stealer, manager = env
    png = make_png_bytes(color=(0, 0, 255))

    async def fake_download(url):
        assert url == "https://example.com/e.gif"
        return png

    monkeypatch.setattr(stealer, "_download_image", fake_download)
    event = make_event([Sticker(sticker="https://example.com/e.gif")])
    await stealer.handle_event(event)
    await stealer.shutdown()
    assert manager.calls == [png]


@pytest.mark.asyncio
async def test_broken_payload_ignored(env):
    stealer, manager = env
    # not base64, not url, not a path -> Sticker() itself would raise in
    # check_file_type, so use a valid base64 of non-image data instead
    event = make_event([Sticker(sticker=b64(b"junk-not-an-image"))])
    await stealer.handle_event(event)
    await stealer.shutdown()
    # still stored (intake validates via Pillow and falls back to .png);
    # extraction-level failures are what must never raise
    assert len(manager.calls) == 1
