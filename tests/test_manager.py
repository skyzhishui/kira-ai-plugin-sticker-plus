"""Manager-level tests: scan, intake dedupe/eviction, pick flow, tagging."""

from __future__ import annotations

import hashlib

import pytest
import pytest_asyncio

from helpers import EmojiDatabase, EmojiManager, FakeTagVLM, make_png_bytes


@pytest_asyncio.fixture
async def env(tmp_path):
    db = EmojiDatabase(tmp_path / "test.db")
    await db.create_tables()
    manager = EmojiManager(
        db=db, vlm=FakeTagVLM(), emoji_dir=tmp_path / "emojis", capacity=50, candidate_count=9
    )
    yield manager
    await manager.shutdown()
    await db.dispose()


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@pytest.mark.asyncio
async def test_scan_directory_registers_and_dedupes(env, tmp_path):
    (tmp_path / "emojis").mkdir()
    (tmp_path / "emojis" / "one.png").write_bytes(make_png_bytes(color=(1, 1, 1)))
    assert await env.scan_directory() == 1
    # second scan: existing file deduped by hash, only the new file registers
    (tmp_path / "emojis" / "two.png").write_bytes(make_png_bytes(color=(2, 2, 2)))
    assert await env.scan_directory() == 1
    rows, total = await env.list_emojis()
    assert total == 2
    # non-image files ignored
    (tmp_path / "emojis" / "notes.txt").write_text("hi")
    assert await env.scan_directory() == 0


@pytest.mark.asyncio
async def test_add_emoji_from_bytes_dedupe_and_file(env):
    data = make_png_bytes(color=(0, 255, 0))
    assert await env.add_emoji_from_bytes(data, "stolen") is True
    file_name = f"{_sha(data)}.png"
    assert (env.emoji_dir / file_name).exists()
    # exact duplicate -> skipped
    assert await env.add_emoji_from_bytes(data, "stolen") is False
    items, total = await env.list_emojis()
    assert total == 1
    record = items[0]
    assert record["source"] == "stolen"


@pytest.mark.asyncio
async def test_add_detects_real_format(env):
    gif = make_png_bytes(fmt="GIF")
    await env.add_emoji_from_bytes(gif, "stolen")
    assert (env.emoji_dir / f"{_sha(gif)}.gif").exists()


@pytest.mark.asyncio
async def test_capacity_eviction_removes_file(env, tmp_path):
    small = EmojiManager(
        db=env._db, vlm=env._vlm, emoji_dir=tmp_path / "cap", capacity=1, candidate_count=3
    )
    first = make_png_bytes(color=(1, 2, 3))
    second = make_png_bytes(color=(4, 5, 6))
    await small.add_emoji_from_bytes(first, "stolen")
    await small.add_emoji_from_bytes(second, "stolen")
    _, total = await small.list_emojis()
    assert total == 1
    files = list((tmp_path / "cap").glob("*"))
    assert len(files) == 1 and files[0].name == f"{_sha(second)}.png"
    await small.shutdown()


@pytest.mark.asyncio
async def test_pick_emoji_flow_and_use_count(env):
    for color in [(10, 10, 10), (20, 20, 20), (30, 30, 30)]:
        await env.add_emoji_from_bytes(make_png_bytes(color=color), "stolen")
    # FakeTagVLM tags everything synchronously? No - tagging is a background
    # task; tag synchronously here instead.
    await env.tag_pending(batch_size=10)
    picked = await env.pick_emoji("开心", "ctx")
    assert picked is not None
    record, path = picked
    assert path.exists()
    stats = await env.stats()
    assert stats["active"] == 3
    # use counter bumped
    assert record.use_count >= 0  # record was read before increment; check fresh row
    fresh = await env.get_emoji(record.id)
    assert fresh["use_count"] == 1


@pytest.mark.asyncio
async def test_pick_empty_library_returns_none(env):
    assert await env.pick_emoji("开心") is None


@pytest.mark.asyncio
async def test_tag_pending_bans_missing_files(env, tmp_path):
    data = make_png_bytes(color=(9, 9, 9))
    await env.add_emoji_from_bytes(data, "stolen")
    # steal the file away before tagging runs
    for f in (tmp_path / "emojis").glob("*"):
        f.unlink()
    tagged = await env.tag_pending(batch_size=5)
    assert tagged == 0
    stats = await env.stats()
    assert stats["banned"] == 1


@pytest.mark.asyncio
async def test_delete_emoji_removes_file(env):
    data = make_png_bytes(color=(5, 5, 5))
    await env.add_emoji_from_bytes(data, "stolen")
    items, _ = await env.list_emojis()
    emoji_id = items[0]["id"]
    file_name = items[0]["path"]
    assert await env.delete_emoji(emoji_id) is True
    assert not (env.emoji_dir / file_name).exists()
    assert await env.delete_emoji(emoji_id) is False
