"""Manager-level tests: scan, intake dedupe/eviction, pick flow, tagging."""

from __future__ import annotations

import asyncio
import hashlib

import pytest
import pytest_asyncio

from helpers import (
    EmojiDatabase,
    EmojiManager,
    EmojiRepository,
    FailingTagVLM,
    FakeTagVLM,
    make_png_bytes,
)


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
async def test_add_rejects_non_image_payload(env):
    # e.g. a Telegram video sticker (webm) - must not become a poison row
    assert await env.add_emoji_from_bytes(b"definitely not an image", "stolen") is False
    _, total = await env.list_emojis()
    assert total == 0


@pytest.mark.asyncio
async def test_scan_skips_corrupt_image_files(env, tmp_path):
    emoji_dir = tmp_path / "emojis"
    emoji_dir.mkdir()
    (emoji_dir / "good.png").write_bytes(make_png_bytes(color=(6, 6, 6)))
    (emoji_dir / "fake.png").write_bytes(b"garbage with an image extension")
    assert await env.scan_directory() == 1
    _, total = await env.list_emojis()
    assert total == 1


@pytest.mark.asyncio
async def test_scan_triggers_background_tagging(env, tmp_path):
    emoji_dir = tmp_path / "emojis"
    emoji_dir.mkdir()
    (emoji_dir / "one.png").write_bytes(make_png_bytes(color=(3, 3, 3)))
    (emoji_dir / "two.png").write_bytes(make_png_bytes(color=(4, 4, 4)))
    assert await env.scan_directory() == 2
    assert env._bg_tasks, "scan must spawn background tagging for new rows"
    await asyncio.gather(*list(env._bg_tasks), return_exceptions=True)
    stats = await env.stats()
    assert stats["active"] == 2


@pytest.mark.asyncio
async def test_background_tagging_drains_backlog(env):
    # 25 rows inserted directly (no per-add spawn): one background run must
    # drain the whole backlog, not just the first batch of 20.
    env.emoji_dir.mkdir(parents=True, exist_ok=True)
    async with env._db.session() as session:
        repo = EmojiRepository(session)
        for i in range(25):
            data = make_png_bytes(color=(i, 50 + i, 100 + i))
            (env.emoji_dir / f"drain{i}.png").write_bytes(data)
            await repo.add(hashlib.sha256(data).hexdigest(), f"drain{i}.png", "manual")
        await session.commit()
    await env._tag_pending_background()
    stats = await env.stats()
    assert stats["active"] == 25


@pytest.mark.asyncio
async def test_tagging_failures_auto_ban(env):
    failing = EmojiManager(
        db=env._db, vlm=FailingTagVLM(), emoji_dir=env.emoji_dir, capacity=50, candidate_count=9
    )
    try:
        assert await failing.add_emoji_from_bytes(make_png_bytes(color=(77, 7, 7)), "stolen")
        # per-add spawn already consumed one strike; run two more rounds
        await asyncio.gather(*list(failing._bg_tasks), return_exceptions=True)
        await failing._tag_pending_background()
        await failing._tag_pending_background()
        stats = await failing.stats()
        assert stats["banned"] == 1
        assert stats["active"] == 0
        async with failing._db.session() as session:
            pending = await EmojiRepository(session).get_unprocessed(limit=10)
        assert pending == []  # banned row left the queue
    finally:
        await failing.shutdown()


@pytest.mark.asyncio
async def test_retag_rewrites_row(env):
    data = make_png_bytes(color=(11, 11, 11))
    await env.add_emoji_from_bytes(data, "stolen")
    items, _ = await env.list_emojis()
    emoji_id = items[0]["id"]
    assert await env.retag(emoji_id) is True
    fresh = await env.get_emoji(emoji_id)
    assert fresh["vlm_processed"] is True
    assert fresh["description"].startswith("desc of ")
    assert await env.retag(99999) is False


@pytest.mark.asyncio
async def test_retag_missing_file_returns_false(env):
    data = make_png_bytes(color=(12, 12, 12))
    await env.add_emoji_from_bytes(data, "stolen")
    items, _ = await env.list_emojis()
    (env.emoji_dir / items[0]["path"]).unlink()
    assert await env.retag(items[0]["id"]) is False


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
