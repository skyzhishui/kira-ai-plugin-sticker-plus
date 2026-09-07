"""Repository-level tests: CRUD, sampling, exclusion, eviction, listing."""

from __future__ import annotations

import pytest
import pytest_asyncio

from helpers import EmojiDatabase, EmojiRepository


@pytest_asyncio.fixture
async def db(tmp_path):
    database = EmojiDatabase(tmp_path / "test.db")
    await database.create_tables()
    yield database
    await database.dispose()


async def _seed(db, rows):
    """rows: list of dicts with tag/ban/use overrides."""
    from datetime import datetime, timedelta

    async with db.session() as session:
        repo = EmojiRepository(session)
        ids = []
        base = datetime(2026, 1, 1)
        for i, spec in enumerate(rows):
            emoji = await repo.add(spec["hash"], spec["path"], spec.get("source", "manual"))
            ids.append(emoji.id)
            if spec.get("tag"):
                await repo.tag(emoji.id, spec["tag"], spec.get("emotions", "开心"))
            if spec.get("banned"):
                await repo.ban(emoji.id)
        await session.commit()
    return ids


@pytest.mark.asyncio
async def test_add_and_find_by_hash(db):
    async with db.session() as session:
        repo = EmojiRepository(session)
        emoji = await repo.add("a" * 64, "a.png", "manual")
        await session.commit()
        assert emoji.id is not None
    async with db.session() as session:
        repo = EmojiRepository(session)
        found = await repo.find_by_hash("a" * 64)
        assert found is not None and found.path == "a.png"
        assert await repo.find_by_hash("b" * 64) is None


@pytest.mark.asyncio
async def test_update_fields_marks_processed(db):
    ids = await _seed(db, [{"hash": "a" * 64, "path": "a.png"}])
    async with db.session() as session:
        repo = EmojiRepository(session)
        emoji = await repo.update_fields(ids[0], description="manually edited", emotions="开心")
        await session.commit()
        assert emoji.vlm_processed is True
        assert emoji.description == "manually edited"
    async with db.session() as session:
        repo = EmojiRepository(session)
        # Manual edits make an emoji selectable without a VLM round.
        picks = await repo.pick_random(limit=5)
        assert [p.path for p in picks] == ["a.png"]


@pytest.mark.asyncio
async def test_pick_by_emotion_like_match_and_exclusion(db):
    await _seed(db, [
        {"hash": "a" * 64, "path": "a.png", "tag": True, "emotions": "开心,调皮"},
        {"hash": "b" * 64, "path": "b.png", "tag": True, "emotions": "悲伤"},
        {"hash": "c" * 64, "path": "c.png", "tag": True, "emotions": "开心"},
        {"hash": "d" * 64, "path": "d.png", "tag": False},
    ])
    async with db.session() as session:
        repo = EmojiRepository(session)
        picks = await repo.pick_by_emotion("开心", limit=10)
        paths = {p.path for p in picks}
        assert paths == {"a.png", "c.png"}
        picks = await repo.pick_by_emotion("开心", limit=10, exclude_ids=[1, 2, 3, 4])
        assert picks == []


@pytest.mark.asyncio
async def test_recent_and_top_used(db):
    from datetime import datetime

    ids = await _seed(db, [
        {"hash": "a" * 64, "path": "a.png", "tag": True},
        {"hash": "b" * 64, "path": "b.png", "tag": True},
        {"hash": "c" * 64, "path": "c.png", "tag": True},
    ])
    async with db.session() as session:
        repo = EmojiRepository(session)
        await repo.increment_use(ids[0])
        await repo.increment_use(ids[0])
        await repo.increment_use(ids[1])
        await session.commit()
        top = await repo.get_top_used_ids(limit=1)
        assert top == [ids[0]]
        recent = await repo.get_recent_used_ids(limit=1)
        # ids[1] was used last -> most recent
        assert recent == [ids[1]]


@pytest.mark.asyncio
async def test_evict_order_prefers_tagged_low_use(db):
    # a: tagged, use 0; b: tagged, use 5; c: untagged (fresh steal)
    ids = await _seed(db, [
        {"hash": "a" * 64, "path": "a.png", "tag": True},
        {"hash": "b" * 64, "path": "b.png", "tag": True},
        {"hash": "c" * 64, "path": "c.png", "tag": False},
    ])
    async with db.session() as session:
        repo = EmojiRepository(session)
        emoji = await repo.get(ids[1])
        emoji.use_count = 5
        await session.commit()
    async with db.session() as session:
        repo = EmojiRepository(session)
        victims = await repo.evict_if_full(capacity=2)
        await session.commit()
        assert [v.path for v in victims] == ["a.png"]  # tagged + least used
    async with db.session() as session:
        repo = EmojiRepository(session)
        assert await repo.count_all() == 2


@pytest.mark.asyncio
async def test_list_emojis_filters_search_pagination(db):
    await _seed(db, [
        {"hash": "a" * 64, "path": "a.png", "tag": True, "tag_desc": ""},
        {"hash": "b" * 64, "path": "b.png", "tag": False},
        {"hash": "c" * 64, "path": "c.png", "tag": True, "banned": True},
    ])
    async with db.session() as session:
        repo = EmojiRepository(session)
        # fix descriptions for search
        emoji = await repo.find_by_hash("a" * 64)
        emoji.description = "a cat waving"
        await session.commit()
    async with db.session() as session:
        repo = EmojiRepository(session)
        items, total = await repo.list_emojis(status="all")
        assert total == 3
        _, total = await repo.list_emojis(status="banned")
        assert total == 1
        _, total = await repo.list_emojis(status="pending")
        assert total == 1
        items, total = await repo.list_emojis(search="cat")
        assert total == 1 and items[0].path == "a.png"
        items, total = await repo.list_emojis(offset=0, limit=2)
        assert total == 3 and len(items) == 2
        # newest first
        assert items[0].path == "c.png"


@pytest.mark.asyncio
async def test_stats(db):
    await _seed(db, [
        {"hash": "a" * 64, "path": "a.png", "tag": True, "banned": True},
        {"hash": "b" * 64, "path": "b.png", "tag": True, "source": "stolen"},
        {"hash": "c" * 64, "path": "c.png", "tag": False},
    ])
    async with db.session() as session:
        stats = await EmojiRepository(session).stats()
    assert stats == {"total": 3, "active": 1, "pending": 1, "banned": 1, "stolen": 1}
