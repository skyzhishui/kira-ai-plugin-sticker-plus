"""Emoji library data-access layer.

Ported from nori_plugin_emoji/repo.py, but without the nori BaseRepository
base class: every method operates directly on an AsyncSession and never
commits - transaction boundaries stay with the caller (EmojiManager), same
convention as upstream.

``func.random()`` resolves to SQLite ``random()`` which exists, so the
sampling queries work unchanged on SQLite.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from .models import EmojiImage


class EmojiRepository:
    def __init__(self, session: AsyncSession):
        self._session = session

    # ------------------------------------------------------------------
    # Write operations (caller commits)
    # ------------------------------------------------------------------

    async def add(self, file_hash: str, path: str, source: str) -> EmojiImage:
        """Create an untagged emoji record; flush so ``id`` is available."""
        emoji = EmojiImage(
            hash=file_hash,
            path=path,
            source=source,
            use_count=0,
            is_banned=False,
            vlm_processed=False,
        )
        self._session.add(emoji)
        await self._session.flush()
        return emoji

    async def tag(self, emoji_id: int, description: str, emotions: str) -> None:
        """Write back a VLM tagging result (resets the failure counter)."""
        emoji = await self._session.get(EmojiImage, emoji_id)
        if emoji is None:
            return
        emoji.description = description
        emoji.emotions = emotions
        emoji.vlm_processed = True
        emoji.tag_fail_count = 0

    async def record_tag_failure(self, emoji_id: int, max_failures: int) -> bool:
        """Count one tagging failure; auto-ban after ``max_failures`` strikes.

        Banned rows leave the unprocessed queue (see ``get_unprocessed``) so a
        permanently broken file cannot starve the tagging pipeline. Returns
        True when this call banned the row.
        """
        emoji = await self._session.get(EmojiImage, emoji_id)
        if emoji is None:
            return False
        emoji.tag_fail_count = int(emoji.tag_fail_count or 0) + 1
        if emoji.tag_fail_count >= max_failures:
            emoji.is_banned = True
            return True
        return False

    async def update_fields(
        self,
        emoji_id: int,
        description: str | None = None,
        emotions: str | None = None,
        is_banned: bool | None = None,
    ) -> EmojiImage | None:
        """Manual edit from the WebUI.

        Setting description/emotions manually marks the emoji as processed so
        it becomes selectable without a VLM round.
        """
        emoji = await self._session.get(EmojiImage, emoji_id)
        if emoji is None:
            return None
        if description is not None:
            emoji.description = description
            emoji.vlm_processed = True
        if emotions is not None:
            emoji.emotions = emotions
            emoji.vlm_processed = True
        if is_banned is not None:
            emoji.is_banned = bool(is_banned)
        await self._session.flush()
        return emoji

    async def delete(self, emoji_id: int) -> EmojiImage | None:
        """Delete a record; returns it (with path) so the caller can unlink the file."""
        emoji = await self._session.get(EmojiImage, emoji_id)
        if emoji is None:
            return None
        await self._session.delete(emoji)
        await self._session.flush()
        return emoji

    async def increment_use(self, emoji_id: int) -> None:
        """Bump use counter and refresh last_used_at (naive local time)."""
        emoji = await self._session.get(EmojiImage, emoji_id)
        if emoji is None:
            return
        emoji.use_count = emoji.use_count + 1
        emoji.last_used_at = datetime.now()

    async def ban(self, emoji_id: int) -> None:
        emoji = await self._session.get(EmojiImage, emoji_id)
        if emoji is not None:
            emoji.is_banned = True

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    async def get(self, emoji_id: int) -> EmojiImage | None:
        return await self._session.get(EmojiImage, emoji_id)

    async def find_by_hash(self, file_hash: str) -> EmojiImage | None:
        result = await self._session.execute(
            select(EmojiImage).where(EmojiImage.hash == file_hash)
        )
        return result.scalar_one_or_none()

    async def get_recent_used_ids(self, limit: int = 3) -> list[int]:
        """Recently used emoji ids (last_used_at desc, NULL excluded)."""
        result = await self._session.execute(
            select(EmojiImage.id)
            .where(
                EmojiImage.vlm_processed.is_(True),
                EmojiImage.is_banned.is_(False),
                EmojiImage.last_used_at.is_not(None),
            )
            .order_by(EmojiImage.last_used_at.desc())
            .limit(limit)
        )
        return list(result.scalars().all())

    async def get_top_used_ids(self, limit: int = 3) -> list[int]:
        """Most used emoji ids (use_count desc, only use_count > 0)."""
        result = await self._session.execute(
            select(EmojiImage.id)
            .where(
                EmojiImage.vlm_processed.is_(True),
                EmojiImage.is_banned.is_(False),
                EmojiImage.use_count > 0,
            )
            .order_by(EmojiImage.use_count.desc())
            .limit(limit)
        )
        return list(result.scalars().all())

    async def pick_by_emotion(
        self, emotion: str, limit: int = 9, exclude_ids: list[int] | None = None
    ) -> list[EmojiImage]:
        """Random sample tagged with the given emotion keyword (LIKE match)."""
        conditions = [
            EmojiImage.vlm_processed.is_(True),
            EmojiImage.is_banned.is_(False),
            EmojiImage.emotions.like(f"%{emotion}%"),
        ]
        if exclude_ids:
            conditions.append(EmojiImage.id.notin_(exclude_ids))
        result = await self._session.execute(
            select(EmojiImage)
            .where(*conditions)
            .order_by(func.random())
            .limit(limit)
        )
        return list(result.scalars().all())

    async def pick_random(
        self, limit: int = 9, exclude_ids: list[int] | None = None
    ) -> list[EmojiImage]:
        """Pure random sample of selectable emojis."""
        conditions = [
            EmojiImage.vlm_processed.is_(True),
            EmojiImage.is_banned.is_(False),
        ]
        if exclude_ids:
            conditions.append(EmojiImage.id.notin_(exclude_ids))
        result = await self._session.execute(
            select(EmojiImage)
            .where(*conditions)
            .order_by(func.random())
            .limit(limit)
        )
        return list(result.scalars().all())

    async def count_active(self) -> int:
        """Selectable count (tagged + not banned)."""
        result = await self._session.execute(
            select(func.count())
            .select_from(EmojiImage)
            .where(
                EmojiImage.vlm_processed.is_(True),
                EmojiImage.is_banned.is_(False),
            )
        )
        return int(result.scalar() or 0)

    async def count_all(self) -> int:
        result = await self._session.execute(
            select(func.count()).select_from(EmojiImage)
        )
        return int(result.scalar() or 0)

    async def list_emojis(
        self,
        status: str = "all",
        search: str = "",
        offset: int = 0,
        limit: int = 60,
    ) -> tuple[list[EmojiImage], int]:
        """Paged listing for the WebUI.

        status: all | active | banned | pending | stolen
        search: substring match on description / emotions / hash prefix.
        """
        conditions = []
        if status == "active":
            conditions.append(EmojiImage.is_banned.is_(False))
        elif status == "banned":
            conditions.append(EmojiImage.is_banned.is_(True))
        elif status == "pending":
            conditions.append(EmojiImage.vlm_processed.is_(False))
        elif status == "stolen":
            conditions.append(EmojiImage.source == "stolen")
        if search:
            like = f"%{search}%"
            conditions.append(
                or_(
                    EmojiImage.description.like(like),
                    EmojiImage.emotions.like(like),
                    EmojiImage.hash.like(like),
                )
            )
        total_result = await self._session.execute(
            select(func.count()).select_from(EmojiImage).where(*conditions)
        )
        total = int(total_result.scalar() or 0)
        result = await self._session.execute(
            select(EmojiImage)
            .where(*conditions)
            .order_by(EmojiImage.id.desc())
            .offset(max(offset, 0))
            .limit(max(limit, 1))
        )
        return list(result.scalars().all()), total

    async def stats(self) -> dict:
        """Library counters for the WebUI header."""
        async def _count(*conditions) -> int:
            result = await self._session.execute(
                select(func.count()).select_from(EmojiImage).where(*conditions)
            )
            return int(result.scalar() or 0)

        return {
            "total": await _count(),
            "active": await _count(
                EmojiImage.vlm_processed.is_(True), EmojiImage.is_banned.is_(False)
            ),
            "pending": await _count(EmojiImage.vlm_processed.is_(False)),
            "banned": await _count(EmojiImage.is_banned.is_(True)),
            "stolen": await _count(EmojiImage.source == "stolen"),
        }

    async def get_unprocessed(self, limit: int = 50) -> list[EmojiImage]:
        """Untagged, not-banned emojis for the background tagger (oldest first).

        Banned rows are excluded: missing files and repeated tagging failures
        get banned, and they must stop occupying queue slots.
        """
        result = await self._session.execute(
            select(EmojiImage)
            .where(EmojiImage.vlm_processed.is_(False), EmojiImage.is_banned.is_(False))
            .order_by(EmojiImage.created_at)
            .limit(limit)
        )
        return list(result.scalars().all())

    # ------------------------------------------------------------------
    # Eviction
    # ------------------------------------------------------------------

    async def evict_if_full(self, capacity: int) -> list[EmojiImage]:
        """Evict least valuable records over capacity (DB side only).

        Priority (ported from upstream):
        1. already-tagged rows evicted first, protecting freshly stolen ones;
        2. lowest use_count first;
        3. oldest last_used_at first (SQLite ASC puts NULL first = oldest);
        4. oldest created_at / id as deterministic tiebreakers, so when the
           keys above all tie (e.g. a full library of never-used untagged
           rows) a freshly added row is never the victim.
        """
        total = await self.count_all()
        if total <= capacity:
            return []
        excess = total - capacity
        result = await self._session.execute(
            select(EmojiImage)
            .order_by(
                EmojiImage.vlm_processed.desc(),
                EmojiImage.use_count.asc(),
                EmojiImage.last_used_at.asc(),
                EmojiImage.created_at.asc(),
                EmojiImage.id.asc(),
            )
            .limit(excess)
        )
        victims = list(result.scalars().all())
        for victim in victims:
            await self._session.delete(victim)
        await self._session.flush()
        return victims
