"""Emoji library manager: scanning, stealing intake, VLM scheduling,
eviction and emotion-based selection.

Ported from nori_plugin_emoji/manager.py with these host adaptations:
- files are addressed by bare file name inside the emoji dir (models.path);
- ``pick_emoji`` returns the selected record + absolute file path; building
  host message elements is the caller's job (keeps this module host-free and
  unit-testable);
- eviction also runs after steal/upload additions, not only at startup;
- tagged records whose file went missing are banned (same as upstream).
"""

from __future__ import annotations

import asyncio
import hashlib
import io
import logging
import random
from pathlib import Path
from typing import Optional

from sqlalchemy.exc import IntegrityError
from PIL import Image

from .db import EmojiDatabase
from .models import EmojiImage
from .repo import EmojiRepository
from .vlm import EmojiVLM

logger = logging.getLogger(__name__)

_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"}

_FORMAT_EXTENSIONS = {
    "png": ".png",
    "gif": ".gif",
    "webp": ".webp",
    "jpeg": ".jpg",
    "jpg": ".jpg",
    "bmp": ".bmp",
}

# A row whose tagging keeps failing is banned after this many strikes, so it
# leaves the unprocessed queue instead of poisoning every batch.
_MAX_TAG_FAILURES = 3


class EmojiManager:
    def __init__(
        self,
        db: EmojiDatabase,
        vlm: EmojiVLM,
        emoji_dir: Path,
        capacity: int = 500,
        candidate_count: int = 9,
    ):
        self._db = db
        self._vlm = vlm
        self._emoji_dir = emoji_dir
        self._capacity = max(int(capacity), 1)
        self._candidate_count = max(int(candidate_count), 1)
        self._bg_tasks: set[asyncio.Task] = set()
        self._tag_lock = asyncio.Lock()

    @property
    def emoji_dir(self) -> Path:
        return self._emoji_dir

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def candidate_count(self) -> int:
        return self._candidate_count

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def startup(self) -> int:
        """Scan the emoji dir, enforce capacity, kick background tagging.

        Returns the number of newly registered files.
        """
        self._emoji_dir.mkdir(parents=True, exist_ok=True)
        new_count = await self.scan_directory()
        evicted = await self._evict_if_full()
        logger.info("EmojiManager started: +%d new, -%d evicted", new_count, evicted)
        self._spawn(self._tag_pending_background())
        return new_count

    async def shutdown(self) -> None:
        """Wait briefly for background tasks, then cancel the stragglers."""
        if not self._bg_tasks:
            return
        tasks = list(self._bg_tasks)
        done, pending = await asyncio.wait(tasks, timeout=5.0)
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        self._bg_tasks.clear()

    def _spawn(self, coro) -> None:
        task = asyncio.create_task(coro)
        self._bg_tasks.add(task)
        task.add_done_callback(self._bg_tasks.discard)

    # ------------------------------------------------------------------
    # Intake
    # ------------------------------------------------------------------

    async def scan_directory(self) -> int:
        """Register image files dropped into the emoji dir (source=manual).

        Payloads Pillow cannot identify are skipped even with an image
        extension (they would poison the tagging queue), and newly registered
        rows trigger background tagging so a manual rescan completes the
        whole import instead of leaving rows stuck at "pending".
        """
        count = 0
        async with self._db.session() as session:
            repo = EmojiRepository(session)
            for file_path in self._emoji_dir.iterdir():
                if not file_path.is_file():
                    continue
                if file_path.suffix.lower() not in _IMAGE_EXTENSIONS:
                    continue
                try:
                    data = file_path.read_bytes()
                except OSError as exc:
                    logger.warning("Failed to read emoji file %s: %s", file_path, exc)
                    continue
                if self._detect_image_extension(data) is None:
                    logger.warning("Skipping non-image file in emoji dir: %s", file_path.name)
                    continue
                file_hash = hashlib.sha256(data).hexdigest()
                if await repo.find_by_hash(file_hash) is not None:
                    continue
                await repo.add(file_hash, file_path.name, "manual")
                count += 1
            await session.commit()
        if count:
            logger.info("Directory scan registered %d new emojis", count)
            self._spawn(self._tag_pending_background())
        return count

    async def add_emoji_from_bytes(self, data: bytes, source: str = "stolen") -> bool:
        """Store raw image bytes: hash -> dedupe -> DB row -> file -> tagging.

        DB row is committed before the file is written so a failed commit
        never leaves an orphan file; concurrent inserts of the same image are
        settled by the unique hash constraint (IntegrityError = duplicate).
        Returns True when a new emoji was added.
        """
        if not data:
            return False
        extension = self._detect_image_extension(data)
        if extension is None:
            # Non-image payloads (corrupt files, video stickers, ...) would
            # fail tagging forever; reject them at the door instead.
            logger.info("Rejected non-image emoji payload (%d bytes)", len(data))
            return False
        file_hash = hashlib.sha256(data).hexdigest()
        file_name = f"{file_hash}{extension}"
        file_path = self._emoji_dir / file_name

        async with self._db.session() as session:
            repo = EmojiRepository(session)
            if await repo.find_by_hash(file_hash) is not None:
                return False
            try:
                await repo.add(file_hash, file_name, source)
                await session.commit()
            except IntegrityError:
                await session.rollback()
                return False

        try:
            self._emoji_dir.mkdir(parents=True, exist_ok=True)
            file_path.write_bytes(data)
        except OSError as exc:
            # DB row stays; missing files are banned during tagging.
            logger.warning("Failed to write emoji file %s: %s", file_path, exc)

        logger.info("Emoji added to library: hash=%s... source=%s", file_hash[:8], source)
        self._spawn(self._tag_pending_background())

        await self._evict_if_full()
        return True

    @staticmethod
    def _detect_image_extension(data: bytes) -> Optional[str]:
        """Sniff the real image format from the header (Pillow lazy open).

        Returns None when Pillow cannot identify the payload or the format is
        not one we can serve - the caller must reject such input rather than
        storing it under a fake ".png" name where it would fail tagging
        forever ("poison" rows).
        """
        try:
            with Image.open(io.BytesIO(data)) as img:
                fmt = (img.format or "").lower()
        except Exception:
            return None
        return _FORMAT_EXTENSIONS.get(fmt)

    # ------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------

    async def pick_emoji(
        self,
        emoji_hint: str,
        recent_context: str = "",
    ) -> Optional[tuple[EmojiImage, Path]]:
        """Select an emoji for the given emotion hint.

        Algorithm (ported from upstream):
        1. exclude the 3 most recently used and the 3 most used ids;
        2. sample up to 6 candidates whose emotion labels match the hint;
        3. fill the remainder with random candidates;
        4. drop candidates whose file is missing;
        5. let the VLM choose one by description text (random on failure);
        6. bump use counters.

        Returns (record, absolute file path) or None when the library has
        nothing usable.
        """
        exclude_recent = 3
        exclude_top_used = 3

        async with self._db.session() as session:
            repo = EmojiRepository(session)
            recent_ids = await repo.get_recent_used_ids(limit=exclude_recent)
            top_used_ids = await repo.get_top_used_ids(limit=exclude_top_used)
            exclude_ids = list(dict.fromkeys(recent_ids + top_used_ids))

            if exclude_ids:
                total_active = await repo.count_active()
                if total_active <= len(exclude_ids):
                    exclude_ids = []

            emotion_count = min(6, self._candidate_count)
            emotion_candidates = await repo.pick_by_emotion(
                emoji_hint, limit=emotion_count, exclude_ids=exclude_ids
            )
            random_count = self._candidate_count - emotion_count
            emotion_ids = [c.id for c in emotion_candidates]
            random_candidates = await repo.pick_random(
                limit=random_count, exclude_ids=exclude_ids + emotion_ids
            )
            candidates = list(emotion_candidates) + list(random_candidates)

            if len(candidates) < self._candidate_count:
                existing_ids = {c.id for c in candidates}
                need = self._candidate_count - len(candidates)
                extra = await repo.pick_random(
                    limit=need, exclude_ids=exclude_ids + list(existing_ids)
                )
                candidates.extend(extra)

            if not candidates:
                logger.warning("Emoji library empty, nothing to pick")
                return None

            valid: list[EmojiImage] = []
            descriptions: list[str] = []
            for candidate in candidates:
                file_path = self._emoji_dir / candidate.path
                if not file_path.exists():
                    logger.warning("Emoji file missing: %s", file_path)
                    continue
                valid.append(candidate)
                descriptions.append(
                    candidate.description or candidate.emotions or "(无描述)"
                )
            if not valid:
                logger.warning("All emoji candidate files missing")
                return None

        # VLM selection runs outside the DB transaction (network IO).
        try:
            idx, reason = await self._vlm.select_emoji_by_description(
                descriptions, emoji_hint, recent_context
            )
            selected = valid[idx]
            logger.info("VLM picked emoji #%d: %s", selected.id, reason)
        except Exception as exc:
            logger.warning("VLM selection failed, falling back to random: %s", exc)
            selected = random.choice(valid)

        async with self._db.session() as session:
            repo = EmojiRepository(session)
            await repo.increment_use(selected.id)
            await session.commit()

        return selected, self._emoji_dir / selected.path

    # ------------------------------------------------------------------
    # Tagging
    # ------------------------------------------------------------------

    async def tag_pending(self, batch_size: int = 10) -> int:
        """Tag up to batch_size unprocessed emojis; returns tagged count.

        Failures are counted per row; a row is auto-banned after
        ``_MAX_TAG_FAILURES`` strikes so permanently broken files leave the
        queue instead of being retried forever.
        """
        tagged = 0
        # Short read-only transaction: collect work, ban missing files.
        pending_work: list[tuple[EmojiImage, Path]] = []
        async with self._db.session() as session:
            repo = EmojiRepository(session)
            for emoji in await repo.get_unprocessed(limit=batch_size):
                file_path = self._emoji_dir / emoji.path
                if not file_path.exists():
                    await repo.ban(emoji.id)
                    continue
                pending_work.append((emoji, file_path))
            await session.commit()

        # VLM calls outside the transaction.
        tagged_rows: list[tuple[int, str, str]] = []
        failed_ids: list[int] = []
        for emoji, file_path in pending_work:
            try:
                description, emotions = await self._vlm.tag_emoji(file_path)
                tagged_rows.append((emoji.id, description, emotions))
                tagged += 1
            except Exception as exc:
                logger.warning("Tagging failed for #%s: %s", emoji.id, exc)
                failed_ids.append(emoji.id)

        if tagged_rows or failed_ids:
            async with self._db.session() as session:
                repo = EmojiRepository(session)
                for emoji_id, description, emotions in tagged_rows:
                    await repo.tag(emoji_id, description, emotions)
                for emoji_id in failed_ids:
                    if await repo.record_tag_failure(emoji_id, _MAX_TAG_FAILURES):
                        logger.warning(
                            "Emoji #%s banned after %d tagging failures",
                            emoji_id, _MAX_TAG_FAILURES,
                        )
                await session.commit()
        return tagged

    async def _tag_pending_background(self) -> None:
        """Serialized fire-and-forget tagging; drains the whole backlog.

        Keeps batching while progress is made and stops on the first
        empty/unproductive batch (queue drained, or only rows that keep
        failing - those accumulate strikes and get banned).
        """
        async with self._tag_lock:
            try:
                while True:
                    tagged = await self.tag_pending(batch_size=20)
                    if tagged:
                        logger.info("Background tagging batch finished: %d emojis", tagged)
                    if tagged == 0:
                        break
            except Exception as exc:
                logger.warning("Background tagging error: %s", exc)

    async def retag(self, emoji_id: int) -> bool:
        """Force a fresh VLM tagging for one emoji (WebUI action)."""
        async with self._db.session() as session:
            repo = EmojiRepository(session)
            emoji = await repo.get(emoji_id)
            if emoji is None:
                return False
            file_path = self._emoji_dir / emoji.path
        if not file_path.exists():
            return False
        # VLM call outside any session scope (same convention as tag/pick);
        # only the short read and the short write-back touch the database.
        description, emotions = await self._vlm.tag_emoji(file_path)
        async with self._db.session() as session:
            repo = EmojiRepository(session)
            await repo.tag(emoji_id, description, emotions)
            await session.commit()
        return True

    # ------------------------------------------------------------------
    # Eviction
    # ------------------------------------------------------------------

    async def enforce_capacity(self) -> int:
        """Public eviction entry point (startup / after adds / manual rescan)."""
        return await self._evict_if_full()

    async def _evict_if_full(self) -> int:
        """Evict over-capacity rows, then unlink their files."""
        async with self._db.session() as session:
            repo = EmojiRepository(session)
            victims = await repo.evict_if_full(self._capacity)
            await session.commit()
        for victim in victims:
            try:
                (self._emoji_dir / victim.path).unlink(missing_ok=True)
            except OSError as exc:
                logger.warning("Failed to delete emoji file %s: %s", victim.path, exc)
        return len(victims)

    # ------------------------------------------------------------------
    # WebUI operations
    # ------------------------------------------------------------------

    async def list_emojis(
        self, status: str = "all", search: str = "", offset: int = 0, limit: int = 60
    ) -> tuple[list[dict], int]:
        async with self._db.session() as session:
            repo = EmojiRepository(session)
            rows, total = await repo.list_emojis(
                status=status, search=search, offset=offset, limit=limit
            )
            return [row.to_dict() for row in rows], total

    async def get_emoji(self, emoji_id: int) -> Optional[dict]:
        async with self._db.session() as session:
            repo = EmojiRepository(session)
            emoji = await repo.get(emoji_id)
            return emoji.to_dict() if emoji else None

    async def emoji_file(self, emoji_id: int) -> Optional[Path]:
        async with self._db.session() as session:
            repo = EmojiRepository(session)
            emoji = await repo.get(emoji_id)
            if emoji is None:
                return None
            file_path = self._emoji_dir / emoji.path
            return file_path if file_path.exists() else None

    async def update_emoji(
        self,
        emoji_id: int,
        description: str | None = None,
        emotions: str | None = None,
        is_banned: bool | None = None,
    ) -> Optional[dict]:
        async with self._db.session() as session:
            repo = EmojiRepository(session)
            emoji = await repo.update_fields(
                emoji_id, description=description, emotions=emotions, is_banned=is_banned
            )
            await session.commit()
            return emoji.to_dict() if emoji else None

    async def delete_emoji(self, emoji_id: int) -> bool:
        async with self._db.session() as session:
            repo = EmojiRepository(session)
            emoji = await repo.delete(emoji_id)
            await session.commit()
        if emoji is None:
            return False
        try:
            (self._emoji_dir / emoji.path).unlink(missing_ok=True)
        except OSError as exc:
            logger.warning("Failed to delete emoji file %s: %s", emoji.path, exc)
        return True

    async def stats(self) -> dict:
        async with self._db.session() as session:
            return await EmojiRepository(session).stats()
