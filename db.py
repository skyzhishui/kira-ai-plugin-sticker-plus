"""Plugin-owned SQLite storage (no extra dependencies: sqlalchemy + aiosqlite
are already host requirements).

Kept fully independent from the host database on purpose:
- the emoji library is plugin-local state, wiping ``data/plugin_data/<id>``
  resets it completely;
- the host DatabaseService API is model-bound and not meant for plugin tables.
"""

from __future__ import annotations

from pathlib import Path

from sqlalchemy import event, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    """Plugin-local declarative base."""


class EmojiDatabase:
    """Async SQLite engine + session factory for the emoji library."""

    def __init__(self, db_path: Path):
        self._db_path = db_path
        # as_posix(): sqlalchemy URL paths must not contain backslashes (Windows)
        self._engine = create_async_engine(
            f"sqlite+aiosqlite:///{db_path.as_posix()}",
            # Single-file library, low write concurrency (one background tagger
            # plus occasional steals): the default pool is fine.
            echo=False,
        )
        self._session_factory = async_sessionmaker(
            self._engine, class_=AsyncSession, expire_on_commit=False
        )
        # WAL + busy_timeout: concurrent read/write sessions (retag, steals,
        # use-count updates) then queue briefly instead of raising
        # "database is locked". journal_mode persists in the DB file, so
        # re-running it per connection is harmless; busy_timeout is
        # per-connection and must be set on every connect.
        @event.listens_for(self._engine.sync_engine, "connect")
        def _sqlite_pragmas(dbapi_connection, _record):
            cursor = dbapi_connection.cursor()
            try:
                cursor.execute("PRAGMA journal_mode=WAL")
                cursor.execute("PRAGMA busy_timeout=5000")
            finally:
                cursor.close()

    @property
    def db_path(self) -> Path:
        return self._db_path

    def session(self) -> AsyncSession:
        """Return an async session; caller is responsible for commit/close.

        Used as ``async with db.session() as session:`` mirroring the
        transaction pattern of the upstream nori plugin.
        """
        return self._session_factory()

    async def create_tables(self) -> None:
        from .models import EmojiImage  # noqa: F401 - register mappings

        async with self._engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
            # create_all never alters existing tables, so columns added after
            # the first release need a hand-rolled upgrade for old DB files.
            def _add_missing_columns(sync_conn) -> None:
                existing = {
                    row[1] for row in sync_conn.execute(text("PRAGMA table_info(emoji_images)"))
                }
                if "tag_fail_count" not in existing:
                    sync_conn.execute(
                        text(
                            "ALTER TABLE emoji_images "
                            "ADD COLUMN tag_fail_count INTEGER NOT NULL DEFAULT 0"
                        )
                    )

            await conn.run_sync(_add_missing_columns)

    async def dispose(self) -> None:
        await self._engine.dispose()
