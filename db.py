"""Plugin-owned SQLite storage (no extra dependencies: sqlalchemy + aiosqlite
are already host requirements).

Kept fully independent from the host database on purpose:
- the emoji library is plugin-local state, wiping ``data/plugin_data/<id>``
  resets it completely;
- the host DatabaseService API is model-bound and not meant for plugin tables.
"""

from __future__ import annotations

from pathlib import Path

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

    async def dispose(self) -> None:
        await self._engine.dispose()
