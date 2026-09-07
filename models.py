"""Emoji library ORM model (plugin-owned table ``emoji_images``).

Ported from nori_plugin_emoji with two deliberate changes:
- ``path`` stores the bare file name inside the emoji directory instead of a
  path relative to some base directory (the emoji dir is fixed per deployment);
- ``created_at`` defaults to naive local time on the Python side (SQLite has
  no ``func.now()`` timezone semantics, and the upstream plugin settled on the
  naive-local convention after an aware/naive mismatch bug).
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import Boolean, DateTime, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from .db import Base


class EmojiImage(Base):
    """Metadata for one emoji image in the library."""

    __tablename__ = "emoji_images"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    hash: Mapped[str] = mapped_column(
        String(64), unique=True, index=True, comment="SHA256 file hash, dedupe key"
    )
    path: Mapped[str] = mapped_column(
        String(512), comment="File name inside the emoji directory"
    )
    description: Mapped[Optional[str]] = mapped_column(
        Text, nullable=True, comment="VLM generated scene description"
    )
    emotions: Mapped[Optional[str]] = mapped_column(
        String(256), nullable=True, comment="VLM emotion labels, comma separated"
    )
    source: Mapped[str] = mapped_column(
        String(16), comment="manual=uploaded/scanned, stolen=from chat"
    )
    use_count: Mapped[int] = mapped_column(Integer, default=0)
    is_banned: Mapped[bool] = mapped_column(Boolean, default=False)
    vlm_processed: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)
    last_used_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    def __repr__(self) -> str:
        return (
            f"<EmojiImage id={self.id} hash={self.hash[:8]}... "
            f"source={self.source} vlm_processed={self.vlm_processed}>"
        )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "hash": self.hash,
            "path": self.path,
            "description": self.description or "",
            "emotions": self.emotions or "",
            "source": self.source,
            "use_count": self.use_count,
            "is_banned": bool(self.is_banned),
            "vlm_processed": bool(self.vlm_processed),
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "last_used_at": self.last_used_at.isoformat() if self.last_used_at else None,
        }
