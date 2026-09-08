"""Shared test doubles for the sticker-plus plugin suite."""

from __future__ import annotations

import base64
import io
from types import SimpleNamespace

from PIL import Image

import importlib

# The plugin folder name contains hyphens, so `from kira-ai-plugin-sticker-plus.x`
# is a syntax error - import through the machinery and re-export here.
_pkg = importlib.import_module("kira-ai-plugin-sticker-plus")
db_mod = importlib.import_module("kira-ai-plugin-sticker-plus.db")
repo_mod = importlib.import_module("kira-ai-plugin-sticker-plus.repo")
manager_mod = importlib.import_module("kira-ai-plugin-sticker-plus.manager")
vlm_mod = importlib.import_module("kira-ai-plugin-sticker-plus.vlm")
stealer_mod = importlib.import_module("kira-ai-plugin-sticker-plus.stealer")

EmojiDatabase = db_mod.EmojiDatabase
EmojiRepository = repo_mod.EmojiRepository
EmojiManager = manager_mod.EmojiManager
EmojiVLM = vlm_mod.EmojiVLM
EmojiStealer = stealer_mod.EmojiStealer


def make_png_bytes(color=(255, 0, 0), size=(8, 8), fmt="PNG") -> bytes:
    """Tiny in-memory image used as emoji payload."""
    img = Image.new("RGB", size, color)
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


def b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


class FakeClient:
    """Stands in for a host LLMModelClient."""

    def __init__(self, replies: list[str]):
        self.replies = list(replies)
        self.requests = []

    async def chat(self, request):
        self.requests.append(request)
        if not self.replies:
            raise AssertionError("FakeClient out of canned replies")
        return SimpleNamespace(text_response=self.replies.pop(0))


class FakeTagVLM(EmojiVLM):
    """Deterministic tagger: description/emotions derived from the file name."""

    def __init__(self):
        super().__init__(client_resolver=lambda: None)
        self.tag_calls: list = []

    async def tag_emoji(self, image_path):
        self.tag_calls.append(image_path)
        name = image_path.name
        return f"desc of {name[:8]}", "开心,调皮"

    async def select_emoji_by_description(self, descriptions, emoji_hint, recent_context=""):
        # Always pick the first candidate, deterministically.
        return 0, "first"


class FailingTagVLM(EmojiVLM):
    """Tagger that always raises - models permanently broken tagging."""

    def __init__(self):
        super().__init__(client_resolver=lambda: None)

    async def tag_emoji(self, image_path):
        raise RuntimeError("tagging always fails")

    async def select_emoji_by_description(self, descriptions, emoji_hint, recent_context=""):
        return 0, "first"


class FakeStealManager:
    """Records add_emoji_from_bytes calls for stealer tests."""

    def __init__(self):
        self.calls: list[bytes] = []

    async def add_emoji_from_bytes(self, data: bytes, source: str = "stolen") -> bool:
        self.calls.append(data)
        return True
