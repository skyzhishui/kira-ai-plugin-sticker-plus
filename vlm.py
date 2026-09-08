"""Emoji VLM interaction: tagging and selection.

Ported from nori_plugin_emoji/vlm.py; the nori TaskRouter/PromptLoader are
replaced by:
- a client resolver callable (returns a host ``LLMModelClient``), resolved
  lazily at call time so config changes apply without rebuilding this object;
- plain ``str.format`` rendering of the plugin-local prompt templates
  (templates keep the upstream ``{{ }}`` escaping convention);
- hand-rolled LLM JSON parsing (fence stripping + json.loads + regex rescue);
  ``json_repair`` is used opportunistically when importable but is not a
  hard dependency.
"""

from __future__ import annotations

import base64
import json
import re
from pathlib import Path
from typing import Callable, Optional

# Same registered name as main.py -> same logger object. Host
# get_logger is required: plain getLogger lines never reach
# data/log.log (the file handler whitelists host-registered names).
from core.logging_manager import get_logger  # noqa: E402

logger = get_logger("sticker-plus", "green")

_PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"

# Minimal magic-number sniffing for the data URL mime type; Pillow already
# validated the payload when the file was stored.
_MIME_BY_EXT = {
    ".png": "image/png",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".bmp": "image/bmp",
}


def _render_prompt(name: str, **vars: str) -> str:
    template = (_PROMPTS_DIR / f"{name}.prompt").read_text(encoding="utf-8")
    return template.format(**vars)


class EmojiVLM:
    """Tag emojis and pick one via the host LLM/VLM client.

    Tagging sends the image to the model (vision required); selection sends
    only the candidate description texts (any chat model is enough). The two
    roles can share one client or be split via separate resolvers.
    """

    def __init__(
        self,
        client_resolver: Callable[[], object],
        selection_client_resolver: Callable[[], object] | None = None,
    ):
        self._resolve_client = client_resolver
        # Selection is text-only; without a dedicated resolver it falls back
        # to the tagging client (single-model setups keep working).
        self._resolve_selection_client = selection_client_resolver or client_resolver

    # ------------------------------------------------------------------
    # Client / request helpers
    # ------------------------------------------------------------------

    def _client(self):
        client = self._resolve_client()
        if client is None:
            raise RuntimeError("no VLM/LLM client available for emoji tagging")
        return client

    def _selection_client(self):
        client = self._resolve_selection_client()
        if client is None:
            raise RuntimeError("no LLM client available for emoji selection")
        return client

    async def _chat(self, content, client) -> str:
        from core.provider import LLMRequest

        request = LLMRequest(messages=[{"role": "user", "content": content}])
        resp = await client.chat(request)
        return resp.text_response or ""

    # ------------------------------------------------------------------
    # Tagging
    # ------------------------------------------------------------------

    async def tag_emoji(self, image_path: Path) -> tuple[str, str]:
        """Describe one emoji image file; returns (description, emotions).

        Raises RuntimeError when the VLM output is empty or not parseable, so
        the caller can retry the record in a later batch.
        """
        data = image_path.read_bytes()
        mime = _MIME_BY_EXT.get(image_path.suffix.lower(), "image/png")
        data_url = f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"
        prompt = _render_prompt("emoji_tagging")
        raw = await self._chat(
            [
                {"type": "image_url", "image_url": {"url": data_url, "detail": "low"}},
                {"type": "text", "text": prompt},
            ],
            client=self._client(),
        )
        parsed = self._parse_json_object(raw)
        description = str(parsed.get("description", "") or "")
        emotions = str(parsed.get("emotions", "") or "")
        if not description or not emotions:
            raise RuntimeError(f"VLM tagging result missing fields: {raw!r}")
        return description, emotions

    # ------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------

    async def select_emoji_by_description(
        self,
        descriptions: list[str],
        emoji_hint: str,
        recent_context: str = "",
    ) -> tuple[int, str]:
        """Pick the best candidate by description text; returns (index, reason).

        Index is 0-based. Raises when nothing parseable comes back or the
        index is out of range - the caller falls back to random selection.
        """
        if not descriptions:
            raise ValueError("candidate description list is empty")

        candidate_lines = "\n".join(f"{i + 1}. {desc}" for i, desc in enumerate(descriptions))
        prompt = _render_prompt(
            "emoji_selection",
            candidate_list=candidate_lines,
            emoji_count=len(descriptions),
            emoji_hint=emoji_hint,
            recent_context=recent_context or "(无)",
        )
        raw = await self._chat(
            [{"type": "text", "text": prompt}], client=self._selection_client()
        )

        try:
            parsed = self._parse_json_object(raw)
        except RuntimeError:
            parsed = self._fallback_extract_emoji_selection(raw)
            if parsed is None:
                raise
            logger.warning("VLM JSON parse failed, regex rescue succeeded: %s", raw[:200])

        emoji_index = parsed.get("emoji_index")
        reason = parsed.get("reason", "")
        if emoji_index is None:
            raise RuntimeError(f"VLM selection result missing index: {raw!r}")
        try:
            idx = int(emoji_index) - 1
        except (TypeError, ValueError) as exc:
            raise RuntimeError(f"VLM returned non-integer index: {emoji_index!r}") from exc
        if idx < 0 or idx >= len(descriptions):
            raise RuntimeError(f"VLM index {idx + 1} out of range [1, {len(descriptions)}]")
        return idx, str(reason)

    # ------------------------------------------------------------------
    # JSON parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_json_object(raw: str) -> dict:
        """Parse an LLM reply into a dict.

        Fence stripping -> slice first ``{`` .. last ``}`` -> json.loads ->
        optional json_repair (only if installed) -> RuntimeError.
        """
        text = (raw or "").strip()
        fence = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
        if fence:
            text = fence.group(1).strip()
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:
            text = text[start : end + 1]
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass
        try:  # optional best-effort repair, not a hard dependency
            import json_repair

            parsed = json_repair.loads(text)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass
        raise RuntimeError(f"LLM output is not a JSON object: {raw!r}")

    @staticmethod
    def _fallback_extract_emoji_selection(raw: str) -> Optional[dict]:
        """Regex rescue for broken selection JSON (ported from upstream)."""
        text = raw.strip()
        fence = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
        if fence:
            text = fence.group(1).strip()
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end == -1 or end < start:
            return None
        text = text[start : end + 1]

        index_match = re.search(r'"emoji_index"\s*:\s*(\d+)', text)
        if not index_match:
            return None

        reason = ""
        reason_match = re.search(r'"reason"\s*:\s*"?(.*?)"?\s*\}\s*$', text, re.DOTALL)
        if reason_match:
            reason = reason_match.group(1).strip()
            if reason.endswith('"'):
                reason = reason[:-1].rstrip()
        if not reason:
            return None
        return {"emoji_index": int(index_match.group(1)), "reason": reason}
