"""Test bootstrap for the sticker-plus plugin suite.

Layout assumptions (mirrors the noriflow/nori convention):
- this plugin lives as ``data/plugins/kira-ai-plugin-sticker-plus`` inside a
  KiraAI checkout -> the host root (containing ``core/`` and ``webui/``) is
  three levels up from this directory;
- fallback: environment variable ``KIRAAI_ROOT`` pointing at a checkout.

The host root is needed for ``core.chat.message_elements`` (Sticker element)
and ``core.provider`` (LLMRequest) imports; the plugin package itself is
importable via its ``__init__.py``. When the host is missing the suite skips
itself.
"""

from pathlib import Path
import os
import sys

_TESTS_DIR = Path(__file__).resolve().parent
_PLUGIN_DIR = _TESTS_DIR.parent


def _locate_host_root() -> Path | None:
    env_root = os.environ.get("KIRAAI_ROOT")
    if env_root and (Path(env_root) / "core" / "plugin").is_dir():
        return Path(env_root)
    for base in _TESTS_DIR.parents:
        if (base / "core" / "plugin").is_dir() and (base / "webui").is_dir():
            return base
    return None


_HOST_ROOT = _locate_host_root()

if _HOST_ROOT is None:
    print(f"[conftest] KiraAI host checkout not found, skipping {_TESTS_DIR} collection")
    collect_ignore_glob = ["test_*.py"]
else:
    for entry in (str(_PLUGIN_DIR.parent), str(_HOST_ROOT)):
        if entry not in sys.path:
            sys.path.insert(0, entry)
