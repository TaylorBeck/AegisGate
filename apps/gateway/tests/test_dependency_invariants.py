"""Lockfile must never pull PyTorch / transformers into the gateway."""

from __future__ import annotations

import tomllib
from pathlib import Path

_LOCK = Path(__file__).resolve().parents[1] / "uv.lock"
_BANNED = frozenset(
    {"torch", "transformers", "sentence-transformers", "sentence_transformers"}
)


def test_banned_packages_absent_from_lockfile() -> None:
    data = tomllib.loads(_LOCK.read_text(encoding="utf-8"))
    names = {str(pkg["name"]) for pkg in data["package"]}
    present = names & _BANNED
    assert not present, f"banned packages leaked into uv.lock: {sorted(present)}"
