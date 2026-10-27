"""Copy JSON and Redis replies into fully typed containers.

`isinstance` narrows `list` and `dict` to unknown type parameters under strict
checking. The casts stay in this module so callers see `object` elements only.
"""

# pyright: reportUnknownArgumentType=false, reportUnknownVariableType=false

from __future__ import annotations

from typing import cast


def object_dict(value: object) -> dict[str, object] | None:
    """Return a string-keyed dict, or None when `value` is not one."""
    if not isinstance(value, dict):
        return None
    typed: dict[str, object] = {}
    for key, item in cast(dict[object, object], value).items():
        if not isinstance(key, str):
            return None
        typed[key] = item
    return typed


def object_list(value: object) -> list[object] | None:
    """Return a list of objects, or None when `value` is not a list."""
    if not isinstance(value, list):
        return None
    return list(cast(list[object], value))


def object_pairs(value: object) -> list[tuple[object, object]] | None:
    """Return key/value pairs from a dict or a flat Redis field list."""
    if isinstance(value, dict):
        pairs: list[tuple[object, object]] = []
        for key, item in cast(dict[object, object], value).items():
            pairs.append((key, item))
        return pairs
    values = object_list(value)
    if values is None or len(values) % 2 != 0:
        return None
    return [(values[index], values[index + 1]) for index in range(0, len(values), 2)]
