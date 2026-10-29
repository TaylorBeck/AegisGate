"""Virtual-key lookup. Raw keys and Authorization headers are never logged."""

from __future__ import annotations

import hmac

import blake3


class VirtualKeyStore:
    """Maps blake3(bearer) -> tenant_id from `AEGIS_VIRTUAL_KEYS`."""

    def __init__(self, raw: str) -> None:
        mapping: dict[str, str] = {}
        for part in raw.split(","):
            item = part.strip()
            if item == "":
                continue
            tenant, sep, key = item.partition(":")
            if sep == "" or tenant == "" or key == "":
                continue
            digest = blake3.blake3(key.encode("utf-8")).hexdigest()
            mapping[digest] = tenant
        self._by_hash = mapping

    def lookup(self, bearer: str) -> str | None:
        digest = blake3.blake3(bearer.encode("utf-8")).hexdigest()
        for known, tenant in self._by_hash.items():
            if hmac.compare_digest(known, digest):
                return tenant
        return None
