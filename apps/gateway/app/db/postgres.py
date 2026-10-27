"""asyncpg pool and idempotent monthly FinOps partitions.

asyncpg stubs leak `Unknown` through `create_pool` / `acquire` / `execute`.
Unknown-member noise stays in this module so the rest of the gateway can
stay strict.
"""

# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false

from __future__ import annotations

import re
from datetime import UTC, date, datetime

import asyncpg

from app.config import Settings

_PARTITION_NAME = re.compile(r"^finops_ledger_\d{4}_\d{2}$")


def _month_start(value: date) -> date:
    return date(value.year, value.month, 1)


def _add_month(value: date) -> date:
    if value.month == 12:
        return date(value.year + 1, 1, 1)
    return date(value.year, value.month + 1, 1)


async def create_pool(settings: Settings) -> asyncpg.Pool:
    """Open the Postgres pool. Caller is responsible for `ensure_partitions`."""
    return await asyncpg.create_pool(
        dsn=settings.database_url,
        min_size=settings.pg_min_size,
        max_size=settings.pg_max_size,
    )


async def ping(pool: asyncpg.Pool) -> bool:
    """Return True when Postgres answers `SELECT 1`."""
    value = await pool.fetchval("SELECT 1")
    return value == 1


async def close_pool(pool: asyncpg.Pool) -> None:
    """Close the pool and wait for connections to be released."""
    await pool.close()


async def ensure_partitions(pool: asyncpg.Pool) -> None:
    """Create the current and next calendar-month partitions if missing."""
    today = datetime.now(UTC).date()
    starts = [_month_start(today), _add_month(_month_start(today))]
    async with pool.acquire() as conn:
        for start in starts:
            end = _add_month(start)
            table = f"finops_ledger_{start.year:04d}_{start.month:02d}"
            if _PARTITION_NAME.fullmatch(table) is None:
                msg = f"refusing to create partition with unexpected name {table}"
                raise ValueError(msg)
            # Table name is regex-validated; bounds come from datetime.date.
            ddl = (
                f"CREATE TABLE IF NOT EXISTS {table} "
                f"PARTITION OF finops_ledger "
                f"FOR VALUES FROM ('{start.isoformat()}') TO ('{end.isoformat()}')"
            )
            await conn.execute(ddl)
