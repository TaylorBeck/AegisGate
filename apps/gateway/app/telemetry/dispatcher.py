"""Off-hot-path queue: cache writeback, affinity, XADD, and ledger COPY."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import UTC, datetime

import asyncpg
import numpy as np
import orjson
from numpy.typing import NDArray
from pydantic import BaseModel
from redis.asyncio import Redis

from app.db.postgres import copy_ledger_rows
from app.db.redis import setex_bytes, xadd_approx
from app.engine.cache import CachedCompletion, SemanticCache
from app.providers.breaker import BreakerState
from app.schemas.finops import LatencyBreakdown, XAegisFinOps
from app.schemas.telemetry import (
    BreakerName,
    BreakerStateEvent,
    RequestCompletedEvent,
    SimulationCompletedEvent,
    SimulationStartedEvent,
)
from app.telemetry.otel import emit_gen_ai_span

logger = logging.getLogger("aegis.gateway.dispatcher")

EVENTS_STREAM = "aegis:events"
_QUEUE_MAX = 10_000
_BATCH_SIZE = 500
_FLUSH_S = 0.25
_STREAM_MAXLEN = 2000


@dataclass(slots=True)
class CacheWrite:
    l1_key: str
    completion: CachedCompletion
    tenant_id: str
    sys_hash: str
    vector: NDArray[np.float32]
    write_l2: bool


@dataclass(slots=True)
class AffinityWrite:
    key: str
    payload: bytes
    ttl_seconds: int


@dataclass(slots=True)
class DispatchJob:
    event: bytes
    cache: CacheWrite | None = None
    affinity: AffinityWrite | None = None
    ledger: tuple[object, ...] | None = None
    finops: XAegisFinOps | None = None
    started_at: datetime | None = None


class TelemetryDispatcher:
    """`put_nowait` from the request path. One worker drains the queue."""

    def __init__(
        self,
        redis: Redis,
        pg: asyncpg.Pool,
        cache: SemanticCache,
        *,
        maxsize: int = _QUEUE_MAX,
        otlp_endpoint: str = "",
    ) -> None:
        self._redis = redis
        self._pg = pg
        self._cache = cache
        self._otlp_endpoint = otlp_endpoint
        self._queue: asyncio.Queue[DispatchJob | None] = asyncio.Queue(maxsize=maxsize)
        self._batch: list[tuple[object, ...]] = []
        self._task: asyncio.Task[None] | None = None
        self._stop = asyncio.Event()
        self.dropped = 0

    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._run(), name="aegis-telemetry")

    async def stop(self) -> None:
        self._stop.set()
        task = self._task
        if task is not None:
            await task
            self._task = None

    async def wait_idle(self) -> None:
        """Wait until every offered job has been processed and the ledger flushed."""
        await self._queue.join()
        await self._flush()

    def offer(self, job: DispatchJob) -> None:
        try:
            self._queue.put_nowait(job)
        except asyncio.QueueFull:
            self.dropped += 1
            logger.warning("telemetry queue full; dropping record")

    def offer_breaker(self, provider: str, state: BreakerState) -> None:
        event = BreakerStateEvent(
            provider=provider,
            state=_breaker_name(state),
            ts=_iso_now(),
        )
        self.offer(DispatchJob(event=_dump_event(event)))

    def offer_simulation_started(self, scenario: str, run_id: str, count: int) -> None:
        event = SimulationStartedEvent(scenario=scenario, run_id=run_id, count=count)
        self.offer(DispatchJob(event=_dump_event(event)))

    def offer_simulation_completed(
        self, scenario: str, run_id: str, count: int
    ) -> None:
        event = SimulationCompletedEvent(scenario=scenario, run_id=run_id, count=count)
        self.offer(DispatchJob(event=_dump_event(event)))

    async def _run(self) -> None:
        while True:
            try:
                job = await asyncio.wait_for(self._queue.get(), timeout=_FLUSH_S)
            except TimeoutError:
                await self._flush()
                if self._stop.is_set() and self._queue.empty():
                    break
                continue
            try:
                if job is not None:
                    await self._handle(job)
            except Exception:
                logger.exception("telemetry worker failed")
            finally:
                self._queue.task_done()
            if self._stop.is_set() and self._queue.empty():
                break
        await self._flush()

    async def _handle(self, job: DispatchJob) -> None:
        if job.cache is not None:
            write = job.cache
            await self._cache.put(
                l1_key=write.l1_key,
                completion=write.completion,
                tenant_id=write.tenant_id,
                sys_hash=write.sys_hash,
                vector=write.vector,
                write_l2=write.write_l2,
            )
        if job.affinity is not None:
            await setex_bytes(
                self._redis,
                job.affinity.key,
                job.affinity.payload,
                job.affinity.ttl_seconds,
            )
        await xadd_approx(
            self._redis,
            EVENTS_STREAM,
            {"payload": job.event},
            maxlen=_STREAM_MAXLEN,
        )
        if job.ledger is not None:
            self._batch.append(job.ledger)
            if len(self._batch) >= _BATCH_SIZE:
                await self._flush()
        if job.finops is not None:
            emit_gen_ai_span(
                job.finops,
                endpoint=self._otlp_endpoint,
                started_at=job.started_at,
            )

    async def _flush(self) -> None:
        if not self._batch:
            return
        rows = self._batch
        self._batch = []
        await copy_ledger_rows(self._pg, rows)


def completed_job(
    *,
    request_id: str,
    tenant_id: str,
    conversation_id: str | None,
    finops: XAegisFinOps,
    breakdown: LatencyBreakdown,
    total_ms: float,
    status: str,
    prompt_preview: str,
    run_id: str | None,
    scenario: str | None,
    cache: CacheWrite | None,
    affinity: AffinityWrite | None,
) -> DispatchJob:
    ts = datetime.now(UTC)
    event = RequestCompletedEvent(
        request_id=request_id,
        ts=ts.isoformat(),
        tenant_id=tenant_id,
        run_id=run_id,
        scenario=scenario,
        prompt_preview=prompt_preview[:80],
        finops=finops,
        total_ms=total_ms,
        status=status,
    )
    return DispatchJob(
        event=_dump_event(event),
        cache=cache,
        affinity=affinity,
        ledger=_ledger_tuple(
            request_id=request_id,
            ts=ts,
            tenant_id=tenant_id,
            conversation_id=conversation_id,
            finops=finops,
            breakdown=breakdown,
            total_ms=total_ms,
            status=status,
        ),
        finops=finops,
        started_at=ts,
    )


def _ledger_tuple(
    *,
    request_id: str,
    ts: datetime,
    tenant_id: str,
    conversation_id: str | None,
    finops: XAegisFinOps,
    breakdown: LatencyBreakdown,
    total_ms: float,
    status: str,
) -> tuple[object, ...]:
    failover = finops.failover.from_provider if finops.failover is not None else None
    return (
        request_id,
        ts,
        tenant_id,
        conversation_id,
        finops.cache_status.value,
        finops.similarity_score,
        finops.routed_tier.value,
        finops.routed_model,
        finops.provider,
        failover,
        finops.tokens_in,
        finops.tokens_out,
        finops.cost_actual_usd,
        finops.cost_saved_usd,
        breakdown.guardrail,
        breakdown.l1_lookup,
        breakdown.onnx_embed,
        breakdown.l2_hnsw,
        breakdown.router,
        finops.gateway_overhead_ms,
        finops.ttft_ms,
        total_ms,
        status,
    )


def _dump_event(event: BaseModel) -> bytes:
    return orjson.dumps(event.model_dump(mode="json"))


def _breaker_name(state: BreakerState) -> BreakerName:
    if state is BreakerState.OPEN:
        return "OPEN"
    if state is BreakerState.HALF_OPEN:
        return "HALF_OPEN"
    return "CLOSED"


def _iso_now() -> str:
    return datetime.now(UTC).isoformat()
