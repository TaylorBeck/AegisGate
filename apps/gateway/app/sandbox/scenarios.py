"""Burst, paraphrase, and outage scripts for Mission Control."""

from __future__ import annotations

import asyncio
import logging
import time

from app.engine.complete import complete_json, new_request_id
from app.engine.pipeline import run_hot_path
from app.providers.registry import MOCK_PRIMARY, use_mock_providers
from app.sandbox.runner import RunState
from app.schemas.openai import ChatCompletionRequest, ChatMessage
from app.state import AppState

logger = logging.getLogger("aegis.gateway.simulate")

PARAPHRASE_A = "Translate the English greeting hello into Spanish."
PARAPHRASE_B = "Translate the English greeting hello into Spanish now."

BURST_PROMPTS: tuple[str, ...] = (
    "Summarize this Q4 revenue variance in one paragraph.",
    "Draft a polite follow-up email to a late invoice.",
    "What is the capital of France?",
    "Translate good morning to Japanese.",
    "Classify this ticket: password reset for SSO.",
    "Write a SQL query to count orders by day.",
    "Explain Kubernetes liveness vs readiness probes.",
    "Refactor this function to be idempotent.",
    "How do I rotate IAM keys without downtime?",
    "Generate a pytest for a token bucket limiter.",
    "What does HTTP 429 mean?",
    "Draft an RCA outline for a Redis failover.",
    "Convert 72 Fahrenheit to Celsius.",
    "Name three CAP theorem trade-offs.",
    "Write a regex for AWS access key ids.",
    "Plan a 3-step migration from monolith to services.",
    "Why is my Postgres index not used?",
    "Spell-check: recieve vs receive.",
    "Create a Terraform snippet for an S3 bucket.",
    "How does BLAKE3 differ from SHA-256?",
    "Estimate tokens for a 2k character prompt.",
    "Design a circuit breaker with half-open probes.",
    "What is HNSW in vector search?",
    "Give a one-line git revert for a merge commit.",
    "Compare gpt-4o-mini vs a frontier model for FAQ.",
    "Write a FastAPI healthz handler.",
    "Explain p95 latency vs average latency.",
    "How do I redact JWTs in logs?",
    "Produce a JSON schema for a chat message.",
    "List steps to enable Redis Stack search.",
    "Why would ONNX beat PyTorch on CPU embeddings?",
    "Draft a budget alert for LLM spend.",
    "Translate 'cache miss' into German.",
    "What is session affinity for LLM providers?",
    "Write an asyncio timeout around an httpx stream.",
    "How do I partition a ledger by month?",
    "Give a 2-sentence summary of RAG vs cache.",
    "Show a curl to POST /v1/chat/completions.",
    "What's a safe similarity threshold for paraphrases?",
    "Generate a changelog entry for a breaking API.",
    "Explain token bucket refill math.",
    "Write a type-safe Pydantic v2 model for FinOps.",
    "How do I keep SSE from being buffered by nginx?",
    "Compare L1 exact cache vs L2 semantic cache.",
    "Draft a runbook for an OPEN circuit breaker.",
    "What headers does EventSource send on reconnect?",
    "Write a Python dataclass with slots=True.",
    "How do I compute USD cost from tokens and $/M?",
    "Give an example of a SIMPLE vs AGENTIC_CODE prompt.",
    "Summarize AegisGate in one sentence for a CFO.",
)


async def run_scenario(state: AppState, run: RunState) -> None:
    try:
        if run.scenario == "burst":
            run.count = await _burst(state, run)
        elif run.scenario == "paraphrase":
            run.count = await _paraphrase(state, run)
        elif run.scenario == "outage":
            run.count = await _outage(state, run)
        else:
            raise ValueError(f"unknown scenario {run.scenario}")
        state.dispatcher.offer_simulation_completed(run.scenario, run.run_id, run.count)
        state.simulations.finish(run)
    except Exception as exc:
        logger.exception("simulation %s failed", run.run_id)
        state.simulations.finish(run, error=str(exc))
        state.dispatcher.offer_simulation_completed(run.scenario, run.run_id, run.count)


async def _burst(state: AppState, run: RunState) -> int:
    tenant = f"sandbox_{run.run_id}"
    sem = asyncio.Semaphore(8)

    async def one(prompt: str) -> None:
        async with sem:
            await _complete(state, tenant, prompt, run)

    await asyncio.gather(*[one(prompt) for prompt in BURST_PROMPTS])
    return len(BURST_PROMPTS)


async def _paraphrase(state: AppState, run: RunState) -> int:
    tenant = f"sandbox_{run.run_id}"
    await _complete(state, tenant, PARAPHRASE_A, run)
    await state.dispatcher.wait_idle()
    await _complete(state, tenant, PARAPHRASE_B, run)
    return 2


async def _outage(state: AppState, run: RunState) -> int:
    tenant = f"sandbox_{run.run_id}"
    state.chaos.inject(MOCK_PRIMARY, 503, count=6)
    try:
        for prompt in BURST_PROMPTS[:3]:
            await _complete(state, tenant, prompt, run)
        return 3
    finally:
        state.chaos.clear(MOCK_PRIMARY)


async def _complete(
    state: AppState,
    tenant: str,
    prompt: str,
    run: RunState,
) -> None:
    body = ChatCompletionRequest(
        model="auto",
        temperature=0.2,
        messages=[ChatMessage(role="user", content=prompt)],
        stream=False,
    )
    t0 = time.perf_counter()
    result = await run_hot_path(
        body,
        tenant,
        state,
        open_providers=state.breakers.open_names(),
    )
    await complete_json(
        state=state,
        request=body,
        tenant=tenant,
        result=result,
        t0=t0,
        request_id=new_request_id(),
        created=int(time.time()),
        use_mock=use_mock_providers(tenant, state.settings, None),
        run_id=run.run_id,
        scenario=run.scenario,
    )
