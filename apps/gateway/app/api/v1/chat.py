"""OpenAI-compatible `POST /v1/chat/completions`."""

from __future__ import annotations

import time
from typing import Annotated

from fastapi import APIRouter, Depends, Request
from starlette.responses import StreamingResponse

from app.api.deps import openai_error_body, require_tenant
from app.engine.admission import AdmissionDeniedError
from app.engine.complete import complete_json, iter_sse, new_request_id
from app.engine.pipeline import run_hot_path
from app.providers.registry import use_mock_providers
from app.responses import ORJSONResponse
from app.schemas.openai import ChatCompletionRequest
from app.state import get_state

router = APIRouter()

_SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "X-Accel-Buffering": "no",
    "Connection": "keep-alive",
}


@router.post("/v1/chat/completions", response_model=None)
async def chat_completions(
    request: Request,
    body: ChatCompletionRequest,
    tenant: Annotated[str, Depends(require_tenant)],
) -> ORJSONResponse | StreamingResponse:
    state = get_state(request)
    t0 = time.perf_counter()
    created = int(time.time())
    request_id = new_request_id()
    upstream_key = request.headers.get("x-aegis-upstream-key")
    use_mock = use_mock_providers(tenant, state.settings, upstream_key)
    try:
        result = await run_hot_path(
            body,
            tenant,
            state,
            open_providers=state.breakers.open_names(),
        )
    except AdmissionDeniedError:
        return ORJSONResponse(
            openai_error_body(
                "Rate limit exceeded",
                error_type="rate_limit_error",
                code="rate_limit_exceeded",
            ),
            status_code=429,
        )
    if body.stream:
        return StreamingResponse(
            iter_sse(
                state=state,
                request=body,
                tenant=tenant,
                result=result,
                t0=t0,
                request_id=request_id,
                created=created,
                use_mock=use_mock,
            ),
            media_type="text/event-stream",
            headers=_SSE_HEADERS,
        )
    status, payload = await complete_json(
        state=state,
        request=body,
        tenant=tenant,
        result=result,
        t0=t0,
        request_id=request_id,
        created=created,
        use_mock=use_mock,
    )
    return ORJSONResponse(payload, status_code=status)
