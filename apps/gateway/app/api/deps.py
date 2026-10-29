"""Tenant resolution from hashed virtual keys. Raw keys are never logged."""

from __future__ import annotations

from fastapi import Request

from app.state import get_state


class GatewayError(Exception):
    """OpenAI-shaped HTTP error (`{"error": ...}`)."""

    def __init__(self, status: int, body: dict[str, object]) -> None:
        super().__init__(str(status))
        self.status = status
        self.body = body


def openai_error(
    message: str,
    *,
    error_type: str,
    code: str,
    status: int,
) -> GatewayError:
    return GatewayError(
        status, openai_error_body(message, error_type=error_type, code=code)
    )


def openai_error_body(message: str, *, error_type: str, code: str) -> dict[str, object]:
    return {
        "error": {
            "message": message,
            "type": error_type,
            "code": code,
            "param": None,
        }
    }


async def require_tenant(request: Request) -> str:
    """Resolve the tenant from `Authorization: Bearer`. Never log the header."""
    state = get_state(request)
    header = request.headers.get("authorization")
    if header is None:
        raise openai_error(
            "Missing bearer token",
            error_type="invalid_request_error",
            code="invalid_api_key",
            status=401,
        )
    scheme, _, token = header.partition(" ")
    if scheme.lower() != "bearer" or token.strip() == "":
        raise openai_error(
            "Missing bearer token",
            error_type="invalid_request_error",
            code="invalid_api_key",
            status=401,
        )
    tenant = state.virtual_keys.lookup(token.strip())
    if tenant is None:
        raise openai_error(
            "Invalid API key",
            error_type="invalid_request_error",
            code="invalid_api_key",
            status=401,
        )
    return tenant
