"""orjson response class used as FastAPI's default serializer."""

from typing import Any

import orjson
from starlette.responses import Response


class ORJSONResponse(Response):
    """Serialize response bodies with `orjson.dumps` (stdlib `json` is banned)."""

    media_type = "application/json"

    def render(self, content: Any) -> bytes:
        return orjson.dumps(content)
