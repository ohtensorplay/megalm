from __future__ import annotations

from typing import Any

import orjson
from starlette.responses import Response


class MegaLMJSONResponse(Response):
    """FastAPI response backed directly by orjson without deprecated helpers."""

    media_type = "application/json"

    def render(self, content: Any) -> bytes:
        return orjson.dumps(content)


__all__ = ["MegaLMJSONResponse"]
