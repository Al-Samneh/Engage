from __future__ import annotations

import time
import uuid
from collections import defaultdict, deque
from typing import Callable, Deque, Dict

from fastapi import Request
from fastapi.responses import JSONResponse

from .config import get_settings


class RateLimiter:
    """Sliding-window counter keyed by API key header."""

    def __init__(self, limit_per_minute: int, window_seconds: int) -> None:
        self.limit_per_minute = limit_per_minute
        self.window_seconds = window_seconds
        self._bucket: Dict[str, Deque[float]] = defaultdict(deque)

    def check(self, key: str) -> bool:
        """
        True → request allowed. False → caller exceeded quota and should be throttled.
        """
        now = time.time()
        window_start = now - self.window_seconds
        window = self._bucket[key]

        while window and window[0] < window_start:
            window.popleft()

        if len(window) >= self.limit_per_minute:
            return False

        window.append(now)
        return True


def make_rate_limit_middleware():
    """
    Factory so we can inject Settings into the middleware stack during app creation.
    """
    settings = get_settings()
    limiter = RateLimiter(
        limit_per_minute=settings.default_rate_limit_per_minute,
        window_seconds=settings.rate_limit_window_seconds,
    )

    async def middleware(request: Request, call_next: Callable):
        trace_id = uuid.uuid4()
        start = time.perf_counter()
        api_key = request.headers.get("x-api-key", "public")
        if not limiter.check(api_key):
            return JSONResponse(
                status_code=429,
                content={
                    "code": "RATE_LIMIT_EXCEEDED",
                    "message": "Too many requests. Try again soon.",
                    "trace_id": str(trace_id),
                    "latency_ms": int((time.perf_counter() - start) * 1000),
                },
            )

        response = await call_next(request)
        response.headers["x-trace-id"] = str(trace_id)
        return response

    return middleware

