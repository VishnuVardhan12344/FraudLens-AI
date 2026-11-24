
import math
import os
import time

from collections import deque
from threading import Lock

from fastapi import HTTPException, Request

from fraudlens.logger import get_logger


logger = get_logger("fraudlens.rate_limiter")


class RateLimiter:

    def __init__(
        self,
        max_requests=60,
        window_seconds=60,
        clock=time.monotonic,
    ):

        if max_requests <= 0 or window_seconds <= 0:
            raise ValueError("Invalid rate limit configuration")

        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._clock = clock

        self._requests = deque()
        self._lock = Lock()

    def allow_request(self):

        with self._lock:

            now = self._clock()

            while (
                self._requests
                and now - self._requests[0]
                >= self.window_seconds
            ):
                self._requests.popleft()

            if len(self._requests) >= self.max_requests:

                retry_after = math.ceil(
                    self.window_seconds
                    - (now - self._requests[0])
                )

                return False, max(1, retry_after)

            self._requests.append(now)

            return True, 0


rate_limiter = RateLimiter(
    max_requests=int(
        os.getenv("FRAUDLENS_RATE_LIMIT", "60")
    ),
    window_seconds=60,
)


def enforce_rate_limit(request: Request):

    allowed, retry_after = rate_limiter.allow_request()

    if not allowed:

        logger.warning(
            "rate_limit_exceeded",
            extra={
                "context": {
                    "request_id": request.state.request_id,
                    "retry_after_seconds": retry_after,
                }
            },
        )

        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded. Try again later.",
            headers={
                "Retry-After": str(retry_after)
            },
        )
