from __future__ import annotations

import math
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True, slots=True)
class RateLimitDecision:
    allowed: bool
    retry_after_seconds: int = 0


class IpRateLimiter:
    def __init__(
        self,
        max_requests: int,
        window_seconds: int,
        time_func: Callable[[], float] | None = None,
    ) -> None:
        self._max_requests = max_requests
        self._window_seconds = window_seconds
        self._time_func = time_func or time.time
        self._requests: defaultdict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def check(self, ip_address: str) -> RateLimitDecision:
        now = self._time_func()

        with self._lock:
            bucket = self._requests[ip_address]
            self._prune(bucket, now)

            if len(bucket) >= self._max_requests:
                retry_after = max(
                    1,
                    math.ceil(self._window_seconds - (now - bucket[0])),
                )
                return RateLimitDecision(
                    allowed=False,
                    retry_after_seconds=retry_after,
                )

            bucket.append(now)
            return RateLimitDecision(allowed=True)

    def _prune(self, bucket: deque[float], now: float) -> None:
        cutoff = now - self._window_seconds
        while bucket and bucket[0] <= cutoff:
            bucket.popleft()
