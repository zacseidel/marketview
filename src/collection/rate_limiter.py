"""
src/collection/rate_limiter.py

Token bucket rate limiter for Polygon.io free tier (5 calls/minute).
"""

from __future__ import annotations

import threading
import time

import structlog

log = structlog.get_logger()


class RateLimiter:
    """Thread-safe token bucket rate limiter."""

    def __init__(self, calls_per_minute: int = 5):
        self._capacity = float(calls_per_minute)
        self._refill_rate = calls_per_minute / 60.0  # tokens per second
        self._tokens = float(calls_per_minute)
        self._last_refill = time.monotonic()
        self._lock = threading.Lock()

    def acquire(self) -> None:
        """Block until a token is available, then consume one."""
        with self._lock:
            self._refill()
            while self._tokens < 1.0:
                wait = (1.0 - self._tokens) / self._refill_rate
                log.debug("rate_limiter.waiting", wait_seconds=round(wait, 2))
                self._lock.release()
                try:
                    time.sleep(wait)
                finally:
                    self._lock.acquire()
                self._refill()
            self._tokens -= 1.0

    def _refill(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_refill
        self._tokens = min(self._capacity, self._tokens + elapsed * self._refill_rate)
        self._last_refill = now
