"""
src/collection/polygon_client.py

Rate-limited HTTP client for the Polygon.io REST API.
"""

from __future__ import annotations

import os
import time
from datetime import date as _date
from typing import Iterator

import requests
import structlog

from src.collection.rate_limiter import RateLimiter

log = structlog.get_logger()

BASE_URL = "https://api.polygon.io"
_RETRY_STATUSES = {429, 500, 502, 503, 504}
_MAX_RETRIES = 5



class PolygonAPIError(Exception):
    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


class PolygonClient:
    def __init__(self, api_key: str | None = None, rate_limiter: RateLimiter | None = None):
        self._api_key = api_key or os.environ["POLYGON_API_KEY"]
        self._limiter = rate_limiter or RateLimiter()
        self._session = requests.Session()
        # Default apiKey on all requests
        self._session.params = {"apiKey": self._api_key}  # type: ignore[assignment]

    # ------------------------------------------------------------------
    # Core request methods
    # ------------------------------------------------------------------

    def get(self, endpoint: str, params: dict | None = None) -> dict:
        """Single-response GET. Raises PolygonAPIError on unrecoverable failure."""
        url = f"{BASE_URL}{endpoint}"
        params = params or {}

        for attempt in range(_MAX_RETRIES + 1):
            self._limiter.acquire()
            log.debug("polygon.request", url=url, params=params, attempt=attempt)
            try:
                resp = self._session.get(url, params=params, timeout=30)
            except requests.RequestException as exc:
                log.warning("polygon.connection_error", error=str(exc), attempt=attempt)
                if attempt == _MAX_RETRIES:
                    raise PolygonAPIError(f"Connection error after {_MAX_RETRIES} retries: {exc}")
                time.sleep(2 ** attempt)
                continue

            log.debug("polygon.response", status=resp.status_code, url=url)

            if resp.status_code == 200:
                return resp.json()
            elif resp.status_code in _RETRY_STATUSES:
                wait = 2 ** attempt
                log.warning("polygon.retry", status=resp.status_code, attempt=attempt, wait_seconds=wait)
                if attempt == _MAX_RETRIES:
                    raise PolygonAPIError(
                        f"Max retries exceeded ({resp.status_code}) for {url}",
                        resp.status_code,
                    )
                time.sleep(wait)
            else:
                raise PolygonAPIError(
                    f"HTTP {resp.status_code}: {resp.text[:200]}",
                    resp.status_code,
                )

        raise PolygonAPIError(f"Max retries exceeded for {url}")

    def get_paginated(self, endpoint: str, params: dict | None = None) -> Iterator[dict]:
        """Follow Polygon cursor pagination, yielding one page dict at a time."""
        params = params or {}
        page = self.get(endpoint, params)
        yield page

        while page.get("next_url"):
            next_url: str = page["next_url"]
            # next_url is absolute; add apiKey if not already embedded
            self._limiter.acquire()
            log.debug("polygon.paginated_request", url=next_url)
            for attempt in range(_MAX_RETRIES + 1):
                if attempt > 0:
                    self._limiter.acquire()
                try:
                    resp = self._session.get(next_url, params={"apiKey": self._api_key}, timeout=30)
                except requests.RequestException as exc:
                    if attempt == _MAX_RETRIES:
                        raise PolygonAPIError(f"Connection error on paginated fetch: {exc}")
                    time.sleep(2 ** attempt)
                    continue

                if resp.status_code == 200:
                    page = resp.json()
                    yield page
                    break
                elif resp.status_code in _RETRY_STATUSES:
                    wait = 2 ** attempt
                    log.warning("polygon.paginated_retry", status=resp.status_code, wait_seconds=wait)
                    if attempt == _MAX_RETRIES:
                        raise PolygonAPIError(f"Max retries on paginated fetch ({resp.status_code})", resp.status_code)
                    time.sleep(wait)
                else:
                    raise PolygonAPIError(f"HTTP {resp.status_code} on paginated fetch", resp.status_code)

    # ------------------------------------------------------------------
    # Domain-specific methods
    # ------------------------------------------------------------------

    def get_grouped_daily(self, date: str) -> dict:
        """All tickers' OHLCV for a single trading date. date: YYYY-MM-DD."""
        return self.get(
            f"/v2/aggs/grouped/locale/us/market/stocks/{date}",
            {"adjusted": "true", "include_otc": "false"},
        )

    def get_ticker_details(self, ticker: str) -> dict:
        """Ticker Details v3 for a single ticker."""
        return self.get(f"/v3/reference/tickers/{ticker}")

    def get_stock_financials(self, ticker: str, limit: int = 20) -> list[dict]:
        """Quarterly financials for a ticker (revenue, net income, shares)."""
        results: list[dict] = []
        for page in self.get_paginated(
            "/vX/reference/financials",
            {"ticker": ticker, "limit": limit, "order": "desc"},
        ):
            results.extend(page.get("results", []))
        return results

    def get_agg_bars(
        self,
        ticker: str,
        from_: str,
        to: str,
        adjusted: bool = True,
    ) -> list[dict]:
        """Daily OHLCV bars for a single ticker. from_/to: YYYY-MM-DD."""
        data = self.get(
            f"/v2/aggs/ticker/{ticker}/range/1/day/{from_}/{to}",
            {"adjusted": str(adjusted).lower(), "sort": "asc", "limit": 50000},
        )
        return data.get("results", [])

    def get_splits(self, ticker: str) -> list[dict]:
        """Corporate split events for a ticker."""
        results: list[dict] = []
        for page in self.get_paginated("/v3/reference/splits", {"ticker": ticker}):
            results.extend(page.get("results", []))
        return results

    def get_reference_tickers(self, market: str = "stocks", active: bool = True) -> list[dict]:
        """All active tickers with basic metadata (paginated)."""
        results: list[dict] = []
        for page in self.get_paginated(
            "/v3/reference/tickers",
            {"market": market, "active": str(active).lower(), "limit": 1000},
        ):
            results.extend(page.get("results", []))
        return results
