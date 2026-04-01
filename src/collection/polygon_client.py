"""
src/collection/polygon_client.py

Rate-limited HTTP client for the Polygon.io REST API.
"""

from __future__ import annotations

import os
import time
from datetime import date as _date, timedelta
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

    def _get_available_contracts(
        self,
        ticker: str,
        contract_type: str,
        exp_min: str,
        exp_max: str,
    ) -> list[dict]:
        """
        Return available option contracts in a DTE window from the reference API.
        Results are sorted by expiration_date ascending (Polygon default).
        Returns [] on any error (e.g. plan restriction).
        """
        results: list[dict] = []
        try:
            for page in self.get_paginated(
                "/v3/reference/options/contracts",
                {
                    "underlying_ticker": ticker,
                    "contract_type": contract_type,
                    "expiration_date.gte": exp_min,
                    "expiration_date.lte": exp_max,
                    "expired": "false",
                    "sort": "expiration_date",
                    "order": "asc",
                    "limit": 250,
                },
            ):
                results.extend(page.get("results", []))
        except PolygonAPIError as exc:
            log.warning(
                "polygon.option_contracts_unavailable",
                ticker=ticker,
                contract_type=contract_type,
                error=str(exc),
            )
        return results

    def _fetch_eod_price(self, option_ticker: str, as_of_date: str) -> float | None:
        """
        Fetch the EOD close price for a specific option contract on as_of_date.
        Returns None if no data exists for that date (illiquid, not yet finalized).
        """
        try:
            data = self.get(
                f"/v2/aggs/ticker/{option_ticker}/range/1/day/{as_of_date}/{as_of_date}",
                {"adjusted": "true"},
            )
        except PolygonAPIError:
            return None
        results = data.get("results", [])
        close = results[0].get("c") if results else None
        return float(close) if close and close > 0 else None

    def get_options_chain(
        self,
        ticker: str,
        stock_price: float | None = None,
        as_of_date: str | None = None,
    ) -> list[dict]:
        """
        EOD options chain using free-plan Polygon endpoints.

        Step 1 — discover what actually exists: queries /v3/reference/options/contracts
        for each DTE band and contract type. This returns real available expirations
        and strikes, so we never probe for non-existent contracts.

        Step 2 — pick best match per strategy target:
          - covered_call / diagonal short: nearest expiry (DTE 21-45), closest to 5% OTM
          - leap_otm:                       nearest LEAP expiry (DTE 450-550), closest to 10% OTM
          - diagonal long:                  same LEAP expiry, closest to 25% ITM (75% of price)
          - csp:                            nearest expiry (DTE 18-25), closest to ATM

        Step 3 — fetch EOD close via /v2/aggs/ticker/{optionTicker}/range/1/day/{date}/{date}
        using the specific eval date, so the result is correct regardless of when
        the task is processed (tasks are queued on eval day, processed the next day).

        Greeks (delta) are omitted; select_contract() falls back to strike-pct
        selection, which matches every template spec exactly.
        """
        if not stock_price or not as_of_date:
            log.warning("polygon.options_chain_missing_params", ticker=ticker)
            return []

        eval_date = _date.fromisoformat(as_of_date)

        # --- Step 1: discover available contracts per DTE band ---
        short_calls = self._get_available_contracts(
            ticker, "call",
            (eval_date + timedelta(21)).isoformat(),
            (eval_date + timedelta(45)).isoformat(),
        )
        short_puts = self._get_available_contracts(
            ticker, "put",
            (eval_date + timedelta(18)).isoformat(),
            (eval_date + timedelta(25)).isoformat(),
        )
        leap_calls = self._get_available_contracts(
            ticker, "call",
            (eval_date + timedelta(450)).isoformat(),
            (eval_date + timedelta(550)).isoformat(),
        )

        if not any([short_calls, short_puts, leap_calls]):
            log.warning("polygon.no_option_contracts", ticker=ticker, as_of_date=as_of_date)
            return []

        # --- Step 2: pick best match per strategy target ---
        # Helper: within a set of contracts, use nearest expiry first,
        # then pick the strike closest to the target price.
        def best_match(contracts: list[dict], target_price: float) -> dict | None:
            if not contracts:
                return None
            nearest_exp = contracts[0]["expiration_date"]  # already sorted asc
            same_exp = [c for c in contracts if c["expiration_date"] == nearest_exp]
            return min(same_exp, key=lambda c: abs(c["strike_price"] - target_price))

        targets: list[dict] = []
        seen_tickers: set[str] = set()

        def enqueue(contract: dict | None) -> None:
            if contract and contract["ticker"] not in seen_tickers:
                targets.append(contract)
                seen_tickers.add(contract["ticker"])

        enqueue(best_match(short_calls, stock_price * 1.05))   # covered_call / diagonal short

        # For LEAP legs, use the nearest LEAP expiry for both so diagonal pricing is consistent
        nearest_leap_exp = leap_calls[0]["expiration_date"] if leap_calls else None
        if nearest_leap_exp:
            leap_in_exp = [c for c in leap_calls if c["expiration_date"] == nearest_leap_exp]
            enqueue(min(leap_in_exp, key=lambda c: abs(c["strike_price"] - stock_price * 1.10)))  # leap_otm
            enqueue(min(leap_in_exp, key=lambda c: abs(c["strike_price"] - stock_price * 0.75)))  # diagonal long

        enqueue(best_match(short_puts, stock_price))           # csp

        # --- Step 3: fetch EOD close for each selected contract ---
        chain: list[dict] = []
        for contract in targets:
            price = self._fetch_eod_price(contract["ticker"], as_of_date)
            if price is None:
                log.debug(
                    "polygon.option_no_eod_price",
                    option_ticker=contract["ticker"],
                    as_of_date=as_of_date,
                )
                continue
            chain.append({
                "details": {
                    "ticker": contract["ticker"],
                    "contract_type": contract["contract_type"],
                    "strike_price": contract["strike_price"],
                    "expiration_date": contract["expiration_date"],
                },
                "greeks": None,
                "implied_volatility": None,
                "last_quote": {"midpoint": price},
            })

        log.info(
            "polygon.eod_chain_built",
            ticker=ticker,
            contracts=len(chain),
            selected=len(targets),
            as_of_date=as_of_date,
        )
        return chain

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
