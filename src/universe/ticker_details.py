"""
src/universe/ticker_details.py

Fetches and stores Polygon Ticker Details v3 for individual tickers.
Handles both single-ticker updates and resumable bulk initialization (~3,000 tickers).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any

import structlog

from src.collection.polygon_client import PolygonAPIError, PolygonClient
from src.collection.queue import WorkQueue

log = structlog.get_logger()

_UNIVERSE_FILE = Path("data/universe/constituents.json")
_UNIVERSE_DIR = Path("data/universe")
_INIT_STATE_FILE = Path("data/universe/.init_state.json")


@dataclass
class TickerDetails:
    ticker: str
    name: str
    description: str
    sector: str
    industry: str
    sic_code: str
    market_cap: float
    shares_outstanding: float
    currency: str = "USD"


def _load_constituents() -> dict[str, dict]:
    if not _UNIVERSE_FILE.exists():
        return {}
    with open(_UNIVERSE_FILE) as f:
        return json.load(f)


def _save_constituents(constituents: dict[str, dict]) -> None:
    _UNIVERSE_DIR.mkdir(parents=True, exist_ok=True)
    with open(_UNIVERSE_FILE, "w") as f:
        json.dump(constituents, f, indent=2)


def _parse_details(ticker: str, response: dict) -> TickerDetails | None:
    """
    Extract fields from Polygon Ticker Details v3 response.
    Returns None if required fields are missing (ticker rejected from universe).
    """
    result = response.get("results", {})
    if not result:
        return None

    description = result.get("description", "").strip()
    name = result.get("name", "").strip()

    # Reject if no company description or name (not a real operating company)
    if not description or not name:
        log.debug("ticker_details.rejected", ticker=ticker, reason="no_company_info")
        return None

    # Branding info has better name sometimes
    branding = result.get("branding", {}) or {}

    # Address → primary exchange for sector mapping (Polygon doesn't give GICS directly)
    # sic_description is the best proxy for industry
    sic_code = result.get("sic_code", "")
    sic_description = result.get("sic_description", "")

    # market_cap and weighted_shares_outstanding from the response
    market_cap = result.get("market_cap") or 0.0
    shares = result.get("weighted_shares_outstanding") or result.get("share_class_shares_outstanding") or 0.0

    return TickerDetails(
        ticker=ticker,
        name=name,
        description=description,
        sector=sic_description,     # Best available; GICS not on free tier
        industry=sic_description,
        sic_code=str(sic_code),
        market_cap=float(market_cap),
        shares_outstanding=float(shares),
    )


def fetch_ticker_details(
    tickers: list[str],
    client: PolygonClient | None = None,
    tier_map: dict[str, str] | None = None,
) -> dict[str, TickerDetails]:
    """
    Fetch Polygon Ticker Details for each ticker.
    Returns {ticker: TickerDetails} for successfully fetched tickers.
    Tickers without company info are silently excluded.
    """
    if client is None:
        client = PolygonClient()
    tier_map = tier_map or {}

    results: dict[str, TickerDetails] = {}
    for ticker in tickers:
        try:
            response = client.get_ticker_details(ticker)
            details = _parse_details(ticker, response)
            if details:
                results[ticker] = details
        except PolygonAPIError as exc:
            if exc.status_code == 404:
                log.debug("ticker_details.not_found", ticker=ticker)
            else:
                log.warning("ticker_details.api_error", ticker=ticker, error=str(exc))

    return results


def fetch_and_admit_new_tickers(
    tickers: list[str],
    client: PolygonClient | None = None,
    tier_map: dict[str, str] | None = None,
) -> list[str]:
    """
    Fetch details for candidate tickers and admit those with valid company info.
    Upserts results into data/universe/constituents.json.
    Returns list of admitted ticker symbols.
    """
    if client is None:
        client = PolygonClient()
    tier_map = tier_map or {}

    details_map = fetch_ticker_details(tickers, client=client, tier_map=tier_map)
    constituents = _load_constituents()
    today = date.today().isoformat()
    admitted: list[str] = []
    new_tickers: list[str] = []

    for ticker, details in details_map.items():
        tier = tier_map.get(ticker, "broad")
        if ticker in constituents:
            # Update existing record
            constituents[ticker].update({
                "name": details.name,
                "description": details.description,
                "sector": details.sector,
                "industry": details.industry,
                "sic_code": details.sic_code,
                "market_cap": details.market_cap,
                "shares_outstanding": details.shares_outstanding,
                "tier": tier,
                "last_details_fetch": today,
                "status": "active",
                "removed_date": None,
                "removal_reason": None,
            })
        else:
            # New admission — also queue a price backfill
            constituents[ticker] = {
                "ticker": ticker,
                "name": details.name,
                "description": details.description,
                "tier": tier,
                "sector": details.sector,
                "industry": details.industry,
                "sic_code": details.sic_code,
                "market_cap": details.market_cap,
                "shares_outstanding": details.shares_outstanding,
                "avg_volume_30d": 0.0,
                "added_date": today,
                "removed_date": None,
                "removal_reason": None,
                "status": "active",
                "last_details_fetch": today,
                "last_financials_fetch": None,
            }
            log.info("ticker_details.admitted", ticker=ticker, tier=tier, name=details.name)
            new_tickers.append(ticker)
        admitted.append(ticker)

    _save_constituents(constituents)

    # Queue price backfills for brand-new admissions
    if new_tickers:
        queue = WorkQueue()
        for ticker in new_tickers:
            queue.enqueue(
                task_type="price_backfill",
                ticker=ticker,
                requested_date=today,
                requested_by="ticker_details",
                priority="normal",
            )

    log.info("ticker_details.upserted", count=len(admitted), new=len(new_tickers))
    return admitted


def bulk_init(
    tickers: list[str],
    client: PolygonClient | None = None,
    tier_map: dict[str, str] | None = None,
) -> None:
    """
    Resumable bulk initialization for ~3,000 tickers.
    State is saved in data/universe/.init_state.json so the run can be interrupted and resumed.
    """
    if client is None:
        client = PolygonClient()
    tier_map = tier_map or {}

    # Load state
    completed: set[str] = set()
    if _INIT_STATE_FILE.exists():
        with open(_INIT_STATE_FILE) as f:
            state = json.load(f)
        completed = set(state.get("completed", []))
        log.info("bulk_init.resuming", already_done=len(completed), remaining=len(tickers) - len(completed))

    remaining = [t for t in tickers if t not in completed]
    total = len(tickers)

    for i, ticker in enumerate(remaining):
        try:
            response = client.get_ticker_details(ticker)
            details = _parse_details(ticker, response)
        except PolygonAPIError as exc:
            if exc.status_code == 404:
                log.debug("bulk_init.not_found", ticker=ticker)
            else:
                log.warning("bulk_init.api_error", ticker=ticker, error=str(exc))
            details = None

        if details:
            tier = tier_map.get(ticker, "broad")
            constituents = _load_constituents()
            today = date.today().isoformat()
            if ticker not in constituents:
                constituents[ticker] = {
                    "ticker": ticker,
                    "name": details.name,
                    "description": details.description,
                    "tier": tier,
                    "sector": details.sector,
                    "industry": details.industry,
                    "sic_code": details.sic_code,
                    "market_cap": details.market_cap,
                    "shares_outstanding": details.shares_outstanding,
                    "avg_volume_30d": 0.0,
                    "added_date": today,
                    "removed_date": None,
                    "removal_reason": None,
                    "status": "active",
                    "last_details_fetch": today,
                    "last_financials_fetch": None,
                }
            else:
                constituents[ticker]["last_details_fetch"] = today
            _save_constituents(constituents)

        completed.add(ticker)

        # Save progress every 25 tickers
        if (i + 1) % 25 == 0 or (i + 1) == len(remaining):
            with open(_INIT_STATE_FILE, "w") as f:
                json.dump({"completed": list(completed)}, f)
            done_total = len(completed)
            log.info("bulk_init.progress", done=done_total, total=total, pct=round(done_total / total * 100, 1))

    log.info("bulk_init.complete", total_admitted=len(_load_constituents()))


if __name__ == "__main__":
    """
    Entry point for local bulk initialization.
    Usage:
        python -m src.universe.ticker_details
    Reads tier_map from Wikipedia scrape, then fetches details for all tickers.
    """
    from dotenv import load_dotenv
    from src.universe.wikipedia import get_index_constituents

    load_dotenv()
    log.info("bulk_init.starting")

    tier_map = get_index_constituents()
    sp_tickers = list(tier_map.keys())

    # Also get broader market from existing constituents or a seed list
    existing = _load_constituents()
    all_tickers = list(set(sp_tickers) | set(existing.keys()))

    log.info("bulk_init.ticker_count", sp_index=len(sp_tickers), total=len(all_tickers))
    bulk_init(all_tickers, tier_map=tier_map)
