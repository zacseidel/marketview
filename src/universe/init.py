"""
src/universe/init.py

Local initialization script for first-time setup.
Runs the full sequence:
  1. Scrape Wikipedia for S&P 500/400 tier tags
  2. Fetch Ticker Details for index members + broad universe (~3,000 tickers, ~10 hrs)
  3. Backfill 2 years of grouped daily prices (~500 API calls, ~2 hrs)

All steps are resumable — re-run after interruption and it picks up where it left off.

Usage:
    python -m src.universe.init [--step {wikipedia,details,prices}]
"""

from __future__ import annotations

import json
import sys
from datetime import date, timedelta
from pathlib import Path

import structlog

log = structlog.get_logger()

_UNIVERSE_FILE = Path("data.nosync/universe/constituents.json")
_PRICES_DIR = Path("data.nosync/prices")
_BACKFILL_STATE_FILE = Path("data.nosync/prices/.backfill_state.json")


def step_wikipedia(client=None, tier_map: dict | None = None) -> dict[str, str]:
    """Step 1: Scrape S&P 500/400 from Wikipedia. Returns {ticker: tier}."""
    from src.universe.wikipedia import get_index_constituents
    log.info("init.step1_wikipedia")
    tier_map = get_index_constituents()
    log.info("init.wikipedia_done", sp500_and_400=len(tier_map))
    return tier_map


def step_ticker_details(tier_map: dict[str, str], client=None) -> None:
    """Step 2: Fetch Ticker Details for all tickers. Resumable."""
    from src.universe.ticker_details import bulk_init
    log.info("init.step2_ticker_details", count=len(tier_map))
    bulk_init(list(tier_map.keys()), client=client, tier_map=tier_map)


def step_price_backfill(client=None, years: int = 2) -> None:
    """
    Step 3: Backfill 2 years of grouped daily prices.
    Resumable via data.nosync/prices/.backfill_state.json.
    """
    from src.collection.polygon_client import PolygonClient
    from src.universe.ingestion import ingest_daily

    if client is None:
        client = PolygonClient()

    _PRICES_DIR.mkdir(parents=True, exist_ok=True)

    # Build list of dates to backfill (trading days only, weekdays)
    end_date = date.today() - timedelta(days=1)
    start_date = end_date - timedelta(days=years * 365)

    all_dates = []
    d = start_date
    while d <= end_date:
        if d.weekday() < 5:  # Mon–Fri
            all_dates.append(d.isoformat())
        d += timedelta(days=1)

    # Load progress
    completed_dates: set[str] = set()
    if _BACKFILL_STATE_FILE.exists():
        with open(_BACKFILL_STATE_FILE) as f:
            completed_dates = set(json.load(f).get("completed", []))

    # Also skip dates that already have price files
    for date_str in all_dates:
        if (_PRICES_DIR / f"{date_str}.json").exists():
            completed_dates.add(date_str)

    remaining = [d for d in all_dates if d not in completed_dates]
    log.info("init.step3_backfill", total=len(all_dates), remaining=len(remaining))

    for i, date_str in enumerate(remaining):
        try:
            result = ingest_daily(target_date=date_str, client=client)
            log.info("init.backfill_progress", date=date_str, records=result.records_written, i=i + 1, total=len(remaining))
        except Exception as exc:
            log.warning("init.backfill_error", date=date_str, error=str(exc))

        completed_dates.add(date_str)

        if (i + 1) % 10 == 0 or (i + 1) == len(remaining):
            with open(_BACKFILL_STATE_FILE, "w") as f:
                json.dump({"completed": sorted(completed_dates)}, f)

    log.info("init.step3_done", price_files=len(list(_PRICES_DIR.glob("*.json"))))


def run_all(step: str | None = None) -> None:
    from dotenv import load_dotenv
    from src.collection.polygon_client import PolygonClient

    load_dotenv()
    client = PolygonClient()

    # Always scrape Wikipedia for the authoritative tier map
    tier_map = step_wikipedia()
    if step == "wikipedia":
        return

    if step in (None, "details"):
        step_ticker_details(tier_map, client=client)
        if step == "details":
            return

    if step in (None, "prices"):
        step_price_backfill(client=client)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Market Tracker initialization")
    parser.add_argument(
        "--step",
        choices=["wikipedia", "details", "prices"],
        default=None,
        help="Run only a specific step (default: all steps in sequence)",
    )
    args = parser.parse_args()
    run_all(step=args.step)
