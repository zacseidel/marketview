"""
src/collection/earnings_refresh.py

Refreshes fundamentals (and earnings events) for tracked tickers whose
estimated next earnings date has passed.

"Tracked" = open positions + latest model new_buy/hold outputs + watchlist.
"Overdue"  = today >= last quarterly filing + 91 days - buffer_days (default 7).
             The 7-day buffer accounts for companies that report a few days
             earlier or later than the prior-quarter cadence.

Calling fetch_and_save() per ticker already triggers earnings.refresh()
internally, so earnings event files are kept in sync automatically.

Usage:
    python -m src.collection.earnings_refresh
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import structlog
import yaml

from src.collection.fundamentals import fetch_and_save
from src.collection.polygon_client import PolygonClient

log = structlog.get_logger()

_FUNDAMENTALS_DIR = Path("data/fundamentals")
_MODELS_DIR       = Path("data/models")
_POSITIONS_FILE   = Path("data/positions/positions.json")
_WATCHLIST_FILE   = Path("config/watchlist.yaml")

_EARNINGS_CADENCE_DAYS = 91   # ~13 weeks between quarterly reports
_BUFFER_DAYS           = 7    # start checking a week before estimate


def _latest_quarterly_filing(ticker: str) -> date | None:
    """Most recent quarterly filing date from data/fundamentals/{ticker}.json."""
    path = _FUNDAMENTALS_DIR / f"{ticker}.json"
    if not path.exists():
        return None
    with open(path) as f:
        records = json.load(f)
    quarterly = [
        r for r in records
        if not r.get("period", "").startswith(("FY", "TTM"))
        and r.get("filing_date")
    ]
    if not quarterly:
        return None
    # records are stored most-recent-first
    return date.fromisoformat(quarterly[0]["filing_date"])


def _get_tracked_tickers() -> set[str]:
    """Open positions + latest model new_buy/hold + watchlist."""
    tickers: set[str] = set()

    # Open positions
    if _POSITIONS_FILE.exists():
        with open(_POSITIONS_FILE) as f:
            positions = json.load(f)
        tickers.update(p["ticker"] for p in positions if p.get("status") == "open")

    # Latest model outputs
    if _MODELS_DIR.exists():
        eval_dirs = sorted(
            d for d in _MODELS_DIR.iterdir()
            if d.is_dir() and d.name[0].isdigit()
        )
        if eval_dirs:
            for fpath in eval_dirs[-1].glob("*.json"):
                if fpath.stem.endswith("_ranks"):
                    continue
                with open(fpath) as f:
                    holdings = json.load(f)
                tickers.update(
                    h["ticker"] for h in holdings
                    if h.get("status") in ("new_buy", "hold")
                )

    # Watchlist
    if _WATCHLIST_FILE.exists():
        with open(_WATCHLIST_FILE) as f:
            cfg = yaml.safe_load(f)
        for entry in cfg.get("watchlist", []):
            if entry.get("ticker"):
                tickers.add(entry["ticker"].upper())

    return tickers


def find_overdue(tickers: set[str], buffer_days: int = _BUFFER_DAYS) -> list[tuple[str, date, int]]:
    """
    Return (ticker, last_filing, days_since) for tickers whose estimated
    next earnings window has passed.
    """
    today = date.today()
    threshold = _EARNINGS_CADENCE_DAYS - buffer_days
    overdue = []
    for ticker in sorted(tickers):
        last_filing = _latest_quarterly_filing(ticker)
        if last_filing is None:
            continue
        days_since = (today - last_filing).days
        if days_since >= threshold:
            overdue.append((ticker, last_filing, days_since))
    return overdue


def run(buffer_days: int = _BUFFER_DAYS) -> None:
    tracked = _get_tracked_tickers()
    log.info("earnings_refresh.tracked", count=len(tracked))

    overdue = find_overdue(tracked, buffer_days=buffer_days)
    if not overdue:
        log.info("earnings_refresh.nothing_overdue")
        print("  No tracked tickers with overdue earnings.")
        return

    print(f"  {len(overdue)} ticker(s) may have new earnings data:")
    for ticker, last_filing, days_since in overdue:
        print(f"    {ticker:<8} last filing {last_filing}  ({days_since}d ago)")

    client = PolygonClient()
    updated, unchanged = 0, 0
    for ticker, _, _ in overdue:
        n = fetch_and_save(ticker, client)
        if n:
            updated += 1
            log.info("earnings_refresh.updated", ticker=ticker, quarters=n)
        else:
            unchanged += 1
            log.debug("earnings_refresh.no_new_data", ticker=ticker)

    print(f"  Updated {updated} ticker(s), {unchanged} unchanged.")


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    run()
