"""
src/universe/reconcile.py

Weekly universe reconciliation.
  1. Scrape Wikipedia for current S&P 500/400 composition
  2. Diff against stored universe
  3. Queue ticker_details tasks for new index additions
  4. Downgrade removed index members to 'broad' tier (not removed — they stay in universe)

Called by universe-refresh.yml (Sunday 12 PM ET).

Usage:
    python -m src.universe.reconcile
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import structlog

from src.universe.wikipedia import get_index_constituents, diff_against_universe
from src.collection.queue import WorkQueue

log = structlog.get_logger()

_UNIVERSE_FILE = Path("data/universe/constituents.json")
_UNIVERSE_DIR = Path("data/universe")


def _load_constituents() -> dict[str, dict]:
    if not _UNIVERSE_FILE.exists():
        return {}
    with open(_UNIVERSE_FILE) as f:
        return json.load(f)


def _save_constituents(constituents: dict[str, dict]) -> None:
    _UNIVERSE_DIR.mkdir(parents=True, exist_ok=True)
    with open(_UNIVERSE_FILE, "w") as f:
        json.dump(constituents, f, indent=2)


def reconcile() -> dict:
    """
    Run the weekly reconciliation cycle.
    Returns a summary dict with added/demoted counts.
    """
    log.info("reconcile.starting")
    constituents = _load_constituents()
    scraped = get_index_constituents()  # {ticker: 'sp500'|'sp400'}

    added, removed_from_index = diff_against_universe(scraped, constituents)

    today = date.today().isoformat()
    queue = WorkQueue()

    # Queue new index additions for ticker_details fetch
    for ticker in added:
        tier = scraped[ticker]
        log.info("reconcile.new_index_member", ticker=ticker, tier=tier)
        if ticker in constituents:
            # Already in universe (broad) — upgrade tier
            constituents[ticker]["tier"] = tier
        else:
            # Brand new — queue details fetch
            queue.enqueue(
                task_type="ticker_details",
                ticker=ticker,
                requested_date=today,
                requested_by="universe_reconcile",
                priority="high",
            )

    # Downgrade tickers removed from index → 'broad' (keep in universe)
    for ticker in removed_from_index:
        if ticker in constituents and constituents[ticker].get("status") == "active":
            old_tier = constituents[ticker].get("tier", "broad")
            constituents[ticker]["tier"] = "broad"
            log.info("reconcile.tier_downgraded", ticker=ticker, old_tier=old_tier, new_tier="broad")

    # Update tier for all tickers still in index (in case tier changed sp500↔sp400)
    for ticker, tier in scraped.items():
        if ticker in constituents:
            constituents[ticker]["tier"] = tier

    _save_constituents(constituents)

    summary = {
        "scraped_index_members": len(scraped),
        "new_additions": len(added),
        "index_removals_downgraded": len(removed_from_index),
        "universe_total": len(constituents),
    }
    log.info("reconcile.complete", **summary)
    return summary


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    summary = reconcile()
    print(summary)
