"""
src/selection/watchlist.py

Manual watchlist model — reads config/watchlist.yaml and produces a holdings list.
All tickers in the watchlist qualify regardless of other signals.
Conviction comes directly from the user-specified value in the YAML.
"""

from __future__ import annotations

from pathlib import Path

import structlog
import yaml

from src.selection.base import DataAccessLayer, HoldingRecord, SelectionModel

log = structlog.get_logger()

_DEFAULT_CONFIG = {
    "watchlist_path": "config/watchlist.yaml",
}


class WatchlistModel(SelectionModel):
    def run(self, config: dict, dal: DataAccessLayer) -> list[HoldingRecord]:
        cfg = {**_DEFAULT_CONFIG, **config}
        eval_date = cfg.get("eval_date", "")
        watchlist_path = Path(cfg["watchlist_path"])

        if not watchlist_path.exists():
            log.warning("watchlist.file_not_found", path=str(watchlist_path))
            return []

        with open(watchlist_path) as f:
            data = yaml.safe_load(f)

        entries = data.get("watchlist", []) if data else []
        if not entries:
            log.info("watchlist.empty")
            return []

        universe_tickers = set(dal.get_all_tickers())

        results: list[HoldingRecord] = []
        for entry in entries:
            ticker = str(entry.get("ticker", "")).upper().strip()
            conviction = float(entry.get("conviction", 0.5))
            notes = entry.get("notes", "")

            if not ticker:
                continue

            if ticker not in universe_tickers:
                log.debug("watchlist.ticker_not_in_universe", ticker=ticker)

            rationale = f"Manual watchlist. Notes: {notes}" if notes else "Manual watchlist entry"

            results.append(HoldingRecord(
                model="watchlist",
                eval_date=eval_date,
                ticker=ticker,
                conviction=round(min(max(conviction, 0.0), 1.0), 3),
                rationale=rationale,
                metadata={"notes": notes, "source": "manual"},
            ))

        max_holdings = int(cfg.get("max_holdings", len(results)))
        if len(results) > max_holdings:
            results.sort(key=lambda r: r.conviction, reverse=True)
            results = results[:max_holdings]

        log.info("watchlist.complete", holdings=len(results))
        return results
