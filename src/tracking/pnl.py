"""
src/tracking/pnl.py

Mark-to-market open positions using the latest daily close prices.
Updates unrealized_pnl and current_value fields in positions.json.

Entry point:
    python -m src.tracking.pnl [YYYY-MM-DD]
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import structlog

log = structlog.get_logger()

_POSITIONS_FILE = Path("data/positions/positions.json")
_PRICES_DIR = Path("data/prices")


@dataclass
class PnLSummary:
    as_of_date: str
    positions_marked: int
    positions_skipped: int
    total_unrealized_pnl: float
    total_realized_pnl: float


def _load_positions() -> list[dict]:
    if not _POSITIONS_FILE.exists():
        return []
    with open(_POSITIONS_FILE) as f:
        return json.load(f)


def _save_positions(positions: list[dict]) -> None:
    _POSITIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = _POSITIONS_FILE.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(positions, f, indent=2)
    tmp.replace(_POSITIONS_FILE)


def _load_close_prices(as_of_date: str) -> dict[str, float]:
    """Load close prices for as_of_date, falling back to the most recent prior file."""
    price_file = _PRICES_DIR / f"{as_of_date}.json"
    if not price_file.exists():
        # Find most recent price file before as_of_date
        files = sorted(
            [f for f in _PRICES_DIR.glob("*.json") if f.stem <= as_of_date],
            reverse=True,
        )
        if not files:
            return {}
        price_file = files[0]
        log.info("pnl.using_fallback_price_file", file=price_file.name)

    with open(price_file) as f:
        records = json.load(f)
    return {r["ticker"]: r["close"] for r in records if "close" in r}


def update_position_marks(as_of_date: str | None = None) -> PnLSummary:
    """
    Mark all open positions to market using close prices.
    Updates unrealized_pnl and current_value in positions.json.
    """
    if as_of_date is None:
        as_of_date = date.today().isoformat()

    log.info("pnl.starting", as_of_date=as_of_date)

    positions = _load_positions()
    if not positions:
        log.info("pnl.no_positions")
        return PnLSummary(
            as_of_date=as_of_date,
            positions_marked=0,
            positions_skipped=0,
            total_unrealized_pnl=0.0,
            total_realized_pnl=0.0,
        )

    prices = _load_close_prices(as_of_date)
    if not prices:
        log.warning("pnl.no_price_data", as_of_date=as_of_date)
        return PnLSummary(
            as_of_date=as_of_date,
            positions_marked=0,
            positions_skipped=len([p for p in positions if p.get("status") == "open"]),
            total_unrealized_pnl=0.0,
            total_realized_pnl=0.0,
        )

    marked = 0
    skipped = 0
    total_unrealized = 0.0
    total_realized = 0.0

    for pos in positions:
        if pos.get("status") == "open":
            ticker = pos["ticker"]
            close = prices.get(ticker)
            if close is None:
                log.warning("pnl.no_price_for_ticker", ticker=ticker)
                skipped += 1
                continue

            entry = pos.get("entry_price", 0.0)
            unrealized = round(close - entry, 4)
            pos["current_value"] = round(close, 4)
            pos["unrealized_pnl"] = unrealized
            pos["mark_date"] = as_of_date
            total_unrealized += unrealized
            marked += 1

            log.debug("pnl.marked", ticker=ticker, entry=entry, close=close, unrealized=unrealized)

        elif pos.get("status") == "closed":
            realized = pos.get("realized_pnl", 0.0) or 0.0
            total_realized += realized

    _save_positions(positions)

    log.info(
        "pnl.done",
        as_of_date=as_of_date,
        marked=marked,
        skipped=skipped,
        total_unrealized=round(total_unrealized, 4),
        total_realized=round(total_realized, 4),
    )

    return PnLSummary(
        as_of_date=as_of_date,
        positions_marked=marked,
        positions_skipped=skipped,
        total_unrealized_pnl=round(total_unrealized, 4),
        total_realized_pnl=round(total_realized, 4),
    )


if __name__ == "__main__":
    import sys
    from dotenv import load_dotenv
    load_dotenv()

    date_arg = sys.argv[1] if len(sys.argv) > 1 else None
    summary = update_position_marks(date_arg)
    print(
        f"Marked: {summary.positions_marked}, Skipped: {summary.positions_skipped}, "
        f"Unrealized P&L: ${summary.total_unrealized_pnl:+.2f}, "
        f"Realized P&L: ${summary.total_realized_pnl:+.2f}"
    )
