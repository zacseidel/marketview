"""
src/tracking/portfolio.py

Tracks the user's actual curated portfolio performance.

Computes portfolio-level P&L from positions.json and appends a daily
snapshot to data/positions/portfolio_history.json.

Entry point:
    compute_portfolio_performance(as_of_date: str | None = None) -> PortfolioPerformance
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path

import structlog

from src.tracking.positions import get_closed_positions, get_open_positions

log = structlog.get_logger()

_PORTFOLIO_HISTORY_FILE = Path("data/positions/portfolio_history.json")


@dataclass
class PortfolioPerformance:
    as_of_date: str
    open_count: int
    closed_count: int
    total_unrealized_pnl: float
    total_realized_pnl: float
    total_pnl: float
    strategy_breakdown: dict    # {strategy: count of open positions}
    open_positions: list[dict]
    closed_positions: list[dict]


def compute_portfolio_performance(as_of_date: str | None = None) -> PortfolioPerformance:
    if as_of_date is None:
        as_of_date = date.today().isoformat()

    open_pos = get_open_positions()
    closed_pos = get_closed_positions()

    total_unrealized = sum(p.get("unrealized_pnl") or 0.0 for p in open_pos)
    total_realized = sum(p.get("realized_pnl") or 0.0 for p in closed_pos)

    strategy_counts: dict[str, int] = {}
    for p in open_pos:
        s = p.get("strategy", "stock")
        strategy_counts[s] = strategy_counts.get(s, 0) + 1

    perf = PortfolioPerformance(
        as_of_date=as_of_date,
        open_count=len(open_pos),
        closed_count=len(closed_pos),
        total_unrealized_pnl=round(total_unrealized, 4),
        total_realized_pnl=round(total_realized, 4),
        total_pnl=round(total_unrealized + total_realized, 4),
        strategy_breakdown=strategy_counts,
        open_positions=open_pos,
        closed_positions=closed_pos,
    )

    _append_history(perf)

    log.info(
        "portfolio.computed",
        as_of_date=as_of_date,
        open=len(open_pos),
        closed=len(closed_pos),
        total_pnl=perf.total_pnl,
    )
    return perf


def _append_history(perf: PortfolioPerformance) -> None:
    """Append/replace today's portfolio snapshot in portfolio_history.json."""
    _PORTFOLIO_HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)

    history: list[dict] = []
    if _PORTFOLIO_HISTORY_FILE.exists():
        with open(_PORTFOLIO_HISTORY_FILE) as f:
            history = json.load(f)

    # Replace snapshot for this date if one already exists
    snapshot = {
        "as_of_date": perf.as_of_date,
        "open_count": perf.open_count,
        "closed_count": perf.closed_count,
        "total_unrealized_pnl": perf.total_unrealized_pnl,
        "total_realized_pnl": perf.total_realized_pnl,
        "total_pnl": perf.total_pnl,
        "strategy_breakdown": perf.strategy_breakdown,
    }
    history = [h for h in history if h.get("as_of_date") != perf.as_of_date]
    history.append(snapshot)
    history.sort(key=lambda h: h["as_of_date"])

    tmp = _PORTFOLIO_HISTORY_FILE.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(history, f, indent=2)
    tmp.replace(_PORTFOLIO_HISTORY_FILE)


if __name__ == "__main__":
    import sys
    from dotenv import load_dotenv
    load_dotenv()

    date_arg = sys.argv[1] if len(sys.argv) > 1 else None
    perf = compute_portfolio_performance(date_arg)
    print(
        f"Open: {perf.open_count}  Closed: {perf.closed_count}  "
        f"Unrealized: ${perf.total_unrealized_pnl:+.2f}  "
        f"Realized: ${perf.total_realized_pnl:+.2f}  "
        f"Total: ${perf.total_pnl:+.2f}"
    )
