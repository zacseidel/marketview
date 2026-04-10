"""
src/strategy/runner.py

Entry point for strategy evaluation (called by evaluate-strategies.yml).

Two responsibilities:
  1. Check all open positions for expired options legs; queue chain fetches
     for any strategies that need new legs opened (covered_call roll, CSP roll, etc.)
  2. Aggregate log returns across all closed strategy observations and save
     a summary to data.nosync/strategy_observations/returns.json.

Usage:
    python -m src.strategy.runner
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import structlog

from src.strategy.returns import (
    aggregate_returns, save_returns, print_summary,
    aggregate_theoretical_returns, save_theoretical_returns,
)

log = structlog.get_logger()

_PRICES_DIR = Path("data.nosync/prices")
_POSITIONS_FILE = Path("data.nosync/positions/positions.json")


def _load_latest_prices() -> dict[str, float]:
    """Load the most recent daily close prices available."""
    if not _PRICES_DIR.exists():
        return {}
    files = sorted(_PRICES_DIR.glob("*.json"), reverse=True)
    if not files:
        return {}
    with open(files[0]) as f:
        records = json.load(f)
    return {
        r["ticker"]: r.get("close") or r.get("ohlc_avg")
        for r in records
        if r.get("close") or r.get("ohlc_avg")
    }


def _load_open_positions() -> list[dict]:
    if not _POSITIONS_FILE.exists():
        return []
    with open(_POSITIONS_FILE) as f:
        positions = json.load(f)
    return [p for p in positions if p.get("status") == "open"]


def check_and_reopen_expirations(eval_date: str) -> int:
    """
    For each open stock position, close any expired options legs (using intrinsic
    value — no chain needed) and queue options_chain tasks for strategies that need
    new legs opened.

    Returns number of reopen tasks queued.
    """
    from src.strategy.snapshot import check_expirations
    from src.collection.queue import WorkQueue

    positions = _load_open_positions()
    if not positions:
        log.info("strategy_runner.no_open_positions")
        return 0

    prices = _load_latest_prices()
    queue = WorkQueue()
    reopen_count = 0

    for pos in positions:
        ticker = pos["ticker"]
        stock_entry_date = pos["entry_date"]
        stock_price = prices.get(ticker)

        if stock_price is None:
            log.warning("strategy_runner.no_price", ticker=ticker)
            continue

        needs_reopen = check_expirations(
            ticker=ticker,
            stock_entry_date=stock_entry_date,
            eval_date=eval_date,
            stock_price=stock_price,
        )

        if needs_reopen:
            queue.enqueue(
                task_type="options_chain",
                ticker=ticker,
                requested_date=eval_date,
                requested_by="strategy_runner",
                priority="high",
                metadata={
                    "reason": "reopen",
                    "strategies_to_reopen": needs_reopen,
                    "stock_entry_date": stock_entry_date,
                    "originating_models": pos.get("originating_models", []),
                },
            )
            log.info(
                "strategy_runner.reopen_queued",
                ticker=ticker,
                strategies=needs_reopen,
            )
            reopen_count += 1

    return reopen_count


def run() -> None:
    from dotenv import load_dotenv
    load_dotenv()

    eval_date = date.today().isoformat()
    log.info("strategy_runner.starting", eval_date=eval_date)

    # Step 1: close expired legs and queue chain fetches for strategies needing reopen
    reopen_queued = check_and_reopen_expirations(eval_date)
    log.info("strategy_runner.expirations_checked", reopen_tasks_queued=reopen_queued)

    # Step 2: aggregate returns across all closed observations
    result = aggregate_returns()
    save_returns(result)

    theoretical = aggregate_theoretical_returns()
    save_theoretical_returns(theoretical)

    total_obs = sum(
        stats["count"]
        for strategies in result.values()
        for stats in strategies.values()
    )
    theoretical_obs = sum(
        stats["count"]
        for strategies in theoretical.values()
        for stats in strategies.values()
    )
    log.info("strategy_runner.done", models=len(result), closed_observations=total_obs,
             theoretical_closed=theoretical_obs)


if __name__ == "__main__":
    run()
