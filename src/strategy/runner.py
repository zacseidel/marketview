"""
src/strategy/runner.py

Entry point for strategy evaluation (called by evaluate-strategies.yml).
Aggregates log returns across all closed strategy observations and saves
a summary to data/strategy_observations/returns.json.

Usage:
    python -m src.strategy.runner
"""

from __future__ import annotations

import structlog

from src.strategy.returns import aggregate_returns, save_returns, print_summary

log = structlog.get_logger()


def run() -> None:
    from dotenv import load_dotenv
    load_dotenv()

    log.info("strategy_runner.starting")
    result = aggregate_returns()
    save_returns(result)

    total_obs = sum(
        stats["count"]
        for strategies in result.values()
        for stats in strategies.values()
    )
    log.info("strategy_runner.done", models=len(result), closed_observations=total_obs)


if __name__ == "__main__":
    run()
