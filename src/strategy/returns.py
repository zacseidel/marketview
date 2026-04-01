"""
src/strategy/returns.py

Aggregates log returns across all strategy observations, grouped by
originating model and strategy type.

Output: data/strategy_observations/returns.json
    {
        "model_name": {
            "strategy_name": {
                "count": 42,
                "mean_log_return": 0.082,
                "std_log_return": 0.041,
                "win_rate": 0.67,
                "returns": [0.12, -0.03, ...]   # all observations
            }
        }
    }
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import structlog

from src.strategy.snapshot import load_all_observations, load_all_theoretical_observations

log = structlog.get_logger()

_RETURNS_FILE = Path("data/strategy_observations/returns.json")
_THEORETICAL_RETURNS_FILE = Path("data/strategy_observations/theoretical/returns.json")


def aggregate_returns() -> dict:
    """
    Read all closed strategy observations and compute return statistics
    grouped by (originating_model, strategy).
    """
    observations = load_all_observations()
    closed = [o for o in observations if o.status == "closed" and o.log_return is not None]

    # Build nested structure: {model: {strategy: [log_returns]}}
    raw: dict[str, dict[str, list[float]]] = {}

    for obs in closed:
        for model in obs.originating_models:
            if model not in raw:
                raw[model] = {}
            if obs.strategy not in raw[model]:
                raw[model][obs.strategy] = []
            raw[model][obs.strategy].append(obs.log_return)

    result: dict = {}
    for model, strategies in raw.items():
        result[model] = {}
        for strategy, returns in strategies.items():
            n = len(returns)
            mean = sum(returns) / n
            variance = sum((r - mean) ** 2 for r in returns) / n if n > 1 else 0.0
            std = math.sqrt(variance)
            wins = sum(1 for r in returns if r > 0)

            result[model][strategy] = {
                "count": n,
                "mean_log_return": round(mean, 6),
                "std_log_return": round(std, 6),
                "win_rate": round(wins / n, 3) if n > 0 else None,
                "returns": [round(r, 6) for r in sorted(returns)],
            }

    log.info(
        "returns.aggregated",
        models=len(result),
        total_observations=len(closed),
    )
    return result


def aggregate_theoretical_returns() -> dict:
    """
    Same aggregation as aggregate_returns() but over theoretical observations only.
    Grouped by (originating_model, strategy) — currently always model="momentum".
    """
    observations = load_all_theoretical_observations()
    closed = [o for o in observations if o.status == "closed" and o.log_return is not None]

    raw: dict[str, dict[str, list[float]]] = {}
    for obs in closed:
        for model in obs.originating_models:
            if model not in raw:
                raw[model] = {}
            if obs.strategy not in raw[model]:
                raw[model][obs.strategy] = []
            raw[model][obs.strategy].append(obs.log_return)

    result: dict = {}
    for model, strategies in raw.items():
        result[model] = {}
        for strategy, returns in strategies.items():
            n = len(returns)
            mean = sum(returns) / n
            variance = sum((r - mean) ** 2 for r in returns) / n if n > 1 else 0.0
            std = math.sqrt(variance)
            wins = sum(1 for r in returns if r > 0)
            result[model][strategy] = {
                "count": n,
                "mean_log_return": round(mean, 6),
                "std_log_return": round(std, 6),
                "win_rate": round(wins / n, 3) if n > 0 else None,
                "returns": [round(r, 6) for r in sorted(returns)],
            }

    log.info("theoretical_returns.aggregated", models=len(result), total_observations=len(closed))
    return result


def save_returns(result: dict) -> None:
    _RETURNS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(_RETURNS_FILE, "w") as f:
        json.dump(result, f, indent=2)


def save_theoretical_returns(result: dict) -> None:
    _THEORETICAL_RETURNS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(_THEORETICAL_RETURNS_FILE, "w") as f:
        json.dump(result, f, indent=2)


def print_summary(result: dict) -> None:
    for model, strategies in sorted(result.items()):
        print(f"\n{model}")
        print(f"  {'Strategy':<20} {'N':>5} {'Mean LogRet':>12} {'StdDev':>8} {'Win%':>6}")
        print(f"  {'-'*55}")
        for strategy, stats in sorted(strategies.items()):
            print(
                f"  {strategy:<20} {stats['count']:>5} "
                f"{stats['mean_log_return']:>12.4f} "
                f"{stats['std_log_return']:>8.4f} "
                f"{stats['win_rate']*100:>5.1f}%"
            )


if __name__ == "__main__":
    result = aggregate_returns()
    save_returns(result)
    if result:
        print_summary(result)
    else:
        print("No closed observations yet.")
