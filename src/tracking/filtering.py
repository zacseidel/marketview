"""
src/tracking/filtering.py

Measures filtering value-add: does user curation improve model returns? (Layer 3)

For each model, compares:
  - Model's full theoretical avg return (from model_scorecard.py)
  - User's subset of that model's recommendations (actual positions)

Filtering alpha = user_avg_return - model_avg_return
  - positive: user's picks outperformed the model's full list
  - negative: user filtering hurt returns

Output stored in data/positions/filtering_analysis.json.

Entry points:
    compute_filtering_value_add(model_name: str, as_of_date: str | None) -> FilteringAnalysis
    get_all_filtering_analysis() -> dict[str, FilteringAnalysis]
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path

import structlog

from src.tracking.model_scorecard import get_all_scorecards

log = structlog.get_logger()

_POSITIONS_FILE = Path("data/positions/positions.json")
_FILTERING_FILE = Path("data/positions/filtering_analysis.json")


@dataclass
class FilteringAnalysis:
    model: str
    as_of_date: str
    model_signal_count: int
    user_position_count: int
    model_avg_return: float | None
    user_avg_return: float | None
    alpha: float | None          # user_avg_log_return - model_avg_log_return; positive = user added value


def _load_positions() -> list[dict]:
    if not _POSITIONS_FILE.exists():
        return []
    with open(_POSITIONS_FILE) as f:
        return json.load(f)


def _user_returns_by_model(positions: list[dict]) -> dict[str, list[float]]:
    """
    For each model, collect log returns from positions where that model was listed
    in originating_models.
    - Closed: math.log(exit_price / entry_price)
    - Open: math.log((entry_price + unrealized_pnl) / entry_price)
    Returns {model: [log_return, ...]}
    """
    result: dict[str, list[float]] = {}
    for pos in positions:
        entry_price = pos.get("entry_price") or 0.0
        if entry_price <= 0:
            continue

        if pos.get("status") == "closed":
            exit_price = pos.get("exit_price")
            if not exit_price or exit_price <= 0:
                continue
            log_ret = math.log(exit_price / entry_price)
        else:
            unrealized = pos.get("unrealized_pnl")
            if unrealized is None:
                continue
            current_price = entry_price + unrealized
            if current_price <= 0:
                continue
            log_ret = math.log(current_price / entry_price)

        for model in pos.get("originating_models", []):
            result.setdefault(model, []).append(round(log_ret, 6))
    return result


def compute_filtering_value_add(
    model_name: str,
    as_of_date: str | None = None,
) -> FilteringAnalysis:
    if as_of_date is None:
        as_of_date = date.today().isoformat()

    scorecards = get_all_scorecards()
    scorecard = scorecards.get(model_name)

    positions = _load_positions()
    user_by_model = _user_returns_by_model(positions)
    user_returns = user_by_model.get(model_name, [])

    model_avg = scorecard.avg_return if scorecard else None
    user_avg = round(sum(user_returns) / len(user_returns), 4) if user_returns else None
    alpha = round(user_avg - model_avg, 4) if user_avg is not None and model_avg is not None else None

    analysis = FilteringAnalysis(
        model=model_name,
        as_of_date=as_of_date,
        model_signal_count=scorecard.signal_count if scorecard else 0,
        user_position_count=len(user_returns),
        model_avg_return=model_avg,
        user_avg_return=user_avg,
        alpha=alpha,
    )

    log.info(
        "filtering.computed",
        model=model_name,
        user_count=len(user_returns),
        model_avg=model_avg,
        user_avg=user_avg,
        alpha=alpha,
    )
    return analysis


def get_all_filtering_analysis(as_of_date: str | None = None) -> dict[str, FilteringAnalysis]:
    """Compute filtering analysis for all enabled models and persist to disk."""
    import yaml

    config_path = Path("config/models.yaml")
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    enabled_models = [name for name, mc in cfg["models"].items() if mc.get("enabled", False)]

    result: dict[str, FilteringAnalysis] = {}
    for model_name in enabled_models:
        result[model_name] = compute_filtering_value_add(model_name, as_of_date)

    _FILTERING_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = _FILTERING_FILE.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump({k: asdict(v) for k, v in result.items()}, f, indent=2)
    tmp.replace(_FILTERING_FILE)

    return result


if __name__ == "__main__":
    import sys
    from dotenv import load_dotenv
    load_dotenv()

    date_arg = sys.argv[1] if len(sys.argv) > 1 else None
    analyses = get_all_filtering_analysis(date_arg)
    for model, a in sorted(analyses.items()):
        alpha_str = f"{a.alpha:+.2%}" if a.alpha is not None else "—"
        model_str = f"{a.model_avg_return:+.2%}" if a.model_avg_return is not None else "—"
        user_str = f"{a.user_avg_return:+.2%}" if a.user_avg_return is not None else "—"
        print(
            f"{model}: model={model_str}  user={user_str}  "
            f"alpha={alpha_str}  (user_positions={a.user_position_count})"
        )
