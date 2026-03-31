"""
src/tracking/model_scorecard.py

Tracks each model's theoretical performance (Layer 1 of three-layer tracking).

For each enabled model, maintains a theoretical portfolio:
  - Enter at OHLC avg (falling back to close) on the first new_buy eval date
  - Exit when the model produces a sell record for that ticker
  - For tickers still held, mark at the latest available price

Computes per-model metrics:
  - signal_count: total tickers ever recommended (open + closed)
  - closed_count: positions already exited
  - hit_rate: fraction of all observations (open + closed) with positive return
  - avg_return: mean log return across all observations
  - total_return: sum of log returns (= log of cumulative return)
  - returns: list of individual simple returns

Scorecards stored in data/models/scorecards/{model}.json.
Updated by weekly-digest.yml.

Entry points:
    update_model_scorecard(model_name: str) -> ModelScorecard
    get_all_scorecards() -> dict[str, ModelScorecard]
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path

import structlog

log = structlog.get_logger()

_MODELS_DIR = Path("data/models")
_PRICES_DIR = Path("data/prices")
_SCORECARDS_DIR = Path("data/models/scorecards")


@dataclass
class ModelScorecard:
    model: str
    as_of_date: str
    signal_count: int
    closed_count: int
    hit_rate: float | None
    avg_return: float | None
    win_rate: float | None
    total_return: float
    returns: list[float] = field(default_factory=list)


def _load_model_evals(model: str) -> list[tuple[str, list[dict]]]:
    """Return sorted [(eval_date, holdings), ...] for a model across all eval dirs."""
    if not _MODELS_DIR.exists():
        return []
    result: list[tuple[str, list[dict]]] = []
    for d in sorted(d for d in _MODELS_DIR.iterdir() if d.is_dir() and d.name[0].isdigit()):
        path = d / f"{model}.json"
        if path.exists():
            with open(path) as f:
                result.append((d.name, json.load(f)))
    return result


def _load_prices_for_date(target_date: str) -> dict[str, float]:
    """
    Load OHLC-avg (or close) prices for target_date.
    Falls back to the most recent prior price file if exact date is missing.
    """
    price_file = _PRICES_DIR / f"{target_date}.json"
    if not price_file.exists():
        candidates = sorted(
            [f for f in _PRICES_DIR.glob("*.json") if f.stem <= target_date],
            reverse=True,
        )
        if not candidates:
            return {}
        price_file = candidates[0]

    with open(price_file) as f:
        records = json.load(f)
    return {
        r["ticker"]: r.get("ohlc_avg") or r.get("close")
        for r in records
        if r.get("ohlc_avg") or r.get("close")
    }


def update_model_scorecard(model_name: str) -> ModelScorecard:
    """
    Replay all eval dates for a model to build its theoretical return series.
    Returns and persists a ModelScorecard.
    """
    evals = _load_model_evals(model_name)
    today = date.today().isoformat()

    if not evals:
        sc = ModelScorecard(
            model=model_name,
            as_of_date=today,
            signal_count=0,
            closed_count=0,
            hit_rate=None,
            avg_return=None,
            win_rate=None,
            total_return=0.0,
        )
        _save_scorecard(sc)
        return sc

    # {ticker: {entry_date, entry_price}}
    open_positions: dict[str, dict] = {}
    closed_returns: list[float] = []

    latest_date = evals[-1][0]

    for eval_date, holdings in evals:
        prices = _load_prices_for_date(eval_date)
        sell_tickers = {h["ticker"] for h in holdings if h.get("status") == "sell"}

        # Open new theoretical positions on new_buy
        for h in holdings:
            if h.get("status") == "new_buy" and h["ticker"] not in open_positions:
                entry_price = prices.get(h["ticker"])
                if entry_price and entry_price > 0:
                    open_positions[h["ticker"]] = {
                        "entry_date": eval_date,
                        "entry_price": entry_price,
                    }

        # Close theoretical positions on sell
        for ticker in sell_tickers:
            if ticker in open_positions:
                exit_price = prices.get(ticker)
                if exit_price and exit_price > 0:
                    entry_price = open_positions[ticker]["entry_price"]
                    ret = math.log(exit_price / entry_price)
                    closed_returns.append(round(ret, 6))
                del open_positions[ticker]

    # Mark still-open positions at latest available prices
    latest_prices = _load_prices_for_date(latest_date)
    open_returns: list[float] = []
    for ticker, pos in open_positions.items():
        current_price = latest_prices.get(ticker)
        if current_price and current_price > 0 and pos["entry_price"] > 0:
            ret = math.log(current_price / pos["entry_price"])
            open_returns.append(round(ret, 6))

    all_returns = closed_returns + open_returns
    signal_count = len(open_positions) + len(closed_returns)

    if not all_returns:
        sc = ModelScorecard(
            model=model_name,
            as_of_date=today,
            signal_count=signal_count,
            closed_count=len(closed_returns),
            hit_rate=None,
            avg_return=None,
            win_rate=None,
            total_return=0.0,
        )
    else:
        n = len(all_returns)
        avg = sum(all_returns) / n
        wins = sum(1 for r in all_returns if r > 0)
        sc = ModelScorecard(
            model=model_name,
            as_of_date=today,
            signal_count=signal_count,
            closed_count=len(closed_returns),
            hit_rate=round(wins / n, 3),
            avg_return=round(avg, 4),
            win_rate=round(wins / n, 3),
            total_return=round(sum(all_returns), 4),
            returns=all_returns,
        )

    _save_scorecard(sc)
    log.info(
        "scorecard.updated",
        model=model_name,
        signals=signal_count,
        closed=len(closed_returns),
        avg_return=sc.avg_return,
    )
    return sc


def _save_scorecard(sc: ModelScorecard) -> None:
    _SCORECARDS_DIR.mkdir(parents=True, exist_ok=True)
    path = _SCORECARDS_DIR / f"{sc.model}.json"
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(asdict(sc), f, indent=2)
    tmp.replace(path)


def get_all_scorecards() -> dict[str, ModelScorecard]:
    """Load all persisted scorecards from disk."""
    if not _SCORECARDS_DIR.exists():
        return {}
    result: dict[str, ModelScorecard] = {}
    for f in _SCORECARDS_DIR.glob("*.json"):
        with open(f) as fp:
            data = json.load(fp)
        result[data["model"]] = ModelScorecard(**data)
    return result


if __name__ == "__main__":
    import sys
    import yaml
    from dotenv import load_dotenv

    load_dotenv()

    config_path = Path("config/models.yaml")
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    enabled_models = [name for name, mc in cfg["models"].items() if mc.get("enabled", False)]

    target = sys.argv[1:] if len(sys.argv) > 1 else enabled_models
    for model_name in target:
        sc = update_model_scorecard(model_name)
        if sc.hit_rate is not None:
            print(
                f"{model_name}: signals={sc.signal_count}, closed={sc.closed_count}, "
                f"hit_rate={sc.hit_rate:.1%}, avg_return={sc.avg_return:+.2%}"
            )
        else:
            print(f"{model_name}: signals={sc.signal_count} — no return data yet")
