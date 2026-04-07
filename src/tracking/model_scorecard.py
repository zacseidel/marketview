"""
src/tracking/model_scorecard.py

Tracks each model's theoretical performance (Layer 1 of three-layer tracking).

For each enabled model, maintains a theoretical portfolio:
  - Enter at OHLC avg (falling back to close) on the first new_buy eval date
  - Exit when the model produces a sell record for that ticker
  - For tickers still held, mark at the latest available price

SPY comparison is computed over the identical date window for every position,
so alpha is measured against a passive hold of the benchmark.

Computes per-model metrics:
  - signal_count:       total tickers ever recommended (open + closed)
  - closed_count:       positions already exited
  - avg_return:         mean log return across all observations
  - avg_excess_return:  mean (log_return - spy_log_return) per position
  - hit_rate:           fraction of observations with positive absolute return
  - beat_spy_rate:      fraction of observations that beat SPY over same window
  - total_return:       sum of log returns

Scorecards stored in data.nosync/models/scorecards/{model}.json.
Updated daily by daily-dashboard.yml.

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

_MODELS_DIR = Path("data.nosync/models")
_PRICES_DIR = Path("data.nosync/prices")
_SCORECARDS_DIR = Path("data.nosync/models/scorecards")


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
    # SPY comparison — all measured over identical per-position date windows
    avg_spy_return: float | None = None     # mean SPY log ret over those same windows
    avg_excess_return: float | None = None  # avg_return - avg_spy_return  (alpha)
    beat_spy_rate: float | None = None      # fraction of positions that beat SPY
    returns: list[float] = field(default_factory=list)
    spy_returns: list[float] = field(default_factory=list)
    excess_returns: list[float] = field(default_factory=list)
    # Per-position detail for drill-down rendering.
    # Each entry: {ticker, status, entry_date, exit_date, log_ret, spy_log_ret, alpha}
    positions: list[dict] = field(default_factory=list)
    # One entry per run date; never overwritten, only appended.
    history: list[dict] = field(default_factory=list)


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
    SPY log return is computed over the identical window for each position so
    that excess return (alpha) is measured against a passive benchmark hold.
    Returns and persists a ModelScorecard.
    """
    evals = _load_model_evals(model_name)
    today = date.today().isoformat()

    if not evals:
        sc = ModelScorecard(
            model=model_name, as_of_date=today, signal_count=0, closed_count=0,
            hit_rate=None, avg_return=None, win_rate=None, total_return=0.0,
        )
        _save_scorecard(sc)
        return sc

    # {ticker: {entry_price, spy_entry, entry_date}}
    open_positions:  dict[str, dict] = {}
    closed_returns:  list[float] = []
    closed_spy_rets: list[float] = []  # SPY log ret over each closed position's window
    positions_detail: list[dict] = []  # per-position drill-down records

    latest_date = evals[-1][0]

    for eval_date, holdings in evals:
        prices       = _load_prices_for_date(eval_date)
        spy_price    = prices.get("SPY")
        sell_tickers = {h["ticker"] for h in holdings if h.get("status") == "sell"}

        # Open new theoretical positions on new_buy
        for h in holdings:
            ticker = h["ticker"]
            if h.get("status") == "new_buy" and ticker not in open_positions:
                entry_price = prices.get(ticker)
                if entry_price and entry_price > 0:
                    open_positions[ticker] = {
                        "entry_price": entry_price,
                        "spy_entry":   spy_price,
                        "entry_date":  eval_date,
                    }

        # Close theoretical positions on sell
        for ticker in sell_tickers:
            if ticker not in open_positions:
                continue
            exit_price = prices.get(ticker)
            if exit_price and exit_price > 0:
                pos = open_positions[ticker]
                ret = round(math.log(exit_price / pos["entry_price"]), 6)
                closed_returns.append(ret)
                spy_entry = pos.get("spy_entry")
                spy_ret = None
                if spy_entry and spy_price and spy_entry > 0 and spy_price > 0:
                    spy_ret = round(math.log(spy_price / spy_entry), 6)
                    closed_spy_rets.append(spy_ret)
                alpha = round(ret - spy_ret, 6) if spy_ret is not None else None
                entry_date = pos.get("entry_date", "")
                days = None
                if entry_date:
                    try:
                        from datetime import date as _date
                        d0 = _date.fromisoformat(entry_date)
                        d1 = _date.fromisoformat(eval_date)
                        days = (d1 - d0).days
                    except Exception:
                        pass
                positions_detail.append({
                    "ticker":      ticker,
                    "status":      "closed",
                    "entry_date":  entry_date,
                    "exit_date":   eval_date,
                    "days":        days,
                    "log_ret":     ret,
                    "spy_log_ret": spy_ret,
                    "alpha":       alpha,
                })
            del open_positions[ticker]

    # Mark still-open positions at latest available prices
    latest_prices    = _load_prices_for_date(latest_date)
    spy_latest       = latest_prices.get("SPY")
    open_returns:    list[float] = []
    open_spy_rets:   list[float] = []

    for ticker, pos in open_positions.items():
        current_price = latest_prices.get(ticker)
        if not (current_price and current_price > 0 and pos["entry_price"] > 0):
            continue
        ret = round(math.log(current_price / pos["entry_price"]), 6)
        open_returns.append(ret)
        spy_entry = pos.get("spy_entry")
        spy_ret = None
        if spy_entry and spy_latest and spy_entry > 0 and spy_latest > 0:
            spy_ret = round(math.log(spy_latest / spy_entry), 6)
            open_spy_rets.append(spy_ret)
        alpha = round(ret - spy_ret, 6) if spy_ret is not None else None
        entry_date = pos.get("entry_date", "")
        days = None
        if entry_date:
            try:
                from datetime import date as _date
                d0 = _date.fromisoformat(entry_date)
                d1 = _date.fromisoformat(latest_date)
                days = (d1 - d0).days
            except Exception:
                pass
        positions_detail.append({
            "ticker":      ticker,
            "status":      "open",
            "entry_date":  entry_date,
            "exit_date":   None,
            "days":        days,
            "log_ret":     ret,
            "spy_log_ret": spy_ret,
            "alpha":       alpha,
        })

    all_returns  = closed_returns  + open_returns
    all_spy_rets = closed_spy_rets + open_spy_rets
    signal_count = len(open_positions) + len(closed_returns)

    # Sort positions: closed first (by exit_date desc), then open (by entry_date desc)
    positions_detail.sort(
        key=lambda p: (p["status"] == "open", -(p.get("alpha") or 0)),
    )

    if not all_returns:
        sc = ModelScorecard(
            model=model_name, as_of_date=today, signal_count=signal_count,
            closed_count=len(closed_returns), hit_rate=None, avg_return=None,
            win_rate=None, total_return=0.0, positions=positions_detail,
        )
    else:
        n    = len(all_returns)
        avg  = sum(all_returns) / n
        wins = sum(1 for r in all_returns if r > 0)

        avg_spy    = round(sum(all_spy_rets) / len(all_spy_rets), 4) if all_spy_rets else None
        # Alpha = difference of averages (not average of differences — same math, explicit intent)
        avg_excess = round(avg - avg_spy, 4) if avg_spy is not None else None
        all_excess = [r - s for r, s in zip(all_returns, all_spy_rets)] if all_spy_rets else []
        beat_spy   = round(sum(1 for e in all_excess if e > 0) / len(all_excess), 3) if all_excess else None

        sc = ModelScorecard(
            model=model_name,
            as_of_date=today,
            signal_count=signal_count,
            closed_count=len(closed_returns),
            hit_rate=round(wins / n, 3),
            avg_return=round(avg, 4),
            win_rate=round(wins / n, 3),
            total_return=round(sum(all_returns), 4),
            avg_spy_return=avg_spy,
            avg_excess_return=avg_excess,
            beat_spy_rate=beat_spy,
            returns=all_returns,
            spy_returns=all_spy_rets,
            excess_returns=all_excess,
            positions=positions_detail,
        )

    # Append a dated snapshot to history (deduplicate by date so re-runs are safe).
    prior_history = _load_existing_history(model_name)
    snapshot = {
        "as_of_date":        today,
        "signal_count":      sc.signal_count,
        "closed_count":      sc.closed_count,
        "avg_return":        sc.avg_return,
        "avg_spy_return":    sc.avg_spy_return,
        "avg_excess_return": sc.avg_excess_return,
        "hit_rate":          sc.hit_rate,
        "beat_spy_rate":     sc.beat_spy_rate,
        "total_return":      sc.total_return,
    }
    history = [e for e in prior_history if e.get("as_of_date") != today]
    history.append(snapshot)
    history.sort(key=lambda e: e["as_of_date"])
    sc.history = history

    _save_scorecard(sc)
    log.info(
        "scorecard.updated",
        model=model_name,
        signals=signal_count,
        closed=len(closed_returns),
        avg_return=sc.avg_return,
        avg_excess=sc.avg_excess_return,
    )
    return sc


def _load_existing_history(model_name: str) -> list[dict]:
    """Load history array from an existing scorecard, if present."""
    path = _SCORECARDS_DIR / f"{model_name}.json"
    if not path.exists():
        return []
    try:
        with open(path) as f:
            data = json.load(f)
        return data.get("history", [])
    except Exception:
        return []


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
        # Tolerate files written before any of these fields existed
        data.setdefault("avg_spy_return", None)
        data.setdefault("avg_excess_return", None)
        data.setdefault("beat_spy_rate", None)
        data.setdefault("spy_returns", [])
        data.setdefault("excess_returns", [])
        data.setdefault("positions", [])
        data.setdefault("history", [])
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
        if sc.avg_return is not None:
            spy_str    = f", spy={sc.avg_spy_return:+.2%}"       if sc.avg_spy_return   is not None else ""
            alpha_str  = f", alpha={sc.avg_excess_return:+.2%}"  if sc.avg_excess_return is not None else ""
            print(
                f"{model_name}: signals={sc.signal_count}, closed={sc.closed_count}, "
                f"avg_return={sc.avg_return:+.2%}{spy_str}{alpha_str}"
            )
        else:
            print(f"{model_name}: signals={sc.signal_count} — no return data yet")
