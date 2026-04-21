"""
src/selection/runner.py

Entry point for the model evaluation run (called by run-models.yml).
Loads all enabled models from config/models.yaml, runs each, saves outputs,
and generates the decision markdown.

Usage:
    python -m src.selection.runner [--eval-date YYYY-MM-DD]
"""

from __future__ import annotations

import importlib
import json
from datetime import date, timedelta
from pathlib import Path

import structlog
import yaml

from src.selection.base import DataAccessLayer, HoldingRecord, SelectionModel

log = structlog.get_logger()

_MODELS_CONFIG = Path("config/models.yaml")


def _load_model_class(module_path: str, class_name: str) -> type:
    mod = importlib.import_module(module_path)
    return getattr(mod, class_name)


def _prev_holdings_tickers(model: str, eval_date: str, dal: DataAccessLayer) -> set[str]:
    """Find the most recent prior eval's holdings for a model (rank-based models)."""
    models_dir = Path("data.nosync/models")
    if not models_dir.exists():
        return set()

    dirs = sorted(
        [d for d in models_dir.iterdir() if d.is_dir() and d.name < eval_date],
        reverse=True,
    )
    for d in dirs:
        path = d / f"{model}.json"
        if path.exists() and path.stat().st_size > 2:
            with open(path) as f:
                records = json.load(f)
            return {r["ticker"] for r in records if r.get("status") != "sell"}
    return set()


def _assign_statuses(
    holdings: list[HoldingRecord],
    prev_tickers: set[str],
    eval_date: str,
    model: str,
    dal: DataAccessLayer,
) -> list[HoldingRecord]:
    """Classify holdings as new_buy/hold; add sell records for tickers that dropped out."""
    current_tickers = {h.ticker for h in holdings}

    for h in holdings:
        h.status = "hold" if h.ticker in prev_tickers else "new_buy"

    for ticker in prev_tickers - current_tickers:
        holdings.append(HoldingRecord(
            model=model,
            eval_date=eval_date,
            ticker=ticker,
            conviction=0.0,
            rationale="Dropped from model holdings",
            status="sell",
        ))

    return holdings


# ---------------------------------------------------------------------------
# Time-based exit logic (quant_gbm_v3 and any future model with
# time_based_exit_days in its params)
# ---------------------------------------------------------------------------

def _count_trading_days(from_date_str: str, to_date_str: str) -> int:
    """Count trading days strictly after from_date_str up to and including to_date_str."""
    prices_dir = Path("data.nosync/prices")
    return sum(
        1 for p in prices_dir.glob("*.json")
        if p.stem[0].isdigit() and from_date_str < p.stem <= to_date_str
    )


def _prev_holdings_with_entry(model: str, eval_date: str) -> dict[str, str]:
    """
    Returns {ticker: entry_eval_date} for all tickers currently held by a model.

    entry_eval_date is the date of the most recent new_buy for each ticker —
    correctly handling sell-and-rebuy cycles by resetting on every sell.

    Algorithm: scan all eval dirs in chronological order.
      - new_buy  → record/overwrite entry_eval_date (use stored field if present)
      - sell     → clear entry (position closed)
      - hold     → inherit stored entry_eval_date if present, else leave as-is
    After the full scan, only tickers with non-empty dates are still held.
    """
    models_dir = Path("data.nosync/models")
    if not models_dir.exists():
        return {}

    all_dirs = sorted(
        d for d in models_dir.iterdir() if d.is_dir() and d.name < eval_date
    )  # ascending — oldest first

    # entry_dates[ticker] = current entry date ("" means not held)
    entry_dates: dict[str, str] = {}

    for d in all_dirs:
        path = d / f"{model}.json"
        if not path.exists():
            continue
        with open(path) as f:
            records = json.load(f)

        for r in records:
            ticker = r.get("ticker", "")
            status = r.get("status", "")
            if status == "sell":
                entry_dates.pop(ticker, None)
            elif status == "new_buy":
                # Prefer stored field (future runs will have it); fall back to dir date
                entry_dates[ticker] = r.get("entry_eval_date") or d.name
            elif status == "hold":
                # Carry forward stored field if the ticker wasn't already tracked
                stored = r.get("entry_eval_date")
                if ticker not in entry_dates:
                    entry_dates[ticker] = stored or d.name

    return entry_dates


def _assign_statuses_time_based(
    holdings: list[HoldingRecord],
    prev_entries: dict[str, str],   # ticker -> entry_eval_date
    eval_date: str,
    model: str,
    exit_days: int,
) -> list[HoldingRecord]:
    """
    Time-based exit strategy:
      - Each new pick is held for exactly exit_days trading days, then sold.
      - New top picks are added as new_buy unless already in the portfolio.
      - Previously held tickers that haven't expired yet remain as hold
        (whether or not the model still ranks them highly).
    """
    result: list[HoldingRecord] = []
    handled: set[str] = set()

    # Build a quick lookup for this run's model picks
    picks_by_ticker = {h.ticker: h for h in holdings}

    # Process all previously held tickers
    for ticker, last_rec_date in prev_entries.items():
        pick = picks_by_ticker.get(ticker)
        if pick:
            # Model re-picked this ticker → reset the clock to today
            pick.status = "hold"
            pick.entry_eval_date = eval_date
            result.append(pick)
        else:
            # Model did not re-pick — count days since last recommendation
            days_since_last_rec = _count_trading_days(last_rec_date, eval_date)
            if days_since_last_rec >= exit_days:
                result.append(HoldingRecord(
                    model=model,
                    eval_date=eval_date,
                    ticker=ticker,
                    conviction=0.0,
                    rationale=(
                        f"Time exit: {days_since_last_rec} trading days since last rec "
                        f"(last rec {last_rec_date}, target {exit_days}d)"
                    ),
                    status="sell",
                    entry_eval_date=last_rec_date,
                ))
            else:
                result.append(HoldingRecord(
                    model=model,
                    eval_date=eval_date,
                    ticker=ticker,
                    conviction=0.0,
                    rationale=(
                        f"Time hold: {days_since_last_rec}/{exit_days}d since last rec "
                        f"({last_rec_date})"
                    ),
                    status="hold",
                    entry_eval_date=last_rec_date,
                ))
        handled.add(ticker)

    # Add fresh picks not already in the portfolio
    for h in holdings:
        if h.ticker not in handled:
            h.status = "new_buy"
            h.entry_eval_date = eval_date
            result.append(h)

    return result



def run_models(eval_date: str | None = None) -> None:
    from dotenv import load_dotenv
    load_dotenv()

    if eval_date is None:
        eval_date = date.today().isoformat()

    log.info("runner.starting", eval_date=eval_date)

    with open(_MODELS_CONFIG) as f:
        models_cfg = yaml.safe_load(f)

    dal = DataAccessLayer()

    for model_name, model_cfg in models_cfg["models"].items():
        if not model_cfg.get("enabled", False):
            continue

        log.info("runner.running_model", model=model_name)
        try:
            ModelClass = _load_model_class(model_cfg["module"], model_cfg["class"])
            instance: SelectionModel = ModelClass()
            config = {**model_cfg.get("params", {}), "eval_date": eval_date}
            holdings = instance.run(config, dal)
        except Exception as exc:
            log.error("runner.model_error", model=model_name, error=str(exc))
            continue

        exit_days = int(model_cfg.get("params", {}).get("time_based_exit_days", 10))
        prev_entries = _prev_holdings_with_entry(model_name, eval_date)
        holdings = _assign_statuses_time_based(
            holdings, prev_entries, eval_date, model_name, exit_days
        )
        dal.save_model_output(holdings, eval_date, model_name)

    log.info("runner.done", eval_date=eval_date)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-date", default=None, help="YYYY-MM-DD (default: today)")
    args = parser.parse_args()
    run_models(eval_date=args.eval_date)
