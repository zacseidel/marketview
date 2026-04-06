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
    """Find the most recent prior eval's holdings for a model."""
    models_dir = Path("data.nosync/models")
    if not models_dir.exists():
        return set()

    # All eval date dirs, sorted descending, skip current
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
    """
    Classify each holding as new_buy, hold.
    Then add sell records for tickers that were held before but dropped out.
    """
    current_tickers = {h.ticker for h in holdings}

    for h in holdings:
        h.status = "hold" if h.ticker in prev_tickers else "new_buy"

    # Add sell records for tickers in previous list but not current
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

        prev_tickers = _prev_holdings_tickers(model_name, eval_date, dal)
        holdings = _assign_statuses(holdings, prev_tickers, eval_date, model_name, dal)
        dal.save_model_output(holdings, eval_date, model_name)

    log.info("runner.done", eval_date=eval_date)

    # Generate decision markdown
    from src.decisions.generate import generate_decision_file
    generate_decision_file(eval_date)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-date", default=None, help="YYYY-MM-DD (default: today)")
    args = parser.parse_args()
    run_models(eval_date=args.eval_date)
