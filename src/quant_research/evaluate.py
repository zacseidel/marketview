"""
src/quant_research/evaluate.py

Shared evaluation utilities for comparing model performance on the validation set.

Given predicted scores for (ticker, date) pairs and actual forward returns,
computes portfolio-level metrics by treating each evaluation date as a
"top-N picks" equal-weight portfolio held for 20 days.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import structlog

log = structlog.get_logger()

_SPY_TICKER = "SPY"
_TOP_N = 20
_FORWARD_DAYS = 20
_TRADING_DAYS_PER_YEAR = 252


def evaluate_model(
    val_df: pd.DataFrame,
    score_fn: Callable[[pd.DataFrame], np.ndarray],
    model_name: str,
    top_n: int = _TOP_N,
    min_score_threshold: float | None = None,
    forward_days: int = _FORWARD_DAYS,
    target_col: str = "fwd_log_ret_20d",
) -> dict:
    """
    Evaluate a model on the validation set.

    Args:
        val_df: DataFrame with columns [date, ticker, <features>, <target_col>]
                filtered to split=='val' only.
        score_fn: callable(df) -> np.ndarray of scores (higher = better), same length as df.
        model_name: name for logging.
        top_n: number of picks per evaluation date.
        min_score_threshold: if set, only include tickers whose score >= threshold.
            Periods with no qualifying tickers are skipped (not counted as flat periods).
            This is the right mode for filter-style models (e.g. cluster) where a
            score of NaN / below threshold means "don't buy anything today".
        forward_days: rebalance cadence in trading days (default 20). Use 10 for
            models trained on 10-day forward returns.
        target_col: column name for the forward return used in evaluation
            (default "fwd_log_ret_20d"). Set to "fwd_log_ret_10d" for v2 models.

    Returns:
        dict with metrics.
    """
    val_df = val_df.copy()

    # Pre-filter to non-overlapping eval dates before scoring — avoids scoring
    # all rows when only every Nth day × all tickers are actually needed.
    all_dates = sorted(val_df["date"].unique())
    eval_dates = all_dates[::forward_days]
    eval_date_set = set(eval_dates)

    # Keep SPY rows (needed for excess return) + rows on eval dates only
    score_df = val_df[(val_df["date"].isin(eval_date_set)) | (val_df["ticker"] == _SPY_TICKER)].copy()

    log.info(
        "evaluate.scoring",
        model=model_name,
        rows_to_score=len(score_df),
        eval_dates=len(eval_dates),
        min_score_threshold=min_score_threshold,
        forward_days=forward_days,
    )
    score_df["score"] = score_fn(score_df)

    spy_returns = (
        score_df[score_df["ticker"] == _SPY_TICKER]
        .set_index("date")[target_col]
        .to_dict()
    )

    portfolio_returns: list[float] = []
    spy_rets: list[float] = []
    hit_rates: list[float] = []

    for d in eval_dates:
        day_df = score_df[(score_df["date"] == d) & (score_df["ticker"] != _SPY_TICKER)]

        # Apply threshold filter: only consider tickers above min_score_threshold
        if min_score_threshold is not None:
            day_df = day_df[day_df["score"] >= min_score_threshold]
            if day_df.empty:
                continue  # no qualifying picks this period — skip entirely
            top = day_df.nlargest(min(top_n, len(day_df)), "score")
        else:
            if len(day_df) < top_n:
                continue
            top = day_df.nlargest(top_n, "score")

        actual_rets = top[target_col].dropna()
        if len(actual_rets) < max(1, top_n // 2 if min_score_threshold is None else 1):
            continue

        port_ret = actual_rets.mean()
        portfolio_returns.append(port_ret)
        hit_rates.append((actual_rets > 0).mean())

        spy_ret = spy_returns.get(d, np.nan)
        if not math.isnan(spy_ret):
            spy_rets.append(spy_ret)

    if not portfolio_returns:
        log.warning("evaluate.no_results", model=model_name)
        return {"model": model_name, "error": "no evaluation periods"}

    port_arr = np.array(portfolio_returns)
    spy_arr = np.array(spy_rets) if spy_rets else np.zeros(len(port_arr))
    excess = port_arr[: len(spy_arr)] - spy_arr

    # Annualized Sharpe on 20-day non-overlapping periods
    periods_per_year = _TRADING_DAYS_PER_YEAR / forward_days
    sharpe = (
        float(excess.mean() / excess.std() * math.sqrt(periods_per_year))
        if excess.std() > 0
        else 0.0
    )

    metrics = {
        "model": model_name,
        "eval_periods": len(portfolio_returns),
        "avg_log_ret": round(float(port_arr.mean()), 5),
        "avg_spy_ret": round(float(spy_arr.mean()), 5) if len(spy_arr) else None,
        "avg_excess_ret": round(float(excess.mean()), 5) if len(excess) else None,
        "hit_rate": round(float(np.mean(hit_rates)), 3),
        "sharpe": round(sharpe, 3),
    }

    log.info("evaluate.result", **{k: v for k, v in metrics.items() if v is not None})
    return metrics


_VAL_METRICS_FILE = Path("data.nosync/quant/val_metrics.json")


def save_val_metrics(result: dict) -> None:
    """Upsert one model's val metrics into data.nosync/quant/val_metrics.json."""
    existing: dict = {}
    if _VAL_METRICS_FILE.exists():
        with open(_VAL_METRICS_FILE) as f:
            existing = json.load(f)
    existing[result["model"]] = result
    _VAL_METRICS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(_VAL_METRICS_FILE, "w") as f:
        json.dump(existing, f, indent=2)


def print_comparison(results: list[dict]) -> None:
    """Print a side-by-side comparison table of model metrics."""
    print("\n" + "=" * 72)
    print(f"{'Model':<18} {'Periods':>8} {'AvgRet':>8} {'SPYRet':>8} "
          f"{'Excess':>8} {'HitRate':>8} {'Sharpe':>8}")
    print("-" * 72)
    for m in results:
        if "error" in m:
            print(f"{m['model']:<18}  ERROR: {m['error']}")
            continue
        print(
            f"{m['model']:<18} "
            f"{m['eval_periods']:>8d} "
            f"{m['avg_log_ret']:>8.4f} "
            f"{(m['avg_spy_ret'] or 0):>8.4f} "
            f"{(m['avg_excess_ret'] or 0):>8.4f} "
            f"{m['hit_rate']:>8.1%} "
            f"{m['sharpe']:>8.3f}"
        )
    print("=" * 72 + "\n")
