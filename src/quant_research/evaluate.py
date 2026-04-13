"""
src/quant_research/evaluate.py

Shared evaluation utilities for comparing model performance on the validation set.

Metrics computed per eval date, then aggregated:
  avg_log_ret      — mean return of top-N picks (equal-weight)
  avg_universe_ret — equal-weight mean of all universe stocks (the benchmark)
  avg_excess_ret   — avg_log_ret - avg_universe_ret (are picks beating the field?)
  avg_spy_ret      — SPY return for reference
  decile_hit_rate  — fraction of top-N picks that landed in the actual top decile
  ic_mean          — mean Spearman IC (rank corr: predicted score vs actual return)
  icir             — IC / IC.std() * sqrt(periods_per_year) — comparable across models
  sharpe           — excess-over-universe Sharpe, annualized

IC and ICIR are the primary cross-model comparables: they are scale-free and
cadence-normalizing, making it valid to compare a 5-day model against a 20-day model.
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
_TOP_DECILE = 0.90   # threshold for decile hit rate


def evaluate_model(
    val_df: pd.DataFrame,
    score_fn: Callable[[pd.DataFrame], np.ndarray],
    model_name: str,
    top_n: int = _TOP_N,
    min_score_threshold: float | None = None,
    forward_days: int = _FORWARD_DAYS,
    target_col: str = "fwd_log_ret_20d",
    eval_weekday: int | None = None,
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
            models trained on 10-day forward returns, 5 for weekly models.
        target_col: column name for the forward return used in evaluation
            (default "fwd_log_ret_20d"). Set to "fwd_log_ret_5d" for weekly models.
        eval_weekday: if set (0=Mon … 6=Sun), restrict eval dates to that weekday only
            and evaluate every qualifying date (no forward_days stride). Use 3 for
            Thursday-trained weekly models where each Thursday is non-overlapping.

    Returns:
        dict with metrics.
    """
    val_df = val_df.copy()

    all_dates = sorted(val_df["date"].unique())
    if eval_weekday is not None:
        eval_dates = [d for d in all_dates if pd.Timestamp(d).weekday() == eval_weekday]
    else:
        eval_dates = all_dates[::forward_days]
    eval_date_set = set(eval_dates)

    # Score all rows on eval dates (including SPY — kept for reference)
    score_df = val_df[val_df["date"].isin(eval_date_set)].copy()

    log.info(
        "evaluate.scoring",
        model=model_name,
        rows_to_score=len(score_df),
        eval_dates=len(eval_dates),
        min_score_threshold=min_score_threshold,
        forward_days=forward_days,
    )
    score_df["score"] = score_fn(score_df)

    # SPY returns — reference only, not used in Sharpe
    spy_returns = (
        score_df[score_df["ticker"] == _SPY_TICKER]
        .set_index("date")[target_col]
        .to_dict()
    )

    portfolio_returns: list[float] = []
    universe_rets: list[float] = []
    spy_rets: list[float] = []
    decile_hits: list[float] = []
    ics: list[float] = []

    for d in eval_dates:
        # All non-SPY stocks this date
        day_df = score_df[(score_df["date"] == d) & (score_df["ticker"] != _SPY_TICKER)]

        if min_score_threshold is not None:
            day_df = day_df[day_df["score"] >= min_score_threshold]
            if day_df.empty:
                continue
            top = day_df.nlargest(min(top_n, len(day_df)), "score")
        else:
            if len(day_df) < top_n:
                continue
            top = day_df.nlargest(top_n, "score")

        actual_rets = top[target_col].dropna()
        min_picks = max(1, top_n // 2 if min_score_threshold is None else 1)
        if len(actual_rets) < min_picks:
            continue

        port_ret = float(actual_rets.mean())
        portfolio_returns.append(port_ret)

        # Universe equal-weight average (all non-SPY stocks with valid target)
        all_rets = day_df[target_col].dropna()
        if len(all_rets) > 0:
            universe_rets.append(float(all_rets.mean()))

        # SPY for reference
        spy_ret = spy_returns.get(d, np.nan)
        if not math.isnan(spy_ret) if spy_ret is not None else False:
            spy_rets.append(float(spy_ret))

        # Decile hit rate: fraction of top picks in actual top 10% of the universe
        if len(all_rets) >= 10:
            threshold = float(all_rets.quantile(_TOP_DECILE))
            hit = float((actual_rets >= threshold).mean())
            decile_hits.append(hit)

        # IC: Spearman rank correlation between scores and actual returns
        # Uses all non-SPY stocks with valid scores and targets on this date
        ic_df = day_df[["score", target_col]].dropna()
        if len(ic_df) >= 10:
            ic = float(ic_df["score"].corr(ic_df[target_col], method="spearman"))
            if not math.isnan(ic):
                ics.append(ic)

    if not portfolio_returns:
        log.warning("evaluate.no_results", model=model_name)
        return {"model": model_name, "error": "no evaluation periods"}

    periods_per_year = _TRADING_DAYS_PER_YEAR / forward_days

    port_arr = np.array(portfolio_returns)
    univ_arr = np.array(universe_rets) if universe_rets else np.zeros(len(port_arr))
    spy_arr = np.array(spy_rets) if spy_rets else None
    ic_arr = np.array(ics) if ics else np.array([0.0])

    # Excess over universe (primary benchmark)
    n = min(len(port_arr), len(univ_arr))
    excess = port_arr[:n] - univ_arr[:n]

    sharpe = (
        float(excess.mean() / excess.std() * math.sqrt(periods_per_year))
        if excess.std() > 0 else 0.0
    )

    # ICIR: Sharpe of IC — comparable across different cadences and horizons
    icir = (
        float(ic_arr.mean() / ic_arr.std() * math.sqrt(periods_per_year))
        if len(ic_arr) > 1 and ic_arr.std() > 0 else 0.0
    )

    metrics = {
        "model": model_name,
        "eval_periods": len(portfolio_returns),
        "avg_log_ret":      round(float(port_arr.mean()), 5),
        "avg_universe_ret": round(float(univ_arr.mean()), 5) if len(univ_arr) else None,
        "avg_excess_ret":   round(float(excess.mean()), 5) if len(excess) else None,
        "avg_spy_ret":      round(float(spy_arr.mean()), 5) if spy_arr is not None else None,
        "decile_hit_rate":  round(float(np.mean(decile_hits)), 3) if decile_hits else None,
        "ic_mean":          round(float(ic_arr.mean()), 4),
        "icir":             round(icir, 3),
        "sharpe":           round(sharpe, 3),
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
    """
    Print a side-by-side comparison table.

    Columns:
      Periods     — number of eval periods in val set
      AvgRet      — mean return of top-N picks per period
      UnivRet     — equal-weight universe mean (the benchmark)
      Excess      — AvgRet - UnivRet (excess over the field)
      DclHit      — fraction of top-N picks in actual top decile
      IC          — mean Spearman IC (predicted rank vs actual return)
      ICIR        — IC / IC.std() * sqrt(periods/yr); cross-model comparable
      Sharpe      — annualized Sharpe of excess-over-universe

    IC and ICIR are the primary cross-model comparables. Sharpe and Excess
    are only directly comparable between models with the same forward_days cadence.
    """
    w = 100
    print("\n" + "=" * w)
    print(
        f"{'Model':<18} {'Periods':>7} {'AvgRet':>7} {'UnivRet':>8} "
        f"{'Excess':>7} {'DclHit':>7} {'IC':>7} {'ICIR':>7} {'Sharpe':>7}"
    )
    print("-" * w)
    for m in results:
        if "error" in m:
            print(f"{m['model']:<18}  ERROR: {m['error']}")
            continue
        print(
            f"{m['model']:<18} "
            f"{m['eval_periods']:>7d} "
            f"{m['avg_log_ret']:>7.4f} "
            f"{(m['avg_universe_ret'] or 0):>8.4f} "
            f"{(m['avg_excess_ret'] or 0):>7.4f} "
            f"{(m['decile_hit_rate'] or 0):>7.1%} "
            f"{m['ic_mean']:>7.4f} "
            f"{m['icir']:>7.3f} "
            f"{m['sharpe']:>7.3f}"
        )
    print("=" * w)
    print(
        "  IC/ICIR: cross-model comparable (scale-free, cadence-normalizing). "
        "Sharpe/Excess: comparable within same forward_days only.\n"
    )
