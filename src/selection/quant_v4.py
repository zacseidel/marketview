"""
src/selection/quant_v4.py

Quantitative factor model v4 — live inference shim.

XGBoost model trained on Thursday-weekly data with a cross-sectional rank-percentile
target (5-day forward return horizon). Predicts the relative rank of each stock
within its Thursday cross-section.

Runs ONLY on the Friday cycle (Thursday EOD data). Returns [] on Tuesday runs.

Usage:
    python -m src.selection.quant_v4
"""

from __future__ import annotations

import json
import math
import pickle
from datetime import date as _date
from pathlib import Path

import numpy as np
import pandas as pd
import structlog

from src.collection.earnings import load_next_dates
from src.selection.base import DataAccessLayer, HoldingRecord, SelectionModel
from src.selection.quant import _load_recent_prices, _write_history_gaps
from src.quant_research.features_v4 import (
    FEATURE_COLS_V4,
    CATEGORICAL_COLS_V4,
    _build_v4_base_features,
    _load_sp500_set,
    _sic_to_broad_sector,
)
from src.quant_research.features_v2 import load_sector_info

log = structlog.get_logger()

_ARTIFACTS_DIR = Path("data.nosync/quant/artifacts/gbm_v4")
_EARNINGS_DIR = Path("data.nosync/earnings")
_UNIVERSE_FILE = Path("data.nosync/universe/constituents.json")

_MAX_HOLDINGS = 5
_MIN_PREDICTED_RANK = 0.5     # only consider stocks predicted above median rank
_DAYS_SINCE_CAP = 20
_DAYS_UNTIL_CAP = 20


# ---------------------------------------------------------------------------
# Artifact helpers
# ---------------------------------------------------------------------------

def _artifacts_exist() -> bool:
    return all((_ARTIFACTS_DIR / n).exists()
               for n in ("model.pkl", "sector_categories.json"))


def _load_artifacts() -> tuple:
    """Returns (model, sector_categories_list)."""
    with open(_ARTIFACTS_DIR / "model.pkl", "rb") as f:
        model = pickle.load(f)
    with open(_ARTIFACTS_DIR / "sector_categories.json") as f:
        sector_categories = json.load(f)
    return model, sector_categories


# ---------------------------------------------------------------------------
# SPY market state
# ---------------------------------------------------------------------------

def _compute_spy_features(prices: pd.DataFrame) -> dict[str, float | None]:
    spy = prices[prices["ticker"] == "SPY"].sort_values("date").reset_index(drop=True)
    if len(spy) < 20:
        return {"spy_ret_20d": None, "spy_vol_20d": None, "spy_pct_above_sma200": None}

    close = spy["close"].values
    log_close = np.log(close)

    spy_ret_20d = float(log_close[-1] - log_close[-21]) if len(log_close) > 20 else None
    daily_lr = np.diff(log_close)
    spy_vol_20d = float(np.std(daily_lr[-20:]) * math.sqrt(252)) if len(daily_lr) >= 20 else None
    sma200 = float(np.mean(close[-200:])) if len(close) >= 200 else None
    spy_pct_above_sma200 = float((close[-1] / sma200 - 1) * 100) if sma200 else None

    return {
        "spy_ret_20d": spy_ret_20d,
        "spy_vol_20d": spy_vol_20d,
        "spy_pct_above_sma200": spy_pct_above_sma200,
    }


# ---------------------------------------------------------------------------
# Earnings timing features
# ---------------------------------------------------------------------------

def _load_all_earnings() -> dict[str, list]:
    """Load all earnings files into memory once. Returns {ticker: records}."""
    result: dict[str, list] = {}
    for path in _EARNINGS_DIR.glob("*.json"):
        if path.name.startswith(".") or path.name == "next_dates.json":
            continue
        try:
            with open(path) as f:
                result[path.stem] = json.load(f)
        except Exception:
            pass
    return result


def _compute_earnings_timing(
    ticker: str,
    as_of: _date,
    earnings_cache: dict[str, list],
    next_dates: dict[str, str],
) -> dict[str, float | None]:
    """
    Compute days_since_earnings and days_until_earnings.
    Both are None when the nearest event exceeds the 20-day cap.
    """
    base: dict[str, float | None] = {
        "days_since_earnings": None,
        "days_until_earnings": None,
    }
    as_of_str = as_of.isoformat()
    records = earnings_cache.get(ticker, [])

    # days_since: most recent past event
    past = [r for r in records if r.get("event_date") and r["event_date"] <= as_of_str]
    if past:
        last_date = _date.fromisoformat(max(r["event_date"] for r in past))
        days = (as_of - last_date).days
        if days <= _DAYS_SINCE_CAP:
            base["days_since_earnings"] = float(days)

    # days_until: next future event (from records or next_dates calendar)
    future = [r for r in records if r.get("event_date") and r["event_date"] > as_of_str]
    if ticker in next_dates:
        try:
            cal_date = _date.fromisoformat(next_dates[ticker])
            days = (cal_date - as_of).days
            if 0 <= days <= _DAYS_UNTIL_CAP:
                base["days_until_earnings"] = float(days)
        except ValueError:
            pass
    elif future:
        next_date = _date.fromisoformat(min(r["event_date"] for r in future))
        days = (next_date - as_of).days
        if 0 <= days <= _DAYS_UNTIL_CAP:
            base["days_until_earnings"] = float(days)

    return base


# ---------------------------------------------------------------------------
# Sector features
# ---------------------------------------------------------------------------

def _compute_sector_features(
    feat_df: pd.DataFrame,
    sector_map: dict[str, str],
) -> pd.DataFrame:
    """Add sector, sector_ret_126d, stock_vs_sector_126d."""
    feat_df = feat_df.copy()
    feat_df["sector"] = feat_df["ticker"].map(sector_map).fillna("Other")

    counts = feat_df.groupby("sector")["log_ret_126d"].transform("count")
    sector_mean = feat_df.groupby("sector")["log_ret_126d"].transform("mean")

    feat_df["sector_ret_126d"] = np.where(counts >= 3, sector_mean, np.nan)
    feat_df["stock_vs_sector_126d"] = np.where(
        counts >= 3,
        feat_df["log_ret_126d"] - sector_mean,
        np.nan,
    )
    return feat_df


# ---------------------------------------------------------------------------
# Feature computation (current snapshot)
# ---------------------------------------------------------------------------

def _compute_current_features_v4(
    prices: pd.DataFrame,
    sp500_set: set[str],
    sector_map: dict[str, str],
) -> pd.DataFrame:
    as_of = prices["date"].max().date()
    spy_features = _compute_spy_features(prices)
    earnings_cache = _load_all_earnings()
    next_dates = load_next_dates()
    log.debug("quant_v4.caches_loaded",
              earnings=len(earnings_cache), next_dates=len(next_dates))

    rows: list[dict] = []
    for ticker, tdf in prices.groupby("ticker"):
        if ticker == "SPY":
            continue
        tdf = tdf.sort_values("date").reset_index(drop=True)
        feat = _build_v4_base_features(tdf)
        if feat.empty:
            continue

        last = feat.iloc[-1]
        technical_cols = [c for c in FEATURE_COLS_V4
                          if c in feat.columns
                          and c not in ("spy_ret_20d", "spy_vol_20d", "spy_pct_above_sma200",
                                        "sector_ret_126d", "stock_vs_sector_126d",
                                        "days_since_earnings", "days_until_earnings",
                                        "in_sp500")]
        row: dict = {"ticker": ticker}
        row.update({col: last[col] for col in technical_cols})
        row.update(_compute_earnings_timing(str(ticker), as_of, earnings_cache, next_dates))
        row.update(spy_features)
        row["in_sp500"] = 1 if ticker in sp500_set else 0
        rows.append(row)

    if not rows:
        return pd.DataFrame(columns=["ticker"] + FEATURE_COLS_V4 + CATEGORICAL_COLS_V4)

    feat_df = pd.DataFrame(rows)
    feat_df = _compute_sector_features(feat_df, sector_map)
    return feat_df


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _score_with_gbm_v4(
    feat_df: pd.DataFrame,
    model,
    sector_categories: list[str],
) -> np.ndarray:
    df = feat_df.copy()
    cat_type = pd.CategoricalDtype(categories=sector_categories, ordered=False)
    df["sector"] = df["sector"].astype(cat_type)
    all_cols = FEATURE_COLS_V4 + CATEGORICAL_COLS_V4
    X = df.reindex(columns=all_cols)
    return model.predict(X)


# ---------------------------------------------------------------------------
# SelectionModel
# ---------------------------------------------------------------------------

class QuantModelV4(SelectionModel):
    def run(self, config: dict, dal: DataAccessLayer) -> list[HoldingRecord]:
        eval_date_str = config.get("eval_date", _date.today().isoformat())
        max_holdings = config.get("max_holdings", _MAX_HOLDINGS)

        # Friday-only guard: this model uses Thursday EOD data, run Friday mornings
        eval_date_obj = _date.fromisoformat(eval_date_str)
        if eval_date_obj.weekday() != 4:  # 4 = Friday
            log.info("quant_v4.skip_non_friday", eval_date=eval_date_str)
            return []

        if not _artifacts_exist():
            log.warning("quant_v4.no_artifacts",
                        msg="Run src.quant_research.train_v4 first")
            return []

        model, sector_categories = _load_artifacts()

        with open(_UNIVERSE_FILE) as f:
            constituents = json.load(f)
        tickers = [v["ticker"] for v in constituents.values() if v.get("status") == "active"]

        prices = _load_recent_prices(list(set(tickers + ["SPY"])))
        if prices.empty:
            return []

        _write_history_gaps(tickers, prices)

        sp500_set = _load_sp500_set()
        sector_map, _ = load_sector_info()

        feat_df = _compute_current_features_v4(prices, sp500_set, sector_map)
        if feat_df.empty:
            log.warning("quant_v4.no_features")
            return []

        log.info("quant_v4.scoring", tickers=len(feat_df))
        predicted = _score_with_gbm_v4(feat_df, model, sector_categories)
        feat_df = feat_df.copy()
        feat_df["predicted_rank"] = predicted

        # Only consider stocks predicted above median rank
        feat_df = feat_df[feat_df["predicted_rank"] > _MIN_PREDICTED_RANK]
        if feat_df.empty:
            log.warning("quant_v4.no_positive_signals")
            return []

        top = feat_df.nlargest(max_holdings, "predicted_rank").reset_index(drop=True)
        max_rank = top["predicted_rank"].max()
        min_rank = top["predicted_rank"].min()
        rank_range = max_rank - min_rank if max_rank != min_rank else 1.0

        holdings: list[HoldingRecord] = []
        for _, row in top.iterrows():
            pred_rank = float(row["predicted_rank"])
            conviction = round((pred_rank - min_rank) / rank_range, 3)

            signals: list[str] = [f"predicted rank pct {pred_rank:.3f}"]
            for label, col in [
                ("days_since_earn", "days_since_earnings"),
                ("days_until_earn", "days_until_earnings"),
                ("spy_20d", "spy_ret_20d"),
                ("sector_126d", "sector_ret_126d"),
                ("vs_sector_126d", "stock_vs_sector_126d"),
            ]:
                val = row.get(col)
                if val is not None and not (isinstance(val, float) and math.isnan(val)):
                    signals.append(f"{label}={float(val):+.3f}")

            metadata: dict = {}
            for col in FEATURE_COLS_V4:
                v = row.get(col)
                if v is not None and not (isinstance(v, float) and math.isnan(v)):
                    metadata[col] = round(float(v), 6)
            metadata["quant_model"] = "gbm_v4"
            metadata["predicted_rank"] = round(pred_rank, 6)
            sector_val = row.get("sector")
            if sector_val:
                metadata["sector"] = str(sector_val)

            holdings.append(HoldingRecord(
                model="quant_v4",
                eval_date=eval_date_str,
                ticker=str(row["ticker"]),
                conviction=conviction,
                rationale=f"Quant v4 (XGB, 5d rank): {'; '.join(signals)}",
                metadata=metadata,
            ))

        log.info("quant_v4.done", selected=len(holdings))
        return holdings


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    dal = DataAccessLayer()
    model_inst = QuantModelV4()
    today = _date.today()
    holdings = model_inst.run({"eval_date": today.isoformat()}, dal)

    if not holdings:
        day_name = today.strftime("%A")
        print(f"\nNo picks returned (today is {day_name} — model runs Fridays only).")
    else:
        print(f"\nTop picks (quant_v4):")
        for h in holdings:
            print(f"  {h.ticker:<6}  conviction={h.conviction:.3f}  {h.rationale}")
