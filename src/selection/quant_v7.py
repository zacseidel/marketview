"""
src/selection/quant_v7.py

Quantitative factor model v7 — live inference shim.

v7 = v6 + 3 additional earnings signals (46 numeric + sector = 47 total):
  ni_qoq_growth       — NI quarter-over-quarter growth
  ni_acceleration     — YoY NI growth acceleration (second derivative)
  earn_ret_5d_to_20d  — Post-PEAD drift: price move from day 5 to day 20

Runs ONLY on the Friday cycle. Returns [] on Tuesday runs.

Usage:
    python -m src.selection.quant_v7
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

from src.selection.base import DataAccessLayer, HoldingRecord, SelectionModel
from src.selection.quant import _load_recent_prices, _write_history_gaps
from src.selection.quant_v6 import (
    _artifacts_exist as _v6_artifacts_exist,
    _compute_current_features_v6,
    _load_all_earnings,
)
from src.quant_research.features_v4 import _load_sp500_set
from src.quant_research.features_v2 import load_sector_info
from src.quant_research.features_v7 import FEATURE_COLS_V7, CATEGORICAL_COLS_V7

log = structlog.get_logger()

_ARTIFACTS_DIR = Path("data.nosync/quant/artifacts/gbm_v7")
_UNIVERSE_FILE = Path("data.nosync/universe/constituents.json")
_MAX_HOLDINGS  = 5

_EXTRA_EARNINGS_FIELDS = ["ni_qoq_growth", "ni_acceleration", "earn_ret_5d_to_20d"]


def _artifacts_exist() -> bool:
    return all((_ARTIFACTS_DIR / n).exists()
               for n in ("model.pkl", "sector_categories.json"))


def _load_artifacts() -> tuple:
    with open(_ARTIFACTS_DIR / "model.pkl", "rb") as f:
        model = pickle.load(f)
    with open(_ARTIFACTS_DIR / "sector_categories.json") as f:
        sector_categories = json.load(f)
    return model, sector_categories


def _attach_extra_earnings(
    feat_df: pd.DataFrame,
    as_of: _date,
    earnings_cache: dict[str, list],
) -> pd.DataFrame:
    """
    Add ni_qoq_growth, ni_acceleration, earn_ret_5d_to_20d from the most
    recent past earnings event per ticker, with leakage masks applied.

    Leakage masks:
      earn_ret_5d_to_20d: NaN when days_since_earnings < 20 (earn_date+20 still future)
      earn_ret_5d:        NaN when days_since_earnings < 5  (earn_date+5 still future)
    Both are safe when days_since_earnings is NaN (>20 days ago — full window elapsed).
    """
    as_of_str = as_of.isoformat()
    rows = feat_df.copy()

    for field in _EXTRA_EARNINGS_FIELDS:
        rows[field] = np.nan

    for idx, row in rows.iterrows():
        ticker = str(row["ticker"])
        records = earnings_cache.get(ticker, [])
        past = [r for r in records if r.get("event_date") and r["event_date"] <= as_of_str]
        if not past:
            continue
        latest = max(past, key=lambda r: r["event_date"])
        for field in _EXTRA_EARNINGS_FIELDS:
            v = latest.get(field)
            if v is not None:
                try:
                    rows.at[idx, field] = float(v)
                except (TypeError, ValueError):
                    pass

    # Anti-leakage: mask earn_ret_5d_to_20d when within 20-day earnings window.
    # earn_date+20 is still in the future when days_since_earnings < 20 → leaks.
    # NaN days_since_earnings means >20 days ago → full window elapsed → safe.
    days_since = rows.get("days_since_earnings")
    if days_since is not None and "earn_ret_5d_to_20d" in rows.columns:
        rows.loc[rows["days_since_earnings"].notna(), "earn_ret_5d_to_20d"] = np.nan

    # earn_ret_5d: NOT masked — behavior must be identical to v6 (unmasked).
    # Masking 0–4 day rows removes genuine PEAD signal that v6 captures.

    # Clip extreme outliers in quarterly NI growth features.
    for col, cap in [("ni_qoq_growth", 3.0), ("ni_acceleration", 3.0)]:
        if col in rows.columns:
            rows[col] = rows[col].clip(-cap, cap)

    return rows


class QuantModelV7(SelectionModel):
    def run(self, config: dict, dal: DataAccessLayer) -> list[HoldingRecord]:
        eval_date_str = config.get("eval_date", _date.today().isoformat())
        max_holdings = config.get("max_holdings", _MAX_HOLDINGS)

        # Tuesday/Friday guard
        eval_date_obj = _date.fromisoformat(eval_date_str)
        if eval_date_obj.weekday() not in (1, 4):  # 1=Tue, 4=Fri
            log.info("quant_v7.skip_non_cycle_day", eval_date=eval_date_str)
            return []

        if not _artifacts_exist():
            log.warning("quant_v7.no_artifacts",
                        msg="Run src.quant_research.train_v7 first")
            return []

        model, sector_categories = _load_artifacts()

        with open(_UNIVERSE_FILE) as f:
            constituents = json.load(f)
        tickers = [t for t, v in constituents.items() if v.get("status") == "active"]

        prices = _load_recent_prices(list(set(tickers + ["SPY"])))
        if prices.empty:
            return []

        _write_history_gaps(tickers, prices)

        sp500_set = _load_sp500_set()
        sector_map, _ = load_sector_info()

        # Build v6 feature snapshot then extend with the 3 extra earnings signals
        feat_df = _compute_current_features_v6(prices, sp500_set, sector_map)
        if feat_df.empty:
            log.warning("quant_v7.no_features")
            return []

        as_of = prices["date"].max().date()
        earnings_cache = _load_all_earnings()
        feat_df = _attach_extra_earnings(feat_df, as_of, earnings_cache)

        # Score
        import xgboost as xgb
        df_score = feat_df.copy()

        # Cast all numeric features to float64 — some (e.g. days_since_earnings)
        # may land as object dtype when the column is all-NaN.
        for col in FEATURE_COLS_V7:
            df_score[col] = pd.to_numeric(df_score[col], errors="coerce").astype("float64")

        # XGBoost 2.x requires pd.Categorical (not CategoricalDtype) for enable_categorical.
        df_score["sector"] = pd.Categorical(df_score["sector"], categories=sector_categories)

        all_cols = FEATURE_COLS_V7 + CATEGORICAL_COLS_V7
        dmatrix = xgb.DMatrix(df_score.reindex(columns=all_cols), enable_categorical=True)
        predicted = model.get_booster().predict(dmatrix)
        feat_df = feat_df.copy()
        feat_df["predicted_score"] = predicted

        log.info("quant_v7.scoring", tickers=len(feat_df),
                 above_zero=int((predicted > 0).sum()))

        top = feat_df.nlargest(max_holdings, "predicted_score").reset_index(drop=True)
        max_score = top["predicted_score"].max()
        min_score = top["predicted_score"].min()
        score_range = max_score - min_score if max_score != min_score else 1.0

        holdings: list[HoldingRecord] = []
        for _, row in top.iterrows():
            pred = float(row["predicted_score"])
            conviction = round((pred - min_score) / score_range, 3)

            signals: list[str] = [f"predicted score {pred:.4f}"]
            for label, col in [
                ("eps_surprise", "eps_surprise_pct"),
                ("earn_ret_5d", "earn_ret_5d"),
                ("earn_drift_5_20", "earn_ret_5d_to_20d"),
                ("ni_yoy", "ni_yoy_growth"),
                ("ni_accel", "ni_acceleration"),
                ("breadth", "mkt_breadth_sma200"),
                ("sector_126d", "sector_ret_126d"),
            ]:
                v = row.get(col)
                if v is not None and not (isinstance(v, float) and math.isnan(v)):
                    signals.append(f"{label}={float(v):+.3f}")

            metadata: dict = {}
            for col in FEATURE_COLS_V7:
                v = row.get(col)
                if v is not None and not (isinstance(v, float) and math.isnan(v)):
                    metadata[col] = round(float(v), 6)
            metadata["quant_model"] = "gbm_v7"
            metadata["predicted_score"] = round(pred, 6)
            sector_val = row.get("sector")
            if sector_val:
                metadata["sector"] = str(sector_val)

            holdings.append(HoldingRecord(
                model="quant_v7",
                eval_date=eval_date_str,
                ticker=str(row["ticker"]),
                conviction=conviction,
                rationale=f"Quant v7 (XGB, 5d raw): {'; '.join(signals)}",
                metadata=metadata,
            ))

        log.info("quant_v7.done", selected=len(holdings))
        return holdings


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    from src.selection.base import DataAccessLayer
    dal = DataAccessLayer()
    model_inst = QuantModelV7()
    today = _date.today()
    holdings = model_inst.run({"eval_date": today.isoformat()}, dal)

    if not holdings:
        day_name = today.strftime("%A")
        print(f"\nNo picks returned (today is {day_name} — model runs Fridays only).")
    else:
        print(f"\nTop picks (quant_v7):")
        for h in holdings:
            print(f"  {h.ticker:<6}  conviction={h.conviction:.3f}  {h.rationale}")
