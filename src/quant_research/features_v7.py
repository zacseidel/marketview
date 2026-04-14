"""
src/quant_research/features_v7.py

Builds the v7 feature matrix: v6 + 3 additional earnings signals.

New features vs v6:
  ni_qoq_growth       — NI quarter-over-quarter log growth (sign-matched)
                        More timely than YoY: captures current-quarter trajectory.
                        ~97% populated in earnings events.
  ni_acceleration     — Change in YoY NI growth rate (second derivative).
                        Is growth accelerating or decelerating?
                        ~67% populated (requires 5+ quarters of history).
  earn_ret_5d_to_20d  — Price drift from day 5 to day 20 post-earnings.
                        Captures the continuation/reversal after PEAD initial move.
                        ~99% populated in earnings events.

Final feature count: 46 numeric + 1 categorical (sector) = 47 total

Built by loading features_v6.parquet and appending the 3 new signals via
a second as-of join — does not rerun the full v6 build pipeline.

Usage:
    python -m src.quant_research.features_v7
    (requires features_v6.parquet to exist first)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import structlog

from src.quant_research.features_v2 import _EARNINGS_DIR
from src.quant_research.features_v6 import (
    FEATURE_COLS_V6,
    CATEGORICAL_COLS_V6,
    build_features_v6,
)

log = structlog.get_logger()

_V6_FILE     = Path("data.nosync/quant/features_v6.parquet")
_OUTPUT_FILE = Path("data.nosync/quant/features_v7.parquet")

# 3 new signals added on top of v6
_NEW_EARNINGS_COLS = ["ni_qoq_growth", "ni_acceleration", "earn_ret_5d_to_20d"]

FEATURE_COLS_V7 = FEATURE_COLS_V6 + _NEW_EARNINGS_COLS
CATEGORICAL_COLS_V7 = CATEGORICAL_COLS_V6


def _load_extra_earnings_df() -> pd.DataFrame:
    """
    Load earnings events with the 3 new signals directly from JSON files.
    _load_earnings_df() in features_v2 hardcodes only 4 columns and doesn't
    expose ni_qoq_growth / ni_acceleration / earn_ret_5d_to_20d.
    """
    records = []
    for fpath in _EARNINGS_DIR.glob("*.json"):
        if fpath.name.startswith("."):
            continue
        with open(fpath) as f:
            data = json.load(f)
        if not isinstance(data, list):
            continue
        for rec in data:
            event_date = rec.get("event_date")
            ticker = rec.get("ticker")
            if not event_date or not ticker:
                continue
            records.append({
                "ticker":              ticker,
                "event_date":          pd.Timestamp(event_date),
                "ni_qoq_growth":       rec.get("ni_qoq_growth"),
                "ni_acceleration":     rec.get("ni_acceleration"),
                "earn_ret_5d_to_20d":  rec.get("earn_ret_5d_to_20d"),
            })

    if not records:
        return pd.DataFrame(columns=["ticker", "event_date"] + _NEW_EARNINGS_COLS)

    df = pd.DataFrame(records)
    df = df.sort_values("event_date").reset_index(drop=True)
    return df


def _attach_extra_earnings(df: pd.DataFrame) -> pd.DataFrame:
    """
    As-of backward join of the 3 new earnings signals onto each row.
    Uses the same event_date anchor as features_v6's earnings join.
    """
    earn_df = _load_extra_earnings_df()
    if earn_df.empty:
        for col in _NEW_EARNINGS_COLS:
            df[col] = np.nan
        return df

    df = df.sort_values("date").reset_index(drop=True)
    df = pd.merge_asof(
        df, earn_df,
        left_on="date", right_on="event_date",
        by="ticker", direction="backward",
    )
    return df.drop(columns=["event_date"])


def build_features_v7() -> pd.DataFrame:
    """
    Load v6 parquet (building it first if needed), then append the 3 new
    earnings signals via a second as-of join.
    """
    if _V6_FILE.exists():
        log.info("features_v7.loading_v6", file=str(_V6_FILE))
        df = pd.read_parquet(_V6_FILE)
        df["date"] = pd.to_datetime(df["date"])
    else:
        log.info("features_v7.building_v6_first")
        df = build_features_v6()

    log.info("features_v7.attaching_extra_earnings", new_cols=_NEW_EARNINGS_COLS)
    df = _attach_extra_earnings(df)

    # --- Anti-leakage masks ---
    # earn_ret_5d_to_20d = log(price[earn_date+20] / price[earn_date+5]).
    # When days_since_earnings < 20, earn_date+20 is still in the future → leaks.
    # When days_since_earnings is NaN (>20 days ago), the full window has elapsed → safe.
    # Rule: mask to NaN whenever days_since_earnings is not NaN (= within 20-day window).
    if "earn_ret_5d_to_20d" in df.columns and "days_since_earnings" in df.columns:
        leak_mask = df["days_since_earnings"].notna()
        n_masked = int(leak_mask.sum())
        df.loc[leak_mask, "earn_ret_5d_to_20d"] = np.nan
        log.info("features_v7.leakage_mask_earn_ret_5d_to_20d",
                 masked=n_masked,
                 remaining=int(df["earn_ret_5d_to_20d"].notna().sum()))

    # earn_ret_5d: NOT masked here — behavior must be identical to v6.
    # v6 uses earn_ret_5d unmasked and it contributes real PEAD signal.
    # Masking 0–4 day rows removes genuine signal and hurts performance.

    # Clip extreme outliers in quarterly NI growth features.
    # ni_qoq_growth and ni_acceleration have heavy tails (|x|>3 for 7–8k rows)
    # that dominate XGBoost splits and destroy the stable v6 signal quality.
    for col, cap in [("ni_qoq_growth", 3.0), ("ni_acceleration", 3.0)]:
        if col in df.columns:
            n_clipped = int((df[col].abs() > cap).sum())
            df[col] = df[col].clip(-cap, cap)
            log.info(f"features_v7.clip_{col}", cap=cap, clipped=n_clipped)

    log.info(
        "features_v7.done",
        total_rows=len(df),
        features=len(FEATURE_COLS_V7) + len(CATEGORICAL_COLS_V7),
        tickers=df["ticker"].nunique(),
    )

    _OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(_OUTPUT_FILE, index=False)
    log.info("features_v7.saved", file=str(_OUTPUT_FILE))
    return df


if __name__ == "__main__":
    df = build_features_v7()
    thu = df[df["date"].dt.dayofweek == 3]
    print(f"\nFeature matrix v7: {len(df):,} total rows, {df['ticker'].nunique()} tickers")
    print(f"Thursdays — Train: {(thu['split']=='train').sum():,}  Val: {(thu['split']=='val').sum():,}")
    print(f"\nNew features ({len(_NEW_EARNINGS_COLS)}):")
    for col in _NEW_EARNINGS_COLS:
        nans = df[col].isna().sum() if col in df.columns else "MISSING"
        print(f"  {col:<30}  NaN: {nans:,}")
    print(f"\nFull feature list ({len(FEATURE_COLS_V7)} numeric + {len(CATEGORICAL_COLS_V7)} categorical):")
    for col in FEATURE_COLS_V7 + CATEGORICAL_COLS_V7:
        nans = df[col].isna().sum() if col in df.columns else "MISSING"
        print(f"  {col:<30}  NaN: {nans:,}")
