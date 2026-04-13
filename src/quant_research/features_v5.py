"""
src/quant_research/features_v5.py

Builds the v5 feature matrix: full union of v3 + v4b features (46 total).

v5 = v4b (v4 tech + earnings fundamentals) + v3-only additions:
  log_price, buyback_pct_12m/1q, days_to_next_earnings,
  sector_ret_20d, sector_vs_spy_20d, sector_ret_rank, sector_size

Built by inner-joining the existing v3 and v4 parquets on (ticker, date).
No need to rerun the full price pipeline — depends on both parquets existing.

Usage:
    python -m src.quant_research.features_v5
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import structlog

log = structlog.get_logger()

_V3_FILE = Path("data.nosync/quant/features_v3.parquet")
_V4_FILE = Path("data.nosync/quant/features_v4.parquet")
_OUTPUT_FILE = Path("data.nosync/quant/features_v5.parquet")

# Columns present in v3 but not v4 — added to v4b base to form v5
_EARNINGS_FUNDAMENTAL_COLS = [
    "eps_surprise_pct", "earn_ret_5d", "ni_yoy_growth", "rev_yoy_growth",
]
_V3_ONLY_COLS = [
    "log_price",
    "buyback_pct_12m", "buyback_pct_1q",
    "days_to_next_earnings",
    "sector_ret_20d", "sector_vs_spy_20d", "sector_ret_rank", "sector_size",
]

FEATURE_COLS_V5 = [
    # Trend (SMA + regression slope + R²)
    "pct_sma10", "pct_sma50", "pct_sma200",
    "slope_10d", "slope_50d", "slope_200d",
    "r2_10d", "r2_50d", "r2_200d",
    # Momentum
    "log_ret_5d", "log_ret_20d", "log_ret_60d", "log_ret_126d", "log_ret_252d", "log_ret_756d",
    "momentum_sharpe", "trend_accel",
    "pct_52w_high", "pct_52w_low", "pct_ath", "pct_time_since_ath",
    # Volatility
    "vol_20d", "vol_60d",
    # Liquidity
    "log_dollar_vol_20d", "dollar_vol_rel_20d",
    # Earnings timing v4 (NaN outside 20-day window)
    "days_since_earnings", "days_until_earnings",
    # Market state
    "spy_ret_20d", "spy_vol_20d", "spy_pct_above_sma200",
    # Sector v4 (126d lookback)
    "sector_ret_126d", "stock_vs_sector_126d",
    # Universe membership
    "in_sp500",
    # Earnings fundamentals
    "eps_surprise_pct", "earn_ret_5d", "ni_yoy_growth", "rev_yoy_growth",
    # Price level
    "log_price",
    # Buyback
    "buyback_pct_12m", "buyback_pct_1q",
    # Earnings timing v3 (0–100 scale, 180-day cap)
    "days_to_next_earnings",
    # Sector v3 (20d lookback)
    "sector_ret_20d", "sector_vs_spy_20d", "sector_ret_rank", "sector_size",
]
CATEGORICAL_COLS_V5 = ["sector"]


def build_features_v5() -> pd.DataFrame:
    """
    Inner-join features_v3 and features_v4 parquets on (ticker, date).
    Keeps all v4 columns plus v3-only additions. Recomputes train/val split
    from the merged date range to stay consistent.
    """
    for path, name in [(_V3_FILE, "features_v3"), (_V4_FILE, "features_v4")]:
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. Rebuild with:\n"
                f"  python -m src.quant_research.{name}"
            )

    log.info("features_v5.loading", v3=str(_V3_FILE), v4=str(_V4_FILE))
    df4 = pd.read_parquet(_V4_FILE)
    extra_cols = _EARNINGS_FUNDAMENTAL_COLS + _V3_ONLY_COLS
    df3 = pd.read_parquet(
        _V3_FILE,
        columns=["ticker", "date"] + extra_cols,
    )

    df4["date"] = pd.to_datetime(df4["date"])
    df3["date"] = pd.to_datetime(df3["date"])

    log.info("features_v5.merging",
             v4_rows=len(df4), v3_rows=len(df3))
    df = df4.merge(df3, on=["ticker", "date"], how="inner")

    # Recompute split from merged date range
    val_cutoff = df["date"].max() - pd.Timedelta(days=int(2 * 365.25))
    df["split"] = "train"
    df.loc[df["date"] > val_cutoff, "split"] = "val"

    train_rows = (df["split"] == "train").sum()
    val_rows = (df["split"] == "val").sum()
    thu_train = ((df["split"] == "train") & (df["date"].dt.dayofweek == 3)).sum()
    thu_val = ((df["split"] == "val") & (df["date"].dt.dayofweek == 3)).sum()

    log.info(
        "features_v5.done",
        total_rows=len(df),
        train_rows=train_rows,
        val_rows=val_rows,
        thursday_train=thu_train,
        thursday_val=thu_val,
        tickers=df["ticker"].nunique(),
        features=len(FEATURE_COLS_V5) + len(CATEGORICAL_COLS_V5),
    )

    _OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(_OUTPUT_FILE, index=False)
    log.info("features_v5.saved", file=str(_OUTPUT_FILE))
    return df


if __name__ == "__main__":
    df = build_features_v5()
    thu = df[df["date"].dt.dayofweek == 3]
    print(f"\nFeature matrix v5: {len(df):,} total rows, {df['ticker'].nunique()} tickers")
    print(f"Thursdays — Train: {(thu['split']=='train').sum():,}  Val: {(thu['split']=='val').sum():,}")
    print(f"\nFeatures ({len(FEATURE_COLS_V5)} numeric + {len(CATEGORICAL_COLS_V5)} categorical):")
    for col in FEATURE_COLS_V5 + CATEGORICAL_COLS_V5:
        nans = df[col].isna().sum() if col in df.columns else "MISSING"
        print(f"  {col:<30}  NaN: {nans:,}")
