"""
src/quant_research/features_v3.py

Builds the v3 feature matrix: v2 features + log_ret_756d = 28 numeric + sector.
Target: fwd_log_ret_10d (20-day, same as v1 for direct Sharpe comparison).
Trained on full 10+ years; val = last 2 years.

Requires ~800 days of price history for live inference (uses recent_prices.parquet
yfinance cache, same as quant.py/quant_v1).

Usage:
    python -m src.quant_research.features_v3
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import structlog

from src.quant_research.features_v2 import (
    _build_spy_lookup,
    _load_fundamentals_df,
    _load_earnings_df,
    load_sector_info,
    add_sector_features,
    _FUNDAMENTALS_DIR,
    _EARNINGS_DIR,
    _UNIVERSE_FILE,
    _DAYS_TO_EARNINGS_CAP,
    _ATH_CAP,
)

log = structlog.get_logger()

_INPUT_FILE = Path("data.nosync/quant/raw_prices.parquet")
_OUTPUT_FILE = Path("data.nosync/quant/features_v3.parquet")

FEATURE_COLS_V3 = [
    # Technical (15 — includes log_ret_756d, requires ~800-day price cache)
    "log_price", "pct_sma10", "pct_sma50", "pct_sma200",
    "pct_ath", "pct_time_since_ath", "pct_52w_low",
    "log_ret_5d", "log_ret_20d", "log_ret_60d", "log_ret_126d", "log_ret_252d", "log_ret_756d",
    "vol_20d", "vol_60d",
    # Repurchase (2)
    "buyback_pct_12m", "buyback_pct_1q",
    # Earnings (5)
    "eps_surprise_pct", "earn_ret_5d", "ni_yoy_growth", "rev_yoy_growth", "days_to_next_earnings",
    # Market state (3)
    "spy_ret_20d", "spy_vol_20d", "spy_pct_above_sma200",
    # Sector (4 numeric)
    "sector_ret_20d", "sector_vs_spy_20d", "sector_ret_rank", "sector_size",
]
CATEGORICAL_COLS_V3 = ["sector"]


def _build_v3_base_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute v3 technical features + fwd_log_ret_10d for one ticker.
    Requires 766 rows minimum (756 lookback + 10 forward).
    Does NOT drop rows where fwd_log_ret_10d is NaN — callers decide.
    """
    close = df["close"].values
    dates = df["date"].values
    n = len(close)

    if n < 766:  # 756 lookback + 10 forward
        return pd.DataFrame()

    log_close = np.log(close)

    def sma(w: int) -> np.ndarray:
        return pd.Series(close).rolling(w).mean().values

    def log_ret(lag: int) -> np.ndarray:
        ret = np.full(n, np.nan)
        ret[lag:] = log_close[lag:] - log_close[:-lag]
        return ret

    def vol(w: int) -> np.ndarray:
        daily_lr = np.diff(log_close, prepend=np.nan)
        return pd.Series(daily_lr).rolling(w).std().values * math.sqrt(252)

    sma10 = sma(10)
    sma50 = sma(50)
    sma200 = sma(200)

    lr5 = log_ret(5)
    lr20 = log_ret(20)
    lr60 = log_ret(60)
    lr126 = log_ret(126)
    lr252 = log_ret(252)
    lr756 = log_ret(756)

    vol20 = vol(20)
    vol60 = vol(60)

    ath = pd.Series(close).expanding().max().values
    pct_ath = (close / ath - 1) * 100

    ath_indices = np.where(close >= ath)[0]
    last_ath_pos = np.searchsorted(ath_indices, np.arange(n), side="right") - 1
    days_since_ath_raw = np.minimum(np.arange(n) - ath_indices[last_ath_pos], _ATH_CAP).astype(float)
    pct_time_since_ath = (days_since_ath_raw / _ATH_CAP) * 100

    low_252 = pd.Series(close).rolling(252).min().values
    pct_52w_low = (close / low_252 - 1) * 100

    fwd10 = np.full(n, np.nan)
    fwd10[:-10] = log_close[10:] - log_close[:-10]

    fwd5 = np.full(n, np.nan)
    fwd5[:-5] = log_close[5:] - log_close[:-5]

    result = pd.DataFrame({
        "date": dates,
        "close": close,
        "log_price": log_close,
        "pct_sma10": (close / sma10 - 1) * 100,
        "pct_sma50": (close / sma50 - 1) * 100,
        "pct_sma200": (close / sma200 - 1) * 100,
        "pct_ath": pct_ath,
        "pct_time_since_ath": pct_time_since_ath,
        "pct_52w_low": pct_52w_low,
        "log_ret_5d": lr5,
        "log_ret_20d": lr20,
        "log_ret_60d": lr60,
        "log_ret_126d": lr126,
        "log_ret_252d": lr252,
        "log_ret_756d": lr756,
        "vol_20d": vol20,
        "vol_60d": vol60,
        "fwd_log_ret_10d": fwd10,
        "fwd_log_ret_5d": fwd5,
    })

    feature_cols = [c for c in result.columns if c not in ("fwd_log_ret_10d", "fwd_log_ret_5d", "date", "close")]
    return result.dropna(subset=feature_cols).reset_index(drop=True)


def build_features_v3() -> pd.DataFrame:
    if not _INPUT_FILE.exists():
        raise FileNotFoundError(f"Run download.py first: {_INPUT_FILE}")

    log.info("features_v3.loading_prices", file=str(_INPUT_FILE))
    prices = pd.read_parquet(_INPUT_FILE)
    prices["date"] = pd.to_datetime(prices["date"])
    prices = prices.sort_values(["ticker", "date"]).reset_index(drop=True)

    spy_df = _build_spy_lookup(prices)
    sector_map, sector_size_map = load_sector_info()
    log.info("features_v3.setup", spy_rows=len(spy_df), sectors=len(sector_size_map))

    tickers = list(prices["ticker"].unique())
    log.info("features_v3.building_ticker_features", tickers=len(tickers))

    all_frames: list[pd.DataFrame] = []
    for i, ticker in enumerate(tickers):
        tdf = prices[prices["ticker"] == ticker].sort_values("date").reset_index(drop=True)
        feat = _build_v3_base_features(tdf)
        if feat.empty:
            continue
        feat["ticker"] = ticker
        all_frames.append(feat)
        if (i + 1) % 100 == 0:
            log.info("features_v3.ticker_progress", done=i + 1, total=len(tickers))

    if not all_frames:
        raise ValueError("No feature rows generated — check raw_prices.parquet")

    df = pd.concat(all_frames, ignore_index=True)
    log.info("features_v3.concat_done", total_rows=len(df))

    df = df.dropna(subset=["fwd_log_ret_10d"])
    before = len(df)
    df = df[df["fwd_log_ret_10d"].abs() <= 1.0]
    log.info("features_v3.outlier_cap", dropped=before - len(df))

    # Attach SPY market state
    df = df.merge(spy_df, on="date", how="left")

    # Attach fundamentals (as-of-date backward join)
    fund_df = _load_fundamentals_df()
    log.info("features_v3.fundamentals_loaded", tickers=fund_df["ticker"].nunique() if not fund_df.empty else 0)
    if not fund_df.empty:
        df = df.sort_values("date").reset_index(drop=True)
        fund_df = fund_df.sort_values("filing_date").reset_index(drop=True)
        df = pd.merge_asof(
            df, fund_df,
            left_on="date", right_on="filing_date",
            by="ticker", direction="backward",
        )
        df = df.drop(columns=["filing_date"])

    # Attach earnings features (as-of-date backward join)
    earn_df = _load_earnings_df()
    log.info("features_v3.earnings_loaded", tickers=earn_df["ticker"].nunique() if not earn_df.empty else 0)
    if not earn_df.empty:
        past_earn = (
            earn_df[["ticker", "event_date", "eps_surprise_pct", "earn_ret_5d",
                      "ni_yoy_growth", "rev_yoy_growth"]]
            .sort_values("event_date")
            .reset_index(drop=True)
        )
        df = df.sort_values("date").reset_index(drop=True)
        df = pd.merge_asof(
            df, past_earn,
            left_on="date", right_on="event_date",
            by="ticker", direction="backward",
        )
        df = df.drop(columns=["event_date"])

        next_earn = (
            earn_df[["ticker", "event_date"]].dropna()
            .rename(columns={"event_date": "next_earnings_date"})
            .sort_values("next_earnings_date")
            .reset_index(drop=True)
        )
        df = df.sort_values("date").reset_index(drop=True)
        df = pd.merge_asof(
            df, next_earn,
            left_on="date", right_on="next_earnings_date",
            by="ticker", direction="forward",
        )
        raw_days = (df["next_earnings_date"] - df["date"]).dt.days.clip(0, _DAYS_TO_EARNINGS_CAP)
        df["days_to_next_earnings"] = (raw_days / _DAYS_TO_EARNINGS_CAP) * 100
        df = df.drop(columns=["next_earnings_date"])

    # Sector features
    df = add_sector_features(df, sector_map, sector_size_map)

    # Train/val split: last 2 years = val (matches v1 for direct Sharpe comparison)
    val_cutoff = df["date"].max() - pd.Timedelta(days=int(2 * 365.25))
    df["split"] = "train"
    df.loc[df["date"] > val_cutoff, "split"] = "val"

    train_rows = (df["split"] == "train").sum()
    val_rows = (df["split"] == "val").sum()
    log.info(
        "features_v3.done",
        total_rows=len(df),
        train_rows=train_rows,
        val_rows=val_rows,
        tickers=df["ticker"].nunique(),
    )

    _OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(_OUTPUT_FILE, index=False)
    log.info("features_v3.saved", file=str(_OUTPUT_FILE))

    return df


if __name__ == "__main__":
    df = build_features_v3()
    print(f"\nFeature matrix v3: {len(df):,} rows, {df['ticker'].nunique()} tickers")
    print(f"Train: {(df['split']=='train').sum():,}  Val: {(df['split']=='val').sum():,}")
