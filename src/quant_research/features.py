"""
src/quant_research/features.py

Builds the feature matrix from raw price data.
Input:  data.nosync/quant/raw_prices.parquet
Output: data.nosync/quant/features.parquet

Each row is one (ticker, date) observation with:
  - 15 features: log_price (absolute, intentional exception) + 14 percentage/log-return features
  - fwd_log_ret_20d: 20-day forward log return (the prediction target)
  - split: 'train' or 'val' (first 10yr = train, last 2yr = val)

Only rows with complete features AND a valid forward label are kept.
Minimum history required per row: 756 days lookback + 20 days forward.

Usage:
    python -m src.quant_research.features
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import structlog

log = structlog.get_logger()

_INPUT_FILE = Path("data.nosync/quant/raw_prices.parquet")
_OUTPUT_FILE = Path("data.nosync/quant/features.parquet")

FEATURE_COLS = [
    # Price (absolute — intentional exception; log-scaled to compress range)
    "log_price",
    # Price vs. moving averages (all percentages)
    "pct_sma10",
    "pct_sma50",
    "pct_sma200",
    # Price vs. highs/lows (all percentages)
    "pct_ath",
    "pct_time_since_ath",   # (days_since_ath / 1260) × 100 — % of 5yr window elapsed
    "pct_52w_low",
    # Momentum (log returns — scale-invariant)
    "log_ret_5d",
    "log_ret_20d",
    "log_ret_60d",
    "log_ret_126d",
    "log_ret_252d",
    "log_ret_756d",
    # Volatility (annualized — scale-invariant)
    "vol_20d",
    "vol_60d",
]

_ATH_CAP = 1260  # cap days_since_ath at 5 years


def _build_ticker_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Given a single-ticker DataFrame sorted by date with columns
    [date, open, high, low, close, volume], return a DataFrame
    of feature rows. Rows with any NaN are dropped.
    """
    close = df["close"].values
    dates = df["date"].values
    n = len(close)

    if n < 776:  # need 756 lookback + 20 forward
        return pd.DataFrame()

    log_close = np.log(close)

    # Rolling SMAs
    def sma(w: int) -> np.ndarray:
        return pd.Series(close).rolling(w).mean().values

    sma10 = sma(10)
    sma50 = sma(50)
    sma200 = sma(200)

    # Log returns at various lookbacks
    def log_ret(lag: int) -> np.ndarray:
        ret = np.full(n, np.nan)
        ret[lag:] = log_close[lag:] - log_close[:-lag]
        return ret

    lr5 = log_ret(5)
    lr20 = log_ret(20)
    lr60 = log_ret(60)
    lr126 = log_ret(126)
    lr252 = log_ret(252)
    lr756 = log_ret(756)

    # 20-day realized volatility (annualized)
    def vol(w: int) -> np.ndarray:
        daily_lr = np.diff(log_close, prepend=np.nan)
        return pd.Series(daily_lr).rolling(w).std().values * math.sqrt(252)

    vol20 = vol(20)
    vol60 = vol(60)

    # All-time high and % below ATH (expanding window)
    ath = pd.Series(close).expanding().max().values
    pct_ath = (close / ath - 1) * 100  # negative = below ATH

    # % of 5-year window elapsed since ATH (0% = just made ATH, 100% = 5+ years ago)
    # close[0] == ath[0] always (expanding max), so ath_indices always contains 0.
    ath_indices = np.where(close >= ath)[0]
    last_ath_pos = np.searchsorted(ath_indices, np.arange(n), side="right") - 1
    days_since_ath_raw = np.minimum(np.arange(n) - ath_indices[last_ath_pos], _ATH_CAP).astype(float)
    pct_time_since_ath = (days_since_ath_raw / _ATH_CAP) * 100

    # 52-week (252-day) low
    low_252 = pd.Series(close).rolling(252).min().values
    pct_52w_low = (close / low_252 - 1) * 100

    # Forward return targets
    fwd20 = np.full(n, np.nan)
    fwd20[:-20] = log_close[20:] - log_close[:-20]

    fwd5 = np.full(n, np.nan)
    fwd5[:-5] = log_close[5:] - log_close[:-5]

    rows = {
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
        "fwd_log_ret_20d": fwd20,
        "fwd_log_ret_5d": fwd5,
    }

    result = pd.DataFrame(rows)
    result = result.dropna()
    return result


def build_features() -> pd.DataFrame:
    if not _INPUT_FILE.exists():
        raise FileNotFoundError(f"Run download.py first: {_INPUT_FILE}")

    log.info("features.loading_prices", file=str(_INPUT_FILE))
    prices = pd.read_parquet(_INPUT_FILE)
    prices["date"] = pd.to_datetime(prices["date"])
    prices = prices.sort_values(["ticker", "date"]).reset_index(drop=True)

    tickers = prices["ticker"].unique()
    log.info("features.building", tickers=len(tickers))

    all_frames: list[pd.DataFrame] = []

    for i, ticker in enumerate(tickers):
        tdf = prices[prices["ticker"] == ticker].reset_index(drop=True)
        feat = _build_ticker_features(tdf)
        if len(feat) == 0:
            continue
        feat["ticker"] = ticker
        all_frames.append(feat)

        if (i + 1) % 100 == 0:
            log.info("features.progress", done=i + 1, total=len(tickers))

    if not all_frames:
        raise ValueError("No feature rows generated — check raw_prices.parquet")

    df = pd.concat(all_frames, ignore_index=True)

    # Drop rows with implausible returns — unadjusted corporate actions / yfinance artifacts.
    # Thresholds are generous enough to keep legitimate extreme years (e.g. NVDA 2023)
    # but catch clear data errors (5-day 28000% gains, etc.)
    ret_filters = {
        "log_ret_5d":   1.0,   # >172% in 5 days
        "log_ret_20d":  1.5,   # >348% in 20 days
        "log_ret_60d":  2.0,
        "log_ret_126d": 2.0,
        "log_ret_252d": 2.3,   # >897% annually
        "log_ret_756d": 2.8,
        "fwd_log_ret_20d": 1.5,
    }
    before = len(df)
    for col, cap in ret_filters.items():
        df = df[df[col].abs() <= cap]
    dropped = before - len(df)
    log.info("features.outliers_dropped", dropped=dropped, pct=f"{dropped/before*100:.2f}%")

    # Train/val split: last 2 years = validation
    cutoff = df["date"].max() - pd.Timedelta(days=int(2 * 365.25))
    df["split"] = "train"
    df.loc[df["date"] > cutoff, "split"] = "val"

    train_rows = (df["split"] == "train").sum()
    val_rows = (df["split"] == "val").sum()

    log.info(
        "features.done",
        total_rows=len(df),
        train_rows=train_rows,
        val_rows=val_rows,
        tickers=df["ticker"].nunique(),
    )

    _OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(_OUTPUT_FILE, index=False)
    log.info("features.saved", file=str(_OUTPUT_FILE))

    return df


if __name__ == "__main__":
    df = build_features()
    print(f"\nFeature matrix: {len(df):,} rows, {df['ticker'].nunique()} tickers")
    print(f"Train: {(df['split']=='train').sum():,}  Val: {(df['split']=='val').sum():,}")
    print(f"\nSample row:\n{df[FEATURE_COLS + ['fwd_log_ret_20d']].iloc[1000].to_string()}")
