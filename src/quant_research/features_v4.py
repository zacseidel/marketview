"""
src/quant_research/features_v4.py

Builds the v4 feature matrix for the Thursday-weekly XGBoost model.

Key design changes from v3:
  - 5-day forward target (fwd_log_ret_5d) instead of 10-day
  - No log_price, no buyback/fundamental features
  - Added: slope + R² of log-price regressions (10/50/200d)
  - Added: dollar volume features (log_dollar_vol_20d, dollar_vol_rel_20d)
  - Added: earnings timing (days_since/until, capped at 20, else NaN)
  - Added: momentum_sharpe, trend_accel, pct_52w_high
  - Added: in_sp500 universe membership flag
  - Sector: 2 features (sector_ret_126d, stock_vs_sector_126d) replacing 4 old ones
  - No StandardScaler (tree model — scaling is a no-op)

Thursday filtering and cross-sectional rank target are applied in train_v4.py.
This file outputs ALL daily rows so the parquet can serve any future use.

Usage:
    python -m src.quant_research.features_v4
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import structlog

from src.quant_research.features_v2 import (
    _build_spy_lookup,
    _load_earnings_df,
    load_sector_info,
    _sic_to_broad_sector,
    _EARNINGS_DIR,
    _UNIVERSE_FILE,
    _ATH_CAP,
    _DAYS_TO_EARNINGS_CAP,
)

log = structlog.get_logger()

_INPUT_FILE = Path("data.nosync/quant/raw_prices.parquet")
_OUTPUT_FILE = Path("data.nosync/quant/features_v4.parquet")

FEATURE_COLS_V4 = [
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
    # Earnings timing (NaN when outside 20-day window — semantically correct)
    "days_since_earnings", "days_until_earnings",
    # Market state
    "spy_ret_20d", "spy_vol_20d", "spy_pct_above_sma200",
    # Sector
    "sector_ret_126d", "stock_vs_sector_126d",
    # Universe membership
    "in_sp500",
]
CATEGORICAL_COLS_V4 = ["sector"]


# ---------------------------------------------------------------------------
# Vectorized rolling linear regression
# ---------------------------------------------------------------------------

def _rolling_linreg(log_close: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Vectorized rolling linear regression of log_close over windows of length n.

    Returns:
        slopes: N-day log-return trend estimate (slope * n). Shape (T,), NaN for first n-1 rows.
        r2s:    R² of the fit [0, 1]. Shape (T,), NaN for first n-1 rows.

    The x-axis is centered (0-mean) for numerical stability.
    slope_n = (slope_raw * n) is directly comparable to log_ret_Nd — it is the
    regression-estimated N-day total log return, robust to endpoint noise.
    """
    T = len(log_close)
    if T < n:
        return np.full(T, np.nan), np.full(T, np.nan)

    # Centered x: [-half, ..., +half]
    x = np.arange(n, dtype=np.float64) - (n - 1) / 2.0
    ss_xx = float((x * x).sum())

    # Build rolling windows via stride tricks — zero-copy view
    shape = (T - n + 1, n)
    strides = (log_close.strides[0], log_close.strides[0])
    windows = np.lib.stride_tricks.as_strided(log_close, shape=shape, strides=strides)

    # For each window: center y, compute ss_xy and ss_yy
    y_mean = windows.mean(axis=1, keepdims=True)
    y_c = windows - y_mean                      # shape (T-n+1, n)

    ss_xy = (y_c * x).sum(axis=1)              # shape (T-n+1,)
    ss_yy = (y_c * y_c).sum(axis=1)            # shape (T-n+1,)

    raw_slope = ss_xy / ss_xx
    slope_n = raw_slope * n                     # N-day trend in log-return units

    # R² = ss_xy² / (ss_xx * ss_yy); flat lines (ss_yy≈0) get R²=1
    r2 = np.where(ss_yy > 1e-12, ss_xy * ss_xy / (ss_xx * ss_yy), 1.0)
    r2 = np.clip(r2, 0.0, 1.0)

    out_slope = np.full(T, np.nan)
    out_r2 = np.full(T, np.nan)
    out_slope[n - 1:] = slope_n
    out_r2[n - 1:] = r2
    return out_slope, out_r2


# ---------------------------------------------------------------------------
# Per-ticker base feature builder
# ---------------------------------------------------------------------------

def _build_v4_base_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute v4 technical + liquidity features for one ticker.
    Requires 766+ rows (756 lookback + 10 buffer for fwd_log_ret_5d).
    Does NOT drop rows where fwd_log_ret_5d is NaN — callers decide.
    """
    close = df["close"].values
    volume = df["volume"].values
    dates = df["date"].values
    n = len(close)

    if n < 766:
        return pd.DataFrame()

    log_close = np.log(close)

    def sma(w: int) -> np.ndarray:
        return pd.Series(close).rolling(w).mean().values

    def log_ret(lag: int) -> np.ndarray:
        ret = np.full(n, np.nan)
        ret[lag:] = log_close[lag:] - log_close[:-lag]
        return ret

    def vol_annualized(w: int) -> np.ndarray:
        daily_lr = np.diff(log_close, prepend=np.nan)
        return pd.Series(daily_lr).rolling(w).std().values * math.sqrt(252)

    # --- Moving averages ---
    sma10 = sma(10)
    sma50 = sma(50)
    sma200 = sma(200)

    # --- Log returns ---
    lr5 = log_ret(5)
    lr20 = log_ret(20)
    lr60 = log_ret(60)
    lr126 = log_ret(126)
    lr252 = log_ret(252)
    lr756 = log_ret(756)

    # --- Volatility ---
    vol20 = vol_annualized(20)
    vol60 = vol_annualized(60)
    vol252 = vol_annualized(252)  # used for momentum_sharpe only

    # --- Regression slopes + R² ---
    slope10, r2_10 = _rolling_linreg(log_close, 10)
    slope50, r2_50 = _rolling_linreg(log_close, 50)
    slope200, r2_200 = _rolling_linreg(log_close, 200)

    # --- ATH features ---
    ath = pd.Series(close).expanding().max().values
    pct_ath = (close / ath - 1) * 100

    ath_indices = np.where(close >= ath)[0]
    last_ath_pos = np.searchsorted(ath_indices, np.arange(n), side="right") - 1
    days_since_ath_raw = np.minimum(
        np.arange(n) - ath_indices[last_ath_pos], _ATH_CAP
    ).astype(float)
    pct_time_since_ath = (days_since_ath_raw / _ATH_CAP) * 100

    # --- 52-week high and low ---
    high_252 = pd.Series(close).rolling(252).max().values
    low_252 = pd.Series(close).rolling(252).min().values
    pct_52w_high = (close / high_252 - 1) * 100
    pct_52w_low = (close / low_252 - 1) * 100

    # --- Derived momentum features ---
    with np.errstate(divide="ignore", invalid="ignore"):
        momentum_sharpe = np.where(vol252 > 0, lr252 / vol252, np.nan)
    trend_accel = lr20 - lr60 / 3.0

    # --- Dollar volume features ---
    dollar_vol = close * volume
    avg_dv_20 = pd.Series(dollar_vol).rolling(20).mean().values
    with np.errstate(divide="ignore", invalid="ignore"):
        log_dollar_vol_20d = np.where(avg_dv_20 > 0, np.log(avg_dv_20), np.nan)
        dollar_vol_rel_20d = np.where(avg_dv_20 > 0, dollar_vol / avg_dv_20, np.nan)

    # --- 5-day forward return (target) ---
    fwd5 = np.full(n, np.nan)
    fwd5[:-5] = log_close[5:] - log_close[:-5]

    result = pd.DataFrame({
        "date": dates,
        "close": close,
        # Trend
        "pct_sma10":  (close / sma10 - 1) * 100,
        "pct_sma50":  (close / sma50 - 1) * 100,
        "pct_sma200": (close / sma200 - 1) * 100,
        "slope_10d":  slope10,
        "slope_50d":  slope50,
        "slope_200d": slope200,
        "r2_10d":     r2_10,
        "r2_50d":     r2_50,
        "r2_200d":    r2_200,
        # Momentum
        "log_ret_5d":   lr5,
        "log_ret_20d":  lr20,
        "log_ret_60d":  lr60,
        "log_ret_126d": lr126,
        "log_ret_252d": lr252,
        "log_ret_756d": lr756,
        "momentum_sharpe": momentum_sharpe,
        "trend_accel": trend_accel,
        "pct_52w_high": pct_52w_high,
        "pct_52w_low":  pct_52w_low,
        "pct_ath":      pct_ath,
        "pct_time_since_ath": pct_time_since_ath,
        # Volatility
        "vol_20d": vol20,
        "vol_60d": vol60,
        # Liquidity
        "log_dollar_vol_20d":  log_dollar_vol_20d,
        "dollar_vol_rel_20d":  dollar_vol_rel_20d,
        # Target
        "fwd_log_ret_5d": fwd5,
    })

    # Drop rows where any non-target feature is NaN (first ~756 rows per ticker)
    feature_cols = [c for c in result.columns if c not in ("fwd_log_ret_5d", "date", "close")]
    return result.dropna(subset=feature_cols).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Sector features
# ---------------------------------------------------------------------------

def _add_sector_features_v4(
    df: pd.DataFrame,
    sector_map: dict[str, str],
) -> pd.DataFrame:
    """
    Add sector (string) + sector_ret_126d + stock_vs_sector_126d.
    Requires log_ret_126d already in df.
    Sectors with fewer than 3 tickers on a given date get NaN sector features.
    """
    df = df.copy()
    df["sector"] = df["ticker"].map(sector_map).fillna("Other")

    counts = df.groupby(["date", "sector"])["log_ret_126d"].transform("count")
    sector_mean = df.groupby(["date", "sector"])["log_ret_126d"].transform("mean")

    df["sector_ret_126d"] = np.where(counts >= 3, sector_mean, np.nan)
    df["stock_vs_sector_126d"] = np.where(
        counts >= 3,
        df["log_ret_126d"] - sector_mean,
        np.nan,
    )
    return df


# ---------------------------------------------------------------------------
# Earnings timing features
# ---------------------------------------------------------------------------

def _build_earnings_date_df() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load all per-ticker earnings files and return two DataFrames:
      past_df:   [ticker, event_date] for all historical events (for days_since)
      future_df: [ticker, event_date] for all events (used as next-event lookup)
    Both sorted by event_date ascending.
    """
    past_records: list[dict] = []
    for fpath in sorted(_EARNINGS_DIR.glob("*.json")):
        if fpath.name.startswith(".") or fpath.name == "next_dates.json":
            continue
        ticker = fpath.stem
        try:
            with open(fpath) as f:
                events = json.load(f)
        except Exception:
            continue
        for e in events:
            ed = e.get("event_date")
            if ed:
                past_records.append({"ticker": ticker, "event_date": pd.Timestamp(ed)})

    if not past_records:
        empty = pd.DataFrame(columns=["ticker", "event_date"])
        return empty, empty

    all_df = (
        pd.DataFrame(past_records)
        .drop_duplicates()
        .sort_values(["ticker", "event_date"])
        .reset_index(drop=True)
    )
    return all_df, all_df.copy()


def _attach_earnings_timing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Attach days_since_earnings and days_until_earnings to df.
    Both are set to NaN when the nearest event is more than 20 days away.
    Uses merge_asof for vectorized as-of date joins.
    """
    past_df, future_df = _build_earnings_date_df()
    if past_df.empty:
        df["days_since_earnings"] = np.nan
        df["days_until_earnings"] = np.nan
        return df

    df = df.sort_values("date").reset_index(drop=True)
    past_df = past_df.sort_values("event_date").reset_index(drop=True)
    future_df = future_df.sort_values("event_date").reset_index(drop=True)

    # days_since: last event on or before current date
    merged_past = pd.merge_asof(
        df[["date", "ticker"]],
        past_df.rename(columns={"event_date": "last_event_date"}),
        left_on="date", right_on="last_event_date",
        by="ticker", direction="backward",
    )
    raw_since = (df["date"] - merged_past["last_event_date"]).dt.days
    df["days_since_earnings"] = np.where(
        raw_since.notna() & (raw_since <= 20), raw_since, np.nan
    )

    # days_until: next event strictly after current date
    merged_future = pd.merge_asof(
        df[["date", "ticker"]],
        future_df.rename(columns={"event_date": "next_event_date"}),
        left_on="date", right_on="next_event_date",
        by="ticker", direction="forward",
    )
    raw_until = (merged_future["next_event_date"] - df["date"]).dt.days
    df["days_until_earnings"] = np.where(
        raw_until.notna() & (raw_until <= 20), raw_until, np.nan
    )

    return df


# ---------------------------------------------------------------------------
# Universe membership
# ---------------------------------------------------------------------------

def _load_sp500_set() -> set[str]:
    """Return set of tickers in S&P 500 tier."""
    with open(_UNIVERSE_FILE) as f:
        constituents = json.load(f)
    return {t for t, v in constituents.items() if v.get("tier") == "sp500"}


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------

def build_features_v4() -> pd.DataFrame:
    if not _INPUT_FILE.exists():
        raise FileNotFoundError(f"Run download.py first: {_INPUT_FILE}")

    log.info("features_v4.loading_prices", file=str(_INPUT_FILE))
    prices = pd.read_parquet(_INPUT_FILE)
    prices["date"] = pd.to_datetime(prices["date"])
    prices = prices.sort_values(["ticker", "date"]).reset_index(drop=True)

    spy_df = _build_spy_lookup(prices)
    sector_map, _ = load_sector_info()
    sp500_set = _load_sp500_set()
    log.info("features_v4.setup",
             spy_rows=len(spy_df),
             sectors=len(set(sector_map.values())),
             sp500_tickers=len(sp500_set))

    tickers = list(prices["ticker"].unique())
    log.info("features_v4.building_ticker_features", tickers=len(tickers))

    all_frames: list[pd.DataFrame] = []
    for i, ticker in enumerate(tickers):
        tdf = prices[prices["ticker"] == ticker].sort_values("date").reset_index(drop=True)
        feat = _build_v4_base_features(tdf)
        if feat.empty:
            continue
        feat["ticker"] = ticker
        feat["in_sp500"] = 1 if ticker in sp500_set else 0
        all_frames.append(feat)
        if (i + 1) % 100 == 0:
            log.info("features_v4.ticker_progress", done=i + 1, total=len(tickers))

    if not all_frames:
        raise ValueError("No feature rows generated — check raw_prices.parquet")

    df = pd.concat(all_frames, ignore_index=True)
    log.info("features_v4.concat_done", total_rows=len(df))

    # Outlier cap on target before attaching (keeps NaN rows intact)
    target_rows = df["fwd_log_ret_5d"].notna()
    extreme = target_rows & (df["fwd_log_ret_5d"].abs() > 0.5)
    log.info("features_v4.outlier_cap", dropped=int(extreme.sum()))
    df.loc[extreme, "fwd_log_ret_5d"] = np.nan

    # Attach SPY market state
    df = df.merge(spy_df, on="date", how="left")

    # Attach earnings timing features
    log.info("features_v4.attaching_earnings_timing")
    df = _attach_earnings_timing(df)

    # Attach sector features (requires log_ret_126d)
    df = _add_sector_features_v4(df, sector_map)

    # Train/val split: last 2 years = val
    val_cutoff = df["date"].max() - pd.Timedelta(days=int(2 * 365.25))
    df["split"] = "train"
    df.loc[df["date"] > val_cutoff, "split"] = "val"

    train_rows = (df["split"] == "train").sum()
    val_rows = (df["split"] == "val").sum()
    thu_train = ((df["split"] == "train") & (df["date"].dt.dayofweek == 3)).sum()
    thu_val = ((df["split"] == "val") & (df["date"].dt.dayofweek == 3)).sum()

    log.info(
        "features_v4.done",
        total_rows=len(df),
        train_rows=train_rows,
        val_rows=val_rows,
        thursday_train_rows=thu_train,
        thursday_val_rows=thu_val,
        tickers=df["ticker"].nunique(),
    )

    _OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(_OUTPUT_FILE, index=False)
    log.info("features_v4.saved", file=str(_OUTPUT_FILE))

    return df


if __name__ == "__main__":
    df = build_features_v4()
    thu = df[df["date"].dt.dayofweek == 3]
    print(f"\nFeature matrix v4: {len(df):,} total rows, {df['ticker'].nunique()} tickers")
    print(f"Thursdays — Train: {(thu['split']=='train').sum():,}  Val: {(thu['split']=='val').sum():,}")
    print(f"\nFeatures ({len(FEATURE_COLS_V4)} numeric + {len(CATEGORICAL_COLS_V4)} categorical):")
    for col in FEATURE_COLS_V4 + CATEGORICAL_COLS_V4:
        nans = df[col].isna().sum() if col in df.columns else "MISSING"
        print(f"  {col:<30}  NaN: {nans:,}")
