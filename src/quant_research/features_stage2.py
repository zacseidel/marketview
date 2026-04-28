"""
src/quant_research/features_stage2.py

Stage 2 trend template feature engineering — Minervini-style conditions.

Builds a per-ticker, per-date feature matrix capturing:
  - Trend alignment (price vs SMA150/200, SMA150 vs SMA200)
  - SMA200 slope (is the 200-day trend turning up?)
  - Higher-high / higher-low trend structure (trailing 13 five-bar windows)
  - Volume quality (up weeks vs down weeks on volume, trailing 13w)
  - Distance from 52-week low
  - Volatility expansion (exit signal: ATR ratio)
  - Distribution pressure (exit signal: heavy selling on volume)

Train/val split: last 2 years = val (consistent with other feature files).
Outputs ALL daily rows; cluster_stage2.py filters as needed.

Usage:
    python -m src.quant_research.features_stage2
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import structlog

log = structlog.get_logger()

_INPUT_FILE = Path("data.nosync/quant/raw_prices.parquet")
_OUTPUT_FILE = Path("data.nosync/quant/features_stage2.parquet")

_MIN_BARS = 210  # 200-day SMA + 10-bar buffer

STAGE2_FEATURE_COLS = [
    # Trend alignment (positive = bullishly above)
    "price_to_sma150",        # (price / SMA150 - 1) * 100
    "price_to_sma200",        # (price / SMA200 - 1) * 100
    "sma150_to_sma200",       # (SMA150 / SMA200 - 1) * 100
    # SMA200 momentum
    "sma200_slope",           # 20-bar % change in SMA200 (is it turning up?)
    # Trend structure
    "hh_hl_score",            # fraction of 13 five-bar windows with HH + HL vs prior
    # Volume quality
    "up_vol_ratio",           # avg volume on up weeks / avg volume on down weeks (13w)
    "up_week_vol_dominance",  # fraction of up weeks that had above-median volume (13w)
    # Distance from trough
    "pct_above_52wk_low",     # (price / 52w low - 1) * 100
    # Exit signals
    "vol_expansion",          # ATR(10) / ATR(50) — volatility regime shift
    "distribution_score",     # count of large-range down days on high volume, trailing 20d
]


# ---------------------------------------------------------------------------
# Vectorized weekly feature helpers
# ---------------------------------------------------------------------------

def _compute_hh_hl_score(high: np.ndarray, low: np.ndarray, n_weeks: int = 13) -> np.ndarray:
    """
    For each bar, fraction of the trailing n_weeks five-bar windows where both
    the weekly high > prior window's high AND the weekly low > prior window's low.
    Returns [0, 1]; NaN for the first n_weeks*5 bars.
    """
    n = len(high)
    window = n_weeks * 5
    result = np.full(n, np.nan)
    if n < window:
        return result

    high_c = np.ascontiguousarray(high, dtype=np.float64)
    low_c = np.ascontiguousarray(low, dtype=np.float64)
    stride = high_c.strides[0]
    n_windows = n - window + 1

    h_strided = np.lib.stride_tricks.as_strided(
        high_c, shape=(n_windows, window), strides=(stride, stride)
    ).copy().reshape(n_windows, n_weeks, 5)

    l_strided = np.lib.stride_tricks.as_strided(
        low_c, shape=(n_windows, window), strides=(stride, stride)
    ).copy().reshape(n_windows, n_weeks, 5)

    h_weekly = h_strided.max(axis=2)    # (n_windows, n_weeks)
    l_weekly = l_strided.min(axis=2)    # (n_windows, n_weeks)

    hh = (h_weekly[:, 1:] > h_weekly[:, :-1]).sum(axis=1).astype(float)
    hl = (l_weekly[:, 1:] > l_weekly[:, :-1]).sum(axis=1).astype(float)

    result[window - 1:] = (hh + hl) / (2.0 * (n_weeks - 1))
    return result


def _compute_weekly_vol_features(
    close: np.ndarray, volume: np.ndarray, n_weeks: int = 13
) -> tuple[np.ndarray, np.ndarray]:
    """
    up_vol_ratio: avg volume on up weeks / avg volume on down weeks.
    up_week_vol_dominance: fraction of up weeks with above-median weekly volume.

    Uses n_weeks+1 five-bar windows so that each week can be classified as
    up/down relative to the prior week. NaN for the first (n_weeks+1)*5 bars.
    """
    n = len(close)
    total_weeks = n_weeks + 1
    total_window = total_weeks * 5

    up_vol_ratio = np.full(n, np.nan)
    up_week_dom = np.full(n, np.nan)

    if n < total_window:
        return up_vol_ratio, up_week_dom

    close_c = np.ascontiguousarray(close, dtype=np.float64)
    vol_c = np.ascontiguousarray(volume, dtype=np.float64)
    cs, vs = close_c.strides[0], vol_c.strides[0]
    n_windows = n - total_window + 1

    c_strided = np.lib.stride_tricks.as_strided(
        close_c, shape=(n_windows, total_window), strides=(cs, cs)
    ).copy().reshape(n_windows, total_weeks, 5)

    v_strided = np.lib.stride_tricks.as_strided(
        vol_c, shape=(n_windows, total_window), strides=(vs, vs)
    ).copy().reshape(n_windows, total_weeks, 5)

    c_weekly = c_strided[:, :, -1]     # last close of each five-bar window
    v_weekly = v_strided.sum(axis=2)   # total volume per window

    # Week is "up" if its close exceeds the prior window's close
    is_up = c_weekly[:, 1:] > c_weekly[:, :-1]   # (n_windows, n_weeks)
    wk_vol = v_weekly[:, 1:]                       # aligned with is_up

    med_vol = np.median(wk_vol, axis=1, keepdims=True)

    up_mask = is_up.astype(float)
    dn_mask = (~is_up).astype(float)

    up_vol_sum = (wk_vol * up_mask).sum(axis=1)
    up_count = up_mask.sum(axis=1)
    dn_vol_sum = (wk_vol * dn_mask).sum(axis=1)
    dn_count = dn_mask.sum(axis=1)

    avg_up = np.where(up_count > 0, up_vol_sum / up_count, np.nan)
    avg_dn = np.where(dn_count > 0, dn_vol_sum / dn_count, np.nan)
    ratio = np.where(avg_dn > 0, avg_up / avg_dn, np.nan)

    dom = (is_up & (wk_vol > med_vol)).sum(axis=1).astype(float) / n_weeks

    start = total_window - 1
    up_vol_ratio[start:] = ratio
    up_week_dom[start:] = dom

    return up_vol_ratio, up_week_dom


# ---------------------------------------------------------------------------
# Per-ticker feature builder
# ---------------------------------------------------------------------------

def _build_stage2_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Stage 2 features for one ticker.
    Requires columns: date, high, low, close, volume.
    Returns empty DataFrame if fewer than _MIN_BARS rows.
    Does NOT drop rows where forward targets are NaN — callers decide.
    """
    if len(df) < _MIN_BARS:
        return pd.DataFrame()

    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    volume = df["volume"].values
    dates = df["date"].values
    n = len(close)

    log_close = np.log(close)

    def sma(arr: np.ndarray, w: int) -> np.ndarray:
        return pd.Series(arr).rolling(w).mean().values

    # Trend alignment
    sma150 = sma(close, 150)
    sma200 = sma(close, 200)
    price_to_sma150 = (close / sma150 - 1.0) * 100.0
    price_to_sma200 = (close / sma200 - 1.0) * 100.0
    sma150_to_sma200 = (sma150 / sma200 - 1.0) * 100.0

    # SMA200 slope: 20-bar % change, normalized to per-bar units
    sma200_slope = np.full(n, np.nan)
    sma200_slope[20:] = (sma200[20:] - sma200[:-20]) / sma200[:-20] * 100.0

    # 52-week low
    low_252 = pd.Series(low).rolling(252).min().values
    pct_above_52wk_low = (close / low_252 - 1.0) * 100.0

    # Trend structure: higher highs / higher lows score
    hh_hl_score = _compute_hh_hl_score(high, low, n_weeks=13)

    # Volume quality features
    up_vol_ratio, up_week_dom = _compute_weekly_vol_features(close, volume, n_weeks=13)

    # Volatility expansion: ATR(10) / ATR(50)
    prev_close = np.empty(n)
    prev_close[0] = close[0]
    prev_close[1:] = close[:-1]
    tr = np.maximum(
        high - low,
        np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)),
    )
    atr10 = pd.Series(tr).rolling(10).mean().values
    atr50 = pd.Series(tr).rolling(50).mean().values
    vol_expansion = np.where(atr50 > 0, atr10 / atr50, np.nan)

    # Distribution score: large-range down days on above-avg volume, trailing 20 bars
    down_day = (close < prev_close).astype(float)
    large_range = ((high - low) / close > 0.02).astype(float)
    avg_vol_20 = pd.Series(volume).rolling(20).mean().values
    high_vol = (volume > 1.5 * avg_vol_20).astype(float)
    dist_day = down_day * large_range * high_vol
    distribution_score = pd.Series(dist_day).rolling(20).sum().values

    # Forward return targets
    fwd5 = np.full(n, np.nan)
    fwd10 = np.full(n, np.nan)
    fwd20 = np.full(n, np.nan)
    fwd5[:-5] = log_close[5:] - log_close[:-5]
    fwd10[:-10] = log_close[10:] - log_close[:-10]
    fwd20[:-20] = log_close[20:] - log_close[:-20]

    result = pd.DataFrame({
        "date": dates,
        "close": close,
        "price_to_sma150": price_to_sma150,
        "price_to_sma200": price_to_sma200,
        "sma150_to_sma200": sma150_to_sma200,
        "sma200_slope": sma200_slope,
        "hh_hl_score": hh_hl_score,
        "up_vol_ratio": up_vol_ratio,
        "up_week_vol_dominance": up_week_dom,
        "pct_above_52wk_low": pct_above_52wk_low,
        "vol_expansion": vol_expansion,
        "distribution_score": distribution_score,
        "fwd_log_ret_5d": fwd5,
        "fwd_log_ret_10d": fwd10,
        "fwd_log_ret_20d": fwd20,
    })

    target_cols = {"fwd_log_ret_5d", "fwd_log_ret_10d", "fwd_log_ret_20d", "date", "close"}
    feature_cols = [c for c in result.columns if c not in target_cols]
    return result.dropna(subset=feature_cols).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------

def build_features_stage2() -> pd.DataFrame:
    if not _INPUT_FILE.exists():
        raise FileNotFoundError(f"Run download.py first: {_INPUT_FILE}")

    log.info("features_stage2.loading", file=str(_INPUT_FILE))
    prices = pd.read_parquet(_INPUT_FILE)
    prices["date"] = pd.to_datetime(prices["date"])
    prices = prices.sort_values(["ticker", "date"]).reset_index(drop=True)

    tickers = list(prices["ticker"].unique())
    log.info("features_stage2.building", tickers=len(tickers))

    all_frames: list[pd.DataFrame] = []
    for i, ticker in enumerate(tickers):
        tdf = prices[prices["ticker"] == ticker].sort_values("date").reset_index(drop=True)
        feat = _build_stage2_features(tdf)
        if feat.empty:
            continue
        feat["ticker"] = ticker
        all_frames.append(feat)
        if (i + 1) % 100 == 0:
            log.info("features_stage2.progress", done=i + 1, total=len(tickers))

    if not all_frames:
        raise ValueError("No feature rows generated — check raw_prices.parquet")

    df = pd.concat(all_frames, ignore_index=True)
    log.info("features_stage2.concat", total_rows=len(df))

    # Outlier cap on targets (keeps NaN rows intact)
    for col, cap in [("fwd_log_ret_5d", 0.5), ("fwd_log_ret_10d", 1.0), ("fwd_log_ret_20d", 1.5)]:
        extreme = df[col].notna() & (df[col].abs() > cap)
        df.loc[extreme, col] = np.nan

    # Train/val split: last 2 years = val
    val_cutoff = df["date"].max() - pd.Timedelta(days=int(2 * 365.25))
    df["split"] = "train"
    df.loc[df["date"] > val_cutoff, "split"] = "val"

    log.info(
        "features_stage2.done",
        total_rows=len(df),
        train_rows=int((df["split"] == "train").sum()),
        val_rows=int((df["split"] == "val").sum()),
        tickers=df["ticker"].nunique(),
    )

    _OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(_OUTPUT_FILE, index=False)
    log.info("features_stage2.saved", file=str(_OUTPUT_FILE))

    return df


if __name__ == "__main__":
    df = build_features_stage2()
    train = df[df["split"] == "train"]
    val = df[df["split"] == "val"]
    print(f"\nStage 2 feature matrix: {len(df):,} rows, {df['ticker'].nunique()} tickers")
    print(f"Train: {len(train):,}  Val: {len(val):,}")
    print(f"\nFeature NaN counts and means:")
    for col in STAGE2_FEATURE_COLS:
        nans = df[col].isna().sum()
        mean = df[col].mean()
        print(f"  {col:<28}  NaN: {nans:>8,}  mean: {mean:>9.3f}")
