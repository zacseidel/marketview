"""
src/quant_research/features_v6.py

Builds the v6 feature matrix from raw price data.
Input:  data.nosync/quant/raw_prices.parquet
Output: data.nosync/quant/features_v6.parquet

v6 = v5 cleaned + expanded market regime + richer sector signal

Changes from v5:
  Dropped (low importance, theory-weak):
    r2_10d, r2_50d, r2_200d        — correlated with slope features, no marginal value
    pct_time_since_ath             — covered by pct_ath + pct_52w_high
    log_price                      — price level is not cross-sectionally predictive
    buyback_pct_12m, buyback_pct_1q — quarterly data, too stale for 5–20d targets
    sector_size                    — too stable to contribute short-horizon signal

  Added (market regime — dominant signal category):
    spy_ret_5d                     — short-term SPY momentum
    spy_pct_above_sma50            — faster regime indicator than SMA200
    mkt_breadth_sma200             — % of universe stocks above their own SMA200
    mkt_breadth_change_20d         — 20d change in breadth (expanding vs contracting)

  Added (sector signal gap):
    sector_ret_60d                 — fills 20d → 126d lookback gap
    stock_vs_sector_20d            — continuous stock vs sector excess at 20d

Final feature count: 43 numeric + 1 categorical (sector) = 44 total

Usage:
    python -m src.quant_research.features_v6
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import structlog

from src.quant_research.features_v2 import (
    _load_earnings_df,
    load_sector_info,
    _DAYS_TO_EARNINGS_CAP,
    _EARNINGS_DIR,
)
from src.quant_research.features_v4 import (
    _build_v4_base_features,
    _attach_earnings_timing,
    _load_sp500_set,
)

log = structlog.get_logger()

_INPUT_FILE  = Path("data.nosync/quant/raw_prices.parquet")
_OUTPUT_FILE = Path("data.nosync/quant/features_v6.parquet")

FEATURE_COLS_V6 = [
    # Trend: SMA ratios + regression slopes (R² dropped)
    "pct_sma10", "pct_sma50", "pct_sma200",
    "slope_10d", "slope_50d", "slope_200d",
    # Momentum
    "log_ret_5d", "log_ret_20d", "log_ret_60d", "log_ret_126d", "log_ret_252d", "log_ret_756d",
    "momentum_sharpe", "trend_accel",
    "pct_52w_high", "pct_52w_low", "pct_ath",
    # Volatility
    "vol_20d", "vol_60d",
    # Liquidity
    "log_dollar_vol_20d", "dollar_vol_rel_20d",
    # Earnings timing — v4 style (NaN outside 20-day window)
    "days_since_earnings", "days_until_earnings",
    # Market state — expanded (5 features vs 3 in v5)
    "spy_ret_5d", "spy_ret_20d", "spy_vol_20d", "spy_pct_above_sma200", "spy_pct_above_sma50",
    # Market breadth — new
    "mkt_breadth_sma200", "mkt_breadth_change_20d",
    # Sector — both 20d and 126d retained, 60d added, stock_vs_sector_20d added
    "sector_ret_20d", "sector_vs_spy_20d", "sector_ret_rank",
    "sector_ret_60d", "stock_vs_sector_20d",
    "sector_ret_126d", "stock_vs_sector_126d",
    # Universe membership
    "in_sp500",
    # Earnings fundamentals
    "eps_surprise_pct", "earn_ret_5d", "ni_yoy_growth", "rev_yoy_growth",
    # Earnings timing — v3 style (0–100 scale, 180-day cap)
    "days_to_next_earnings",
]
CATEGORICAL_COLS_V6 = ["sector"]

# Columns computed by _build_v4_base_features that v6 drops
_V4_COLS_TO_DROP = {"r2_10d", "r2_50d", "r2_200d", "pct_time_since_ath"}


# ---------------------------------------------------------------------------
# SPY features — extended
# ---------------------------------------------------------------------------

def _build_spy_lookup_v6(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Compute 5 SPY market state features for every date.
    Extends the v2/v4 3-feature lookup with spy_ret_5d and spy_pct_above_sma50.
    """
    spy = prices[prices["ticker"] == "SPY"].sort_values("date").reset_index(drop=True)
    if spy.empty:
        return pd.DataFrame(columns=[
            "date", "spy_ret_5d", "spy_ret_20d", "spy_vol_20d",
            "spy_pct_above_sma200", "spy_pct_above_sma50",
        ])

    close = spy["close"].values
    log_close = np.log(close)
    n = len(close)

    # Returns
    spy_ret_5d  = np.full(n, np.nan)
    spy_ret_20d = np.full(n, np.nan)
    spy_ret_5d[5:]  = log_close[5:]  - log_close[:-5]
    spy_ret_20d[20:] = log_close[20:] - log_close[:-20]

    # Realized vol (annualized)
    daily_lr = np.diff(log_close, prepend=np.nan)
    spy_vol_20d = pd.Series(daily_lr).rolling(20).std().values * math.sqrt(252)

    # SMA200 and SMA50
    sma200 = pd.Series(close).rolling(200).mean().values
    sma50  = pd.Series(close).rolling(50).mean().values
    spy_pct_above_sma200 = (close / sma200 - 1) * 100
    spy_pct_above_sma50  = (close / sma50  - 1) * 100

    return pd.DataFrame({
        "date":                spy["date"].values,
        "spy_ret_5d":          spy_ret_5d,
        "spy_ret_20d":         spy_ret_20d,
        "spy_vol_20d":         spy_vol_20d,
        "spy_pct_above_sma200": spy_pct_above_sma200,
        "spy_pct_above_sma50":  spy_pct_above_sma50,
    })


# ---------------------------------------------------------------------------
# Market breadth — cross-sectional per date
# ---------------------------------------------------------------------------

def _add_market_breadth(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add mkt_breadth_sma200 and mkt_breadth_change_20d.

    mkt_breadth_sma200: % of universe stocks with pct_sma200 > 0 (price above SMA200)
      on each date. Excludes tickers with NaN pct_sma200 (insufficient history).
    mkt_breadth_change_20d: 20-trading-day change in breadth percentage.

    Both are date-level features broadcast to all stocks.
    """
    # Compute daily breadth time series from non-NaN pct_sma200 rows
    valid = df[df["pct_sma200"].notna()][["date", "pct_sma200"]].copy()
    breadth_ts = (
        valid.groupby("date")["pct_sma200"]
        .apply(lambda x: (x > 0).mean() * 100)
        .reset_index()
        .rename(columns={"pct_sma200": "mkt_breadth_sma200"})
        .sort_values("date")
        .reset_index(drop=True)
    )
    breadth_ts["mkt_breadth_change_20d"] = breadth_ts["mkt_breadth_sma200"].diff(20)

    return df.merge(
        breadth_ts[["date", "mkt_breadth_sma200", "mkt_breadth_change_20d"]],
        on="date", how="left",
    )


# ---------------------------------------------------------------------------
# Sector features — v6 (20d / 60d / 126d lookbacks + new features)
# ---------------------------------------------------------------------------

def _add_sector_features_v6(
    df: pd.DataFrame,
    sector_map: dict[str, str],
) -> pd.DataFrame:
    """
    Add sector label and 7 sector numeric features.

    Lookbacks and features:
      20d:  sector_ret_20d, sector_vs_spy_20d, sector_ret_rank, stock_vs_sector_20d (new)
      60d:  sector_ret_60d (new)
      126d: sector_ret_126d, stock_vs_sector_126d

    Requires log_ret_20d, log_ret_60d, log_ret_126d, spy_ret_20d in df.
    Sectors with fewer than 3 tickers on a given date get NaN sector features.
    """
    df = df.copy()
    df["sector"] = df["ticker"].map(sector_map).fillna("Other")

    for lookback, base_col, ret_col, vs_col in [
        (20,  "log_ret_20d",  "sector_ret_20d",  "stock_vs_sector_20d"),
        (60,  "log_ret_60d",  "sector_ret_60d",  None),
        (126, "log_ret_126d", "sector_ret_126d", "stock_vs_sector_126d"),
    ]:
        counts = df.groupby(["date", "sector"])[base_col].transform("count")
        mean   = df.groupby(["date", "sector"])[base_col].transform("mean")

        df[ret_col] = np.where(counts >= 3, mean, np.nan)
        if vs_col:
            df[vs_col] = np.where(counts >= 3, df[base_col] - mean, np.nan)

    # sector_vs_spy_20d: sector 20d return vs SPY 20d return
    df["sector_vs_spy_20d"] = df["sector_ret_20d"] - df["spy_ret_20d"]
    df.loc[df["sector_ret_20d"].isna(), "sector_vs_spy_20d"] = np.nan

    # sector_ret_rank: within-sector percentile rank using 20d return
    df["sector_ret_rank"] = df.groupby(["date", "sector"])["log_ret_20d"].rank(pct=True)
    counts_20 = df.groupby(["date", "sector"])["log_ret_20d"].transform("count")
    df.loc[counts_20 < 3, "sector_ret_rank"] = np.nan

    return df


# ---------------------------------------------------------------------------
# Earnings fundamentals join (v3-style, no buyback)
# ---------------------------------------------------------------------------

def _attach_earnings_fundamentals(df: pd.DataFrame) -> pd.DataFrame:
    """
    As-of backward join of past earnings events onto each row.
    Attaches: eps_surprise_pct, earn_ret_5d, ni_yoy_growth, rev_yoy_growth.
    """
    earn_df = _load_earnings_df()
    if earn_df.empty:
        for col in ("eps_surprise_pct", "earn_ret_5d", "ni_yoy_growth", "rev_yoy_growth"):
            df[col] = np.nan
        return df

    past = (
        earn_df[["ticker", "event_date", "eps_surprise_pct", "earn_ret_5d",
                 "ni_yoy_growth", "rev_yoy_growth"]]
        .sort_values("event_date")
        .reset_index(drop=True)
    )
    df = df.sort_values("date").reset_index(drop=True)
    df = pd.merge_asof(
        df, past,
        left_on="date", right_on="event_date",
        by="ticker", direction="backward",
    )
    # Anti-leakage: earn_ret_5d = log(price[earn_date+5] / price[earn_date]).
    # earn_date+5 is still in the future when < 5 days have elapsed → leaks.
    days_since = (df["date"] - df["event_date"]).dt.days
    leak_mask = days_since.notna() & (days_since < 5)
    n_masked = int(leak_mask.sum())
    df.loc[leak_mask, "earn_ret_5d"] = np.nan
    log.info("features_v6.leakage_mask_earn_ret_5d",
             masked=n_masked, remaining=int(df["earn_ret_5d"].notna().sum()))
    return df.drop(columns=["event_date"])


def _attach_days_to_next_earnings(df: pd.DataFrame) -> pd.DataFrame:
    """
    Forward join of next earnings event; computes days_to_next_earnings
    on a 0–100 scale capped at _DAYS_TO_EARNINGS_CAP days.
    """
    earn_df = _load_earnings_df()
    if earn_df.empty:
        df["days_to_next_earnings"] = np.nan
        return df

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
    return df.drop(columns=["next_earnings_date"])


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------

def build_features_v6() -> pd.DataFrame:
    if not _INPUT_FILE.exists():
        raise FileNotFoundError(f"Run download.py first: {_INPUT_FILE}")

    log.info("features_v6.loading_prices", file=str(_INPUT_FILE))
    prices = pd.read_parquet(_INPUT_FILE)
    prices["date"] = pd.to_datetime(prices["date"])
    prices = prices.sort_values(["ticker", "date"]).reset_index(drop=True)

    spy_df   = _build_spy_lookup_v6(prices)
    sector_map, _ = load_sector_info()
    sp500_set = _load_sp500_set()
    log.info("features_v6.setup",
             spy_rows=len(spy_df),
             sectors=len(set(sector_map.values())),
             sp500_tickers=len(sp500_set))

    tickers = list(prices["ticker"].unique())
    log.info("features_v6.building_ticker_features", tickers=len(tickers))

    # --- Per-ticker base features (v4 base, minus r2 + pct_time_since_ath) ---
    all_frames: list[pd.DataFrame] = []
    for i, ticker in enumerate(tickers):
        tdf = prices[prices["ticker"] == ticker].sort_values("date").reset_index(drop=True)
        feat = _build_v4_base_features(tdf)
        if feat.empty:
            continue
        feat = feat.drop(columns=[c for c in _V4_COLS_TO_DROP if c in feat.columns])
        feat["ticker"]   = ticker
        feat["in_sp500"] = 1 if ticker in sp500_set else 0
        all_frames.append(feat)
        if (i + 1) % 100 == 0:
            log.info("features_v6.ticker_progress", done=i + 1, total=len(tickers))

    if not all_frames:
        raise ValueError("No feature rows generated — check raw_prices.parquet")

    df = pd.concat(all_frames, ignore_index=True)
    log.info("features_v6.concat_done", total_rows=len(df))

    # --- Outlier cap on all three forward targets ---
    for _col, _cap in [("fwd_log_ret_5d", 0.5), ("fwd_log_ret_10d", 1.0), ("fwd_log_ret_20d", 1.5)]:
        extreme = df[_col].notna() & (df[_col].abs() > _cap)
        df.loc[extreme, _col] = np.nan
    log.info("features_v6.outlier_cap", remaining_5d=int(df["fwd_log_ret_5d"].notna().sum()))

    # --- SPY features ---
    df = df.merge(spy_df, on="date", how="left")

    # --- Market breadth (cross-sectional) ---
    log.info("features_v6.computing_market_breadth")
    df = _add_market_breadth(df)

    # --- Earnings timing v4-style (days_since / days_until, NaN outside 20d) ---
    log.info("features_v6.attaching_earnings_timing")
    df = _attach_earnings_timing(df)

    # --- Earnings fundamentals ---
    log.info("features_v6.attaching_earnings_fundamentals")
    df = _attach_earnings_fundamentals(df)

    # --- days_to_next_earnings (v3-style, 0–100 scale) ---
    log.info("features_v6.attaching_days_to_next_earnings")
    df = _attach_days_to_next_earnings(df)

    # --- Sector features ---
    log.info("features_v6.computing_sector_features")
    df = _add_sector_features_v6(df, sector_map)

    # --- Train/val split: last 2 years = val ---
    val_cutoff = df["date"].max() - pd.Timedelta(days=int(2 * 365.25))
    df["split"] = "train"
    df.loc[df["date"] > val_cutoff, "split"] = "val"

    train_rows = (df["split"] == "train").sum()
    val_rows   = (df["split"] == "val").sum()
    thu_train  = ((df["split"] == "train") & (df["date"].dt.dayofweek == 3)).sum()
    thu_val    = ((df["split"] == "val")   & (df["date"].dt.dayofweek == 3)).sum()

    log.info(
        "features_v6.done",
        total_rows=len(df),
        train_rows=train_rows,
        val_rows=val_rows,
        thursday_train=thu_train,
        thursday_val=thu_val,
        tickers=df["ticker"].nunique(),
        features=len(FEATURE_COLS_V6) + len(CATEGORICAL_COLS_V6),
    )

    _OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(_OUTPUT_FILE, index=False)
    log.info("features_v6.saved", file=str(_OUTPUT_FILE))
    return df


if __name__ == "__main__":
    df = build_features_v6()
    thu = df[df["date"].dt.dayofweek == 3]
    print(f"\nFeature matrix v6: {len(df):,} total rows, {df['ticker'].nunique()} tickers")
    print(f"Thursdays — Train: {(thu['split']=='train').sum():,}  Val: {(thu['split']=='val').sum():,}")
    print(f"\nFeatures ({len(FEATURE_COLS_V6)} numeric + {len(CATEGORICAL_COLS_V6)} categorical):")
    for col in FEATURE_COLS_V6 + CATEGORICAL_COLS_V6:
        nans = df[col].isna().sum() if col in df.columns else "MISSING"
        print(f"  {col:<30}  NaN: {nans:,}")
