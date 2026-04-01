"""
src/quant_research/features_v2.py

Builds the v2 feature matrix: 14 technical + 2 repurchase + 5 earnings +
3 market state + 3 sector numeric = 27 numeric features, plus sector categorical.

Trained on full 10+ years of raw_prices.parquet; features require only 252 days
of price history per row so the model can run on Polygon's ~2yr data window.
(log_ret_756d excluded — it requires 3yr lookback.)

Val split: last 2 years (same period as Polygon data coverage).
Target: fwd_log_ret_10d (10-day forward log return).

Usage:
    python -m src.quant_research.features_v2
"""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import structlog

log = structlog.get_logger()

_INPUT_FILE = Path("data/quant/raw_prices.parquet")
_OUTPUT_FILE = Path("data/quant/features_v2.parquet")
_FUNDAMENTALS_DIR = Path("data/fundamentals")
_EARNINGS_DIR = Path("data/earnings")
_UNIVERSE_FILE = Path("data/universe/constituents.json")

_DAYS_TO_EARNINGS_CAP = 180   # cap days_to_next_earnings
_ATH_CAP = 1260               # cap days_since_ath at 5 years (same as v1)

FEATURE_COLS_V2 = [
    # Technical (14 — log_ret_756d excluded; max lookback is 252 days)
    "log_price", "pct_sma10", "pct_sma50", "pct_sma200",
    "pct_ath", "pct_time_since_ath", "pct_52w_low",
    "log_ret_5d", "log_ret_20d", "log_ret_60d", "log_ret_126d", "log_ret_252d",
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
CATEGORICAL_COLS_V2 = ["sector"]


# ---------------------------------------------------------------------------
# Per-ticker feature builder (252-day max lookback)
# ---------------------------------------------------------------------------

def _build_v2_base_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute v2 technical features + fwd_log_ret_10d for one ticker.
    Max lookback: 252 days — runs on any 2yr price window (Polygon data compatible).

    Unlike features.py's _build_ticker_features, this function:
      - Excludes log_ret_756d (requires 3yr lookback)
      - Computes fwd_log_ret_10d as the prediction target (not 20d)
      - Does NOT drop rows where fwd_log_ret_10d is NaN — callers decide
        whether to filter on the target (training yes, inference no)
    """
    close = df["close"].values
    dates = df["date"].values
    n = len(close)

    if n < 262:  # 252 lookback + 10 forward
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

    vol20 = vol(20)
    vol60 = vol(60)

    # ATH uses all available history (expanding window); pct_time capped at _ATH_CAP
    ath = pd.Series(close).expanding().max().values
    pct_ath = (close / ath - 1) * 100

    ath_indices = np.where(close >= ath)[0]
    last_ath_pos = np.searchsorted(ath_indices, np.arange(n), side="right") - 1
    days_since_ath_raw = np.minimum(np.arange(n) - ath_indices[last_ath_pos], _ATH_CAP).astype(float)
    pct_time_since_ath = (days_since_ath_raw / _ATH_CAP) * 100

    low_252 = pd.Series(close).rolling(252).min().values
    pct_52w_low = (close / low_252 - 1) * 100

    # 10-day forward return (training target; NaN for last 10 rows — kept intentionally)
    fwd10 = np.full(n, np.nan)
    fwd10[:-10] = log_close[10:] - log_close[:-10]

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
        "vol_20d": vol20,
        "vol_60d": vol60,
        "fwd_log_ret_10d": fwd10,
    })

    # Drop rows where any feature (not the target) is NaN — e.g., first 251 rows
    feature_cols = [c for c in result.columns if c not in ("fwd_log_ret_10d", "date", "close")]
    return result.dropna(subset=feature_cols).reset_index(drop=True)


# ---------------------------------------------------------------------------
# SPY market state
# ---------------------------------------------------------------------------

def _build_spy_lookup(prices_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute SPY market state features for every date.
    Returns DataFrame with [date, spy_ret_20d, spy_vol_20d, spy_pct_above_sma200].
    """
    spy = prices_df[prices_df["ticker"] == "SPY"].sort_values("date").reset_index(drop=True)
    if spy.empty:
        return pd.DataFrame(columns=["date", "spy_ret_20d", "spy_vol_20d", "spy_pct_above_sma200"])

    close = spy["close"].values
    log_close = np.log(close)
    n = len(close)

    spy_ret_20d = np.full(n, np.nan)
    spy_ret_20d[20:] = log_close[20:] - log_close[:-20]

    daily_lr = np.diff(log_close, prepend=np.nan)
    spy_vol_20d = pd.Series(daily_lr).rolling(20).std().values * math.sqrt(252)

    sma200 = pd.Series(close).rolling(200).mean().values
    spy_pct_above_sma200 = (close / sma200 - 1) * 100

    return pd.DataFrame({
        "date": spy["date"].values,
        "spy_ret_20d": spy_ret_20d,
        "spy_vol_20d": spy_vol_20d,
        "spy_pct_above_sma200": spy_pct_above_sma200,
    })


# ---------------------------------------------------------------------------
# Fundamentals
# ---------------------------------------------------------------------------

def _load_fundamentals_df() -> pd.DataFrame:
    """
    Load all fundamentals into one DataFrame with pre-computed lagged share columns.
    Returns DataFrame sorted by [ticker, filing_date] with buyback feature columns.
    """
    records = []
    for fpath in _FUNDAMENTALS_DIR.glob("*.json"):
        if fpath.name.startswith("."):
            continue
        with open(fpath) as f:
            data = json.load(f)
        for rec in data:
            shares = rec.get("shares_outstanding")
            filing = rec.get("filing_date")
            ticker = rec.get("ticker")
            if not filing or not ticker or shares is None:
                continue
            records.append({
                "ticker": ticker,
                "filing_date": pd.Timestamp(filing),
                "shares_outstanding": float(shares),
            })

    if not records:
        return pd.DataFrame(columns=["ticker", "filing_date", "buyback_pct_12m", "buyback_pct_1q"])

    df = pd.DataFrame(records)
    df = df.sort_values(["ticker", "filing_date"]).reset_index(drop=True)

    # shift(1) = 1 quarter ago, shift(4) = 4 quarters (12 months) ago — within each ticker
    df["shares_1q_ago"] = df.groupby("ticker")["shares_outstanding"].shift(1)
    df["shares_4q_ago"] = df.groupby("ticker")["shares_outstanding"].shift(4)

    df["buyback_pct_1q"] = (
        (df["shares_1q_ago"] - df["shares_outstanding"]) / df["shares_1q_ago"] * 100
    )
    df["buyback_pct_12m"] = (
        (df["shares_4q_ago"] - df["shares_outstanding"]) / df["shares_4q_ago"] * 100
    )

    return df[["ticker", "filing_date", "buyback_pct_12m", "buyback_pct_1q"]]


# ---------------------------------------------------------------------------
# Earnings
# ---------------------------------------------------------------------------

def _load_earnings_df() -> pd.DataFrame:
    """
    Load all earnings events into one DataFrame sorted by [ticker, event_date].
    """
    def _pct(v: object) -> float | None:
        return float(v) * 100 if v is not None else None

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
                "ticker": ticker,
                "event_date": pd.Timestamp(event_date),
                # Convert decimal ratios → percentage (× 100) for consistency with pct_* features
                "eps_surprise_pct": _pct(rec.get("eps_surprise_pct")),
                "earn_ret_5d": rec.get("earn_ret_5d"),   # already a log return
                "ni_yoy_growth": _pct(rec.get("ni_yoy_growth")),
                "rev_yoy_growth": _pct(rec.get("rev_yoy_growth")),
            })

    if not records:
        return pd.DataFrame(columns=["ticker", "event_date", "eps_surprise_pct",
                                     "earn_ret_5d", "ni_yoy_growth", "rev_yoy_growth"])

    df = pd.DataFrame(records)
    df = df.sort_values(["ticker", "event_date"]).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Sector
# ---------------------------------------------------------------------------

def _sic_to_broad_sector(sic_code: str) -> str:
    """
    Map a SIC code string to one of 11 GICS-like broad sectors.
    Reduces ~257 raw SIC strings → 11 categories for a regularized categorical feature.
    Carve-outs checked before broader ranges to avoid conflicts.
    """
    try:
        sic = int(sic_code)
    except (ValueError, TypeError):
        return "Other"

    # Carve-outs first (specific sub-ranges within broader categories)
    if (2830 <= sic <= 2836) or (3841 <= sic <= 3851) or (8000 <= sic <= 8099):
        return "Health Care"
    if (6500 <= sic <= 6599) or sic == 6798:
        return "Real Estate"
    if (3570 <= sic <= 3579) or (3670 <= sic <= 3679) or (7370 <= sic <= 7379):
        return "Information Technology"
    if (4800 <= sic <= 4899) or (7810 <= sic <= 7849):
        return "Communication Services"
    if 4900 <= sic <= 4999:
        return "Utilities"

    # Broad categories
    if (1300 <= sic <= 1399) or (2900 <= sic <= 2999):
        return "Energy"
    if (2100 <= sic <= 2199) or (5000 <= sic <= 5199):
        return "Consumer Staples"
    if ((3700 <= sic <= 3799) or (5200 <= sic <= 5999) or
            (7000 <= sic <= 7369) or (7500 <= sic <= 7599) or
            (7900 <= sic <= 7999) or (8200 <= sic <= 8299)):
        return "Consumer Discretionary"
    if (6000 <= sic <= 6499) or (6700 <= sic <= 6799):
        return "Financials"
    if ((1500 <= sic <= 1999) or (3400 <= sic <= 3569) or (3580 <= sic <= 3669) or
            (3680 <= sic <= 3699) or (3800 <= sic <= 3840) or (3900 <= sic <= 3999) or
            (4000 <= sic <= 4799) or (7380 <= sic <= 7499) or (8700 <= sic <= 8799)):
        return "Industrials"
    # Materials: mining, metals, chemicals, rubber, plastics, glass, textiles
    if ((100 <= sic <= 1299) or (1400 <= sic <= 1499) or
            (2000 <= sic <= 2099) or (2200 <= sic <= 2829) or
            (2837 <= sic <= 2899) or (3000 <= sic <= 3399)):
        return "Materials"
    return "Other"


def load_sector_info() -> tuple[dict[str, str], dict[str, float]]:
    """
    Returns (sector_map, sector_size_map):
      sector_map: {ticker: broad_sector_string}  (11 GICS-like categories)
      sector_size_map: {sector_string: log(count_active_tickers)}
    Maps sic_code → broad sector to reduce from ~257 raw SIC strings → 11 categories.
    """
    with open(_UNIVERSE_FILE) as f:
        constituents = json.load(f)

    sector_map = {
        t: _sic_to_broad_sector(v.get("sic_code", ""))
        for t, v in constituents.items()
    }

    active_sectors = [
        sector_map[t]
        for t, v in constituents.items()
        if v.get("status") == "active"
    ]
    sector_counts = Counter(active_sectors)
    sector_size_map = {s: math.log(max(c, 1)) for s, c in sector_counts.items()}

    return sector_map, sector_size_map


def add_sector_features(
    df: pd.DataFrame,
    sector_map: dict[str, str],
    sector_size_map: dict[str, float],
) -> pd.DataFrame:
    """Add sector, sector_ret_20d, sector_ret_rank, sector_size to df."""
    df = df.copy()
    df["sector"] = df["ticker"].map(sector_map).fillna("")

    # Require at least 3 same-sector tickers on the same date for sector features
    sector_counts = df.groupby(["date", "sector"])["log_ret_20d"].transform("count")
    df["sector_ret_20d"] = df.groupby(["date", "sector"])["log_ret_20d"].transform("mean")
    df.loc[sector_counts < 3, "sector_ret_20d"] = np.nan

    df["sector_ret_rank"] = df.groupby(["date", "sector"])["log_ret_20d"].rank(pct=True)
    df.loc[sector_counts < 3, "sector_ret_rank"] = np.nan

    # Sector excess return vs SPY — captures sector rotation relative to market
    df["sector_vs_spy_20d"] = df["sector_ret_20d"] - df["spy_ret_20d"]

    df["sector_size"] = df["sector"].map(sector_size_map)

    return df


# ---------------------------------------------------------------------------
# Main feature builder
# ---------------------------------------------------------------------------

def build_features_v2() -> pd.DataFrame:
    if not _INPUT_FILE.exists():
        raise FileNotFoundError(f"Run download.py first: {_INPUT_FILE}")

    log.info("features_v2.loading_prices", file=str(_INPUT_FILE))
    prices = pd.read_parquet(_INPUT_FILE)
    prices["date"] = pd.to_datetime(prices["date"])
    prices = prices.sort_values(["ticker", "date"]).reset_index(drop=True)

    spy_df = _build_spy_lookup(prices)
    log.info("features_v2.spy_lookup_built", rows=len(spy_df))

    sector_map, sector_size_map = load_sector_info()
    log.info("features_v2.sector_info_loaded", sectors=len(sector_size_map))

    # Include SPY so evaluate.py can look up its fwd_log_ret_10d as a benchmark
    tickers = list(prices["ticker"].unique())
    log.info("features_v2.building_ticker_features", tickers=len(tickers))

    all_frames: list[pd.DataFrame] = []

    for i, ticker in enumerate(tickers):
        tdf = prices[prices["ticker"] == ticker].sort_values("date").reset_index(drop=True)
        feat = _build_v2_base_features(tdf)
        if feat.empty:
            continue
        feat["ticker"] = ticker
        all_frames.append(feat)

        if (i + 1) % 100 == 0:
            log.info("features_v2.ticker_progress", done=i + 1, total=len(tickers))

    if not all_frames:
        raise ValueError("No feature rows generated — check raw_prices.parquet")

    df = pd.concat(all_frames, ignore_index=True)
    log.info("features_v2.concat_done", total_rows=len(df))

    # Drop rows without valid 10-day target and apply outlier cap
    # (keeps all years of data for training; just removes the last 10 rows per ticker)
    df = df.dropna(subset=["fwd_log_ret_10d"])
    before = len(df)
    df = df[df["fwd_log_ret_10d"].abs() <= 1.0]
    log.info("features_v2.outlier_cap", dropped=before - len(df))

    # Attach SPY market state
    df = df.merge(spy_df, on="date", how="left")

    # Attach fundamentals (as-of-date backward join: filing_date <= row date)
    fund_df = _load_fundamentals_df()
    log.info("features_v2.fundamentals_loaded", tickers=fund_df["ticker"].nunique() if not fund_df.empty else 0)
    if not fund_df.empty:
        # merge_asof requires the join key to be globally sorted (not just within groups)
        df = df.sort_values("date").reset_index(drop=True)
        fund_df = fund_df.sort_values("filing_date").reset_index(drop=True)
        df = pd.merge_asof(
            df, fund_df,
            left_on="date", right_on="filing_date",
            by="ticker", direction="backward",
        )
        df = df.drop(columns=["filing_date"])

    # Attach earnings features (as-of-date backward join: event_date <= row date)
    earn_df = _load_earnings_df()
    log.info("features_v2.earnings_loaded", tickers=earn_df["ticker"].nunique() if not earn_df.empty else 0)
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

        # days_to_next_earnings (forward join: next event_date after row date)
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
        # Normalize to [0, 100] as % of cap — consistent with pct_time_since_ath convention
        raw_days = (df["next_earnings_date"] - df["date"]).dt.days.clip(0, _DAYS_TO_EARNINGS_CAP)
        df["days_to_next_earnings"] = (raw_days / _DAYS_TO_EARNINGS_CAP) * 100
        df = df.drop(columns=["next_earnings_date"])

    # Sector features (computed across all tickers simultaneously — no lookahead)
    df = add_sector_features(df, sector_map, sector_size_map)

    # Train/val split: last 2 years = val (matches Polygon data coverage period)
    val_cutoff = df["date"].max() - pd.Timedelta(days=int(2 * 365.25))
    df["split"] = "train"
    df.loc[df["date"] > val_cutoff, "split"] = "val"

    # Log NaN rates for new features
    new_features = [c for c in FEATURE_COLS_V2 if c not in ("log_price", "pct_sma10",
        "pct_sma50", "pct_sma200", "pct_ath", "pct_time_since_ath", "pct_52w_low",
        "log_ret_5d", "log_ret_20d", "log_ret_60d", "log_ret_126d", "log_ret_252d",
        "vol_20d", "vol_60d")]
    nan_rates = {c: round(df[c].isna().mean(), 3) for c in new_features if c in df.columns}
    log.info("features_v2.nan_rates", **nan_rates)

    train_rows = (df["split"] == "train").sum()
    val_rows = (df["split"] == "val").sum()
    log.info(
        "features_v2.done",
        total_rows=len(df),
        train_rows=train_rows,
        val_rows=val_rows,
        tickers=df["ticker"].nunique(),
    )

    _OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(_OUTPUT_FILE, index=False)
    log.info("features_v2.saved", file=str(_OUTPUT_FILE))

    return df


if __name__ == "__main__":
    df = build_features_v2()
    print(f"\nFeature matrix v2: {len(df):,} rows, {df['ticker'].nunique()} tickers")
    print(f"Train: {(df['split']=='train').sum():,}  Val: {(df['split']=='val').sum():,}")
    available = [c for c in FEATURE_COLS_V2 + CATEGORICAL_COLS_V2 if c in df.columns]
    print(f"Features available ({len(available)}/{len(FEATURE_COLS_V2) + len(CATEGORICAL_COLS_V2)}): {available}")
