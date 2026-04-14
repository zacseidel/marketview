"""
src/selection/quant_v6.py

Quantitative factor model v6 — live inference shim.

v6 feature set (43 numeric + sector categorical = 44 total):
  - Trend: SMA ratios + slopes (R² dropped)
  - Momentum: 6 windows, sharpe, accel, ATH/52w proximity
  - Volatility: 20d/60d
  - Liquidity: log dollar vol, relative dollar vol
  - Earnings timing: days since/until (v4 style, NaN outside 20d window)
  - Market regime: SPY ret 5d/20d, vol 20d, pct above SMA200 + SMA50
  - Market breadth: breadth_sma200, breadth_change_20d
  - Sector: 20d/60d/126d returns, stock vs sector 20d/126d, rank, vs SPY
  - Universe membership: in_sp500
  - Earnings fundamentals: eps_surprise, earn_ret_5d, ni/rev yoy growth
  - Earnings timing v3 style: days_to_next_earnings

Runs ONLY on the Friday cycle. Returns [] on Tuesday runs.

Usage:
    python -m src.selection.quant_v6
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
from src.quant_research.features_v4 import _build_v4_base_features, _load_sp500_set
from src.quant_research.features_v2 import load_sector_info, _DAYS_TO_EARNINGS_CAP
from src.quant_research.features_v6 import FEATURE_COLS_V6, CATEGORICAL_COLS_V6

log = structlog.get_logger()

_ARTIFACTS_DIR = Path("data.nosync/quant/artifacts/gbm_v6")
_EARNINGS_DIR  = Path("data.nosync/earnings")
_UNIVERSE_FILE = Path("data.nosync/universe/constituents.json")

_MAX_HOLDINGS  = 5
_DAYS_SINCE_CAP  = 20
_DAYS_UNTIL_CAP  = 20


# ---------------------------------------------------------------------------
# Artifact helpers
# ---------------------------------------------------------------------------

def _artifacts_exist() -> bool:
    return all((_ARTIFACTS_DIR / n).exists()
               for n in ("model.pkl", "sector_categories.json"))


def _load_artifacts() -> tuple:
    with open(_ARTIFACTS_DIR / "model.pkl", "rb") as f:
        model = pickle.load(f)
    with open(_ARTIFACTS_DIR / "sector_categories.json") as f:
        sector_categories = json.load(f)
    return model, sector_categories


# ---------------------------------------------------------------------------
# SPY features — v6 (5 features: adds spy_ret_5d + spy_pct_above_sma50)
# ---------------------------------------------------------------------------

def _compute_spy_features_v6(prices: pd.DataFrame) -> dict[str, float | None]:
    spy = prices[prices["ticker"] == "SPY"].sort_values("date").reset_index(drop=True)
    base: dict[str, float | None] = {
        "spy_ret_5d": None, "spy_ret_20d": None, "spy_vol_20d": None,
        "spy_pct_above_sma200": None, "spy_pct_above_sma50": None,
    }
    if len(spy) < 20:
        return base

    close = spy["close"].values
    log_close = np.log(close)
    daily_lr = np.diff(log_close)

    if len(log_close) > 5:
        base["spy_ret_5d"] = float(log_close[-1] - log_close[-6])
    if len(log_close) > 20:
        base["spy_ret_20d"] = float(log_close[-1] - log_close[-21])
    if len(daily_lr) >= 20:
        base["spy_vol_20d"] = float(np.std(daily_lr[-20:]) * math.sqrt(252))
    if len(close) >= 200:
        sma200 = float(np.mean(close[-200:]))
        base["spy_pct_above_sma200"] = float((close[-1] / sma200 - 1) * 100)
    if len(close) >= 50:
        sma50 = float(np.mean(close[-50:]))
        base["spy_pct_above_sma50"] = float((close[-1] / sma50 - 1) * 100)

    return base


# ---------------------------------------------------------------------------
# Market breadth — cross-sectional (new in v6)
# ---------------------------------------------------------------------------

def _compute_market_breadth(
    prices: pd.DataFrame,
    feat_df: pd.DataFrame,
) -> tuple[float | None, float | None]:
    """
    Compute mkt_breadth_sma200 (today) and mkt_breadth_change_20d.

    mkt_breadth_sma200: % of universe stocks with pct_sma200 > 0 today.
    mkt_breadth_change_20d: today's breadth minus breadth 20 trading days ago.

    Uses feat_df for current pct_sma200 (already computed per-ticker).
    Uses prices to reconstruct pct_sma200 at t-20 for the change calculation.
    """
    valid_today = feat_df["pct_sma200"].dropna()
    if valid_today.empty:
        return None, None
    breadth_today = float((valid_today > 0).mean() * 100)

    # Find the date 20 trading days ago
    all_dates = sorted(prices["date"].unique())
    if len(all_dates) < 22:
        return breadth_today, None

    date_20d_ago = all_dates[-21]  # index -1 is today, -21 is 20 sessions back

    above_count = 0
    total_count = 0
    for ticker, tdf in prices.groupby("ticker"):
        if ticker == "SPY":
            continue
        tdf = tdf.sort_values("date").reset_index(drop=True)
        past = tdf[tdf["date"] <= date_20d_ago]
        if len(past) < 200:
            continue
        close_at = float(past["close"].iloc[-1])
        sma200_at = float(past["close"].iloc[-200:].mean())
        if sma200_at > 0:
            total_count += 1
            if close_at > sma200_at:
                above_count += 1

    if total_count == 0:
        return breadth_today, None

    breadth_20d_ago = (above_count / total_count) * 100
    return breadth_today, breadth_today - breadth_20d_ago


# ---------------------------------------------------------------------------
# Earnings — timing (v4 style) + fundamentals
# ---------------------------------------------------------------------------

def _load_all_earnings() -> dict[str, list]:
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
    result: dict[str, float | None] = {
        "days_since_earnings": None,
        "days_until_earnings": None,
    }
    as_of_str = as_of.isoformat()
    records = earnings_cache.get(ticker, [])

    past = [r for r in records if r.get("event_date") and r["event_date"] <= as_of_str]
    if past:
        last_date = _date.fromisoformat(max(r["event_date"] for r in past))
        days = (as_of - last_date).days
        if days <= _DAYS_SINCE_CAP:
            result["days_since_earnings"] = float(days)

    future = [r for r in records if r.get("event_date") and r["event_date"] > as_of_str]
    if ticker in next_dates:
        try:
            cal_date = _date.fromisoformat(next_dates[ticker])
            days = (cal_date - as_of).days
            if 0 <= days <= _DAYS_UNTIL_CAP:
                result["days_until_earnings"] = float(days)
        except ValueError:
            pass
    elif future:
        next_date = _date.fromisoformat(min(r["event_date"] for r in future))
        days = (next_date - as_of).days
        if 0 <= days <= _DAYS_UNTIL_CAP:
            result["days_until_earnings"] = float(days)

    return result


def _compute_earnings_fundamentals(
    ticker: str,
    as_of: _date,
    earnings_cache: dict[str, list],
) -> dict[str, float | None]:
    result: dict[str, float | None] = {
        "eps_surprise_pct": None,
        "earn_ret_5d": None,
        "ni_yoy_growth": None,
        "rev_yoy_growth": None,
    }
    as_of_str = as_of.isoformat()
    records = earnings_cache.get(ticker, [])
    past = [r for r in records if r.get("event_date") and r["event_date"] <= as_of_str]
    if not past:
        return result
    latest = max(past, key=lambda r: r["event_date"])
    for field in result:
        v = latest.get(field)
        if v is not None:
            try:
                result[field] = float(v)
            except (TypeError, ValueError):
                pass
    return result


def _compute_days_to_next_earnings(
    ticker: str,
    as_of: _date,
    earnings_cache: dict[str, list],
    next_dates: dict[str, str],
) -> float | None:
    as_of_str = as_of.isoformat()
    records = earnings_cache.get(ticker, [])
    raw_days: int | None = None

    if ticker in next_dates:
        try:
            cal_date = _date.fromisoformat(next_dates[ticker])
            d = (cal_date - as_of).days
            if d >= 0:
                raw_days = d
        except ValueError:
            pass

    if raw_days is None:
        future = [r for r in records if r.get("event_date") and r["event_date"] > as_of_str]
        if future:
            next_date = _date.fromisoformat(min(r["event_date"] for r in future))
            raw_days = (next_date - as_of).days

    if raw_days is None:
        return None
    return float(min(raw_days, _DAYS_TO_EARNINGS_CAP) / _DAYS_TO_EARNINGS_CAP * 100)


# ---------------------------------------------------------------------------
# Sector features — v6 (20d / 60d / 126d + stock_vs_sector_20d)
# ---------------------------------------------------------------------------

def _compute_sector_features_v6(
    feat_df: pd.DataFrame,
    sector_map: dict[str, str],
    spy_ret_20d: float | None,
) -> pd.DataFrame:
    """
    Add sector label and 7 numeric sector features across 20d/60d/126d lookbacks.
    Requires log_ret_20d, log_ret_60d, log_ret_126d already in feat_df.
    """
    feat_df = feat_df.copy()
    feat_df["sector"] = feat_df["ticker"].map(sector_map).fillna("Other")

    # 20d sector features
    counts_20 = feat_df.groupby("sector")["log_ret_20d"].transform("count")
    mean_20   = feat_df.groupby("sector")["log_ret_20d"].transform("mean")
    feat_df["sector_ret_20d"]    = np.where(counts_20 >= 3, mean_20, np.nan)
    feat_df["stock_vs_sector_20d"] = np.where(
        counts_20 >= 3, feat_df["log_ret_20d"] - mean_20, np.nan
    )
    feat_df["sector_ret_rank"]   = feat_df.groupby("sector")["log_ret_20d"].rank(pct=True)
    feat_df.loc[counts_20 < 3, "sector_ret_rank"] = np.nan

    spy_val = spy_ret_20d if spy_ret_20d is not None else np.nan
    feat_df["sector_vs_spy_20d"] = np.where(
        counts_20 >= 3, feat_df["sector_ret_20d"] - spy_val, np.nan
    )

    # 60d sector return (new in v6)
    counts_60 = feat_df.groupby("sector")["log_ret_60d"].transform("count")
    mean_60   = feat_df.groupby("sector")["log_ret_60d"].transform("mean")
    feat_df["sector_ret_60d"] = np.where(counts_60 >= 3, mean_60, np.nan)

    # 126d sector features
    counts_126 = feat_df.groupby("sector")["log_ret_126d"].transform("count")
    mean_126   = feat_df.groupby("sector")["log_ret_126d"].transform("mean")
    feat_df["sector_ret_126d"]       = np.where(counts_126 >= 3, mean_126, np.nan)
    feat_df["stock_vs_sector_126d"]  = np.where(
        counts_126 >= 3, feat_df["log_ret_126d"] - mean_126, np.nan
    )

    return feat_df


# ---------------------------------------------------------------------------
# Feature computation (current snapshot)
# ---------------------------------------------------------------------------

# Columns populated by cross-sectional passes — excluded from per-ticker loop
_CROSS_SECTIONAL_COLS = {
    "spy_ret_5d", "spy_ret_20d", "spy_vol_20d",
    "spy_pct_above_sma200", "spy_pct_above_sma50",
    "mkt_breadth_sma200", "mkt_breadth_change_20d",
    "sector_ret_20d", "sector_vs_spy_20d", "sector_ret_rank",
    "sector_ret_60d", "stock_vs_sector_20d",
    "sector_ret_126d", "stock_vs_sector_126d",
    "days_since_earnings", "days_until_earnings",
    "eps_surprise_pct", "earn_ret_5d", "ni_yoy_growth", "rev_yoy_growth",
    "days_to_next_earnings",
    "in_sp500",
}

# v4 base features dropped in v6 (computed but not passed to model)
_V4_DROPPED = {"r2_10d", "r2_50d", "r2_200d", "pct_time_since_ath", "log_price"}


def _compute_current_features_v6(
    prices: pd.DataFrame,
    sp500_set: set[str],
    sector_map: dict[str, str],
) -> pd.DataFrame:
    as_of = prices["date"].max().date()
    spy_features = _compute_spy_features_v6(prices)
    earnings_cache = _load_all_earnings()
    next_dates = load_next_dates()
    log.debug("quant_v6.caches_loaded",
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
        technical_cols = [
            c for c in FEATURE_COLS_V6
            if c in feat.columns and c not in _CROSS_SECTIONAL_COLS and c not in _V4_DROPPED
        ]

        row: dict = {"ticker": ticker}
        row.update({col: last[col] for col in technical_cols})
        row.update(_compute_earnings_timing(str(ticker), as_of, earnings_cache, next_dates))
        row.update(_compute_earnings_fundamentals(str(ticker), as_of, earnings_cache))
        row["days_to_next_earnings"] = _compute_days_to_next_earnings(
            str(ticker), as_of, earnings_cache, next_dates
        )
        row.update(spy_features)
        row["in_sp500"] = 1 if ticker in sp500_set else 0
        rows.append(row)

    if not rows:
        return pd.DataFrame(columns=["ticker"] + FEATURE_COLS_V6 + CATEGORICAL_COLS_V6)

    feat_df = pd.DataFrame(rows)

    # Cross-sectional: market breadth
    breadth, breadth_change = _compute_market_breadth(prices, feat_df)
    feat_df["mkt_breadth_sma200"]    = breadth
    feat_df["mkt_breadth_change_20d"] = breadth_change
    log.debug("quant_v6.breadth", mkt_breadth_sma200=breadth,
              mkt_breadth_change_20d=breadth_change)

    # Cross-sectional: sector features
    feat_df = _compute_sector_features_v6(
        feat_df, sector_map, spy_features.get("spy_ret_20d")
    )

    return feat_df


# ---------------------------------------------------------------------------
# SelectionModel
# ---------------------------------------------------------------------------

class QuantModelV6(SelectionModel):
    def run(self, config: dict, dal: DataAccessLayer) -> list[HoldingRecord]:
        eval_date_str = config.get("eval_date", _date.today().isoformat())
        max_holdings = config.get("max_holdings", _MAX_HOLDINGS)

        # Friday-only guard
        eval_date_obj = _date.fromisoformat(eval_date_str)
        if eval_date_obj.weekday() != 4:
            log.info("quant_v6.skip_non_friday", eval_date=eval_date_str)
            return []

        if not _artifacts_exist():
            log.warning("quant_v6.no_artifacts",
                        msg="Run src.quant_research.train_v6 first")
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

        feat_df = _compute_current_features_v6(prices, sp500_set, sector_map)
        if feat_df.empty:
            log.warning("quant_v6.no_features")
            return []

        # Score
        df_score = feat_df.copy()
        cat_type = pd.CategoricalDtype(categories=sector_categories, ordered=False)
        df_score["sector"] = df_score["sector"].astype(cat_type)
        all_cols = FEATURE_COLS_V6 + CATEGORICAL_COLS_V6
        predicted = model.predict(df_score.reindex(columns=all_cols))
        feat_df = feat_df.copy()
        feat_df["predicted_score"] = predicted

        log.info("quant_v6.scoring", tickers=len(feat_df),
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
                ("ni_yoy", "ni_yoy_growth"),
                ("breadth", "mkt_breadth_sma200"),
                ("sector_60d", "sector_ret_60d"),
                ("sector_126d", "sector_ret_126d"),
            ]:
                v = row.get(col)
                if v is not None and not (isinstance(v, float) and math.isnan(v)):
                    signals.append(f"{label}={float(v):+.3f}")

            metadata: dict = {}
            for col in FEATURE_COLS_V6:
                v = row.get(col)
                if v is not None and not (isinstance(v, float) and math.isnan(v)):
                    metadata[col] = round(float(v), 6)
            metadata["quant_model"] = "gbm_v6"
            metadata["predicted_score"] = round(pred, 6)
            sector_val = row.get("sector")
            if sector_val:
                metadata["sector"] = str(sector_val)

            holdings.append(HoldingRecord(
                model="quant_v6",
                eval_date=eval_date_str,
                ticker=str(row["ticker"]),
                conviction=conviction,
                rationale=f"Quant v6 (XGB, 5d raw): {'; '.join(signals)}",
                metadata=metadata,
            ))

        log.info("quant_v6.done", selected=len(holdings))
        return holdings


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    dal = DataAccessLayer()
    model_inst = QuantModelV6()
    today = _date.today()
    holdings = model_inst.run({"eval_date": today.isoformat()}, dal)

    if not holdings:
        day_name = today.strftime("%A")
        print(f"\nNo picks returned (today is {day_name} — model runs Fridays only).")
    else:
        print(f"\nTop picks (quant_v6):")
        for h in holdings:
            print(f"  {h.ticker:<6}  conviction={h.conviction:.3f}  {h.rationale}")
