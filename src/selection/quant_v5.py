"""
src/selection/quant_v5.py

Quantitative factor model v5 — live inference shim.

Full union feature set (45 numeric + sector categorical):
  v4 technical features + slope/R² + dollar volume + earnings timing (v4 style)
  + earnings fundamentals (eps_surprise, earn_ret_5d, ni/rev yoy growth)
  + log_price + buyback + days_to_next_earnings (v3 style)
  + sector_ret_20d/vs_spy/rank/size (v3 sector 20d)
  + sector_ret_126d/stock_vs_sector_126d (v4 sector 126d)

Runs ONLY on the Friday cycle. Returns [] on Tuesday runs.

Usage:
    python -m src.selection.quant_v5
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
from src.quant_research.features_v4 import (
    _build_v4_base_features,
    _load_sp500_set,
)
from src.quant_research.features_v2 import load_sector_info, _DAYS_TO_EARNINGS_CAP
from src.quant_research.features_v5 import FEATURE_COLS_V5, CATEGORICAL_COLS_V5

log = structlog.get_logger()

_ARTIFACTS_DIR = Path("data.nosync/quant/artifacts/gbm_v5")
_FUNDAMENTALS_DIR = Path("data.nosync/fundamentals")
_EARNINGS_DIR = Path("data.nosync/earnings")
_UNIVERSE_FILE = Path("data.nosync/universe/constituents.json")

_MAX_HOLDINGS = 5
_DAYS_SINCE_CAP = 20
_DAYS_UNTIL_CAP = 20


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
# SPY market state
# ---------------------------------------------------------------------------

def _compute_spy_features(prices: pd.DataFrame) -> dict[str, float | None]:
    spy = prices[prices["ticker"] == "SPY"].sort_values("date").reset_index(drop=True)
    if len(spy) < 20:
        return {"spy_ret_20d": None, "spy_vol_20d": None, "spy_pct_above_sma200": None}

    close = spy["close"].values
    log_close = np.log(close)

    spy_ret_20d = float(log_close[-1] - log_close[-21]) if len(log_close) > 20 else None
    daily_lr = np.diff(log_close)
    spy_vol_20d = float(np.std(daily_lr[-20:]) * math.sqrt(252)) if len(daily_lr) >= 20 else None
    sma200 = float(np.mean(close[-200:])) if len(close) >= 200 else None
    spy_pct_above_sma200 = float((close[-1] / sma200 - 1) * 100) if sma200 else None

    return {
        "spy_ret_20d": spy_ret_20d,
        "spy_vol_20d": spy_vol_20d,
        "spy_pct_above_sma200": spy_pct_above_sma200,
    }


# ---------------------------------------------------------------------------
# Earnings — timing (v4 style, 20-day cap) + fundamentals
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
    """days_since/until (v4 style, NaN outside 20-day window)."""
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
    """Most recent past earnings event's fundamental fields."""
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


def _compute_days_to_next_earnings_v3(
    ticker: str,
    as_of: _date,
    earnings_cache: dict[str, list],
    next_dates: dict[str, str],
) -> float | None:
    """
    V3-style days_to_next_earnings: (raw_days / _DAYS_TO_EARNINGS_CAP) * 100.
    Returns None if no next event found within cap.
    """
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
    clipped = min(raw_days, _DAYS_TO_EARNINGS_CAP)
    return float(clipped / _DAYS_TO_EARNINGS_CAP * 100)


# ---------------------------------------------------------------------------
# Fundamentals (buyback)
# ---------------------------------------------------------------------------

def _load_buyback_cache() -> dict[str, dict]:
    """Load most recent filing per ticker with buyback features."""
    cache: dict[str, dict] = {}
    for path in _FUNDAMENTALS_DIR.glob("*.json"):
        if path.name.startswith("."):
            continue
        ticker = path.stem
        try:
            with open(path) as f:
                records = json.load(f)
        except Exception:
            continue
        # Sort by filing_date descending, compute from most-recent 5 entries
        valid = [r for r in records if r.get("filing_date") and r.get("shares_outstanding")]
        if not valid:
            continue
        valid.sort(key=lambda r: r["filing_date"], reverse=True)

        shares_now = float(valid[0]["shares_outstanding"])
        shares_1q = float(valid[1]["shares_outstanding"]) if len(valid) > 1 else None
        shares_4q = float(valid[4]["shares_outstanding"]) if len(valid) > 4 else None

        buyback_1q = (
            (shares_1q - shares_now) / shares_1q * 100
            if shares_1q and shares_1q > 0 else None
        )
        buyback_12m = (
            (shares_4q - shares_now) / shares_4q * 100
            if shares_4q and shares_4q > 0 else None
        )
        cache[ticker] = {
            "buyback_pct_1q": buyback_1q,
            "buyback_pct_12m": buyback_12m,
        }
    return cache


# ---------------------------------------------------------------------------
# Sector features (both v4 126d and v3 20d)
# ---------------------------------------------------------------------------

def _compute_sector_features_v5(
    feat_df: pd.DataFrame,
    sector_map: dict[str, str],
    sector_size_map: dict[str, float],
    spy_ret_20d: float | None,
) -> pd.DataFrame:
    """
    Add sector (string) + v4 sector features (126d) + v3 sector features (20d).
    Requires log_ret_126d and log_ret_20d already in feat_df.
    """
    feat_df = feat_df.copy()
    feat_df["sector"] = feat_df["ticker"].map(sector_map).fillna("Other")

    # v4 sector features (126d lookback)
    counts_126 = feat_df.groupby("sector")["log_ret_126d"].transform("count")
    mean_126 = feat_df.groupby("sector")["log_ret_126d"].transform("mean")
    feat_df["sector_ret_126d"] = np.where(counts_126 >= 3, mean_126, np.nan)
    feat_df["stock_vs_sector_126d"] = np.where(
        counts_126 >= 3,
        feat_df["log_ret_126d"] - mean_126,
        np.nan,
    )

    # v3 sector features (20d lookback)
    counts_20 = feat_df.groupby("sector")["log_ret_20d"].transform("count")
    mean_20 = feat_df.groupby("sector")["log_ret_20d"].transform("mean")
    feat_df["sector_ret_20d"] = np.where(counts_20 >= 3, mean_20, np.nan)
    feat_df["sector_ret_rank"] = feat_df.groupby("sector")["log_ret_20d"].rank(pct=True)
    feat_df.loc[counts_20 < 3, "sector_ret_rank"] = np.nan

    spy_val = spy_ret_20d if spy_ret_20d is not None else np.nan
    feat_df["sector_vs_spy_20d"] = np.where(
        counts_20 >= 3,
        feat_df["sector_ret_20d"] - spy_val,
        np.nan,
    )

    feat_df["sector_size"] = feat_df["sector"].map(sector_size_map)

    return feat_df


# ---------------------------------------------------------------------------
# Feature computation (current snapshot)
# ---------------------------------------------------------------------------

def _compute_current_features_v5(
    prices: pd.DataFrame,
    sp500_set: set[str],
    sector_map: dict[str, str],
    sector_size_map: dict[str, float],
) -> pd.DataFrame:
    as_of = prices["date"].max().date()
    spy_features = _compute_spy_features(prices)
    earnings_cache = _load_all_earnings()
    next_dates = load_next_dates()
    buyback_cache = _load_buyback_cache()
    log.debug("quant_v5.caches_loaded",
              earnings=len(earnings_cache),
              next_dates=len(next_dates),
              buyback=len(buyback_cache))

    # v4 technical cols that come from per-ticker computation
    _SPY_COLS = {"spy_ret_20d", "spy_vol_20d", "spy_pct_above_sma200"}
    _SECTOR_COLS = {
        "sector_ret_126d", "stock_vs_sector_126d",
        "sector_ret_20d", "sector_vs_spy_20d", "sector_ret_rank", "sector_size",
    }
    _EARNINGS_COLS = {
        "days_since_earnings", "days_until_earnings",
        "eps_surprise_pct", "earn_ret_5d", "ni_yoy_growth", "rev_yoy_growth",
        "days_to_next_earnings",
    }
    _BUYBACK_COLS = {"buyback_pct_12m", "buyback_pct_1q"}
    _EXTRA_COLS = _SPY_COLS | _SECTOR_COLS | _EARNINGS_COLS | _BUYBACK_COLS | {"in_sp500", "log_price"}

    rows: list[dict] = []
    for ticker, tdf in prices.groupby("ticker"):
        if ticker == "SPY":
            continue
        tdf = tdf.sort_values("date").reset_index(drop=True)
        feat = _build_v4_base_features(tdf)
        if feat.empty:
            continue

        last = feat.iloc[-1]
        technical_cols = [c for c in FEATURE_COLS_V5
                          if c in feat.columns and c not in _EXTRA_COLS]

        row: dict = {"ticker": ticker}
        row.update({col: last[col] for col in technical_cols})

        # log_price
        row["log_price"] = float(last["log_price"]) if "log_price" in last.index else (
            float(np.log(tdf["close"].iloc[-1]))
        )

        # SPY features (broadcast — same for all tickers)
        row.update(spy_features)

        # Earnings timing v4 style
        row.update(_compute_earnings_timing(str(ticker), as_of, earnings_cache, next_dates))

        # Earnings fundamentals (most recent past event)
        row.update(_compute_earnings_fundamentals(str(ticker), as_of, earnings_cache))

        # days_to_next_earnings v3 style
        row["days_to_next_earnings"] = _compute_days_to_next_earnings_v3(
            str(ticker), as_of, earnings_cache, next_dates
        )

        # Buyback
        row.update(buyback_cache.get(str(ticker), {
            "buyback_pct_12m": None, "buyback_pct_1q": None,
        }))

        row["in_sp500"] = 1 if ticker in sp500_set else 0
        rows.append(row)

    if not rows:
        return pd.DataFrame(columns=["ticker"] + FEATURE_COLS_V5 + CATEGORICAL_COLS_V5)

    feat_df = pd.DataFrame(rows)

    # Cross-sectional sector features (requires full snapshot)
    feat_df = _compute_sector_features_v5(
        feat_df, sector_map, sector_size_map, spy_features.get("spy_ret_20d")
    )

    return feat_df


# ---------------------------------------------------------------------------
# SelectionModel
# ---------------------------------------------------------------------------

class QuantModelV5(SelectionModel):
    def run(self, config: dict, dal: DataAccessLayer) -> list[HoldingRecord]:
        eval_date_str = config.get("eval_date", _date.today().isoformat())
        max_holdings = config.get("max_holdings", _MAX_HOLDINGS)

        # Friday-only guard
        eval_date_obj = _date.fromisoformat(eval_date_str)
        if eval_date_obj.weekday() != 4:
            log.info("quant_v5.skip_non_friday", eval_date=eval_date_str)
            return []

        if not _artifacts_exist():
            log.warning("quant_v5.no_artifacts",
                        msg="Run src.quant_research.train_v5 first")
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
        sector_map, sector_size_map = load_sector_info()

        feat_df = _compute_current_features_v5(prices, sp500_set, sector_map, sector_size_map)
        if feat_df.empty:
            log.warning("quant_v5.no_features")
            return []

        # Score
        df_score = feat_df.copy()
        cat_type = pd.CategoricalDtype(categories=sector_categories, ordered=False)
        df_score["sector"] = df_score["sector"].astype(cat_type)
        all_cols = FEATURE_COLS_V5 + CATEGORICAL_COLS_V5
        X = df_score.reindex(columns=all_cols)
        predicted = model.predict(X)
        feat_df = feat_df.copy()
        feat_df["predicted_score"] = predicted

        log.info("quant_v5.scoring", tickers=len(feat_df),
                 above_zero=(predicted > 0).sum())

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
                ("sector_20d", "sector_ret_20d"),
                ("sector_126d", "sector_ret_126d"),
                ("buyback_12m", "buyback_pct_12m"),
            ]:
                v = row.get(col)
                if v is not None and not (isinstance(v, float) and math.isnan(v)):
                    signals.append(f"{label}={float(v):+.3f}")

            metadata: dict = {}
            for col in FEATURE_COLS_V5:
                v = row.get(col)
                if v is not None and not (isinstance(v, float) and math.isnan(v)):
                    metadata[col] = round(float(v), 6)
            metadata["quant_model"] = "gbm_v5"
            metadata["predicted_score"] = round(pred, 6)
            sector_val = row.get("sector")
            if sector_val:
                metadata["sector"] = str(sector_val)

            holdings.append(HoldingRecord(
                model="quant_v5",
                eval_date=eval_date_str,
                ticker=str(row["ticker"]),
                conviction=conviction,
                rationale=f"Quant v5 (XGB, 5d raw): {'; '.join(signals)}",
                metadata=metadata,
            ))

        log.info("quant_v5.done", selected=len(holdings))
        return holdings


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    dal = DataAccessLayer()
    model_inst = QuantModelV5()
    today = _date.today()
    holdings = model_inst.run({"eval_date": today.isoformat()}, dal)

    if not holdings:
        day_name = today.strftime("%A")
        print(f"\nNo picks returned (today is {day_name} — model runs Fridays only).")
    else:
        print(f"\nTop picks (quant_v5):")
        for h in holdings:
            print(f"  {h.ticker:<6}  conviction={h.conviction:.3f}  {h.rationale}")
