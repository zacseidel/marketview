"""
src/selection/quant_v2.py

Quantitative factor model v2 — live inference shim.

Uses GBM v2 artifacts (28 technical + buyback + earnings + market state + sector
features) to generate HoldingRecord signals predicting 10-day forward log return.

Shares the price cache from quant.py. Reads fundamentals from data.nosync/fundamentals/
and earnings from data.nosync/earnings/ for point-in-time feature computation.

Usage:
    python -m src.selection.quant_v2
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

from src.selection.base import DataAccessLayer, HoldingRecord, SelectionModel
from src.selection.quant import _load_recent_prices, _write_history_gaps
from src.quant_research.features_v2 import (
    _build_v2_base_features,
    FEATURE_COLS_V2, CATEGORICAL_COLS_V2,
    load_sector_info, add_sector_features,
)
from src.quant_research.train_v2 import encode_sector

log = structlog.get_logger()

_ARTIFACTS_DIR = Path("data.nosync/quant/artifacts/gbm_v2")
_FUNDAMENTALS_DIR = Path("data.nosync/fundamentals")
_EARNINGS_DIR = Path("data.nosync/earnings")
_UNIVERSE_FILE = Path("data.nosync/universe/constituents.json")

_MAX_HOLDINGS = 20
_DAYS_TO_EARNINGS_CAP = 180


# ---------------------------------------------------------------------------
# SPY market state (point-in-time)
# ---------------------------------------------------------------------------

def _compute_spy_features(prices: pd.DataFrame) -> dict[str, float | None]:
    """Compute current SPY market state features from the price cache."""
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
# Point-in-time buyback features
# ---------------------------------------------------------------------------

def _compute_buyback_features(ticker: str) -> dict[str, float | None]:
    """
    Read data.nosync/fundamentals/{ticker}.json and compute buyback_pct_12m, buyback_pct_1q.
    Records are stored most-recent-first; index 0 = latest, index 4 = ~12m ago.
    """
    fpath = _FUNDAMENTALS_DIR / f"{ticker}.json"
    if not fpath.exists():
        return {"buyback_pct_12m": None, "buyback_pct_1q": None}

    with open(fpath) as f:
        records = json.load(f)

    if not records:
        return {"buyback_pct_12m": None, "buyback_pct_1q": None}

    shares_latest = records[0].get("shares_outstanding")
    shares_1q = records[1].get("shares_outstanding") if len(records) > 1 else None
    shares_4q = records[4].get("shares_outstanding") if len(records) > 4 else None

    buyback_pct_1q = None
    if shares_latest is not None and shares_1q is not None and shares_1q != 0:
        buyback_pct_1q = (shares_1q - shares_latest) / shares_1q * 100

    buyback_pct_12m = None
    if shares_latest is not None and shares_4q is not None and shares_4q != 0:
        buyback_pct_12m = (shares_4q - shares_latest) / shares_4q * 100

    return {"buyback_pct_12m": buyback_pct_12m, "buyback_pct_1q": buyback_pct_1q}


# ---------------------------------------------------------------------------
# Point-in-time earnings features
# ---------------------------------------------------------------------------

def _compute_earnings_features(ticker: str, as_of: _date) -> dict[str, float | None]:
    """
    Read data.nosync/earnings/{ticker}.json and compute point-in-time earnings features.
    Uses event_date as the availability cutoff (earnings are public on announcement day).
    """
    fpath = _EARNINGS_DIR / f"{ticker}.json"
    if not fpath.exists():
        return {
            "eps_surprise_pct": None, "earn_ret_5d": None,
            "ni_yoy_growth": None, "rev_yoy_growth": None,
            "days_to_next_earnings": None,
        }

    with open(fpath) as f:
        records = json.load(f)

    as_of_str = as_of.isoformat()
    records_with_date = [r for r in records if r.get("event_date")]
    records_sorted = sorted(records_with_date, key=lambda r: r["event_date"])

    # Past events: most recent event on or before as_of
    past = [r for r in records_sorted if r["event_date"] <= as_of_str]
    earn_features: dict[str, float | None] = {
        "eps_surprise_pct": None, "earn_ret_5d": None,
        "ni_yoy_growth": None, "rev_yoy_growth": None,
    }
    def _pct(v: object) -> float | None:
        return float(v) * 100 if v is not None else None

    if past:
        latest = past[-1]
        # Convert decimal ratios → percentage (× 100) to match training data normalization
        earn_features["eps_surprise_pct"] = _pct(latest.get("eps_surprise_pct"))
        earn_features["earn_ret_5d"] = latest.get("earn_ret_5d")  # already a log return
        earn_features["ni_yoy_growth"] = _pct(latest.get("ni_yoy_growth"))
        earn_features["rev_yoy_growth"] = _pct(latest.get("rev_yoy_growth"))

    # days_to_next_earnings: normalize to [0, 100] as % of cap (matches training data)
    future = [r for r in records_sorted if r["event_date"] > as_of_str]
    if future:
        raw_days = min(max((pd.Timestamp(future[0]["event_date"]).date() - as_of).days, 0), _DAYS_TO_EARNINGS_CAP)
        earn_features["days_to_next_earnings"] = float(raw_days / _DAYS_TO_EARNINGS_CAP * 100)
    elif past:
        from datetime import timedelta
        raw_days = min(max((pd.Timestamp(past[-1]["event_date"]).date() + timedelta(days=91) - as_of).days, 0), _DAYS_TO_EARNINGS_CAP)
        earn_features["days_to_next_earnings"] = float(raw_days / _DAYS_TO_EARNINGS_CAP * 100)
    else:
        earn_features["days_to_next_earnings"] = None

    return earn_features


# ---------------------------------------------------------------------------
# Feature computation
# ---------------------------------------------------------------------------

def _compute_current_features_v2(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Compute v2 feature vector for each ticker's most recent date.
    Returns DataFrame with [ticker] + FEATURE_COLS_V2 (NaN for missing signals).
    """
    as_of = prices["date"].max().date()
    spy_features = _compute_spy_features(prices)

    rows: list[dict] = []

    for ticker, tdf in prices.groupby("ticker"):
        if ticker == "SPY":
            continue
        tdf = tdf.sort_values("date").reset_index(drop=True)
        feat = _build_v2_base_features(tdf)
        if feat.empty:
            continue

        # Use the most recent row — _build_v2_base_features keeps the last row
        # even when fwd_log_ret_10d is NaN (we're predicting the future)
        last = feat.iloc[-1]
        row: dict = {"ticker": ticker}
        row.update({col: last[col] for col in FEATURE_COLS_V2
                    if col in feat.columns and col not in
                    ("buyback_pct_12m", "buyback_pct_1q", "eps_surprise_pct",
                     "earn_ret_5d", "ni_yoy_growth", "rev_yoy_growth",
                     "days_to_next_earnings", "spy_ret_20d", "spy_vol_20d",
                     "spy_pct_above_sma200", "sector_ret_20d", "sector_ret_rank",
                     "sector_size")})

        # Repurchase features
        row.update(_compute_buyback_features(str(ticker)))

        # Earnings features
        row.update(_compute_earnings_features(str(ticker), as_of))

        # Market state
        row.update(spy_features)

        # Sector features added later (cross-ticker)
        rows.append(row)

    if not rows:
        return pd.DataFrame(columns=["ticker"] + FEATURE_COLS_V2)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _score_with_gbm_v2(feat_df: pd.DataFrame) -> np.ndarray:
    """Score rows using GBM v2 artifacts. NaN features handled natively by LightGBM."""
    with open(_ARTIFACTS_DIR / "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open(_ARTIFACTS_DIR / "model.pkl", "rb") as f:
        model = pickle.load(f)
    with open(_ARTIFACTS_DIR / "sector_mapping.json") as f:
        sector_mapping = json.load(f)

    numeric_features = [c for c in FEATURE_COLS_V2 if c in feat_df.columns]
    X_numeric = feat_df.reindex(columns=numeric_features).values.astype(np.float64)
    X_scaled = scaler.transform(X_numeric)

    sector_encoded, _ = encode_sector(feat_df, sector_mapping)
    X = np.column_stack([X_scaled, sector_encoded])
    return model.predict(X)


def _artifacts_exist() -> bool:
    return all(
        (_ARTIFACTS_DIR / name).exists()
        for name in ("scaler.pkl", "model.pkl", "sector_mapping.json")
    )


# ---------------------------------------------------------------------------
# SelectionModel implementation
# ---------------------------------------------------------------------------

class QuantModelV2(SelectionModel):
    def run(self, config: dict, dal: DataAccessLayer) -> list[HoldingRecord]:
        max_holdings = config.get("max_holdings", _MAX_HOLDINGS)
        eval_date = config.get("eval_date", "")

        if not _artifacts_exist():
            log.warning(
                "quant_v2.no_artifacts",
                msg="Run src.quant_research.train_v2 first",
            )
            return []

        with open(_UNIVERSE_FILE) as f:
            constituents = json.load(f)
        tickers = [
            v["ticker"] for v in constituents.values() if v.get("status") == "active"
        ]

        prices = _load_recent_prices(list(set(tickers + ["SPY"])))
        if prices.empty:
            return []

        _write_history_gaps(tickers, prices)
        feat_df = _compute_current_features_v2(prices)
        if feat_df.empty:
            log.warning("quant_v2.no_features")
            return []

        # Add sector features (requires all tickers' log_ret_20d on the same date)
        # inject a single shared date so add_sector_features can groupby ["date","sector"]
        feat_df["date"] = pd.Timestamp.today().normalize()
        sector_map, sector_size_map = load_sector_info()
        feat_df = add_sector_features(feat_df, sector_map, sector_size_map)
        feat_df = feat_df.drop(columns=["date"])

        log.info("quant_v2.scoring", tickers=len(feat_df))
        predicted = _score_with_gbm_v2(feat_df)
        feat_df = feat_df.copy()
        feat_df["predicted_log_ret"] = predicted
        feat_df = feat_df[feat_df["predicted_log_ret"] > 0]

        if feat_df.empty:
            log.warning("quant_v2.no_positive_signals")
            return []

        top = feat_df.nlargest(max_holdings, "predicted_log_ret").reset_index(drop=True)

        max_ret = top["predicted_log_ret"].max()
        min_ret = top["predicted_log_ret"].min()
        ret_range = max_ret - min_ret if max_ret != min_ret else 1.0

        holdings: list[HoldingRecord] = []
        for _, row in top.iterrows():
            pred_ret = float(row["predicted_log_ret"])
            conviction = round((pred_ret - min_ret) / ret_range, 3)

            # Build rationale highlighting v2-specific signals where available
            signals: list[str] = [f"predicted 10d log return {pred_ret:+.4f}"]

            bb12 = row.get("buyback_pct_12m")
            if bb12 is not None and not math.isnan(float(bb12)):
                signals.append(f"buyback_12m={float(bb12):+.2f}%")

            ni_g = row.get("ni_yoy_growth")
            if ni_g is not None and not math.isnan(float(ni_g)):
                signals.append(f"ni_yoy={float(ni_g):+.1%}")

            eps_s = row.get("eps_surprise_pct")
            if eps_s is not None and not math.isnan(float(eps_s)):
                signals.append(f"eps_surprise={float(eps_s):+.1%}")

            dte = row.get("days_to_next_earnings")
            if dte is not None and not math.isnan(float(dte)):
                signals.append(f"days_to_earn={int(dte)}")

            spy_r = row.get("spy_ret_20d")
            if spy_r is not None and not math.isnan(float(spy_r)):
                signals.append(f"spy_20d={float(spy_r):+.3f}")

            rationale = f"Quant v2 (GBM): {'; '.join(signals)}"

            metadata: dict = {}
            for col in FEATURE_COLS_V2:
                val = row.get(col)
                if val is not None and not (isinstance(val, float) and math.isnan(val)):
                    metadata[col] = round(float(val), 6)
            metadata["quant_model"] = "gbm_v2"
            metadata["predicted_log_ret"] = round(pred_ret, 6)
            sector_val = row.get("sector")
            if sector_val:
                metadata["sector"] = str(sector_val)

            holdings.append(HoldingRecord(
                model="quant_v2",
                eval_date=eval_date,
                ticker=str(row["ticker"]),
                conviction=conviction,
                rationale=rationale,
                metadata=metadata,
            ))

        log.info("quant_v2.done", selected=len(holdings))
        return holdings


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    dal = DataAccessLayer()
    model = QuantModelV2()
    holdings = model.run({"eval_date": _date.today().isoformat()}, dal)

    print(f"\nTop picks (quant_v2):")
    for h in holdings:
        print(f"  {h.ticker:<6}  conviction={h.conviction:.3f}  {h.rationale}")
