"""
src/selection/quant_v3.py

Quantitative factor model v3 — live inference shim.

All v2 features + log_ret_756d. Predicts 20-day forward log return.
Uses the same yfinance price cache as quant.py (needs ~800 days of history).

Usage:
    python -m src.selection.quant_v3
"""

from __future__ import annotations

import json
import math
import pickle
from datetime import date as _date
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import structlog

from src.selection.base import DataAccessLayer, HoldingRecord, SelectionModel
from src.selection.quant import _load_recent_prices, _write_history_gaps
from src.quant_research.features_v3 import FEATURE_COLS_V3, _build_v3_base_features
from src.quant_research.features_v2 import load_sector_info, add_sector_features
from src.quant_research.train_v2 import encode_sector

log = structlog.get_logger()

_ARTIFACTS_DIR = Path("data/quant/artifacts/gbm_v3")
_FUNDAMENTALS_DIR = Path("data/fundamentals")
_EARNINGS_DIR = Path("data/earnings")
_UNIVERSE_FILE = Path("data/universe/constituents.json")

_MAX_HOLDINGS = 20
_DAYS_TO_EARNINGS_CAP = 180


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

    return {"spy_ret_20d": spy_ret_20d, "spy_vol_20d": spy_vol_20d,
            "spy_pct_above_sma200": spy_pct_above_sma200}


def _compute_buyback_features(ticker: str) -> dict[str, float | None]:
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


def _compute_earnings_features(ticker: str, as_of: _date) -> dict[str, float | None]:
    fpath = _EARNINGS_DIR / f"{ticker}.json"
    if not fpath.exists():
        return {"eps_surprise_pct": None, "earn_ret_5d": None,
                "ni_yoy_growth": None, "rev_yoy_growth": None, "days_to_next_earnings": None}

    with open(fpath) as f:
        records = json.load(f)

    def _pct(v: object) -> float | None:
        return float(v) * 100 if v is not None else None

    as_of_str = as_of.isoformat()
    records_sorted = sorted([r for r in records if r.get("event_date")],
                            key=lambda r: r["event_date"])

    past = [r for r in records_sorted if r["event_date"] <= as_of_str]
    earn_features: dict[str, float | None] = {
        "eps_surprise_pct": None, "earn_ret_5d": None,
        "ni_yoy_growth": None, "rev_yoy_growth": None,
    }
    if past:
        latest = past[-1]
        earn_features["eps_surprise_pct"] = _pct(latest.get("eps_surprise_pct"))
        earn_features["earn_ret_5d"] = latest.get("earn_ret_5d")
        earn_features["ni_yoy_growth"] = _pct(latest.get("ni_yoy_growth"))
        earn_features["rev_yoy_growth"] = _pct(latest.get("rev_yoy_growth"))

    future = [r for r in records_sorted if r["event_date"] > as_of_str]
    if future:
        raw_days = min(max((pd.Timestamp(future[0]["event_date"]).date() - as_of).days, 0), _DAYS_TO_EARNINGS_CAP)
    elif past:
        raw_days = min(max((pd.Timestamp(past[-1]["event_date"]).date() + timedelta(days=91) - as_of).days, 0), _DAYS_TO_EARNINGS_CAP)
    else:
        raw_days = None

    earn_features["days_to_next_earnings"] = float(raw_days / _DAYS_TO_EARNINGS_CAP * 100) if raw_days is not None else None
    return earn_features


def _compute_current_features_v3(prices: pd.DataFrame) -> pd.DataFrame:
    as_of = prices["date"].max().date()
    spy_features = _compute_spy_features(prices)

    rows: list[dict] = []
    for ticker, tdf in prices.groupby("ticker"):
        if ticker == "SPY":
            continue
        tdf = tdf.sort_values("date").reset_index(drop=True)
        feat = _build_v3_base_features(tdf)
        if feat.empty:
            continue

        last = feat.iloc[-1]
        technical_cols = [c for c in FEATURE_COLS_V3 if c in feat.columns]
        row: dict = {"ticker": ticker}
        row.update({col: last[col] for col in technical_cols})
        row.update(_compute_buyback_features(str(ticker)))
        row.update(_compute_earnings_features(str(ticker), as_of))
        row.update(spy_features)
        rows.append(row)

    if not rows:
        return pd.DataFrame(columns=["ticker"] + FEATURE_COLS_V3)
    return pd.DataFrame(rows)


def _score_with_gbm_v3(feat_df: pd.DataFrame) -> np.ndarray:
    with open(_ARTIFACTS_DIR / "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open(_ARTIFACTS_DIR / "model.pkl", "rb") as f:
        model = pickle.load(f)
    with open(_ARTIFACTS_DIR / "sector_mapping.json") as f:
        sector_mapping = json.load(f)

    numeric_features = [c for c in FEATURE_COLS_V3 if c in feat_df.columns]
    X_numeric = feat_df.reindex(columns=numeric_features).values.astype(np.float64)
    X_scaled = scaler.transform(X_numeric)
    sector_encoded, _ = encode_sector(feat_df, sector_mapping)
    return model.predict(np.column_stack([X_scaled, sector_encoded]))


def _artifacts_exist() -> bool:
    return all((_ARTIFACTS_DIR / n).exists()
               for n in ("scaler.pkl", "model.pkl", "sector_mapping.json"))


class QuantModelV3(SelectionModel):
    def run(self, config: dict, dal: DataAccessLayer) -> list[HoldingRecord]:
        max_holdings = config.get("max_holdings", _MAX_HOLDINGS)
        eval_date = config.get("eval_date", "")

        if not _artifacts_exist():
            log.warning("quant_v3.no_artifacts", msg="Run src.quant_research.train_v3 first")
            return []

        with open(_UNIVERSE_FILE) as f:
            constituents = json.load(f)
        tickers = [v["ticker"] for v in constituents.values() if v.get("status") == "active"]

        prices = _load_recent_prices(list(set(tickers + ["SPY"])))
        if prices.empty:
            return []

        _write_history_gaps(tickers, prices)
        feat_df = _compute_current_features_v3(prices)
        if feat_df.empty:
            log.warning("quant_v3.no_features")
            return []

        feat_df["date"] = pd.Timestamp.today().normalize()
        sector_map, sector_size_map = load_sector_info()
        feat_df = add_sector_features(feat_df, sector_map, sector_size_map)
        feat_df = feat_df.drop(columns=["date"])

        log.info("quant_v3.scoring", tickers=len(feat_df))
        predicted = _score_with_gbm_v3(feat_df)
        feat_df = feat_df.copy()
        feat_df["predicted_log_ret"] = predicted
        feat_df = feat_df[feat_df["predicted_log_ret"] > 0]

        if feat_df.empty:
            log.warning("quant_v3.no_positive_signals")
            return []

        top = feat_df.nlargest(max_holdings, "predicted_log_ret").reset_index(drop=True)
        max_ret = top["predicted_log_ret"].max()
        min_ret = top["predicted_log_ret"].min()
        ret_range = max_ret - min_ret if max_ret != min_ret else 1.0

        holdings: list[HoldingRecord] = []
        for _, row in top.iterrows():
            pred_ret = float(row["predicted_log_ret"])
            conviction = round((pred_ret - min_ret) / ret_range, 3)

            signals: list[str] = [f"predicted 10d log return {pred_ret:+.4f}"]
            for label, col in [("buyback_12m", "buyback_pct_12m"), ("ni_yoy", "ni_yoy_growth"),
                                ("eps_surprise", "eps_surprise_pct"), ("days_to_earn", "days_to_next_earnings"),
                                ("spy_20d", "spy_ret_20d")]:
                val = row.get(col)
                if val is not None and not (isinstance(val, float) and math.isnan(val)):
                    signals.append(f"{label}={float(val):+.2f}")

            metadata: dict = {}
            for col in FEATURE_COLS_V3:
                v = row.get(col)
                if v is not None and not (isinstance(v, float) and math.isnan(v)):
                    metadata[col] = round(float(v), 6)
            metadata["quant_model"] = "gbm_v3"
            metadata["predicted_log_ret"] = round(pred_ret, 6)
            sector_val = row.get("sector")
            if sector_val:
                metadata["sector"] = str(sector_val)

            holdings.append(HoldingRecord(
                model="quant_v3",
                eval_date=eval_date,
                ticker=str(row["ticker"]),
                conviction=conviction,
                rationale=f"Quant v3 (GBM, 10d): {'; '.join(signals)}",
                metadata=metadata,
            ))

        log.info("quant_v3.done", selected=len(holdings))
        return holdings


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    dal = DataAccessLayer()
    model = QuantModelV3()
    holdings = model.run({"eval_date": _date.today().isoformat()}, dal)

    print(f"\nTop picks (quant_v3):")
    for h in holdings:
        print(f"  {h.ticker:<6}  conviction={h.conviction:.3f}  {h.rationale}")
