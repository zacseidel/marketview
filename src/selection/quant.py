"""
src/selection/quant.py

Quantitative factor model — pipeline integration shim.

Loads trained artifacts from data.nosync/quant/artifacts/ and applies them to
current market data to generate HoldingRecord signals.

Uses yfinance for recent price data (needs 756+ days of history, beyond
the 2-year window available in data.nosync/prices/). Maintains its own price
cache at data.nosync/quant/recent_prices.parquet, refreshed on each run.

Enabled in config/models.yaml after training and validation are complete.
Set `model` param to "knn", "gbm", "cluster", or "ensemble" (averages all three).

Usage:
    python -m src.selection.quant [--model knn|gbm|cluster|ensemble]
"""

from __future__ import annotations

import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import structlog

from src.selection.base import DataAccessLayer, HoldingRecord, SelectionModel
from src.quant_research.features import FEATURE_COLS, _build_ticker_features
from src.quant_research.train import CLUSTER_SCORE_THRESHOLD  # top-3-cluster threshold

log = structlog.get_logger()

_ARTIFACTS_DIR = Path("data.nosync/quant/artifacts")
_RECENT_PRICES_FILE = Path("data.nosync/quant/recent_prices.parquet")
_HISTORY_GAPS_FILE = Path("data.nosync/quant/history_gaps.json")
_UNIVERSE_FILE = Path("data.nosync/universe/constituents.json")
_PRICES_DIR = Path("data.nosync/prices")

_LOOKBACK_DAYS = 800   # need 756 + buffer for feature computation
_MIN_HISTORY_DAYS = 756
_MAX_HOLDINGS = 20
_DEFAULT_MODEL = "ensemble"


# ---------------------------------------------------------------------------
# One-time yfinance backfill (run with --backfill; not used in daily pipeline)
# ---------------------------------------------------------------------------

def _backfill_via_yfinance(tickers: list[str]) -> pd.DataFrame:
    """Download extended price history via yfinance. One-time use only."""
    import yfinance as yf
    from datetime import date, timedelta

    end = date.today()
    start = end - timedelta(days=int(_LOOKBACK_DAYS * 1.5))  # ~1200 calendar days

    log.info("quant.backfill_start", tickers=len(tickers))

    batch_size = 50
    frames: list[pd.DataFrame] = []

    for i in range(0, len(tickers), batch_size):
        batch = tickers[i: i + batch_size]
        try:
            raw = yf.download(
                batch,
                start=start.isoformat(),
                end=end.isoformat(),
                auto_adjust=True,
                progress=False,
                group_by="ticker",
            )
        except Exception as exc:
            log.warning("quant.backfill_fetch_error", error=str(exc))
            continue

        for ticker in batch:
            try:
                df = raw[ticker][["Open", "High", "Low", "Close", "Volume"]].copy()
                df.columns = ["open", "high", "low", "close", "volume"]
                df = df.dropna(subset=["close"])
                df["ticker"] = ticker
                df.index.name = "date"
                frames.append(df.reset_index())
            except (KeyError, Exception):
                pass
        time.sleep(0.5)

    if not frames:
        raise RuntimeError("No price data downloaded during backfill")

    combined = pd.concat(frames, ignore_index=True)
    combined["date"] = pd.to_datetime(combined["date"])
    combined = combined.sort_values(["ticker", "date"]).reset_index(drop=True)

    _RECENT_PRICES_FILE.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(_RECENT_PRICES_FILE, index=False)
    log.info("quant.backfill_complete", tickers=combined["ticker"].nunique(), rows=len(combined))
    return combined


# ---------------------------------------------------------------------------
# Price cache — local only (daily pipeline)
# ---------------------------------------------------------------------------

def _append_local_prices(df: pd.DataFrame) -> pd.DataFrame:
    """Append any local data.nosync/prices/ files newer than the cached parquet."""
    import json as _json

    latest_cached = df["date"].max() if not df.empty else pd.Timestamp("2000-01-01")
    new_frames: list[pd.DataFrame] = []

    for price_file in sorted(_PRICES_DIR.glob("*.json")):
        if not price_file.stem[0].isdigit():
            continue
        file_date = pd.Timestamp(price_file.stem)
        if file_date <= latest_cached:
            continue
        with open(price_file) as f:
            records = _json.load(f)
        if not isinstance(records, list) or not records:
            continue
        day_df = pd.DataFrame(records)
        cols = [c for c in ["date", "ticker", "open", "high", "low", "close", "volume"] if c in day_df.columns]
        day_df = day_df[cols].copy()
        day_df["date"] = pd.to_datetime(day_df["date"])
        new_frames.append(day_df)

    if not new_frames:
        return df

    combined = pd.concat([df] + new_frames, ignore_index=True)
    combined = combined.sort_values(["ticker", "date"]).reset_index(drop=True)
    combined.to_parquet(_RECENT_PRICES_FILE, index=False)
    log.info("quant.appended_local_prices", new_days=len(new_frames))
    return combined


def _write_history_gaps(tickers: list[str], prices: pd.DataFrame) -> list[str]:
    """Find tickers with fewer than MIN_HISTORY_DAYS trading days; write gaps file."""
    import json as _json
    from datetime import datetime as _dt

    counts = prices.groupby("ticker")["date"].count()
    gaps = sorted(t for t in tickers if counts.get(t, 0) < _MIN_HISTORY_DAYS)
    _HISTORY_GAPS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(_HISTORY_GAPS_FILE, "w") as f:
        _json.dump({"as_of": _dt.today().strftime("%Y-%m-%d"), "count": len(gaps), "tickers": gaps}, f, indent=2)
    if gaps:
        log.warning("quant.history_gaps", count=len(gaps), sample=gaps[:5])
    return gaps


def _load_recent_prices(tickers: list[str]) -> pd.DataFrame:
    """Load price cache and append any new local files. No network calls."""
    if not _RECENT_PRICES_FILE.exists():
        log.warning(
            "quant.no_price_cache",
            msg="Run one-time backfill first: python -m src.selection.quant --backfill",
        )
        return pd.DataFrame()
    df = pd.read_parquet(_RECENT_PRICES_FILE)
    df["date"] = pd.to_datetime(df["date"])
    return _append_local_prices(df)


# ---------------------------------------------------------------------------
# Feature computation for current market state
# ---------------------------------------------------------------------------

def _compute_current_features(prices: pd.DataFrame) -> pd.DataFrame:
    """
    For each ticker, compute the feature vector for the most recent date.
    Returns DataFrame with columns [ticker] + FEATURE_COLS.
    """
    rows: list[dict] = []

    for ticker, tdf in prices.groupby("ticker"):
        tdf = tdf.sort_values("date").reset_index(drop=True)
        feat = _build_ticker_features(tdf)
        if len(feat) == 0:
            continue
        last = feat.iloc[-1]
        row = {"ticker": ticker}
        row.update({col: last[col] for col in FEATURE_COLS})
        rows.append(row)

    if not rows:
        return pd.DataFrame(columns=["ticker"] + FEATURE_COLS)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Scoring functions (ndarray output — used by evaluate.py)
# ---------------------------------------------------------------------------

def _score_with_knn(X: np.ndarray) -> np.ndarray:
    from sklearn.neighbors import BallTree

    d = _ARTIFACTS_DIR / "knn"
    with open(d / "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open(d / "balltree.pkl", "rb") as f:
        tree: BallTree = pickle.load(f)
    labels = np.load(d / "labels.npy")

    X_scaled = scaler.transform(X.astype(np.float32)).astype(np.float32)
    distances, indices = tree.query(X_scaled, k=50)
    weights = 1.0 / (distances + 1e-8)
    return (weights * labels[indices]).sum(axis=1) / weights.sum(axis=1)


def _score_with_gbm(X: np.ndarray) -> np.ndarray:
    d = _ARTIFACTS_DIR / "gbm"
    with open(d / "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open(d / "model.pkl", "rb") as f:
        model = pickle.load(f)
    return model.predict(scaler.transform(X))


def _score_with_cluster(X: np.ndarray) -> np.ndarray:
    d = _ARTIFACTS_DIR / "cluster"
    with open(d / "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open(d / "kmeans.pkl", "rb") as f:
        kmeans = pickle.load(f)
    with open(d / "cluster_stats.json") as f:
        cluster_stats = json.load(f)

    cluster_ids = kmeans.predict(scaler.transform(X))
    return np.array([cluster_stats[str(cid)]["mean_fwd_ret"] for cid in cluster_ids])


_SCORERS = {
    "knn": _score_with_knn,
    "gbm": _score_with_gbm,
    "cluster": _score_with_cluster,
}


# ---------------------------------------------------------------------------
# Detail scoring (rich dict output — used by QuantModel.run())
# ---------------------------------------------------------------------------

def _detail_knn(X: np.ndarray) -> list[dict]:
    """Returns per-row: predicted_log_ret (distance-weighted avg of K neighbors)."""
    from sklearn.neighbors import BallTree

    d = _ARTIFACTS_DIR / "knn"
    with open(d / "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open(d / "balltree.pkl", "rb") as f:
        tree: BallTree = pickle.load(f)
    labels = np.load(d / "labels.npy")

    X_scaled = scaler.transform(X.astype(np.float32)).astype(np.float32)
    distances, indices = tree.query(X_scaled, k=50)
    weights = 1.0 / (distances + 1e-8)
    predicted = (weights * labels[indices]).sum(axis=1) / weights.sum(axis=1)

    return [
        {
            "predicted_log_ret": float(predicted[i]),
            "knn_avg_neighbor_dist": round(float(distances[i].mean()), 5),
        }
        for i in range(len(X))
    ]


def _detail_gbm(X: np.ndarray) -> list[dict]:
    """Returns per-row: predicted_log_ret from GBM regression."""
    d = _ARTIFACTS_DIR / "gbm"
    with open(d / "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open(d / "model.pkl", "rb") as f:
        model = pickle.load(f)
    predicted = model.predict(scaler.transform(X))
    return [{"predicted_log_ret": float(p)} for p in predicted]


def _detail_cluster(X: np.ndarray) -> list[dict]:
    """Returns per-row: cluster_id, cluster stats (mean log ret, hit rate, n)."""
    d = _ARTIFACTS_DIR / "cluster"
    with open(d / "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open(d / "kmeans.pkl", "rb") as f:
        kmeans = pickle.load(f)
    with open(d / "cluster_stats.json") as f:
        cluster_stats = json.load(f)

    cluster_ids = kmeans.predict(scaler.transform(X))
    return [
        {
            "predicted_log_ret": cluster_stats[str(cid)]["mean_fwd_ret"],
            "cluster_id": int(cid),
            "cluster_mean_log_ret": cluster_stats[str(cid)]["mean_fwd_ret"],
            "cluster_hit_rate": cluster_stats[str(cid)]["hit_rate"],
            "cluster_n": cluster_stats[str(cid)]["n"],
        }
        for cid in cluster_ids
    ]


_DETAILERS = {
    "knn": _detail_knn,
    "gbm": _detail_gbm,
    "cluster": _detail_cluster,
}


def _artifacts_exist(model: str) -> bool:
    if model == "ensemble":
        return all((_ARTIFACTS_DIR / name).exists() for name in ("knn", "gbm", "cluster"))
    return (_ARTIFACTS_DIR / model).exists()


# ---------------------------------------------------------------------------
# SelectionModel implementation
# ---------------------------------------------------------------------------

class QuantModel(SelectionModel):
    def run(self, config: dict, dal: DataAccessLayer) -> list[HoldingRecord]:
        model_name = config.get("model", _DEFAULT_MODEL)
        max_holdings = config.get("max_holdings", _MAX_HOLDINGS)
        eval_date = config.get("eval_date", "")

        if not _artifacts_exist(model_name):
            log.warning(
                "quant.no_artifacts",
                model=model_name,
                msg="Run src.quant_research.train first",
            )
            return []

        with open(_UNIVERSE_FILE) as f:
            constituents = json.load(f)
        tickers = [
            v["ticker"]
            for v in constituents.values()
            if v.get("status") == "active"
        ]

        prices = _load_recent_prices(list(set(tickers + ["SPY"])))
        if prices.empty:
            return []

        _write_history_gaps(tickers, prices)
        feat_df = _compute_current_features(prices)

        if feat_df.empty:
            log.warning("quant.no_features")
            return []

        log.info("quant.scoring", tickers=len(feat_df), model=model_name)
        X = feat_df[FEATURE_COLS].values.astype(np.float64)
        feat_df = feat_df.copy()

        # Get rich per-ticker details from each model
        if model_name == "ensemble":
            all_details: dict[str, list[dict]] = {}
            for name, detailer in _DETAILERS.items():
                try:
                    all_details[name] = detailer(X)
                except Exception as exc:
                    log.warning("quant.detail_error", model=name, error=str(exc))

            if not all_details:
                return []

            # Ensemble score = rank-normalized predicted_log_ret averaged across models
            scores_list = []
            for name, details in all_details.items():
                raw = np.array([d["predicted_log_ret"] for d in details])
                scores_list.append(pd.Series(raw).rank(pct=True).values)

            feat_df["predicted_log_ret"] = np.mean(
                [np.array([d["predicted_log_ret"] for d in dets])
                 for dets in all_details.values()],
                axis=0,
            )
            feat_df["score"] = np.mean(scores_list, axis=0)
            detail_list: list[dict] = [
                {name: all_details[name][i] for name in all_details}
                for i in range(len(feat_df))
            ]
        else:
            details = _DETAILERS[model_name](X)
            feat_df["predicted_log_ret"] = [d["predicted_log_ret"] for d in details]
            feat_df["score"] = feat_df["predicted_log_ret"]
            detail_list = details

        # Exclude SPY; exclude negative predicted returns (no point holding expected losers)
        feat_df = feat_df[feat_df["ticker"] != "SPY"]
        # For cluster model: apply minimum score threshold (only buy top-performing clusters)
        if model_name == "cluster":
            feat_df = feat_df[feat_df["predicted_log_ret"] >= CLUSTER_SCORE_THRESHOLD]
        else:
            feat_df = feat_df[feat_df["predicted_log_ret"] > 0]

        if feat_df.empty:
            log.warning("quant.no_positive_signals")
            return []

        top = feat_df.nlargest(max_holdings, "score").reset_index(drop=True)

        # Conviction = predicted_log_ret normalized to [0,1] within top-N
        max_ret = top["predicted_log_ret"].max()
        min_ret = top["predicted_log_ret"].min()
        ret_range = max_ret - min_ret if max_ret != min_ret else 1.0

        holdings: list[HoldingRecord] = []
        for i, row in top.iterrows():
            conviction = round((row["predicted_log_ret"] - min_ret) / ret_range, 3)
            pred_ret = float(row["predicted_log_ret"])
            detail = detail_list[i] if i < len(detail_list) else {}

            # Build rationale and metadata based on model type
            if model_name == "cluster":
                rationale = (
                    f"Quant (cluster {detail.get('cluster_id', '?')}): "
                    f"predicted 20d log return {pred_ret:+.4f} "
                    f"(cluster avg={detail.get('cluster_mean_log_ret', 0):+.4f}, "
                    f"hit_rate={detail.get('cluster_hit_rate', 0):.1%}, "
                    f"n={detail.get('cluster_n', 0):,}); "
                    f"pct_sma200={row['pct_sma200']:.1f}%, pct_ath={row['pct_ath']:.1f}%"
                )
            elif model_name == "ensemble":
                sub_rets = {
                    name: f"{d['predicted_log_ret']:+.4f}"
                    for name, d in detail.items()
                }
                rationale = (
                    f"Quant (ensemble): predicted 20d log return {pred_ret:+.4f} "
                    f"[knn={sub_rets.get('knn','?')}, gbm={sub_rets.get('gbm','?')}, "
                    f"cluster={sub_rets.get('cluster','?')}]; "
                    f"pct_sma200={row['pct_sma200']:.1f}%, pct_ath={row['pct_ath']:.1f}%"
                )
            else:
                rationale = (
                    f"Quant ({model_name}): predicted 20d log return {pred_ret:+.4f}; "
                    f"pct_sma200={row['pct_sma200']:.1f}%, "
                    f"pct_ath={row['pct_ath']:.1f}%, "
                    f"log_ret_252d={row['log_ret_252d']:.3f}"
                )

            metadata: dict = {col: round(float(row[col]), 5) for col in FEATURE_COLS}
            metadata["quant_model"] = model_name
            # Merge in model-specific detail fields first, then overwrite predicted_log_ret
            # with the correct row value (detail_list index may not align with top after
            # filtering + reset_index, so the detail's predicted_log_ret can be wrong)
            if isinstance(detail, dict):
                for k, v in detail.items():
                    if not isinstance(v, dict):
                        metadata[k] = v
            metadata["predicted_log_ret"] = round(pred_ret, 6)

            holdings.append(HoldingRecord(
                model="quant",
                eval_date=eval_date,
                ticker=row["ticker"],
                conviction=conviction,
                rationale=rationale,
                metadata=metadata,
            ))

        log.info("quant.done", selected=len(holdings), model=model_name)
        return holdings


if __name__ == "__main__":
    import argparse
    from dotenv import load_dotenv
    load_dotenv()

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=_DEFAULT_MODEL,
                        choices=["knn", "gbm", "cluster", "ensemble"])
    parser.add_argument("--backfill", action="store_true",
                        help="One-time: download extended price history via yfinance and save to cache")
    parser.add_argument("--refresh-cache", action="store_true",
                        help="Append any new local price files to the recent prices cache")
    args = parser.parse_args()

    if args.refresh_cache:
        if not _RECENT_PRICES_FILE.exists():
            print("No price cache found — skipping (run --backfill locally first)")
        else:
            import pandas as _pd
            df = _pd.read_parquet(_RECENT_PRICES_FILE)
            df["date"] = _pd.to_datetime(df["date"])
            updated = _append_local_prices(df)
            print(f"Price cache updated: {len(updated)} rows")
    elif args.backfill:
        import json as _json
        with open(_UNIVERSE_FILE) as f:
            constituents = _json.load(f)
        all_tickers = [v["ticker"] for v in constituents.values() if v.get("status") == "active"]
        _backfill_via_yfinance(list(set(all_tickers + ["SPY"])))
        print(f"Backfill complete. Cache saved to {_RECENT_PRICES_FILE}")
    else:
        dal = DataAccessLayer()
        quant = QuantModel()
        holdings = quant.run({"model": args.model, "eval_date": "2026-03-21"}, dal)

        print(f"\nTop picks ({args.model}):")
        for h in holdings:
            print(f"  {h.ticker:<6}  conviction={h.conviction:.3f}  {h.rationale}")
