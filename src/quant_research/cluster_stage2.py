"""
src/quant_research/cluster_stage2.py

K-means clustering on Stage 2 trend template features.

Pipeline:
  1. Load features_stage2.parquet (build with features_stage2.py first)
  2. Drop NaN rows, StandardScaler on train features
  3. Silhouette-optimal k-means search (k=3..10) on a 50k-row sample
  4. Fit final model on all train rows with the best k
  5. Compute cluster profiles: centroid + forward return distributions (5d/10d/20d)
  6. Validate: do cluster return rankings from train hold in the val period?
  7. Save artifacts to data.nosync/quant/artifacts/stage2/

Artifacts:
  cluster_model.pkl      — fitted KMeans
  scaler.pkl             — fitted StandardScaler
  cluster_profiles.json  — centroids + train/val return stats per cluster

Usage:
    python -m src.quant_research.cluster_stage2
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import structlog
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from src.quant_research.features_stage2 import STAGE2_FEATURE_COLS

log = structlog.get_logger()

_FEATURES_FILE = Path("data.nosync/quant/features_stage2.parquet")
_ARTIFACTS_DIR = Path("data.nosync/quant/artifacts/stage2")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not _FEATURES_FILE.exists():
        raise FileNotFoundError(f"Run features_stage2.py first: {_FEATURES_FILE}")

    df = pd.read_parquet(_FEATURES_FILE)
    df = df.dropna(subset=STAGE2_FEATURE_COLS)

    train = df[df["split"] == "train"].copy()
    val = df[df["split"] == "val"].copy()

    log.info(
        "cluster_stage2.data_loaded",
        train_rows=len(train),
        val_rows=len(val),
        tickers=df["ticker"].nunique(),
    )
    return train, val


# ---------------------------------------------------------------------------
# K selection via silhouette score
# ---------------------------------------------------------------------------

def _find_best_k(X_scaled: np.ndarray, k_range: range) -> tuple[int, float]:
    """
    Silhouette score search over k_range. Evaluates on a 50k-row sample
    to keep computation tractable (silhouette is O(n²)).
    """
    n_sample = min(50_000, len(X_scaled))
    rng = np.random.default_rng(42)
    idx = rng.choice(len(X_scaled), size=n_sample, replace=False)
    X_sample = X_scaled[idx]

    best_k, best_score = k_range.start, -1.0
    scores: dict[int, float] = {}

    for k in k_range:
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = km.fit_predict(X_sample)
        score = float(silhouette_score(X_sample, labels))
        scores[k] = round(score, 4)
        log.info("cluster_stage2.silhouette", k=k, score=round(score, 4))
        if score > best_score:
            best_score = score
            best_k = k

    print("\nSilhouette scores by k:")
    for k, s in scores.items():
        marker = "  <-- best" if k == best_k else ""
        print(f"  k={k}: {s:.4f}{marker}")

    return best_k, best_score


# ---------------------------------------------------------------------------
# Cluster characterization
# ---------------------------------------------------------------------------

def _return_stats(series: pd.Series) -> dict:
    s = series.dropna()
    if len(s) == 0:
        return {"n": 0, "mean": None, "median": None, "std": None, "hit_rate": None}
    return {
        "n": int(len(s)),
        "mean": round(float(s.mean()), 5),
        "median": round(float(s.median()), 5),
        "std": round(float(s.std()), 5),
        "hit_rate": round(float((s > 0).mean()), 3),
    }


def _cluster_profile(df: pd.DataFrame, cluster_id: int) -> dict:
    sub = df[df["cluster"] == cluster_id]
    return {
        "n_rows": int(len(sub)),
        "fwd_5d": _return_stats(sub["fwd_log_ret_5d"]),
        "fwd_10d": _return_stats(sub["fwd_log_ret_10d"]),
        "fwd_20d": _return_stats(sub["fwd_log_ret_20d"]),
    }


# ---------------------------------------------------------------------------
# Main clustering pipeline
# ---------------------------------------------------------------------------

def _winsorize(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    cols: list[str],
    lo: float = 0.02,
    hi: float = 0.98,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Clip each feature to the [lo, hi] percentile range computed on train only.
    Prevents outlier rows (e.g. pct_above_52wk_low = 100,000%) from dominating
    k-means distance calculations and pulling clusters off the real distribution.
    """
    train_df = train_df.copy()
    val_df = val_df.copy()
    for col in cols:
        lo_val = float(train_df[col].quantile(lo))
        hi_val = float(train_df[col].quantile(hi))
        train_df[col] = train_df[col].clip(lo_val, hi_val)
        val_df[col] = val_df[col].clip(lo_val, hi_val)
        log.info("cluster_stage2.winsorize", feature=col,
                 lo=round(lo_val, 3), hi=round(hi_val, 3))
    return train_df, val_df


def run_clustering(k_range: range = range(3, 11)) -> None:
    train, val = _load_data()

    log.info("cluster_stage2.winsorizing", lo="2%", hi="98%")
    train, val = _winsorize(train, val, STAGE2_FEATURE_COLS)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(train[STAGE2_FEATURE_COLS].values)
    X_val = scaler.transform(val[STAGE2_FEATURE_COLS].values)

    log.info("cluster_stage2.silhouette_search", k_min=k_range.start, k_max=k_range.stop - 1)
    best_k, sil_score = _find_best_k(X_train, k_range)
    log.info("cluster_stage2.best_k", k=best_k, silhouette=round(sil_score, 4))

    log.info("cluster_stage2.fitting_final", k=best_k, n_train=len(X_train))
    km = KMeans(n_clusters=best_k, n_init=20, random_state=42)
    train = train.copy()
    train["cluster"] = km.fit_predict(X_train)
    val = val.copy()
    val["cluster"] = km.predict(X_val)

    # Centroids in winsorized-but-unscaled space for interpretability
    centroids_orig = scaler.inverse_transform(km.cluster_centers_)

    # Return stats per cluster on train and val
    train_profiles = {i: _cluster_profile(train, i) for i in range(best_k)}
    val_profiles = {i: _cluster_profile(val, i) for i in range(best_k)}

    # Rank clusters by 20d mean train return (descending = best first)
    def safe_mean_20d(profiles: dict, i: int) -> float:
        v = profiles[i]["fwd_20d"]["mean"]
        return v if v is not None else -999.0

    train_order = sorted(range(best_k), key=lambda i: safe_mean_20d(train_profiles, i), reverse=True)
    val_order = sorted(range(best_k), key=lambda i: safe_mean_20d(val_profiles, i), reverse=True)

    train_rank = {cid: rank for rank, cid in enumerate(train_order)}
    val_rank = {cid: rank for rank, cid in enumerate(val_order)}

    # Spearman rank correlation: do train return rankings hold in val?
    tr_ranks = [train_rank[i] for i in range(best_k)]
    vl_ranks = [val_rank[i] for i in range(best_k)]
    rank_corr = float(spearmanr(tr_ranks, vl_ranks).correlation)

    # Build output structure
    clusters = []
    for i in range(best_k):
        centroid = {
            feat: round(float(centroids_orig[i, j]), 4)
            for j, feat in enumerate(STAGE2_FEATURE_COLS)
        }
        clusters.append({
            "id": i,
            "centroid": centroid,
            "train_rank": int(train_rank[i]),
            "val_rank": int(val_rank[i]),
            "train": train_profiles[i],
            "val": val_profiles[i],
        })
    clusters.sort(key=lambda c: c["train_rank"])

    profiles = {
        "k": best_k,
        "silhouette_score": round(sil_score, 4),
        "rank_correlation_train_vs_val": round(rank_corr, 4),
        "features": STAGE2_FEATURE_COLS,
        "clusters": clusters,
    }

    _print_results(clusters, best_k, sil_score, rank_corr)

    _ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(_ARTIFACTS_DIR / "cluster_model.pkl", "wb") as f:
        pickle.dump(km, f)
    with open(_ARTIFACTS_DIR / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    with open(_ARTIFACTS_DIR / "cluster_profiles.json", "w") as f:
        json.dump(profiles, f, indent=2)

    log.info("cluster_stage2.saved", dir=str(_ARTIFACTS_DIR))
    print(f"\nArtifacts saved to {_ARTIFACTS_DIR}/")


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def _print_results(
    clusters: list[dict], k: int, sil_score: float, rank_corr: float
) -> None:
    width = 100
    print(f"\n{'=' * width}")
    print(
        f"Stage 2 Cluster Analysis  "
        f"k={k}  silhouette={sil_score:.4f}  "
        f"rank_corr(train↔val)={rank_corr:.4f}"
    )
    print(f"{'=' * width}")
    print(
        f"{'Rank':<5} {'ID':<4} {'N(tr)':>8} {'N(val)':>8}  "
        f"{'5d(tr)':>8} {'10d(tr)':>8} {'20d(tr)':>8} {'hit%(tr)':>9}  "
        f"{'20d(val)':>9} {'hit%(val)':>10}"
    )
    print("-" * width)

    for c in clusters:
        tr = c["train"]
        vl = c["val"]
        print(
            f"{c['train_rank']:<5} {c['id']:<4} {tr['n_rows']:>8,} {vl['n_rows']:>8,}  "
            f"{(tr['fwd_5d']['mean'] or 0.0):>8.4f} "
            f"{(tr['fwd_10d']['mean'] or 0.0):>8.4f} "
            f"{(tr['fwd_20d']['mean'] or 0.0):>8.4f} "
            f"{(tr['fwd_20d']['hit_rate'] or 0.0):>9.1%}  "
            f"{(vl['fwd_20d']['mean'] or 0.0):>9.4f} "
            f"{(vl['fwd_20d']['hit_rate'] or 0.0):>10.1%}"
        )

    print(f"\n{'Top cluster (train rank 0) centroid':}")
    top = clusters[0]
    for feat, val in top["centroid"].items():
        print(f"  {feat:<28} {val:>10.3f}")

    print(f"\n{'Bottom cluster (train rank -1) centroid':}")
    bottom = clusters[-1]
    for feat, val in bottom["centroid"].items():
        print(f"  {feat:<28} {val:>10.3f}")


if __name__ == "__main__":
    run_clustering(k_range=range(3, 11))
