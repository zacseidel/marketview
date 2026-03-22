"""
src/quant_research/train.py

Trains three models on the training split and evaluates each on the validation split.
Saves serialized artifacts to data/quant/artifacts/{knn,gbm,cluster}/.

Models:
  1. KNN analog matching (sklearn BallTree, K=50, distance-weighted)
  2. LightGBM regression (target = 20d forward log return)
  3. K-Means clustering (k=30, scored by cluster mean forward return)

Prints a comparison table at the end.

Usage:
    python -m src.quant_research.train [--model knn|gbm|cluster|all]
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import structlog

from src.quant_research.evaluate import evaluate_model, print_comparison
from src.quant_research.features import FEATURE_COLS

log = structlog.get_logger()

_FEATURES_FILE = Path("data/quant/features.parquet")
_ARTIFACTS_DIR = Path("data/quant/artifacts")

_KNN_K = 50
_KNN_TRAIN_SUBSAMPLE = 300_000   # subsample training rows for BallTree — 300K is representative and much faster
_KNN_QUERY_BATCH = 10_000        # query BallTree in batches so progress is visible
_KMEANS_K = 30
_GBM_PARAMS = {
    "n_estimators": 500,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "n_jobs": -1,
    "verbose": -1,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not _FEATURES_FILE.exists():
        raise FileNotFoundError(f"Run features.py first: {_FEATURES_FILE}")
    df = pd.read_parquet(_FEATURES_FILE)
    train = df[df["split"] == "train"].dropna(subset=FEATURE_COLS + ["fwd_log_ret_20d"])
    val = df[df["split"] == "val"].dropna(subset=FEATURE_COLS + ["fwd_log_ret_20d"])
    log.info("train.data_loaded", train_rows=len(train), val_rows=len(val))
    return train, val


def _save_artifact(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    log.info("train.artifact_saved", path=str(path))


def _fit_scaler(X_train: np.ndarray):
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.fit(X_train)
    return scaler


# ---------------------------------------------------------------------------
# Model 1: KNN
# ---------------------------------------------------------------------------

def train_knn(train: pd.DataFrame) -> None:
    from sklearn.neighbors import BallTree

    # Subsample training rows for BallTree — avoids 1.4M-point tree that makes
    # querying prohibitively slow. 300K rows captures the distribution well.
    if len(train) > _KNN_TRAIN_SUBSAMPLE:
        train = train.sample(n=_KNN_TRAIN_SUBSAMPLE, random_state=42)
        log.info("train.knn.subsampled", rows=len(train))

    log.info("train.knn.starting", rows=len(train), k=_KNN_K)

    X = train[FEATURE_COLS].values.astype(np.float32)
    y = train["fwd_log_ret_20d"].values.astype(np.float32)

    scaler = _fit_scaler(X)
    X_scaled = scaler.transform(X).astype(np.float32)

    log.info("train.knn.building_balltree")
    tree = BallTree(X_scaled, metric="euclidean")

    out_dir = _ARTIFACTS_DIR / "knn"
    out_dir.mkdir(parents=True, exist_ok=True)

    _save_artifact(scaler, out_dir / "scaler.pkl")
    _save_artifact(tree, out_dir / "balltree.pkl")

    # Save labels aligned to tree leaf order (same row order as X_scaled)
    np.save(out_dir / "labels.npy", y)

    log.info("train.knn.done")


def score_knn(val: pd.DataFrame) -> np.ndarray:
    from sklearn.neighbors import BallTree

    out_dir = _ARTIFACTS_DIR / "knn"
    with open(out_dir / "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open(out_dir / "balltree.pkl", "rb") as f:
        tree: BallTree = pickle.load(f)
    labels = np.load(out_dir / "labels.npy")

    X = val[FEATURE_COLS].values.astype(np.float32)
    X_scaled = scaler.transform(X).astype(np.float32)

    # Query in batches so progress is visible and memory stays manageable
    n = len(X_scaled)
    all_distances = np.empty((n, _KNN_K), dtype=np.float32)
    all_indices = np.empty((n, _KNN_K), dtype=np.int64)

    for start in range(0, n, _KNN_QUERY_BATCH):
        end = min(start + _KNN_QUERY_BATCH, n)
        d, idx = tree.query(X_scaled[start:end], k=_KNN_K)
        all_distances[start:end] = d
        all_indices[start:end] = idx
        log.info("train.knn.scoring_progress", done=end, total=n)

    weights = 1.0 / (all_distances + 1e-8)
    scores = (weights * labels[all_indices]).sum(axis=1) / weights.sum(axis=1)
    return scores


# ---------------------------------------------------------------------------
# Model 2: GBM (LightGBM)
# ---------------------------------------------------------------------------

def train_gbm(train: pd.DataFrame) -> None:
    import lightgbm as lgb

    log.info("train.gbm.starting", rows=len(train))

    X = train[FEATURE_COLS].values
    y = train["fwd_log_ret_20d"].values

    scaler = _fit_scaler(X)
    X_scaled = scaler.transform(X)

    model = lgb.LGBMRegressor(**_GBM_PARAMS)
    model.fit(X_scaled, y)

    out_dir = _ARTIFACTS_DIR / "gbm"
    _save_artifact(scaler, out_dir / "scaler.pkl")
    _save_artifact(model, out_dir / "model.pkl")

    # Log feature importances
    importances = sorted(
        zip(FEATURE_COLS, model.feature_importances_),
        key=lambda x: x[1],
        reverse=True,
    )
    log.info("train.gbm.feature_importances", importances=importances[:5])
    print("\nGBM Feature Importances:")
    for feat, imp in importances:
        print(f"  {feat:<20} {imp:>6}")

    log.info("train.gbm.done")


def score_gbm(val: pd.DataFrame) -> np.ndarray:
    out_dir = _ARTIFACTS_DIR / "gbm"
    with open(out_dir / "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open(out_dir / "model.pkl", "rb") as f:
        model = pickle.load(f)

    X = val[FEATURE_COLS].values
    X_scaled = scaler.transform(X)
    return model.predict(X_scaled)


# ---------------------------------------------------------------------------
# Model 3: K-Means Clustering
# ---------------------------------------------------------------------------

def train_cluster(train: pd.DataFrame) -> None:
    from sklearn.cluster import KMeans

    log.info("train.cluster.starting", rows=len(train), k=_KMEANS_K)

    X = train[FEATURE_COLS].values
    y = train["fwd_log_ret_20d"].values

    scaler = _fit_scaler(X)
    X_scaled = scaler.transform(X)

    kmeans = KMeans(n_clusters=_KMEANS_K, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)

    # Compute per-cluster statistics
    cluster_stats: dict[str, dict] = {}
    for cluster_id in range(_KMEANS_K):
        mask = labels == cluster_id
        cluster_rets = y[mask]
        n = int(mask.sum())
        mean_ret = float(cluster_rets.mean()) if n > 0 else 0.0
        std_ret = float(cluster_rets.std()) if n > 1 else 0.0
        hit_rate = float((cluster_rets > 0).mean()) if n > 0 else 0.0
        cluster_stats[str(cluster_id)] = {
            "n": n,
            "mean_fwd_ret": round(mean_ret, 6),
            "std_fwd_ret": round(std_ret, 6),
            "hit_rate": round(hit_rate, 3),
        }

    # Print cluster summary sorted by mean return
    print("\nCluster Summary (sorted by mean forward return):")
    print(f"  {'Cluster':>8} {'N':>8} {'MeanRet':>10} {'StdRet':>8} {'HitRate':>8}")
    for cid, stats in sorted(cluster_stats.items(), key=lambda x: x[1]["mean_fwd_ret"], reverse=True):
        print(f"  {cid:>8} {stats['n']:>8,} {stats['mean_fwd_ret']:>10.4f} "
              f"{stats['std_fwd_ret']:>8.4f} {stats['hit_rate']:>8.1%}")

    out_dir = _ARTIFACTS_DIR / "cluster"
    _save_artifact(scaler, out_dir / "scaler.pkl")
    _save_artifact(kmeans, out_dir / "kmeans.pkl")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "cluster_stats.json", "w") as f:
        json.dump(cluster_stats, f, indent=2)

    log.info("train.cluster.done")


def score_cluster(val: pd.DataFrame) -> np.ndarray:
    out_dir = _ARTIFACTS_DIR / "cluster"
    with open(out_dir / "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open(out_dir / "kmeans.pkl", "rb") as f:
        kmeans = pickle.load(f)
    with open(out_dir / "cluster_stats.json") as f:
        cluster_stats = json.load(f)

    X = val[FEATURE_COLS].values
    X_scaled = scaler.transform(X)
    cluster_ids = kmeans.predict(X_scaled)
    return np.array([cluster_stats[str(cid)]["mean_fwd_ret"] for cid in cluster_ids])


# Threshold used for cluster filter mode evaluation and live inference.
# Only buy stocks in clusters whose historical mean 20d forward return >= this value.
CLUSTER_SCORE_THRESHOLD = 0.09  # top 3 clusters only: 23 (16%), 12 (11%), 13 (11%)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def train_all(models: list[str] = ("knn", "gbm", "cluster")) -> None:
    train, val = _load_data()

    results = []

    if "knn" in models:
        train_knn(train)
        result = evaluate_model(val, score_knn, model_name="knn")
        results.append(result)

    if "gbm" in models:
        train_gbm(train)
        result = evaluate_model(val, score_gbm, model_name="gbm")
        results.append(result)

    if "cluster" in models:
        train_cluster(train)
        # Evaluate in two modes:
        # 1. Ranker (always top-N) — baseline
        result_ranker = evaluate_model(val, score_cluster, model_name="cluster_ranker")
        results.append(result_ranker)
        # 2. Filter (only buy from top-performing clusters) — threshold mode
        result_filter = evaluate_model(
            val,
            score_cluster,
            model_name="cluster_filter",
            min_score_threshold=CLUSTER_SCORE_THRESHOLD,
        )
        results.append(result_filter)

    print_comparison(results)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="all",
        choices=["knn", "gbm", "cluster", "all"],
        help="Which model(s) to train",
    )
    args = parser.parse_args()

    models = ["knn", "gbm", "cluster"] if args.model == "all" else [args.model]
    train_all(models=models)
