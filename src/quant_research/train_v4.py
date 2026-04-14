"""
src/quant_research/train_v4.py

Trains GBM v4 (XGBoost): Thursday-weekly data, cross-sectional rank-percentile
target, 5-day forward return horizon. No StandardScaler.

Key differences from v3:
  - XGBoost instead of LightGBM
  - Thursday-only training rows (non-overlapping 5-day targets)
  - Target: cross-sectional rank percentile of fwd_log_ret_5d within each date
  - No StandardScaler (tree models are invariant to monotonic transforms)
  - sector handled as pd.Categorical with XGBoost enable_categorical=True
  - Artifacts: model.pkl + sector_categories.json

Usage:
    python -m src.quant_research.train_v4
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import structlog

from src.quant_research.evaluate import evaluate_model, print_comparison, save_val_metrics
from src.quant_research.features_v4 import FEATURE_COLS_V4, CATEGORICAL_COLS_V4

log = structlog.get_logger()

_FEATURES_FILE = Path("data.nosync/quant/features_v4.parquet")
_ARTIFACTS_DIR = Path("data.nosync/quant/artifacts/gbm_v4")

_GBM_V4_PARAMS = {
    "n_estimators": 600,
    "max_depth": 5,
    "learning_rate": 0.03,
    "subsample": 0.8,
    "colsample_bytree": 0.7,
    "min_child_weight": 20,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "tree_method": "hist",
    "enable_categorical": True,
    "n_jobs": -1,
    "verbosity": 0,
    "random_state": 42,
}


def _load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not _FEATURES_FILE.exists():
        raise FileNotFoundError(f"Run features_v4.py first: {_FEATURES_FILE}")

    df = pd.read_parquet(_FEATURES_FILE)

    # Thursday-only: dayofweek 3
    df = df[df["date"].dt.dayofweek == 3].copy()
    log.info("train_v4.thursday_filter", thursday_rows=len(df))

    # Drop rows where target is missing or was capped to NaN during feature build
    df = df.dropna(subset=["fwd_log_ret_5d"])

    train = df[(df["split"] == "train") & (df["ticker"] != "SPY")].copy()
    val = df[df["split"] == "val"].copy()

    # Cross-sectional rank percentile within each Thursday's universe.
    # Computed independently on train and val (no cross-split leakage — each date
    # is self-contained). Range [0, 1]; 1.0 = best performer that week.
    train["rank_target"] = train.groupby("date")["fwd_log_ret_5d"].rank(pct=True)

    log.info("train_v4.data_loaded", train_rows=len(train), val_rows=len(val),
             rank_mean=round(float(train["rank_target"].mean()), 4))
    return train, val


def _encode_sector(
    train: pd.DataFrame,
    val: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """
    Encode sector as pd.Categorical using categories derived from the full dataset
    (train + val) so inference never sees an unknown category.
    Returns modified copies and the sorted category list (saved as artifact).
    """
    all_sectors = sorted(
        set(train["sector"].dropna().tolist()) | set(val["sector"].dropna().tolist())
    )
    cat_type = pd.CategoricalDtype(categories=all_sectors, ordered=False)
    train = train.copy()
    val = val.copy()
    train["sector"] = train["sector"].astype(cat_type)
    val["sector"] = val["sector"].astype(cat_type)
    return train, val, all_sectors


def train_gbm_v4(
    train: pd.DataFrame,
    sector_categories: list[str],
) -> None:
    """Train XGBoost v4 and save artifacts."""
    import xgboost as xgb

    log.info("train_v4.gbm_v4.starting",
             rows=len(train),
             features=len(FEATURE_COLS_V4) + len(CATEGORICAL_COLS_V4))

    all_cols = FEATURE_COLS_V4 + CATEGORICAL_COLS_V4
    X = train.reindex(columns=all_cols)
    y = train["rank_target"].values  # cross-sectional rank percentile [0, 1]

    model = xgb.XGBRegressor(**_GBM_V4_PARAMS)
    model.fit(X, y)

    _ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(_ARTIFACTS_DIR / "model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open(_ARTIFACTS_DIR / "sector_categories.json", "w") as f:
        json.dump(sector_categories, f, indent=2)

    # Feature importances
    importances = sorted(
        zip(all_cols, model.feature_importances_),
        key=lambda x: x[1], reverse=True,
    )
    log.info("train_v4.gbm_v4.feature_importances", top5=[f for f, _ in importances[:5]])
    print("\nGBM v4 Feature Importances (top 15):")
    for feat, imp in importances[:15]:
        print(f"  {feat:<30} {imp:>8.4f}")

    log.info("train_v4.gbm_v4.done", artifacts=str(_ARTIFACTS_DIR))


def score_gbm_v4(df: pd.DataFrame) -> np.ndarray:
    """Score a DataFrame with the trained v4 model. Returns predicted rank percentiles."""
    with open(_ARTIFACTS_DIR / "model.pkl", "rb") as f:
        model = pickle.load(f)
    with open(_ARTIFACTS_DIR / "sector_categories.json") as f:
        sector_categories = json.load(f)

    df = df.copy()
    cat_type = pd.CategoricalDtype(categories=sector_categories, ordered=False)
    df["sector"] = df["sector"].astype(cat_type)

    all_cols = FEATURE_COLS_V4 + CATEGORICAL_COLS_V4
    X = df.reindex(columns=all_cols)
    return model.predict(X)


def train_all_v4() -> None:
    train, val = _load_data()
    train, val, sector_categories = _encode_sector(train, val)

    train_gbm_v4(train, sector_categories)

    result = evaluate_model(
        val,
        score_gbm_v4,
        model_name="gbm_v4",
        forward_days=5,
        target_col="fwd_log_ret_5d",
        eval_weekday=3,   # Thursdays only
        top_n=20,
    )
    save_val_metrics(result)
    print_comparison([result])

    print(f"\nThursday training dates: {train['date'].nunique()} unique Thursdays")
    print(f"Thursday val dates:     {val['date'].nunique()} unique Thursdays")
    print(f"Rank target — mean: {train['rank_target'].mean():.4f}  "
          f"std: {train['rank_target'].std():.4f}  (should be ~0.5 / ~0.29)")


if __name__ == "__main__":
    train_all_v4()
