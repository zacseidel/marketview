"""
src/quant_research/train_v3.py

Trains GBM v3: 28 numeric features + sector categorical, predicting 20-day forward
log return. Direct comparison to v1 (same horizon, same rebalance cadence).

Saves artifacts to data/quant/artifacts/gbm_v3/.

Usage:
    python -m src.quant_research.train_v3
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import structlog

from src.quant_research.evaluate import evaluate_model, print_comparison, save_val_metrics
from src.quant_research.features_v3 import FEATURE_COLS_V3, CATEGORICAL_COLS_V3
from src.quant_research.train import _save_artifact
from src.quant_research.train_v2 import _fit_scaler_nan_safe, encode_sector

log = structlog.get_logger()

_FEATURES_FILE = Path("data/quant/features_v3.parquet")
_ARTIFACTS_DIR = Path("data/quant/artifacts/gbm_v3")

# Same params as v1 — full 10yr training set is large enough for deeper trees
_GBM_V3_PARAMS = {
    "n_estimators": 500,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_samples": 50,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "n_jobs": -1,
    "verbose": -1,
}


def _load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not _FEATURES_FILE.exists():
        raise FileNotFoundError(f"Run features_v3.py first: {_FEATURES_FILE}")
    df = pd.read_parquet(_FEATURES_FILE)
    df = df.dropna(subset=["fwd_log_ret_10d"])

    train = df[(df["split"] == "train") & (df["ticker"] != "SPY")].copy()
    val = df[df["split"] == "val"].copy()
    log.info("train_v3.data_loaded", train_rows=len(train), val_rows=len(val))
    return train, val


def train_gbm_v3(train: pd.DataFrame) -> dict[str, int]:
    """Train GBM v3. Returns sector_mapping."""
    import lightgbm as lgb

    log.info("train_v3.gbm_v3.starting", rows=len(train))

    numeric_features = [c for c in FEATURE_COLS_V3 if c in train.columns]
    X_numeric = train.reindex(columns=numeric_features).values.astype(np.float64)

    scaler = _fit_scaler_nan_safe(X_numeric)
    X_scaled = scaler.transform(X_numeric)

    sector_encoded, sector_mapping = encode_sector(train)
    X = np.column_stack([X_scaled, sector_encoded])
    y = train["fwd_log_ret_10d"].values

    feature_names = numeric_features + ["sector"]
    cat_idx = len(numeric_features)

    model = lgb.LGBMRegressor(**_GBM_V3_PARAMS)
    model.fit(X, y, feature_name=feature_names, categorical_feature=[cat_idx])

    _ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    _save_artifact(scaler, _ARTIFACTS_DIR / "scaler.pkl")
    _save_artifact(model, _ARTIFACTS_DIR / "model.pkl")
    with open(_ARTIFACTS_DIR / "sector_mapping.json", "w") as f:
        json.dump(sector_mapping, f, indent=2)

    importances = sorted(
        zip(feature_names, model.feature_importances_),
        key=lambda x: x[1], reverse=True,
    )
    log.info("train_v3.gbm_v3.feature_importances", top5=importances[:5])
    print("\nGBM v3 Feature Importances:")
    for feat, imp in importances:
        print(f"  {feat:<25} {imp:>6}")

    log.info("train_v3.gbm_v3.done")
    return sector_mapping


def score_gbm_v3(df: pd.DataFrame) -> np.ndarray:
    with open(_ARTIFACTS_DIR / "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open(_ARTIFACTS_DIR / "model.pkl", "rb") as f:
        model = pickle.load(f)
    with open(_ARTIFACTS_DIR / "sector_mapping.json") as f:
        sector_mapping = json.load(f)

    numeric_features = [c for c in FEATURE_COLS_V3 if c in df.columns]
    X_numeric = df.reindex(columns=numeric_features).values.astype(np.float64)
    X_scaled = scaler.transform(X_numeric)

    sector_encoded, _ = encode_sector(df, sector_mapping)
    X = np.column_stack([X_scaled, sector_encoded])
    return model.predict(X)


def train_all_v3() -> None:
    train, val = _load_data()
    train_gbm_v3(train)
    result = evaluate_model(
        val,
        score_gbm_v3,
        model_name="gbm_v3",
        forward_days=10,
        target_col="fwd_log_ret_10d",
    )
    save_val_metrics(result)
    print_comparison([result])


if __name__ == "__main__":
    train_all_v3()
