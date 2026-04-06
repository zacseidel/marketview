"""
src/quant_research/train_v2.py

Trains GBM v2: 28 numeric features + sector categorical, predicting 10-day forward
log return. Trained on the most recent 2-year observation window.

Saves artifacts to data.nosync/quant/artifacts/gbm_v2/:
  scaler.pkl           — StandardScaler fitted on numeric features
  model.pkl            — LightGBM regressor
  sector_mapping.json  — {sector_string: int} encoding

Usage:
    python -m src.quant_research.train_v2
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import structlog

from src.quant_research.evaluate import evaluate_model, print_comparison, save_val_metrics
from src.quant_research.features_v2 import FEATURE_COLS_V2, CATEGORICAL_COLS_V2
from src.quant_research.train import _save_artifact

log = structlog.get_logger()

_FEATURES_FILE = Path("data.nosync/quant/features_v2.parquet")
_ARTIFACTS_DIR = Path("data.nosync/quant/artifacts/gbm_v2")

_GBM_V2_PARAMS = {
    "n_estimators": 300,
    "max_depth": 5,
    "learning_rate": 0.05,
    "subsample": 0.7,
    "colsample_bytree": 0.7,
    "min_child_samples": 50,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "n_jobs": -1,
    "verbose": -1,
}


def encode_sector(df: pd.DataFrame, sector_mapping: dict[str, int] | None = None) -> tuple[np.ndarray, dict[str, int]]:
    """
    Integer-encode the 'sector' column.
    If sector_mapping is None, builds one from df (training mode).
    Unknown sectors (inference-time) map to -1.
    Returns (encoded_array, mapping_dict).
    """
    if sector_mapping is None:
        unique_sectors = sorted(df["sector"].dropna().unique())
        sector_mapping = {s: i for i, s in enumerate(unique_sectors)}

    encoded = df["sector"].map(sector_mapping).fillna(-1).astype(int).values
    return encoded, sector_mapping


def _fit_scaler_nan_safe(X: np.ndarray):
    """
    Fit StandardScaler on non-NaN values per feature.
    Uses median-imputed data for fitting so NaN features don't corrupt mean/std.
    NaN values are preserved in transform output — LightGBM handles them natively.
    """
    from sklearn.preprocessing import StandardScaler

    col_medians = np.nanmedian(X, axis=0)
    X_for_fit = np.where(np.isnan(X), col_medians, X)
    scaler = StandardScaler()
    scaler.fit(X_for_fit)
    return scaler


def _load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not _FEATURES_FILE.exists():
        raise FileNotFoundError(f"Run features_v2.py first: {_FEATURES_FILE}")
    df = pd.read_parquet(_FEATURES_FILE)
    df = df.dropna(subset=["fwd_log_ret_10d"])

    # Exclude SPY from training — it's an index ETF, not a selectable stock.
    # SPY rows are kept in val so evaluate.py can use them as the benchmark.
    train = df[(df["split"] == "train") & (df["ticker"] != "SPY")].copy()
    val = df[df["split"] == "val"].copy()
    log.info("train_v2.data_loaded", train_rows=len(train), val_rows=len(val))
    return train, val


def train_gbm_v2(train: pd.DataFrame) -> dict[str, int]:
    """Train GBM v2. Returns sector_mapping (also saved to disk)."""
    import lightgbm as lgb

    log.info("train_v2.gbm_v2.starting", rows=len(train))

    numeric_features = [c for c in FEATURE_COLS_V2 if c in train.columns]
    X_numeric = train.reindex(columns=numeric_features).values.astype(np.float64)

    scaler = _fit_scaler_nan_safe(X_numeric)
    X_scaled = scaler.transform(X_numeric)  # NaN values preserved in output

    sector_encoded, sector_mapping = encode_sector(train)
    X = np.column_stack([X_scaled, sector_encoded])
    y = train["fwd_log_ret_10d"].values

    feature_names = numeric_features + ["sector"]
    cat_idx = len(numeric_features)  # sector is the last column

    model = lgb.LGBMRegressor(**_GBM_V2_PARAMS)
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
    log.info("train_v2.gbm_v2.feature_importances", top5=importances[:5])
    print("\nGBM v2 Feature Importances:")
    for feat, imp in importances:
        print(f"  {feat:<25} {imp:>6}")

    log.info("train_v2.gbm_v2.done")
    return sector_mapping


def score_gbm_v2(df: pd.DataFrame) -> np.ndarray:
    """Score rows using trained GBM v2 artifacts. Handles NaN features natively."""
    with open(_ARTIFACTS_DIR / "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open(_ARTIFACTS_DIR / "model.pkl", "rb") as f:
        model = pickle.load(f)
    with open(_ARTIFACTS_DIR / "sector_mapping.json") as f:
        sector_mapping = json.load(f)

    numeric_features = [c for c in FEATURE_COLS_V2 if c in df.columns]
    X_numeric = df.reindex(columns=numeric_features).values.astype(np.float64)
    X_scaled = scaler.transform(X_numeric)

    sector_encoded, _ = encode_sector(df, sector_mapping)
    X = np.column_stack([X_scaled, sector_encoded])
    return model.predict(X)


def train_all_v2() -> None:
    train, val = _load_data()
    train_gbm_v2(train)
    result = evaluate_model(
        val,
        score_gbm_v2,
        model_name="gbm_v2",
        forward_days=10,
        target_col="fwd_log_ret_10d",
    )
    save_val_metrics(result)
    print_comparison([result])


if __name__ == "__main__":
    train_all_v2()
