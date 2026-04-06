"""
src/quant_research/compare.py

Head-to-head comparison of all GBM variants on a consistent 10-day cadence.
Trains a "v1_10d" control in-memory (v1 features, 10d target) to isolate
whether the v2/v3 Sharpe gains come from the new features vs just the shorter
prediction horizon.

Models compared:
  gbm_v1      — 15 technical features, 20d target (published baseline)
  gbm_v1_10d  — 15 technical features, 10d target (control: horizon change only)
  gbm_v2      — 27 technical features, 10d target (adds buyback/earnings/SPY/sector)
  gbm_v3      — 28 technical features, 10d target (v2 + log_ret_756d)

Usage:
    python -m src.quant_research.compare
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import structlog
from pathlib import Path
from sklearn.preprocessing import StandardScaler

from src.quant_research.evaluate import evaluate_model, print_comparison
from src.quant_research.features import FEATURE_COLS
from src.quant_research.train import score_gbm
from src.quant_research.train_v2 import score_gbm_v2, _fit_scaler_nan_safe
from src.quant_research.train_v3 import score_gbm_v3

log = structlog.get_logger()

_GBM_PARAMS = {
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


def _train_v1_10d(train: pd.DataFrame) -> tuple[StandardScaler, object]:
    """Train v1 features on 10d target. Returns (scaler, model)."""
    import lightgbm as lgb

    X = train.reindex(columns=FEATURE_COLS).values.astype(np.float64)
    scaler = _fit_scaler_nan_safe(X)
    X_scaled = scaler.transform(X)
    y = train["fwd_log_ret_10d"].values

    model = lgb.LGBMRegressor(**_GBM_PARAMS)
    model.fit(X_scaled, y)

    importances = sorted(
        zip(FEATURE_COLS, model.feature_importances_),
        key=lambda x: x[1], reverse=True,
    )
    print("\nGBM v1_10d Feature Importances:")
    for feat, imp in importances:
        print(f"  {feat:<25} {imp:>6}")

    return scaler, model


def main() -> None:
    # Load feature matrices
    df1 = pd.read_parquet("data.nosync/quant/features.parquet")
    df3 = pd.read_parquet("data.nosync/quant/features_v3.parquet")
    df2 = pd.read_parquet("data.nosync/quant/features_v2.parquet")

    val1 = df1[df1["split"] == "val"].copy()
    val2 = df2[df2["split"] == "val"].copy()
    val3 = df3[df3["split"] == "val"].copy()

    # Train v1_10d control in-memory on features_v3 train split
    # (features_v3 has v1's 15 technical cols + fwd_log_ret_10d)
    train3 = df3[(df3["split"] == "train") & (df3["ticker"] != "SPY")].copy()
    train3 = train3.dropna(subset=["fwd_log_ret_10d"])
    log.info("compare.training_v1_10d", rows=len(train3))
    scaler, model_v1_10d = _train_v1_10d(train3)

    def score_v1_10d(df: pd.DataFrame) -> np.ndarray:
        X = df.reindex(columns=FEATURE_COLS).values.astype(np.float64)
        return model_v1_10d.predict(scaler.transform(X))

    # Evaluate all four
    r1 = evaluate_model(val1, score_gbm,      model_name="gbm_v1",     forward_days=20, target_col="fwd_log_ret_20d")
    r1c = evaluate_model(val3, score_v1_10d,  model_name="gbm_v1_10d", forward_days=10, target_col="fwd_log_ret_10d")
    r2 = evaluate_model(val2, score_gbm_v2,   model_name="gbm_v2",     forward_days=10, target_col="fwd_log_ret_10d")
    r3 = evaluate_model(val3, score_gbm_v3,   model_name="gbm_v3",     forward_days=10, target_col="fwd_log_ret_10d")

    print_comparison([r1, r1c, r2, r3])

    # Annotate what each model adds
    print("What each model adds over the previous:")
    print(f"  v1 → v1_10d : target horizon 20d → 10d (same 15 features)")
    print(f"  v1_10d → v2 : +buyback, +earnings, +SPY regime, +sector  (drops log_ret_756d)")
    print(f"  v2 → v3     : +log_ret_756d (3yr momentum)")


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    main()
