"""
src/quant_research/train_v5.py

Trains GBM v5 (XGBoost): full union feature set, Thursday-weekly,
raw fwd_log_ret_5d target. Same hyperparameters as v4.

v5 features = v4b (v4 + earnings fundamentals) + v3-only additions:
  log_price, buyback_pct_12m/1q, days_to_next_earnings (v3 style),
  sector_ret_20d/vs_spy/rank/size (v3 sector 20d features)
  — 45 numeric + sector categorical = 46 features total

Prerequisite:
    python -m src.quant_research.features_v5

Usage:
    python -m src.quant_research.train_v5
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import structlog

from src.quant_research.evaluate import evaluate_model, print_comparison, save_val_metrics
from src.quant_research.features_v5 import FEATURE_COLS_V5, CATEGORICAL_COLS_V5

log = structlog.get_logger()

_FEATURES_FILE = Path("data.nosync/quant/features_v5.parquet")
_ARTIFACTS_DIR = Path("data.nosync/quant/artifacts/gbm_v5")

_GBM_V5_PARAMS = {
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
        raise FileNotFoundError(
            f"Run features_v5.py first: {_FEATURES_FILE}\n"
            "  python -m src.quant_research.features_v5"
        )

    df = pd.read_parquet(_FEATURES_FILE)

    # Thursday-only: dayofweek 3
    df = df[df["date"].dt.dayofweek == 3].copy()
    log.info("train_v5.thursday_filter", thursday_rows=len(df))

    df = df.dropna(subset=["fwd_log_ret_5d"])
    df = df[df["fwd_log_ret_5d"].abs() <= 0.5]

    train = df[(df["split"] == "train") & (df["ticker"] != "SPY")].copy()
    val = df[df["split"] == "val"].copy()
    log.info("train_v5.data_loaded", train_rows=len(train), val_rows=len(val))
    return train, val


def _encode_sector(
    train: pd.DataFrame,
    val: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    all_sectors = sorted(
        set(train["sector"].dropna().tolist()) | set(val["sector"].dropna().tolist())
    )
    cat_type = pd.CategoricalDtype(categories=all_sectors, ordered=False)
    train = train.copy()
    val = val.copy()
    train["sector"] = train["sector"].astype(cat_type)
    val["sector"] = val["sector"].astype(cat_type)
    return train, val, all_sectors


def train_gbm_v5(train: pd.DataFrame, sector_categories: list[str]) -> None:
    import xgboost as xgb

    log.info("train_v5.starting",
             rows=len(train),
             features=len(FEATURE_COLS_V5) + len(CATEGORICAL_COLS_V5))

    all_cols = FEATURE_COLS_V5 + CATEGORICAL_COLS_V5
    X = train.reindex(columns=all_cols)
    y = train["fwd_log_ret_5d"].values

    model = xgb.XGBRegressor(**_GBM_V5_PARAMS)
    model.fit(X, y)

    _ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(_ARTIFACTS_DIR / "model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open(_ARTIFACTS_DIR / "sector_categories.json", "w") as f:
        json.dump(sector_categories, f, indent=2)

    importances = sorted(
        zip(all_cols, model.feature_importances_),
        key=lambda x: x[1], reverse=True,
    )
    print("\nGBM v5 Feature Importances (top 20):")
    for feat, imp in importances[:20]:
        print(f"  {feat:<30} {imp:>8.4f}")

    log.info("train_v5.done", artifacts=str(_ARTIFACTS_DIR))


def score_gbm_v5(df: pd.DataFrame) -> np.ndarray:
    with open(_ARTIFACTS_DIR / "model.pkl", "rb") as f:
        model = pickle.load(f)
    with open(_ARTIFACTS_DIR / "sector_categories.json") as f:
        sector_categories = json.load(f)

    df = df.copy()
    cat_type = pd.CategoricalDtype(categories=sector_categories, ordered=False)
    df["sector"] = df["sector"].astype(cat_type)

    all_cols = FEATURE_COLS_V5 + CATEGORICAL_COLS_V5
    X = df.reindex(columns=all_cols)
    return model.predict(X)


def train_all_v5() -> None:
    train, val = _load_data()
    train, val, sector_categories = _encode_sector(train, val)

    train_gbm_v5(train, sector_categories)

    result = evaluate_model(
        val,
        score_gbm_v5,
        model_name="gbm_v5",
        forward_days=5,
        target_col="fwd_log_ret_5d",
        eval_weekday=3,
        top_n=20,
    )
    save_val_metrics(result)
    print_comparison([result])

    print(f"\nThursday training dates: {train['date'].nunique()} unique Thursdays")
    print(f"Thursday val dates:     {val['date'].nunique()} unique Thursdays")
    print(f"Target — mean: {train['fwd_log_ret_5d'].mean():.5f}  "
          f"std: {train['fwd_log_ret_5d'].std():.5f}")


if __name__ == "__main__":
    train_all_v5()
