"""
src/quant_research/train_v6.py

Trains GBM v6 (XGBoost): Thursday-weekly data, raw log-return target,
5-day forward return horizon.

v6 feature set (44 total: 43 numeric + sector categorical):
  - Trend: SMA ratios + slopes (R² dropped)
  - Momentum: 6 log-return windows, sharpe, accel, ATH/52w proximity
  - Volatility: 20d/60d
  - Liquidity: log dollar vol, relative dollar vol
  - Earnings timing: days since/until (v4-style), days to next (v3-style)
  - Market regime: SPY ret 5d/20d, vol 20d, pct above SMA200/50
  - Market breadth: breadth_sma200, breadth_change_20d
  - Sector: 20d/60d/126d returns, stock vs sector 20d/126d, sector rank, vs SPY
  - Universe: in_sp500
  - Earnings fundamentals: EPS surprise, earn ret, NI/rev YoY growth

Usage:
    python -m src.quant_research.train_v6
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import structlog

from src.quant_research.evaluate import evaluate_model, print_comparison, save_val_metrics
from src.quant_research.features_v6 import FEATURE_COLS_V6, CATEGORICAL_COLS_V6

log = structlog.get_logger()

_FEATURES_FILE = Path("data.nosync/quant/features_v6.parquet")
_ARTIFACTS_DIR = Path("data.nosync/quant/artifacts/gbm_v6")

_XGB_PARAMS = {
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
            f"Run features_v6.py first: {_FEATURES_FILE}\n"
            f"  python -m src.quant_research.features_v6"
        )

    df = pd.read_parquet(_FEATURES_FILE)
    df["date"] = pd.to_datetime(df["date"])

    # Thursday-only: non-overlapping 5-day target windows
    df = df[df["date"].dt.dayofweek == 3].copy()
    log.info("train_v6.thursday_filter", thursday_rows=len(df))

    df = df.dropna(subset=["fwd_log_ret_5d"])
    df = df[df["fwd_log_ret_5d"].abs() <= 0.5]

    train = df[(df["split"] == "train") & (df["ticker"] != "SPY")].copy()
    val   = df[df["split"] == "val"].copy()

    log.info("train_v6.data_loaded", train_rows=len(train), val_rows=len(val))
    return train, val


def _encode_sector(
    train: pd.DataFrame,
    val: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """
    Encode sector as pd.Categorical using categories from train ∪ val
    so inference never sees an unknown category.
    """
    all_sectors = sorted(
        set(train["sector"].dropna().tolist()) | set(val["sector"].dropna().tolist())
    )
    cat_type = pd.CategoricalDtype(categories=all_sectors, ordered=False)
    train = train.copy()
    val   = val.copy()
    train["sector"] = train["sector"].astype(cat_type)
    val["sector"]   = val["sector"].astype(cat_type)
    return train, val, all_sectors


def train_gbm_v6(
    train: pd.DataFrame,
    sector_categories: list[str],
) -> None:
    """Train XGBoost v6 on raw log-return target and save artifacts."""
    import xgboost as xgb

    all_cols = FEATURE_COLS_V6 + CATEGORICAL_COLS_V6
    log.info("train_v6.gbm_v6.starting",
             rows=len(train), features=len(all_cols))

    X = train.reindex(columns=all_cols)
    y = train["fwd_log_ret_5d"].values

    model = xgb.XGBRegressor(**_XGB_PARAMS)
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
    log.info("train_v6.gbm_v6.feature_importances", top5=[f for f, _ in importances[:5]])
    print("\nGBM v6 Feature Importances (top 20):")
    for feat, imp in importances[:20]:
        print(f"  {feat:<35} {imp:>8.4f}")

    log.info("train_v6.gbm_v6.done", artifacts=str(_ARTIFACTS_DIR))


def score_gbm_v6(df: pd.DataFrame) -> np.ndarray:
    """Score a DataFrame with the trained v6 model. Returns predicted 5d log returns."""
    with open(_ARTIFACTS_DIR / "model.pkl", "rb") as f:
        model = pickle.load(f)
    with open(_ARTIFACTS_DIR / "sector_categories.json") as f:
        sector_categories = json.load(f)

    df = df.copy()
    cat_type = pd.CategoricalDtype(categories=sector_categories, ordered=False)
    df["sector"] = df["sector"].astype(cat_type)

    all_cols = FEATURE_COLS_V6 + CATEGORICAL_COLS_V6
    return model.predict(df.reindex(columns=all_cols))


def train_all_v6() -> None:
    train, val = _load_data()
    train, val, sector_categories = _encode_sector(train, val)

    train_gbm_v6(train, sector_categories)

    result = evaluate_model(
        val,
        score_gbm_v6,
        model_name="gbm_v6",
        forward_days=5,
        target_col="fwd_log_ret_5d",
        eval_weekday=3,
        top_n=20,
    )
    save_val_metrics(result)
    print_comparison([result])

    print(f"\nThursday training dates: {train['date'].nunique()} unique Thursdays")
    print(f"Thursday val dates:     {val['date'].nunique()} unique Thursdays")
    print(f"Target — mean: {train['fwd_log_ret_5d'].mean():.4f}  "
          f"std: {train['fwd_log_ret_5d'].std():.4f}")


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    train_all_v6()
