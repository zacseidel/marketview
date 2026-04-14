"""
src/quant_research/train_compare_windows.py

Multi-window, multi-target-type comparison.
Same feature sets, same XGBoost hyperparameters across all runs.

Matrix:
  feature sets  × target windows × target type = up to 30 training runs
  ─────────────────────────────────────────────────────────────────────────
  v1feat (15)   × 5d / 10d / 20d × raw [/ rank]
  v3feat (28)   × 5d / 10d / 20d × raw [/ rank]
  v5feat (45)   × 5d / 10d / 20d × raw [/ rank]
  v6feat (44)   × 5d / 10d / 20d × raw [/ rank]
  v7feat (47)   × 5d / 10d / 20d × raw [/ rank]

Target types:
  raw  — XGBoost regresses directly on fwd_log_ret_Nd  (default)
  rank — XGBoost regresses on cross-sectional rank percentile of fwd_log_ret_Nd
         (computed per eval date; range [0,1]; ~U(0,1) by construction)

Evaluation always uses raw log returns (IC/Sharpe/DclHit), so results are
directly comparable across target types. IC/ICIR are cadence-normalizing and
the primary cross-window comparables.

All runs:
  - Thursday-only rows (non-overlapping 5d windows; 10d/20d have label overlap
    but cadence is held constant for apples-to-apples training set sizes)
  - Identical XGBoost hyperparameters
  - Evaluated at Thursday cadence with matching forward_days

Prerequisites:
    python -m src.quant_research.features        # adds fwd_log_ret_10d to v1 parquet
    python -m src.quant_research.features_v3     # adds fwd_log_ret_20d to v3 parquet
    python -m src.quant_research.features_v4     # adds fwd_log_ret_10d/20d to v4 parquet
    python -m src.quant_research.features_v5     # rebuilds v5 from v3+v4 inner join
    python -m src.quant_research.features_v6     # builds v6 from raw_prices.parquet
    python -m src.quant_research.features_v7     # extends v6 with 3 earnings signals

Usage:
    python -m src.quant_research.train_compare_windows           # 15 raw-target runs (default)
    python -m src.quant_research.train_compare_windows --rank    # 15 rank-target runs only
    python -m src.quant_research.train_compare_windows --both    # all 30 runs
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import structlog

from src.quant_research.evaluate import evaluate_model, print_comparison
from src.quant_research.features import FEATURE_COLS as FEATURE_COLS_V1
from src.quant_research.features_v3 import FEATURE_COLS_V3, CATEGORICAL_COLS_V3
from src.quant_research.features_v5 import FEATURE_COLS_V5, CATEGORICAL_COLS_V5
from src.quant_research.features_v6 import FEATURE_COLS_V6, CATEGORICAL_COLS_V6
from src.quant_research.features_v7 import FEATURE_COLS_V7, CATEGORICAL_COLS_V7

log = structlog.get_logger()

_FILES = {
    "v1": Path("data.nosync/quant/features.parquet"),
    "v3": Path("data.nosync/quant/features_v3.parquet"),
    "v4": Path("data.nosync/quant/features_v4.parquet"),
    "v5": Path("data.nosync/quant/features_v5.parquet"),
    "v6": Path("data.nosync/quant/features_v6.parquet"),
    "v7": Path("data.nosync/quant/features_v7.parquet"),
}

# v3-only columns needed when building v5 on-the-fly from v4+v3 parquets
_EARNINGS_FUNDAMENTAL_COLS = [
    "eps_surprise_pct", "earn_ret_5d", "ni_yoy_growth", "rev_yoy_growth",
]
_V3_ONLY_COLS = [
    "log_price",
    "buyback_pct_12m", "buyback_pct_1q",
    "days_to_next_earnings",
    "sector_ret_20d", "sector_vs_spy_20d",
    "sector_ret_rank", "sector_size",
]

# Outlier caps per window — applied as NaN rather than row drop so rows remain
# valid for other windows when iterating
_OUTLIER_CAPS = {
    "fwd_log_ret_5d":  0.5,
    "fwd_log_ret_10d": 1.0,
    "fwd_log_ret_20d": 1.5,
}

_WINDOWS = [
    ("5d",  "fwd_log_ret_5d",   5),
    ("10d", "fwd_log_ret_10d", 10),
    ("20d", "fwd_log_ret_20d", 20),
]

_TARGET_TYPES = ["raw", "rank"]  # default run: raw only (--rank or --both to add rank)

# Identical hyperparameters across all 18 runs
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


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_thursday_df(
    path: Path,
    categorical_cols: list[str],
    target_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load a parquet, filter to Thursday rows with a valid target,
    encode sector categoricals, return (train, val).
    """
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Rebuild with:\n"
            f"  python -m src.quant_research.{path.stem}"
        )

    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])

    if target_col not in df.columns:
        raise ValueError(
            f"{path.name} is missing '{target_col}'. "
            f"Rebuild with: python -m src.quant_research.{path.stem}"
        )

    df = df[df["date"].dt.dayofweek == 3].copy()
    df = df.dropna(subset=[target_col])
    cap = _OUTLIER_CAPS[target_col]
    df = df[df[target_col].abs() <= cap]

    if categorical_cols and "sector" in df.columns:
        all_cats = sorted(df["sector"].dropna().unique().tolist())
        df["sector"] = df["sector"].astype(
            pd.CategoricalDtype(categories=all_cats, ordered=False)
        )

    train = df[(df["split"] == "train") & (df["ticker"] != "SPY")].copy()
    val   = df[df["split"] == "val"].copy()
    log.info("windows.loaded", path=path.name, target=target_col,
             train_rows=len(train), val_rows=len(val))
    return train, val


def _load_v5_thursday_df(
    categorical_cols: list[str],
    target_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build v5 dataset on-the-fly: inner-join v4 + v3-only columns,
    filtered to Thursday rows with a valid target.
    """
    for key in ("v4", "v3"):
        if not _FILES[key].exists():
            raise FileNotFoundError(
                f"{_FILES[key]} not found. Rebuild with: "
                f"python -m src.quant_research.features{'_v4' if key == 'v4' else '_v3'}"
            )

    extra_cols = _EARNINGS_FUNDAMENTAL_COLS + _V3_ONLY_COLS
    df4 = pd.read_parquet(_FILES["v4"])
    df3 = pd.read_parquet(
        _FILES["v3"],
        columns=["ticker", "date"] + extra_cols,
    )

    df4["date"] = pd.to_datetime(df4["date"])
    df3["date"] = pd.to_datetime(df3["date"])

    df4 = df4[df4["date"].dt.dayofweek == 3].copy()
    df3 = df3[df3["date"].dt.dayofweek == 3].copy()

    df = df4.merge(df3, on=["ticker", "date"], how="inner")

    if target_col not in df.columns:
        raise ValueError(f"'{target_col}' missing from v4 parquet — rebuild features_v4")

    df = df.dropna(subset=[target_col])
    cap = _OUTLIER_CAPS[target_col]
    df = df[df[target_col].abs() <= cap]

    if categorical_cols and "sector" in df.columns:
        all_cats = sorted(df["sector"].dropna().unique().tolist())
        df["sector"] = df["sector"].astype(
            pd.CategoricalDtype(categories=all_cats, ordered=False)
        )

    train = df[(df["split"] == "train") & (df["ticker"] != "SPY")].copy()
    val   = df[df["split"] == "val"].copy()
    log.info("windows.loaded_v5", target=target_col,
             train_rows=len(train), val_rows=len(val))
    return train, val


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _apply_rank_target(train: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """
    Replace target_col with its cross-sectional rank percentile within each
    eval date. Range [0, 1]; ~U(0, 1) by construction.

    Computed on train only — val rank percentiles are not needed since
    evaluate_model uses raw returns for IC/Sharpe regardless of training target.
    """
    train = train.copy()
    train[target_col] = train.groupby("date")[target_col].rank(pct=True)
    return train


def _make_score_fn(model, all_cols: list[str], categorical_cols: list[str],
                   sector_categories: list[str]):
    def score_fn(df: pd.DataFrame) -> np.ndarray:
        df2 = df.copy()
        if categorical_cols and "sector" in df2.columns:
            df2["sector"] = df2["sector"].astype(
                pd.CategoricalDtype(categories=sector_categories, ordered=False)
            )
        return model.predict(df2.reindex(columns=all_cols))
    return score_fn


def _train_one(
    run_name: str,
    train: pd.DataFrame,
    val: pd.DataFrame,
    feature_cols: list[str],
    categorical_cols: list[str],
    target_col: str,
    forward_days: int,
    target_type: str,
) -> dict:
    import xgboost as xgb

    all_cols = feature_cols + categorical_cols

    # Align sector categories across train+val before any modification
    sector_categories: list[str] = []
    if categorical_cols and "sector" in train.columns:
        sector_categories = sorted(
            set(train["sector"].cat.categories.tolist())
            | set(val["sector"].cat.categories.tolist())
        )
        cat_type = pd.CategoricalDtype(categories=sector_categories, ordered=False)
        train = train.copy()
        val   = val.copy()
        train["sector"] = train["sector"].astype(cat_type)
        val["sector"]   = val["sector"].astype(cat_type)

    # For rank target: replace raw return with cross-sectional rank percentile
    # on train only. Val target stays raw — evaluation uses raw returns.
    if target_type == "rank":
        train = _apply_rank_target(train, target_col)
        log.info("windows.rank_target_applied", run=run_name,
                 mean=round(float(train[target_col].mean()), 4),
                 std=round(float(train[target_col].std()), 4))

    X_train = train.reindex(columns=all_cols)
    y_train = train[target_col].values

    log.info("windows.training", run=run_name,
             rows=len(train), features=len(all_cols),
             target=target_col, target_type=target_type)
    model = xgb.XGBRegressor(**_XGB_PARAMS)
    model.fit(X_train, y_train)

    # Top-10 feature importances
    importances = sorted(
        zip(all_cols, model.feature_importances_),
        key=lambda x: x[1], reverse=True,
    )
    print(f"\n{run_name} — top 10 feature importances:")
    for feat, imp in importances[:10]:
        print(f"  {feat:<30} {imp:.4f}")

    score_fn = _make_score_fn(model, all_cols, categorical_cols, sector_categories)

    # Evaluation always uses raw target_col (not ranks) for IC/Sharpe
    result = evaluate_model(
        val, score_fn,
        model_name=run_name,
        forward_days=forward_days,
        target_col=target_col,
        eval_weekday=3,
        top_n=20,
    )
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_window_comparison(run_raw: bool = True, run_rank: bool = True) -> None:
    import xgboost as xgb  # noqa: F401 — early import to surface missing dep

    feat_configs = [
        ("v1feat",  "v1",  FEATURE_COLS_V1,  []),
        ("v3feat",  "v3",  FEATURE_COLS_V3,  CATEGORICAL_COLS_V3),
        ("v5feat",  "v5",  FEATURE_COLS_V5,  CATEGORICAL_COLS_V5),
        ("v6feat",  "v6",  FEATURE_COLS_V6,  CATEGORICAL_COLS_V6),
        ("v7feat",  "v7",  FEATURE_COLS_V7,  CATEGORICAL_COLS_V7),
    ]
    target_types = [t for t in _TARGET_TYPES
                    if (t == "raw" and run_raw) or (t == "rank" and run_rank)]

    # Collect results along multiple axes for flexible summary views
    results_by_type_window: dict[tuple[str, str], list[dict]] = {}
    results_by_type_feat:   dict[tuple[str, str], list[dict]] = {}
    all_results: list[dict] = []

    for target_type in target_types:
        for feat_label, version, feat_cols, cat_cols in feat_configs:
            for window_label, target_col, fwd_days in _WINDOWS:
                run_name = f"{feat_label}_{window_label}_{target_type}"
                print(f"\n{'='*65}")
                print(f"Training {run_name} ...")

                if version == "v5":
                    train, val = _load_v5_thursday_df(cat_cols, target_col)
                else:
                    train, val = _load_thursday_df(
                        _FILES[version], cat_cols, target_col
                    )

                result = _train_one(
                    run_name, train, val, feat_cols, cat_cols,
                    target_col, fwd_days, target_type,
                )

                key_tw = (target_type, window_label)
                key_tf = (target_type, feat_label)
                results_by_type_window.setdefault(key_tw, []).append(result)
                results_by_type_feat.setdefault(key_tf, []).append(result)
                all_results.append(result)

    # -----------------------------------------------------------------------
    # Summary tables
    # -----------------------------------------------------------------------

    # Group by target type → window: see if rank beats raw within each horizon
    for target_type in target_types:
        print(f"\n\n{'='*70}")
        print(f"TARGET TYPE: {target_type.upper()}  — results by window")
        print("=" * 70)
        for window_label, _, _ in _WINDOWS:
            key = (target_type, window_label)
            if key in results_by_type_window:
                print(f"\n--- {window_label} ---")
                print_comparison(results_by_type_window[key])

    # Group by target type → feature set: see if more features help per type
    for target_type in target_types:
        print(f"\n\n{'='*70}")
        print(f"TARGET TYPE: {target_type.upper()}  — results by feature set")
        print("=" * 70)
        for feat_label, _, feat_cols, cat_cols in feat_configs:
            key = (target_type, feat_label)
            if key in results_by_type_feat:
                print(f"\n--- {feat_label} ({len(feat_cols)+len(cat_cols)} features) ---")
                print_comparison(results_by_type_feat[key])

    # Raw vs rank head-to-head: same feature set + window, different target type
    if run_raw and run_rank:
        print(f"\n\n{'='*70}")
        print("RAW vs RANK HEAD-TO-HEAD — same feature set + window")
        print("=" * 70)
        for window_label, _, _ in _WINDOWS:
            print(f"\n--- {window_label} window ---")
            combined = []
            for target_type in ["raw", "rank"]:
                combined.extend(results_by_type_window.get((target_type, window_label), []))
            print_comparison(combined)

    # Full grid
    print(f"\n\n{'='*70}")
    print(f"FULL GRID — all {len(all_results)} runs")
    print("=" * 70)
    print_comparison(all_results)

    print("\nFeature counts:")
    for feat_label, _, fcols, ccols in feat_configs:
        print(f"  {feat_label:<12} {len(fcols)+len(ccols)} features  "
              f"({len(fcols)} numeric, {len(ccols)} categorical)")
    print(f"\nTarget types run: {', '.join(target_types)}")


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--rank", action="store_true", help="Run rank-target runs only")
    group.add_argument("--both", action="store_true", help="Run both raw and rank targets")
    args = parser.parse_args()

    # Default: raw only. --rank = rank only. --both = both.
    run_raw  = not args.rank              # True unless --rank
    run_rank = args.rank or args.both     # False unless --rank or --both

    run_window_comparison(run_raw=run_raw, run_rank=run_rank)
