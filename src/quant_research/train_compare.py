"""
src/quant_research/train_compare.py

Controlled feature-set comparison: same model (XGBoost), same target
(fwd_log_ret_5d), same training cadence (Thursday-only), same eval framework.
Only the feature set varies.

Models compared:
  xgb_v1feat  — v1's 15 technical features  (no sector, no fundamentals)
  xgb_v3feat  — v3's 28 features            (+ buyback, earnings, SPY, sector 20d)
  xgb_v4feat  — v4's 33 features            (+ slope/R², dollar vol, earnings timing,
                                               sector 126d, in_sp500; no buyback/fundamentals)
  xgb_v4bfeat — v4 + earnings fundamentals  (+ eps_surprise, earn_ret_5d, ni_yoy_growth,
                                               rev_yoy_growth joined from v3 parquet;
                                               tests whether fundamentals add to v4's features)

All trained on Thursday-only rows, raw fwd_log_ret_5d target, XGBoost with
identical hyperparameters. Evaluated with eval_weekday=3 (Thursday) and
forward_days=5 so the Sharpe and IC metrics are directly comparable.

Prerequisite: rebuild parquets with fwd_log_ret_5d column:
    python -m src.quant_research.features       # adds fwd_log_ret_5d to features.parquet
    python -m src.quant_research.features_v3    # adds fwd_log_ret_5d to features_v3.parquet
    (features_v4.parquet already has fwd_log_ret_5d)

Usage:
    python -m src.quant_research.train_compare
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import structlog

from src.quant_research.evaluate import evaluate_model, print_comparison
from src.quant_research.features import FEATURE_COLS as FEATURE_COLS_V1
from src.quant_research.features_v3 import FEATURE_COLS_V3, CATEGORICAL_COLS_V3
from src.quant_research.features_v4 import FEATURE_COLS_V4, CATEGORICAL_COLS_V4

log = structlog.get_logger()

_FILES = {
    "v1": Path("data.nosync/quant/features.parquet"),
    "v3": Path("data.nosync/quant/features_v3.parquet"),
    "v4": Path("data.nosync/quant/features_v4.parquet"),
}
_TARGET = "fwd_log_ret_5d"
_OUTLIER_CAP = 0.5

# Earnings fundamental columns available in v3 parquet but not v4
_EARNINGS_FUNDAMENTAL_COLS = [
    "eps_surprise_pct", "earn_ret_5d", "ni_yoy_growth", "rev_yoy_growth",
]

# v3-only columns not present in v4 parquet at all (beyond earnings fundamentals)
_V3_ONLY_COLS = [
    "log_price",                                              # price level
    "buyback_pct_12m", "buyback_pct_1q",                    # capital return
    "days_to_next_earnings",                                  # v3 timing (0-100 scale)
    "sector_ret_20d", "sector_vs_spy_20d",                   # v3 sector (20d lookback)
    "sector_ret_rank", "sector_size",                        # v3 sector rank + size
]

# v4b = v4 features + earnings fundamentals from v3 (no buyback, no log_price)
FEATURE_COLS_V4B = FEATURE_COLS_V4 + _EARNINGS_FUNDAMENTAL_COLS

# v5 = full union of v3 + v4b: all 46 features; let XGBoost sort out redundancy
FEATURE_COLS_V5 = FEATURE_COLS_V4B + _V3_ONLY_COLS

# Identical hyperparameters for all three
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


def _load_and_filter(path: Path, feature_cols: list[str],
                     categorical_cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load a feature parquet, filter to Thursday-only rows with valid 5d target,
    encode categorical columns, return (train, val).
    """
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Rebuild it first:\n"
            f"  python -m src.quant_research.{path.stem}"
        )
    df = pd.read_parquet(path)

    if _TARGET not in df.columns:
        raise ValueError(
            f"{path.name} is missing '{_TARGET}'. "
            f"Rebuild with: python -m src.quant_research.{path.stem}"
        )

    # Thursday-only, valid target, outlier cap
    df = df[df["date"].dt.dayofweek == 3].copy()
    df = df.dropna(subset=[_TARGET])
    df = df[df[_TARGET].abs() <= _OUTLIER_CAP]

    # Encode categoricals
    if categorical_cols:
        all_cats = sorted(
            set(df["sector"].dropna().tolist()) if "sector" in df.columns else []
        )
        if all_cats:
            cat_type = pd.CategoricalDtype(categories=all_cats, ordered=False)
            df["sector"] = df["sector"].astype(cat_type)

    train = df[(df["split"] == "train") & (df["ticker"] != "SPY")].copy()
    val   = df[df["split"] == "val"].copy()
    log.info("compare.loaded", path=path.name,
             train_rows=len(train), val_rows=len(val),
             features=len(feature_cols) + len(categorical_cols))
    return train, val


def _train_and_score(
    name: str,
    train: pd.DataFrame,
    feature_cols: list[str],
    categorical_cols: list[str],
) -> callable:
    """Train one XGBoost model, return a scoring function."""
    import xgboost as xgb

    all_cols = feature_cols + categorical_cols
    X = train.reindex(columns=all_cols)
    y = train[_TARGET].values

    log.info("compare.training", model=name, rows=len(train), features=len(all_cols))
    model = xgb.XGBRegressor(**_XGB_PARAMS)
    model.fit(X, y)

    # Print top-10 importances
    importances = sorted(
        zip(all_cols, model.feature_importances_),
        key=lambda x: x[1], reverse=True,
    )
    print(f"\n{name} — top 10 feature importances:")
    for feat, imp in importances[:10]:
        print(f"  {feat:<30} {imp:.4f}")

    # Capture for closure
    _model = model
    _all_cols = all_cols
    _cat_cols = categorical_cols

    def score_fn(df: pd.DataFrame) -> np.ndarray:
        df2 = df.copy()
        if _cat_cols and "sector" in df2.columns:
            cats = [c.name for c in [_model.get_booster().feature_names] if False] or None
            # Reconstruct Categorical with same categories as training
            existing_cats = df2["sector"].cat.categories if hasattr(
                df2["sector"], "cat") else None
            if existing_cats is None:
                # Re-encode if not already categorical
                all_train_cats = sorted(df2["sector"].dropna().unique().tolist())
                df2["sector"] = pd.Categorical(df2["sector"],
                                               categories=all_train_cats)
        X2 = df2.reindex(columns=_all_cols)
        return _model.predict(X2)

    return score_fn


def _load_v4b_and_filter(
    feature_cols: list[str],
    categorical_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load v4 features and join the 4 earnings fundamental columns from v3.
    No parquet rebuild needed — merges on (ticker, date) after Thursday filter.
    """
    for key in ("v4", "v3"):
        if not _FILES[key].exists():
            raise FileNotFoundError(
                f"{_FILES[key]} not found. Rebuild with: "
                f"python -m src.quant_research.features{'_v4' if key=='v4' else '_v3'}"
            )

    df4 = pd.read_parquet(_FILES["v4"])
    # Only load the columns we need from v3 to keep memory low
    df3 = pd.read_parquet(
        _FILES["v3"],
        columns=["ticker", "date"] + _EARNINGS_FUNDAMENTAL_COLS,
    )

    df4["date"] = pd.to_datetime(df4["date"])
    df3["date"] = pd.to_datetime(df3["date"])

    # Thursday-only filter before merge
    df4 = df4[df4["date"].dt.dayofweek == 3].copy()
    df3 = df3[df3["date"].dt.dayofweek == 3].copy()

    # Merge earnings fundamentals onto v4 rows
    df = df4.merge(df3, on=["ticker", "date"], how="left")

    if _TARGET not in df.columns:
        raise ValueError(f"'{_TARGET}' missing from v4 parquet")

    df = df.dropna(subset=[_TARGET])
    df = df[df[_TARGET].abs() <= _OUTLIER_CAP]

    # Encode sector
    if categorical_cols and "sector" in df.columns:
        all_cats = sorted(df["sector"].dropna().unique().tolist())
        cat_type = pd.CategoricalDtype(categories=all_cats, ordered=False)
        df["sector"] = df["sector"].astype(cat_type)

    train = df[(df["split"] == "train") & (df["ticker"] != "SPY")].copy()
    val   = df[df["split"] == "val"].copy()
    log.info("compare.loaded_v4b",
             train_rows=len(train), val_rows=len(val),
             features=len(feature_cols) + len(categorical_cols))
    return train, val


def _make_score_fn(model, all_cols: list[str], categorical_cols: list[str],
                   sector_categories: list[str]) -> callable:
    """Return a score function that re-encodes sector against training categories."""
    def score_fn(df: pd.DataFrame) -> np.ndarray:
        df2 = df.copy()
        if categorical_cols and "sector" in df2.columns:
            cat_type = pd.CategoricalDtype(categories=sector_categories, ordered=False)
            df2["sector"] = df2["sector"].astype(cat_type)
        return model.predict(df2.reindex(columns=all_cols))
    return score_fn


def _analyze_pick_overlap(
    scored_vals: dict[str, pd.DataFrame],
    model_a: str,
    model_b: str,
    top_n: int = 20,
) -> None:
    """
    Compare top-N picks between two models on each eval Thursday.
    Reports average overlap count, Jaccard similarity, and rank correlation.
    """
    df_a = scored_vals.get(model_a)
    df_b = scored_vals.get(model_b)
    if df_a is None or df_b is None:
        print(f"\nOverlap skipped: missing scored val for {model_a} or {model_b}")
        return

    common_dates = sorted(set(df_a["date"]) & set(df_b["date"]))

    overlaps: list[int] = []
    jaccards: list[float] = []
    rank_corrs: list[float] = []

    for d in common_dates:
        day_a = df_a[(df_a["date"] == d) & (df_a["ticker"] != "SPY")]
        day_b = df_b[(df_b["date"] == d) & (df_b["ticker"] != "SPY")]

        if len(day_a) < top_n or len(day_b) < top_n:
            continue

        picks_a = set(day_a.nlargest(top_n, "score")["ticker"])
        picks_b = set(day_b.nlargest(top_n, "score")["ticker"])

        n_overlap = len(picks_a & picks_b)
        n_union = len(picks_a | picks_b)
        overlaps.append(n_overlap)
        jaccards.append(n_overlap / n_union if n_union > 0 else 0.0)

        # Rank correlation across the full universe on this date
        common_tickers = set(day_a["ticker"]) & set(day_b["ticker"])
        if len(common_tickers) >= 10:
            rank_a = day_a.set_index("ticker")["score"].reindex(common_tickers)
            rank_b = day_b.set_index("ticker")["score"].reindex(common_tickers)
            corr = float(rank_a.corr(rank_b, method="spearman"))
            if not np.isnan(corr):
                rank_corrs.append(corr)

    if not overlaps:
        print(f"\nOverlap: no common eval dates for {model_a} vs {model_b}")
        return

    print(f"\nPick overlap — {model_a} vs {model_b}  ({len(overlaps)} Thursdays, top {top_n})")
    print(f"  Avg stocks in common:   {np.mean(overlaps):.1f} / {top_n}  ({np.mean(overlaps)/top_n:.0%})")
    print(f"  Avg Jaccard similarity: {np.mean(jaccards):.3f}  (0=no overlap, 1=identical)")
    print(f"  Min / Max overlap:      {min(overlaps)} / {max(overlaps)}")
    if rank_corrs:
        print(f"  Avg rank correlation:   {np.mean(rank_corrs):.3f}  (full universe Spearman)")


def _load_v5_and_filter(
    feature_cols: list[str],
    categorical_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load v5 features: v4 parquet as base, merge ALL v3-only columns onto it.
    Inner join on (ticker, date) — only rows present in both parquets.
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

    # Thursday-only before merge
    df4 = df4[df4["date"].dt.dayofweek == 3].copy()
    df3 = df3[df3["date"].dt.dayofweek == 3].copy()

    # Inner join: only rows where both parquets have data
    df = df4.merge(df3, on=["ticker", "date"], how="inner")

    if _TARGET not in df.columns:
        raise ValueError(f"'{_TARGET}' missing from v4 parquet")

    df = df.dropna(subset=[_TARGET])
    df = df[df[_TARGET].abs() <= _OUTLIER_CAP]

    if categorical_cols and "sector" in df.columns:
        all_cats = sorted(df["sector"].dropna().unique().tolist())
        cat_type = pd.CategoricalDtype(categories=all_cats, ordered=False)
        df["sector"] = df["sector"].astype(cat_type)

    train = df[(df["split"] == "train") & (df["ticker"] != "SPY")].copy()
    val   = df[df["split"] == "val"].copy()
    log.info("compare.loaded_v5",
             train_rows=len(train), val_rows=len(val),
             features=len(feature_cols) + len(categorical_cols))
    return train, val


def _evaluate_ensemble(
    scored_vals: dict[str, pd.DataFrame],
    model_a: str,
    model_b: str,
    ensemble_name: str,
    target_col: str = "fwd_log_ret_5d",
) -> dict:
    """
    Evaluate a simple average-score ensemble of two already-scored models.
    Inner-joins on (date, ticker) — only rows where both models have scores.
    No retraining; scores are combined post-hoc.
    """
    cols_needed = ["date", "ticker", "score", target_col]
    df_a = scored_vals[model_a][cols_needed].copy()
    df_b = scored_vals[model_b][["date", "ticker", "score"]].copy()

    merged = df_a.merge(
        df_b.rename(columns={"score": "score_b"}),
        on=["date", "ticker"],
        how="inner",
    )
    merged["score_avg"] = (merged["score"] + merged["score_b"]) / 2.0

    score_lookup = dict(zip(
        zip(merged["date"], merged["ticker"]),
        merged["score_avg"],
    ))

    def score_fn(df: pd.DataFrame) -> np.ndarray:
        return np.array([
            score_lookup.get((d, t), np.nan)
            for d, t in zip(df["date"], df["ticker"])
        ])

    return evaluate_model(
        merged,
        score_fn,
        model_name=ensemble_name,
        forward_days=5,
        target_col=target_col,
        eval_weekday=3,
        top_n=20,
    )


def run_comparison() -> None:
    import xgboost as xgb

    results = []
    scored_vals: dict[str, pd.DataFrame] = {}  # model_name -> val df with "score" column

    configs = [
        ("xgb_v1feat",  "v1",  FEATURE_COLS_V1,  []),
        ("xgb_v3feat",  "v3",  FEATURE_COLS_V3,  CATEGORICAL_COLS_V3),
        ("xgb_v4feat",  "v4",  FEATURE_COLS_V4,  CATEGORICAL_COLS_V4),
        ("xgb_v4bfeat", "v4b", FEATURE_COLS_V4B, CATEGORICAL_COLS_V4),
        ("xgb_v5feat",  "v5",  FEATURE_COLS_V5,  CATEGORICAL_COLS_V4),
    ]

    for model_name, version, feat_cols, cat_cols in configs:
        print(f"\n{'='*60}")
        print(f"Training {model_name} ...")

        if version == "v4b":
            train, val = _load_v4b_and_filter(feat_cols, cat_cols)
        elif version == "v5":
            train, val = _load_v5_and_filter(feat_cols, cat_cols)
        else:
            train, val = _load_and_filter(_FILES[version], feat_cols, cat_cols)

        all_cols = feat_cols + cat_cols
        X_train = train.reindex(columns=all_cols)
        y_train = train[_TARGET].values

        # Derive and save sector categories for consistent inference
        sector_categories: list[str] = []
        if cat_cols and "sector" in train.columns:
            sector_categories = sorted(
                set(train["sector"].cat.categories.tolist())
                | set(val["sector"].cat.categories.tolist())
            )
            cat_type = pd.CategoricalDtype(categories=sector_categories, ordered=False)
            train = train.copy()
            val   = val.copy()
            train["sector"] = train["sector"].astype(cat_type)
            val["sector"]   = val["sector"].astype(cat_type)
            X_train = train.reindex(columns=all_cols)

        log.info("compare.training", model=model_name,
                 rows=len(train), features=len(all_cols))
        model = xgb.XGBRegressor(**_XGB_PARAMS)
        model.fit(X_train, y_train)

        # Feature importances
        importances = sorted(
            zip(all_cols, model.feature_importances_),
            key=lambda x: x[1], reverse=True,
        )
        print(f"\n{model_name} — top 10 feature importances:")
        for feat, imp in importances[:10]:
            print(f"  {feat:<30} {imp:.4f}")

        score_fn = _make_score_fn(model, all_cols, cat_cols, sector_categories)

        result = evaluate_model(
            val, score_fn,
            model_name=model_name,
            forward_days=5,
            target_col=_TARGET,
            eval_weekday=3,
            top_n=20,
        )
        results.append(result)

        # Collect scored val for pick-overlap analysis
        sv = val[val["date"].dt.dayofweek == 3].copy()
        sv["score"] = score_fn(sv)
        scored_vals[model_name] = sv

    print("\n\n" + "=" * 60)
    print("CONTROLLED COMPARISON — same model, target, cadence; feature set only varies")
    print("=" * 60)
    print_comparison(results)
    print("Feature counts:")
    for name, _, fcols, ccols in configs:
        print(f"  {name:<18} {len(fcols)+len(ccols)} features  "
              f"({len(fcols)} numeric, {len(ccols)} categorical)")

    # Pick-overlap analysis
    print("\n" + "=" * 60)
    print("PICK OVERLAP ANALYSIS")
    print("=" * 60)
    _analyze_pick_overlap(scored_vals, "xgb_v3feat",  "xgb_v4bfeat")
    _analyze_pick_overlap(scored_vals, "xgb_v3feat",  "xgb_v5feat")
    _analyze_pick_overlap(scored_vals, "xgb_v4bfeat", "xgb_v5feat")

    # Ensemble evaluation: average scores from v3 + v4b (best two independent models)
    print("\n" + "=" * 60)
    print("ENSEMBLE — average score of xgb_v3feat + xgb_v4bfeat (no retraining)")
    print("=" * 60)
    ensemble_result = _evaluate_ensemble(
        scored_vals, "xgb_v3feat", "xgb_v4bfeat", "ensemble_v3_v4b",
    )
    print_comparison([ensemble_result])


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    run_comparison()
