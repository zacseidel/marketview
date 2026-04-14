"""
src/quant_research/build_all.py

One-shot script: build all feature matrices then train all active models.

Dependency order
────────────────
raw_prices.parquet          (must already exist — run download.py first)
  ├─ features_v3.parquet    → train_v3  → artifacts/gbm_v3
  ├─ features_v4.parquet ─┐
  │  features_v3.parquet ─┴─ features_v5.parquet → train_v5 → artifacts/gbm_v5
  └─ features_v6.parquet
       └─ features_v7.parquet → train_v7 → artifacts/gbm_v7

Usage:
    python -m src.quant_research.build_all           # build features + train
    python -m src.quant_research.build_all --train-only   # skip feature builds
    python -m src.quant_research.build_all --features-only  # skip training
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import structlog

log = structlog.get_logger()

_RAW_PRICES = Path("data.nosync/quant/raw_prices.parquet")


def _step(label: str) -> None:
    log.info("build_all.step", step=label)
    print(f"\n{'='*60}\n  {label}\n{'='*60}")


def _elapsed(t0: float) -> str:
    s = int(time.time() - t0)
    return f"{s//60}m{s%60:02d}s"


def build_features() -> None:
    _step("features_v3  (raw_prices → features_v3.parquet)")
    from src.quant_research.features_v3 import build_features_v3
    build_features_v3()

    _step("features_v4  (raw_prices → features_v4.parquet)")
    from src.quant_research.features_v4 import build_features_v4
    build_features_v4()

    _step("features_v5  (v3 + v4 → features_v5.parquet)")
    from src.quant_research.features_v5 import build_features_v5
    build_features_v5()

    _step("features_v6  (raw_prices → features_v6.parquet)")
    from src.quant_research.features_v6 import build_features_v6
    build_features_v6()

    _step("features_v7  (v6 → features_v7.parquet)")
    from src.quant_research.features_v7 import build_features_v7
    build_features_v7()


def train_models() -> None:
    _step("train_v3  → artifacts/gbm_v3")
    from src.quant_research.train_v3 import train_all_v3
    train_all_v3()

    _step("train_v5  → artifacts/gbm_v5")
    from src.quant_research.train_v5 import train_all_v5
    train_all_v5()

    _step("train_v7  → artifacts/gbm_v7")
    from src.quant_research.train_v7 import train_all_v7
    train_all_v7()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build all quant features and train all active models.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--train-only", action="store_true",
                       help="Skip feature builds; use existing parquet files")
    group.add_argument("--features-only", action="store_true",
                       help="Build features only; skip training")
    args = parser.parse_args()

    if not _RAW_PRICES.exists():
        print(f"ERROR: {_RAW_PRICES} not found. Run 'python -m src.quant_research.download' first.")
        sys.exit(1)

    t0 = time.time()

    if not args.train_only:
        build_features()

    if not args.features_only:
        train_models()

    print(f"\nDone in {_elapsed(t0)}.")
    print("Next: python -m src.quant_research.train_compare_windows")


if __name__ == "__main__":
    main()
