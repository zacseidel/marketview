"""
src/collection/convert_prices_to_parquet.py

One-time migration: consolidates data.nosync/prices/*.json → data.nosync/prices/prices.parquet.
Run once locally before deploying the Parquet-aware DAL.

    python -m src.collection.convert_prices_to_parquet
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

import pandas as pd
import structlog

log = structlog.get_logger()

_PRICES_DIR = Path("data.nosync/prices")
_OUTPUT = _PRICES_DIR / "prices.parquet"
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}\.json$")


def convert() -> None:
    files = sorted(f for f in _PRICES_DIR.glob("*.json") if _DATE_RE.match(f.name))
    log.info("convert.start", files=len(files))

    records: list[dict] = []
    skipped = 0
    for i, f in enumerate(files):
        if f.stat().st_size == 0:
            skipped += 1
            continue
        if i % 25 == 0:
            log.info("convert.progress", file=i, total=len(files), pct=f"{i/len(files)*100:.0f}%")
        t0 = time.time()
        with open(f) as fp:
            data = fp.read()
        elapsed = time.time() - t0
        if elapsed > 0.5:
            log.warning("slow_read", file=str(f), seconds=round(elapsed, 2))
        records.extend(json.loads(data))
    if skipped:
        log.info("convert.skipped_empty", count=skipped)

    log.info("convert.building_dataframe", records=len(records))
    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    df = df.drop_duplicates(subset=["date", "ticker"], keep="last")
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    df.to_parquet(_OUTPUT, index=False)
    log.info("convert.done", rows=len(df), output=str(_OUTPUT))


if __name__ == "__main__":
    convert()
