"""
src/collection/backfill_spy.py

One-time backfill: fetches SPY (and QQQ) daily bars from Polygon for the full
range covered by existing price files, then injects them into any file where
they are missing.

Safe to re-run — skips dates that already have the ticker.

Usage:
    python -m src.collection.backfill_spy
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import structlog
from dotenv import load_dotenv

from src.collection.polygon_client import PolygonClient

load_dotenv()
log = structlog.get_logger()

_PRICES_DIR = Path("data.nosync/prices")
_BACKFILL_TICKERS = ["SPY", "QQQ"]


def _bar_to_record(ticker: str, bar: dict) -> dict | None:
    """Convert a Polygon aggregate bar dict to our standard price record format."""
    t_ms = bar.get("t")
    if t_ms is None:
        return None
    date_str = datetime.fromtimestamp(t_ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
    o  = bar.get("o", 0.0)
    h  = bar.get("h", 0.0)
    lo = bar.get("l", 0.0)
    c  = bar.get("c", 0.0)
    return {
        "date":     date_str,
        "ticker":   ticker,
        "open":     o,
        "high":     h,
        "low":      lo,
        "close":    c,
        "volume":   bar.get("v", 0),
        "vwap":     bar.get("vw", 0.0),
        "ohlc_avg": round((o + h + lo + c) / 4.0, 4),
    }


def backfill(tickers: list[str] = _BACKFILL_TICKERS) -> None:
    price_files = sorted(f for f in _PRICES_DIR.glob("*.json") if f.stem[0].isdigit())
    if not price_files:
        log.error("backfill_spy.no_price_files")
        return

    from_date = price_files[0].stem
    to_date   = price_files[-1].stem
    log.info("backfill_spy.start", tickers=tickers, from_date=from_date, to_date=to_date)

    client = PolygonClient()

    for ticker in tickers:
        log.info("backfill_spy.fetching", ticker=ticker)
        bars = client.get_agg_bars(ticker, from_=from_date, to=to_date)
        log.info("backfill_spy.bars_received", ticker=ticker, count=len(bars))

        # Build {date_str: record} from Polygon response
        bar_by_date: dict[str, dict] = {}
        for bar in bars:
            record = _bar_to_record(ticker, bar)
            if record:
                bar_by_date[record["date"]] = record

        injected = 0
        for price_file in price_files:
            date_str = price_file.stem
            if date_str not in bar_by_date:
                continue  # no bar for this date (holiday, etc.)

            with open(price_file) as f:
                records: list[dict] = json.load(f)

            if any(r.get("ticker") == ticker for r in records):
                continue  # already present

            records.append(bar_by_date[date_str])
            tmp = price_file.with_suffix(".tmp")
            with open(tmp, "w") as f:
                json.dump(records, f, indent=2)
            tmp.replace(price_file)
            injected += 1

        log.info("backfill_spy.done", ticker=ticker, injected=injected,
                 skipped=len(price_files) - injected)
        print(f"  {ticker}: injected into {injected} files, {len(price_files) - injected} already had it")


if __name__ == "__main__":
    backfill()
    print("Backfill complete.")
