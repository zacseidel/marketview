"""
src/universe/ingestion.py

Daily price ingestion from Polygon's grouped daily bars endpoint.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import structlog

from src.collection.polygon_client import PolygonClient
from src.collection.queue import WorkQueue

log = structlog.get_logger()

_PRICES_DIR = Path("data/prices")
_UNIVERSE_FILE = Path("data/universe/constituents.json")
_SPLIT_THRESHOLD = 0.40  # ±40% single-day move flags a potential split
_BENCHMARK_TICKERS = {"SPY", "QQQ"}  # always stored alongside universe tickers


@dataclass
class IngestResult:
    date: str
    records_written: int
    tickers_flagged_for_split: list[str]
    skipped: bool = False


def _prev_trading_day(d: date) -> date:
    """Step back one calendar day, skipping weekends."""
    d -= timedelta(days=1)
    while d.weekday() >= 5:  # Saturday=5, Sunday=6
        d -= timedelta(days=1)
    return d


def _load_universe() -> dict:
    """Load constituents.json as {ticker: record}."""
    if not _UNIVERSE_FILE.exists():
        log.warning("ingestion.no_universe_file", path=str(_UNIVERSE_FILE))
        return {}
    with open(_UNIVERSE_FILE) as f:
        return json.load(f)


def _load_prev_closes(prev_date_str: str) -> dict[str, float]:
    """Load previous day's close prices as {ticker: close}."""
    prev_file = _PRICES_DIR / f"{prev_date_str}.json"
    if not prev_file.exists():
        return {}
    with open(prev_file) as f:
        records = json.load(f)
    return {r["ticker"]: r["close"] for r in records if "close" in r}


def ingest_daily(target_date: str | None = None, client: PolygonClient | None = None) -> IngestResult:
    """
    Fetch and store grouped daily prices for target_date (default: previous trading day).

    - Filters to universe members only
    - Computes ohlc_avg per bar
    - Detects potential splits (±40% vs previous close)
    - Queues split_correction tasks for flagged tickers
    - Idempotent: skips if output file already exists
    """
    _PRICES_DIR.mkdir(parents=True, exist_ok=True)

    # Resolve target date
    if target_date is None:
        target_date = _prev_trading_day(date.today()).isoformat()

    out_file = _PRICES_DIR / f"{target_date}.json"

    if out_file.exists():
        log.info("ingestion.skipped", date=target_date, reason="file_exists")
        with open(out_file) as f:
            existing = json.load(f)
        return IngestResult(date=target_date, records_written=len(existing), tickers_flagged_for_split=[], skipped=True)

    universe = _load_universe()
    if not universe:
        log.error("ingestion.empty_universe")
        return IngestResult(date=target_date, records_written=0, tickers_flagged_for_split=[])

    universe_tickers = {t for t, r in universe.items() if r.get("status") == "active"} | _BENCHMARK_TICKERS
    log.info("ingestion.starting", date=target_date, universe_size=len(universe_tickers))

    if client is None:
        client = PolygonClient()

    # Fetch grouped daily (1 API call for entire market)
    raw = client.get_grouped_daily(target_date)
    bars = raw.get("results", [])

    if not bars:
        log.warning("ingestion.no_bars", date=target_date, status=raw.get("status"))
        # Write an empty file so backfill doesn't retry this date (likely a market holiday)
        with open(out_file, "w") as f:
            json.dump([], f)
        return IngestResult(date=target_date, records_written=0, tickers_flagged_for_split=[])

    log.info("ingestion.bars_received", date=target_date, total_bars=len(bars))

    # Load previous closes for split detection
    prev_date_str = _prev_trading_day(date.fromisoformat(target_date)).isoformat()
    prev_closes = _load_prev_closes(prev_date_str)

    records = []
    split_flagged: list[str] = []

    for bar in bars:
        ticker = bar.get("T", "")
        if ticker not in universe_tickers:
            continue

        o = bar.get("o", 0.0)
        h = bar.get("h", 0.0)
        lo = bar.get("l", 0.0)
        c = bar.get("c", 0.0)
        v = bar.get("v", 0)
        vw = bar.get("vw", 0.0)

        ohlc_avg = (o + h + lo + c) / 4.0

        record = {
            "date": target_date,
            "ticker": ticker,
            "open": o,
            "high": h,
            "low": lo,
            "close": c,
            "volume": v,
            "vwap": vw,
            "ohlc_avg": round(ohlc_avg, 4),
        }
        records.append(record)

        # Split detection
        prev_close = prev_closes.get(ticker)
        if prev_close and prev_close > 0 and c > 0:
            move = abs(c / prev_close - 1.0)
            if move >= _SPLIT_THRESHOLD:
                log.warning(
                    "ingestion.split_flagged",
                    ticker=ticker,
                    date=target_date,
                    prev_close=prev_close,
                    new_close=c,
                    move_pct=round(move * 100, 1),
                )
                split_flagged.append(ticker)

    # Write output
    with open(out_file, "w") as f:
        json.dump(records, f, indent=2)

    log.info(
        "ingestion.complete",
        date=target_date,
        records=len(records),
        split_flags=len(split_flagged),
    )

    # Queue split correction tasks
    if split_flagged:
        queue = WorkQueue()
        for ticker in split_flagged:
            queue.enqueue(
                task_type="split_correction",
                ticker=ticker,
                requested_date=target_date,
                requested_by="ingestion",
                priority="high",
            )

    return IngestResult(
        date=target_date,
        records_written=len(records),
        tickers_flagged_for_split=split_flagged,
    )


def _missing_weekdays(lookback_days: int = 7) -> list[str]:
    """Return ISO date strings for weekdays in the past lookback_days that have no price file."""
    today = date.today()
    missing = []
    for offset in range(1, lookback_days + 1):
        d = today - timedelta(days=offset)
        if d.weekday() >= 5:  # skip weekends
            continue
        if not (_PRICES_DIR / f"{d.isoformat()}.json").exists():
            missing.append(d.isoformat())
    return missing


if __name__ == "__main__":
    import sys
    from dotenv import load_dotenv

    load_dotenv()
    target = sys.argv[1] if len(sys.argv) > 1 else None

    client = PolygonClient()

    # Backfill any missing weekdays in the past 7 days before the primary ingest
    missing = _missing_weekdays()
    if target:
        # Explicit date passed — skip backfill, just run that date
        missing = []
    for d in sorted(missing):
        log.info("ingestion.backfill", date=d)
        ingest_daily(d, client=client)

    result = ingest_daily(target, client=client)
    print(f"Ingested {result.records_written} records for {result.date}")
    if result.tickers_flagged_for_split:
        print(f"Split flags: {result.tickers_flagged_for_split}")
