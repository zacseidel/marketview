"""
src/collection/process_queue.py

Queue processor — runs pending tasks from the work queue.
Called by process-queue.yml workflow (weekdays 7 PM ET).

Handles:
  - split_correction: confirm and correct split-adjusted prices
  - ticker_details: fetch and admit new tickers to the universe
  - options_chain: fetch options data (Phase 3)

Usage:
    python -m src.collection.process_queue
"""

from __future__ import annotations

import structlog

from src.collection.queue import WorkQueue
from src.collection.polygon_client import PolygonClient

log = structlog.get_logger()

# Max tasks per run — keeps within rate limit budget (5/min × ~50 min)
_MAX_TASKS_PER_RUN = 50


def process_split_corrections(queue: WorkQueue, client: PolygonClient) -> int:
    from src.universe.splits import confirm_and_correct_split

    tasks = queue.get_pending(task_type="split_correction")
    processed = 0
    for task in tasks:
        if processed >= _MAX_TASKS_PER_RUN:
            break
        log.info("process_queue.split_correction", ticker=task.ticker, date=task.requested_date)
        try:
            confirm_and_correct_split(task.ticker, task.requested_date, client=client)
            queue.mark_complete(task.task_id, data_path=f"data.nosync/splits/{task.ticker}_{task.requested_date}.json")
        except Exception as exc:
            log.warning("process_queue.split_error", ticker=task.ticker, error=str(exc))
            queue.mark_failed(task.task_id, str(exc))
        processed += 1
    return processed


def process_ticker_details(queue: WorkQueue, client: PolygonClient) -> int:
    from src.universe.ticker_details import fetch_and_admit_new_tickers

    tasks = queue.get_pending(task_type="ticker_details")
    if not tasks:
        return 0

    tickers = [t.ticker for t in tasks[:_MAX_TASKS_PER_RUN]]
    log.info("process_queue.ticker_details", count=len(tickers))
    try:
        fetch_and_admit_new_tickers(tickers, client=client)
        for task in tasks[:len(tickers)]:
            queue.mark_complete(task.task_id, data_path="data.nosync/universe/constituents.json")
    except Exception as exc:
        log.warning("process_queue.ticker_details_error", error=str(exc))
        for task in tasks[:len(tickers)]:
            queue.mark_failed(task.task_id, str(exc))
    return len(tickers)


def process_price_backfills(queue: WorkQueue, client: PolygonClient) -> int:
    from src.collection.polygon_client import PolygonClient
    from datetime import date, timedelta
    import json
    from pathlib import Path

    tasks = queue.get_pending(task_type="price_backfill")
    processed = 0
    prices_dir = Path("data.nosync/prices")

    for task in tasks:
        if processed >= _MAX_TASKS_PER_RUN:
            break
        ticker = task.ticker
        log.info("process_queue.price_backfill", ticker=ticker)
        try:
            # 2-year backfill via aggregate bars
            to_date = date.today().isoformat()
            from_date = (date.today() - timedelta(days=730)).isoformat()
            bars = client.get_agg_bars(ticker, from_date, to_date, adjusted=True)

            if not bars:
                log.warning("process_queue.backfill_no_bars", ticker=ticker)
                queue.mark_failed(task.task_id, "no bars returned")
                processed += 1
                continue

            # Merge into existing daily price files
            from datetime import date as date_cls
            updated = 0
            for bar in bars:
                bar_date = date_cls.fromtimestamp(bar["t"] / 1000).isoformat()
                price_file = prices_dir / f"{bar_date}.json"
                o, h, lo, c = bar.get("o", 0.0), bar.get("h", 0.0), bar.get("l", 0.0), bar.get("c", 0.0)
                record = {
                    "date": bar_date, "ticker": ticker,
                    "open": o, "high": h, "low": lo, "close": c,
                    "volume": bar.get("v", 0), "vwap": bar.get("vw", 0.0),
                    "ohlc_avg": round((o + h + lo + c) / 4, 4),
                }
                if price_file.exists():
                    with open(price_file) as f:
                        records = json.load(f)
                    records = [r for r in records if r.get("ticker") != ticker]
                    records.append(record)
                    records.sort(key=lambda r: r["ticker"])
                    with open(price_file, "w") as f:
                        json.dump(records, f, indent=2)
                    updated += 1

            # Rebuild consolidated Parquet so the DAL doesn't serve stale backfill prices
            if updated > 0:
                from src.collection.convert_prices_to_parquet import convert as rebuild_parquet
                rebuild_parquet()
                log.info("process_queue.parquet_rebuilt", ticker=ticker)

            queue.mark_complete(task.task_id, data_path=f"data.nosync/prices/ ({updated} files)")
            log.info("process_queue.backfill_done", ticker=ticker, bars=len(bars), files_updated=updated)
        except Exception as exc:
            log.warning("process_queue.backfill_error", ticker=ticker, error=str(exc))
            queue.mark_failed(task.task_id, str(exc))
        processed += 1
    return processed



def run() -> None:
    from dotenv import load_dotenv
    load_dotenv()

    client = PolygonClient()
    queue = WorkQueue()

    expired = queue.expire_old_tasks(max_age_days=7)
    if expired:
        log.info("process_queue.expired_tasks", count=expired)

    log.info("process_queue.starting", **queue.stats())

    processed = 0
    processed += process_split_corrections(queue, client)
    processed += process_ticker_details(queue, client)
    processed += process_price_backfills(queue, client)

    log.info("process_queue.done", processed=processed, **queue.stats())


if __name__ == "__main__":
    run()
