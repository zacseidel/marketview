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
            queue.mark_complete(task.task_id, data_path=f"data/splits/{task.ticker}_{task.requested_date}.json")
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
            queue.mark_complete(task.task_id, data_path="data/universe/constituents.json")
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
    prices_dir = Path("data/prices")

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

            queue.mark_complete(task.task_id, data_path=f"data/prices/ ({updated} files)")
            log.info("process_queue.backfill_done", ticker=ticker, bars=len(bars), files_updated=updated)
        except Exception as exc:
            log.warning("process_queue.backfill_error", ticker=ticker, error=str(exc))
            queue.mark_failed(task.task_id, str(exc))
        processed += 1
    return processed


def process_options_chains(queue: WorkQueue, client: PolygonClient) -> int:
    """
    Fetch options chains queued for strategy snapshot entry/exit.
    Task metadata carries: reason ("strategy_entry"|"strategy_close"), stock_entry_date.
    """
    import json
    from pathlib import Path

    tasks = queue.get_pending(task_type="options_chain", priority="high")
    tasks += queue.get_pending(task_type="options_chain", priority="normal")
    processed = 0

    for task in tasks:
        if processed >= _MAX_TASKS_PER_RUN:
            break

        ticker = task.ticker
        log.info("process_queue.options_chain", ticker=ticker, date=task.requested_date)

        try:
            chain = client.get_options_chain(ticker)
        except Exception as exc:
            log.warning("process_queue.options_chain_error", ticker=ticker, error=str(exc))
            queue.mark_failed(task.task_id, str(exc))
            processed += 1
            continue

        if not chain:
            log.warning("process_queue.empty_chain", ticker=ticker)
            queue.mark_failed(task.task_id, "empty chain")
            processed += 1
            continue

        # Save chain snapshot
        chain_dir = Path("data/options")
        chain_dir.mkdir(parents=True, exist_ok=True)
        chain_path = chain_dir / f"{ticker}_{task.requested_date}.json"
        with open(chain_path, "w") as f:
            json.dump(chain, f)

        # Load the stock price for this date
        price_file = Path("data/prices") / f"{task.requested_date}.json"
        stock_price = None
        if price_file.exists():
            with open(price_file) as f:
                prices = json.load(f)
            for rec in prices:
                if rec.get("ticker") == ticker:
                    stock_price = rec.get("close") or rec.get("ohlc_avg")
                    break

        # Handle strategy observations waiting on this chain
        from src.strategy.snapshot import (
            create_observation_set, save_observations,
            load_observations, close_awaiting_chain,
            reopen_expired_strategies,
        )

        reason = task.metadata.get("reason", "strategy_entry")

        # stock_entry_date: use metadata if present (reopen tasks set this explicitly),
        # otherwise look up from the open position record.
        stock_entry_date = task.metadata.get("stock_entry_date", task.requested_date)
        if stock_entry_date == task.requested_date:
            positions_file = Path("data/positions/positions.json")
            if positions_file.exists():
                with open(positions_file) as f:
                    positions = json.load(f)
                for pos in positions:
                    if pos.get("ticker") == ticker and pos.get("status") == "open":
                        stock_entry_date = pos.get("entry_date", task.requested_date)
                        break

        if stock_price is None:
            log.warning("process_queue.no_stock_price", ticker=ticker, date=task.requested_date)
            queue.mark_complete(task.task_id, data_path=str(chain_path))
            processed += 1
            continue

        # Close any awaiting_chain observations regardless of reason
        resolved = close_awaiting_chain(
            ticker=ticker,
            stock_entry_date=stock_entry_date,
            chain=chain,
            close_date=task.requested_date,
            stock_price=stock_price,
        )
        if resolved:
            log.info("process_queue.resolved_awaiting", ticker=ticker, resolved=resolved)

        if reason == "reopen":
            # Expired legs were already closed by strategy_runner; open new generation legs
            strategies_to_reopen = task.metadata.get("strategies_to_reopen", [])
            originating_models = task.metadata.get("originating_models", [])
            if strategies_to_reopen:
                reopen_expired_strategies(
                    ticker=ticker,
                    stock_entry_date=stock_entry_date,
                    strategies_to_reopen=strategies_to_reopen,
                    eval_date=task.requested_date,
                    stock_price=stock_price,
                    chain=chain,
                    originating_models=originating_models,
                )
                log.info(
                    "process_queue.strategies_reopened",
                    ticker=ticker,
                    strategies=strategies_to_reopen,
                )

        else:
            # strategy_entry: create initial observations if none exist yet
            obs = load_observations(ticker, stock_entry_date)
            if not obs:
                positions_file = Path("data/positions/positions.json")
                models = task.metadata.get("originating_models", [])
                if not models and positions_file.exists():
                    with open(positions_file) as f:
                        positions = json.load(f)
                    for pos in positions:
                        if pos.get("ticker") == ticker and pos.get("entry_date") == stock_entry_date:
                            models = pos.get("originating_models", [])
                            break

                new_obs = create_observation_set(
                    ticker=ticker,
                    stock_price=stock_price,
                    chain=chain,
                    entry_date=stock_entry_date,
                    originating_models=models,
                )
                save_observations(ticker, stock_entry_date, new_obs)
                log.info("process_queue.strategy_snapshots_created", ticker=ticker, count=len(new_obs))

        queue.mark_complete(task.task_id, data_path=str(chain_path))
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
    processed += process_options_chains(queue, client)

    log.info("process_queue.done", processed=processed, **queue.stats())


if __name__ == "__main__":
    run()
