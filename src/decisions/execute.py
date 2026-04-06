"""
src/decisions/execute.py

Records execution prices and creates/closes portfolio positions.
Runs Tuesday and Friday after market close (record-executions.yml).

When called with no date argument, processes ALL pending decisions where
execution_date <= today, in chronological order (buys before sells within
each date). Fetches price data from Polygon if the local file is missing.
"""

from __future__ import annotations

import json
import uuid
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from pathlib import Path

import structlog

log = structlog.get_logger()

_DECISIONS_DIR = Path("data.nosync/decisions")
_POSITIONS_FILE = Path("data.nosync/positions/positions.json")
_PRICES_DIR = Path("data.nosync/prices")

_MAX_DATE_ADVANCE = 3   # trading days to walk forward when a date has no market data


@dataclass
class Position:
    position_id: str
    ticker: str
    strategy: str
    entry_date: str
    entry_price: float
    entry_details: dict
    status: str                    # 'open' | 'closed'
    originating_models: list[str]
    current_value: float | None = None
    unrealized_pnl: float | None = None
    exit_date: str | None = None
    exit_price: float | None = None
    realized_pnl: float | None = None


@dataclass
class ExecutionResult:
    execution_date: str
    positions_opened: int
    positions_closed: int
    skipped_no_price: int
    skipped_no_position: int


def _load_positions() -> list[dict]:
    if not _POSITIONS_FILE.exists():
        return []
    with open(_POSITIONS_FILE) as f:
        return json.load(f)


def _save_positions(positions: list[dict]) -> None:
    _POSITIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = _POSITIONS_FILE.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(positions, f, indent=2)
    tmp.replace(_POSITIONS_FILE)


def _load_local_ohlc(price_date: str) -> dict[str, float]:
    """Load ohlc_avg per ticker from a local price file, or {} if missing."""
    price_file = _PRICES_DIR / f"{price_date}.json"
    if not price_file.exists():
        return {}
    with open(price_file) as f:
        records = json.load(f)
    return {r["ticker"]: r["ohlc_avg"] for r in records if "ohlc_avg" in r}


def _next_trading_day(d: date) -> date:
    d += timedelta(days=1)
    while d.weekday() >= 5:
        d += timedelta(days=1)
    return d


def _fetch_and_cache_grouped_daily(price_date: str) -> dict[str, float]:
    """
    Fetch grouped daily bars from Polygon for price_date, write to the local
    price file, and return {ticker: ohlc_avg}.  Returns {} on API error.
    """
    from src.collection.polygon_client import PolygonClient, PolygonAPIError
    try:
        client = PolygonClient()
        resp = client.get_grouped_daily(price_date)
    except PolygonAPIError as exc:
        log.warning("execute.polygon_fetch_failed", date=price_date, error=str(exc))
        return {}

    bars = resp.get("results") or []
    if not bars:
        return {}

    records = []
    ohlc: dict[str, float] = {}
    for bar in bars:
        ticker = bar.get("T", "")
        if not ticker:
            continue
        o = bar.get("o", 0.0)
        h = bar.get("h", 0.0)
        lo = bar.get("l", 0.0)
        c = bar.get("c", 0.0)
        avg = round((o + h + lo + c) / 4.0, 4)
        records.append({
            "date": price_date,
            "ticker": ticker,
            "open": o,
            "high": h,
            "low": lo,
            "close": c,
            "volume": bar.get("v", 0),
            "vwap": bar.get("vw", 0.0),
            "ohlc_avg": avg,
        })
        ohlc[ticker] = avg

    # Cache locally so subsequent runs don't re-fetch
    price_file = _PRICES_DIR / f"{price_date}.json"
    _PRICES_DIR.mkdir(parents=True, exist_ok=True)
    tmp = price_file.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(records, f)
    tmp.replace(price_file)
    log.info("execute.cached_prices", date=price_date, tickers=len(ohlc))
    return ohlc


def _resolve_ohlc(execution_date: str) -> tuple[str, dict[str, float]]:
    """
    Return (actual_date, {ticker: ohlc_avg}) for execution_date.

    Resolution order:
      1. Local price file for execution_date
      2. Polygon API fetch → cached locally
      3. Walk forward up to _MAX_DATE_ADVANCE trading days, trying each

    Returns ("", {}) if no data can be found.
    """
    d = date.fromisoformat(execution_date)

    for _ in range(_MAX_DATE_ADVANCE + 1):
        ds = d.isoformat()

        # Try local first
        ohlc = _load_local_ohlc(ds)
        if ohlc:
            if ds != execution_date:
                log.info("execute.date_advanced", requested=execution_date, used=ds)
            return ds, ohlc

        # Try Polygon
        ohlc = _fetch_and_cache_grouped_daily(ds)
        if ohlc:
            if ds != execution_date:
                log.info("execute.date_advanced", requested=execution_date, used=ds)
            return ds, ohlc

        # No data for this date (holiday / weekend) — advance
        d = _next_trading_day(d)

    log.warning("execute.no_price_data", execution_date=execution_date)
    return "", {}


def _load_all_pending_decisions(as_of: str) -> dict[str, list[tuple[str, dict]]]:
    """
    Scan data.nosync/decisions/*.json for all records where:
      - status == "pending"
      - user_approved == True
      - action in ("buy", "sell")
      - execution_date <= as_of

    Returns {execution_date: [(eval_date, record), ...]} sorted by execution_date.
    Buys come before sells within each execution_date.
    """
    by_date: dict[str, list[tuple[str, dict]]] = defaultdict(list)

    if not _DECISIONS_DIR.exists():
        return {}

    for path in sorted(_DECISIONS_DIR.glob("*.json")):
        if path.stem.startswith(".") or path.parent.name in ("processed", "pending"):
            continue
        with open(path) as f:
            records = json.load(f)
        for r in records:
            exec_date = r.get("execution_date", "")
            if (
                r.get("status") == "pending"
                and r.get("user_approved")
                and r.get("action") in ("buy", "sell")
                and exec_date <= as_of
            ):
                by_date[exec_date].append((path.stem, r))

    # Sort buys before sells within each date
    action_order = {"buy": 0, "sell": 1}
    for exec_date in by_date:
        by_date[exec_date].sort(key=lambda x: action_order.get(x[1].get("action", "sell"), 1))

    return dict(sorted(by_date.items()))


def _save_decision_updates(eval_date: str, updates: dict[tuple, dict]) -> None:
    """Update specific decision records in data.nosync/decisions/{eval_date}.json."""
    path = _DECISIONS_DIR / f"{eval_date}.json"
    if not path.exists():
        return
    with open(path) as f:
        records = json.load(f)
    for r in records:
        key = (r["ticker"], r["action"])
        if key in updates:
            r.update(updates[key])
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(records, f, indent=2)
    tmp.replace(path)


def record_executions(execution_date: str | None = None) -> list[ExecutionResult]:
    """
    Fill execution prices for all approved pending decisions with
    execution_date <= today (or == execution_date if supplied).

    Creates/closes positions accordingly. Returns one ExecutionResult per
    execution_date processed.
    """
    as_of = execution_date or date.today().isoformat()
    log.info("execute.starting", as_of=as_of)

    pending_by_date = _load_all_pending_decisions(as_of)
    if not pending_by_date:
        log.info("execute.no_pending", as_of=as_of)
        return []

    positions = _load_positions()
    open_by_ticker: dict[str, dict] = {
        p["ticker"]: p for p in positions if p["status"] == "open"
    }

    results: list[ExecutionResult] = []

    for exec_date, pending in pending_by_date.items():
        actual_date, ohlc = _resolve_ohlc(exec_date)

        opened = 0
        closed = 0
        skipped_price = 0
        skipped_no_pos = 0
        decision_updates: dict[str, dict[tuple, dict]] = {}

        for eval_date, record in pending:
            ticker = record["ticker"]
            action = record["action"]

            if not ohlc:
                log.warning("execute.no_price_data_for_date", exec_date=exec_date, ticker=ticker)
                skipped_price += 1
                continue

            fill_price = ohlc.get(ticker)
            if fill_price is None:
                log.warning("execute.no_price_for_ticker", ticker=ticker, date=actual_date)
                skipped_price += 1
                continue

            if eval_date not in decision_updates:
                decision_updates[eval_date] = {}

            if action == "buy":
                if ticker in open_by_ticker:
                    log.debug("execute.already_open", ticker=ticker)
                    decision_updates[eval_date][(ticker, action)] = {"status": "skipped"}
                    continue

                position_id = f"{ticker}_stock_{actual_date}_{str(uuid.uuid4())[:8]}"
                new_pos = Position(
                    position_id=position_id,
                    ticker=ticker,
                    strategy="stock",
                    entry_date=actual_date,
                    entry_price=fill_price,
                    entry_details={"fill_method": "ohlc_avg", "requested_date": exec_date},
                    status="open",
                    originating_models=record.get("recommending_models", []),
                )
                positions.append(asdict(new_pos))
                open_by_ticker[ticker] = asdict(new_pos)
                opened += 1
                log.info("execute.opened", ticker=ticker, price=fill_price, id=position_id)

                from src.collection.queue import WorkQueue
                WorkQueue().enqueue(
                    task_type="options_chain",
                    ticker=ticker,
                    requested_date=actual_date,
                    requested_by="execute",
                    priority="high",
                )

                decision_updates[eval_date][(ticker, action)] = {
                    "execution_price": fill_price,
                    "status": "executed",
                }

            elif action == "sell":
                if ticker not in open_by_ticker:
                    log.warning("execute.no_open_position", ticker=ticker)
                    skipped_no_pos += 1
                    decision_updates[eval_date][(ticker, action)] = {"status": "skipped"}
                    continue

                pos = open_by_ticker[ticker]
                realized_pnl = round(fill_price - pos["entry_price"], 4)
                pos.update({
                    "status": "closed",
                    "exit_date": actual_date,
                    "exit_price": fill_price,
                    "realized_pnl": realized_pnl,
                })
                closed += 1
                del open_by_ticker[ticker]
                log.info("execute.closed", ticker=ticker, entry=pos["entry_price"], exit=fill_price, pnl=realized_pnl)

                from src.strategy.snapshot import close_all_for_model_sell
                from src.collection.queue import WorkQueue
                all_closed = close_all_for_model_sell(
                    ticker=ticker,
                    stock_entry_date=pos["entry_date"],
                    close_date=actual_date,
                    stock_price=fill_price,
                )
                if not all_closed:
                    WorkQueue().enqueue(
                        task_type="options_chain",
                        ticker=ticker,
                        requested_date=actual_date,
                        requested_by="execute_close",
                        priority="high",
                    )

                decision_updates[eval_date][(ticker, action)] = {
                    "execution_price": fill_price,
                    "status": "executed",
                }

        for eval_date, updates in decision_updates.items():
            _save_decision_updates(eval_date, updates)

        log.info(
            "execute.date_done",
            exec_date=exec_date,
            actual_date=actual_date,
            opened=opened,
            closed=closed,
            skipped_price=skipped_price,
            skipped_no_pos=skipped_no_pos,
        )
        results.append(ExecutionResult(
            execution_date=exec_date,
            positions_opened=opened,
            positions_closed=closed,
            skipped_no_price=skipped_price,
            skipped_no_position=skipped_no_pos,
        ))

    _save_positions(positions)
    return results


if __name__ == "__main__":
    import sys
    from dotenv import load_dotenv
    load_dotenv()
    exec_date = sys.argv[1] if len(sys.argv) > 1 else None
    results = record_executions(exec_date)
    if not results:
        print("No pending decisions to process.")
    for r in results:
        print(
            f"{r.execution_date}: opened={r.positions_opened} closed={r.positions_closed} "
            f"skipped_price={r.skipped_no_price} skipped_no_pos={r.skipped_no_position}"
        )
