"""
src/decisions/execute.py

Records execution prices and creates/closes portfolio positions.
Runs Tuesday and Friday after market close (record-executions.yml).
"""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from pathlib import Path

import structlog

log = structlog.get_logger()

_DECISIONS_DIR = Path("data/decisions")
_POSITIONS_FILE = Path("data/positions/positions.json")
_PRICES_DIR = Path("data/prices")


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


def _load_positions() -> list[dict]:
    if not _POSITIONS_FILE.exists():
        return []
    with open(_POSITIONS_FILE) as f:
        return json.load(f)


def _save_positions(positions: list[dict]) -> None:
    _POSITIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(_POSITIONS_FILE, "w") as f:
        json.dump(positions, f, indent=2)


def _load_ohlc(execution_date: str) -> dict[str, float]:
    """Load execution day's ohlc_avg per ticker."""
    price_file = _PRICES_DIR / f"{execution_date}.json"
    if not price_file.exists():
        return {}
    with open(price_file) as f:
        records = json.load(f)
    return {r["ticker"]: r["ohlc_avg"] for r in records if "ohlc_avg" in r}


def _load_pending_decisions(execution_date: str) -> list[tuple[str, dict]]:
    """
    Find all decision records where execution_date matches and status is pending.
    Returns [(eval_date, record), ...].
    """
    pending: list[tuple[str, dict]] = []
    if not _DECISIONS_DIR.exists():
        return pending

    for path in sorted(_DECISIONS_DIR.glob("*.json")):
        if path.stem.startswith(".") or path.parent.name == "processed":
            continue
        with open(path) as f:
            records = json.load(f)
        for r in records:
            if r.get("execution_date") == execution_date and r.get("status") == "pending":
                if r.get("user_approved") and r.get("action") in ("buy", "sell"):
                    pending.append((path.stem, r))

    return pending


def _save_decision_updates(eval_date: str, updates: dict[str, dict]) -> None:
    """Update specific decision records in data/decisions/{eval_date}.json."""
    path = _DECISIONS_DIR / f"{eval_date}.json"
    if not path.exists():
        return
    with open(path) as f:
        records = json.load(f)
    for r in records:
        key = (r["ticker"], r["action"])
        if key in updates:
            r.update(updates[key])
    with open(path, "w") as f:
        json.dump(records, f, indent=2)


def record_executions(execution_date: str | None = None) -> ExecutionResult:
    """
    Fill execution prices for all approved decisions on execution_date.
    Creates/closes positions accordingly.
    """
    if execution_date is None:
        execution_date = date.today().isoformat()

    log.info("execute.starting", execution_date=execution_date)

    ohlc = _load_ohlc(execution_date)
    if not ohlc:
        log.warning("execute.no_price_data", execution_date=execution_date)
        return ExecutionResult(execution_date=execution_date, positions_opened=0, positions_closed=0, skipped_no_price=0)

    pending = _load_pending_decisions(execution_date)
    if not pending:
        log.info("execute.no_pending", execution_date=execution_date)
        return ExecutionResult(execution_date=execution_date, positions_opened=0, positions_closed=0, skipped_no_price=0)

    positions = _load_positions()
    # Index open positions by ticker for fast lookup
    open_by_ticker: dict[str, dict] = {
        p["ticker"]: p for p in positions if p["status"] == "open"
    }

    opened = 0
    closed = 0
    skipped = 0
    decision_updates: dict[str, dict[tuple, dict]] = {}

    for eval_date, record in pending:
        ticker = record["ticker"]
        action = record["action"]

        fill_price = ohlc.get(ticker)
        if fill_price is None:
            log.warning("execute.no_price_for_ticker", ticker=ticker, date=execution_date)
            skipped += 1
            continue

        if eval_date not in decision_updates:
            decision_updates[eval_date] = {}

        if action == "buy":
            if ticker in open_by_ticker:
                log.debug("execute.already_open", ticker=ticker)
                continue

            position_id = f"{ticker}_stock_{execution_date}_{str(uuid.uuid4())[:8]}"
            new_pos = Position(
                position_id=position_id,
                ticker=ticker,
                strategy="stock",
                entry_date=execution_date,
                entry_price=fill_price,
                entry_details={"fill_method": "ohlc_avg"},
                status="open",
                originating_models=record.get("recommending_models", []),
            )
            positions.append(asdict(new_pos))
            open_by_ticker[ticker] = asdict(new_pos)
            opened += 1
            log.info("execute.opened", ticker=ticker, price=fill_price, id=position_id)

            # Queue options chain fetch to snapshot all strategy templates
            from src.collection.queue import WorkQueue
            WorkQueue().enqueue(
                task_type="options_chain",
                ticker=ticker,
                requested_date=execution_date,
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
                continue

            pos = open_by_ticker[ticker]
            realized_pnl = round(fill_price - pos["entry_price"], 4)
            pos.update({
                "status": "closed",
                "exit_date": execution_date,
                "exit_price": fill_price,
                "realized_pnl": realized_pnl,
            })
            closed += 1
            del open_by_ticker[ticker]
            log.info("execute.closed", ticker=ticker, entry=pos["entry_price"], exit=fill_price, pnl=realized_pnl)

            # Close strategy observations; queue chain for any unexpired options legs
            from src.strategy.snapshot import close_all_for_model_sell
            from src.collection.queue import WorkQueue
            all_closed = close_all_for_model_sell(
                ticker=ticker,
                stock_entry_date=pos["entry_date"],
                close_date=execution_date,
                stock_price=fill_price,
            )
            if not all_closed:
                WorkQueue().enqueue(
                    task_type="options_chain",
                    ticker=ticker,
                    requested_date=execution_date,
                    requested_by="execute_close",
                    priority="high",
                )

            decision_updates[eval_date][(ticker, action)] = {
                "execution_price": fill_price,
                "status": "executed",
            }

    _save_positions(positions)

    # Patch decision files
    for eval_date, updates in decision_updates.items():
        _save_decision_updates(eval_date, updates)

    log.info("execute.done", opened=opened, closed=closed, skipped=skipped)
    return ExecutionResult(
        execution_date=execution_date,
        positions_opened=opened,
        positions_closed=closed,
        skipped_no_price=skipped,
    )


if __name__ == "__main__":
    import sys
    from dotenv import load_dotenv
    load_dotenv()
    exec_date = sys.argv[1] if len(sys.argv) > 1 else None
    result = record_executions(exec_date)
    print(f"Opened: {result.positions_opened}, Closed: {result.positions_closed}, Skipped: {result.skipped_no_price}")
