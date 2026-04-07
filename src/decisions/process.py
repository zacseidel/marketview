"""
src/decisions/process.py

Parses user-edited decision markdown and records approved transitions.
Triggered by process-decisions.yml on push to decisions/pending/.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from pathlib import Path

import structlog

from src.collection.queue import WorkQueue

log = structlog.get_logger()

_DECISIONS_DATA_DIR = Path("data.nosync/decisions")
_PROCESSED_DIR = Path("data.nosync/decisions/processed")


@dataclass
class DecisionRecord:
    eval_date: str
    execution_date: str
    ticker: str
    action: str                      # 'buy' | 'sell' | 'hold'
    recommending_models: list[str]
    user_approved: bool
    execution_price: float | None = None
    status: str = "pending"          # 'pending' | 'executed'
    notes: str | None = None


@dataclass
class ProcessResult:
    eval_date: str
    buys_approved: int
    sells_approved: int
    holds_confirmed: int
    total_decisions: int


def _next_trading_day(eval_date: str) -> str:
    d = date.fromisoformat(eval_date)
    d += timedelta(days=1)
    while d.weekday() >= 5:
        d += timedelta(days=1)
    return d.isoformat()


def _parse_models(line: str) -> list[str]:
    """Extract model names from a line like '- [x] AAPL — momentum (0.90) + buyback (0.72)'."""
    model_pattern = re.compile(r'\b(momentum|buyback|watchlist|quant|thirteen_f|earnings)\b')
    return model_pattern.findall(line)


def _parse_decision_file(file_path: Path) -> tuple[str, str, list[dict]]:
    """
    Parse the markdown file.
    Returns (eval_date, execution_date, list of raw decision dicts).
    """
    content = file_path.read_text()
    lines = content.splitlines()

    eval_date = ""
    execution_date = ""

    # Extract dates from header
    for line in lines[:5]:
        m = re.search(r'Evaluation:\s+(\d{4}-\d{2}-\d{2})', line)
        if m:
            eval_date = m.group(1)
        m = re.search(r'Execute:\s+\w+\s+(\d{4}-\d{2}-\d{2})', line)
        if m:
            execution_date = m.group(1)

    if not eval_date:
        raise ValueError(f"Could not parse eval_date from {file_path}")
    if not execution_date:
        execution_date = _next_trading_day(eval_date)

    # Parse sections and checkboxes
    current_section = None
    decisions: list[dict] = []

    section_map = {
        "## New Buy Recommendations": "new_buy",
        "## Current Holdings": "hold",
        "## Sell Recommendations": "sell",
    }

    for line in lines:
        # Detect section headers
        for header, section in section_map.items():
            if line.startswith(header):
                current_section = section
                break

        if current_section is None:
            continue

        # Match checkbox lines: - [x] TICKER — ...
        m = re.match(r'^-\s+\[( |x)\]\s+([A-Z.\-]+)\s*[—-]?\s*(.*)', line, re.IGNORECASE)
        if not m:
            continue

        checked = m.group(1).lower() == "x"
        ticker = m.group(2).upper()
        rest = m.group(3)
        models = _parse_models(rest)

        # Translate section + checked state to action + approved
        if current_section == "new_buy":
            action = "buy"
            approved = checked
        elif current_section == "hold":
            # unchecked hold = early sell
            action = "hold" if checked else "sell"
            approved = True
        else:  # sell
            action = "sell" if checked else "hold"  # uncheck = override, keep
            approved = True

        decisions.append({
            "ticker": ticker,
            "action": action,
            "user_approved": approved,
            "recommending_models": models,
            "section": current_section,
        })

    return eval_date, execution_date, decisions


def process_decision_file(file_path: str | Path) -> ProcessResult:
    """
    Parse a decision markdown file, write DecisionRecords, queue price fetches.
    Idempotent: skips if decision records already exist for this eval_date.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Decision file not found: {file_path}")

    eval_date, execution_date, raw_decisions = _parse_decision_file(file_path)

    _DECISIONS_DATA_DIR.mkdir(parents=True, exist_ok=True)
    _PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    out_path = _DECISIONS_DATA_DIR / f"{eval_date}.json"

    # Skip re-processing only for historical decisions (execution already happened).
    # Current/upcoming decisions are always re-processed so user vetoes take effect.
    if out_path.exists() and execution_date < date.today().isoformat():
        log.info("process.already_done", eval_date=eval_date)
        with open(out_path) as f:
            existing = json.load(f)
        buys = sum(1 for r in existing if r["action"] == "buy" and r["user_approved"])
        sells = sum(1 for r in existing if r["action"] == "sell" and r["user_approved"])
        holds = sum(1 for r in existing if r["action"] == "hold")
        return ProcessResult(
            eval_date=eval_date,
            buys_approved=buys,
            sells_approved=sells,
            holds_confirmed=holds,
            total_decisions=len(existing),
        )
    if out_path.exists():
        log.info("process.reprocessing", eval_date=eval_date, execution_date=execution_date)

    queue = WorkQueue()
    records: list[DecisionRecord] = []

    buys_approved = 0
    sells_approved = 0
    holds_confirmed = 0

    for d in raw_decisions:
        record = DecisionRecord(
            eval_date=eval_date,
            execution_date=execution_date,
            ticker=d["ticker"],
            action=d["action"],
            recommending_models=d["recommending_models"],
            user_approved=d["user_approved"],
        )
        records.append(record)

        if d["action"] == "buy" and d["user_approved"]:
            buys_approved += 1
            queue.enqueue(
                task_type="price_fetch",
                ticker=d["ticker"],
                requested_date=execution_date,
                requested_by="decision_processor",
                priority="high",
            )
        elif d["action"] == "sell" and d["user_approved"]:
            sells_approved += 1
            queue.enqueue(
                task_type="price_fetch",
                ticker=d["ticker"],
                requested_date=execution_date,
                requested_by="decision_processor",
                priority="high",
            )
        elif d["action"] == "hold":
            holds_confirmed += 1

    with open(out_path, "w") as f:
        json.dump([asdict(r) for r in records], f, indent=2)

    log.info(
        "process.complete",
        eval_date=eval_date,
        execution_date=execution_date,
        buys=buys_approved,
        sells=sells_approved,
        holds=holds_confirmed,
    )

    return ProcessResult(
        eval_date=eval_date,
        buys_approved=buys_approved,
        sells_approved=sells_approved,
        holds_confirmed=holds_confirmed,
        total_decisions=len(records),
    )


def process_all_pending() -> None:
    """Process all unprocessed decision files in decisions/pending/."""
    pending_dir = Path("decisions/pending")
    if not pending_dir.exists():
        return

    for md_file in sorted(pending_dir.glob("*.md")):
        eval_date = md_file.stem
        out_path = _DECISIONS_DATA_DIR / f"{eval_date}.json"
        if out_path.exists():
            log.debug("process.skipping_existing", eval_date=eval_date)
            continue
        log.info("process.processing", file=md_file.name)
        try:
            process_decision_file(md_file)
        except Exception as exc:
            log.error("process.error", file=md_file.name, error=str(exc))


if __name__ == "__main__":
    process_all_pending()
