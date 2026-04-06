"""
src/tracking/positions.py

Position lifecycle management for the paper trading portfolio.

Stores all positions in data.nosync/positions/positions.json.
position_id: {ticker}_{strategy}_{entry_date}
originating_models: tracks which selection models recommended the stock.

Entry points:
    open_position(ticker, strategy, entry_date, entry_price, originating_models, entry_details) -> str
    close_position(position_id, exit_date, exit_price) -> dict
    get_open_positions() -> list[dict]
    get_closed_positions() -> list[dict]
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import structlog

log = structlog.get_logger()

_POSITIONS_FILE = Path("data.nosync/positions/positions.json")


@dataclass
class Position:
    position_id: str
    ticker: str
    strategy: str
    entry_date: str
    entry_price: float
    originating_models: list[str]
    status: str = "open"
    entry_details: dict = field(default_factory=dict)
    current_value: float | None = None
    unrealized_pnl: float | None = None
    mark_date: str | None = None
    exit_date: str | None = None
    exit_price: float | None = None
    realized_pnl: float | None = None

    def to_dict(self) -> dict:
        return asdict(self)


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


def open_position(
    ticker: str,
    strategy: str,
    entry_date: str,
    entry_price: float,
    originating_models: list[str],
    entry_details: dict | None = None,
) -> str:
    """
    Open a new position. Returns the position_id.
    Idempotent: if a position with this id already exists, returns it unchanged.
    """
    position_id = f"{ticker}_{strategy}_{entry_date}"
    positions = _load_positions()

    if any(p["position_id"] == position_id for p in positions):
        log.info("positions.already_open", position_id=position_id)
        return position_id

    position = Position(
        position_id=position_id,
        ticker=ticker,
        strategy=strategy,
        entry_date=entry_date,
        entry_price=round(float(entry_price), 4),
        originating_models=originating_models or [],
        entry_details=entry_details or {},
    )
    positions.append(position.to_dict())
    _save_positions(positions)
    log.info(
        "positions.opened",
        position_id=position_id,
        ticker=ticker,
        strategy=strategy,
        entry_price=entry_price,
    )
    return position_id


def close_position(position_id: str, exit_date: str, exit_price: float) -> dict:
    """
    Close an open position. Computes realized P&L per share.
    Returns the updated position dict (empty dict if not found).
    """
    positions = _load_positions()
    for pos in positions:
        if pos["position_id"] == position_id and pos.get("status") == "open":
            exit_price_r = round(float(exit_price), 4)
            realized = round(exit_price_r - pos["entry_price"], 4)
            pos["status"] = "closed"
            pos["exit_date"] = exit_date
            pos["exit_price"] = exit_price_r
            pos["realized_pnl"] = realized
            pos["unrealized_pnl"] = None
            pos["current_value"] = None
            _save_positions(positions)
            log.info(
                "positions.closed",
                position_id=position_id,
                exit_price=exit_price_r,
                realized_pnl=realized,
            )
            return pos

    log.warning("positions.not_found_or_already_closed", position_id=position_id)
    return {}


def get_open_positions() -> list[dict]:
    return [p for p in _load_positions() if p.get("status") == "open"]


def get_closed_positions() -> list[dict]:
    return [p for p in _load_positions() if p.get("status") == "closed"]
