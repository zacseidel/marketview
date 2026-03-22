"""
src/universe/splits.py

Split detection confirmation and price history correction.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import structlog

from src.collection.polygon_client import PolygonClient

log = structlog.get_logger()

_PRICES_DIR = Path("data/prices")
_SPLITS_DIR = Path("data/splits")
_UNIVERSE_FILE = Path("data/universe/constituents.json")

# A confirmed split must have occurred within this many days of the flagged date
_SPLIT_WINDOW_DAYS = 5


@dataclass
class SplitResult:
    ticker: str
    flagged_date: str
    confirmed: bool
    split_ratio: float | None = None  # e.g. 2.0 for 2-for-1; 0.5 for 1-for-2 reverse
    split_date: str | None = None
    records_corrected: int = 0
    error: str | None = None


def _find_split_near_date(splits: list[dict], flagged_date: str) -> dict | None:
    """Return the split event within _SPLIT_WINDOW_DAYS of flagged_date, or None."""
    flagged = date.fromisoformat(flagged_date)
    for split in splits:
        split_date_str = split.get("execution_date") or split.get("ex_date", "")
        if not split_date_str:
            continue
        try:
            split_date = date.fromisoformat(split_date_str)
        except ValueError:
            continue
        if abs((split_date - flagged).days) <= _SPLIT_WINDOW_DAYS:
            return split
    return None


def _compute_ratio(split: dict) -> float | None:
    """
    Compute split ratio from Polygon split record.
    split_from / split_to gives the price multiplier (e.g. 1/2 for 2-for-1 = halves price).
    We want the adjustment factor for historical prices: split_to / split_from.
    """
    split_from = split.get("split_from")
    split_to = split.get("split_to")
    if not split_from or not split_to or split_from == 0:
        return None
    # A 2-for-1 split: split_from=1, split_to=2. Price halves. Ratio = 2.0 means "2 new shares per 1 old".
    return split_to / split_from


def _backfill_adjusted_prices(
    ticker: str,
    client: PolygonClient,
    from_date: str = "2020-01-01",
) -> dict[str, dict]:
    """
    Re-download full adjusted price history for ticker.
    Returns {date_str: bar_record}.
    """
    to_date = date.today().isoformat()
    bars = client.get_agg_bars(ticker, from_date, to_date, adjusted=True)
    result: dict[str, dict] = {}
    for bar in bars:
        # Polygon timestamp is milliseconds since epoch; convert to date
        ts_ms = bar.get("t", 0)
        bar_date = date.fromtimestamp(ts_ms / 1000).isoformat()
        o, h, lo, c = bar.get("o", 0.0), bar.get("h", 0.0), bar.get("l", 0.0), bar.get("c", 0.0)
        result[bar_date] = {
            "date": bar_date,
            "ticker": ticker,
            "open": o,
            "high": h,
            "low": lo,
            "close": c,
            "volume": bar.get("v", 0),
            "vwap": bar.get("vw", 0.0),
            "ohlc_avg": round((o + h + lo + c) / 4, 4),
        }
    return result


def _update_price_files(ticker: str, adjusted_bars: dict[str, dict]) -> int:
    """
    Overwrite the ticker's records in all matching data/prices/{date}.json files.
    Returns the number of daily files updated.
    """
    updated = 0
    for price_file in sorted(_PRICES_DIR.glob("*.json")):
        file_date = price_file.stem  # YYYY-MM-DD
        if file_date not in adjusted_bars:
            continue
        with open(price_file) as f:
            records: list[dict] = json.load(f)

        new_records = [r for r in records if r.get("ticker") != ticker]
        new_records.append(adjusted_bars[file_date])
        new_records.sort(key=lambda r: r["ticker"])

        with open(price_file, "w") as f:
            json.dump(new_records, f, indent=2)
        updated += 1

    return updated


def confirm_and_correct_split(
    ticker: str,
    flagged_date: str,
    client: PolygonClient | None = None,
) -> SplitResult:
    """
    Cross-reference Polygon splits endpoint to confirm a suspected split.
    If confirmed, re-download full adjusted history and patch all price files.
    """
    _SPLITS_DIR.mkdir(parents=True, exist_ok=True)

    if client is None:
        client = PolygonClient()

    log.info("splits.checking", ticker=ticker, flagged_date=flagged_date)

    # Check for existing split record (idempotent)
    split_record_path = _SPLITS_DIR / f"{ticker}_{flagged_date}.json"
    if split_record_path.exists():
        with open(split_record_path) as f:
            existing = json.load(f)
        log.info("splits.already_processed", ticker=ticker, flagged_date=flagged_date)
        return SplitResult(
            ticker=ticker,
            flagged_date=flagged_date,
            confirmed=existing.get("confirmed", False),
            split_ratio=existing.get("split_ratio"),
            split_date=existing.get("split_date"),
            records_corrected=existing.get("records_corrected", 0),
        )

    try:
        splits = client.get_splits(ticker)
    except Exception as exc:
        log.warning("splits.fetch_error", ticker=ticker, error=str(exc))
        return SplitResult(ticker=ticker, flagged_date=flagged_date, confirmed=False, error=str(exc))

    split_event = _find_split_near_date(splits, flagged_date)

    if not split_event:
        log.info("splits.not_confirmed", ticker=ticker, flagged_date=flagged_date)
        result = SplitResult(ticker=ticker, flagged_date=flagged_date, confirmed=False)
        with open(split_record_path, "w") as f:
            json.dump(asdict(result), f, indent=2)
        return result

    ratio = _compute_ratio(split_event)
    split_date_str = split_event.get("execution_date") or split_event.get("ex_date", "")
    log.info("splits.confirmed", ticker=ticker, split_date=split_date_str, ratio=ratio)

    # Re-download full adjusted history
    try:
        adjusted_bars = _backfill_adjusted_prices(ticker, client)
    except Exception as exc:
        log.error("splits.backfill_error", ticker=ticker, error=str(exc))
        return SplitResult(
            ticker=ticker,
            flagged_date=flagged_date,
            confirmed=True,
            split_ratio=ratio,
            split_date=split_date_str,
            error=str(exc),
        )

    records_corrected = _update_price_files(ticker, adjusted_bars)
    log.info("splits.corrected", ticker=ticker, files_updated=records_corrected)

    result = SplitResult(
        ticker=ticker,
        flagged_date=flagged_date,
        confirmed=True,
        split_ratio=ratio,
        split_date=split_date_str,
        records_corrected=records_corrected,
    )
    with open(split_record_path, "w") as f:
        json.dump(asdict(result), f, indent=2)

    return result
