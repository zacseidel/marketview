"""
src/collection/fundamentals.py

Fetches and stores quarterly financial data from Polygon's Stock Financials endpoint.
Used by the buyback model (shares_outstanding) and future quant models.

Stores per-ticker in data.nosync/fundamentals/{ticker}.json as a list of quarterly records,
most recent first. Resumable via data.nosync/fundamentals/.init_state.json.

Usage:
    python -m src.collection.fundamentals            # bulk init / refresh
    python -m src.collection.fundamentals AAPL MSFT  # specific tickers
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import structlog

from src.collection.polygon_client import PolygonAPIError, PolygonClient

log = structlog.get_logger()

_FUNDAMENTALS_DIR = Path("data.nosync/fundamentals")
_UNIVERSE_FILE = Path("data.nosync/universe/constituents.json")
_INIT_STATE_FILE = _FUNDAMENTALS_DIR / ".init_state.json"


def _parse_quarterly_record(result: dict, ticker: str) -> dict | None:
    """
    Extract the fields we care about from one Polygon financials result entry.
    Returns None if essential fields are missing.
    """
    # Polygon financials nests data under financials.income_statement / balance_sheet
    fin = result.get("financials", {})
    income = fin.get("income_statement", {})
    balance = fin.get("balance_sheet", {})
    comprehensive = fin.get("comprehensive_income", {})

    filing_date = result.get("filing_date") or result.get("period_of_report_date", "")
    period = result.get("fiscal_period", "") + " " + str(result.get("fiscal_year", ""))
    period = period.strip()

    # Shares — weighted average is best for buyback detection
    shares = (
        income.get("diluted_average_shares", {}).get("value")
        or income.get("basic_average_shares", {}).get("value")
        or balance.get("equity", {}).get("value")  # fallback
    )

    revenue = income.get("revenues", {}).get("value")
    net_income = income.get("net_income_loss", {}).get("value")

    if not filing_date:
        return None

    return {
        "ticker": ticker,
        "period": period,
        "filing_date": filing_date,
        "shares_outstanding": float(shares) if shares is not None else None,
        "revenue": float(revenue) if revenue is not None else None,
        "net_income": float(net_income) if net_income is not None else None,
        "market_cap": result.get("market_cap"),
    }


def fetch_fundamentals(ticker: str, client: PolygonClient) -> list[dict]:
    """
    Fetch up to 20 quarters of financials for one ticker.
    Returns list of quarterly records, most recent first.
    """
    try:
        raw_results = client.get_stock_financials(ticker, limit=20)
    except PolygonAPIError as exc:
        if exc.status_code == 404:
            log.debug("fundamentals.not_found", ticker=ticker)
        else:
            log.warning("fundamentals.api_error", ticker=ticker, error=str(exc))
        return []

    records = []
    for result in raw_results:
        parsed = _parse_quarterly_record(result, ticker)
        if parsed:
            records.append(parsed)

    # Sort most-recent first
    records.sort(key=lambda r: r["filing_date"], reverse=True)
    return records


def save_fundamentals(ticker: str, records: list[dict]) -> None:
    _FUNDAMENTALS_DIR.mkdir(parents=True, exist_ok=True)
    path = _FUNDAMENTALS_DIR / f"{ticker}.json"
    with open(path, "w") as f:
        json.dump(records, f, indent=2)


def fetch_and_save(ticker: str, client: PolygonClient) -> int:
    """Fetch and persist fundamentals for one ticker. Returns number of records saved."""
    records = fetch_fundamentals(ticker, client)
    if records:
        save_fundamentals(ticker, records)
        log.debug("fundamentals.saved", ticker=ticker, quarters=len(records))

    # Update last_financials_fetch in constituents (atomic write to avoid truncation)
    universe_file = _UNIVERSE_FILE
    if universe_file.exists():
        try:
            with open(universe_file) as f:
                constituents = json.load(f)
        except (json.JSONDecodeError, OSError):
            log.warning("fundamentals.constituents_read_error", ticker=ticker)
            return len(records)
        if ticker in constituents:
            constituents[ticker]["last_financials_fetch"] = date.today().isoformat()
            tmp = universe_file.with_suffix(".tmp")
            with open(tmp, "w") as f:
                json.dump(constituents, f, indent=2)
            tmp.replace(universe_file)

    # Keep earnings events in sync with updated fundamentals
    try:
        from src.collection import earnings as _earnings
        _earnings.refresh([ticker])
    except Exception as exc:
        log.debug("fundamentals.earnings_sync_failed", ticker=ticker, error=str(exc))

    return len(records)


def bulk_fetch(tickers: list[str], client: PolygonClient | None = None, cap: int | None = None) -> None:
    """
    Resumable bulk fetch for a list of tickers.
    Progress saved every 25 tickers in data.nosync/fundamentals/.init_state.json.
    """
    if client is None:
        client = PolygonClient()

    _FUNDAMENTALS_DIR.mkdir(parents=True, exist_ok=True)

    completed: set[str] = set()
    if _INIT_STATE_FILE.exists():
        with open(_INIT_STATE_FILE) as f:
            completed = set(json.load(f).get("completed", []))
        log.info("fundamentals.resuming", done=len(completed), remaining=len(tickers) - len(completed))

    remaining = [t for t in tickers if t not in completed]
    if cap:
        remaining = remaining[:cap]
    total = len(tickers)

    for i, ticker in enumerate(remaining):
        count = fetch_and_save(ticker, client)
        completed.add(ticker)

        if (i + 1) % 25 == 0 or (i + 1) == len(remaining):
            with open(_INIT_STATE_FILE, "w") as f:
                json.dump({"completed": sorted(completed)}, f)
            log.info(
                "fundamentals.progress",
                done=len(completed),
                total=total,
                pct=round(len(completed) / total * 100, 1),
            )

    log.info("fundamentals.bulk_complete", total=total)


if __name__ == "__main__":
    import sys
    from dotenv import load_dotenv

    load_dotenv()
    client = PolygonClient()

    # Parse --cap N flag
    cap: int | None = None
    args = sys.argv[1:]
    if "--cap" in args:
        idx = args.index("--cap")
        cap = int(args[idx + 1])
        args = [a for i, a in enumerate(args) if i != idx and i != idx + 1]

    if args:
        # Specific tickers passed as args
        tickers = [t.upper() for t in args]
        log.info("fundamentals.fetching_specific", tickers=tickers)
        for ticker in tickers:
            n = fetch_and_save(ticker, client)
            print(f"{ticker}: {n} quarters")
    else:
        # Bulk fetch for entire active universe
        if not _UNIVERSE_FILE.exists():
            print("No universe file found. Run init first.")
            sys.exit(1)
        with open(_UNIVERSE_FILE) as f:
            constituents = json.load(f)
        tickers = [t for t, r in constituents.items() if r.get("status") == "active"]
        log.info("fundamentals.bulk_starting", count=len(tickers), cap=cap)
        bulk_fetch(tickers, client=client, cap=cap)
