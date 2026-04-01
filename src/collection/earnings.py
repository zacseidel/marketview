"""
src/collection/earnings.py

Builds and maintains per-ticker earnings event files at data/earnings/{ticker}.json.

Each event record combines:
  - Actual announcement date + EPS estimate/actual/surprise from yfinance (one-time backfill)
  - Revenue, net income, growth rates from Polygon fundamentals (updated quarterly)
  - Post-announcement price reactions computed from price history

Two entry points:
  backfill_all()  — one-time: fetches yfinance history for all tickers, merges with
                    existing fundamentals and price history. Resume-safe.
  refresh(tickers) — ongoing: re-derives events from already-stored yfinance data +
                     updated fundamentals. Called automatically from fundamentals.py
                     after each fetch. No new yfinance calls.

Usage:
    python -m src.collection.earnings --backfill       # one-time historical pull
    python -m src.collection.earnings                  # refresh all from stored data
    python -m src.collection.earnings AAPL MSFT        # specific tickers
"""

from __future__ import annotations

import json
import math
import time
from datetime import date, datetime
from pathlib import Path

import structlog

log = structlog.get_logger()


def _clean_float(v) -> float | None:
    """Convert pandas NA / None to None, otherwise to float."""
    if v is None:
        return None
    try:
        import pandas as pd
        if pd.isna(v):
            return None
    except Exception:
        pass
    return float(v)


_EARNINGS_DIR     = Path("data/earnings")
_FUNDAMENTALS_DIR = Path("data/fundamentals")
_UNIVERSE_FILE    = Path("data/universe/constituents.json")
_RAW_PRICES_FILE  = Path("data/quant/raw_prices.parquet")
_STATE_FILE       = _EARNINGS_DIR / ".backfill_state.json"


# ---------------------------------------------------------------------------
# yfinance helpers (backfill only)
# ---------------------------------------------------------------------------

def _fetch_yfinance_earnings(ticker: str) -> list[dict]:
    """
    Fetch earnings history for one ticker from yfinance.
    Returns list of dicts: {announcement_date, eps_estimate, eps_actual, surprise_pct}
    sorted most-recent-first. Returns [] on any error.
    """
    try:
        import yfinance as yf
        t = yf.Ticker(ticker)
        df = t.earnings_dates
        if df is None or df.empty:
            return []

        # earnings_dates index is the announcement datetime (tz-aware)
        df = df.reset_index()
        date_col = df.columns[0]  # "Earnings Date" or similar
        results = []
        for _, row in df.iterrows():
            try:
                ann_dt = row[date_col]
                if hasattr(ann_dt, "date"):
                    ann_date = ann_dt.date().isoformat()
                else:
                    ann_date = str(ann_dt)[:10]

                eps_est    = _clean_float(row.get("EPS Estimate"))
                eps_actual = _clean_float(row.get("Reported EPS"))
                surprise   = _clean_float(row.get("Surprise(%)"))

                # Compute surprise_pct ourselves if we have both values
                surprise_pct = None
                if eps_est is not None and eps_actual is not None and eps_est != 0:
                    surprise_pct = round((eps_actual - eps_est) / abs(eps_est), 4)
                elif surprise is not None:
                    surprise_pct = round(surprise / 100, 4)

                results.append({
                    "announcement_date": ann_date,
                    "eps_estimate":      round(eps_est, 4) if eps_est is not None else None,
                    "eps_actual":        round(eps_actual, 4) if eps_actual is not None else None,
                    "surprise_pct":      surprise_pct,
                })
            except Exception:
                continue

        results.sort(key=lambda r: r["announcement_date"], reverse=True)
        return results

    except Exception as exc:
        log.debug("earnings.yfinance_error", ticker=ticker, error=str(exc))
        return []


# ---------------------------------------------------------------------------
# Growth rate computation
# ---------------------------------------------------------------------------

def _compute_growth_rates(quarters: list[dict]) -> list[dict]:
    """
    For each quarterly record, compute:
      ni_yoy_growth, ni_qoq_growth, rev_yoy_growth, ni_acceleration
    Adds these fields in-place and returns the list.
    Only processes records with period starting Q (quarterly).
    """
    # Filter to actual quarterly records, sort oldest-first for iteration
    q_only = [
        q for q in quarters
        if not q.get("period", "").startswith(("FY", "TTM"))
        and q.get("filing_date")
    ]
    q_only.sort(key=lambda q: q["filing_date"])

    for i, q in enumerate(q_only):
        ni  = q.get("net_income")
        rev = q.get("revenue")

        # QoQ (previous quarter)
        ni_qoq = None
        if i >= 1:
            prev = q_only[i - 1]
            prev_ni = prev.get("net_income")
            if ni is not None and prev_ni is not None and prev_ni != 0 and ni != 0:
                # Only compute when signs match — sign changes (loss→profit or vice
                # versa) are not meaningful as a log return and are left as None.
                if (ni > 0) == (prev_ni > 0):
                    ni_qoq = round(math.log(abs(ni) / abs(prev_ni)) * (1 if ni >= 0 else -1), 4)

        # YoY (4 quarters back)
        ni_yoy  = None
        rev_yoy = None
        ni_yoy_prior = None  # for acceleration
        if i >= 4:
            yoy = q_only[i - 4]
            yoy_ni  = yoy.get("net_income")
            yoy_rev = yoy.get("revenue")
            if ni is not None and yoy_ni is not None and yoy_ni != 0 and ni != 0 and yoy_ni > 0 and ni > 0:
                ni_yoy = round(math.log(ni / yoy_ni), 4)
            if rev is not None and yoy_rev is not None and yoy_rev > 0 and rev > 0:
                rev_yoy = round(math.log(rev / yoy_rev), 4)

        # Prior-year YoY growth (for acceleration = current - prior)
        if i >= 5:
            prior = q_only[i - 1]
            prior_ni     = prior.get("net_income")
            prior_yoy_ni = q_only[i - 5].get("net_income")
            if prior_ni is not None and prior_yoy_ni is not None and prior_yoy_ni > 0 and prior_ni > 0:
                ni_yoy_prior = round(math.log(prior_ni / prior_yoy_ni), 4)

        ni_acceleration = None
        if ni_yoy is not None and ni_yoy_prior is not None:
            ni_acceleration = round(ni_yoy - ni_yoy_prior, 4)

        q["ni_yoy_growth"]  = ni_yoy
        q["ni_qoq_growth"]  = ni_qoq
        q["rev_yoy_growth"] = rev_yoy
        q["ni_acceleration"] = ni_acceleration

    return q_only


# ---------------------------------------------------------------------------
# Price reaction computation
# ---------------------------------------------------------------------------

def _compute_price_reactions(event_date: str, df) -> dict:
    """
    Given an event date and a pre-processed single-ticker price DataFrame
    (sorted by date, with a date_str column), compute price reactions.
    Returns dict with price_on_event_date, price_5d_after, price_20d_after,
    earn_ret_5d, earn_ret_5d_to_20d. All None if dates not in data.
    """
    result = {
        "price_on_event_date": None,
        "price_5d_after":      None,
        "price_20d_after":     None,
        "earn_ret_5d":         None,
        "earn_ret_5d_to_20d":  None,
    }

    if df is None or df.empty:
        return result

    # Find the index of the first trading day on or after event_date
    on_or_after = df[df["date_str"] >= event_date]
    if on_or_after.empty:
        return result

    base_pos = on_or_after.index[0]  # 0-based position (df was reset_index'd above)
    p0 = df.iloc[base_pos]["close"]
    result["price_on_event_date"] = round(float(p0), 4)

    # +5 trading days
    if base_pos + 5 < len(df):
        p5 = df.iloc[base_pos + 5]["close"]
        result["price_5d_after"] = round(float(p5), 4)
        if p0 > 0 and p5 > 0:
            result["earn_ret_5d"] = round(math.log(p5 / p0), 4)

    # +20 trading days
    if base_pos + 20 < len(df):
        p20 = df.iloc[base_pos + 20]["close"]
        result["price_20d_after"] = round(float(p20), 4)
        p5_val = result["price_5d_after"]
        if p5_val and p5_val > 0 and p20 > 0:
            result["earn_ret_5d_to_20d"] = round(math.log(p20 / p5_val), 4)

    return result


# ---------------------------------------------------------------------------
# Event builder
# ---------------------------------------------------------------------------

def build_earnings_events(
    ticker: str,
    fundamentals: list[dict],
    prices_df,
    yf_events: list[dict],
) -> list[dict]:
    """
    Merge fundamentals + yfinance announcement data + price reactions into a
    unified list of earnings event dicts, sorted most-recent-first.
    """
    quarters = _compute_growth_rates([dict(q) for q in fundamentals])
    if not quarters:
        return []

    # Index yfinance events by announcement_date for fast lookup
    yf_by_date: dict[str, dict] = {e["announcement_date"]: e for e in yf_events}
    yf_dates_sorted = sorted(yf_by_date.keys(), reverse=True)

    # Pre-process prices once for all per-event lookups
    if prices_df is not None and not prices_df.empty:
        import pandas as pd
        prices_df = prices_df.sort_values("date").reset_index(drop=True)
        prices_df = prices_df.assign(date_str=prices_df["date"].dt.strftime("%Y-%m-%d"))
    else:
        prices_df = None

    events = []
    for q in quarters:
        filing_date = q.get("filing_date")
        if not filing_date:
            continue

        # Find yfinance event within 60 days before the filing date
        filing_dt = datetime.strptime(filing_date, "%Y-%m-%d")
        yf = None
        for yf_date in yf_dates_sorted:
            if yf_date <= filing_date:
                if (filing_dt - datetime.strptime(yf_date, "%Y-%m-%d")).days <= 60:
                    yf = yf_by_date[yf_date]
                break

        announcement_date = yf["announcement_date"] if yf else None
        event_date = announcement_date or filing_date

        price_rx = _compute_price_reactions(event_date, prices_df)

        event = {
            "ticker":              ticker,
            "period":              q.get("period"),
            "announcement_date":   announcement_date,
            "filing_date":         filing_date,
            "event_date":          event_date,
            "eps_estimate":        yf["eps_estimate"]   if yf else None,
            "eps_actual":          yf["eps_actual"]     if yf else None,
            "eps_surprise_pct":    yf["surprise_pct"]   if yf else None,
            "revenue":             q.get("revenue"),
            "net_income":          q.get("net_income"),
            "shares_outstanding":  q.get("shares_outstanding"),
            "ni_yoy_growth":       q.get("ni_yoy_growth"),
            "rev_yoy_growth":      q.get("rev_yoy_growth"),
            "ni_qoq_growth":       q.get("ni_qoq_growth"),
            "ni_acceleration":     q.get("ni_acceleration"),
            **price_rx,
        }
        events.append(event)

    events.sort(key=lambda e: e["event_date"], reverse=True)
    return events


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def _save_events(ticker: str, events: list[dict]) -> None:
    _EARNINGS_DIR.mkdir(parents=True, exist_ok=True)
    path = _EARNINGS_DIR / f"{ticker}.json"
    with open(path, "w") as f:
        json.dump(events, f, indent=2)


def _load_fundamentals(ticker: str) -> list[dict]:
    path = _FUNDAMENTALS_DIR / f"{ticker}.json"
    if not path.exists():
        return []
    with open(path) as f:
        return json.load(f)


_prices_cache: dict | None = None


def _get_prices_cache() -> dict:
    """Load raw_prices.parquet once and cache by ticker for the process lifetime."""
    global _prices_cache
    if _prices_cache is not None:
        return _prices_cache
    if not _RAW_PRICES_FILE.exists():
        _prices_cache = {}
        return _prices_cache
    try:
        import pandas as pd
        df = pd.read_parquet(_RAW_PRICES_FILE, columns=["ticker", "date", "close"])
        df["date"] = pd.to_datetime(df["date"])
        _prices_cache = {
            t: grp[["date", "close"]].reset_index(drop=True)
            for t, grp in df.groupby("ticker")
        }
        log.debug("earnings.prices_cache_loaded", tickers=len(_prices_cache))
    except Exception as exc:
        log.debug("earnings.prices_load_error", error=str(exc))
        _prices_cache = {}
    return _prices_cache


def _load_prices_df(ticker: str):
    """Return price history for a single ticker (from module-level cache)."""
    return _get_prices_cache().get(ticker)


def _load_stored_yf_events(ticker: str) -> list[dict]:
    """Load previously-fetched yfinance data from the earnings file."""
    path = _EARNINGS_DIR / f"{ticker}.json"
    if not path.exists():
        return []
    with open(path) as f:
        events = json.load(f)
    # Extract the yfinance fields from stored events
    yf_events = []
    for e in events:
        if e.get("announcement_date"):
            yf_events.append({
                "announcement_date": e["announcement_date"],
                "eps_estimate":      e.get("eps_estimate"),
                "eps_actual":        e.get("eps_actual"),
                "surprise_pct":      e.get("eps_surprise_pct"),
            })
    return yf_events


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def refresh(tickers: list[str]) -> None:
    """
    Re-derive earnings events for specified tickers from stored yfinance data +
    current fundamentals. No new yfinance calls. Called from fundamentals.py.
    """
    for ticker in tickers:
        fundamentals = _load_fundamentals(ticker)
        if not fundamentals:
            continue
        yf_events = _load_stored_yf_events(ticker)
        prices_df = _load_prices_df(ticker)
        events = build_earnings_events(ticker, fundamentals, prices_df, yf_events)
        if events:
            _save_events(ticker, events)
            log.debug("earnings.refreshed", ticker=ticker, events=len(events))


def backfill_all(tickers: list[str] | None = None) -> None:
    """
    One-time backfill: fetch yfinance earnings history for all tickers, merge with
    fundamentals and price history, write data/earnings/{ticker}.json.
    Resume-safe via data/earnings/.backfill_state.json.
    """
    if tickers is None:
        if not _UNIVERSE_FILE.exists():
            log.error("earnings.no_universe")
            return
        with open(_UNIVERSE_FILE) as f:
            constituents = json.load(f)
        tickers = [t for t, r in constituents.items() if r.get("status") == "active"]

    # Load resume state
    completed: set[str] = set()
    if _STATE_FILE.exists():
        with open(_STATE_FILE) as f:
            completed = set(json.load(f).get("completed", []))
        log.info("earnings.resuming", done=len(completed), remaining=len(tickers) - len(completed))

    remaining = [t for t in tickers if t not in completed]
    total = len(tickers)

    log.info("earnings.backfill_starting", total=total, remaining=len(remaining))
    _EARNINGS_DIR.mkdir(parents=True, exist_ok=True)

    for i, ticker in enumerate(remaining):
        fundamentals = _load_fundamentals(ticker)
        if not fundamentals:
            log.debug("earnings.no_fundamentals", ticker=ticker)
            completed.add(ticker)
            continue

        yf_events = _fetch_yfinance_earnings(ticker)
        prices_df = _load_prices_df(ticker)
        events = build_earnings_events(ticker, fundamentals, prices_df, yf_events)

        if events:
            _save_events(ticker, events)

        completed.add(ticker)
        time.sleep(0.5)  # be polite to yfinance

        if (i + 1) % 25 == 0 or (i + 1) == len(remaining):
            with open(_STATE_FILE, "w") as f:
                json.dump({"completed": sorted(completed)}, f)
            log.info(
                "earnings.progress",
                done=len(completed),
                total=total,
                pct=round(len(completed) / total * 100, 1),
            )

    log.info("earnings.backfill_complete", total=total)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from dotenv import load_dotenv
    load_dotenv()

    args = sys.argv[1:]
    backfill = "--backfill" in args
    tickers_arg = [a for a in args if not a.startswith("--")]

    if backfill:
        tickers = [t.upper() for t in tickers_arg] if tickers_arg else None
        backfill_all(tickers)
    elif tickers_arg:
        tickers = [t.upper() for t in tickers_arg]
        log.info("earnings.refreshing_specific", tickers=tickers)
        refresh(tickers)
        for t in tickers:
            path = _EARNINGS_DIR / f"{t}.json"
            if path.exists():
                with open(path) as f:
                    events = json.load(f)
                print(f"{t}: {len(events)} events")
                if events:
                    e = events[0]
                    print(f"  Latest: {e['event_date']}  NI YoY={e.get('ni_yoy_growth')}  "
                          f"EPS surprise={e.get('eps_surprise_pct')}  earn_ret_5d={e.get('earn_ret_5d')}")
    else:
        # Refresh all from stored data
        if not _UNIVERSE_FILE.exists():
            print("No universe file found.")
            sys.exit(1)
        with open(_UNIVERSE_FILE) as f:
            constituents = json.load(f)
        tickers = [t for t, r in constituents.items() if r.get("status") == "active"]
        log.info("earnings.refresh_all", count=len(tickers))
        refresh(tickers)
        count = sum(1 for t in tickers if (_EARNINGS_DIR / f"{t}.json").exists())
        print(f"Refreshed {count} earnings files.")
