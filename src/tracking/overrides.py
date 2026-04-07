"""
src/tracking/overrides.py

Records and scores user overrides of model recommendations (Layer 2 of judgment tracking).

An override is when the user deviates from the model default:
  - veto_buy:  model said new_buy, user rejected it (will not buy)
  - keep_sell: model said sell, user overrode it (will keep holding)

Each override is scored 20 trading days after the eval date by looking up prices:
  - veto_buy value:  -ticker_log_ret  (positive = you avoided a loss)
  - keep_sell value: ticker_log_ret   (positive = you were right to hold)

Both include SPY return over the same window for context.

Data: data.nosync/overrides/log.json
Entry points:
    record_override(eval_date, ticker, override_type, models) -> None
    score_pending_overrides() -> int  (returns number newly scored)
    get_all_overrides() -> list[dict]
"""

from __future__ import annotations

import json
import math
from datetime import date
from pathlib import Path

import structlog

log = structlog.get_logger()

_OVERRIDES_FILE = Path("data.nosync/overrides/log.json")
_PRICES_DIR     = Path("data.nosync/prices")

# Score after this many trading-day-equivalent price files
_SCORE_AFTER_DAYS = 20


def _load_overrides() -> list[dict]:
    if not _OVERRIDES_FILE.exists():
        return []
    with open(_OVERRIDES_FILE) as f:
        return json.load(f)


def _save_overrides(overrides: list[dict]) -> None:
    _OVERRIDES_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = _OVERRIDES_FILE.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(overrides, f, indent=2)
    tmp.replace(_OVERRIDES_FILE)


def _price_files_from(from_date: str) -> list[Path]:
    """Return price files at or after from_date, sorted ascending."""
    return sorted(
        f for f in _PRICES_DIR.glob("*.json")
        if f.stem[0].isdigit() and f.stem >= from_date
    )


def _price_on_date(ticker: str, target_date: str) -> float | None:
    """Return the close price for ticker on or after target_date."""
    candidates = sorted(
        f for f in _PRICES_DIR.glob("*.json")
        if f.stem[0].isdigit() and f.stem >= target_date
    )
    for pf in candidates[:5]:  # look at up to 5 files (skip holidays)
        with open(pf) as fp:
            records = json.load(fp)
        for r in records:
            if r.get("ticker") == ticker:
                return r.get("ohlc_avg") or r.get("close")
    return None


def record_override(
    eval_date: str,
    ticker: str,
    override_type: str,   # "veto_buy" | "keep_sell"
    models: list[str],
) -> None:
    """
    Append an override record. Idempotent: skips if this eval_date+ticker+type exists.
    """
    overrides = _load_overrides()
    oid = f"{eval_date}_{ticker}_{override_type}"
    if any(o.get("override_id") == oid for o in overrides):
        log.debug("overrides.already_recorded", override_id=oid)
        return

    entry: dict = {
        "override_id":   oid,
        "eval_date":     eval_date,
        "ticker":        ticker,
        "override_type": override_type,   # "veto_buy" | "keep_sell"
        "models":        models,
        "recorded_at":   date.today().isoformat(),
        "scored":        False,
        "score_date":    None,
        "entry_price":   _price_on_date(ticker, eval_date),
        "ticker_log_ret": None,
        "spy_log_ret":   None,
        "override_value": None,
    }
    overrides.append(entry)
    _save_overrides(overrides)
    log.info("overrides.recorded", override_id=oid, type=override_type)


def score_pending_overrides() -> int:
    """
    Score any unscored overrides that are at least _SCORE_AFTER_DAYS price files old.
    Returns the number of overrides newly scored.
    """
    overrides = _load_overrides()
    if not overrides:
        return 0

    newly_scored = 0
    for o in overrides:
        if o.get("scored"):
            continue

        eval_date = o["eval_date"]
        ticker    = o["ticker"]

        # Find the price file that is _SCORE_AFTER_DAYS trading days after eval_date
        future_files = _price_files_from(eval_date)
        if len(future_files) < _SCORE_AFTER_DAYS + 1:
            continue  # not enough data yet

        score_file  = future_files[_SCORE_AFTER_DAYS]
        score_date  = score_file.stem

        entry_price = o.get("entry_price")
        if not entry_price or entry_price <= 0:
            # Try to find it now
            entry_price = _price_on_date(ticker, eval_date)
            o["entry_price"] = entry_price
        if not entry_price or entry_price <= 0:
            continue

        # Look up exit price on score_date
        with open(score_file) as fp:
            records = json.load(fp)

        ticker_price = next((r.get("ohlc_avg") or r.get("close") for r in records if r.get("ticker") == ticker), None)
        spy_price    = next((r.get("ohlc_avg") or r.get("close") for r in records if r.get("ticker") == "SPY"), None)

        if not ticker_price or ticker_price <= 0:
            continue

        # SPY entry price on eval_date
        spy_entry = _price_on_date("SPY", eval_date)

        ticker_log_ret = round(math.log(ticker_price / entry_price), 6)
        spy_log_ret    = round(math.log(spy_price / spy_entry), 6) if (spy_price and spy_entry and spy_entry > 0) else None

        # Override value: positive means your override was correct
        override_type = o["override_type"]
        if override_type == "veto_buy":
            # You avoided buying; good if stock fell
            override_value = round(-ticker_log_ret, 6)
        elif override_type == "keep_sell":
            # You kept holding; good if stock rose
            override_value = round(ticker_log_ret, 6)
        else:
            override_value = None

        o["scored"]          = True
        o["score_date"]      = score_date
        o["ticker_log_ret"]  = ticker_log_ret
        o["spy_log_ret"]     = spy_log_ret
        o["override_value"]  = override_value
        newly_scored += 1
        log.info(
            "overrides.scored",
            override_id=o["override_id"],
            ticker_log_ret=ticker_log_ret,
            override_value=override_value,
        )

    if newly_scored:
        _save_overrides(overrides)

    return newly_scored


def get_all_overrides() -> list[dict]:
    return _load_overrides()


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    n = score_pending_overrides()
    print(f"Newly scored: {n}")

    overrides = get_all_overrides()
    scored = [o for o in overrides if o.get("scored")]
    pending = [o for o in overrides if not o.get("scored")]

    if scored:
        values = [o["override_value"] for o in scored if o.get("override_value") is not None]
        avg = sum(values) / len(values) if values else 0
        good = sum(1 for v in values if v > 0)
        print(f"\nScored overrides ({len(scored)}):")
        print(f"  Good: {good}/{len(values)}   Avg value: {avg:+.2%}")
        for o in sorted(scored, key=lambda x: x["eval_date"]):
            v = o.get("override_value")
            v_str = f"{v:+.2%}" if v is not None else "—"
            print(f"  {o['eval_date']}  {o['ticker']:<6}  {o['override_type']:<10}  {v_str}")

    if pending:
        print(f"\nPending (not yet scored): {len(pending)}")
        for o in pending:
            print(f"  {o['eval_date']}  {o['ticker']:<6}  {o['override_type']}")
