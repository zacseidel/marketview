"""
action.py — Model pipeline

Run on Tue/Fri: downloads prices, processes the queue, refreshes earnings,
runs all selection models, and regenerates the dashboard. Prints a
recommendations summary when the pipeline finishes.

Usage:
    python action.py
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path

_ROOT = Path(__file__).parent.resolve()

STEPS = [
    ("Download prices",      ["python", "-m", "src.universe.ingestion"]),
    ("Process work queue",   ["python", "-m", "src.collection.process_queue"]),
    ("Refresh earnings data",["python", "-m", "src.collection.earnings_refresh"]),
    ("Run selection models", ["python", "-m", "src.selection.runner"]),
    ("Regenerate dashboard", ["python", "-m", "src.reports.daily"]),
]


def _prev_trading_day() -> str:
    d = date.today() - timedelta(days=1)
    while d.weekday() >= 5:
        d -= timedelta(days=1)
    return d.isoformat()


def _queue_preflight() -> None:
    """Print pending task counts by type before running the queue processor."""
    queue_file = _ROOT / "data.nosync/queue/pending.json"
    if not queue_file.exists():
        print("  Queue: empty")
        return
    with open(queue_file) as f:
        tasks = json.load(f)
    pending = [t for t in tasks if t.get("status") in ("pending", "ready")]
    if not pending:
        print("  Queue: no pending tasks")
        return
    by_type: dict[str, int] = {}
    for t in pending:
        by_type[t["task_type"]] = by_type.get(t["task_type"], 0) + 1
    summary = ", ".join(f"{k}: {v}" for k, v in sorted(by_type.items()))
    print(f"  Pending tasks: {len(pending)}  ({summary})")


def _models_preflight() -> None:
    """Print which models are enabled before running selection."""
    try:
        import yaml
        config_file = _ROOT / "config/models.yaml"
        with open(config_file) as f:
            cfg = yaml.safe_load(f)
        enabled = [name for name, mc in cfg["models"].items() if mc.get("enabled", False)]
        print(f"  Models: {', '.join(enabled)}")
    except Exception:
        pass


def _prices_preflight() -> None:
    """Print the target date and whether it will be a fresh fetch or skip."""
    target = _prev_trading_day()
    out = _ROOT / "data.nosync/prices" / f"{target}.json"
    if out.exists():
        print(f"  Target date: {target}  (already exists — will skip)")
    else:
        print(f"  Target date: {target}  (fetching)")

_STALE_DAYS = 90
_STALE_THRESHOLD = 50  # warn if more than this many tickers are stale


def _check_fundamentals() -> None:
    """
    Warn if a significant number of tickers haven't had fundamentals fetched
    in over 90 days. Offer to kick off a capped background batch run.
    """
    universe_file = _ROOT / "data.nosync/universe/constituents.json"
    if not universe_file.exists():
        return

    with open(universe_file) as f:
        constituents = json.load(f)

    cutoff = (date.today() - timedelta(days=_STALE_DAYS)).isoformat()
    stale = [
        t for t, r in constituents.items()
        if r.get("status") == "active"
        and (not r.get("last_financials_fetch") or r["last_financials_fetch"] < cutoff)
    ]

    if len(stale) < _STALE_THRESHOLD:
        return

    print(f"  ⚠  {len(stale)} tickers have fundamentals older than {_STALE_DAYS} days.")
    print(f"     This affects the repurchase model and future earnings signals.")
    print(f"     Run in batches of 100 (~20 min each) until caught up:")
    print(f"       python -m src.collection.fundamentals --cap 100")
    print()
    ans = input("  Run a batch of 100 now in the background? [y/N] ").strip().lower()
    if ans == "y":
        proc = subprocess.Popen(
            ["python", "-m", "src.collection.fundamentals", "--cap", "100"],
            cwd=_ROOT,
        )
        print(f"  Fundamentals fetch started (PID {proc.pid}) — running in background.")
        print(f"  You can commit the results after action.py finishes.\n")
    else:
        print(f"  Skipping — run manually when convenient.\n")


def _print_recommendations() -> None:
    """Print a concise summary of the latest model recommendations."""
    models_dir = _ROOT / "data.nosync/models"
    if not models_dir.exists():
        return
    eval_dirs = sorted(
        [d for d in models_dir.iterdir() if d.is_dir() and d.name[0].isdigit()],
        reverse=True,
    )
    if not eval_dirs:
        return
    latest = eval_dirs[0]

    # Load price changes for 1W return column
    prices_dir = _ROOT / "data.nosync/prices"
    price_files = sorted(f for f in prices_dir.glob("*.json") if f.stem[0].isdigit())
    populated = [f for f in reversed(price_files) if f.stat().st_size > 2][:6]
    week_rets: dict[str, float] = {}
    if len(populated) >= 2:
        import math
        with open(populated[-1]) as f:
            early = {r["ticker"]: r["close"] for r in json.load(f) if r.get("close")}
        with open(populated[0]) as f:
            latest_prices = {r["ticker"]: r["close"] for r in json.load(f) if r.get("close")}
        week_rets = {
            t: math.log(latest_prices[t] / early[t])
            for t in latest_prices if t in early and early[t] > 0
        }

    print(f"\n{'═'*64}")
    print(f"  RECOMMENDATIONS — {latest.name}")
    print(f"{'═'*64}")

    # Track confluence
    ticker_models: dict[str, list[str]] = {}

    for json_file in sorted(latest.glob("*.json")):
        if json_file.stem.endswith("_ranks") or json_file.stem.endswith("_universe") or json_file.stem.endswith("_overflow"):
            continue
        with open(json_file) as f:
            holdings = json.load(f)

        active = [h for h in holdings if h.get("status") != "sell"]
        sells  = [h for h in holdings if h.get("status") == "sell"]
        new_count = sum(1 for h in active if h.get("status") == "new_buy")

        for h in active:
            ticker_models.setdefault(h["ticker"], []).append(json_file.stem)

        header = f"  {json_file.stem.upper()}"
        counts = f"{len(active)} holding{'s' if len(active) != 1 else ''}"
        if new_count:
            counts += f"  (+{new_count} new)"
        if not active and not sells:
            print(f"\n{header}  — no signals")
            continue
        print(f"\n{header}  {counts}")

        status_label = {"new_buy": "NEW ", "hold": "HOLD", "sell": "SELL"}
        sort_key = (lambda x: x["ticker"]) if json_file.stem == "munger" else (lambda x: -x.get("conviction", 0))
        for h in sorted(active, key=sort_key):
            status = status_label.get(h.get("status", "hold"), "    ")
            rat = h.get("rationale", "")[:62]
            ret = week_rets.get(h["ticker"])
            ret_str = f"  {ret:+.1%}" if ret is not None else ""
            print(f"    {status}  {h['ticker']:<7}{ret_str:<9}  {rat}")

        if sells:
            sell_tickers = ", ".join(h["ticker"] for h in sells)
            print(f"    EXIT  {sell_tickers}")

    # Confluence
    multi = {t: m for t, m in ticker_models.items() if len(m) >= 2}
    if multi:
        print(f"\n  Confluence:")
        for ticker, mods in sorted(multi.items(), key=lambda x: -len(x[1])):
            ret = week_rets.get(ticker)
            ret_str = f" {ret:+.1%}" if ret is not None else ""
            print(f"    {ticker}{ret_str}  —  {', '.join(sorted(mods))}")

    # Munger overflow: qualifiers dropped by max_holdings cap
    overflow_file = latest / "munger_overflow.json"
    if overflow_file.exists():
        with open(overflow_file) as f:
            overflow = json.load(f)
        dropped = overflow.get("dropped", [])
        if dropped:
            cap = overflow.get("cap", "?")
            tickers = ", ".join(r["ticker"] for r in dropped)
            print(f"\n  Note (Munger): {len(dropped)} qualifier(s) dropped by {cap}-ticker cap — {tickers}")

    print(f"\n{'═'*64}\n")


_PREFLIGHTS: dict[str, object] = {
    "Download prices":     _prices_preflight,
    "Process work queue":  _queue_preflight,
    "Run selection models":_models_preflight,
}


def run() -> None:
    print(f"\n=== action.py — {datetime.now().strftime('%Y-%m-%d %H:%M')} ===\n")

    _check_fundamentals()

    pipeline_start = time.time()
    failed = False
    for label, cmd in STEPS:
        print(f"[{label}]")
        preflight = _PREFLIGHTS.get(label)
        if preflight:
            preflight()
        t0 = time.time()
        result = subprocess.run(cmd, cwd=_ROOT)
        elapsed = time.time() - t0
        if result.returncode != 0:
            print(f"  FAILED (exit {result.returncode}) after {elapsed:.0f}s — stopping\n")
            failed = True
            break
        print(f"  done  ({elapsed:.0f}s)\n")

    total = time.time() - pipeline_start
    if failed:
        print(f"Pipeline stopped early after {total:.0f}s. Fix the error above and re-run.")
        sys.exit(1)

    print(f"=== Done in {total:.0f}s ===")
    _print_recommendations()


if __name__ == "__main__":
    run()
