"""
action.py — Pre-decision pipeline

Run this Mon/Thu evening to download prices, process the queue,
run all selection models, and regenerate the dashboard.

After this completes:
  1. Review decisions/pending/YYYY-MM-DD.md (or run: python review.py)
  2. Check boxes for buys you want, uncheck any sells you want to keep
  3. Commit and push — triggers process-decisions workflow on GitHub

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
    ("Download prices",         ["python", "-m", "src.universe.ingestion"]),
    ("Process work queue",      ["python", "-m", "src.collection.process_queue"]),
    ("Refresh earnings data",   ["python", "-m", "src.collection.earnings_refresh"]),
    ("Run selection models",    ["python", "-m", "src.selection.runner"]),
    ("Evaluate strategies",     ["python", "-m", "src.strategy.runner"]),
    ("Regenerate dashboard",    ["python", "-m", "src.reports.daily"]),
]


def _prev_trading_day() -> str:
    d = date.today() - timedelta(days=1)
    while d.weekday() >= 5:
        d -= timedelta(days=1)
    return d.isoformat()


def _queue_preflight() -> None:
    """Print pending task counts by type before running the queue processor."""
    queue_file = _ROOT / "data/queue/pending.json"
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
    out = _ROOT / "data/prices" / f"{target}.json"
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
    universe_file = _ROOT / "data/universe/constituents.json"
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


_PREFLIGHTS: dict[str, object] = {
    "Download prices":      _prices_preflight,
    "Process work queue":   _queue_preflight,
    "Run selection models": _models_preflight,
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
    print("Next: python review.py  (or review decisions/pending/ and push your approved checkboxes.)")


if __name__ == "__main__":
    run()
