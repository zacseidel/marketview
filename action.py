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
from datetime import date, datetime, timedelta
from pathlib import Path

_ROOT = Path(__file__).parent.resolve()

STEPS = [
    ("Download prices",         ["python", "-m", "src.universe.ingestion"]),
    ("Process work queue",      ["python", "-m", "src.collection.process_queue"]),
    ("Run selection models",    ["python", "-m", "src.selection.runner"]),
    ("Regenerate dashboard",    ["python", "-m", "src.reports.daily"]),
]

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


def run() -> None:
    print(f"\n=== action.py — {datetime.now().strftime('%Y-%m-%d %H:%M')} ===\n")

    _check_fundamentals()

    failed = False
    for label, cmd in STEPS:
        print(f"[{label}]")
        result = subprocess.run(cmd, cwd=_ROOT)
        if result.returncode != 0:
            print(f"  FAILED (exit {result.returncode}) — stopping\n")
            failed = True
            break
        print(f"  done\n")

    if failed:
        print("Pipeline stopped early. Fix the error above and re-run.")
        sys.exit(1)

    print("=== Done ===")
    print("Next: python review.py  (or review decisions/pending/ and push your approved checkboxes.)")


if __name__ == "__main__":
    run()
