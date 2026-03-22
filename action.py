"""
action.py — Pre-decision pipeline

Run this Mon/Thu evening to download prices, process the queue,
run all selection models, and regenerate the dashboard.

After this completes:
  1. Review decisions/pending/YYYY-MM-DD.md
  2. Check boxes for buys you want, uncheck any sells you want to keep
  3. Commit and push — triggers process-decisions workflow on GitHub

Usage:
    python action.py
"""

from __future__ import annotations

import subprocess
import sys
from datetime import datetime

STEPS = [
    ("Download prices",         ["python", "-m", "src.universe.ingestion"]),
    ("Process work queue",      ["python", "-m", "src.collection.process_queue"]),
    ("Run selection models",    ["python", "-m", "src.selection.runner"]),
    ("Regenerate dashboard",    ["python", "-m", "src.reports.daily"]),
]


def run() -> None:
    print(f"\n=== action.py — {datetime.now().strftime('%Y-%m-%d %H:%M')} ===\n")
    failed = False

    for label, cmd in STEPS:
        print(f"[{label}]")
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"  FAILED (exit {result.returncode}) — stopping\n")
            failed = True
            break
        print(f"  done\n")

    if failed:
        print("Pipeline stopped early. Fix the error above and re-run.")
        sys.exit(1)

    print("=== Done ===")
    print("Next: review decisions/pending/ and push your approved checkboxes.")


if __name__ == "__main__":
    run()
