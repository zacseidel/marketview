"""
finish.py — Post-execution pipeline

Run this Tue/Fri after execution day to process approved decisions,
record fills, evaluate strategies, update P&L, and regenerate the dashboard.

Requires:
  - decisions/pending/YYYY-MM-DD.md already pushed and processed by GitHub Actions
    (or pass --date YYYY-MM-DD to process a specific local decision file)

Usage:
    python finish.py
    python finish.py --date 2026-03-21
"""

from __future__ import annotations

import subprocess
import sys
from datetime import datetime


def run() -> None:
    date_arg = None
    if "--date" in sys.argv:
        idx = sys.argv.index("--date")
        if idx + 1 < len(sys.argv):
            date_arg = sys.argv[idx + 1]

    print(f"\n=== finish.py — {datetime.now().strftime('%Y-%m-%d %H:%M')} ===\n")

    steps = [
        ("Process decision file",       ["python", "-m", "src.decisions.process"] + ([date_arg] if date_arg else [])),
        ("Record fills",                ["python", "-m", "src.decisions.execute"] + ([date_arg] if date_arg else [])),
        ("Evaluate strategies",         ["python", "-m", "src.strategy.runner"]),
        ("Update position P&L",         ["python", "-m", "src.tracking.pnl"]),
        ("Update portfolio history",    ["python", "-m", "src.tracking.portfolio"]),
        ("Regenerate dashboard",        ["python", "-m", "src.reports.daily"]),
    ]

    failed = False
    for label, cmd in steps:
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
    print("Dashboard updated. Open docs/index.html to review positions.")


if __name__ == "__main__":
    run()
