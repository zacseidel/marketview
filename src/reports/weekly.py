"""
src/reports/weekly.py

Generates the weekly digest as docs/weekly.md.

Covers the trailing 7 days ending on week_ending (default: most recent Saturday):
  - Decisions made (buys/sells approved)
  - Model activity (new buys, exits, hold counts)
  - Model scorecards (hit rate, avg return)
  - Portfolio summary (open/closed positions, P&L)
  - Strategy returns table (all-time)

Called by weekly-digest.yml workflow (Saturday 10 AM ET).

Entry point:
    generate_weekly_digest(week_ending: str | None = None) -> str
    Returns the path to the written file.
"""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import structlog

log = structlog.get_logger()

_DOCS_DIR = Path("docs")
_MODELS_DIR = Path("data.nosync/models")
_DECISIONS_DIR = Path("data.nosync/decisions")
_POSITIONS_FILE = Path("data.nosync/positions/positions.json")
_SCORECARDS_DIR = Path("data.nosync/models/scorecards")
_RETURNS_FILE = Path("data.nosync/strategy_observations/returns.json")


def _week_range(week_ending: str) -> tuple[str, str]:
    end = date.fromisoformat(week_ending)
    start = end - timedelta(days=6)
    return start.isoformat(), week_ending


def _load_week_decisions(week_start: str, week_end: str) -> list[dict]:
    if not _DECISIONS_DIR.exists():
        return []
    records: list[dict] = []
    for f in sorted(_DECISIONS_DIR.glob("*.json")):
        if week_start <= f.stem <= week_end:
            with open(f) as fp:
                records.extend(json.load(fp))
    return records


def _load_week_model_evals(week_start: str, week_end: str) -> dict[str, list[dict]]:
    """Collect model holdings from eval dirs that fall within the week."""
    if not _MODELS_DIR.exists():
        return {}
    result: dict[str, list[dict]] = {}
    for d in _MODELS_DIR.iterdir():
        if d.is_dir() and week_start <= d.name <= week_end:
            for json_file in sorted(d.glob("*.json")):
                model = json_file.stem
                with open(json_file) as f:
                    holdings = json.load(f)
                result.setdefault(model, []).extend(holdings)
    return result


def _load_scorecards() -> dict[str, dict]:
    if not _SCORECARDS_DIR.exists():
        return {}
    result: dict[str, dict] = {}
    for f in sorted(_SCORECARDS_DIR.glob("*.json")):
        with open(f) as fp:
            data = json.load(fp)
        result[data["model"]] = data
    return result


def _load_positions() -> tuple[list[dict], list[dict]]:
    if not _POSITIONS_FILE.exists():
        return [], []
    with open(_POSITIONS_FILE) as f:
        positions = json.load(f)
    open_pos = [p for p in positions if p.get("status") == "open"]
    closed_pos = [p for p in positions if p.get("status") == "closed"]
    return open_pos, closed_pos


def _load_strategy_returns() -> dict:
    if not _RETURNS_FILE.exists():
        return {}
    with open(_RETURNS_FILE) as f:
        return json.load(f)


def generate_weekly_digest(week_ending: str | None = None) -> str:
    """
    Generate docs/weekly.md. Returns the path to the written file.
    If week_ending is None, defaults to the most recent Saturday.
    """
    if week_ending is None:
        today = date.today()
        # Saturday = weekday 5; shift back to most recent Saturday
        days_since_saturday = (today.weekday() - 5) % 7
        week_ending = (today - timedelta(days=days_since_saturday)).isoformat()

    week_start, week_end = _week_range(week_ending)

    decisions = _load_week_decisions(week_start, week_end)
    model_evals = _load_week_model_evals(week_start, week_end)
    scorecards = _load_scorecards()
    open_pos, closed_pos = _load_positions()
    strategy_returns = _load_strategy_returns()

    buys = [d for d in decisions if d.get("action") == "buy" and d.get("user_approved")]
    sells = [d for d in decisions if d.get("action") == "sell" and d.get("user_approved")]

    lines: list[str] = [
        f"# Weekly Digest — Week ending {week_end}",
        f"_Period: {week_start} → {week_end}_",
        "",
    ]

    # --- Decisions ---
    lines += ["## Decisions This Week", ""]

    if buys:
        lines.append(f"**Buys ({len(buys)}):**")
        for d in buys:
            models_str = ", ".join(d.get("recommending_models", []))
            price_str = f"${d['execution_price']:.2f}" if d.get("execution_price") else "pending"
            lines.append(f"- {d['ticker']} — {price_str} | {models_str}")
    else:
        lines.append("_No buys this week._")
    lines.append("")

    if sells:
        lines.append(f"**Sells ({len(sells)}):**")
        for d in sells:
            price_str = f"${d['execution_price']:.2f}" if d.get("execution_price") else "pending"
            lines.append(f"- {d['ticker']} — {price_str}")
    else:
        lines.append("_No sells this week._")
    lines.append("")

    # --- Model Activity ---
    lines += ["## Model Activity", ""]
    if model_evals:
        for model, holdings in sorted(model_evals.items()):
            new_buys = [h for h in holdings if h.get("status") == "new_buy"]
            exits = [h for h in holdings if h.get("status") == "sell"]
            holds = [h for h in holdings if h.get("status") == "hold"]
            lines.append(
                f"**{model}**: {len(holds)} holds, "
                f"+{len(new_buys)} new buys, "
                f"-{len(exits)} exits"
            )
            if new_buys:
                tickers = ", ".join(h["ticker"] for h in new_buys[:10])
                lines.append(f"  _New: {tickers}_")
    else:
        lines.append("_No model evaluations this week._")
    lines.append("")

    # --- Model Scorecards ---
    lines += ["## Model Scorecards", ""]
    if scorecards:
        lines.append("| Model | Signals | Hit Rate | Avg Return |")
        lines.append("|---|---|---|---|")
        for model, sc in sorted(scorecards.items()):
            hit_str = f"{sc['hit_rate']*100:.1f}%" if sc.get("hit_rate") is not None else "—"
            avg_str = f"{sc['avg_return']*100:+.2f}%" if sc.get("avg_return") is not None else "—"
            lines.append(f"| {model} | {sc.get('signal_count', 0)} | {hit_str} | {avg_str} |")
    else:
        lines.append("_No scorecards yet — run `python -m src.tracking.model_scorecard`._")
    lines.append("")

    # --- Portfolio Summary ---
    total_unrealized = sum(p.get("unrealized_pnl") or 0.0 for p in open_pos)
    total_realized = sum(p.get("realized_pnl") or 0.0 for p in closed_pos)

    lines += ["## Portfolio", ""]
    lines.append(f"- Open positions: {len(open_pos)}")
    lines.append(f"- Closed positions: {len(closed_pos)}")
    lines.append(f"- Unrealized P&L: ${total_unrealized:+.2f}")
    lines.append(f"- Realized P&L: ${total_realized:+.2f}")
    lines.append(f"- Total P&L: ${total_unrealized + total_realized:+.2f}")
    lines.append("")

    if open_pos:
        lines.append("**Open positions:**")
        for p in sorted(open_pos, key=lambda x: x.get("entry_date", ""), reverse=True):
            pnl = p.get("unrealized_pnl")
            pnl_str = f"${pnl:+.2f}" if pnl is not None else "—"
            lines.append(
                f"- {p['ticker']} ({p.get('strategy','stock')}) "
                f"entered {p['entry_date']} @ ${p.get('entry_price',0):.2f} — {pnl_str}"
            )
        lines.append("")

    # --- Strategy Returns ---
    lines += ["## Strategy Returns (All Time)", ""]
    if strategy_returns:
        lines.append("| Model | Strategy | N | Mean Log Return | Win Rate |")
        lines.append("|---|---|---|---|---|")
        for model in sorted(strategy_returns):
            for strategy, stats in sorted(strategy_returns[model].items()):
                win_str = f"{stats['win_rate']*100:.0f}%" if stats.get("win_rate") is not None else "—"
                lines.append(
                    f"| {model} | {strategy} | {stats['count']} "
                    f"| {stats['mean_log_return']:+.4f} | {win_str} |"
                )
    else:
        lines.append("_No closed strategy observations yet._")
    lines.append("")

    lines.append(f"_Generated {date.today().isoformat()}_")

    content = "\n".join(lines) + "\n"

    _DOCS_DIR.mkdir(exist_ok=True)
    out_path = _DOCS_DIR / "weekly.md"
    out_path.write_text(content)

    log.info("weekly.generated", path=str(out_path), week_ending=week_end)
    return str(out_path)


if __name__ == "__main__":
    import sys
    from dotenv import load_dotenv
    load_dotenv()

    week_arg = sys.argv[1] if len(sys.argv) > 1 else None
    path = generate_weekly_digest(week_arg)
    print(f"Written: {path}")
