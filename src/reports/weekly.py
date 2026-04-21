"""
src/reports/weekly.py

Generates the weekly digest as docs/weekly.md.

Covers the trailing 7 days ending on week_ending (default: most recent Saturday).
For each model eval date in the window, shows every model's full pick list
with 1-week price change per ticker.

Called by weekly-digest.yml workflow (Saturday 10 AM ET).

Entry point:
    generate_weekly_digest(week_ending: str | None = None) -> str
    Returns the path to the written file.
"""

from __future__ import annotations

import json
import math
from datetime import date, timedelta
from pathlib import Path

import structlog

log = structlog.get_logger()

_DOCS_DIR   = Path("docs")
_MODELS_DIR = Path("data.nosync/models")
_PRICES_DIR = Path("data.nosync/prices")


def _week_range(week_ending: str) -> tuple[str, str]:
    end = date.fromisoformat(week_ending)
    start = end - timedelta(days=6)
    return start.isoformat(), week_ending


def _load_week_model_evals(week_start: str, week_end: str) -> dict[str, dict[str, list[dict]]]:
    """
    Returns {eval_date: {model_name: [holding, ...]}} for all eval dirs in the week.
    """
    if not _MODELS_DIR.exists():
        return {}
    result: dict[str, dict[str, list[dict]]] = {}
    for d in sorted(_MODELS_DIR.iterdir()):
        if not d.is_dir() or not d.name[0].isdigit():
            continue
        if not (week_start <= d.name <= week_end):
            continue
        models: dict[str, list[dict]] = {}
        for json_file in sorted(d.glob("*.json")):
            if json_file.stem.endswith("_ranks"):
                continue
            with open(json_file) as f:
                models[json_file.stem] = json.load(f)
        if models:
            result[d.name] = models
    return result


def _load_week_price_changes(week_start: str, week_end: str) -> dict[str, float]:
    """
    Returns {ticker: log_return} using the first and last populated price files
    within [week_start, week_end].
    """
    files = sorted(
        f for f in _PRICES_DIR.glob("*.json")
        if f.stem[0].isdigit() and week_start <= f.stem <= week_end and f.stat().st_size > 2
    )
    if len(files) < 2:
        return {}
    with open(files[0]) as f:
        early = {r["ticker"]: r["close"] for r in json.load(f) if r.get("close")}
    with open(files[-1]) as f:
        latest = {r["ticker"]: r["close"] for r in json.load(f) if r.get("close")}
    return {
        t: math.log(latest[t] / early[t])
        for t in latest
        if t in early and early[t] > 0
    }


def generate_weekly_digest(week_ending: str | None = None) -> str:
    """
    Generate docs/weekly.md. Returns the path to the written file.
    If week_ending is None, defaults to the most recent Saturday.
    """
    if week_ending is None:
        today = date.today()
        days_since_saturday = (today.weekday() - 5) % 7
        week_ending = (today - timedelta(days=days_since_saturday)).isoformat()

    week_start, week_end = _week_range(week_ending)

    evals        = _load_week_model_evals(week_start, week_end)
    price_changes = _load_week_price_changes(week_start, week_end)

    lines: list[str] = [
        f"# Weekly Model Report — Week ending {week_end}",
        f"_Period: {week_start} → {week_end}_",
        "",
    ]

    if not evals:
        lines.append("_No model evaluations this week._")
    else:
        for eval_date, models in sorted(evals.items()):
            lines += [f"## {eval_date}", ""]

            for model_name, holdings in sorted(models.items()):
                active = [h for h in holdings if h.get("status") != "sell"]
                exits  = [h for h in holdings if h.get("status") == "sell"]
                new_count  = sum(1 for h in active if h.get("status") == "new_buy")
                hold_count = sum(1 for h in active if h.get("status") == "hold")

                header = f"### {model_name} — {hold_count} hold"
                if new_count:
                    header += f", +{new_count} new"
                if exits:
                    header += f", -{len(exits)} exit"
                lines += [header, ""]

                if active:
                    lines.append("| Ticker | Status | 1W Return | Rationale |")
                    lines.append("|---|---|---|---|")
                    for h in sorted(active, key=lambda x: -x.get("conviction", 0)):
                        ret = price_changes.get(h["ticker"])
                        ret_str = f"{math.exp(ret)-1:+.1%}" if ret is not None else "—"
                        status  = h.get("status", "hold").upper().replace("_", " ")
                        rat     = h.get("rationale", "").replace("|", "\\|")
                        if len(rat) > 80:
                            rat = rat[:77] + "…"
                        lines.append(f"| {h['ticker']} | {status} | {ret_str} | {rat} |")
                    lines.append("")

                if exits:
                    exit_tickers = ", ".join(h["ticker"] for h in exits)
                    lines.append(f"_Exits: {exit_tickers}_")
                    lines.append("")

        # Multi-model confluence across all eval dates in the week
        ticker_models: dict[str, set[str]] = {}
        for models in evals.values():
            for model_name, holdings in models.items():
                for h in holdings:
                    if h.get("status") != "sell":
                        ticker_models.setdefault(h["ticker"], set()).add(model_name)
        multi = {t: m for t, m in ticker_models.items() if len(m) >= 2}

        lines += ["## Multi-Model Confluence", ""]
        if multi:
            for ticker, models in sorted(multi.items(), key=lambda x: -len(x[1])):
                ret = price_changes.get(ticker)
                ret_str = f" ({math.exp(ret)-1:+.1%})" if ret is not None else ""
                lines.append(f"- **{ticker}**{ret_str} — {', '.join(sorted(models))}")
        else:
            lines.append("_No tickers recommended by multiple models this week._")
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
