"""
src/decisions/generate.py

Generates the markdown decision file for user review after models run.
Writes decisions/pending/{eval_date}.md with checkbox-based action items.
"""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import structlog
import yaml

log = structlog.get_logger()

_MODELS_CONFIG = Path("config/models.yaml")
_DECISIONS_DIR = Path("decisions/pending")

_TEMPLATE = """\
# Evaluation: {eval_date} ({weekday})
Execute: {exec_day} {exec_date}

## New Buy Recommendations (uncheck to veto)
{new_buys}

## Current Holdings — Confirm Continue
{holds}

## Sell Recommendations (uncheck to override and keep)
{sells}
"""

_NO_ITEMS = "_None_"


def _exec_date(eval_date: str) -> tuple[str, str]:
    """Return (exec_date_iso, exec_day_name) for the next trading day after eval_date."""
    d = date.fromisoformat(eval_date)
    d += timedelta(days=1)
    while d.weekday() >= 5:
        d += timedelta(days=1)
    return d.isoformat(), d.strftime("%A")


def _weekday_name(eval_date: str) -> str:
    return date.fromisoformat(eval_date).strftime("%A")


def _load_enabled_models() -> list[str]:
    with open(_MODELS_CONFIG) as f:
        cfg = yaml.safe_load(f)
    return [name for name, mc in cfg["models"].items() if mc.get("enabled", False)]


def _load_all_model_outputs(eval_date: str) -> dict[str, list[dict]]:
    """Load {model: [holding_record, ...]} for all enabled models on eval_date."""
    models = _load_enabled_models()
    result: dict[str, list[dict]] = {}
    for model in models:
        path = Path("data.nosync/models") / eval_date / f"{model}.json"
        if path.exists():
            with open(path) as f:
                result[model] = json.load(f)
    return result


def _build_ticker_summary(all_outputs: dict[str, list[dict]]) -> dict[str, dict]:
    """
    Merge all model outputs into per-ticker view.
    Returns {ticker: {status, models: [{name, conviction}]}}
    """
    ticker_view: dict[str, dict] = {}

    for model_name, holdings in all_outputs.items():
        for h in holdings:
            ticker = h["ticker"]
            status = h["status"]

            if ticker not in ticker_view:
                ticker_view[ticker] = {"status": status, "models": []}

            # Status priority: sell < hold < new_buy
            existing_status = ticker_view[ticker]["status"]
            if status == "new_buy" or (status == "hold" and existing_status == "sell"):
                ticker_view[ticker]["status"] = status

            ticker_view[ticker]["models"].append({
                "name": model_name,
                "conviction": h["conviction"],
            })

    return ticker_view


def _format_ticker_line(ticker: str, models: list[dict], checked: bool) -> str:
    checkbox = "[x]" if checked else "[ ]"
    model_str = " + ".join(
        f"{m['name']} ({m['conviction']:.2f})"
        for m in sorted(models, key=lambda m: m["conviction"], reverse=True)
    )
    return f"- {checkbox} {ticker} — {model_str}"


def generate_decision_file(eval_date: str) -> str:
    """
    Generate decisions/pending/{eval_date}.md and return the file path.
    Idempotent: overwrites if it already exists.
    """
    _DECISIONS_DIR.mkdir(parents=True, exist_ok=True)

    all_outputs = _load_all_model_outputs(eval_date)
    if not all_outputs:
        log.warning("generate.no_model_outputs", eval_date=eval_date)
        return ""

    ticker_summary = _build_ticker_summary(all_outputs)
    exec_date, exec_day = _exec_date(eval_date)

    new_buy_lines: list[str] = []
    hold_lines: list[str] = []
    sell_lines: list[str] = []

    # Sort each section by: primary model name (highest conviction), then conviction desc
    def sort_key(kv: tuple) -> tuple:
        entry = kv[1]
        top_model = max(entry["models"], key=lambda m: m["conviction"], default={"name": "z", "conviction": 0.0})
        return (top_model["name"], -top_model["conviction"])

    for ticker, entry in sorted(ticker_summary.items(), key=sort_key):
        status = entry["status"]
        models = entry["models"]

        if status == "new_buy":
            new_buy_lines.append(_format_ticker_line(ticker, models, checked=True))
        elif status == "hold":
            hold_lines.append(_format_ticker_line(ticker, models, checked=True))
        elif status == "sell":
            sell_lines.append(_format_ticker_line(ticker, models, checked=True))

    content = _TEMPLATE.format(
        eval_date=eval_date,
        weekday=_weekday_name(eval_date),
        exec_day=exec_day,
        exec_date=exec_date,
        new_buys="\n".join(new_buy_lines) if new_buy_lines else _NO_ITEMS,
        holds="\n".join(hold_lines) if hold_lines else _NO_ITEMS,
        sells="\n".join(sell_lines) if sell_lines else _NO_ITEMS,
    )

    out_path = _DECISIONS_DIR / f"{eval_date}.md"
    out_path.write_text(content)
    log.info(
        "generate.written",
        path=str(out_path),
        new_buys=len(new_buy_lines),
        holds=len(hold_lines),
        sells=len(sell_lines),
    )
    return str(out_path)


if __name__ == "__main__":
    import sys
    eval_date = sys.argv[1] if len(sys.argv) > 1 else date.today().isoformat()
    path = generate_decision_file(eval_date)
    print(f"Written: {path}")
