"""
src/reports/daily.py

Generates the daily GitHub Pages dashboard (docs/index.html).
Reads from flat files — no API calls. Safe to run anytime.

Each run also writes a dated snapshot to docs/reports/{eval_date}.html
for historical browsing.

Entry point:
    generate_daily_dashboard(as_of_date: str | None = None) -> None
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import structlog

log = structlog.get_logger()

_DOCS_DIR = Path("docs")
_UNIVERSE_FILE = Path("data.nosync/universe/constituents.json")
_PRICES_DIR = Path("data.nosync/prices")
_RECENT_PRICES_FILE = Path("data.nosync/quant/recent_prices.parquet")
_MODELS_DIR = Path("data.nosync/models")


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def _load_universe_stats() -> dict:
    if not _UNIVERSE_FILE.exists():
        return {"total": 0, "sp500": 0, "sp400": 0, "broad": 0}
    with open(_UNIVERSE_FILE) as f:
        constituents = json.load(f)
    active = [r for r in constituents.values() if r.get("status") == "active"]
    return {
        "total": len(active),
        "sp500": sum(1 for r in active if r.get("tier") == "sp500"),
        "sp400": sum(1 for r in active if r.get("tier") == "sp400"),
        "broad": sum(1 for r in active if r.get("tier") == "broad"),
    }


def _load_benchmarks() -> dict:
    """
    Load close + 5d and 252d log returns for SPY and QQQ.
    Tries recent_prices.parquet first (has full history), then falls back to price files.
    """
    def _build_entry(closes: list[float], date_str: str) -> dict:
        close = closes[-1]
        ret_5d   = math.log(closes[-1] / closes[-6])   if len(closes) >= 6   else None
        ret_252d = math.log(closes[-1] / closes[-253]) if len(closes) >= 253 else None
        return {"close": close, "date": date_str, "ret_5d": ret_5d, "ret_252d": ret_252d}

    if _RECENT_PRICES_FILE.exists():
        try:
            import pandas as pd
            df = pd.read_parquet(_RECENT_PRICES_FILE)
            out: dict = {"date": None, "spy": None, "qqq": None}
            for ticker, key in (("SPY", "spy"), ("QQQ", "qqq")):
                t = df[df["ticker"] == ticker].sort_values("date").reset_index(drop=True)
                if len(t) < 2:
                    continue
                closes = t["close"].tolist()
                date_str = str(t.iloc[-1]["date"])[:10]
                out[key] = _build_entry(closes, date_str)
                if out["date"] is None:
                    out["date"] = date_str
            if out["spy"] or out["qqq"]:
                return out
        except Exception:
            pass

    files = sorted(f for f in _PRICES_DIR.glob("*.json") if f.stem[0].isdigit())
    spy_closes: list[float] = []
    qqq_closes: list[float] = []
    last_date: str | None = None
    for f in files[-253:]:
        with open(f) as fp:
            lookup = {r["ticker"]: r for r in json.load(fp)}
        if "SPY" in lookup:
            spy_closes.append(lookup["SPY"]["close"])
            last_date = f.stem
        if "QQQ" in lookup:
            qqq_closes.append(lookup["QQQ"]["close"])

    return {
        "date": last_date,
        "spy": _build_entry(spy_closes, last_date) if spy_closes else None,
        "qqq": _build_entry(qqq_closes, last_date) if qqq_closes else None,
    }


def _load_week_price_changes(as_of: str | None = None) -> dict[str, float]:
    """
    Returns {ticker: log_return} over the last ~5 trading days up to as_of.
    Skips empty files (market holidays where ingestion ran but returned no bars).
    """
    files = sorted(f for f in _PRICES_DIR.glob("*.json") if f.stem[0].isdigit())
    if as_of:
        files = [f for f in files if f.stem <= as_of]
    populated: list[Path] = []
    for f in reversed(files):
        if f.stat().st_size > 2:
            populated.append(f)
            if len(populated) == 6:
                break
    if len(populated) < 2:
        return {}
    populated.reverse()
    with open(populated[0]) as f:
        early = {r["ticker"]: r["close"] for r in json.load(f) if r.get("close")}
    with open(populated[-1]) as f:
        latest = {r["ticker"]: r["close"] for r in json.load(f) if r.get("close")}
    return {
        t: math.log(latest[t] / early[t])
        for t in latest
        if t in early and early[t] > 0
    }


def _load_latest_model_eval(price_changes: dict[str, float], as_of: str | None = None) -> dict:
    """Load summary stats for the latest eval — used in the overview card."""
    if not _MODELS_DIR.exists():
        return {"eval_date": None, "models": {}}
    eval_dirs = sorted(
        [d for d in _MODELS_DIR.iterdir() if d.is_dir() and d.name[0].isdigit()],
        reverse=True,
    )
    if as_of:
        eval_dirs = [d for d in eval_dirs if d.name <= as_of]
    if not eval_dirs:
        return {"eval_date": None, "models": {}}

    latest_dir = eval_dirs[0]
    models: dict[str, dict] = {}
    for json_file in sorted(latest_dir.glob("*.json")):
        model_name = json_file.stem
        if model_name.endswith("_ranks"):
            continue
        with open(json_file) as f:
            holdings = json.load(f)
        active = [h for h in holdings if h.get("status") != "sell"]
        new_buys = [h for h in active if h.get("status") == "new_buy"]
        rets = [price_changes[h["ticker"]] for h in active if h["ticker"] in price_changes]
        avg_log_ret_1w = round(sum(rets) / len(rets), 4) if rets else None
        models[model_name] = {
            "total": len(active),
            "new_buys": len(new_buys),
            "avg_log_ret_1w": avg_log_ret_1w,
            "top": sorted(active, key=lambda h: h.get("conviction", 0), reverse=True)[:5],
        }
    return {"eval_date": latest_dir.name, "models": models}


def _load_all_model_holdings(as_of: str | None = None) -> tuple[str | None, dict[str, list[dict]]]:
    """
    Load full holdings for every model in the latest eval dir up to as_of.
    Returns (eval_date, {model_name: [holding, ...]}).
    Holdings sorted: new_buy first, then hold, then sell.
    """
    if not _MODELS_DIR.exists():
        return None, {}
    eval_dirs = sorted(
        [d for d in _MODELS_DIR.iterdir() if d.is_dir() and d.name[0].isdigit()],
        reverse=True,
    )
    if as_of:
        eval_dirs = [d for d in eval_dirs if d.name <= as_of]
    if not eval_dirs:
        return None, {}

    latest_dir = eval_dirs[0]
    status_order = {"new_buy": 0, "hold": 1, "sell": 2}
    result: dict[str, list[dict]] = {}

    for json_file in sorted(latest_dir.glob("*.json")):
        if json_file.stem.endswith("_ranks"):
            continue
        with open(json_file) as f:
            holdings = json.load(f)
        holdings.sort(key=lambda h: (status_order.get(h.get("status", "hold"), 1), -h.get("conviction", 0)))
        result[json_file.stem] = holdings

    return latest_dir.name, result




def _load_past_reports(limit: int = 10) -> list[dict]:
    """
    Scan docs/reports/ for dated HTML snapshots.
    Returns [{"date": "YYYY-MM-DD", "path": "reports/YYYY-MM-DD.html"}, ...] newest-first.
    """
    reports_dir = _DOCS_DIR / "reports"
    if not reports_dir.exists():
        return []
    entries = []
    for f in reports_dir.glob("*.html"):
        if len(f.stem) == 10 and f.stem[4] == "-" and f.stem[7] == "-":
            entries.append({"date": f.stem, "path": f"reports/{f.name}"})
    return sorted(entries, key=lambda e: e["date"], reverse=True)[:limit]


# ---------------------------------------------------------------------------
# HTML rendering helpers
# ---------------------------------------------------------------------------

def _pct_color(pct: float) -> str:
    if pct > 0:
        return "#3fb950"
    elif pct < 0:
        return "#f85149"
    return "#8b949e"


def _status_badge(status: str) -> str:
    styles = {
        "new_buy": "background:#1f3a1f;color:#3fb950",
        "hold":    "background:#1c2333;color:#8b949e",
        "sell":    "background:#3a1f1f;color:#f85149",
    }
    labels = {"new_buy": "NEW", "hold": "HOLD", "sell": "SELL"}
    style = styles.get(status, "color:#8b949e")
    label = labels.get(status, status.upper())
    return f'<span style="{style};border-radius:4px;font-size:0.7rem;padding:0.15rem 0.5rem;font-weight:700">{label}</span>'


def _render_market_row(ticker: str, data: dict | None) -> str:
    if not data:
        return f'<tr><td class="ticker">{ticker}</td><td colspan="3" class="muted">—</td></tr>'
    close = data.get("close", 0)
    ret_5d   = data.get("ret_5d")
    ret_252d = data.get("ret_252d")
    def _fmt_ret(r: float | None) -> str:
        if r is None:
            return '<span class="muted">—</span>'
        color = _pct_color(r)
        return f'<span style="color:{color}">{r:+.2%}</span>'
    return (
        f'<tr>'
        f'<td class="ticker">{ticker}</td>'
        f'<td>${close:,.2f}</td>'
        f'<td>{_fmt_ret(ret_5d)}</td>'
        f'<td>{_fmt_ret(ret_252d)}</td>'
        f'</tr>'
    )


def _render_model_summary_row(name: str, stats: dict) -> str:
    new_badge = (
        f' <span style="background:#1f3a1f;color:#3fb950;border-radius:4px;font-size:0.7rem;'
        f'padding:0.1rem 0.4rem;font-weight:600">+{stats["new_buys"]} new</span>'
    ) if stats["new_buys"] else ""
    top_tickers = ", ".join(h["ticker"] for h in stats["top"][:5])
    ret = stats.get("avg_log_ret_1w")
    if ret is not None:
        ret_str = f'<span style="color:{_pct_color(ret)};font-family:monospace">{ret:+.4f}</span>'
    else:
        ret_str = '<span class="muted">—</span>'
    return (
        f'<tr>'
        f'<td class="ticker">{name}</td>'
        f'<td>{stats["total"]}{new_badge}</td>'
        f'<td>{ret_str}</td>'
        f'<td class="muted small">{top_tickers}</td>'
        f'</tr>'
    )


def _render_holdings_card(model_name: str, holdings: list[dict], price_changes: dict[str, float]) -> str:
    _DISPLAY_MAX = 10
    total_count = len(holdings)
    display = holdings[:_DISPLAY_MAX]
    truncated = total_count > _DISPLAY_MAX

    if not holdings:
        body = '<tr><td colspan="4" class="muted">No holdings for this model</td></tr>'
    else:
        rows = ""
        for h in display:
            status = h.get("status", "hold")
            rationale = h.get("rationale", "")
            if len(rationale) > 110:
                rationale = rationale[:107] + "…"
            ret = price_changes.get(h["ticker"])
            if ret is not None:
                ret_cell = f'<span style="color:{_pct_color(ret)};font-family:monospace">{ret:+.4f}</span>'
            else:
                ret_cell = '<span class="muted">—</span>'
            rows += (
                f'<tr>'
                f'<td class="ticker">{h["ticker"]}</td>'
                f'<td>{_status_badge(status)}</td>'
                f'<td>{ret_cell}</td>'
                f'<td class="muted small">{rationale}</td>'
                f'</tr>'
            )
        if truncated:
            rows += (
                f'<tr><td colspan="4" class="muted small" style="text-align:center">'
                f'+ {total_count - _DISPLAY_MAX} more holdings not shown</td></tr>'
            )
        body = rows

    new_count  = sum(1 for h in holdings if h.get("status") == "new_buy")
    hold_count = sum(1 for h in holdings if h.get("status") == "hold")
    sell_count = sum(1 for h in holdings if h.get("status") == "sell")
    summary = f'{new_count} new &nbsp;·&nbsp; {hold_count} hold &nbsp;·&nbsp; {sell_count} sell'

    return f"""
    <div class="card wide">
      <h2>{model_name} &nbsp;<span class="muted" style="font-weight:400;text-transform:none;letter-spacing:0">{summary}</span></h2>
      <table>
        <thead><tr>
          <td class="muted small">Ticker</td>
          <td class="muted small">Status</td>
          <td class="muted small">1W Log Ret</td>
          <td class="muted small">Rationale</td>
        </tr></thead>
        <tbody>{body}</tbody>
      </table>
    </div>"""


def _render_confluence(all_holdings: dict[str, list[dict]]) -> str:
    ticker_models: dict[str, list[str]] = defaultdict(list)
    for model_name, holdings in all_holdings.items():
        for h in holdings:
            if h.get("status") != "sell":
                ticker_models[h["ticker"]].append(model_name)

    multi = {t: models for t, models in ticker_models.items() if len(models) >= 2}

    if not multi:
        body = '<tr><td colspan="3" class="muted">No tickers recommended by multiple models</td></tr>'
    else:
        rows = ""
        for ticker, models in sorted(multi.items(), key=lambda x: -len(x[1])):
            rows += (
                f'<tr>'
                f'<td class="ticker">{ticker}</td>'
                f'<td>{len(models)}</td>'
                f'<td class="muted small">{", ".join(sorted(models))}</td>'
                f'</tr>'
            )
        body = rows

    return f"""
    <div class="card wide">
      <h2>Multi-Model Confluence</h2>
      <table>
        <thead><tr>
          <td class="muted small">Ticker</td>
          <td class="muted small">Model Count</td>
          <td class="muted small">Models</td>
        </tr></thead>
        <tbody>{body}</tbody>
      </table>
    </div>"""




def _render_past_reports_card(reports: list[dict]) -> str:
    if not reports:
        return ""
    rows = ""
    for r in reports:
        rows += (
            f'<tr><td>'
            f'<a href="{r["path"]}" style="color:#58a6ff;text-decoration:none">{r["date"]}</a>'
            f'</td></tr>'
        )
    return f"""
    <div class="card">
      <h2>Past Reports</h2>
      <table><tbody>{rows}</tbody></table>
    </div>"""


# ---------------------------------------------------------------------------
# HTML assembly
# ---------------------------------------------------------------------------

def _build_html(
    generated_at: str,
    universe: dict,
    market: dict,
    model_eval: dict,
    all_holdings: dict[str, list[dict]],
    price_changes: dict[str, float],
    past_reports: list[dict],
) -> str:

    market_rows = (
        _render_market_row("SPY", market.get("spy"))
        + _render_market_row("QQQ", market.get("qqq"))
    )
    market_date = market.get("date") or "—"

    model_summary_rows = ""
    if model_eval["models"]:
        for name, stats in model_eval["models"].items():
            model_summary_rows += _render_model_summary_row(name, stats)
    else:
        model_summary_rows = '<tr><td colspan="4" class="muted">No model evaluations yet — run python -m src.selection.runner</td></tr>'

    eval_date = model_eval.get("eval_date") or "—"

    holdings_cards = ""
    if all_holdings:
        for model_name, holdings in all_holdings.items():
            holdings_cards += _render_holdings_card(model_name, holdings, price_changes)
    else:
        holdings_cards = '<div class="card wide"><h2>Model Signals</h2><p class="muted">No model evaluations yet</p></div>'

    confluence_card   = _render_confluence(all_holdings)
    past_reports_card = _render_past_reports_card(past_reports)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>marketview</title>
  <style>
    *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      background: #0d1117; color: #c9d1d9; padding: 1.5rem;
    }}
    .header {{
      display: flex; align-items: baseline; justify-content: space-between;
      margin-bottom: 1.5rem; flex-wrap: wrap; gap: 0.5rem;
    }}
    .header h1 {{ font-size: 1.5rem; font-weight: 700; color: #e6edf3; }}
    .header .meta {{ font-size: 0.8rem; color: #8b949e; }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
      gap: 1rem;
    }}
    .card {{
      background: #161b22; border: 1px solid #30363d;
      border-radius: 8px; padding: 1rem;
    }}
    .card h2 {{
      font-size: 0.7rem; font-weight: 600; color: #8b949e;
      text-transform: uppercase; letter-spacing: 0.08em;
      margin-bottom: 0.75rem;
    }}
    .card.wide {{ grid-column: 1 / -1; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 0.875rem; }}
    td {{ padding: 0.4rem 0.5rem; border-bottom: 1px solid #21262d; vertical-align: middle; }}
    tr:last-child td {{ border-bottom: none; }}
    .ticker {{ font-weight: 600; color: #e6edf3; font-family: monospace; }}
    .muted {{ color: #8b949e; }}
    .small {{ font-size: 0.78rem; }}
    .stat-row {{
      display: flex; gap: 1.5rem; flex-wrap: wrap; margin-bottom: 0.5rem;
    }}
    .stat {{ text-align: center; }}
    .stat .value {{ font-size: 1.5rem; font-weight: 700; color: #e6edf3; }}
    .stat .label {{ font-size: 0.7rem; color: #8b949e; text-transform: uppercase; }}
    .section-header {{
      font-size: 1rem; font-weight: 600; color: #e6edf3;
      margin: 1.5rem 0 0.75rem; padding-bottom: 0.4rem;
      border-bottom: 1px solid #30363d;
      grid-column: 1 / -1;
    }}
  </style>
</head>
<body>
  <div class="header">
    <h1>marketview</h1>
    <span class="meta">Updated {generated_at}</span>
  </div>

  <div class="grid">

    <!-- Universe -->
    <div class="card">
      <h2>Universe</h2>
      <div class="stat-row">
        <div class="stat"><div class="value">{universe['total']}</div><div class="label">Total</div></div>
        <div class="stat"><div class="value">{universe['sp500']}</div><div class="label">S&amp;P 500</div></div>
        <div class="stat"><div class="value">{universe['sp400']}</div><div class="label">S&amp;P 400</div></div>
        <div class="stat"><div class="value">{universe['broad']}</div><div class="label">Broad</div></div>
      </div>
    </div>

    <!-- Market -->
    <div class="card">
      <h2>Market — {market_date}</h2>
      <table>
        <thead><tr>
          <td class="muted small">Ticker</td>
          <td class="muted small">Close</td>
          <td class="muted small">5d</td>
          <td class="muted small">12m</td>
        </tr></thead>
        <tbody>{market_rows}</tbody>
      </table>
    </div>

    <!-- Past Reports -->
    {past_reports_card}

    <!-- Model Overview Summary -->
    <div class="card wide">
      <h2>Model Overview — {eval_date}</h2>
      <table>
        <thead><tr>
          <td class="muted small">Model</td>
          <td class="muted small">Holdings</td>
          <td class="muted small">Avg 1W Log Ret</td>
          <td class="muted small">Top Picks</td>
        </tr></thead>
        <tbody>{model_summary_rows}</tbody>
      </table>
    </div>

    <!-- Multi-Model Confluence -->
    {confluence_card}

    <!-- Per-Model Signals (one card per model) -->
    <div class="section-header">Model Signals</div>
    {holdings_cards}

  </div>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def generate_daily_dashboard(as_of_date: str | None = None) -> None:
    _DOCS_DIR.mkdir(exist_ok=True)
    reports_dir = _DOCS_DIR / "reports"
    reports_dir.mkdir(exist_ok=True)

    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    universe     = _load_universe_stats()
    market       = _load_benchmarks()
    price_changes = _load_week_price_changes(as_of=as_of_date)
    model_eval   = _load_latest_model_eval(price_changes, as_of=as_of_date)
    eval_date, all_holdings = _load_all_model_holdings(as_of=as_of_date)
    past_reports = _load_past_reports()

    html = _build_html(
        generated_at, universe, market, model_eval,
        all_holdings, price_changes, past_reports,
    )

    out = _DOCS_DIR / "index.html"
    out.write_text(html)
    log.info("dashboard.generated", path=str(out), universe=universe["total"], eval_date=eval_date)

    # Write dated archive snapshot
    if eval_date and eval_date != "—":
        archive = reports_dir / f"{eval_date}.html"
        archive.write_text(html)
        log.info("dashboard.archived", path=str(archive))


if __name__ == "__main__":
    import sys
    date_arg = sys.argv[1] if len(sys.argv) > 1 else None
    generate_daily_dashboard(date_arg)
    print("Dashboard written to docs/index.html")
