"""
src/reports/daily.py

Generates the daily GitHub Pages dashboard (docs/index.html).
Reads from flat files — no API calls. Safe to run anytime.

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
_SCORECARDS_DIR = Path("data.nosync/models/scorecards")
_DECISIONS_DIR = Path("data.nosync/decisions")
_POSITIONS_FILE = Path("data.nosync/positions/positions.json")
_QUEUE_FILE = Path("data.nosync/queue/pending.json")
_HISTORY_GAPS_FILE = Path("data.nosync/quant/history_gaps.json")
_VAL_METRICS_FILE = Path("data.nosync/quant/val_metrics.json")


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
    Tries recent_prices.parquet first (has full history), then falls back to price files
    (populated once ingestion includes benchmark tickers).
    """
    def _build_entry(closes: list[float], date_str: str) -> dict:
        close = closes[-1]
        ret_5d   = math.log(closes[-1] / closes[-6])   if len(closes) >= 6   else None
        ret_252d = math.log(closes[-1] / closes[-253]) if len(closes) >= 253 else None
        return {"close": close, "date": date_str, "ret_5d": ret_5d, "ret_252d": ret_252d}

    # --- parquet path (local runs + fresh quant cache) ---
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

    # --- price file fallback (GitHub Actions, after ingestion includes benchmarks) ---
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


def _load_week_price_changes() -> dict[str, float]:
    """
    Returns {ticker: log_return} over the last ~5 trading days.
    Uses the earliest and latest of the 6 most recent daily price files that have data.
    Skips empty files (e.g. market holidays where ingestion ran but returned no bars).
    """
    files = sorted(f for f in _PRICES_DIR.glob("*.json") if f.stem[0].isdigit())
    # Walk newest-first and collect up to 6 files that actually have data (> 2 bytes = not `[]`)
    populated: list[Path] = []
    for f in reversed(files):
        if f.stat().st_size > 2:
            populated.append(f)
            if len(populated) == 6:
                break
    if len(populated) < 2:
        return {}
    populated.reverse()  # back to chronological order
    with open(populated[0]) as f:
        early = {r["ticker"]: r["close"] for r in json.load(f) if r.get("close")}
    with open(populated[-1]) as f:
        latest = {r["ticker"]: r["close"] for r in json.load(f) if r.get("close")}
    return {
        t: math.log(latest[t] / early[t])
        for t in latest
        if t in early and early[t] > 0
    }


def _load_latest_model_eval(price_changes: dict[str, float]) -> dict:
    """Load summary stats for the latest eval — used in the overview card."""
    if not _MODELS_DIR.exists():
        return {"eval_date": None, "models": {}}
    eval_dirs = sorted([d for d in _MODELS_DIR.iterdir() if d.is_dir() and d.name[0].isdigit()], reverse=True)
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


def _load_all_model_holdings() -> tuple[str | None, dict[str, list[dict]]]:
    """
    Load full holdings for every model in the latest eval dir.
    Returns (eval_date, {model_name: [holding, ...]}).
    Holdings sorted: new_buy first, then hold, then sell.
    """
    if not _MODELS_DIR.exists():
        return None, {}
    eval_dirs = sorted([d for d in _MODELS_DIR.iterdir() if d.is_dir() and d.name[0].isdigit()], reverse=True)
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


def _load_recent_decisions(limit: int = 10) -> list[dict]:
    if not _DECISIONS_DIR.exists():
        return []
    files = sorted(_DECISIONS_DIR.glob("*.json"), reverse=True)
    results = []
    for f in files[:3]:
        with open(f) as fp:
            records = json.load(fp)
        for r in records:
            if r.get("user_approved") and r.get("action") in ("buy", "sell"):
                results.append(r)
        if len(results) >= limit:
            break
    return results[:limit]


def _load_open_positions() -> list[dict]:
    if not _POSITIONS_FILE.exists():
        return []
    with open(_POSITIONS_FILE) as f:
        positions = json.load(f)
    return [p for p in positions if p.get("status") == "open"]


def _load_history_gaps() -> dict:
    if not _HISTORY_GAPS_FILE.exists():
        return {}
    with open(_HISTORY_GAPS_FILE) as f:
        return json.load(f)


def _load_quant_val_metrics() -> dict:
    if not _VAL_METRICS_FILE.exists():
        return {}
    with open(_VAL_METRICS_FILE) as f:
        return json.load(f)


def _load_live_scorecards() -> list[dict]:
    """
    Load all scorecard files. Returns list sorted by avg_return descending.
    Only includes models with at least one signal.
    """
    if not _SCORECARDS_DIR.exists():
        return []
    result = []
    for f in _SCORECARDS_DIR.glob("*.json"):
        try:
            with open(f) as fp:
                data = json.load(fp)
        except Exception:
            continue
        if data.get("signal_count", 0) == 0:
            continue
        data.setdefault("history", [])
        data.setdefault("positions", [])
        result.append(data)
    result.sort(key=lambda d: d.get("avg_excess_return") or d.get("avg_return") or -999, reverse=True)
    return result


def _load_overrides() -> list[dict]:
    path = Path("data.nosync/overrides/log.json")
    if not path.exists():
        return []
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return []


def _load_queue_stats() -> dict:
    if not _QUEUE_FILE.exists():
        return {"total": 0, "by_type": {}}
    with open(_QUEUE_FILE) as f:
        tasks = json.load(f)
    pending = [t for t in tasks if t.get("status") in ("pending", "ready")]
    by_type: dict[str, int] = {}
    for t in pending:
        task_type = t.get("task_type", "unknown")
        by_type[task_type] = by_type.get(task_type, 0) + 1
    return {"total": len(pending), "by_type": by_type}


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
    new_badge = f' <span style="background:#1f3a1f;color:#3fb950;border-radius:4px;font-size:0.7rem;padding:0.1rem 0.4rem;font-weight:600">+{stats["new_buys"]} new</span>' if stats["new_buys"] else ""
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
            rows += f'<tr><td colspan="4" class="muted small" style="text-align:center">+ {total_count - _DISPLAY_MAX} more holdings not shown</td></tr>'
        body = rows

    new_count = sum(1 for h in holdings if h.get("status") == "new_buy")
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


def _render_decision_row(r: dict) -> str:
    action = r.get("action", "")
    action_color = "#3fb950" if action == "buy" else "#f85149"
    exec_price = f'${r["execution_price"]:.2f}' if r.get("execution_price") else "pending"
    models = ", ".join(r.get("recommending_models", []))
    return (
        f'<tr>'
        f'<td class="ticker">{r["ticker"]}</td>'
        f'<td style="color:{action_color};font-weight:600">{action.upper()}</td>'
        f'<td>{r.get("execution_date","")}</td>'
        f'<td>{exec_price}</td>'
        f'<td class="muted small">{models}</td>'
        f'</tr>'
    )


def _render_position_row(p: dict) -> str:
    pnl = p.get("unrealized_pnl")
    pnl_str = f'${pnl:+.2f}' if pnl is not None else "—"
    pnl_color = _pct_color(pnl or 0)
    strategy = p.get("strategy", "stock")
    models = ", ".join(p.get("originating_models", [])) or "—"
    return (
        f'<tr>'
        f'<td class="ticker">{p["ticker"]}</td>'
        f'<td class="small">{strategy}</td>'
        f'<td class="muted small">{models}</td>'
        f'<td>{p.get("entry_date","")}</td>'
        f'<td>${p.get("entry_price",0):.2f}</td>'
        f'<td style="color:{pnl_color}">{pnl_str}</td>'
        f'</tr>'
    )


def _render_quant_scorecard(metrics: dict) -> str:
    if not metrics:
        return ""

    _MODEL_LABELS = {
        "gbm":    ("quant_gbm",    "15d",  "15 technical, 20d target"),
        "gbm_v3": ("quant_gbm_v3", "10d",  "28 features + sector/earnings, 10d target"),
        "gbm_v4": ("quant_gbm_v4", "5d",   "34 features: slope/R² + dollar vol + earnings timing"),
        "gbm_v5": ("quant_gbm_v5", "5d",   "45 features: full union v3+v4b"),
        "gbm_v6": ("quant_gbm_v6", "5d",   "44 features: v5 cleaned + breadth + sector 60d"),
        "gbm_v7": ("quant_gbm_v7", "5d",   "47 features: v6 + ni_qoq + ni_accel + earn drift"),
    }

    rows = ""
    for key in ("gbm_v7",):
        m = metrics.get(key)
        if not m or "error" in m:
            continue
        label, cadence, description = _MODEL_LABELS.get(key, (key, "?d", ""))
        sharpe = m.get("sharpe") or 0
        sharpe_color = "#3fb950" if sharpe >= 1.5 else "#f0883e" if sharpe >= 0.7 else "#8b949e"
        excess = m.get("avg_excess_ret") or 0
        excess_color = _pct_color(excess)
        periods = m.get("eval_periods", 0)

        # IC / ICIR — present in new-format entries, absent in old
        ic = m.get("ic_mean")
        icir = m.get("icir")
        dcl = m.get("decile_hit_rate") or m.get("hit_rate")

        ic_cell = f"{ic:.4f}" if ic is not None else "—"
        icir_str = f"{icir:.3f}" if icir is not None else "—"
        icir_color = "#3fb950" if (icir or 0) >= 2.0 else "#f0883e" if (icir or 0) >= 1.0 else "#8b949e"
        dcl_cell = f"{dcl:.1%}" if dcl is not None else "—"

        rows += (
            f'<tr>'
            f'<td class="ticker">{label}</td>'
            f'<td style="color:{sharpe_color};font-weight:600">{sharpe:.3f}</td>'
            f'<td style="color:{excess_color}">{excess:+.2%}</td>'
            f'<td style="color:{icir_color};font-weight:600">{icir_str}</td>'
            f'<td>{ic_cell}</td>'
            f'<td>{dcl_cell}</td>'
            f'<td class="muted">{periods} × {cadence}</td>'
            f'<td class="muted small">{description}</td>'
            f'</tr>'
        )

    if not rows:
        return ""

    return f"""
    <div class="card wide">
      <h2>Quant Model Val Performance</h2>
      <table>
        <thead><tr>
          <td class="muted small">Model</td>
          <td class="muted small">Sharpe</td>
          <td class="muted small">Excess/Period</td>
          <td class="muted small">ICIR</td>
          <td class="muted small">IC</td>
          <td class="muted small">DclHit</td>
          <td class="muted small">Val Periods</td>
          <td class="muted small">Features</td>
        </tr></thead>
        <tbody>{rows}</tbody>
      </table>
      <p class="small muted" style="margin-top:0.5rem">ICIR = IC / IC.std() × √(periods/yr) — primary cross-model comparable (scale-free, cadence-normalizing). DclHit = fraction of top-20 picks in actual top decile of returns.</p>
    </div>"""


def _render_history_gaps_card(gaps: dict) -> str:
    if not gaps or not gaps.get("tickers"):
        return ""
    tickers = gaps["tickers"]
    as_of = gaps.get("as_of", "")
    ticker_list = ", ".join(tickers[:30])
    overflow = f" + {len(tickers) - 30} more" if len(tickers) > 30 else ""
    return f"""
    <div class="card wide" style="border-color:#d29922;background:#1c1a12">
      <h2 style="color:#d29922">Quant Model — Insufficient Price History ({len(tickers)} tickers as of {as_of})</h2>
      <p class="small muted" style="margin-bottom:0.5rem">These tickers have fewer than 756 trading days of history and were skipped by the quant models. New IPOs and spinoffs will appear here until they accumulate enough history.</p>
      <p class="small" style="color:#c9d1d9">{ticker_list}{overflow}</p>
    </div>"""


def _render_queue_rows(by_type: dict) -> str:
    if not by_type:
        return '<tr><td colspan="2" class="muted">No pending tasks</td></tr>'
    rows = ""
    for task_type, count in sorted(by_type.items()):
        rows += f'<tr><td class="muted">{task_type.replace("_"," ")}</td><td>{count}</td></tr>'
    return rows


# ---------------------------------------------------------------------------
# Live scorecard rendering
# ---------------------------------------------------------------------------

def _sparkline(values: list[float], width: int = 80, height: int = 22) -> str:
    """Inline SVG sparkline of avg_return over time. Zero baseline shown."""
    clean = [v for v in values if v is not None]
    if len(clean) < 2:
        # Single point — just draw a dot
        if len(clean) == 1:
            color = "#3fb950" if clean[0] >= 0 else "#f85149"
            cx, cy = width // 2, height // 2
            return (
                f'<svg width="{width}" height="{height}" style="vertical-align:middle">'
                f'<circle cx="{cx}" cy="{cy}" r="2.5" fill="{color}"/>'
                f'</svg>'
            )
        return f'<svg width="{width}" height="{height}"></svg>'

    pad = 3
    w = width - pad * 2
    h = height - pad * 2
    mn, mx = min(clean), max(clean)
    span = mx - mn if mx != mn else 0.001

    def _y(v: float) -> float:
        return pad + (1 - (v - mn) / span) * h

    def _x(i: int) -> float:
        return pad + (i / (len(clean) - 1)) * w

    points = " ".join(f"{_x(i):.1f},{_y(v):.1f}" for i, v in enumerate(clean))
    color = "#3fb950" if clean[-1] >= 0 else "#f85149"

    # Zero baseline (only if range crosses zero)
    baseline = ""
    if mn < 0 < mx:
        y0 = _y(0)
        baseline = (
            f'<line x1="{pad}" y1="{y0:.1f}" x2="{width - pad}" y2="{y0:.1f}"'
            f' stroke="#30363d" stroke-width="0.8" stroke-dasharray="2,2"/>'
        )

    return (
        f'<svg width="{width}" height="{height}" style="vertical-align:middle">'
        f'{baseline}'
        f'<polyline points="{points}" fill="none" stroke="{color}"'
        f' stroke-width="1.5" stroke-linejoin="round" stroke-linecap="round"/>'
        f'</svg>'
    )


def _render_live_performance_card(scorecards: list[dict]) -> str:
    if not scorecards:
        return ""

    # Sort by avg_excess_return if available, otherwise avg_return
    scorecards = sorted(
        scorecards,
        key=lambda d: d.get("avg_excess_return") or d.get("avg_return") or -999,
        reverse=True,
    )

    rows = ""
    for sc in scorecards:
        model   = sc.get("model", "?")
        signals = sc.get("signal_count", 0)
        closed  = sc.get("closed_count", 0)
        avg_ret = sc.get("avg_return")
        avg_spy = sc.get("avg_spy_return")
        alpha   = sc.get("avg_excess_return")
        as_of   = sc.get("as_of_date", "")
        history = sc.get("history", [])

        # Sparkline tracks alpha over time
        spark_values = [e.get("avg_excess_return") for e in history]
        spark_svg = _sparkline(spark_values)

        def _ret(v: float | None, bold: bool = False) -> str:
            if v is None:
                return '<span class="muted">—</span>'
            color  = _pct_color(v)
            weight = "font-weight:600;" if bold else ""
            return f'<span style="color:{color};font-family:monospace;{weight}">{v:+.2%}</span>'

        thin = ""
        if closed < 5:
            thin = ' <span class="muted" style="font-size:0.7rem" title="Fewer than 5 closed positions — interpret with caution">*</span>'

        model_link = f'<a href="#model-{model}" style="color:#58a6ff;text-decoration:none">{model}</a>'
        rows += (
            f'<tr>'
            f'<td class="ticker">{model_link}</td>'
            f'<td class="muted">{signals}</td>'
            f'<td class="muted">{closed}{thin}</td>'
            f'<td>{_ret(avg_ret)}</td>'
            f'<td>{_ret(avg_spy)}</td>'
            f'<td>{_ret(alpha, bold=True)}</td>'
            f'<td style="white-space:nowrap">{spark_svg}</td>'
            f'<td class="muted small">{as_of}</td>'
            f'</tr>'
        )

    max_history = max((len(sc.get("history", [])) for sc in scorecards), default=0)
    note = (
        f'<p class="muted small" style="margin-top:0.5rem">'
        f'All returns are log returns. Open positions marked to latest available price. '
        f'SPY Log Ret = average SPY log return over the identical holding window for each signal. '
        f'Alpha = Avg Log Ret − SPY Log Ret. '
        f'Sparkline tracks alpha over time ({max_history} daily snapshot{"s" if max_history != 1 else ""} accumulated).'
        f'</p>'
    )

    return f"""
    <div class="card wide">
      <h2>Live Model Performance — Signal Returns Since Launch</h2>
      <table>
        <thead><tr>
          <td class="muted small">Model</td>
          <td class="muted small">Signals</td>
          <td class="muted small">Closed</td>
          <td class="muted small">Avg Log Ret</td>
          <td class="muted small">SPY Log Ret</td>
          <td class="muted small">Alpha vs. SPY</td>
          <td class="muted small">Alpha Trend</td>
          <td class="muted small">As Of</td>
        </tr></thead>
        <tbody>{rows}</tbody>
      </table>
      {note}
    </div>"""


def _render_model_detail_sections(scorecards: list[dict]) -> str:
    """
    Render one collapsible <details> section per model, showing every position
    that contributed to performance. Intended to appear below the main table.
    """
    if not scorecards:
        return ""

    def _ret_cell(v: float | None) -> str:
        if v is None:
            return '<span class="muted">—</span>'
        color = "#3fb950" if v > 0 else ("#f85149" if v < 0 else "#8b949e")
        return f'<span style="color:{color};font-family:monospace">{v:+.2%}</span>'

    sections = ""
    for sc in scorecards:
        model     = sc.get("model", "?")
        positions = sc.get("positions", [])

        if not positions:
            sections += f"""
    <div class="card wide" id="model-{model}" style="margin-top:0.5rem">
      <details>
        <summary style="cursor:pointer;font-size:0.7rem;font-weight:600;color:#8b949e;text-transform:uppercase;letter-spacing:0.08em">{model} — no position detail available</summary>
      </details>
    </div>"""
            continue

        pos_rows = ""
        for p in positions:
            ticker     = p.get("ticker", "?")
            status     = p.get("status", "?")
            entry_date = p.get("entry_date") or "—"
            exit_date  = p.get("exit_date")  or "—"
            days       = p.get("days")
            log_ret    = p.get("log_ret")
            spy_ret    = p.get("spy_log_ret")
            alpha      = p.get("alpha")

            days_str = str(days) if days is not None else "—"
            status_color = "#3fb950" if status == "open" else "#8b949e"
            status_badge = f'<span style="color:{status_color};font-size:0.75rem;font-weight:600">{status.upper()}</span>'

            pos_rows += (
                f'<tr>'
                f'<td class="ticker">{ticker}</td>'
                f'<td>{status_badge}</td>'
                f'<td class="muted small">{entry_date}</td>'
                f'<td class="muted small">{exit_date}</td>'
                f'<td class="muted">{days_str}</td>'
                f'<td>{_ret_cell(log_ret)}</td>'
                f'<td>{_ret_cell(spy_ret)}</td>'
                f'<td>{_ret_cell(alpha)}</td>'
                f'</tr>'
            )

        n_pos = len(positions)
        n_closed = sum(1 for p in positions if p.get("status") == "closed")
        sections += f"""
    <div class="card wide" id="model-{model}" style="margin-top:0.5rem">
      <details>
        <summary style="cursor:pointer;font-size:0.7rem;font-weight:600;color:#8b949e;text-transform:uppercase;letter-spacing:0.08em">{model} — {n_pos} positions ({n_closed} closed)</summary>
        <table style="margin-top:0.75rem">
          <thead><tr>
            <td class="muted small">Ticker</td>
            <td class="muted small">Status</td>
            <td class="muted small">Entry</td>
            <td class="muted small">Exit</td>
            <td class="muted small">Days</td>
            <td class="muted small">Log Ret</td>
            <td class="muted small">SPY</td>
            <td class="muted small">Alpha</td>
          </tr></thead>
          <tbody>{pos_rows}</tbody>
        </table>
      </details>
    </div>"""

    return f"""
    <div class="section-header">Model Position Detail</div>
    {sections}"""


def _render_overrides_card(overrides: list[dict]) -> str:
    if not overrides:
        return ""

    scored   = [o for o in overrides if o.get("scored")]
    pending  = [o for o in overrides if not o.get("scored")]

    def _ret_cell(v: float | None) -> str:
        if v is None:
            return '<span class="muted">—</span>'
        color = "#3fb950" if v > 0 else ("#f85149" if v < 0 else "#8b949e")
        return f'<span style="color:{color};font-family:monospace">{v:+.2%}</span>'

    # Summary stats
    values = [o["override_value"] for o in scored if o.get("override_value") is not None]
    if values:
        avg_val = sum(values) / len(values)
        good    = sum(1 for v in values if v > 0)
        avg_color = "#3fb950" if avg_val > 0 else "#f85149"
        summary = (
            f'<div style="margin-bottom:0.75rem;font-size:0.875rem">'
            f'<span style="color:#8b949e">Scored overrides: </span>'
            f'<strong>{len(scored)}</strong>'
            f'<span class="muted"> &nbsp;|&nbsp; </span>'
            f'<span style="color:#8b949e">Good calls: </span>'
            f'<strong>{good}/{len(values)}</strong>'
            f'<span class="muted"> &nbsp;|&nbsp; </span>'
            f'<span style="color:#8b949e">Avg override value: </span>'
            f'<strong style="color:{avg_color}">{avg_val:+.2%}</strong>'
            f'<span class="muted small"> (log return; positive = override added value)</span>'
            f'</div>'
        )
    else:
        summary = '<p class="muted">No scored overrides yet — overrides are scored after 20 trading days.</p>'

    # Scored rows
    rows = ""
    for o in sorted(scored, key=lambda x: x["eval_date"], reverse=True):
        otype    = o["override_type"]
        label    = "veto buy" if otype == "veto_buy" else "keep sell"
        label_color = "#f85149" if otype == "veto_buy" else "#3fb950"
        models   = ", ".join(o.get("models", [])) or "—"
        rows += (
            f'<tr>'
            f'<td class="muted small">{o["eval_date"]}</td>'
            f'<td class="ticker">{o["ticker"]}</td>'
            f'<td><span style="color:{label_color};font-size:0.78rem">{label}</span></td>'
            f'<td class="muted small">{models}</td>'
            f'<td>{_ret_cell(o.get("ticker_log_ret"))}</td>'
            f'<td>{_ret_cell(o.get("spy_log_ret"))}</td>'
            f'<td>{_ret_cell(o.get("override_value"))}</td>'
            f'<td class="muted small">{o.get("score_date","")}</td>'
            f'</tr>'
        )

    # Pending rows
    for o in sorted(pending, key=lambda x: x["eval_date"], reverse=True):
        otype  = o["override_type"]
        label  = "veto buy" if otype == "veto_buy" else "keep sell"
        label_color = "#f85149" if otype == "veto_buy" else "#3fb950"
        models = ", ".join(o.get("models", [])) or "—"
        rows += (
            f'<tr>'
            f'<td class="muted small">{o["eval_date"]}</td>'
            f'<td class="ticker">{o["ticker"]}</td>'
            f'<td><span style="color:{label_color};font-size:0.78rem">{label}</span></td>'
            f'<td class="muted small">{models}</td>'
            f'<td colspan="3" class="muted small" style="font-style:italic">pending (scores after 20 trading days)</td>'
            f'<td class="muted small">—</td>'
            f'</tr>'
        )

    if not rows:
        return ""

    return f"""
    <div class="card wide">
      <h2>Your Override Record</h2>
      {summary}
      <table>
        <thead><tr>
          <td class="muted small">Date</td>
          <td class="muted small">Ticker</td>
          <td class="muted small">Override</td>
          <td class="muted small">Model(s)</td>
          <td class="muted small">Stock Return</td>
          <td class="muted small">SPY Return</td>
          <td class="muted small">Override Value</td>
          <td class="muted small">Scored On</td>
        </tr></thead>
        <tbody>{rows}</tbody>
      </table>
      <p class="muted small" style="margin-top:0.5rem">
        veto buy: good if stock fell (you avoided a loss).
        keep sell: good if stock rose (you were right to hold).
        Override Value = log return (veto buy: negated stock ret; keep sell: stock ret).
      </p>
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
    decisions: list[dict],
    positions: list[dict],
    queue: dict,
    history_gaps: dict,
    quant_metrics: dict,
    price_changes: dict[str, float],
    live_scorecards: list[dict],
    overrides: list[dict],
) -> str:

    market_rows = (
        _render_market_row("SPY", market.get("spy"))
        + _render_market_row("QQQ", market.get("qqq"))
    )
    market_date = market.get("date") or "—"

    # Overview summary table
    model_summary_rows = ""
    if model_eval["models"]:
        for name, stats in model_eval["models"].items():
            model_summary_rows += _render_model_summary_row(name, stats)
    else:
        model_summary_rows = '<tr><td colspan="4" class="muted">No model evaluations yet — run python -m src.selection.runner</td></tr>'

    eval_date = model_eval.get("eval_date") or "—"

    # Per-model holdings cards
    holdings_cards = ""
    if all_holdings:
        for model_name, holdings in all_holdings.items():
            holdings_cards += _render_holdings_card(model_name, holdings, price_changes)
    else:
        holdings_cards = '<div class="card wide"><h2>Model Holdings</h2><p class="muted">No model evaluations yet</p></div>'

    confluence_card = _render_confluence(all_holdings)
    history_gaps_card = _render_history_gaps_card(history_gaps)
    quant_scorecard = _render_quant_scorecard(quant_metrics)
    live_performance_card = _render_live_performance_card(live_scorecards)
    model_detail_sections = _render_model_detail_sections(live_scorecards)
    overrides_card        = _render_overrides_card(overrides)

    decision_rows = ""
    if decisions:
        for r in decisions:
            decision_rows += _render_decision_row(r)
    else:
        decision_rows = '<tr><td colspan="5" class="muted">No decisions yet</td></tr>'

    position_rows = ""
    if positions:
        for p in positions:
            position_rows += _render_position_row(p)
    else:
        position_rows = '<tr><td colspan="6" class="muted">No open positions</td></tr>'

    queue_rows = _render_queue_rows(queue.get("by_type", {}))
    queue_total = queue.get("total", 0)

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

    <!-- Queue -->
    <div class="card">
      <h2>Queue — {queue_total} pending</h2>
      <table><tbody>{queue_rows}</tbody></table>
    </div>

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

    <!-- History Gaps Warning -->
    {history_gaps_card}

    <!-- Multi-Model Confluence -->
    {confluence_card}

    <!-- Quant Model Val Scorecard -->
    {quant_scorecard}

    <!-- Live Signal Performance (all models, accumulates over time) -->
    {live_performance_card}

    <!-- Per-Model Position Drill-Down -->
    {model_detail_sections}

    <!-- Override Record -->
    {overrides_card}

    <!-- Per-Model Holdings (one card per model) -->
    <div class="section-header">Positions by Model</div>
    {holdings_cards}

    <!-- Recent Decisions -->
    <div class="card wide">
      <h2>Recent Decisions</h2>
      <table>
        <thead><tr>
          <td class="muted small">Ticker</td>
          <td class="muted small">Action</td>
          <td class="muted small">Exec Date</td>
          <td class="muted small">Fill Price</td>
          <td class="muted small">Models</td>
        </tr></thead>
        <tbody>{decision_rows}</tbody>
      </table>
    </div>

    <!-- Open Positions -->
    <div class="card wide">
      <h2>Open Positions</h2>
      <table>
        <thead><tr>
          <td class="muted small">Ticker</td>
          <td class="muted small">Strategy</td>
          <td class="muted small">Model</td>
          <td class="muted small">Entry Date</td>
          <td class="muted small">Entry Price</td>
          <td class="muted small">Unrealized P&amp;L</td>
        </tr></thead>
        <tbody>{position_rows}</tbody>
      </table>
    </div>

  </div>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def generate_daily_dashboard(as_of_date: str | None = None) -> None:
    _DOCS_DIR.mkdir(exist_ok=True)

    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    universe = _load_universe_stats()
    market = _load_benchmarks()
    price_changes = _load_week_price_changes()
    model_eval = _load_latest_model_eval(price_changes)
    eval_date, all_holdings = _load_all_model_holdings()
    decisions = _load_recent_decisions()
    positions = _load_open_positions()
    queue = _load_queue_stats()
    history_gaps = _load_history_gaps()
    quant_metrics = _load_quant_val_metrics()
    live_scorecards = _load_live_scorecards()
    overrides = _load_overrides()

    html = _build_html(
        generated_at, universe, market, model_eval,
        all_holdings, decisions, positions, queue, history_gaps,
        quant_metrics, price_changes, live_scorecards, overrides,
    )

    out = _DOCS_DIR / "index.html"
    out.write_text(html)
    log.info("dashboard.generated", path=str(out), universe=universe["total"], eval_date=eval_date)


if __name__ == "__main__":
    import sys
    date_arg = sys.argv[1] if len(sys.argv) > 1 else None
    generate_daily_dashboard(date_arg)
    print("Dashboard written to docs/index.html")
