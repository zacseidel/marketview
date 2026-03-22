"""
src/reports/daily.py

Generates the daily GitHub Pages dashboard (docs/index.html).
Reads from flat files — no API calls. Safe to run anytime.

Entry point:
    generate_daily_dashboard(as_of_date: str | None = None) -> None
"""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import structlog

log = structlog.get_logger()

_DOCS_DIR = Path("docs")
_UNIVERSE_FILE = Path("data/universe/constituents.json")
_PRICES_DIR = Path("data/prices")
_MODELS_DIR = Path("data/models")
_DECISIONS_DIR = Path("data/decisions")
_POSITIONS_FILE = Path("data/positions/positions.json")
_QUEUE_FILE = Path("data/queue/pending.json")
_RETURNS_FILE = Path("data/strategy_observations/returns.json")
_HISTORY_GAPS_FILE = Path("data/quant/history_gaps.json")


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


def _load_spy_qqq() -> dict:
    files = sorted(f for f in _PRICES_DIR.glob("*.json") if f.stem[0].isdigit())
    for f in reversed(files):
        with open(f) as fp:
            records = json.load(fp)
        lookup = {r["ticker"]: r for r in records}
        spy = lookup.get("SPY")
        qqq = lookup.get("QQQ")
        if spy or qqq:
            return {"date": f.stem, "spy": spy, "qqq": qqq}
    return {"date": None, "spy": None, "qqq": None}


def _load_latest_model_eval() -> dict:
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
        avg_conviction = (
            round(sum(h.get("conviction", 0) for h in active) / len(active), 2)
            if active else 0.0
        )
        models[model_name] = {
            "total": len(active),
            "new_buys": len(new_buys),
            "avg_conviction": avg_conviction,
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


def _load_strategy_returns() -> dict:
    if not _RETURNS_FILE.exists():
        return {}
    with open(_RETURNS_FILE) as f:
        return json.load(f)


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


def _conviction_bar(conviction: float, width: int = 20) -> str:
    filled = round(conviction * width)
    color = "#f0883e" if conviction >= 0.7 else "#58a6ff" if conviction >= 0.5 else "#8b949e"
    return f'<span style="color:{color};font-family:monospace">{"█" * filled}{"░" * (width - filled)}</span>'


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
    ohlc = data.get("ohlc_avg", close)
    vol = data.get("volume", 0)
    return (
        f'<tr>'
        f'<td class="ticker">{ticker}</td>'
        f'<td>${close:,.2f}</td>'
        f'<td>${ohlc:,.2f}</td>'
        f'<td>{vol:,.0f}</td>'
        f'</tr>'
    )


def _render_model_summary_row(name: str, stats: dict) -> str:
    bar = _conviction_bar(stats["avg_conviction"], width=15)
    new_badge = f' <span style="background:#1f3a1f;color:#3fb950;border-radius:4px;font-size:0.7rem;padding:0.1rem 0.4rem;font-weight:600">+{stats["new_buys"]} new</span>' if stats["new_buys"] else ""
    top_tickers = ", ".join(h["ticker"] for h in stats["top"][:5])
    return (
        f'<tr>'
        f'<td class="ticker">{name}</td>'
        f'<td>{stats["total"]}{new_badge}</td>'
        f'<td>{bar} {stats["avg_conviction"]:.2f}</td>'
        f'<td class="muted small">{top_tickers}</td>'
        f'</tr>'
    )


def _render_holdings_card(model_name: str, holdings: list[dict]) -> str:
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
            conviction = h.get("conviction", 0)
            rationale = h.get("rationale", "")
            if len(rationale) > 110:
                rationale = rationale[:107] + "…"
            rows += (
                f'<tr>'
                f'<td class="ticker">{h["ticker"]}</td>'
                f'<td>{_status_badge(status)}</td>'
                f'<td>{_conviction_bar(conviction, width=12)} {conviction:.2f}</td>'
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
          <td class="muted small">Conviction</td>
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


def _render_strategies_reference() -> str:
    strategies = [
        (
            "stock",
            "Own stock outright",
            "Default — full upside/downside exposure. No options overlay.",
            "Always available",
        ),
        (
            "covered_call",
            "Stock + short call &Delta;0.20–0.25, 21–45 DTE",
            "Generates premium income on existing stock position. Caps upside at the short strike.",
            "Sideways or mild bull market; high IV environments",
        ),
        (
            "leap_otm",
            "Long call 10% OTM, ~500 DTE",
            "Leveraged bullish bet with defined max loss (premium paid). Requires ~2yr runway for thesis to play out.",
            "High-conviction long with limited capital at risk",
        ),
        (
            "diagonal",
            "Long ITM call ~500 DTE + short call &Delta;0.20–0.25, 21–45 DTE",
            "Owns deep ITM long call for delta exposure; sells near-term call to offset cost. Complex — requires active management of the short leg.",
            "Bull market; want to reduce cost basis of long call over time",
        ),
        (
            "csp",
            "Short put ATM, ~21 DTE",
            "Collects premium; obligated to buy stock at strike if assigned. Max profit = premium; loss = strike &minus; premium if stock falls to zero.",
            "Willing to own stock at lower price; high IV; near support levels",
        ),
    ]

    rows = ""
    for name, structure, description, when in strategies:
        rows += (
            f'<tr>'
            f'<td class="ticker">{name}</td>'
            f'<td class="small">{structure}</td>'
            f'<td class="muted small">{description}</td>'
            f'<td class="muted small" style="color:#58a6ff">{when}</td>'
            f'</tr>'
        )

    return f"""
    <div class="card wide">
      <h2>Options Strategy Reference</h2>
      <table>
        <thead><tr>
          <td class="muted small">Strategy</td>
          <td class="muted small">Structure</td>
          <td class="muted small">Mechanics</td>
          <td class="muted small">Use When</td>
        </tr></thead>
        <tbody>{rows}</tbody>
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
    models = ", ".join(p.get("recommending_models", [])) or "—"
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


def _render_strategy_rows(returns: dict) -> str:
    if not returns:
        return '<tr><td colspan="6" class="muted">No closed strategy observations yet</td></tr>'
    rows = ""
    for model in sorted(returns):
        for strategy in sorted(returns[model]):
            s = returns[model][strategy]
            mean = s["mean_log_return"]
            win = s.get("win_rate")
            win_str = f"{win*100:.0f}%" if win is not None else "—"
            mean_color = "#3fb950" if mean > 0 else "#f85149" if mean < 0 else "#8b949e"
            rows += (
                f'<tr>'
                f'<td class="muted small">{model}</td>'
                f'<td class="ticker">{strategy}</td>'
                f'<td>{s["count"]}</td>'
                f'<td style="color:{mean_color}">{mean:+.4f}</td>'
                f'<td class="muted">{s["std_log_return"]:.4f}</td>'
                f'<td>{win_str}</td>'
                f'</tr>'
            )
    return rows


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
    strategy_returns: dict,
    history_gaps: dict,
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
            holdings_cards += _render_holdings_card(model_name, holdings)
    else:
        holdings_cards = '<div class="card wide"><h2>Model Holdings</h2><p class="muted">No model evaluations yet</p></div>'

    confluence_card = _render_confluence(all_holdings)
    strategies_reference = _render_strategies_reference()
    history_gaps_card = _render_history_gaps_card(history_gaps)

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
    strategy_rows = _render_strategy_rows(strategy_returns)

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
          <td class="muted small">OHLC Avg</td>
          <td class="muted small">Volume</td>
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
          <td class="muted small">Avg Conviction</td>
          <td class="muted small">Top Picks</td>
        </tr></thead>
        <tbody>{model_summary_rows}</tbody>
      </table>
    </div>

    <!-- History Gaps Warning -->
    {history_gaps_card}

    <!-- Multi-Model Confluence -->
    {confluence_card}

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

    <!-- Options Strategy Reference -->
    {strategies_reference}

    <!-- Strategy Returns by Model -->
    <div class="card wide">
      <h2>Strategy Returns by Model</h2>
      <table>
        <thead><tr>
          <td class="muted small">Model</td>
          <td class="muted small">Strategy</td>
          <td class="muted small">N</td>
          <td class="muted small">Mean Log Return</td>
          <td class="muted small">Std Dev</td>
          <td class="muted small">Win Rate</td>
        </tr></thead>
        <tbody>{strategy_rows}</tbody>
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
    market = _load_spy_qqq()
    model_eval = _load_latest_model_eval()
    eval_date, all_holdings = _load_all_model_holdings()
    decisions = _load_recent_decisions()
    positions = _load_open_positions()
    queue = _load_queue_stats()
    strategy_returns = _load_strategy_returns()
    history_gaps = _load_history_gaps()

    html = _build_html(
        generated_at, universe, market, model_eval,
        all_holdings, decisions, positions, queue, strategy_returns, history_gaps,
    )

    out = _DOCS_DIR / "index.html"
    out.write_text(html)
    log.info("dashboard.generated", path=str(out), universe=universe["total"], eval_date=eval_date)


if __name__ == "__main__":
    import sys
    date_arg = sys.argv[1] if len(sys.argv) > 1 else None
    generate_daily_dashboard(date_arg)
    print("Dashboard written to docs/index.html")
