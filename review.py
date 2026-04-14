"""
review.py — Decision review and veto interface

Default: accept all model recommendations (new buys, holds, sells).
Action required only to override: veto a buy or keep a sell.

Usage:
    python review.py
    python review.py --date 2026-03-21
"""

from __future__ import annotations

import json
import math
import os
import re
import sys
from pathlib import Path

import yaml

_ROOT = Path(__file__).parent.resolve()
os.chdir(_ROOT)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_MODELS_DIR       = _ROOT / "data.nosync/models"
_MODELS_CONFIG    = _ROOT / "config/models.yaml"
_UNIVERSE_FILE    = _ROOT / "data.nosync/universe/constituents.json"
_FUNDAMENTALS_DIR = _ROOT / "data.nosync/fundamentals"
_PRICES_DIR       = _ROOT / "data.nosync/prices"
_DECISIONS_DIR    = _ROOT / "decisions/pending"
_POSITIONS_FILE   = _ROOT / "data.nosync/positions/positions.json"
_SCORECARDS_DIR   = _ROOT / "data.nosync/models/scorecards"

_MODEL_DESCRIPTIONS = {
    "momentum":      "Top 5 S&P 500 by trailing 12-month return, rank-stable.",
    "munger":        "Top 100 by market cap; buy when price dips to SMA200 then recovers above EMA15.",
    "repurchase":    "Top 5 by trailing 12-month share buyback %; above 21d EMA.",
    "watchlist":     "User-curated tickers.",
    "quant_gbm_v7":  "XGBoost v7: 47 features — v6 + ni_qoq_growth + ni_acceleration + earn_ret_5d_to_20d; 5d target. Val ICIR 1.425. Tue/Fri only.",
}

_FEATURE_LABELS = {
    "predicted_log_ret": ("Predicted 20d return", lambda v: f"{math.exp(v)-1:+.1%}"),
    "pct_ath":           ("vs All-Time High",      lambda v: f"{v:+.1f}%"),
    "pct_sma200":        ("vs 200d SMA",           lambda v: f"{v:+.1f}%"),
    "pct_sma50":         ("vs 50d SMA",            lambda v: f"{v:+.1f}%"),
    "pct_52w_low":       ("vs 52-week Low",        lambda v: f"{v:+.1f}%"),
    "log_ret_252d":      ("12-month return",       lambda v: f"{math.exp(v)-1:+.1%}"),
    "log_ret_60d":       ("3-month return",        lambda v: f"{math.exp(v)-1:+.1%}"),
    "log_ret_20d":       ("1-month return",        lambda v: f"{math.exp(v)-1:+.1%}"),
    "vol_20d":           ("20d volatility (ann.)", lambda v: f"{v:.2f}"),
}

_FEATURE_LABELS_V7 = {
    "predicted_score":    ("Predicted score",        lambda v: f"{v:+.4f}"),
    "eps_surprise_pct":   ("EPS surprise",           lambda v: f"{v:+.1f}%"),
    "earn_ret_5d":        ("Post-earn 5d return",    lambda v: f"{math.exp(v)-1:+.1%}"),
    "earn_ret_5d_to_20d": ("Post-earn 5–20d drift",  lambda v: f"{math.exp(v)-1:+.1%}"),
    "ni_yoy_growth":      ("NI YoY growth",          lambda v: f"{v:+.1f}%"),
    "ni_qoq_growth":      ("NI QoQ growth",          lambda v: f"{v:+.3f}"),
    "ni_acceleration":    ("NI accel (2nd deriv)",   lambda v: f"{v:+.3f}"),
    "mkt_breadth_sma200": ("Mkt breadth SMA200",     lambda v: f"{v:.1%}"),
    "sector_ret_126d":    ("Sector 126d return",     lambda v: f"{math.exp(v)-1:+.1%}"),
    "log_ret_5d":         ("5d return",              lambda v: f"{math.exp(v)-1:+.1%}"),
    "log_ret_60d":        ("3-month return",         lambda v: f"{math.exp(v)-1:+.1%}"),
    "vol_20d":            ("20d volatility (ann.)",  lambda v: f"{v:.2f}"),
}


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def _load_universe() -> dict:
    if not _UNIVERSE_FILE.exists():
        return {}
    with open(_UNIVERSE_FILE) as f:
        return json.load(f)


def _load_fundamentals(ticker: str) -> list[dict]:
    path = _FUNDAMENTALS_DIR / f"{ticker}.json"
    if not path.exists():
        return []
    with open(path) as f:
        data = json.load(f)
    return sorted(
        [q for q in data if not q.get("period", "").startswith(("TTM", "FY"))],
        key=lambda q: q.get("filing_date", ""),
        reverse=True,
    )


def _price_files() -> list[Path]:
    return sorted(f for f in _PRICES_DIR.glob("*.json") if f.stem[0].isdigit())


def _load_latest_price(ticker: str) -> dict | None:
    for f in reversed(_price_files()):
        with open(f) as fp:
            records = json.load(fp)
        for r in records:
            if r.get("ticker") == ticker:
                return r
    return None


def _recent_return(ticker: str, n_days: int = 5) -> float | None:
    files = _price_files()
    prices: list[float] = []
    for f in reversed(files):
        with open(f) as fp:
            records = json.load(fp)
        for r in records:
            if r.get("ticker") == ticker:
                c = r.get("close")
                if c:
                    prices.append(c)
                break
        if len(prices) > n_days:
            break
    if len(prices) < 2:
        return None
    return prices[0] / prices[min(n_days, len(prices) - 1)] - 1


def _load_enabled_model_names() -> set[str]:
    if not _MODELS_CONFIG.exists():
        return set()
    with open(_MODELS_CONFIG) as f:
        cfg = yaml.safe_load(f)
    return {name for name, mc in cfg.get("models", {}).items() if mc.get("enabled", False)}


def _load_model_holdings(eval_date: str, enabled_only: set[str] | None = None) -> dict[str, list[dict]]:
    model_dir = _MODELS_DIR / eval_date
    if not model_dir.exists():
        return {}
    result = {}
    for f in sorted(model_dir.glob("*.json")):
        if f.stem.endswith("_ranks"):
            continue
        if enabled_only is not None and f.stem not in enabled_only:
            continue
        with open(f) as fp:
            result[f.stem] = json.load(fp)
    return result


def _latest_eval_date() -> str | None:
    if not _MODELS_DIR.exists():
        return None
    dirs = sorted(d for d in _MODELS_DIR.iterdir() if d.is_dir() and d.name[0].isdigit())
    return dirs[-1].name if dirs else None


def _load_open_positions() -> list[dict]:
    if not _POSITIONS_FILE.exists():
        return []
    with open(_POSITIONS_FILE) as f:
        positions = json.load(f)
    return [p for p in positions if p.get("status") == "open"]


def _load_scorecards() -> list[dict]:
    if not _SCORECARDS_DIR.exists():
        return []
    result = []
    for f in _SCORECARDS_DIR.glob("*.json"):
        try:
            with open(f) as fp:
                data = json.load(fp)
            if data.get("signal_count", 0) > 0:
                result.append(data)
        except Exception:
            continue
    return sorted(
        result,
        key=lambda d: d.get("avg_excess_return") or d.get("avg_return") or -999,
        reverse=True,
    )


def _parse_decision_file(path: Path) -> dict:
    """
    Parse decision markdown. Returns per-ticker state.
    {ticker: {section, checked, models}}
    """
    if not path.exists():
        return {}
    result: dict[str, dict] = {}
    section = None
    section_map = {
        "## New Buy":           "new_buy",
        "## Current Holdings":  "hold",
        "## Sell":              "sell",
    }
    for line in path.read_text().splitlines():
        for header, sec in section_map.items():
            if line.startswith(header):
                section = sec
                break
        if not section:
            continue
        m = re.match(r'^-\s+\[( |x)\]\s+([A-Z.\-]+)\s*', line, re.IGNORECASE)
        if not m:
            continue
        checked = m.group(1).lower() == "x"
        ticker  = m.group(2).upper()
        # Extract model names from the rest of the line
        rest = line[m.end():]
        model_names = re.findall(r'\b(\w+)\s+\([\d.]+\)', rest)
        result[ticker] = {"section": section, "checked": checked, "models": model_names}
    return result


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def _hr(char: str = "─", width: int = 72) -> str:
    return char * width


def _fmt_millions(v: float | None) -> str:
    if v is None:
        return "—"
    if abs(v) >= 1e9:
        return f"${v/1e9:.2f}B"
    if abs(v) >= 1e6:
        return f"${v/1e6:.0f}M"
    return f"${v:.0f}"


def _ret_str(r: float | None, width: int = 7) -> str:
    if r is None:
        return "—".rjust(width)
    sign = "+" if r >= 0 else ""
    return f"{sign}{r:.1%}".rjust(width)


def _ret_str_log(r: float | None) -> str:
    """Format a log return as a percentage string."""
    if r is None:
        return "   —  "
    sign = "+" if r >= 0 else ""
    return f"{sign}{r:.2%}"


# ---------------------------------------------------------------------------
# Model scorecard header
# ---------------------------------------------------------------------------

def _print_scorecard_header(scorecards: list[dict]) -> None:
    print(f"\n{_hr('═')}")
    print("  MODEL PERFORMANCE")
    print(_hr("═"))
    if not scorecards:
        print("  No scorecard data yet — run: python -m src.tracking.model_scorecard")
        return

    print(f"  {'Model':<16} {'Signals':>7}  {'Closed':>6}  {'Avg Ret':>8}  {'SPY':>8}  {'Alpha':>8}  {'Beat%':>6}")
    print(f"  {'─'*16} {'─'*7}  {'─'*6}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*6}")
    for sc in scorecards:
        model  = sc.get("model", "?")
        sigs   = sc.get("signal_count", 0)
        closed = sc.get("closed_count", 0)
        avg_r  = sc.get("avg_return")
        avg_s  = sc.get("avg_spy_return")
        alpha  = sc.get("avg_excess_return")
        beat   = sc.get("beat_spy_rate")
        thin   = " *" if closed < 5 else "  "

        avg_r_s  = _ret_str_log(avg_r)
        avg_s_s  = _ret_str_log(avg_s)
        alpha_s  = _ret_str_log(alpha) if alpha is not None else "   —  "
        beat_s   = f"{beat:.0%}" if beat is not None else "  —  "
        print(f"  {model:<16} {sigs:>7}  {closed:>4}{thin}  {avg_r_s:>8}  {avg_s_s:>8}  {alpha_s:>8}  {beat_s:>6}")
    print(f"  {'─'*16}")
    print("  * fewer than 5 closed positions — interpret with caution")
    print("  Alpha = Avg Return − SPY return over same holding windows (log returns)")


# ---------------------------------------------------------------------------
# Overrides log
# ---------------------------------------------------------------------------

def _record_overrides(vetoed_buys: list[tuple[str, list[str]]], kept_sells: list[tuple[str, list[str]]], eval_date: str) -> None:
    """Record overrides. Runs inside review.py to avoid circular imports."""
    try:
        from src.tracking.overrides import record_override
        for ticker, models in vetoed_buys:
            record_override(eval_date, ticker, "veto_buy", models)
        for ticker, models in kept_sells:
            record_override(eval_date, ticker, "keep_sell", models)
    except Exception as exc:
        print(f"  (Warning: could not record overrides: {exc})")


# ---------------------------------------------------------------------------
# Stock detail
# ---------------------------------------------------------------------------

def _print_stock_detail(ticker: str, universe: dict, all_holdings: dict[str, list[dict]]) -> None:
    info = universe.get(ticker, {})
    print(f"\n{_hr('═')}")
    print(f"  {ticker}  —  {info.get('name', 'Unknown')}")
    print(f"  {info.get('sector', '')}  |  {info.get('tier', '').upper()}")
    print(_hr("═"))

    desc = info.get("description", "")
    if desc:
        words = desc.split()
        line, lines = [], []
        for w in words:
            if sum(len(x) + 1 for x in line) + len(w) > 72:
                lines.append(" ".join(line))
                line = [w]
            else:
                line.append(w)
        if line:
            lines.append(" ".join(line))
        for ln in lines[:6]:
            print(f"  {ln}")
        if len(lines) > 6:
            print(f"  [... {len(lines)-6} more lines]")
    print()

    price_data = _load_latest_price(ticker)
    if price_data:
        print(f"  Price:  ${price_data.get('close', 0):,.2f}  "
              f"(OHLC avg ${price_data.get('ohlc_avg', 0):,.2f})  "
              f"Date: {price_data.get('date', '—')}")
        r5 = _recent_return(ticker, 5)
        r20 = _recent_return(ticker, 20)
        print(f"  5d: {_ret_str(r5)}   20d: {_ret_str(r20)}")
    print()

    quarters = _load_fundamentals(ticker)
    if quarters:
        q, q1 = quarters[0], quarters[1] if len(quarters) > 1 else {}
        rev, rev1 = q.get("revenue"), q1.get("revenue")
        ni = q.get("net_income")
        mc = q.get("market_cap") or info.get("market_cap")
        rev_chg = f"  ({rev/rev1-1:+.1%} QoQ)" if rev and rev1 else ""
        print(f"  Fundamentals — {q.get('period','?')} (filed {q.get('filing_date','?')})")
        print(f"    Revenue:     {_fmt_millions(rev)}{rev_chg}")
        print(f"    Net Income:  {_fmt_millions(ni)}")
        print(f"    Market Cap:  {_fmt_millions(mc)}")
        shares = q.get("shares_outstanding")
        if shares:
            print(f"    Shares Out:  {shares/1e6:.1f}M")
        print()

    for model_name, holdings in all_holdings.items():
        for h in holdings:
            if h["ticker"] != ticker:
                continue
            status_tag = {"new_buy": "[NEW]", "hold": "[HOLD]", "sell": "[SELL]"}.get(h.get("status", ""), "")
            print(f"  {model_name.upper()} {status_tag}  conviction={h['conviction']:.2f}")
            print(f"    {h.get('rationale', '')}")
            meta = h.get("metadata", {})
            if meta.get("quant_model") in ("gbm", "knn"):
                print(f"    {'─'*50}")
                for feat_key, (label, fmt) in _FEATURE_LABELS.items():
                    val = meta.get(feat_key)
                    if val is not None:
                        print(f"    {label:<24} {fmt(val)}")
            print()

    print(_hr())


# ---------------------------------------------------------------------------
# Veto prompt helper
# ---------------------------------------------------------------------------

def _veto_prompt(
    label: str,
    tickers: list[str],
    universe: dict,
    all_holdings: dict[str, list[dict]],
) -> set[str]:
    """
    Show tickers and return the set of vetoed ones.
    Enter = veto none. Comma/space-separated tickers = veto those. ?TICKER = detail.
    """
    print(f"\n  {label}")
    print(f"  {' '.join(tickers)}")
    print("  Veto (space-separated) or Enter to accept all, ?TICKER to research:")

    while True:
        raw = input("  > ").strip()
        if not raw:
            return set()

        vetoed: set[str] = set()
        unknown: list[str] = []
        for token in raw.split():
            if token.startswith("?"):
                _print_stock_detail(token[1:].upper(), universe, all_holdings)
                continue
            t = token.upper().rstrip(",")
            if t in tickers:
                vetoed.add(t)
            else:
                unknown.append(t)

        if unknown:
            print(f"  Not in list: {', '.join(unknown)} — try again or Enter to accept all")
            continue

        if vetoed:
            print(f"  Vetoing: {', '.join(sorted(vetoed))}")
            confirm = input("  Confirm? [Y/n] > ").strip().lower()
            if confirm == "n":
                print("  Cleared — try again or Enter to accept all")
                continue
        return vetoed


# ---------------------------------------------------------------------------
# Markdown writer
# ---------------------------------------------------------------------------

def _update_markdown(
    path: Path,
    vetoed_buys: set[str],
    kept_sells: set[str],
) -> None:
    """
    Uncheck vetoed new buys and unchecked-to-keep overridden sells.
    All other checkboxes remain unchanged.
    """
    lines = path.read_text().splitlines()
    updated = []
    section = None
    section_map = {
        "## New Buy":          "new_buy",
        "## Current Holdings": "hold",
        "## Sell":             "sell",
    }
    for line in lines:
        for header, sec in section_map.items():
            if line.startswith(header):
                section = sec
                break

        if line.startswith("- [x] ") or line.startswith("- [ ] "):
            rest   = line[6:].strip()
            ticker = rest.split()[0]
            if section == "new_buy" and ticker in vetoed_buys:
                line = "- [ ] " + rest   # uncheck = veto
            elif section == "sell" and ticker in kept_sells:
                line = "- [ ] " + rest   # uncheck = override sell, keep holding
        updated.append(line)

    path.write_text("\n".join(updated) + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Per-model review section
# ---------------------------------------------------------------------------

def _print_model_review_section(
    model_name: str,
    sc: dict | None,
    holdings_today: list[dict],
) -> None:
    """Print the full review block for one model."""
    desc = _MODEL_DESCRIPTIONS.get(model_name, "")
    print(f"\n{_hr('═')}")
    print(f"  {model_name.upper()}")
    if desc:
        print(f"  {desc}")
    print(_hr("─"))

    # One-line scorecard summary
    if sc and sc.get("signal_count", 0) > 0:
        sigs  = sc.get("signal_count", 0)
        avg_r = sc.get("avg_return")
        alpha = sc.get("avg_excess_return")
        beat  = sc.get("beat_spy_rate")
        beat_s = f"{beat:.0%}" if beat is not None else "—"
        print(f"  {sigs} signals  |  avg return {_ret_str_log(avg_r)}  "
              f"|  alpha {_ret_str_log(alpha)}  |  beat SPY {beat_s}")
    else:
        print("  No scorecard data yet.")
    print()

    new_buys_h  = [h for h in holdings_today if h.get("status") == "new_buy"]
    sells_h     = [h for h in holdings_today if h.get("status") == "sell"]
    sell_tickers = {h["ticker"] for h in sells_h}
    new_tickers  = {h["ticker"] for h in new_buys_h}

    # Current theoretical open positions (scorecard), excluding today's exits
    open_pos = [p for p in (sc or {}).get("positions", [])
                if p.get("status") == "open" and p["ticker"] not in sell_tickers]

    # Holdings table: current holds + new buys (marked)
    rows = []
    for p in open_pos:
        rows.append({
            "ticker":     p["ticker"],
            "entry_date": p.get("entry_date", "?"),
            "days":       p.get("days", 0) or 0,
            "log_ret":    p.get("log_ret"),
            "alpha":      p.get("alpha"),
            "new":        False,
        })
    for h in new_buys_h:
        rows.append({
            "ticker":     h["ticker"],
            "entry_date": "today",
            "days":       0,
            "log_ret":    None,
            "alpha":      None,
            "new":        True,
        })
    # Sort: existing holds by alpha desc, new buys appended at bottom
    rows.sort(key=lambda r: (r["new"], -(r["alpha"] or 0)))

    if rows:
        count_after = len(rows)
        print(f"  Portfolio after this run ({count_after} positions):")
        print(f"  {'Ticker':<8} {'Entry':>11}  {'Days':>4}  {'Return':>8}  {'vs SPY':>8}")
        print(f"  {_hr('─', 48)}")
        for r in rows:
            tag = "  ← NEW" if r["new"] else ""
            print(f"  {r['ticker']:<8} {r['entry_date']:>11}  {r['days']:>4}  "
                  f"{_ret_str_log(r['log_ret']):>8}  {_ret_str_log(r['alpha']):>8}{tag}")
        print()

    # Exits
    if sells_h:
        print(f"  Exiting ({len(sells_h)}):  {', '.join(h['ticker'] for h in sells_h)}")
        for h in sells_h:
            if h.get("rationale"):
                print(f"    {h['ticker']}  {h['rationale']}")
        print()

    # New buy detail (conviction + rationale + quant features)
    if new_buys_h:
        print(f"  New buy detail:")
        for h in new_buys_h:
            print(f"    {h['ticker']:<8}  conviction {h.get('conviction', 0):.2f}"
                  f"  —  {h.get('rationale', '')}")
            meta = h.get("metadata", {})
            if meta.get("quant_model") == "gbm_v7":
                for feat_key, (label, fmt) in _FEATURE_LABELS_V7.items():
                    val = meta.get(feat_key)
                    if val is not None:
                        print(f"             {label:<26} {fmt(val)}")
            elif meta.get("quant_model") in ("gbm", "knn"):
                for feat_key, (label, fmt) in _FEATURE_LABELS.items():
                    val = meta.get(feat_key)
                    if val is not None:
                        print(f"             {label:<26} {fmt(val)}")
        print()

    if not rows and not sells_h:
        print("  No holdings and no signals this run.")
        print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    from dotenv import load_dotenv
    load_dotenv()

    date_arg = None
    if "--date" in sys.argv:
        idx = sys.argv.index("--date")
        if idx + 1 < len(sys.argv):
            date_arg = sys.argv[idx + 1]

    eval_date = date_arg or _latest_eval_date()
    if not eval_date:
        print("No model evaluations found. Run python action.py first.")
        sys.exit(1)

    decision_file  = _DECISIONS_DIR / f"{eval_date}.md"
    universe       = _load_universe()
    enabled        = _load_enabled_model_names()
    all_holdings   = _load_model_holdings(eval_date, enabled_only=enabled)
    scorecards_raw = _load_scorecards()
    sc_by_model    = {sc["model"]: sc for sc in scorecards_raw}
    decision_state = _parse_decision_file(decision_file)

    # Build model attribution maps from decision file
    buy_models:  dict[str, list[str]] = {}
    sell_models: dict[str, list[str]] = {}
    for ticker, state in decision_state.items():
        if state["section"] == "new_buy":
            buy_models[ticker] = state["models"]
        elif state["section"] == "sell":
            sell_models[ticker] = state["models"]

    # ── Scorecard header ─────────────────────────────────────────────────────
    _print_scorecard_header([sc for sc in scorecards_raw if sc.get("model") in enabled])

    print(f"\n{_hr('═')}")
    print(f"  MODEL REVIEW — {eval_date}")
    print(_hr("═"))
    print("  Default: all model signals accepted (actual portfolio mirrors theoretical).")
    print("  Type ?TICKER to research. Press Enter to advance. Type  y  to add exceptions.")

    # Order models by alpha desc (best alpha first), models with no scorecard last
    ordered_models = sorted(
        [m for m in all_holdings if m in enabled],
        key=lambda m: sc_by_model.get(m, {}).get("avg_excess_return") or -999,
        reverse=True,
    )

    # ── Per-model review loop ────────────────────────────────────────────────
    vetoed_buys: set[str] = set()
    kept_sells:  set[str] = set()

    for model_name in ordered_models:
        holdings_today = all_holdings.get(model_name, [])
        sc = sc_by_model.get(model_name)

        _print_model_review_section(model_name, sc, holdings_today)

        new_buys_m = [h["ticker"] for h in holdings_today if h.get("status") == "new_buy"]
        sells_m    = [h["ticker"] for h in holdings_today if h.get("status") == "sell"]
        has_actions = bool(new_buys_m or sells_m)

        hint = "  ?TICKER · y (exceptions) · Enter" if has_actions else "  ?TICKER · Enter"
        while True:
            raw = input(f"{hint} > ").strip()
            if not raw:
                break
            if raw.startswith("?"):
                _print_stock_detail(raw[1:].upper(), universe, all_holdings)
            elif raw.lower() == "y" and has_actions:
                if new_buys_m:
                    v = _veto_prompt(
                        "Veto any new buys? (space-separated or Enter to accept all):",
                        new_buys_m, universe, all_holdings,
                    )
                    vetoed_buys |= v
                if sells_m:
                    k = _veto_prompt(
                        "Keep any sells? (space-separated or Enter to sell all):",
                        sells_m, universe, all_holdings,
                    )
                    kept_sells |= k
                break
            else:
                print(f"  ({hint.strip()})")

    # ── Final summary ─────────────────────────────────────────────────────────
    all_new_buys = [t for t, s in decision_state.items()
                    if s["section"] == "new_buy" and s["checked"]]
    all_sells    = [t for t, s in decision_state.items()
                    if s["section"] == "sell" and s["checked"]]
    final_buys   = [t for t in all_new_buys if t not in vetoed_buys]
    final_sells  = [t for t in all_sells    if t not in kept_sells]

    print(f"\n{_hr('═')}")
    print(f"  FINAL PLAN")
    print(_hr("═"))
    if final_buys:
        print(f"  Buying  ({len(final_buys)}):  {', '.join(final_buys)}")
    if vetoed_buys:
        print(f"    Vetoed: {', '.join(sorted(vetoed_buys))}")
    if final_sells:
        print(f"  Selling ({len(final_sells)}):  {', '.join(final_sells)}")
    if kept_sells:
        print(f"    Kept (override sell): {', '.join(sorted(kept_sells))}")
    if not final_buys and not final_sells:
        print("  No buys or sells this run.")
    print()

    no_changes = not vetoed_buys and not kept_sells
    if no_changes:
        print("  No overrides — all model signals accepted.")
        print("  Commit decisions/pending/ and push to trigger processing.")
        return

    if not decision_file.exists():
        print("  No decision file found — nothing to write.")
        return

    confirm = input(f"  Write overrides to {decision_file.relative_to(_ROOT)}? [Y/n] > ").strip().lower()
    if confirm == "n":
        print("  Aborted — no changes written.")
        return

    _update_markdown(decision_file, vetoed_buys, kept_sells)

    vetoed_list = [(t, buy_models.get(t, [])) for t in vetoed_buys]
    kept_list   = [(t, sell_models.get(t, [])) for t in kept_sells]
    _record_overrides(vetoed_list, kept_list, eval_date)

    print("  Written. Commit and push to trigger processing.")
    print()


if __name__ == "__main__":
    main()
