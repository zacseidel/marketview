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
    "momentum":     "Top 10 S&P 500 by trailing 12-month return, rank-stable.",
    "munger":       "Top 100 by market cap; buy when price dips to SMA200 then recovers above EMA15.",
    "repurchase":   "Top 5 by trailing 12-month share buyback %; above 21d EMA.",
    "quant_gbm":    "LightGBM v1: 15 technical features; predicts 20d forward return. Val Sharpe 0.794.",
    "quant_gbm_v3": "LightGBM v3: 28 features + sector/earnings; predicts 10d forward return. Val Sharpe 1.125.",
    "watchlist":    "User-curated tickers.",
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

    decision_file = _DECISIONS_DIR / f"{eval_date}.md"

    universe      = _load_universe()
    enabled       = _load_enabled_model_names()
    all_holdings  = _load_model_holdings(eval_date, enabled_only=enabled)
    open_positions = _load_open_positions()
    scorecards    = _load_scorecards()
    decision_state = _parse_decision_file(decision_file)

    # Categorise tickers from the decision file
    new_buys:  list[str] = []  # all pre-approved new buys (checked in markdown)
    holds:     list[str] = []
    sells:     list[str] = []
    buy_models:  dict[str, list[str]] = {}  # ticker -> [model names]
    sell_models: dict[str, list[str]] = {}

    for ticker, state in decision_state.items():
        sec     = state["section"]
        checked = state["checked"]
        models  = state["models"]
        if sec == "new_buy" and checked:
            new_buys.append(ticker)
            buy_models[ticker] = models
        elif sec == "hold" and checked:
            holds.append(ticker)
        elif sec == "sell" and checked:
            sells.append(ticker)
            sell_models[ticker] = models

    # ── Scorecard header ─────────────────────────────────────────────────────
    _print_scorecard_header(scorecards)

    # ── Overview ─────────────────────────────────────────────────────────────
    print(f"\n{_hr('═')}")
    print(f"  DECISION REVIEW — {eval_date}")
    print(_hr("═"))
    print(f"  Decision file: {decision_file.relative_to(_ROOT)}")
    print()
    print(f"  Default plan (all model recommendations accepted):")
    print(f"    Buying  ({len(new_buys)}):  {', '.join(new_buys) or '—'}")
    print(f"    Holding ({len(holds)}):  {', '.join(holds[:8])}" + (" ..." if len(holds) > 8 else ""))
    print(f"    Selling ({len(sells)}):  {', '.join(sells) or '—'}")
    print()
    print(f"  Open positions: {len(open_positions)}")
    print()

    # Quick overview of open positions
    if open_positions:
        print(f"  {'Ticker':<8} {'Entry':>8} {'Now':>8} {'Return':>8}  {'5d':>6}  Models")
        print(f"  {'─'*8} {'─'*8}  {'─'*8}  {'─'*8}  {'─'*6}  {'─'*20}")
        for p in open_positions:
            ticker = p["ticker"]
            entry  = p.get("entry_price", 0)
            pd     = _load_latest_price(ticker)
            curr   = pd.get("close") if pd else None
            ret    = math.log(curr / entry) if curr and entry else None
            r5     = _recent_return(ticker, 5)
            mods   = ", ".join(p.get("originating_models", []))
            curr_s = f"${curr:,.2f}" if curr else "—"
            ret_s  = _ret_str_log(ret) if ret is not None else "—"
            print(f"  {ticker:<8} ${entry:>7,.2f}  {curr_s:>8}  {ret_s:>8}  {_ret_str(r5):>6}  {mods}")
        print()

    print("  Research any ticker — type ?TICKER before vetoing.")

    # ── Veto new buys ─────────────────────────────────────────────────────────
    vetoed_buys: set[str] = set()
    if new_buys:
        vetoed_buys = _veto_prompt(
            "NEW BUYS — veto any? (space-separated or Enter to accept all):",
            new_buys, universe, all_holdings,
        )
    else:
        print("\n  No new buy recommendations.")

    # ── Override sells ────────────────────────────────────────────────────────
    kept_sells: set[str] = set()
    if sells:
        kept_sells = _veto_prompt(
            "SELLS — keep any? (space-separated or Enter to sell all):",
            sells, universe, all_holdings,
        )
    else:
        print("\n  No sell recommendations.")

    # ── Research loop ─────────────────────────────────────────────────────────
    print(f"\n{_hr()}")
    print("  Research anything else before confirming — ?TICKER or Enter to continue.")
    while True:
        raw = input("  > ").strip()
        if not raw:
            break
        if raw.startswith("?"):
            _print_stock_detail(raw[1:].upper(), universe, all_holdings)
        else:
            print("  (?TICKER for detail, or press Enter)")

    # ── Summary ───────────────────────────────────────────────────────────────
    final_buys  = [t for t in new_buys if t not in vetoed_buys]
    final_sells = [t for t in sells if t not in kept_sells]

    print(f"\n{_hr('═')}")
    print(f"  FINAL PLAN")
    print(_hr("═"))
    print(f"  Buying  ({len(final_buys)}):  {', '.join(final_buys) or '—'}")
    if vetoed_buys:
        print(f"    Vetoed: {', '.join(sorted(vetoed_buys))}")
    print(f"  Holding ({len(holds)}):  {', '.join(holds[:8])}" + (" ..." if len(holds) > 8 else ""))
    print(f"  Selling ({len(final_sells)}):  {', '.join(final_sells) or '—'}")
    if kept_sells:
        print(f"    Kept (override sell): {', '.join(sorted(kept_sells))}")
    print()

    no_changes = not vetoed_buys and not kept_sells
    if no_changes:
        print("  No overrides — accepting all model recommendations.")
        print("  Nothing to write. Commit decisions/pending/ and push to trigger processing.")
        return

    if not decision_file.exists():
        print("  No decision file found — nothing to write.")
        return

    confirm = input(f"  Write overrides to {decision_file.relative_to(_ROOT)}? [Y/n] > ").strip().lower()
    if confirm == "n":
        print("  Aborted — no changes written.")
        return

    _update_markdown(decision_file, vetoed_buys, kept_sells)

    # Record overrides for future scoring
    vetoed_list = [(t, buy_models.get(t, [])) for t in vetoed_buys]
    kept_list   = [(t, sell_models.get(t, [])) for t in kept_sells]
    _record_overrides(vetoed_list, kept_list, eval_date)

    print(f"  Written. Commit and push to trigger processing.")
    print()


if __name__ == "__main__":
    main()
