"""
review.py — Interactive decision review

Walk through current positions, pending buys, new recommendations, and sell
signals. Research any stock with ?TICKER. Updates the pending decision file.

Usage:
    python review.py
    python review.py --date 2026-03-21
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

# Always run relative to the repo root, regardless of where python was invoked
_ROOT = Path(__file__).parent.resolve()
os.chdir(_ROOT)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_MODELS_DIR       = _ROOT / "data/models"
_UNIVERSE_FILE    = _ROOT / "data/universe/constituents.json"
_FUNDAMENTALS_DIR = _ROOT / "data/fundamentals"
_PRICES_DIR       = _ROOT / "data/prices"
_DECISIONS_DIR    = _ROOT / "decisions/pending"
_POSITIONS_FILE   = _ROOT / "data/positions/positions.json"

_MODEL_DESCRIPTIONS = {
    "momentum":   "Top 10 S&P 500 by trailing 12-month return, rank-stable. Pure price momentum.",
    "munger":     "Top 100 by market cap; buy when price dips to SMA200 then recovers above EMA15. Quality on dips.",
    "repurchase": "Top 5 by trailing 12-month share buyback %; above 21d EMA. Aggressive capital return.",
    "quant_gbm":  "LightGBM on 15 technical factors; predicts 20-day forward return. Val Sharpe: 0.794.",
    "quant_knn":  "KNN (K=50) finds 50 most historically similar setups, averages forward returns. Val Sharpe: 0.373.",
    "watchlist":  "User-curated tickers.",
    "composite":  "Tickers where 2+ models agree, conviction-weighted.",
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
    """Return simple % change over the last n trading days."""
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


def _load_model_holdings(eval_date: str) -> dict[str, list[dict]]:
    model_dir = _MODELS_DIR / eval_date
    if not model_dir.exists():
        return {}
    result = {}
    for f in sorted(model_dir.glob("*.json")):
        if f.stem.endswith("_ranks"):
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


def _parse_decision_file(path: Path) -> dict[str, list[str]]:
    """
    Parse decision markdown. Returns:
      { "approved_buys": [...], "pending_buys": [...],
        "approved_sells": [...], "kept_sells": [...] }
    approved = [x], pending = [ ]
    """
    if not path.exists():
        return {}

    result: dict[str, list[str]] = {
        "approved_buys": [], "pending_buys": [],
        "approved_sells": [], "kept_sells": [],
    }
    section = None

    for line in path.read_text().splitlines():
        if "## New Buy" in line:
            section = "buy"
        elif "## Current Holdings" in line:
            section = "hold"
        elif "## Sell" in line:
            section = "sell"
        elif line.startswith("- ["):
            checked = line.startswith("- [x]")
            ticker = line[6:].strip().split()[0]
            if section == "buy":
                (result["approved_buys"] if checked else result["pending_buys"]).append(ticker)
            elif section == "sell":
                (result["approved_sells"] if checked else result["kept_sells"]).append(ticker)

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


def _ret_str(r: float | None) -> str:
    if r is None:
        return "  —  "
    sign = "+" if r >= 0 else ""
    return f"{sign}{r:.1%}"


def _holding_models(ticker: str, all_holdings: dict[str, list[dict]], statuses: tuple) -> list[str]:
    """Return model names where ticker appears with given statuses."""
    return [
        model for model, holdings in all_holdings.items()
        for h in holdings
        if h["ticker"] == ticker and h.get("status") in statuses
    ]


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
        for l in lines[:6]:
            print(f"  {l}")
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
            status_tag = {"new_buy": "[NEW]", "hold": "[HOLD]", "sell": "[SELL]"}.get(h.get("status",""), "")
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
# Section renderers
# ---------------------------------------------------------------------------

def _show_portfolio_section(
    open_positions: list[dict],
    approved_buys: list[str],
    universe: dict,
    all_holdings: dict[str, list[dict]],
) -> None:
    """Display current open positions and approved-but-pending buys."""

    print(f"\n{_hr('═')}")
    print(f"  CURRENT PORTFOLIO")
    print(_hr("═"))

    # Open positions
    if open_positions:
        print(f"\n  Open Positions ({len(open_positions)})")
        print(f"  {'─'*68}")
        print(f"  {'Ticker':<8} {'Entry':>8} {'Current':>8} {'Since Entry':>12} {'5d':>7}  Models (hold)")
        print(f"  {'─'*68}")
        for p in open_positions:
            ticker = p["ticker"]
            entry = p.get("entry_price", 0)
            price_data = _load_latest_price(ticker)
            current = price_data.get("close") if price_data else None
            since_entry = (math.log(current / entry) if current and entry else None)
            r5 = _recent_return(ticker, 5)
            hold_models = _holding_models(ticker, all_holdings, ("hold", "new_buy"))
            models_str = ", ".join(hold_models) if hold_models else "—"
            pnl_str = f"{math.exp(since_entry)-1:+.1%}" if since_entry is not None else "  —  "
            curr_str = f"${current:,.2f}" if current else "  —  "
            print(f"  {ticker:<8} ${entry:>7,.2f} {curr_str:>8} {pnl_str:>12} {_ret_str(r5):>7}  {models_str}")

    # Pending buys (approved, awaiting execution)
    if approved_buys:
        print(f"\n  Pending Execution — approved, executes next trading day ({len(approved_buys)})")
        print(f"  {'─'*68}")
        print(f"  {'Ticker':<8} {'Current':>8} {'5d':>7} {'20d':>7}  Models")
        print(f"  {'─'*68}")
        for ticker in approved_buys:
            price_data = _load_latest_price(ticker)
            current = price_data.get("close") if price_data else None
            r5 = _recent_return(ticker, 5)
            r20 = _recent_return(ticker, 20)
            rec_models = _holding_models(ticker, all_holdings, ("new_buy", "hold"))
            models_str = ", ".join(rec_models) if rec_models else "—"
            curr_str = f"${current:,.2f}" if current else "  —  "
            print(f"  {ticker:<8} {curr_str:>8} {_ret_str(r5):>7} {_ret_str(r20):>7}  {models_str}")

    if not open_positions and not approved_buys:
        print("\n  No open positions or pending buys yet.")

    print()
    print("  Research any ticker — type ?TICKER, or Enter to continue.")
    while True:
        raw = input("  > ").strip()
        if not raw:
            break
        if raw.startswith("?"):
            _print_stock_detail(raw[1:].upper(), universe, all_holdings)
        else:
            print("  (type ?TICKER or press Enter)")


def _prompt_section(
    section_label: str,
    tickers: list[str],
    already_selected: set[str],
    universe: dict,
    all_holdings: dict[str, list[dict]],
    pre_checked: bool = False,
) -> set[str]:
    if not tickers:
        return set()

    selected = set(t for t in tickers if pre_checked)

    while True:
        if already_selected:
            print(f"\n  Already buying: {', '.join(sorted(already_selected))}")
        print(f"  Tickers: {', '.join(tickers)}")
        if pre_checked:
            print(f"  [pre-approved: {', '.join(sorted(selected))}]")
        print()
        print("  Commands:")
        print("    <tickers>   — select (e.g. 'AAPL MSFT')")
        if pre_checked:
            print("    -<tickers>  — remove from approved (e.g. '-NVDA')")
        print("    ?<ticker>   — show detail (e.g. '?AAPL')")
        print("    done/Enter  — move to next section")
        print("    skip        — skip section")
        print()
        raw = input(f"  {section_label} > ").strip()

        if raw.lower() in ("done", ""):
            break
        if raw.lower() == "skip":
            selected = set(t for t in tickers if pre_checked)
            break

        for token in raw.split():
            if token.startswith("?"):
                lookup = token[1:].upper()
                if lookup:
                    _print_stock_detail(lookup, universe, all_holdings)
            elif token.startswith("-") and pre_checked:
                selected.discard(token[1:].upper())
                print(f"    Removed {token[1:].upper()}")
            else:
                t = token.upper()
                if t in tickers:
                    selected.add(t)
                    print(f"    Added {t}")
                else:
                    print(f"    '{t}' not in this section — ignored")

    return selected


# ---------------------------------------------------------------------------
# Markdown updater
# ---------------------------------------------------------------------------

def _update_markdown(path: Path, approved: set[str], keep_sells: set[str]) -> None:
    lines = path.read_text().splitlines()
    updated = []
    for line in lines:
        if line.startswith("- [ ] ") or line.startswith("- [x] "):
            rest = line[6:].strip()
            ticker = rest.split()[0]
            if line.startswith("- [ ] ") and ticker in approved:
                line = "- [x] " + rest
            elif line.startswith("- [x] ") and ticker in keep_sells:
                line = "- [ ] " + rest
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

    universe = _load_universe()
    all_holdings = _load_model_holdings(eval_date)
    open_positions = _load_open_positions()
    decision_state = _parse_decision_file(decision_file)

    approved_buys  = decision_state.get("approved_buys", [])
    pending_buys   = decision_state.get("pending_buys", [])
    approved_sells = decision_state.get("approved_sells", [])

    # Model new buys (from output files, not the markdown)
    model_new_buys: dict[str, list[tuple[str, float, str | None]]] = {}
    sell_tickers: list[tuple[str, str]] = []

    for model_name, holdings in all_holdings.items():
        active = [(h["ticker"], h["conviction"], h.get("status")) for h in holdings if h.get("status") in ("new_buy", "hold")]
        if active:
            model_new_buys[model_name] = active
        for h in holdings:
            if h.get("status") == "sell":
                sell_tickers.append((h["ticker"], model_name))

    # Header
    print()
    print(_hr("═"))
    print(f"  DECISION REVIEW — {eval_date}")
    print(_hr("═"))
    if decision_file.exists():
        print(f"  Decision file: {decision_file.relative_to(_ROOT)}")
    print(f"  Open positions: {len(open_positions)}  |  "
          f"Approved buys pending execution: {len(approved_buys)}  |  "
          f"New recommendations: {sum(1 for v in model_new_buys.values() for _, _, st in v if st == 'new_buy')}  |  "
          f"Sell signals: {len(sell_tickers)}")
    print(_hr())
    print("  Tip: type ?TICKER at any prompt to see full stock details.")

    # ── Section 1: Current portfolio + pending buys ──────────────────────────
    _show_portfolio_section(open_positions, approved_buys, universe, all_holdings)

    # ── Section 2: New buy recommendations ───────────────────────────────────
    approved: set[str] = set(approved_buys)  # seed with already-approved

    if not pending_buys and not model_new_buys:
        print("\n  No new buy recommendations to review.")
    elif not model_new_buys:
        print("\n  No model outputs found for this eval date.")
    else:
        pending_set = set(pending_buys)
        for model_name, picks in model_new_buys.items():
            model_selectable = [t for t, _, st in picks if st == "new_buy" and t in pending_set]

            print(f"\n{_hr()}")
            print(f"  MODEL: {model_name.upper()}")
            print(_hr())
            desc = _MODEL_DESCRIPTIONS.get(model_name, "")
            if desc:
                print(f"  {desc}")
            print()

            print(f"  {'Ticker':<8} {'Conv':>5}  {'5d':>7}  {'Also in':<28}  {'Status'}")
            print(f"  {'─'*8} {'─'*5}  {'─'*7}  {'─'*28}  {'─'*14}")
            for ticker, conviction, model_status in picks:
                other_models = [
                    m for m, holdings in all_holdings.items()
                    if m != model_name
                    and any(h["ticker"] == ticker and h.get("status") != "sell" for h in holdings)
                ]
                r5 = _recent_return(ticker, 5)
                others = ", ".join(other_models) if other_models else "—"
                if model_status == "hold":
                    status = "  holding"
                elif ticker in approved:
                    status = "✓ approved"
                elif ticker not in pending_set:
                    status = "— skipped"
                else:
                    status = "  new"
                print(f"  {ticker:<8} {conviction:>5.2f}  {_ret_str(r5):>7}  {others:<28}  {status}")

            if not model_selectable:
                input("  (press Enter to continue) ")
                continue

            tickers = model_selectable
            selected = _prompt_section(model_name, tickers, approved, universe, all_holdings)
            approved.update(selected)

    # ── Section 3: Sells ──────────────────────────────────────────────────────
    keep_sells: set[str] = set()
    if sell_tickers:
        print(f"\n{_hr()}")
        print(f"  SELL RECOMMENDATIONS")
        print(_hr())
        print("  These positions are flagged for sale. Prefix with - to keep.\n")
        for ticker, model in sell_tickers:
            r5 = _recent_return(ticker, 5)
            print(f"  {ticker:<8} — {model}  (5d: {_ret_str(r5)})")

        sell_only = [t for t, _ in sell_tickers]
        keeps = _prompt_section(
            "sells (-TICKER to keep)",
            sell_only, set(), universe, all_holdings, pre_checked=True,
        )
        keep_sells = set(sell_only) - keeps

    # ── Research prompt (always available before confirming) ─────────────────
    print(f"\n{_hr()}")
    print("  Research anything else before confirming. ?TICKER for details, Enter to finish.")
    while True:
        raw = input("  > ").strip()
        if not raw:
            break
        if raw.startswith("?"):
            _print_stock_detail(raw[1:].upper(), universe, all_holdings)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{_hr('═')}")
    print(f"  SUMMARY")
    print(_hr("═"))

    new_approvals = approved - set(approved_buys)
    if approved:
        print(f"  Buying ({len(approved)}):  {', '.join(sorted(approved))}")
        if new_approvals:
            print(f"    (new this session: {', '.join(sorted(new_approvals))})")
    else:
        print("  Buying: none")

    if keep_sells:
        print(f"  Keeping (override sell): {', '.join(sorted(keep_sells))}")
    sell_confirmed = set(t for t, _ in sell_tickers) - keep_sells
    if sell_confirmed:
        print(f"  Selling: {', '.join(sorted(sell_confirmed))}")
    print()

    if not decision_file.exists():
        print("  No decision file found — nothing to write.")
        return

    if not new_approvals and not keep_sells:
        print("  No changes to write to decision file.")
        return

    confirm = input(f"  Write to {decision_file.relative_to(_ROOT)}? [Y/n] > ").strip().lower()
    if confirm == "n":
        print("  Aborted — no changes written.")
        return

    _update_markdown(decision_file, approved, keep_sells)
    print(f"  Written. Commit and push to trigger processing.")
    print()


if __name__ == "__main__":
    main()
