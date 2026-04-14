"""
trades.py — Options & stock trade tracker

Menu-driven interactive session. Session state (account + date) persists
across entries so you can log multiple trades without re-entering context.

Usage:
    python trades.py
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path

import requests

_ROOT = Path(__file__).parent.resolve()
os.chdir(_ROOT)

TRADES_DIR    = _ROOT / "trades"
ACCOUNTS_FILE = TRADES_DIR / "accounts.json"
STRATEGIES_FILE = TRADES_DIR / "strategies.json"
POSITIONS_FILE  = TRADES_DIR / "positions.json"
PRICES_DIR    = _ROOT / "data.nosync/prices"

TODAY = date.today().isoformat()

LEG_TYPES    = ["short_call", "short_put", "long_call", "long_put"]
EXIT_REASONS = ["expired", "assigned", "bought_back_profit", "bought_back_loss", "rolled"]

EVALS_FILE          = TRADES_DIR / "strategy_evals.json"
POLYGON_BASE_URL    = "https://api.polygon.io"
POLYGON_RATE_SLEEP  = 12    # seconds between Polygon calls (free tier = 5/min)
EVAL_STALE_DAYS     = 5     # warn if option prices older than this
EVAL_STRATEGY_TYPES = ["covered_call", "leap_cc", "diagonal", "csp", "naked_leap"]
EVAL_STRATEGY_NAMES = {
    "covered_call": "Covered Call",
    "leap_cc":      "LEAP + Covered Call",
    "diagonal":     "Diagonal Spread",
    "csp":          "Cash Secured Put",
    "naked_leap":   "Naked LEAP",
}

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

@dataclass
class Session:
    account: dict | None = None
    trade_date: str = field(default_factory=lambda: date.today().isoformat())

    @property
    def account_name(self) -> str:
        return self.account["name"] if self.account else "(none)"

    def status_line(self) -> str:
        return f"Account: {self.account_name}  |  Date: {self.trade_date}"


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def hr(char: str = "─", width: int = 72) -> str:
    return char * width


def sep() -> None:
    print()


def fmt_dollars(v: float | None, signed: bool = False) -> str:
    if v is None:
        return "—"
    if signed and v > 0:
        return f"+${v:,.2f}"
    return f"${v:,.2f}"


def fmt_ret(r: float | None) -> str:
    if r is None:
        return "  —  "
    sign = "+" if r >= 0 else ""
    return f"{sign}{r:.1%}"


def _wrap(text: str, width: int = 64, indent: str = "    ") -> list[str]:
    words, lines, line = text.split(), [], []
    for w in words:
        if sum(len(x) + 1 for x in line) + len(w) > width:
            lines.append(indent + " ".join(line))
            line = [w]
        else:
            line.append(w)
    if line:
        lines.append(indent + " ".join(line))
    return lines


def print_header(title: str, session: Session | None = None) -> None:
    print()
    print(hr("═"))
    print(f"  {title}")
    if session:
        print(f"  {session.status_line()}")
    print(hr("═"))


def print_section(title: str) -> None:
    print()
    print(f"  {title}")
    print(f"  {hr('─', 66)}")


# ---------------------------------------------------------------------------
# Prompt helpers
# ---------------------------------------------------------------------------

def prompt(label: str, default: str | None = None) -> str:
    suffix = f" [{default}]" if default is not None else ""
    try:
        raw = input(f"  {label}{suffix}: ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return ""
    return raw if raw else (default or "")


def prompt_float(label: str, default: float | None = None) -> float | None:
    default_str = str(default) if default is not None else None
    while True:
        raw = prompt(label, default_str)
        if not raw:
            return None
        try:
            return float(raw)
        except ValueError:
            print(f"  Invalid number: {raw!r}")


def prompt_int(label: str, default: int | None = None) -> int | None:
    default_str = str(default) if default is not None else None
    while True:
        raw = prompt(label, default_str)
        if not raw:
            return None
        try:
            return int(raw)
        except ValueError:
            print(f"  Invalid integer: {raw!r}")


def prompt_date(label: str, default: str | None = None) -> str:
    while True:
        raw = prompt(label, default or TODAY)
        result = _parse_date_flexible(raw)
        if result:
            return result
        print("  Format: YYYY-MM-DD, MM/DD/YYYY, or MM-DD")


def _parse_date_flexible(raw: str) -> str | None:
    """Parse date strings into YYYY-MM-DD.

    Accepted formats:
      YYYY-MM-DD, YYYY/MM/DD
      MM-DD, M-D, MM/DD, M/D          → year inferred; advances to next year if past
      MM/DD/YYYY, M/DD/YYYY, M/D/YYYY
      MM/DD/YY, M/D/YY                → 2-digit year: 00-49 → 2000s, 50-99 → 1900s
    Returns None if unparseable.
    """
    raw = raw.strip()
    # Normalise: replace slashes with dashes
    normalised = raw.replace("/", "-")
    parts = normalised.split("-")

    if len(parts) == 3:
        # Could be YYYY-MM-DD or MM-DD-YYYY or MM-DD-YY
        p0, p1, p2 = parts
        if len(p0) == 4:
            # YYYY-MM-DD
            try:
                datetime.strptime(normalised, "%Y-%m-%d")
                return normalised
            except ValueError:
                return None
        else:
            # MM-DD-YYYY or MM-DD-YY
            try:
                month, day = int(p0), int(p1)
                year = int(p2)
                if year < 100:
                    year += 2000 if year < 50 else 1900
                return date(year, month, day).isoformat()
            except ValueError:
                return None

    if len(parts) == 2:
        # MM-DD (year inferred)
        try:
            today = date.today()
            parsed = date(today.year, int(parts[0]), int(parts[1]))
            if parsed < today:
                parsed = parsed.replace(year=today.year + 1)
            return parsed.isoformat()
        except ValueError:
            pass

    return None


def prompt_expiry(label: str = "  Expiry") -> str:
    """Prompt for an options expiry date.
    Accepts YYYY-MM-DD, MM-DD, M-D, M/D (year inferred; next year if already past).
    """
    while True:
        raw = prompt(label).strip()
        if not raw:
            continue
        result = _parse_date_flexible(raw)
        if result:
            return result
        print("  Format: YYYY-MM-DD or MM-DD")


def prompt_choice(label: str, choices: list[str], default: str | None = None) -> str:
    while True:
        raw = prompt(f"{label} ({'/'.join(choices)})", default)
        if raw in choices:
            return raw
        print(f"  Must be one of: {', '.join(choices)}")


def pick_from_list(items: list[str], label: str = "Select") -> int | None:
    """Print numbered list, return 0-based index. Returns None on blank/cancel."""
    for i, item in enumerate(items):
        print(f"    {i + 1}.  {item}")
    while True:
        raw = prompt(label)
        if not raw:
            return None
        try:
            idx = int(raw) - 1
            if 0 <= idx < len(items):
                return idx
            print(f"  Enter 1–{len(items)}")
        except ValueError:
            print("  Enter a number")


def confirm(label: str = "Confirm") -> bool:
    return prompt(label, "Y").lower() in ("y", "yes", "")


# ---------------------------------------------------------------------------
# Data I/O
# ---------------------------------------------------------------------------

def _ensure_dir() -> None:
    TRADES_DIR.mkdir(exist_ok=True)


def _load(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path) as f:
        return json.load(f)


def _save(path: Path, data: list[dict]) -> None:
    _ensure_dir()
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    tmp.replace(path)


def load_accounts() -> list[dict]:   return _load(ACCOUNTS_FILE)
def save_accounts(d: list[dict]):    _save(ACCOUNTS_FILE, d)
def load_strategies() -> list[dict]: return _load(STRATEGIES_FILE)
def save_strategies(d: list[dict]):  _save(STRATEGIES_FILE, d)
def load_positions() -> list[dict]:  return _load(POSITIONS_FILE)
def save_positions(d: list[dict]):   _save(POSITIONS_FILE, d)
def load_evals() -> list[dict]:      return _load(EVALS_FILE)
def save_evals(d: list[dict]):       _save(EVALS_FILE, d)


# ---------------------------------------------------------------------------
# Price utilities
# ---------------------------------------------------------------------------

_price_cache: dict[str, dict[str, float]] = {}


def _price_dates() -> list[str]:
    return sorted(f.stem for f in PRICES_DIR.glob("*.json") if f.stem[0].isdigit())


def _load_price_file(date_str: str) -> dict[str, float]:
    if date_str in _price_cache:
        return _price_cache[date_str]
    path = PRICES_DIR / f"{date_str}.json"
    if not path.exists():
        return {}
    with open(path) as f:
        records = json.load(f)
    result = {r["ticker"]: r["close"] for r in records if r.get("close")}
    _price_cache[date_str] = result
    return result


def get_price(ticker: str, date_str: str) -> float | None:
    for d in reversed([d for d in _price_dates() if d <= date_str]):
        prices = _load_price_file(d)
        if ticker in prices:
            return prices[ticker]
    return None


def get_latest_price(ticker: str) -> tuple[str, float] | None:
    for d in reversed(_price_dates()):
        prices = _load_price_file(d)
        if ticker in prices:
            return (d, prices[ticker])
    return None


def build_occ_ticker(ticker: str, expiry: str, opt_type: str, strike: float) -> str:
    """Build OCC options contract ticker.
    expiry: YYYY-MM-DD, opt_type: 'call' or 'put'
    Example: AAPL, 2026-05-16, call, 195.0 → O:AAPL260516C00195000
    """
    yy, mm, dd = expiry[2:4], expiry[5:7], expiry[8:10]
    cp = "C" if opt_type.lower() == "call" else "P"
    strike_int = int(round(strike * 1000))
    return f"O:{ticker}{yy}{mm}{dd}{cp}{strike_int:08d}"


def fetch_option_price(occ_ticker: str, as_of_date: str) -> float | None:
    """Fetch most recent closing price for an OCC options ticker from Polygon.
    Searches up to 10 calendar days back from as_of_date.
    Returns closing premium per share, or None on failure.
    """
    api_key = os.environ.get("POLYGON_API_KEY")
    if not api_key:
        print("  [!] POLYGON_API_KEY not set — cannot fetch option prices.")
        return None
    from_date = (datetime.strptime(as_of_date, "%Y-%m-%d") - timedelta(days=30)).strftime("%Y-%m-%d")
    url = f"{POLYGON_BASE_URL}/v2/aggs/ticker/{occ_ticker}/range/1/day/{from_date}/{as_of_date}"
    try:
        resp = requests.get(url, params={"apiKey": api_key, "adjusted": "true",
                                         "sort": "desc", "limit": 1}, timeout=30)
        if resp.status_code == 200:
            results = resp.json().get("results", [])
            if results:
                return results[0]["c"]
        elif resp.status_code != 404:
            print(f"  [!] Polygon HTTP {resp.status_code} for {occ_ticker}")
    except requests.RequestException as exc:
        print(f"  [!] Network error: {exc}")
    return None


# ---------------------------------------------------------------------------
# Analytics
# ---------------------------------------------------------------------------

def compute_analytics(pos: dict) -> dict:
    ticker      = pos["ticker"]
    entry_price = pos["entry_price"]
    entry_date  = pos["entry_date"]
    shares      = pos.get("shares", 0)
    status      = pos.get("status", "open")

    out: dict = {
        "stock_log_ret": None, "spy_log_ret": None, "excess_ret": None,
        "options_pnl": None,   "options_yield": None, "total_log_ret": None,
        "holding_days": None,  "current_price": None, "current_date": None,
    }

    if status == "closed":
        exit_price = pos.get("exit_price")
        exit_date  = pos.get("exit_date")
        if not exit_price or not exit_date:
            return out
        current_price, current_date = exit_price, exit_date
    else:
        info = get_latest_price(ticker)
        if not info:
            return out
        current_date, current_price = info

    out["current_price"] = current_price
    out["current_date"]  = current_date
    out["stock_log_ret"] = math.log(current_price / entry_price)

    d0 = datetime.strptime(entry_date, "%Y-%m-%d").date()
    d1 = datetime.strptime(current_date, "%Y-%m-%d").date()
    out["holding_days"] = (d1 - d0).days

    spy_entry   = get_price("SPY", entry_date)
    spy_current = get_price("SPY", current_date)
    if spy_entry and spy_current:
        out["spy_log_ret"] = math.log(spy_current / spy_entry)
        out["excess_ret"]  = out["stock_log_ret"] - out["spy_log_ret"]

    pnl, has_closed = 0.0, False
    for leg in pos.get("legs", []):
        if leg.get("exit_premium") is None:
            continue
        has_closed = True
        ep, xp     = leg.get("entry_premium", 0), leg.get("exit_premium", 0)
        contracts  = leg.get("contracts", 1)
        if leg.get("leg_type", "").startswith("short"):
            pnl += (ep - xp) * contracts * 100
        else:
            pnl += (xp - ep) * contracts * 100

    if has_closed:
        out["options_pnl"] = pnl
        stock_val = entry_price * shares
        if stock_val > 0:
            out["options_yield"] = pnl / stock_val
            # True log return: log((ending_stock_value + options_pnl) / entry_stock_value)
            ending_stock_val = current_price * shares
            out["total_log_ret"] = math.log((ending_stock_val + pnl) / stock_val)

    return out


def _leg_pnl(leg: dict) -> float | None:
    if leg.get("exit_premium") is None:
        return None
    ep, xp    = leg.get("entry_premium", 0), leg.get("exit_premium", 0)
    contracts = leg.get("contracts", 1)
    if leg.get("leg_type", "").startswith("short"):
        return (ep - xp) * contracts * 100
    return (xp - ep) * contracts * 100


# ---------------------------------------------------------------------------
# Shared position/leg display
# ---------------------------------------------------------------------------

def show_open_positions_table(positions: list[dict]) -> None:
    open_pos = [p for p in positions if p.get("status") == "open"]
    if not open_pos:
        print("  No open positions.")
        return

    print(f"  {'#':<3} {'Ticker':<6} {'Account':<22} {'Shrs':>5} {'Entry':>8} "
          f"{'Current':>8} {'Return':>7} {'vs SPY':>7} {'Legs':>5}")
    print(f"  {hr('─', 68)}")
    for i, pos in enumerate(open_pos, 1):
        a     = compute_analytics(pos)
        acct  = pos.get("account_name", pos.get("account_id", "?"))[:22]
        curr  = a.get("current_price")
        legs  = sum(1 for l in pos.get("legs", []) if l.get("exit_date") is None)
        curr_str = f"${curr:,.2f}" if curr else "  —  "
        print(f"  {i:<3} {pos['ticker']:<6} {acct:<22} {pos.get('shares',0):>5} "
              f"${pos['entry_price']:>7,.2f} "
              f"{curr_str:>8} "
              f"{fmt_ret(a.get('stock_log_ret')):>7} "
              f"{fmt_ret(a.get('excess_ret')):>7} {legs:>5}")


def show_position_detail(pos: dict) -> None:
    a = compute_analytics(pos)
    print()
    print(f"  {pos['ticker']}  —  {pos.get('account_name', pos.get('account_id', '?'))}")
    print(f"  Entry {pos['entry_date']} @ ${pos['entry_price']:,.2f}  x{pos.get('shares',0)} shares"
          + (f"  →  exit {pos['exit_date']} @ ${pos.get('exit_price',0):,.2f}" if pos.get("exit_date") else ""))

    if a.get("stock_log_ret") is not None:
        label = "Return" if pos["status"] == "closed" else "Return (unrealized)"
        print(f"  {label}: {fmt_ret(a['stock_log_ret'])}  "
              f"SPY: {fmt_ret(a.get('spy_log_ret'))}  "
              f"Excess: {fmt_ret(a.get('excess_ret'))}")
    if a.get("options_pnl") is not None:
        print(f"  Options P&L: {fmt_dollars(a['options_pnl'], signed=True)}  "
              f"Yield: {fmt_ret(a.get('options_yield'))}")

    legs = pos.get("legs", [])
    if legs:
        print(f"  {hr('─', 60)}")
        for leg in legs:
            pnl = _leg_pnl(leg)
            delta_str = f"  δ={leg['entry_delta']:.2f}" if leg.get("entry_delta") else ""
            iv_str    = f"  iv={leg['entry_iv']:.0%}" if leg.get("entry_iv") else ""
            if leg.get("exit_date"):
                outcome = f"→ closed {leg['exit_date']} ({leg.get('exit_reason','?')})  P&L: {fmt_dollars(pnl, signed=True)}"
            else:
                outcome = "→ OPEN"
            print(f"  [{leg['leg_id']}] {leg.get('strategy_id','?')}  {leg['leg_type']}"
                  f"  ${leg['strike']}  exp {leg['expiry']}  x{leg['contracts']}"
                  f"{delta_str}{iv_str}")
            print(f"    entry {leg['entry_date']} @ ${leg['entry_premium']:.2f}  {outcome}")


# ---------------------------------------------------------------------------
# Strategy evaluation helpers
# ---------------------------------------------------------------------------

def _eval_stale_warning(evals: list[dict]) -> None:
    """Print a one-line warning if any open eval has prices older than EVAL_STALE_DAYS."""
    today = date.today()
    stale_tickers: list[str] = []
    for ev in evals:
        if ev.get("status") != "open":
            continue
        for contract in ev.get("contracts", {}).values():
            if not contract:
                continue
            price_date = contract.get("price_date")
            if not price_date:
                stale_tickers.append(ev.get("ticker", "?"))
                break
            age = (today - datetime.strptime(price_date, "%Y-%m-%d").date()).days
            if age > EVAL_STALE_DAYS:
                stale_tickers.append(ev.get("ticker", "?"))
                break
    if stale_tickers:
        unique = sorted(set(stale_tickers))
        print(f"  [!] Stale option prices (>{EVAL_STALE_DAYS}d): {', '.join(unique)}"
              f"  —  run  r  Reprice to update")


def _compute_eval_analytics(ev: dict, current_stock_price: float | None) -> None:
    """(Re)compute all strategy analytics on an eval record in-place.

    All returns are true log returns so they can be averaged across positions.
    Works on a per-share basis — share count does not affect the result.

    Formulas:
      Covered Call : log( (S1 + C_entry - C_current) / S0 )
      Naked LEAP   : log( L_current / L_entry )
      LEAP + CC    : log( (L_current - C_current) / (L_entry - C_entry) )
      Diagonal     : log( (ITM_current - C_current) / (ITM_entry - C_entry) )
      CSP          : log( (K + P_entry - P_current) / K )

    Each active strategy stores: log_ret, excess_ret (vs SPY), annualized_ret.
    ev also gets: stock_log_ret, stock_excess_ret, stock_annualized_ret, holding_days.
    """
    contracts = ev.get("contracts", {})
    S0 = ev.get("underlying_price") or 0
    S1 = current_stock_price if current_stock_price is not None else S0

    sc  = contracts.get("short_call")
    atm = contracts.get("atm_leap")
    itm = contracts.get("itm_leap")
    sp  = contracts.get("short_put")

    def _has(c: dict | None) -> bool:
        return bool(c and c.get("entry_premium") is not None
                    and c.get("current_premium") is not None)

    def _log_ret(ending: float, beginning: float) -> float | None:
        if beginning > 0 and ending > 0:
            return math.log(ending / beginning)
        return None

    # ── Holding period & SPY benchmark ──────────────────────────────────────
    eval_date_str = ev.get("eval_date")
    price_dates   = [c["price_date"] for c in contracts.values()
                     if c and c.get("price_date")]
    latest_pd     = max(price_dates) if price_dates else eval_date_str
    holding_days  = 1
    if eval_date_str and latest_pd:
        d0 = datetime.strptime(eval_date_str, "%Y-%m-%d").date()
        d1 = datetime.strptime(latest_pd, "%Y-%m-%d").date()
        holding_days = max(1, (d1 - d0).days)

    spy0   = get_price("SPY", eval_date_str) if eval_date_str else None
    spy1   = get_price("SPY", latest_pd)     if latest_pd     else None
    spy_lr = math.log(spy1 / spy0) if spy0 and spy1 else None

    def _augment(lr: float | None) -> dict:
        """Build a strategy dict with all 3 metrics."""
        if lr is None:
            return {"active": False}
        return {
            "active":         True,
            "log_ret":        lr,
            "excess_ret":     (lr - spy_lr) if spy_lr is not None else None,
            "annualized_ret": lr * 252 / holding_days,
        }

    strategies: dict = {}

    # Covered Call: ending value per share = S1 + net premium collected
    if _has(sc) and S0 > 0:
        ending = S1 + sc["entry_premium"] - sc["current_premium"]
        strategies["covered_call"] = _augment(_log_ret(ending, S0))
    else:
        strategies["covered_call"] = {"active": False}

    # LEAP + Covered Call: return on net debit (long - short)
    if _has(atm) and _has(sc):
        net_entry   = atm["entry_premium"]   - sc["entry_premium"]
        net_current = atm["current_premium"] - sc["current_premium"]
        strategies["leap_cc"] = _augment(_log_ret(net_current, net_entry))
    else:
        strategies["leap_cc"] = {"active": False}

    # Diagonal: return on net debit (ITM long - short)
    if _has(itm) and _has(sc):
        net_entry   = itm["entry_premium"]   - sc["entry_premium"]
        net_current = itm["current_premium"] - sc["current_premium"]
        strategies["diagonal"] = _augment(_log_ret(net_current, net_entry))
    else:
        strategies["diagonal"] = {"active": False}

    # CSP: return on cash committed (strike)
    if _has(sp):
        K = sp.get("strike") or 0
        ending = K + sp["entry_premium"] - sp["current_premium"]
        strategies["csp"] = _augment(_log_ret(ending, K))
    else:
        strategies["csp"] = {"active": False}

    # Naked LEAP: return on premium paid
    if _has(atm):
        strategies["naked_leap"] = _augment(_log_ret(atm["current_premium"], atm["entry_premium"]))
    else:
        strategies["naked_leap"] = {"active": False}

    ev["strategies"] = strategies

    # ── Stock comparison over the same eval window ───────────────────────────
    if S0 > 0 and S1:
        slr = math.log(S1 / S0)
        ev["stock_log_ret"]        = slr
        ev["stock_excess_ret"]     = (slr - spy_lr) if spy_lr is not None else None
        ev["stock_annualized_ret"] = slr * 252 / holding_days
    else:
        ev["stock_log_ret"] = ev["stock_excess_ret"] = ev["stock_annualized_ret"] = None
    ev["holding_days"] = holding_days


def _fetch_contracts_batch(
    contracts_to_fetch: list[tuple[str, str]],
    as_of_date: str,
) -> dict[str, float | None]:
    """Fetch prices for a list of (display_label, occ_ticker) tuples.
    Shows progress, sleeps between calls for rate limit.
    Returns dict mapping occ_ticker -> closing price or None.
    """
    results: dict[str, float | None] = {}
    total = len(contracts_to_fetch)
    for i, (label, occ_ticker) in enumerate(contracts_to_fetch, 1):
        print(f"  [{i}/{total}] {occ_ticker} ({label}) ...", end=" ", flush=True)
        price = fetch_option_price(occ_ticker, as_of_date)
        if price is not None:
            print(f"${price:.2f} ✓")
        else:
            print("no data")
        results[occ_ticker] = price
        if i < total:
            next_label, next_occ = contracts_to_fetch[i]
            print(f"  Next up [{i+1}/{total}]: {next_label}  —  {next_occ}")
            print(f"  (waiting {POLYGON_RATE_SLEEP}s ...)")
            time.sleep(POLYGON_RATE_SLEEP)
    return results


# ---------------------------------------------------------------------------
# Core flows (used from multiple menus)
# ---------------------------------------------------------------------------

def flow_add_leg(pos: dict, strategies: list[dict], positions: list[dict],
                 trade_date: str) -> bool:
    """Prompt for and save a new options leg on pos. Returns True if saved."""
    active = [s for s in strategies if s.get("active", True)]
    if not active:
        print("  No strategies defined yet. Add one from the main menu (t).")
        return False

    print()
    print(f"  Add leg to: {pos['id']}")
    print()

    # Pick strategy
    labels = [f"{s['name']}  ({s['id']})" for s in active]
    print("  Strategy:")
    idx = pick_from_list(labels, "Strategy number")
    if idx is None:
        return False
    strategy = active[idx]

    # Show playbook reminder
    if strategy.get("approach") or strategy.get("notes"):
        print(f"\n  {hr('─', 60)}")
        print(f"  {strategy['name']} — your approach:")
        if strategy.get("approach"):
            for line in _wrap(strategy["approach"]):
                print(line)
        notes = strategy.get("notes", [])
        if notes:
            latest = sorted(notes, key=lambda n: n["date"])[-1]
            print(f"\n  Latest note ({latest['date']}):")
            for line in _wrap(latest["text"]):
                print(line)
        print(f"  {hr('─', 60)}")

    print()
    leg_type = prompt_choice("Leg type", LEG_TYPES)
    strike   = prompt_float("Strike")
    if strike is None:
        return False
    expiry     = prompt_date("Expiry", None)
    contracts  = prompt_int("Contracts", 1) or 1
    entry_date = prompt_date("Entry date", trade_date)
    entry_prem = prompt_float("Entry premium (per share)")
    if entry_prem is None:
        return False

    delta_raw  = prompt("Delta (optional, e.g. 0.24)")
    iv_raw     = prompt("IV (optional, e.g. 0.38)")
    entry_delta = float(delta_raw) if delta_raw else None
    entry_iv    = float(iv_raw)    if iv_raw    else None

    existing = {l["leg_id"] for l in pos.get("legs", [])}
    n = 1
    while f"leg_{n}" in existing:
        n += 1

    leg: dict = {
        "leg_id": f"leg_{n}",
        "strategy_id": strategy["id"],
        "leg_type": leg_type,
        "strike": strike,
        "expiry": expiry,
        "contracts": contracts,
        "entry_date": entry_date,
        "entry_premium": entry_prem,
        "entry_delta": entry_delta,
        "entry_iv": entry_iv,
        "exit_date": None,
        "exit_premium": None,
        "exit_reason": None,
    }

    already_closed = prompt("Already closed (past leg)? [y/N]", "N").lower() == "y"
    if already_closed:
        leg["exit_date"]    = prompt_date("Exit date", expiry)
        leg["exit_premium"] = prompt_float("Exit premium (0 if expired)") or 0.0
        leg["exit_reason"]  = prompt_choice("Exit reason", EXIT_REASONS, "expired")

    pos.setdefault("legs", []).append(leg)
    save_positions(positions)

    pnl_str = f"  P&L: {fmt_dollars(_leg_pnl(leg), signed=True)}" if leg.get("exit_date") else ""
    print(f"  Saved {leg['leg_id']} ({strategy['name']}, {leg_type} ${strike} exp {expiry}).{pnl_str}")
    return True


def flow_close_leg(positions: list[dict], filter_account_id: str | None = None) -> None:
    """Show all open legs, let user pick one to close."""
    source = [p for p in positions if p.get("status") == "open"]
    if filter_account_id:
        source = [p for p in source if p.get("account_id") == filter_account_id]

    open_legs: list[tuple[dict, dict]] = [
        (pos, leg)
        for pos in source
        for leg in pos.get("legs", [])
        if leg.get("exit_date") is None
    ]

    if not open_legs:
        print("  No open legs found.")
        return

    print()
    labels = [
        f"{pos['ticker']}  [{leg['leg_id']}]  {leg.get('strategy_id','?')}  "
        f"{leg['leg_type']}  ${leg['strike']}  exp {leg['expiry']}  "
        f"prem ${leg['entry_premium']:.2f}  —  {pos.get('account_name', pos.get('account_id','?'))}"
        for pos, leg in open_legs
    ]
    idx = pick_from_list(labels, "Leg to close")
    if idx is None:
        return

    pos, leg = open_legs[idx]
    print()
    leg["exit_date"]    = prompt_date("Exit date", leg["expiry"])
    leg["exit_premium"] = prompt_float("Exit premium (0 if expired)") if True else 0.0
    if leg["exit_premium"] is None:
        leg["exit_premium"] = 0.0
    leg["exit_reason"]  = prompt_choice("Exit reason", EXIT_REASONS, "expired")

    save_positions(positions)
    pnl = _leg_pnl(leg)
    print(f"  Closed {leg['leg_id']} on {pos['ticker']}.  P&L: {fmt_dollars(pnl, signed=True)}")


def flow_close_position(positions: list[dict], filter_account_id: str | None = None) -> None:
    """Show open positions, let user pick one to close."""
    source = [p for p in positions if p.get("status") == "open"]
    if filter_account_id:
        source = [p for p in source if p.get("account_id") == filter_account_id]

    if not source:
        print("  No open positions found.")
        return

    print()
    labels = [
        f"{p['ticker']}  entry {p['entry_date']} @ ${p['entry_price']:,.2f}"
        f"  x{p.get('shares',0)}  —  {p.get('account_name', p.get('account_id','?'))}"
        for p in source
    ]
    idx = pick_from_list(labels, "Position to close")
    if idx is None:
        return

    pos = source[idx]
    open_legs = [l for l in pos.get("legs", []) if l.get("exit_date") is None]

    print()
    pos["exit_date"]  = prompt_date("Exit date", TODAY)

    # Fetch the historical price based on the ticker and exit date
    proposed_exit_price = get_price(pos["ticker"], pos["exit_date"])

    exit_price = prompt_float("Exit price", proposed_exit_price)

    if exit_price is None:
        return
    pos["exit_price"] = exit_price
    pos["status"]     = "closed"
    save_positions(positions)

    # Mark any open strategy evals for this position as closed
    evals = load_evals()
    changed = False
    for ev in evals:
        if ev.get("position_id") == pos["id"] and ev.get("status") == "open":
            ev["status"] = "closed"
            changed = True
    if changed:
        save_evals(evals)

    a = compute_analytics(pos)
    print(f"  Closed {pos['ticker']}.  "
          f"Stock return: {fmt_ret(a.get('stock_log_ret'))}  "
          f"vs SPY: {fmt_ret(a.get('excess_ret'))}")

    # Offer to close any open options legs
    if open_legs and prompt(f"\n  Close {len(open_legs)} open option leg(s)? [Y/n]", "Y").lower() in ("y", "yes", ""):
        for leg in open_legs:
            print()
            print(f"  [{leg['leg_id']}] {leg.get('strategy_id','?')}  "
                  f"{leg['leg_type']}  ${leg['strike']}  exp {leg['expiry']}  "
                  f"entry ${leg['entry_premium']:.2f}")
            leg["exit_date"]    = prompt_date("  Exit date", pos["exit_date"])
            leg["exit_premium"] = prompt_float("  Exit premium (0 if expired)") or 0.0
            leg["exit_reason"]  = prompt_choice("  Exit reason", EXIT_REASONS, "expired")
            pnl = _leg_pnl(leg)
            print(f"  P&L: {fmt_dollars(pnl, signed=True)}")
        save_positions(positions)


def flow_evaluate_strategies(pos: dict, evals: list[dict], next_ticker: str | None = None) -> bool:
    """Prompt for up to 4 contract inputs, batch-fetch prices from Polygon,
    compute all 5 strategy evaluations, and save. Returns True if saved.

    At any strike or expiry prompt, enter  b  to go back one step.
    At the price-date, fetch, and save prompts, b also steps backward.
    """
    CONTRACT_DEFS = [
        ("short_call", "Short Call",    "Covered Call, LEAP+CC, Diagonal", "call"),
        ("atm_leap",   "ATM/OTM LEAP", "LEAP+CC, Naked LEAP",             "call"),
        ("itm_leap",   "ITM LEAP",      "Diagonal only",                   "call"),
        ("short_put",  "Short Put",     "CSP only",                        "put"),
    ]
    LABEL_MAP = {"short_call": "Short Call", "atm_leap": "ATM LEAP",
                 "itm_leap": "ITM LEAP", "short_put": "Short Put"}

    ticker      = pos["ticker"]
    entry_price = pos["entry_price"]
    shares      = pos.get("shares", 0)

    info = get_latest_price(ticker)
    current_price = info[1] if info else entry_price
    current_date  = info[0] if info else TODAY
    stock_log_ret_display = math.log(current_price / entry_price) if entry_price and current_price else None

    print()
    print(f"  Evaluating strategies for {ticker}  "
          f"(entry ${entry_price:,.2f}  ×{shares} shares  |  "
          f"current ${current_price:,.2f}  {fmt_ret(stock_log_ret_display)})")
    print(f"  Enter b at any prompt to go back one step.")
    print()

    # ── Suggest reusing contracts from another position with same ticker ────────
    same_ticker_evals = sorted(
        [ev for ev in evals
         if ev.get("ticker") == ticker
         and ev.get("position_id") != pos["id"]
         and ev.get("status") == "open"
         and any(v for v in ev.get("contracts", {}).values() if v)],
        key=lambda e: e.get("eval_date", ""),
        reverse=True,
    )
    reuse_contracts: dict[str, dict | None] = {}
    reuse_has_prices = False
    reuse_price_date: str | None = None
    if same_ticker_evals:
        src = same_ticker_evals[0]
        print(f"  Found existing {ticker} eval from {src.get('eval_date', '?')}.")
        parts = []
        for ck, cv in src.get("contracts", {}).items():
            if cv:
                lbl = {"short_call": "SC", "atm_leap": "ATM", "itm_leap": "ITM", "short_put": "SP"}.get(ck, ck)
                prem_str = f" @ ${cv['entry_premium']:.2f}" if cv.get("entry_premium") is not None else ""
                parts.append(f"{lbl} ${cv['strike']} exp {cv['expiry']}{prem_str}")
        print(f"  Contracts: {', '.join(parts)}")
        if confirm("  Reuse these contracts?"):
            for ck, cv in src.get("contracts", {}).items():
                if cv:
                    reuse_contracts[ck] = dict(cv)  # copy all fields including premiums
                else:
                    reuse_contracts[ck] = None
            # Check if prices are already populated
            price_dates = [cv["price_date"] for cv in reuse_contracts.values()
                           if cv and cv.get("price_date")]
            reuse_has_prices = any(cv.get("entry_premium") is not None
                                   for cv in reuse_contracts.values() if cv)
            reuse_price_date = max(price_dates) if price_dates else src.get("eval_date")
        print()

    # ── Phase state machine ──────────────────────────────────────────────────
    # Phases: "contracts" → "price_date" → "fetch_prompt" → "fetch_or_manual" → "save"
    phase         = "contract_review" if reuse_contracts else "contracts"
    contract_idx  = 0
    raw_contracts: dict[str, dict | None] = dict(reuse_contracts)
    short_expiry: str | None = None   # default for short_call / short_put
    leap_expiry:  str | None = None   # default for atm_leap / itm_leap
    price_date    = TODAY
    use_polygon   = True
    prices: dict[str, float | None] = {}
    ev: dict = {}

    while True:

        # ── Contract entry ───────────────────────────────────────────────────
        if phase == "contracts":
            if contract_idx >= len(CONTRACT_DEFS):
                phase = "contract_review"
                continue

            key, label, used_by, opt_type = CONTRACT_DEFS[contract_idx]
            print(f"  ── {label}  (used by: {used_by})")
            strike_raw = prompt("  Strike (Enter to skip, b to go back)").strip()

            if strike_raw.lower() == "b":
                if contract_idx == 0:
                    return False
                contract_idx -= 1
                raw_contracts.pop(CONTRACT_DEFS[contract_idx][0], None)
                continue

            if not strike_raw:
                raw_contracts[key] = None
                contract_idx += 1
                print()
                continue

            try:
                strike = float(strike_raw)
            except ValueError:
                print("  Invalid strike — skipping.")
                raw_contracts[key] = None
                contract_idx += 1
                print()
                continue

            # Expiry sub-prompt — b goes back to this contract's strike
            expiry: str | None = None
            _is_leap   = key in ("atm_leap", "itm_leap")
            expiry_default = (leap_expiry if _is_leap else short_expiry) or ""
            while True:
                raw = prompt("  Expiry (YYYY-MM-DD, MM-DD, or b)", expiry_default).strip()
                if raw.lower() == "b":
                    break  # expiry = None → re-prompt strike for same contract
                result = _parse_date_flexible(raw)
                if result:
                    expiry = result
                    if _is_leap:
                        leap_expiry = result
                    else:
                        short_expiry = result
                    break
                print("  Format: YYYY-MM-DD or MM-DD")

            if expiry is None:
                print()
                continue  # re-prompt strike for same contract_idx

            occ = build_occ_ticker(ticker, expiry, opt_type, strike)
            print(f"    → {occ}")
            print()
            raw_contracts[key] = {
                "strike": strike, "expiry": expiry,
                "occ_ticker": occ, "option_type": opt_type,
                "entry_premium": None, "current_premium": None, "price_date": None,
            }
            contract_idx += 1

        # ── Contract review ──────────────────────────────────────────────────
        elif phase == "contract_review":
            sc  = raw_contracts.get("short_call")
            atm = raw_contracts.get("atm_leap")
            itm = raw_contracts.get("itm_leap")
            sp  = raw_contracts.get("short_put")

            # Build list of missing contracts that would unlock at least one strategy
            missing: list[tuple[str, str, str, str]] = []
            if not sc:
                missing.append(("short_call", "Short Call",    "Covered Call, LEAP+CC, Diagonal", "call"))
            if not atm:
                missing.append(("atm_leap",   "ATM/OTM LEAP", "LEAP+CC, Naked LEAP",             "call"))
            if not itm:
                missing.append(("itm_leap",   "ITM LEAP",      "Diagonal only",                   "call"))
            if not sp:
                missing.append(("short_put",  "Short Put",     "CSP only",                        "put"))

            if missing:
                inactive_names: list[str] = []
                if not sc:
                    inactive_names += ["Covered Call", "LEAP+CC", "Diagonal"]
                if not atm:
                    inactive_names += [n for n in ["LEAP+CC", "Naked LEAP"] if n not in inactive_names]
                if not itm:
                    inactive_names += [n for n in ["Diagonal"] if n not in inactive_names]
                if not sp:
                    inactive_names += ["CSP"]

                print()
                print(f"  Inactive strategies: {', '.join(inactive_names)}")
                for _, lbl, used_by, _ in missing:
                    print(f"    {lbl:<16}  would enable: {used_by}")
                print()

                rev = prompt("  Add missing contracts? [y/N/b]", "N").strip().lower()
                if rev == "b":
                    contract_idx = len(CONTRACT_DEFS) - 1
                    raw_contracts.pop(CONTRACT_DEFS[contract_idx][0], None)
                    phase = "contracts"
                    continue
                if rev in ("y", "yes"):
                    for m_key, m_label, m_used_by, m_opt_type in missing:
                        print(f"\n  ── {m_label}  (used by: {m_used_by})")
                        # Strike
                        m_strike: float | None = None
                        while True:
                            s_raw = prompt("  Strike (Enter to skip)").strip()
                            if not s_raw:
                                break
                            try:
                                m_strike = float(s_raw)
                                break
                            except ValueError:
                                print("  Invalid strike.")
                        if m_strike is None:
                            raw_contracts[m_key] = None
                            continue
                        # Expiry
                        m_expiry: str | None = None
                        _m_is_leap = m_key in ("atm_leap", "itm_leap")
                        _m_default = (leap_expiry if _m_is_leap else short_expiry) or ""
                        while True:
                            e_raw = prompt("  Expiry (YYYY-MM-DD or MM-DD)", _m_default).strip()
                            m_expiry = _parse_date_flexible(e_raw)
                            if m_expiry:
                                if _m_is_leap:
                                    leap_expiry = m_expiry
                                else:
                                    short_expiry = m_expiry
                                break
                            print("  Format: YYYY-MM-DD or MM-DD")
                        m_occ = build_occ_ticker(ticker, m_expiry, m_opt_type, m_strike)
                        print(f"    → {m_occ}")
                        raw_contracts[m_key] = {
                            "strike": m_strike, "expiry": m_expiry,
                            "occ_ticker": m_occ, "option_type": m_opt_type,
                            "entry_premium": None, "current_premium": None, "price_date": None,
                        }
                    print()

            if reuse_has_prices:
                # Prices already copied from source eval — build ev and skip to save
                price_date = reuse_price_date or TODAY
                ev = {
                    "position_id":      pos["id"],
                    "ticker":           ticker,
                    "eval_date":        price_date,
                    "underlying_price": current_price,
                    "status":           "open",
                    "contracts":        dict(raw_contracts),
                    "strategies":       {},
                }
                _compute_eval_analytics(ev, current_price)
                phase = "save"
            else:
                phase = "price_date"

        # ── Price date ───────────────────────────────────────────────────────
        elif phase == "price_date":
            raw = prompt("  Entry price date (or b to go back)", TODAY).strip()
            if raw.lower() == "b":
                phase = "contract_review"
                continue
            result = _parse_date_flexible(raw) or (raw if raw == TODAY else None)
            if result:
                price_date = result
                phase = "fetch_prompt"
            else:
                print("  Format: YYYY-MM-DD or MM-DD")

        # ── Fetch prompt ─────────────────────────────────────────────────────
        elif phase == "fetch_prompt":
            filled = {k: v for k, v in raw_contracts.items() if v is not None}
            if not filled:
                print("  No contracts entered — nothing to evaluate.")
                return False

            active = []
            if raw_contracts.get("short_call"):                                    active.append("Covered Call")
            if raw_contracts.get("atm_leap") and raw_contracts.get("short_call"): active.append("LEAP+CC")
            if raw_contracts.get("itm_leap") and raw_contracts.get("short_call"): active.append("Diagonal")
            if raw_contracts.get("short_put"):                                     active.append("CSP")
            if raw_contracts.get("atm_leap"):                                      active.append("Naked LEAP")
            print(f"  Strategies: {', '.join(active)}")
            print()

            if next_ticker:
                print(f"  Next position after this: {next_ticker}")
            raw = prompt("  Fetch prices from Polygon? [Y/n/b]", "Y").strip().lower()
            if raw == "b":
                phase = "price_date"
                continue
            use_polygon = raw in ("y", "yes", "")
            phase = "fetch_or_manual"

        # ── Fetch or manual entry ────────────────────────────────────────────
        elif phase == "fetch_or_manual":
            filled = {k: v for k, v in raw_contracts.items() if v is not None}
            to_fetch = [(LABEL_MAP[k], v["occ_ticker"]) for k, v in filled.items()]
            print()
            if use_polygon:
                prices = _fetch_contracts_batch(to_fetch, price_date)
            else:
                for k, contract in filled.items():
                    p = prompt_float(f"  Premium for {contract['occ_ticker']}")
                    prices[contract["occ_ticker"]] = p
            print()

            # Write prices into contracts
            for k, contract in filled.items():
                p = prices.get(contract["occ_ticker"])
                contract["entry_premium"]   = p
                contract["current_premium"] = p
                contract["price_date"]      = price_date if p is not None else None

            # Build eval and compute
            ev = {
                "position_id":      pos["id"],
                "ticker":           ticker,
                "eval_date":        price_date,
                "underlying_price": current_price,
                "status":           "open",
                "contracts":        dict(raw_contracts),
                "strategies":       {},
            }
            _compute_eval_analytics(ev, current_price)
            phase = "save"

        # ── Save ─────────────────────────────────────────────────────────────
        elif phase == "save":
            print_section("Contracts Captured")
            LABEL_MAP2 = {
                "short_call": "Short Call",
                "atm_leap":   "ATM LEAP",
                "itm_leap":   "ITM LEAP",
                "short_put":  "Short Put",
            }
            any_fetched = False
            for ck, cv in ev.get("contracts", {}).items():
                if not cv:
                    continue
                lbl = LABEL_MAP2.get(ck, ck)
                prem = cv.get("entry_premium")
                prem_str = f"${prem:.2f}" if prem is not None else "price not fetched"
                print(f"  {lbl:<16}  {cv['occ_ticker']}  @ {prem_str}")
                if prem is not None:
                    any_fetched = True
            print()
            print(f"  Stock entry ${ev['underlying_price']:,.2f}  |  current ${current_price:,.2f}"
                  f"  {fmt_ret(stock_log_ret_display)}  (as of {current_date})")
            print()
            if any_fetched:
                print("  Strategy returns will appear after the first  r  Reprice.")
            else:
                print("  No option prices fetched — run  r  Reprice to populate premiums.")
            print()

            raw = prompt("  Save evaluation? [Y/n/b]", "Y").strip().lower()
            if raw == "b":
                phase = "fetch_prompt"
                continue
            if raw not in ("y", "yes", ""):
                return False

            existing_ids = {e["eval_id"] for e in evals if e.get("eval_id")}
            n = 1
            while f"{pos['id']}_eval_{n}" in existing_ids:
                n += 1
            ev["eval_id"] = f"{pos['id']}_eval_{n}"
            evals.append(ev)
            save_evals(evals)
            print(f"  Saved {ev['eval_id']}.")
            return True


# ---------------------------------------------------------------------------
# Menu: Dashboard
# ---------------------------------------------------------------------------

def menu_dashboard(session: Session) -> None:
    positions = load_positions()
    evals     = load_evals()
    open_pos  = [p for p in positions if p.get("status") == "open"]
    closed_pos = sorted(
        [p for p in positions if p.get("status") == "closed"],
        key=lambda p: p.get("exit_date", ""),
        reverse=True,
    )[:6]

    print_header("DASHBOARD", session)
    _eval_stale_warning(evals)
    print(f"  Open: {len(open_pos)}  |  Closed: {len(positions)-len(open_pos)}")

    # Build eval index early so per-account detail can reference it inline
    open_eval_by_pos: dict[str, dict] = {}
    for ev in evals:
        if ev.get("status") != "open":
            continue
        pid = ev.get("position_id", "")
        if pid not in open_eval_by_pos or ev.get("eval_date", "") > open_eval_by_pos[pid].get("eval_date", ""):
            open_eval_by_pos[pid] = ev

    # ── Open positions: account summary then per-account breakdown ───────────
    if open_pos:
        # Compute analytics for all open positions once
        open_analytics = [(p, compute_analytics(p)) for p in open_pos]

        # Build per-account buckets preserving insertion order (account name → [(pos, a)])
        by_acct: dict[str, list[tuple[dict, dict]]] = {}
        for p, a in open_analytics:
            key = p.get("account_id", "?")
            by_acct.setdefault(key, []).append((p, a))

        # ── Account summary table ────────────────────────────────────────────
        print_section("Open Positions — by Account")
        print(f"  {'Account':<24} {'Pos':>4} {'Avg Ret':>9} {'vs SPY':>9}")
        print(f"  {hr('─', 50)}")
        for aid, items in by_acct.items():
            acct_name = items[0][0].get("account_name", aid)[:24]
            rets  = [a["stock_log_ret"] for _, a in items if a.get("stock_log_ret") is not None]
            excess = [a["excess_ret"]   for _, a in items if a.get("excess_ret")   is not None]
            avg_r  = sum(rets)   / len(rets)   if rets   else None
            avg_e  = sum(excess) / len(excess) if excess else None
            print(f"  {acct_name:<24} {len(items):>4} "
                  f"{fmt_ret(avg_r):>9} {fmt_ret(avg_e):>9}")

        # ── Per-account position detail ──────────────────────────────────────
        for aid, items in by_acct.items():
            acct_name = items[0][0].get("account_name", aid)
            print()
            print(f"  ── {acct_name} ──")
            print(f"  {'#':<3} {'Ticker':<6} {'Shrs':>5} {'Entry':>8} "
                  f"{'Current':>8} {'Return':>7} {'vs SPY':>7} {'Legs':>5}")
            print(f"  {hr('─', 55)}")
            for i, (pos, a) in enumerate(items, 1):
                curr  = a.get("current_price")
                legs  = sum(1 for l in pos.get("legs", []) if l.get("exit_date") is None)
                curr_str = f"${curr:,.2f}" if curr else "  —  "
                print(f"  {i:<3} {pos['ticker']:<6} {pos.get('shares',0):>5} "
                      f"${pos['entry_price']:>7,.2f} "
                      f"{curr_str:>8} "
                      f"{fmt_ret(a.get('stock_log_ret')):>7} "
                      f"{fmt_ret(a.get('excess_ret')):>7} {legs:>5}")
                ev = open_eval_by_pos.get(pos["id"])
                if ev:
                    strats = ev.get("strategies", {})
                    hdays  = ev.get("holding_days", 0)
                    price_dates = [c["price_date"] for c in ev.get("contracts", {}).values()
                                   if c and c.get("price_date")]
                    age_days = (date.today() - datetime.strptime(max(price_dates), "%Y-%m-%d").date()).days \
                               if price_dates else None
                    parts = []
                    for skey, slabel in [("covered_call", "CC"), ("leap_cc", "L+CC"),
                                         ("diagonal", "Diag"), ("csp", "CSP"), ("naked_leap", "LEAP")]:
                        s = strats.get(skey, {})
                        if s.get("active"):
                            parts.append(f"{slabel} {fmt_ret(s.get('log_ret'))}")
                    if parts:
                        suffix = f"  {hdays}d held"
                        if age_days is not None:
                            suffix += f"  · priced {age_days}d ago"
                        print(f"       ↳ {' | '.join(parts)}{suffix}")

    # ── Recently closed ──────────────────────────────────────────────────────
    if closed_pos:
        print_section("Recently Closed")
        print(f"  {'Ticker':<6} {'Account':<22} {'Exit':>11} "
              f"{'Return':>7} {'vs SPY':>7} {'Opts P&L':>10}")
        print(f"  {hr('─', 66)}")
        for pos in closed_pos:
            a = compute_analytics(pos)
            acct = pos.get("account_name", pos.get("account_id", "?"))[:22]
            opts  = a.get("options_pnl")
            print(f"  {pos['ticker']:<6} {acct:<22} "
                  f"{pos.get('exit_date','?'):>11} "
                  f"{fmt_ret(a.get('stock_log_ret')):>7} "
                  f"{fmt_ret(a.get('excess_ret')):>7} "
                  f"{fmt_dollars(opts, signed=True) if opts is not None else '—':>10}")

    if not open_pos and not closed_pos:
        print("\n  No trades recorded yet. Choose  1 — Enter trades  from the main menu.")

    # ── Strategy evaluation summary ──────────────────────────────────────────
    if open_eval_by_pos:
        print_section("Strategy Evaluations")
        print(f"  {'Ticker':<6} {'CC':>7} {'LEAP+CC':>9} {'Diag':>7} {'CSP':>7} {'LEAP':>7}  {'Days':>5} {'Age':>6}")
        print(f"  {hr('─', 64)}")
        pos_order = {pos["id"]: i for i, pos in enumerate(open_pos)}
        for pid, ev in sorted(open_eval_by_pos.items(),
                              key=lambda x: pos_order.get(x[0], 999)):
            ticker = ev.get("ticker", "?")
            strats = ev.get("strategies", {})
            hdays  = ev.get("holding_days", 0)
            price_dates = [
                c["price_date"] for c in ev.get("contracts", {}).values()
                if c and c.get("price_date")
            ]
            age_str = "?"
            if price_dates:
                age = (date.today() - datetime.strptime(max(price_dates), "%Y-%m-%d").date()).days
                age_str = f"{age}d"

            def _fmt_s(key: str) -> str:
                s = strats.get(key, {})
                if not s.get("active"):
                    return "  —  "
                return fmt_ret(s.get("log_ret"))

            print(f"  {ticker:<6} "
                  f"{_fmt_s('covered_call'):>7} "
                  f"{_fmt_s('leap_cc'):>9} "
                  f"{_fmt_s('diagonal'):>7} "
                  f"{_fmt_s('csp'):>7} "
                  f"{_fmt_s('naked_leap'):>7}  {hdays:>5} {age_str:>6}")

    sep()
    prompt("Press Enter to continue")


# ---------------------------------------------------------------------------
# Menu: Enter trades  (the main fast-entry loop)
# ---------------------------------------------------------------------------

def _handle_ticker(ticker: str, session: Session,
                   positions: list[dict], strategies: list[dict]) -> None:
    """Handle one ticker in the enter-trades loop."""
    account_id = session.account["id"]

    # Find open positions for this ticker in the current account
    account_pos = [p for p in positions
                   if p["ticker"] == ticker
                   and p.get("account_id") == account_id
                   and p.get("status") == "open"]

    if account_pos:
        print()
        print(f"  {ticker} — {len(account_pos)} open position(s) in {session.account_name}:")
        for i, pos in enumerate(account_pos, 1):
            open_legs = [l for l in pos.get("legs", []) if l.get("exit_date") is None]
            legs_str  = f"  {len(open_legs)} open leg(s)" if open_legs else "  no open legs"
            print(f"    {i}.  entry {pos['entry_date']} @ ${pos['entry_price']:,.2f}"
                  f"  x{pos.get('shares',0)} shares{legs_str}")
        print()
        print("    l  Add leg to a position above")
        print("    n  New position")
        print("    v  View position detail")
        print("    b  Back (different ticker)")
        print()
        action = prompt("Action", "l").lower()

        if action == "b":
            return
        elif action == "v":
            pos_idx = 0
            if len(account_pos) > 1:
                n = prompt_int("Position number", 1) or 1
                pos_idx = n - 1
            show_position_detail(account_pos[pos_idx])
            sep()
            prompt("Press Enter to continue")
        elif action == "l":
            pos_idx = 0
            if len(account_pos) > 1:
                n = prompt_int("Position number", 1) or 1
                pos_idx = (n - 1)
            flow_add_leg(account_pos[pos_idx], strategies, positions, session.trade_date)
        elif action == "n":
            _add_new_position(ticker, session, positions, strategies)
    else:
        print(f"\n  {ticker} — no open position in {session.account_name}.")
        print()
        print("    n  New position")
        print("    b  Back (different ticker)")
        print()
        action = prompt("Action", "n").lower()
        if action == "n":
            _add_new_position(ticker, session, positions, strategies)


def _add_new_position(ticker: str, session: Session,
                      positions: list[dict], strategies: list[dict]) -> None:
    account = session.account
    print()
    entry_date  = prompt_date("Entry date", session.trade_date)
    
    # Fetch the historical price based on the ticker and entry date
    proposed_price = get_price(ticker, entry_date)
    
    # Pass the proposed price as the default value
    entry_price = prompt_float("Entry price", proposed_price)


    if entry_price is None:
        print("  Cancelled.")
        return
    shares = prompt_int("Shares")
    if shares is None:
        print("  Cancelled.")
        return

    pos_id   = f"{ticker}_{account['id']}_{entry_date}"
    existing = {p["id"] for p in positions}
    if pos_id in existing:
        n = 2
        while f"{pos_id}_{n}" in existing:
            n += 1
        pos_id = f"{pos_id}_{n}"

    pos: dict = {
        "id": pos_id,
        "account_id":   account["id"],
        "account_name": account["name"],
        "ticker":       ticker,
        "entry_date":   entry_date,
        "entry_price":  entry_price,
        "shares":       shares,
        "status":       "open",
        "exit_date":    None,
        "exit_price":   None,
        "legs":         [],
    }
    positions.append(pos)
    save_positions(positions)
    print(f"  Saved position: {pos_id}")

    if prompt("Add an options leg now? [y/N]", "N").lower() == "y":
        flow_add_leg(pos, strategies, positions, session.trade_date)

    if prompt("Option strategies? [y/N]", "N").lower() == "y":
        evals = load_evals()
        flow_evaluate_strategies(pos, evals)


def menu_enter_trades(session: Session) -> None:
    if not session.account:
        print("\n  No account selected. Switch account first (s from main menu).")
        prompt("Press Enter to continue")
        return

    strategies = load_strategies()

    while True:
        positions = load_positions()  # reload each loop in case of saves
        print_header("ENTER TRADES", session)
        print()
        print("  Enter a ticker to add a position or leg.")
        print("  Press Enter (blank) to return to the main menu.")
        print()

        ticker = prompt("Ticker").upper()
        if not ticker:
            break

        _handle_ticker(ticker, session, positions, strategies)


# ---------------------------------------------------------------------------
# Menu: Close leg / Close position
# ---------------------------------------------------------------------------

def menu_close_leg(session: Session) -> None:
    print_header("CLOSE A LEG", session)
    positions = load_positions()

    # Ask: current account only, or all?
    scope = "all"
    if session.account:
        raw = prompt(f"Current account only ({session.account_name})? [Y/n]", "Y")
        scope = "account" if raw.lower() in ("y", "yes", "") else "all"

    aid = session.account["id"] if scope == "account" and session.account else None
    flow_close_leg(positions, filter_account_id=aid)
    sep()
    prompt("Press Enter to continue")


def menu_close_position(session: Session) -> None:
    print_header("CLOSE A POSITION", session)
    positions = load_positions()

    scope = "all"
    if session.account:
        raw = prompt(f"Current account only ({session.account_name})? [Y/n]", "Y")
        scope = "account" if raw.lower() in ("y", "yes", "") else "all"

    aid = session.account["id"] if scope == "account" and session.account else None
    flow_close_position(positions, filter_account_id=aid)
    sep()
    prompt("Press Enter to continue")


# ---------------------------------------------------------------------------
# Menu: Evaluate strategies
# ---------------------------------------------------------------------------

def menu_evaluate_strategies(session: Session) -> None:
    print_header("OPTION STRATEGIES", session)
    positions = load_positions()
    open_pos  = [p for p in positions if p.get("status") == "open"]

    if not open_pos:
        print("  No open positions.")
        prompt("Press Enter to continue")
        return

    evals = load_evals()

    # Split into unevaluated (no open eval) and already evaluated
    evaluated_ids = {ev["position_id"] for ev in evals if ev.get("status") == "open"}
    unevaluated = [p for p in open_pos if p["id"] not in evaluated_ids]
    already_done = [p for p in open_pos if p["id"] in evaluated_ids]

    if unevaluated:
        print(f"  {len(unevaluated)} position(s) without an evaluation"
              + (f"  |  {len(already_done)} already evaluated" if already_done else ""))
    else:
        print(f"  All {len(open_pos)} open position(s) already have evaluations.")

    print()

    # Offer the unevaluated queue first, then let user pick any if they want
    labels = [
        f"{p['ticker']}  entry {p['entry_date']} @ ${p['entry_price']:,.2f}"
        f"  x{p.get('shares',0)}  —  {p.get('account_name', p.get('account_id','?'))}"
        + ("" if p["id"] not in evaluated_ids else "  ✓")
        for p in open_pos
    ]

    if unevaluated:
        # Queue mode: work through unevaluated positions one by one
        queue = list(unevaluated)
        while queue:
            pos = queue.pop(0)
            print_header(f"OPTION STRATEGIES — {pos['ticker']}  ({queue[0]['ticker']} next)" if queue
                         else f"OPTION STRATEGIES — {pos['ticker']}  (last one)", session)
            evals = load_evals()  # reload in case a prior save added entries
            flow_evaluate_strategies(pos, evals, next_ticker=queue[0]["ticker"] if queue else None)
            if queue:
                sep()
                if prompt(f"Continue to {queue[0]['ticker']}? [Y/n]", "Y").lower() not in ("y", "yes", ""):
                    break
    else:
        # All evaluated — let user pick one to re-evaluate
        print("  Pick a position to re-evaluate:")
        idx = pick_from_list(labels, "Position")
        if idx is None:
            return
        evals = load_evals()
        flow_evaluate_strategies(open_pos[idx], evals)

    sep()
    prompt("Press Enter to continue")


# ---------------------------------------------------------------------------
# Menu: Reprice options
# ---------------------------------------------------------------------------

def menu_reprice_options(session: Session) -> None:
    print_header("REPRICE OPTIONS", session)
    evals = load_evals()
    open_evals = [ev for ev in evals if ev.get("status") == "open"]

    if not open_evals:
        print("  No open strategy evaluations to reprice.")
        prompt("Press Enter to continue")
        return

    # Collect unique OCC tickers across all open evals, with label from first occurrence
    ticker_label: dict[str, str] = {}
    ticker_refs: dict[str, list[tuple[int, str]]] = {}
    for ev_idx, ev in enumerate(open_evals):
        for contract_key, contract in ev.get("contracts", {}).items():
            if not contract:
                continue
            occ = contract.get("occ_ticker")
            if not occ:
                continue
            if occ not in ticker_label:
                label_map = {
                    "short_call": "Short Call", "atm_leap": "ATM LEAP",
                    "itm_leap": "ITM LEAP", "short_put": "Short Put",
                }
                ticker_label[occ] = f"{ev.get('ticker','?')} {label_map.get(contract_key, contract_key)}"
            ticker_refs.setdefault(occ, []).append((ev_idx, contract_key))

    if not ticker_refs:
        print("  No options contracts found in open evaluations.")
        prompt("Press Enter to continue")
        return

    print(f"  {len(ticker_refs)} unique contract(s) across {len(open_evals)} open eval(s).")
    if not confirm("Fetch current prices from Polygon?"):
        sep()
        prompt("Press Enter to continue")
        return

    to_fetch = [(ticker_label[occ], occ) for occ in sorted(ticker_refs)]
    print()
    prices = _fetch_contracts_batch(to_fetch, TODAY)
    print()

    # Write prices back to all referencing evals
    updated = 0
    for occ, price in prices.items():
        if price is None:
            continue
        for ev_idx, contract_key in ticker_refs[occ]:
            ev = open_evals[ev_idx]
            contract = ev["contracts"][contract_key]
            if contract.get("entry_premium") is None:
                contract["entry_premium"] = price
            contract["current_premium"] = price
            contract["price_date"] = TODAY
            updated += 1

    # Recompute analytics for each affected eval
    positions = load_positions()
    pos_map = {p["id"]: p for p in positions}
    for ev in open_evals:
        pos = pos_map.get(ev.get("position_id", ""))
        current_stock_price = None
        if pos:
            info = get_latest_price(pos["ticker"])
            if info:
                current_stock_price = info[1]
        _compute_eval_analytics(ev, current_stock_price)

    save_evals(evals)
    print(f"  Updated {updated} contract price(s). Analytics recomputed.")
    sep()
    prompt("Press Enter to continue")


# ---------------------------------------------------------------------------
# Menu: Report
# ---------------------------------------------------------------------------

def menu_report(session: Session) -> None:
    print_header("REPORT", session)
    print()
    print("  Filter options (press Enter to include all):")
    filter_account  = prompt("Account ID filter (Enter = all)")
    filter_strategy = prompt("Strategy ID filter (Enter = all)")

    positions  = load_positions()
    strategies = load_strategies()

    if filter_account:
        positions = [p for p in positions if p.get("account_id") == filter_account]

    analytics = [(pos, compute_analytics(pos)) for pos in positions]
    closed_a  = [(p, a) for p, a in analytics if p.get("status") == "closed"]
    open_a    = [(p, a) for p, a in analytics if p.get("status") == "open"]

    # ── Stock performance ────────────────────────────────────────────────────
    print_section("STOCK PERFORMANCE")

    closed_ret = [(p, a) for p, a in closed_a if a.get("stock_log_ret") is not None]
    if closed_ret:
        rets    = [a["stock_log_ret"] for _, a in closed_ret]
        spy_r   = [a["spy_log_ret"]   for _, a in closed_ret if a.get("spy_log_ret") is not None]
        excess  = [a["excess_ret"]    for _, a in closed_ret if a.get("excess_ret")  is not None]
        days    = [a["holding_days"]  for _, a in closed_ret if a.get("holding_days")]
        win_r   = sum(1 for r in rets if r > 0) / len(rets)
        spy_b   = sum(1 for e in excess if e > 0) / len(excess) if excess else None

        print(f"  Closed positions:   {len(closed_ret)}")
        print(f"  Avg log return:     {fmt_ret(sum(rets)/len(rets))}")
        print(f"  Avg SPY (same wdw): {fmt_ret(sum(spy_r)/len(spy_r)) if spy_r else '—'}")
        print(f"  Avg excess vs SPY:  {fmt_ret(sum(excess)/len(excess)) if excess else '—'}")
        print(f"  Win rate (>0):      {win_r:.0%}")
        if spy_b is not None:
            print(f"  Beat SPY rate:      {spy_b:.0%}")
        if days:
            print(f"  Avg days held:      {sum(days)/len(days):.0f}")
    else:
        print("  No closed positions with return data yet.")

    open_ret = [(p, a) for p, a in open_a if a.get("stock_log_ret") is not None]
    if open_ret:
        avg_o   = sum(a["stock_log_ret"] for _, a in open_ret) / len(open_ret)
        excess_o = [a["excess_ret"] for _, a in open_ret if a.get("excess_ret") is not None]
        print(f"\n  Open positions ({len(open_ret)}) — unrealized:")
        print(f"  Avg log return:     {fmt_ret(avg_o)}")
        if excess_o:
            print(f"  Avg excess vs SPY:  {fmt_ret(sum(excess_o)/len(excess_o))}")

    # ── Options by strategy ──────────────────────────────────────────────────
    print_section("OPTIONS BY STRATEGY")

    strat_legs: dict[str, list[dict]] = {}
    for pos, _ in analytics:
        sv = pos["entry_price"] * pos.get("shares", 0)
        for leg in pos.get("legs", []):
            sid = leg.get("strategy_id", "unknown")
            if filter_strategy and sid != filter_strategy:
                continue
            strat_legs.setdefault(sid, []).append({**leg, "_sv": sv})

    if not strat_legs:
        print("  No options legs recorded yet.")
    else:
        strat_map = {s["id"]: s["name"] for s in strategies}
        for sid, legs in sorted(strat_legs.items()):
            name    = strat_map.get(sid, sid)
            closed  = [l for l in legs if l.get("exit_date") is not None]
            open_l  = [l for l in legs if l.get("exit_date") is None]
            print(f"\n  {name}  ({sid})")
            print(f"    Total: {len(legs)}  |  Closed: {len(closed)}  |  Open: {len(open_l)}")

            if closed:
                pnls   = [p for p in (_leg_pnl(l) for l in closed) if p is not None]
                wins   = [p > 0 for p in pnls]
                yields = [p / l["_sv"] for p, l in zip(pnls, closed) if l.get("_sv", 0) > 0]
                dtes: list[int] = []
                for leg in closed:
                    try:
                        d0 = datetime.strptime(leg["entry_date"], "%Y-%m-%d").date()
                        d1 = datetime.strptime(leg["exit_date"],  "%Y-%m-%d").date()
                        dtes.append((d1 - d0).days)
                    except (ValueError, KeyError):
                        pass
                reasons: dict[str, int] = {}
                for leg in closed:
                    r = leg.get("exit_reason", "unknown")
                    reasons[r] = reasons.get(r, 0) + 1

                if wins:
                    print(f"    Win rate:      {sum(wins)/len(wins):.0%}")
                print(f"    Total P&L:     {fmt_dollars(sum(pnls), signed=True)}")
                print(f"    Avg P&L/leg:   {fmt_dollars(sum(pnls)/len(pnls), signed=True)}")
                if yields:
                    print(f"    Avg yield:     {fmt_ret(sum(yields)/len(yields))}")
                if dtes:
                    print(f"    Avg days held: {sum(dtes)/len(dtes):.0f}")
                reasons_str = "  ".join(f"{r}: {c}" for r, c in sorted(reasons.items()))
                print(f"    Outcomes:      {reasons_str}")

            if open_l:
                parts = [f"[{l['leg_id']}] {l['leg_type']} ${l['strike']} exp {l['expiry']}"
                         for l in open_l]
                print(f"    Open:          {',  '.join(parts)}")

    # ── By account ───────────────────────────────────────────────────────────
    print_section("BY ACCOUNT")

    by_acct: dict[str, list[tuple[dict, dict]]] = {}
    for pos, a in analytics:
        by_acct.setdefault(pos.get("account_id", "?"), []).append((pos, a))

    for aid_key, items in sorted(by_acct.items()):
        acct_name   = items[0][0].get("account_name", aid_key)
        open_count  = sum(1 for p, _ in items if p.get("status") == "open")
        closed_count = sum(1 for p, _ in items if p.get("status") == "closed")
        c_rets  = [a["stock_log_ret"] for p, a in items if p.get("status") == "closed" and a.get("stock_log_ret") is not None]
        c_exc   = [a["excess_ret"]    for p, a in items if p.get("status") == "closed" and a.get("excess_ret")  is not None]
        o_rets  = [a["stock_log_ret"] for p, a in items if p.get("status") == "open"   and a.get("stock_log_ret") is not None]
        opts_pnl = sum(a.get("options_pnl") or 0 for _, a in items if a.get("options_pnl") is not None)
        has_opts = any(a.get("options_pnl") is not None for _, a in items)

        print(f"\n  {acct_name}  ({aid_key})")
        print(f"    Open: {open_count}  Closed: {closed_count}")
        if c_rets:
            print(f"    Avg closed return:   {fmt_ret(sum(c_rets)/len(c_rets))}"
                  + (f"  vs SPY: {fmt_ret(sum(c_exc)/len(c_exc))}" if c_exc else ""))
        if o_rets:
            print(f"    Avg open return:     {fmt_ret(sum(o_rets)/len(o_rets))}  (unrealized)")
        if has_opts:
            print(f"    Total options P&L:   {fmt_dollars(opts_pnl, signed=True)}")

    # ── Strategy evaluations ─────────────────────────────────────────────────
    print_section("STRATEGY EVALUATIONS")

    all_evals = load_evals()
    if not all_evals:
        print("  No strategy evaluations recorded yet. Use  e  from the main menu.")
    else:
        # Aggregate per strategy type + stock across all evals (open + closed)
        _agg_keys = ["stock"] + EVAL_STRATEGY_TYPES
        agg: dict[str, dict[str, list[float]]] = {
            k: {"log_ret": [], "excess_ret": [], "annualized_ret": []}
            for k in _agg_keys
        }
        for ev in all_evals:
            # Stock row
            for field in ("log_ret", "excess_ret", "annualized_ret"):
                v = ev.get(f"stock_{field}")
                if v is not None:
                    agg["stock"][field].append(v)
            # Strategy rows
            for stype, sdata in ev.get("strategies", {}).items():
                if not sdata.get("active") or stype not in agg:
                    continue
                for field in ("log_ret", "excess_ret", "annualized_ret"):
                    v = sdata.get(field)
                    if v is not None:
                        agg[stype][field].append(v)

        has_any = any(agg[k]["log_ret"] for k in _agg_keys)
        if has_any:
            print(f"  {'Strategy':<22} {'N':>4} {'Avg Ret':>9} {'vs SPY':>9} {'Ann.':>9}")
            print(f"  {hr('─', 58)}")
            row_labels = {"stock": "Stock (underlying)", **EVAL_STRATEGY_NAMES}
            for key in _agg_keys:
                name  = row_labels[key]
                rets  = agg[key]["log_ret"]
                if not rets:
                    print(f"  {name:<22} {'—':>4}")
                    continue
                avg_r = sum(rets) / len(rets)
                exc   = agg[key]["excess_ret"]
                avg_e = sum(exc) / len(exc) if exc else None
                ann   = agg[key]["annualized_ret"]
                avg_a = sum(ann) / len(ann) if ann else None
                print(f"  {name:<22} {len(rets):>4} "
                      f"{fmt_ret(avg_r):>9} {fmt_ret(avg_e):>9} {fmt_ret(avg_a):>9}")

        # Per-position latest eval detail (vertical block per position)
        pos_map_recent: dict[str, dict] = {}
        for ev in all_evals:
            pid = ev.get("position_id", "")
            if pid not in pos_map_recent or ev.get("eval_date", "") > pos_map_recent[pid].get("eval_date", ""):
                pos_map_recent[pid] = ev

        if pos_map_recent:
            print()
            print(f"  Latest eval per position:")
            col_hdr = f"    {'Strategy':<22} {'Ret':>7} {'vs SPY':>9} {'Ann.':>9}"
            col_div = f"    {hr('─', 49)}"
            for pid, ev in sorted(pos_map_recent.items()):
                ticker  = ev.get("ticker", "?")
                edate   = ev.get("eval_date", "?")
                hdays   = ev.get("holding_days", 0)
                strats  = ev.get("strategies", {})
                print()
                print(f"  {ticker}  (eval {edate}  ·  {hdays}d held)")
                print(col_hdr)
                print(col_div)
                # Stock row
                print(f"    {'Stock':<22} "
                      f"{fmt_ret(ev.get('stock_log_ret')):>7} "
                      f"{fmt_ret(ev.get('stock_excess_ret')):>9} "
                      f"{fmt_ret(ev.get('stock_annualized_ret')):>9}")
                # Strategy rows
                for stype in EVAL_STRATEGY_TYPES:
                    name  = EVAL_STRATEGY_NAMES[stype]
                    sdata = strats.get(stype, {})
                    if not sdata.get("active"):
                        print(f"    {name:<22}  —")
                        continue
                    print(f"    {name:<22} "
                          f"{fmt_ret(sdata.get('log_ret')):>7} "
                          f"{fmt_ret(sdata.get('excess_ret')):>9} "
                          f"{fmt_ret(sdata.get('annualized_ret')):>9}")

    sep()
    prompt("Press Enter to continue")


# ---------------------------------------------------------------------------
# Menu: Accounts
# ---------------------------------------------------------------------------

def menu_accounts(session: Session) -> None:
    while True:
        accounts = load_accounts()
        print_header("ACCOUNTS", session)
        print()

        if accounts:
            for i, a in enumerate(accounts, 1):
                current = " ◀ current" if session.account and session.account["id"] == a["id"] else ""
                print(f"  {i}.  {a['name']:<30} {a.get('type',''):<10} [{a['id']}]{current}")
        else:
            print("  No accounts yet.")

        print()
        print("  a  Add account")
        print("  b  Back")
        print()
        choice = prompt("Choice", "b").lower()

        if choice == "b":
            break
        elif choice == "a":
            print()
            name = prompt("Account name (e.g. Fidelity Taxable)")
            if not name:
                continue
            acct_type = prompt_choice("Type", ["taxable", "ira", "roth", "hsa", "other"])
            acct_id   = name.lower().replace(" ", "_")
            existing  = {a["id"] for a in accounts}
            if acct_id in existing:
                n = 2
                while f"{acct_id}_{n}" in existing:
                    n += 1
                acct_id = f"{acct_id}_{n}"
            accounts.append({"id": acct_id, "name": name, "type": acct_type})
            save_accounts(accounts)
            print(f"  Saved: {acct_id}")


# ---------------------------------------------------------------------------
# Menu: Strategies
# ---------------------------------------------------------------------------

def menu_strategies(session: Session) -> None:
    while True:
        strategies = load_strategies()
        print_header("STRATEGIES", session)
        print()

        if strategies:
            for i, s in enumerate(strategies, 1):
                active_tag = "" if s.get("active", True) else "  [inactive]"
                note_count = len(s.get("notes", []))
                print(f"  {i}.  {s['name']:<28} {note_count} note(s){active_tag}")
        else:
            print("  No strategies yet.")

        print()
        print("  a   Add strategy")
        if strategies:
            print("  #   View / add note to strategy  (e.g. '1')")
        print("  b   Back")
        print()
        choice = prompt("Choice", "b").lower()

        if choice == "b":
            break
        elif choice == "a":
            print()
            name = prompt("Strategy name (e.g. Covered Call)")
            if not name:
                continue
            sid = name.lower().replace(" ", "_")
            if any(s["id"] == sid for s in strategies):
                print(f"  '{sid}' already exists.")
                continue
            approach = prompt("Approach / how you think about it (Enter to skip)")
            strategies.append({
                "id": sid, "name": name, "active": True,
                "approach": approach, "notes": [], "created": TODAY,
            })
            save_strategies(strategies)
            print(f"  Saved: {sid}")
        else:
            # Treat as a number — show strategy detail + note options
            try:
                idx = int(choice) - 1
                if not (0 <= idx < len(strategies)):
                    raise ValueError
            except ValueError:
                print("  Unknown choice.")
                continue
            _strategy_detail_menu(strategies, idx)


def _strategy_detail_menu(strategies: list[dict], idx: int) -> None:
    while True:
        s = strategies[idx]
        print()
        print(hr("═"))
        print(f"  {s['name'].upper()}  [{s['id']}]")
        print(hr("═"))

        if s.get("approach"):
            print()
            print("  Approach:")
            for line in _wrap(s["approach"]):
                print(line)

        notes = sorted(s.get("notes", []), key=lambda n: n["date"])
        if notes:
            print()
            print(f"  Notes ({len(notes)}):")
            for n in notes:
                print(f"\n  {n['date']}")
                for line in _wrap(n["text"]):
                    print(line)
        else:
            print("\n  No notes yet.")

        print()
        print("  n  Add note")
        print("  e  Edit approach")
        print("  b  Back")
        print()
        choice = prompt("Choice", "b").lower()

        if choice == "b":
            break
        elif choice == "n":
            print()
            text = prompt("Note")
            if not text:
                continue
            note_date = prompt_date("Date", TODAY)
            s.setdefault("notes", []).append({"date": note_date, "text": text})
            save_strategies(strategies)
            print("  Note saved.")
        elif choice == "e":
            print()
            print("  Current approach:")
            print(f"    {s.get('approach', '(none)')}")
            approach = prompt("New approach (Enter to keep)")
            if approach:
                s["approach"] = approach
                save_strategies(strategies)
                print("  Updated.")


# ---------------------------------------------------------------------------
# Menu: Help / documentation
# ---------------------------------------------------------------------------

def menu_help(session: Session) -> None:
    print_header("HELP — HOW TO USE TRADES", session)

    sections = [
        ("OVERVIEW", [
            "trades.py is a personal ledger for stock positions and options overlays.",
            "It is completely separate from the marketview automated pipeline — this",
            "is your manual accounting layer on top of whatever you actually hold.",
            "",
            "Everything is organized around three concepts:",
            "  Accounts  — your brokerage accounts (Fidelity Taxable, Schwab IRA, etc.)",
            "  Strategies — named approaches with your own playbook notes (Covered Call, etc.)",
            "  Positions  — a stock you hold, with zero or more options legs attached.",
        ]),
        ("SESSION STATE", [
            "The top of every screen shows your current Account and Date. These persist",
            "across entries within a session so you don't re-enter them for every trade.",
            "",
            "  s  — Switch to a different account at any time.",
            "  d  — Change the session date (useful for entering past trades).",
            "       Defaults to today. All new position and leg entry dates default",
            "       to this value — you only override if a specific trade differs.",
        ]),
        ("ENTERING TRADES  (menu option 1)", [
            "This is the main fast-entry loop. Type a ticker, and the tool shows",
            "what's already open in your current account:",
            "",
            "  • If an open position exists: choose  l (add leg)  or  n (new position).",
            "  • If no position: goes straight to new position entry.",
            "",
            "Within a session you can enter many tickers back-to-back. Press Enter",
            "(blank ticker) to return to the main menu when you're done.",
            "",
            "When adding an options leg, your strategy's approach and most recent note",
            "are shown as a reminder before you enter the details.",
        ]),
        ("CLOSING TRADES", [
            "  2 — Close a leg: shows all open options legs across your positions.",
            "       You record the exit premium and reason (expired, bought_back_profit,",
            "       assigned, bought_back_loss, rolled). P&L is computed immediately.",
            "",
            "  3 — Close a position: records the stock exit price and date.",
            "       Open legs are not auto-closed — close them separately (option 2)",
            "       if they were part of the exit.",
            "",
            "  Both default to the current account but you can scope to all accounts.",
        ]),
        ("PAST TRADES", [
            "You can enter trades from any date — there is no restriction to today.",
            "Change the session date (d) to the trade date before entering, or simply",
            "override the date field for each individual entry. Useful for backfilling",
            "a history of covered calls or CSPs from your brokerage statements.",
        ]),
        ("STRATEGIES & PLAYBOOK  (menu option t)", [
            "Strategies are your named approaches. Each has:",
            "  Approach — a free-text description of how you think about the strategy.",
            "  Notes    — timestamped journal entries, shown newest-first.",
            "",
            "When you add a leg (option 1 → l), the approach and latest note are",
            "displayed as a reminder before you enter the trade details. This keeps",
            "your own rules in front of you at the moment of entry.",
            "",
            "Add a note any time your thinking evolves (t → pick strategy → n).",
            "Over time this becomes a searchable record of how your approach changed.",
        ]),
        ("REPORT  (menu option 4)", [
            "The report aggregates across all closed positions and options legs.",
            "You can filter by account ID or strategy ID at the prompt.",
            "",
            "Stock performance section: log returns, SPY comparison, win rates.",
            "Options by strategy: win rate, total P&L, avg yield per leg, avg days held,",
            "  and an outcome breakdown (expired / bought_back / assigned / rolled).",
            "By account: closed and open return summary per brokerage account.",
        ]),
        ("STRATEGY EVALUATIONS  (menu options e and r)", [
            "Option  e — Option strategies  runs on any open position.",
            "You enter up to 4 contract specs (strike + expiry for each):",
            "  Short call    → used by Covered Call, LEAP+CC, Diagonal",
            "  ATM/OTM LEAP  → used by LEAP+CC, Naked LEAP",
            "  ITM LEAP      → used by Diagonal only",
            "  Short put     → used by CSP only",
            "",
            "Prices are fetched from Polygon in a batch at the end",
            f"(free tier: 5 calls/min — program waits {POLYGON_RATE_SLEEP}s between fetches).",
            "Results are stored in trades/strategy_evals.json.",
            "",
            "Option  r — Reprice options  refreshes current prices for all",
            "open evaluations in one batch. A stale warning appears on the",
            f"main menu and dashboard if prices are more than {EVAL_STALE_DAYS} days old.",
            "(The warning also fires if underlying stock prices are stale.)",
            "",
            "Blank strike = skip that contract (strategies needing it show '—').",
            "You can re-run  e  on a position at any time to create a new eval.",
        ]),
        ("DATA FILES", [
            "All data lives in trades/ (four JSON files):",
            "  trades/accounts.json       — account definitions",
            "  trades/strategies.json     — strategy playbook",
            "  trades/positions.json      — positions with legs embedded",
            "  trades/strategy_evals.json — strategy evaluations",
            "",
            "Files are written atomically (write to .tmp, then rename) so a crash",
            "during a save never corrupts existing data.",
            "",
            "For calculation details including the log return formulas, choose",
            "  i — Data & calculations  from the main menu.",
        ]),
    ]

    for title, lines in sections:
        print()
        print(f"  {title}")
        print(f"  {hr('─', 66)}")
        for line in lines:
            print() if line == "" else print(f"  {line}")

    sep()
    prompt("Press Enter to return to main menu")


# ---------------------------------------------------------------------------
# Menu: Data & calculations
# ---------------------------------------------------------------------------

def menu_data_info(session: Session) -> None:
    print_header("DATA & CALCULATIONS", session)

    sections = [
        ("PRICE DATA SOURCE", [
            "Stock prices are read from the marketview price cache:",
            "  data.nosync/prices/YYYY-MM-DD.json",
            "",
            "Each file is a list of daily OHLCV bars for all universe tickers",
            "(S&P 500 + S&P 400, ~900 tickers). SPY is included.",
            "",
            "Price lookup: for a given date, the tool searches on or before that",
            "date for the nearest available trading day. This handles weekends,",
            "holidays, and dates before the price history begins gracefully.",
            "",
            "Open positions are marked to market using the most recent price file.",
            "The date of that mark is shown in position detail view.",
        ]),
        ("LOG RETURN — STOCK", [
            "All stock returns are log returns (continuously compounded):",
            "",
            "  log_return = ln( exit_price / entry_price )",
            "",
            "Log returns are used instead of simple returns because they are:",
            "  • Additive across time  (ln(P2/P0) = ln(P2/P1) + ln(P1/P0))",
            "  • Symmetric  (a +50% gain and a -33% loss are equal in magnitude)",
            "  • Consistent with the rest of the marketview pipeline",
            "",
            "To convert to a simple percentage return for intuition:",
            "  simple_return = exp(log_return) - 1",
            "",
            "The report and dashboard display log returns directly, labeled as",
            "'Return'. A value of +5.1% means a log return of 0.051.",
        ]),
        ("LOG RETURN — SPY BENCHMARK", [
            "For each position, SPY's log return is computed over the identical",
            "holding window:",
            "",
            "  spy_log_return = ln( SPY_price_at_exit / SPY_price_at_entry )",
            "",
            "  excess_return  = stock_log_return - spy_log_return",
            "",
            "For open positions, 'exit' is the latest available SPY price.",
            "A positive excess return means the stock outperformed SPY over",
            "the same calendar period you held it.",
            "",
            "Averages in the report are arithmetic means of log returns across",
            "positions — this is appropriate for cross-sectional comparison.",
        ]),
        ("OPTIONS P&L", [
            "Options P&L is computed in dollars per leg:",
            "",
            "  Short leg (sold premium):  P&L = (entry_prem - exit_prem) × contracts × 100",
            "  Long leg  (bought premium): P&L = (exit_prem - entry_prem) × contracts × 100",
            "",
            "Examples:",
            "  Sold a covered call for $2.30, bought back at $0.80:",
            "    P&L = (2.30 - 0.80) × 1 × 100 = +$150",
            "",
            "  Sold a CSP for $1.50, expired worthless:",
            "    P&L = (1.50 - 0.00) × 1 × 100 = +$150",
            "",
            "  Sold a call for $2.00, bought back at $3.50 (loss):",
            "    P&L = (2.00 - 3.50) × 1 × 100 = -$150",
            "",
            "Only closed legs are included in P&L — open legs are excluded",
            "because their current value requires live options pricing data.",
        ]),
        ("OPTIONS YIELD", [
            "Options yield is the dollar P&L expressed as a fraction of the",
            "stock position value at entry:",
            "",
            "  options_yield = options_P&L / (entry_price × shares)",
            "",
            "This is a simple ratio, not a log return. It is added to the stock",
            "log return to produce 'Total (stk+opts)' — an approximation that",
            "is accurate when yields are small (< ~5%).",
            "",
            "The report shows avg yield per closed leg across all legs of a",
            "strategy. Annualized yield is not computed automatically — divide",
            "by avg days held and multiply by 365 to approximate.",
        ]),
        ("STRATEGY EVALUATION ANALYTICS", [
            "Prices come from Polygon's aggregate bars endpoint using OCC format:",
            "  O:{TICKER}{YYMMDD}{C|P}{STRIKE×1000:08d}",
            "  Example: AAPL 195C 2026-05-16 → O:AAPL260516C00195000",
            "",
            "On creation, entry_premium = current_premium = fetched price (yield = 0).",
            "As you reprice, current_premium moves and returns accrue.",
            "",
            "Covered Call:",
            "  option_yield = (entry_prem − current_prem) / underlying_price",
            "  total_return = stock_log_ret + option_yield",
            "",
            "Naked LEAP / LEAP legs:",
            "  leap_log_ret = ln(current_prem / entry_prem)",
            "",
            "LEAP + Covered Call:",
            "  total_return = leap_log_ret(ATM LEAP) + option_yield(short call)",
            "",
            "Diagonal Spread:",
            "  total_return = leap_log_ret(ITM LEAP) + option_yield(short call)",
            "",
            "Cash Secured Put:",
            "  option_yield = (entry_prem − current_prem) / strike",
            "  (denominator = strike because cash reserved = strike × 100)",
        ]),
        ("WHAT IS NOT TRACKED", [
            "The following are intentionally out of scope:",
            "  • Open options leg mark-to-market  (requires live options quotes)",
            "  • Dividends  (not reflected in entry/exit prices)",
            "  • Commissions and fees  (enter net prices if you want them included)",
            "  • Tax lot accounting  (multiple positions in same account/ticker",
            "    are tracked as separate records, not FIFO/LIFO lots)",
            "  • Options assignment partial fills or complex multi-leg orders",
            "    (model as individual legs entered separately)",
        ]),
        ("DATA FILE SCHEMA", [
            "positions.json — one record per stock position:",
            "  id, account_id, account_name, ticker",
            "  entry_date, entry_price, shares",
            "  exit_date, exit_price  (null if open)",
            "  status: 'open' | 'closed'",
            "  legs: list of options leg records",
            "",
            "Each leg:",
            "  leg_id, strategy_id, leg_type (short_call / short_put / long_call / long_put)",
            "  strike, expiry, contracts",
            "  entry_date, entry_premium, entry_delta, entry_iv",
            "  exit_date, exit_premium, exit_reason",
            "  exit_reason: expired | assigned | bought_back_profit |",
            "               bought_back_loss | rolled",
        ]),
    ]

    for title, lines in sections:
        print()
        print(f"  {title}")
        print(f"  {hr('─', 66)}")
        for line in lines:
            print() if line == "" else print(f"  {line}")

    sep()
    prompt("Press Enter to return to main menu")


# ---------------------------------------------------------------------------
# Switch account
# ---------------------------------------------------------------------------

def switch_account(session: Session) -> None:
    accounts = load_accounts()
    if not accounts:
        print("\n  No accounts yet. Add one from the Accounts menu (a).")
        prompt("Press Enter to continue")
        return

    print()
    print("  Select account:")
    labels = [f"{a['name']}  ({a.get('type','')})  [{a['id']}]" for a in accounts]
    idx = pick_from_list(labels, "Account number")
    if idx is not None:
        session.account = accounts[idx]
        print(f"  Switched to: {session.account_name}")


# ---------------------------------------------------------------------------
# Main menu loop
# ---------------------------------------------------------------------------

def main_menu(session: Session) -> None:
    while True:
        positions  = load_positions()
        evals      = load_evals()
        open_count = sum(1 for p in positions if p.get("status") == "open")
        open_legs  = sum(
            sum(1 for l in p.get("legs", []) if l.get("exit_date") is None)
            for p in positions if p.get("status") == "open"
        )

        print()
        print(hr("═"))
        print(f"  TRADES  —  {TODAY}")
        print(f"  {session.status_line()}")
        print(hr("─"))
        print(f"  Open positions: {open_count}  |  Open legs: {open_legs}")
        print(hr("═"))
        _eval_stale_warning(evals)
        print()
        print("  1   Enter trades  (add positions / legs)")
        print("  2   Close a leg")
        print("  3   Close a position")
        print("  4   Report")
        print("  5   Dashboard")
        print(hr("─"))
        print("  a   Accounts")
        print("  t   Strategies")
        print("  e   Option strategies")
        print("  r   Reprice options")
        print("  s   Switch account")
        print("  d   Change session date")
        print(hr("─"))
        print("  h   Help & instructions")
        print("  i   Data sources & calculations")
        print("  q   Quit")
        print()

        choice = prompt("Choice").lower()

        if choice == "q":
            print()
            break
        elif choice == "1":
            menu_enter_trades(session)
        elif choice == "2":
            menu_close_leg(session)
        elif choice == "3":
            menu_close_position(session)
        elif choice == "4":
            menu_report(session)
        elif choice == "5":
            menu_dashboard(session)
        elif choice == "a":
            menu_accounts(session)
        elif choice == "t":
            menu_strategies(session)
        elif choice == "e":
            menu_evaluate_strategies(session)
        elif choice == "r":
            menu_reprice_options(session)
        elif choice == "h":
            menu_help(session)
        elif choice == "i":
            menu_data_info(session)
        elif choice == "s":
            switch_account(session)
        elif choice == "d":
            session.trade_date = prompt_date("Session date", session.trade_date)
            print(f"  Date set to {session.trade_date}")
        else:
            print("  Unknown choice.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    session = Session()

    # Auto-select account if there's only one
    accounts = load_accounts()
    if len(accounts) == 1:
        session.account = accounts[0]
    elif len(accounts) > 1:
        print()
        print(hr("═"))
        print(f"  TRADES  —  {TODAY}")
        print(hr("═"))
        print("\n  Select account to start with (Enter to skip):")
        labels = [f"{a['name']}  ({a.get('type','')})  [{a['id']}]" for a in accounts]
        idx = pick_from_list(labels, "Account number")
        if idx is not None:
            session.account = accounts[idx]

    main_menu(session)


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    main()
