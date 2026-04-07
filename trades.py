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
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path

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
        try:
            datetime.strptime(raw, "%Y-%m-%d")
            return raw
        except ValueError:
            print("  Format: YYYY-MM-DD")


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
            out["options_yield"]  = pnl / stock_val
            out["total_log_ret"]  = out["stock_log_ret"] + out["options_yield"]

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
    if open_legs:
        print(f"  Note: {len(open_legs)} open leg(s) remain — close them separately if needed.")

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

    a = compute_analytics(pos)
    print(f"  Closed {pos['ticker']}.  "
          f"Stock return: {fmt_ret(a.get('stock_log_ret'))}  "
          f"vs SPY: {fmt_ret(a.get('excess_ret'))}")


# ---------------------------------------------------------------------------
# Menu: Dashboard
# ---------------------------------------------------------------------------

def menu_dashboard(session: Session) -> None:
    positions = load_positions()
    open_pos  = [p for p in positions if p.get("status") == "open"]
    closed_pos = sorted(
        [p for p in positions if p.get("status") == "closed"],
        key=lambda p: p.get("exit_date", ""),
        reverse=True,
    )[:6]

    print_header("DASHBOARD", session)
    print(f"  Open: {len(open_pos)}  |  Closed: {len(positions)-len(open_pos)}")

    if open_pos:
        print_section("Open Positions")
        show_open_positions_table(positions)

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
        ("DATA FILES", [
            "All data lives in trades/ (three JSON files):",
            "  trades/accounts.json   — account definitions",
            "  trades/strategies.json — strategy playbook",
            "  trades/positions.json  — positions with legs embedded",
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
        print()
        print("  1   Enter trades  (add positions / legs)")
        print("  2   Close a leg")
        print("  3   Close a position")
        print("  4   Report")
        print("  5   Dashboard")
        print(hr("─"))
        print("  a   Accounts")
        print("  t   Strategies")
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
    main()
