"""
src/strategy/templates.py

Strategy template definitions and options contract selection logic.

Each template describes the legs of one strategy. The selector functions
scan an options chain and return the best-matching contract for each leg.

Strategies:
    stock         — own stock outright
    covered_call  — stock + short call (delta 0.20–0.25, 21–45 DTE)
    leap_otm      — long call, 10% OTM strike, ~500 DTE
    diagonal      — long call 25% ITM ~500 DTE + short call (delta 0.20–0.25, 21–45 DTE)
    csp           — short put, ATM strike, ~21 DTE
"""

from __future__ import annotations

import math
from datetime import date
from typing import Any

import structlog

log = structlog.get_logger()

# ---------------------------------------------------------------------------
# Template definitions
# ---------------------------------------------------------------------------

TEMPLATES: dict[str, dict] = {
    "stock": {
        "description": "Own stock outright",
        "legs": [],   # no options legs; handled separately
    },
    "covered_call": {
        "description": "Stock + short call, delta 0.20–0.25, 21–45 DTE",
        "legs": [
            {
                "leg_type": "short_call",
                "dte_min": 21, "dte_max": 45,
                "select_by": "delta",
                "target_delta": 0.225,
                "delta_min": 0.20, "delta_max": 0.25,
            }
        ],
    },
    "leap_otm": {
        "description": "Long call, 10% OTM, ~500 DTE",
        "legs": [
            {
                "leg_type": "long_call",
                "dte_min": 450, "dte_max": 550,
                "select_by": "strike_pct",
                "target_strike_pct": 1.10,   # 10% above stock price
            }
        ],
    },
    "diagonal": {
        "description": "Long call 25% ITM ~500 DTE + short call delta 0.20–0.25 21–45 DTE",
        "legs": [
            {
                "leg_type": "long_call",
                "dte_min": 450, "dte_max": 550,
                "select_by": "strike_pct",
                "target_strike_pct": 0.75,   # 25% below stock price = ITM call
            },
            {
                "leg_type": "short_call",
                "dte_min": 21, "dte_max": 45,
                "select_by": "delta",
                "target_delta": 0.225,
                "delta_min": 0.20, "delta_max": 0.25,
            },
        ],
    },
    "csp": {
        "description": "Short put, ATM, ~21 DTE",
        "legs": [
            {
                "leg_type": "short_put",
                "dte_min": 18, "dte_max": 25,
                "select_by": "strike_pct",
                "target_strike_pct": 1.00,   # at-the-money
            }
        ],
    },
}


# ---------------------------------------------------------------------------
# Contract selection
# ---------------------------------------------------------------------------

def _dte(contract: dict, eval_date: str) -> int:
    exp = date.fromisoformat(contract["details"]["expiration_date"])
    return (exp - date.fromisoformat(eval_date)).days


def _mid_price(contract: dict) -> float | None:
    q = contract.get("last_quote", {})
    mid = q.get("midpoint")
    if mid and mid > 0:
        return float(mid)
    bid = q.get("bid", 0)
    ask = q.get("ask", 0)
    if ask > 0:
        return (bid + ask) / 2
    return None


def select_contract(
    chain: list[dict],
    leg_spec: dict,
    stock_price: float,
    eval_date: str,
) -> dict | None:
    """
    Scan the options chain and return the best contract for a leg spec.
    Returns None if no suitable contract is found.
    """
    leg_type = leg_spec["leg_type"]
    contract_type = "call" if "call" in leg_type else "put"
    dte_min = leg_spec["dte_min"]
    dte_max = leg_spec["dte_max"]

    # Filter: type + DTE window + must have a usable price
    candidates = [
        c for c in chain
        if c["details"]["contract_type"] == contract_type
        and dte_min <= _dte(c, eval_date) <= dte_max
        and _mid_price(c) is not None
    ]

    if not candidates:
        return None

    select_by = leg_spec["select_by"]

    if select_by == "delta":
        # Require greeks; filter to those within the delta range
        with_greeks = [c for c in candidates if c.get("greeks") and c["greeks"].get("delta") is not None]
        if not with_greeks:
            # Fall back to strike-based selection near target delta-equivalent strike
            log.debug("templates.no_greeks_fallback", leg_type=leg_type)
            target_strike = stock_price * 1.05  # rough 0.225 delta proxy
            return min(candidates, key=lambda c: abs(c["details"]["strike_price"] - target_strike))

        target = leg_spec["target_delta"]
        # Use absolute delta (puts have negative delta)
        return min(with_greeks, key=lambda c: abs(abs(c["greeks"]["delta"]) - target))

    elif select_by == "strike_pct":
        target_strike = stock_price * leg_spec["target_strike_pct"]
        return min(candidates, key=lambda c: abs(c["details"]["strike_price"] - target_strike))

    return None


# ---------------------------------------------------------------------------
# P&L calculation at close
# ---------------------------------------------------------------------------

def compute_leg_close_value(
    leg: dict,
    stock_price: float,
    chain: list[dict] | None = None,
    close_date: str | None = None,
) -> tuple[float | None, str]:
    """
    Compute the close value of one options leg.

    For expired options: uses intrinsic value from stock_price (no chain needed).
    For active options: uses mid price from chain if provided.

    Returns (close_value, method) where method is "intrinsic" | "mid_price" | "unavailable".
    close_value is None if we can't determine it yet.
    """
    leg_type = leg["leg_type"]
    expiration = leg.get("expiration")
    strike = leg.get("strike")
    contract_symbol = leg.get("contract_symbol")

    if leg_type == "long_stock":
        return stock_price, "stock_price"

    if leg_type == "cash_reserved":
        # CSP cash collateral — always returned at close
        return leg["entry_price"], "cash"

    # Check if the option has expired
    expired = False
    if expiration and close_date:
        expired = expiration <= close_date

    if expired:
        # Intrinsic value only (time value = 0 at expiry)
        if leg_type in ("long_call", "short_call"):
            intrinsic = max(0.0, stock_price - strike)
            sign = 1 if leg_type == "long_call" else -1
            return sign * intrinsic, "intrinsic"
        elif leg_type in ("long_put", "short_put"):
            intrinsic = max(0.0, strike - stock_price)
            sign = 1 if leg_type == "long_put" else -1
            return sign * intrinsic, "intrinsic"

    # Option not expired — need options chain
    if chain is None:
        return None, "unavailable"

    # Find the contract in the chain by symbol or strike/type/expiry
    matched = None
    if contract_symbol:
        matched = next(
            (c for c in chain if c["details"]["ticker"] == contract_symbol),
            None,
        )

    if matched is None and strike and expiration:
        # Fallback: match by strike + expiry + type
        contract_type = "call" if "call" in leg_type else "put"
        matched = next(
            (
                c for c in chain
                if c["details"]["contract_type"] == contract_type
                and c["details"]["strike_price"] == strike
                and c["details"]["expiration_date"] == expiration
            ),
            None,
        )

    if matched is None:
        log.warning("templates.contract_not_found_in_chain", symbol=contract_symbol, strike=strike, expiry=expiration)
        return None, "unavailable"

    mid = _mid_price(matched)
    if mid is None:
        return None, "unavailable"

    sign = 1 if leg_type.startswith("long_") else -1
    return sign * mid, "mid_price"


def compute_strategy_exit_value(
    legs: list[dict],
    stock_price: float,
    chain: list[dict] | None = None,
    close_date: str | None = None,
) -> tuple[float | None, bool]:
    """
    Compute total exit value of all legs in a strategy.

    Returns (exit_value, all_resolved).
    exit_value is None if any leg could not be priced.
    all_resolved is False if any leg returned "unavailable".
    """
    total = 0.0
    all_resolved = True

    for leg in legs:
        value, method = compute_leg_close_value(leg, stock_price, chain=chain, close_date=close_date)
        if value is None:
            all_resolved = False
            return None, False
        total += value

    return total, all_resolved
