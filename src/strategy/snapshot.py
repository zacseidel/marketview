"""
src/strategy/snapshot.py

Creates and manages strategy observations for each stock position.

Lifecycle:
  1. Stock buy executed → create_observation_set() → snapshots all 5 templates
  2. Model evaluation (still holding) → check_expirations() → close expired legs,
     open new legs for strategies whose options have expired
  3. Model sell → close_all_for_model_sell() → closes stock leg immediately,
     marks options legs "awaiting_chain"; queue handler closes them when chain arrives

Storage: data/strategy_observations/{ticker}_{entry_date}.json
"""

from __future__ import annotations

import json
import math
import uuid
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path

import structlog

from src.strategy.templates import (
    TEMPLATES,
    compute_strategy_exit_value,
    select_contract,
    _mid_price,
    _dte,
)

log = structlog.get_logger()

_OBS_DIR = Path("data/strategy_observations")
_THEORETICAL_OBS_DIR = Path("data/strategy_observations/theoretical")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class StrategyLeg:
    leg_type: str                    # "long_stock" | "long_call" | "short_call" | "short_put"
    entry_price: float               # premium paid (positive) or received (stored positive, sign handled in math)
    contract_symbol: str | None = None
    strike: float | None = None
    expiration: str | None = None    # YYYY-MM-DD
    entry_delta: float | None = None
    entry_iv: float | None = None
    close_price: float | None = None
    close_date: str | None = None
    close_reason: str | None = None  # "expiry" | "model_sell" | "unavailable"


@dataclass
class StrategyObservation:
    obs_id: str
    ticker: str
    strategy: str
    description: str
    entry_date: str
    entry_cost: float                # net capital deployed (positive = cash out)
    originating_models: list[str]
    legs: list[StrategyLeg]
    generation: int = 1              # increments when strategy re-opens after expiry
    status: str = "open"             # "open" | "awaiting_chain" | "closed" | "no_contracts"
    close_date: str | None = None
    close_value: float | None = None
    log_return: float | None = None
    close_reason: str | None = None  # "expiry" | "model_sell"
    notes: str | None = None


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def _obs_path(ticker: str, stock_entry_date: str, theoretical: bool = False) -> Path:
    obs_dir = _THEORETICAL_OBS_DIR if theoretical else _OBS_DIR
    obs_dir.mkdir(parents=True, exist_ok=True)
    return obs_dir / f"{ticker}_{stock_entry_date}.json"


def load_observations(ticker: str, stock_entry_date: str, theoretical: bool = False) -> list[StrategyObservation]:
    path = _obs_path(ticker, stock_entry_date, theoretical)
    if not path.exists():
        return []
    with open(path) as f:
        raw = json.load(f)
    obs = []
    for r in raw:
        legs = [StrategyLeg(**leg) for leg in r.pop("legs", [])]
        obs.append(StrategyObservation(legs=legs, **r))
    return obs


def save_observations(ticker: str, stock_entry_date: str, observations: list[StrategyObservation], theoretical: bool = False) -> None:
    path = _obs_path(ticker, stock_entry_date, theoretical)
    data = [asdict(obs) for obs in observations]
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    tmp.replace(path)


def load_all_observations() -> list[StrategyObservation]:
    """Load every observation across all executed stock positions."""
    all_obs = []
    for path in _OBS_DIR.glob("*.json"):
        with open(path) as f:
            raw = json.load(f)
        for r in raw:
            legs = [StrategyLeg(**leg) for leg in r.pop("legs", [])]
            all_obs.append(StrategyObservation(legs=legs, **r))
    return all_obs


def load_all_theoretical_observations() -> list[StrategyObservation]:
    """Load every theoretical (model-signal) observation, independent of user buy decisions."""
    if not _THEORETICAL_OBS_DIR.exists():
        return []
    all_obs = []
    for path in _THEORETICAL_OBS_DIR.glob("*.json"):
        with open(path) as f:
            raw = json.load(f)
        for r in raw:
            legs = [StrategyLeg(**leg) for leg in r.pop("legs", [])]
            all_obs.append(StrategyObservation(legs=legs, **r))
    return all_obs


# ---------------------------------------------------------------------------
# Entry cost calculation
# ---------------------------------------------------------------------------

def _entry_cost(strategy: str, legs: list[StrategyLeg], stock_price: float) -> float:
    """
    Net capital deployed for the strategy (per unit / per share equivalent).

    Stock:         stock_price
    Covered call:  stock_price - call_premium
    LEAP OTM:      call_premium
    Diagonal:      long_call_premium - short_call_premium
    CSP:           strike - put_premium  (cash reserved minus premium received)
    """
    if strategy == "stock":
        return stock_price

    premiums = {leg.leg_type: leg.entry_price for leg in legs}

    if strategy == "covered_call":
        return stock_price - premiums.get("short_call", 0)

    if strategy == "leap_otm":
        return premiums.get("long_call", 0)

    if strategy == "diagonal":
        return premiums.get("long_call", 0) - premiums.get("short_call", 0)

    if strategy == "csp":
        cash_leg = next((l for l in legs if l.leg_type == "cash_reserved"), None)
        put_leg = next((l for l in legs if l.leg_type == "short_put"), None)
        if cash_leg and put_leg:
            return cash_leg.entry_price - put_leg.entry_price
        return 0.0

    return stock_price


# ---------------------------------------------------------------------------
# Create observations
# ---------------------------------------------------------------------------

def create_observation_set(
    ticker: str,
    stock_price: float,
    chain: list[dict],
    entry_date: str,
    originating_models: list[str],
    generation: int = 1,
    existing_strategies: set[str] | None = None,
) -> list[StrategyObservation]:
    """
    Snapshot all strategy templates against the current options chain.
    Returns a list of new StrategyObservation objects (not yet saved).

    existing_strategies: skip strategies that are already open (used for re-opening expired legs)
    """
    observations: list[StrategyObservation] = []
    skip = existing_strategies or set()

    for strategy_name, template in TEMPLATES.items():
        if strategy_name in skip:
            continue

        obs_id = f"{ticker}_{strategy_name}_{entry_date}_g{generation}"
        legs: list[StrategyLeg] = []
        failed = False

        # Stock leg (for strategies that include owning the stock)
        if strategy_name in ("stock", "covered_call", "diagonal"):
            legs.append(StrategyLeg(
                leg_type="long_stock",
                entry_price=stock_price,
            ))

        # Options legs
        for leg_spec in template["legs"]:
            contract = select_contract(chain, leg_spec, stock_price, entry_date)

            if contract is None:
                log.warning(
                    "snapshot.no_contract",
                    ticker=ticker,
                    strategy=strategy_name,
                    leg_type=leg_spec["leg_type"],
                )
                failed = True
                break

            mid = _mid_price(contract)
            if mid is None:
                log.warning("snapshot.no_price", ticker=ticker, strategy=strategy_name)
                failed = True
                break

            details = contract["details"]
            greeks = contract.get("greeks", {}) or {}

            legs.append(StrategyLeg(
                leg_type=leg_spec["leg_type"],
                entry_price=mid,
                contract_symbol=details.get("ticker"),
                strike=details.get("strike_price"),
                expiration=details.get("expiration_date"),
                entry_delta=greeks.get("delta"),
                entry_iv=contract.get("implied_volatility"),
            ))

        # CSP: add a cash_reserved leg so exit value accounts for cash returned at expiry
        if strategy_name == "csp" and not failed:
            put_leg = next((l for l in legs if l.leg_type == "short_put"), None)
            if put_leg and put_leg.strike:
                legs.append(StrategyLeg(
                    leg_type="cash_reserved",
                    entry_price=put_leg.strike,
                ))

        if failed:
            observations.append(StrategyObservation(
                obs_id=obs_id,
                ticker=ticker,
                strategy=strategy_name,
                description=template["description"],
                entry_date=entry_date,
                entry_cost=0.0,
                originating_models=originating_models,
                legs=[],
                generation=generation,
                status="no_contracts",
                notes="No suitable contracts found in chain",
            ))
            continue

        entry_cost = _entry_cost(strategy_name, legs, stock_price)

        observations.append(StrategyObservation(
            obs_id=obs_id,
            ticker=ticker,
            strategy=strategy_name,
            description=template["description"],
            entry_date=entry_date,
            entry_cost=entry_cost,
            originating_models=originating_models,
            legs=legs,
            generation=generation,
        ))
        log.info(
            "snapshot.created",
            ticker=ticker,
            strategy=strategy_name,
            entry_cost=round(entry_cost, 2),
            legs=len(legs),
            generation=generation,
        )

    return observations


# ---------------------------------------------------------------------------
# Close / check expirations
# ---------------------------------------------------------------------------

def _close_observation(
    obs: StrategyObservation,
    close_date: str,
    stock_price: float,
    reason: str,
    chain: list[dict] | None = None,
) -> bool:
    """
    Attempt to close an observation. Returns True if fully resolved.
    If options can't be priced (need chain), sets status to "awaiting_chain".
    """
    # Build the leg dicts for compute_strategy_exit_value
    leg_dicts = [asdict(leg) for leg in obs.legs]

    exit_value, all_resolved = compute_strategy_exit_value(
        leg_dicts, stock_price, chain=chain, close_date=close_date
    )

    if not all_resolved:
        obs.status = "awaiting_chain"
        obs.close_date = close_date
        obs.close_reason = reason
        log.info("snapshot.awaiting_chain", ticker=obs.ticker, strategy=obs.strategy, obs_id=obs.obs_id)
        return False

    obs.close_date = close_date
    obs.close_value = round(exit_value, 4)
    obs.close_reason = reason
    obs.status = "closed"

    if obs.entry_cost > 0 and exit_value > 0:
        obs.log_return = round(math.log(exit_value / obs.entry_cost), 6)
    elif obs.entry_cost > 0 and exit_value <= 0:
        # Total loss — log return is -inf, store as a large negative
        obs.log_return = -10.0
        obs.notes = (obs.notes or "") + " | total loss"
    else:
        obs.log_return = None

    log.info(
        "snapshot.closed",
        ticker=obs.ticker,
        strategy=obs.strategy,
        entry_cost=obs.entry_cost,
        exit_value=exit_value,
        log_return=obs.log_return,
        reason=reason,
    )
    return True


def check_expirations(
    ticker: str,
    stock_entry_date: str,
    eval_date: str,
    stock_price: float,
) -> list[str]:
    """
    At each model evaluation, close any options legs that have expired.
    Returns list of strategies that need new legs opened (expired since last eval).
    Uses stock price + intrinsic value — no chain needed for expired options.
    """
    observations = load_observations(ticker, stock_entry_date)
    needs_reopen: list[str] = []

    for obs in observations:
        if obs.status != "open":
            continue

        # Check if any options legs have expired
        options_legs = [l for l in obs.legs if l.leg_type != "long_stock"]
        if not options_legs:
            continue

        earliest_expiry = min(l.expiration for l in options_legs if l.expiration)
        if earliest_expiry and earliest_expiry <= eval_date:
            resolved = _close_observation(obs, earliest_expiry, stock_price, reason="expiry")
            if resolved:
                needs_reopen.append(obs.strategy)

    save_observations(ticker, stock_entry_date, observations)
    return needs_reopen


def reopen_expired_strategies(
    ticker: str,
    stock_entry_date: str,
    strategies_to_reopen: list[str],
    eval_date: str,
    stock_price: float,
    chain: list[dict],
    originating_models: list[str],
) -> None:
    """Open new generation observations for strategies whose options have expired."""
    observations = load_observations(ticker, stock_entry_date)

    for strategy in strategies_to_reopen:
        # Find highest generation already recorded for this strategy
        existing_gens = [o.generation for o in observations if o.strategy == strategy]
        next_gen = max(existing_gens, default=0) + 1

        new_obs = create_observation_set(
            ticker=ticker,
            stock_price=stock_price,
            chain=chain,
            entry_date=eval_date,
            originating_models=originating_models,
            generation=next_gen,
            existing_strategies=set(TEMPLATES.keys()) - {strategy},
        )
        observations.extend(new_obs)
        log.info("snapshot.reopened", ticker=ticker, strategy=strategy, generation=next_gen)

    save_observations(ticker, stock_entry_date, observations)


def close_all_for_model_sell(
    ticker: str,
    stock_entry_date: str,
    close_date: str,
    stock_price: float,
    chain: list[dict] | None = None,
    theoretical: bool = False,
) -> bool:
    """
    Model says sell. Close all open observations.
    Stock legs close immediately at stock_price.
    Options legs: if expired → intrinsic; if chain provided → mid price; else → awaiting_chain.
    Returns True if all observations fully closed, False if any are awaiting chain.
    """
    observations = load_observations(ticker, stock_entry_date, theoretical)
    all_closed = True

    for obs in observations:
        if obs.status in ("closed", "no_contracts"):
            continue
        resolved = _close_observation(obs, close_date, stock_price, reason="model_sell", chain=chain)
        if not resolved:
            all_closed = False

    save_observations(ticker, stock_entry_date, observations, theoretical)
    return all_closed


def close_theoretical_for_ticker(ticker: str, close_date: str, stock_price: float) -> tuple[bool, list[str]]:
    """
    Close all open theoretical observations for a ticker across all entry dates.
    Returns (all_resolved, entry_dates_needing_chain) where entry_dates_needing_chain
    lists entry dates that have awaiting_chain observations requiring a chain fetch.
    """
    if not _THEORETICAL_OBS_DIR.exists():
        return True, []

    all_resolved = True
    needs_chain: list[str] = []

    for obs_file in _THEORETICAL_OBS_DIR.glob(f"{ticker}_*.json"):
        entry_date = obs_file.stem[len(ticker) + 1:]
        resolved = close_all_for_model_sell(
            ticker=ticker,
            stock_entry_date=entry_date,
            close_date=close_date,
            stock_price=stock_price,
            theoretical=True,
        )
        if not resolved:
            all_resolved = False
            needs_chain.append(entry_date)

    return all_resolved, needs_chain


def close_awaiting_chain(
    ticker: str,
    stock_entry_date: str,
    chain: list[dict],
    close_date: str,
    stock_price: float,
    theoretical: bool = False,
) -> int:
    """
    Called by queue processor after fetching the options chain.
    Closes all "awaiting_chain" observations using the provided chain.
    Returns number of observations resolved.
    """
    observations = load_observations(ticker, stock_entry_date, theoretical)
    resolved_count = 0

    for obs in observations:
        if obs.status != "awaiting_chain":
            continue
        resolved = _close_observation(
            obs, obs.close_date or close_date, stock_price,
            reason=obs.close_reason or "model_sell", chain=chain
        )
        if resolved:
            resolved_count += 1

    save_observations(ticker, stock_entry_date, observations, theoretical)
    return resolved_count
