"""
src/selection/munger.py

Munger model: buy quality large-caps that dipped to/below their 200-day SMA
and have since recovered above their 15-day EMA.

Universe: top 100 S&P 500 tickers by market cap (from constituents.json).

Buy/hold signal (both must be true):
  - Price touched at or below SMA200 at least once in the last 21 trading days
  - Current price is above EMA15

Sell signal:
  - Current price is at or below EMA15

Conviction scoring (0–1):
  - How far current price is above EMA15, normalized (5% above = full score)
"""

from __future__ import annotations

import math
from pathlib import Path

import structlog

from src.selection.base import DataAccessLayer, HoldingRecord, SelectionModel

log = structlog.get_logger()

_UNIVERSE_FILE = Path("data/universe/constituents.json")

_DEFAULT_CONFIG = {
    "universe_size": 100,       # top N S&P 500 tickers by market cap
    "sma_long": 200,
    "ema_short": 15,
    "dip_lookback": 21,         # trading days to look back for SMA200 touch
    "max_holdings": 20,
}


def _ema(close_series, span: int) -> float:
    """Compute EMA of a pandas Series, return most recent value."""
    return float(close_series.ewm(span=span, adjust=False).mean().iloc[-1])


def _top100_sp500_tickers(universe_size: int) -> list[str]:
    """Return top `universe_size` S&P 500 tickers sorted by market_cap descending."""
    import json
    with open(_UNIVERSE_FILE) as f:
        constituents = json.load(f)

    sp500 = [
        v for v in constituents.values()
        if v.get("status") == "active" and v.get("tier") == "sp500"
    ]
    sp500.sort(key=lambda v: v.get("market_cap") or 0.0, reverse=True)
    return [v["ticker"] for v in sp500[:universe_size]]


def _score_ticker(ticker: str, dal: DataAccessLayer, config: dict) -> tuple[float, str, dict] | None:
    """
    Returns (conviction, rationale, metadata) if ticker qualifies, else None.
    """
    sma_long = config["sma_long"]
    ema_short = config["ema_short"]
    dip_lookback = config["dip_lookback"]

    prices = dal.get_prices(ticker, lookback_days=sma_long + 10)
    close = prices["close"]

    if len(close) < sma_long:
        return None

    current = float(close.iloc[-1])
    sma200 = float(close.rolling(sma_long).mean().iloc[-1])
    ema15 = _ema(close, ema_short)

    # Sell condition: current price at or below EMA15 → not a hold
    if current <= ema15:
        return None

    # Buy condition: price must have touched at or below SMA200 in last `dip_lookback` days
    recent = close.iloc[-dip_lookback:]
    sma200_series = close.rolling(sma_long).mean().iloc[-dip_lookback:]
    touched_below_sma200 = bool((recent <= sma200_series).any())

    if not touched_below_sma200:
        return None

    # Conviction: how far above EMA15 is current price (5% above = 1.0)
    pct_above_ema = (current - ema15) / ema15
    conviction = round(min(pct_above_ema / 0.05, 1.0), 3)

    rationale = (
        f"Price ${current:.2f} > EMA{ema_short} ${ema15:.2f} "
        f"({pct_above_ema*100:.1f}% above); "
        f"touched SMA{sma_long} ${sma200:.2f} within last {dip_lookback}d"
    )
    metadata = {
        "current_price": round(current, 2),
        "sma200": round(sma200, 2),
        "ema15": round(ema15, 2),
        "pct_above_ema15": round(pct_above_ema, 4),
        "touched_sma200_in_lookback": True,
    }
    return conviction, rationale, metadata


class MungerModel(SelectionModel):
    def run(self, config: dict, dal: DataAccessLayer) -> list[HoldingRecord]:
        cfg = {**_DEFAULT_CONFIG, **config}
        eval_date = cfg.get("eval_date", "")

        tickers = _top100_sp500_tickers(cfg["universe_size"])
        log.info("munger.universe", count=len(tickers))

        holdings: list[HoldingRecord] = []

        for ticker in tickers:
            try:
                result = _score_ticker(ticker, dal, cfg)
            except Exception as exc:
                log.debug("munger.score_error", ticker=ticker, error=str(exc))
                continue

            if result is None:
                continue

            conviction, rationale, metadata = result
            holdings.append(HoldingRecord(
                model="munger",
                eval_date=eval_date,
                ticker=ticker,
                conviction=conviction,
                rationale=rationale,
                metadata=metadata,
            ))

        holdings.sort(key=lambda h: h.conviction, reverse=True)
        result_list = holdings[:cfg["max_holdings"]]

        log.info("munger.complete", qualifying=len(holdings), selected=len(result_list))
        return result_list
