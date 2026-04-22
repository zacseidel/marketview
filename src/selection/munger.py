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

import json
import math
from pathlib import Path

import structlog

from src.selection.base import DataAccessLayer, HoldingRecord, SelectionModel

log = structlog.get_logger()

_UNIVERSE_FILE = Path("data.nosync/universe/constituents.json")
_MODELS_DIR = Path("data.nosync/models")

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


def _top100_sp500_tickers(universe_size: int, eval_date: str = "") -> list[str]:
    """Return top `universe_size` S&P 500 tickers sorted by market_cap descending."""
    with open(_UNIVERSE_FILE) as f:
        constituents = json.load(f)

    sp500 = [
        v for v in constituents.values()
        if v.get("status") == "active" and v.get("tier") == "sp500"
    ]
    sp500.sort(key=lambda v: v.get("market_cap") or 0.0, reverse=True)
    top = sp500[:universe_size]

    if eval_date:
        _save_universe(top, eval_date)

    return [v["ticker"] for v in top]


def _save_universe(ranked: list[dict], eval_date: str) -> None:
    """Persist the top-N market-cap ranking as a sidecar for the dashboard to diff."""
    out_dir = _MODELS_DIR / eval_date
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "munger_universe.json"
    records = [
        {"ticker": v["ticker"], "rank": i + 1, "market_cap": v.get("market_cap") or 0.0, "name": v.get("name", "")}
        for i, v in enumerate(ranked)
    ]
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(records, f, indent=2)
    tmp.replace(path)
    log.info("munger.universe_saved", path=str(path), count=len(records))


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

        tickers = _top100_sp500_tickers(cfg["universe_size"], eval_date=eval_date)
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

        # Check previously-held tickers that weren't re-picked for EMA15 violations.
        # If EMA15 is broken → immediate sell. If still above EMA15 but the SMA200
        # touch window expired → let the time-based exit handle it (no explicit sell here).
        prev_tickers: set[str] = set(cfg.get("prev_tickers", []))
        re_picked = {h.ticker for h in result_list}
        top100_set = set(tickers)
        for ticker in prev_tickers - re_picked:
            if ticker not in top100_set:
                continue  # fell out of top-100 universe; time-based exit applies
            try:
                prices = dal.get_prices(ticker, lookback_days=cfg["sma_long"] + 10)
                close = prices["close"]
                if len(close) < 2:
                    continue
                current = float(close.iloc[-1])
                ema15 = _ema(close, cfg["ema_short"])
                if current <= ema15:
                    result_list.append(HoldingRecord(
                        model="munger",
                        eval_date=eval_date,
                        ticker=ticker,
                        conviction=0.0,
                        rationale=f"EMA{cfg['ema_short']} exit: ${current:.2f} ≤ EMA ${ema15:.2f}",
                        status="sell",
                    ))
            except Exception as exc:
                log.debug("munger.ema_exit_check_error", ticker=ticker, error=str(exc))

        log.info("munger.complete", qualifying=len(holdings), selected=len(result_list))
        return result_list
