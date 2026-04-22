"""
src/selection/repurchase.py

Share repurchase model.

Buys the top N S&P 500 stocks by trailing-12-month share buyback percentage
(shares repurchased / shares outstanding at start of period), provided they
are currently trading above their 21-day EMA.

Sell signals are issued when a holding either:
  - Falls out of the top-N buyback ranking, or
  - Drops below its 21-day EMA.
"""

from __future__ import annotations

import math
import time
from datetime import datetime, timedelta

import structlog

from src.selection.base import DataAccessLayer, HoldingRecord, SelectionModel

log = structlog.get_logger()

_LOOKBACK_DAYS = 365          # 12-month buyback window (calendar days)
_EMA_PERIOD = 21              # 21-day EMA for price filter
_MIN_QUARTERS = 2             # need at least 2 quarterly data points to compute a delta
_DEFAULT_MAX_HOLDINGS = 5


def _compute_ema(closes: list[float], period: int) -> float | None:
    """Return the most recent EMA value, or None if insufficient data."""
    if len(closes) < period:
        return None
    k = 2.0 / (period + 1)
    ema = closes[0]
    for price in closes[1:]:
        ema = price * k + ema * (1 - k)
    return ema


def _trailing_buyback_pct(quarters: list[dict], cutoff_date: str) -> tuple[float | None, list[dict]]:
    """
    Compute shares repurchased as a fraction of shares outstanding 12 months ago.

    Strategy:
      - Find the most recent quarter with valid shares_outstanding filed on or
        before cutoff_date (current shares).
      - Find the most recent quarter with a filing_date at least 365 days before
        the current quarter's filing_date (baseline shares).
      - buyback_pct = (baseline - current) / baseline
        Positive means shares were reduced (bought back).
    """
    valid = [
        q for q in quarters
        if (q.get("shares_outstanding") or 0) > 0
        and q.get("filing_date", "") <= cutoff_date
        and not q.get("period", "").startswith("TTM")   # skip trailing-twelve-month summaries
        and not q.get("period", "").startswith("FY")    # skip full-year summaries
    ]
    if len(valid) < _MIN_QUARTERS:
        return None, []

    # Sort most-recent first
    valid.sort(key=lambda q: q["filing_date"], reverse=True)
    current = valid[0]
    current_date = datetime.strptime(current["filing_date"], "%Y-%m-%d")
    baseline_cutoff = (current_date - timedelta(days=_LOOKBACK_DAYS)).strftime("%Y-%m-%d")

    # Find most recent quarter filed on or before baseline_cutoff
    baseline_candidates = [q for q in valid if q["filing_date"] <= baseline_cutoff]
    if not baseline_candidates:
        return None, []

    baseline = baseline_candidates[0]
    baseline_shares = baseline["shares_outstanding"]
    current_shares = current["shares_outstanding"]

    if baseline_shares <= 0:
        return None, []

    return (baseline_shares - current_shares) / baseline_shares, valid


class RepurchaseModel(SelectionModel):
    def run(self, config: dict, dal: DataAccessLayer) -> list[HoldingRecord]:
        t_start = time.perf_counter()
        eval_date = config.get("eval_date", "")
        max_holdings = config.get("max_holdings", _DEFAULT_MAX_HOLDINGS)
        cutoff_date = eval_date or datetime.today().strftime("%Y-%m-%d")

        t1 = time.perf_counter()
        tickers = dal.get_all_tickers()
        log.info("repurchase.running", universe=len(tickers), get_tickers_ms=round((time.perf_counter() - t1) * 1000))

        # Pre-warm fundamentals cache (logs dal.fundamentals_loaded if not already loaded)
        t2 = time.perf_counter()
        dal.get_fundamentals("__warmup__")  # triggers _ensure_fundamentals()
        log.debug("repurchase.fundamentals_ready", elapsed_ms=round((time.perf_counter() - t2) * 1000))

        candidates: list[tuple[str, float, float, dict]] = []  # (ticker, buyback_pct, ema, meta)
        t_loop = time.perf_counter()

        for i, ticker in enumerate(tickers):
            if i > 0 and i % 100 == 0:
                elapsed = time.perf_counter() - t_loop
                log.info(
                    "repurchase.progress",
                    processed=i,
                    total=len(tickers),
                    candidates=len(candidates),
                    elapsed_s=round(elapsed, 1),
                    rate=round(i / elapsed, 1),
                )

            t_fund = time.perf_counter()
            quarters = dal.get_fundamentals(ticker)
            fund_ms = round((time.perf_counter() - t_fund) * 1000)
            if fund_ms > 50:
                log.warning("repurchase.slow_fundamentals", ticker=ticker, elapsed_ms=fund_ms)

            if not quarters:
                continue

            try:
                t_bb = time.perf_counter()
                buyback_pct, valid = _trailing_buyback_pct(quarters, cutoff_date)
                bb_ms = round((time.perf_counter() - t_bb) * 1000)
                if bb_ms > 50:
                    log.warning("repurchase.slow_buyback_calc", ticker=ticker, elapsed_ms=bb_ms)
            except Exception as exc:
                log.debug("repurchase.buyback_error", ticker=ticker, error=str(exc))
                continue

            if buyback_pct is None or buyback_pct <= 0:
                continue  # no buyback or shares increased

            # Check 21-day EMA gate
            t_price = time.perf_counter()
            prices = dal.get_prices(ticker, lookback_days=63)  # ~3 months for stable EMA
            price_ms = round((time.perf_counter() - t_price) * 1000)
            if price_ms > 50:
                log.warning("repurchase.slow_price_lookup", ticker=ticker, elapsed_ms=price_ms)

            if prices.empty or "close" not in prices.columns:
                continue

            closes = prices["close"].dropna().tolist()
            ema21 = _compute_ema(closes, _EMA_PERIOD)
            if ema21 is None:
                continue

            current_price = closes[-1]
            if current_price <= ema21:
                continue  # below EMA — not a buy

            latest = valid[0]  # already filtered and sorted by _trailing_buyback_pct

            meta = {
                "buyback_pct_12m": round(buyback_pct * 100, 3),
                "current_shares": latest.get("shares_outstanding"),
                "latest_period": latest.get("period"),
                "latest_filing": latest.get("filing_date"),
                "ema21": round(ema21, 3),
                "current_price": round(current_price, 3),
                "pct_above_ema21": round((current_price / ema21 - 1) * 100, 2),
            }
            candidates.append((ticker, buyback_pct, ema21, meta))

        log.info("repurchase.loop_done", tickers=len(tickers), candidates=len(candidates), elapsed_s=round(time.perf_counter() - t_loop, 2))

        # Rank by buyback percentage descending, take top N
        candidates.sort(key=lambda x: x[1], reverse=True)
        top = candidates[:max_holdings]

        if not top:
            log.info("repurchase.no_candidates")
            return []

        # Conviction = rank-normalized within top N (rank 1 = 1.0, rank N = 0.0)
        n = len(top)
        holdings: list[HoldingRecord] = []
        for rank, (ticker, buyback_pct, ema21, meta) in enumerate(top):
            conviction = round(0.2 + 0.8 * (1.0 - rank / max(n - 1, 1)), 3)
            pct_above = meta["pct_above_ema21"]
            rationale = (
                f"Repurchase: {buyback_pct:.1%} of shares bought back in trailing 12mo "
                f"(rank {rank + 1}/{n}); "
                f"price {pct_above:+.1f}% above 21d EMA"
            )
            holdings.append(HoldingRecord(
                model="repurchase",
                eval_date=eval_date,
                ticker=ticker,
                conviction=conviction,
                rationale=rationale,
                metadata=meta,
            ))

        # Check previously-held tickers that weren't re-picked for EMA21 violations.
        # If EMA21 is broken → immediate sell. If still above EMA21 but dropped from
        # the top-N ranking → let the time-based exit handle it (no explicit sell here).
        prev_tickers: set[str] = set(config.get("prev_tickers", []))
        re_picked = {h.ticker for h in holdings}
        for ticker in prev_tickers - re_picked:
            try:
                prices = dal.get_prices(ticker, lookback_days=63)
                if prices.empty or "close" not in prices.columns:
                    continue
                closes = prices["close"].dropna().tolist()
                ema21 = _compute_ema(closes, _EMA_PERIOD)
                if ema21 is None:
                    continue
                current_price = closes[-1]
                if current_price <= ema21:
                    holdings.append(HoldingRecord(
                        model="repurchase",
                        eval_date=eval_date,
                        ticker=ticker,
                        conviction=0.0,
                        rationale=f"EMA{_EMA_PERIOD} exit: ${current_price:.2f} ≤ EMA ${ema21:.2f}",
                        status="sell",
                    ))
            except Exception as exc:
                log.debug("repurchase.ema_exit_check_error", ticker=ticker, error=str(exc))

        log.info("repurchase.complete", candidates=len(candidates), selected=len(holdings), total_elapsed_s=round(time.perf_counter() - t_start, 2))
        return holdings
