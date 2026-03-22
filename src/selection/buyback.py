"""
src/selection/buyback.py

Buyback-based stock selection model.

Flags tickers with 2+ consecutive quarters of declining shares_outstanding (≥1%/quarter).
Conviction scales with the number of qualifying quarters and magnitude of reduction.
"""

from __future__ import annotations

import structlog

from src.selection.base import DataAccessLayer, HoldingRecord, SelectionModel

log = structlog.get_logger()

_DEFAULT_CONFIG = {
    "min_consecutive_quarters": 2,
    "min_quarterly_reduction_pct": 0.01,
    "min_conviction": 0.50,
    "max_holdings": 30,
}


def _analyze_buybacks(
    quarters: list[dict],
    min_consec: int,
    min_pct: float,
) -> tuple[bool, float, int, float]:
    """
    Examine quarterly fundamentals for a declining share count trend.

    Returns:
        qualifies       — True if trend meets thresholds
        conviction      — 0.0–1.0
        consec_quarters — how many consecutive qualifying quarters
        total_reduction — cumulative fractional reduction in shares
    """
    if not quarters:
        return False, 0.0, 0, 0.0

    # Sort most-recent first; filter to records with valid share counts
    valid = [q for q in quarters if q.get("shares_outstanding", 0) > 0]
    valid.sort(key=lambda q: q.get("filing_date", ""), reverse=True)

    if len(valid) < min_consec + 1:
        return False, 0.0, 0, 0.0

    # Walk back through consecutive quarters
    consec = 0
    cumulative_reduction = 1.0

    for i in range(len(valid) - 1):
        newer = valid[i]["shares_outstanding"]
        older = valid[i + 1]["shares_outstanding"]
        if older <= 0:
            break

        reduction_pct = (older - newer) / older  # positive = fewer shares this quarter
        if reduction_pct >= min_pct:
            consec += 1
            cumulative_reduction *= (1.0 - reduction_pct)
        else:
            break

    if consec < min_consec:
        return False, 0.0, consec, 0.0

    total_reduction = 1.0 - cumulative_reduction  # e.g. 0.05 = 5% total reduction

    # Conviction formula:
    # Base 0.5 for meeting minimum threshold
    # +0.05 per additional quarter beyond minimum (up to +0.20)
    # +0.15 for magnitude: total_reduction / 0.10 (10% total = max bonus)
    base = 0.50
    extra_quarters = min(consec - min_consec, 4) * 0.05
    magnitude_bonus = min(total_reduction / 0.10, 1.0) * 0.15
    conviction = round(min(base + extra_quarters + magnitude_bonus, 1.0), 3)

    return True, conviction, consec, total_reduction


class BuybackModel(SelectionModel):
    def run(self, config: dict, dal: DataAccessLayer) -> list[HoldingRecord]:
        cfg = {**_DEFAULT_CONFIG, **config}
        eval_date = cfg.get("eval_date", "")
        min_consec = cfg["min_consecutive_quarters"]
        min_pct = cfg["min_quarterly_reduction_pct"]
        min_conviction = cfg["min_conviction"]
        max_holdings = cfg["max_holdings"]

        tickers = dal.get_all_tickers()
        log.info("buyback.running", universe=len(tickers))

        results: list[HoldingRecord] = []

        for ticker in tickers:
            quarters = dal.get_fundamentals(ticker)
            if not quarters:
                continue

            try:
                qualifies, conviction, consec, total_reduction = _analyze_buybacks(
                    quarters, min_consec, min_pct
                )
            except Exception as exc:
                log.debug("buyback.analysis_error", ticker=ticker, error=str(exc))
                continue

            if not qualifies or conviction < min_conviction:
                continue

            latest = quarters[0]
            rationale = (
                f"{consec} consecutive quarters of share reduction; "
                f"cumulative reduction {total_reduction:.1%}; "
                f"latest shares {latest.get('shares_outstanding', 0):,.0f}"
            )

            results.append(HoldingRecord(
                model="buyback",
                eval_date=eval_date,
                ticker=ticker,
                conviction=conviction,
                rationale=rationale,
                metadata={
                    "consecutive_quarters": consec,
                    "total_reduction_pct": round(total_reduction * 100, 2),
                    "quarters_analyzed": len(quarters),
                    "latest_shares": latest.get("shares_outstanding"),
                    "latest_period": latest.get("period"),
                },
            ))

        results.sort(key=lambda h: h.conviction, reverse=True)
        result = results[:max_holdings]

        log.info("buyback.complete", candidates=len(results), selected=len(result))
        return result
