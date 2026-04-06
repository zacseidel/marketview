"""
src/selection/earnings.py

Earnings momentum stock selection model.

Signals used (derived from quarterly fundamentals stored in data.nosync/fundamentals/):
  - Net income growth: consecutive quarters of YoY net income improvement
  - Revenue growth: positive revenue trend over recent quarters
  - Acceleration: whether the most recent quarter's growth rate exceeds the prior quarter's

Conviction scoring (0–1):
  - 0.40  net income growing YoY in most recent quarter (vs. same quarter prior year)
  - 0.20  net income growing QoQ in most recent quarter
  - 0.20  revenue growing YoY in most recent quarter
  - 0.10  both net income and revenue accelerating (growth rate > prior quarter)
  - 0.10  three or more consecutive quarters of positive net income

Requires fundamentals data (run `python -m src.collection.fundamentals` first).
Set enabled: true in config/models.yaml to include in model runs.
"""

from __future__ import annotations

import structlog

from src.selection.base import DataAccessLayer, HoldingRecord, SelectionModel

log = structlog.get_logger()

_DEFAULT_CONFIG = {
    "min_conviction": 0.50,
    "max_holdings": 30,
    "min_quarters": 4,          # minimum quarters needed to score
}


def _score_ticker(ticker: str, dal: DataAccessLayer, config: dict) -> tuple[float, str, dict]:
    """
    Returns (conviction, rationale, metadata).
    conviction == 0.0 means the ticker fails minimum data requirements.
    """
    min_quarters = config.get("min_quarters", 4)
    quarters = dal.get_fundamentals(ticker)

    # Filter to records with both revenue and net_income
    valid = [
        q for q in quarters
        if q.get("net_income") is not None and q.get("revenue") is not None
    ]
    # Sort oldest → newest so we can walk them in order
    valid.sort(key=lambda q: q.get("filing_date", ""))

    if len(valid) < min_quarters:
        return 0.0, "insufficient fundamentals data", {}

    score = 0.0
    meta: dict = {}

    latest = valid[-1]
    prior_qtr = valid[-2]

    # Locate same quarter prior year (roughly 4 quarters back)
    yoy_idx = len(valid) - 5
    yoy = valid[yoy_idx] if yoy_idx >= 0 else None

    net_income_latest = latest["net_income"]
    net_income_prior_qtr = prior_qtr["net_income"]
    net_income_yoy = yoy["net_income"] if yoy else None

    revenue_latest = latest["revenue"]
    revenue_yoy = yoy["revenue"] if yoy else None

    # 1. Net income YoY growth (+0.40)
    ni_yoy_growth: float | None = None
    if net_income_yoy is not None and net_income_yoy != 0:
        ni_yoy_growth = (net_income_latest - net_income_yoy) / abs(net_income_yoy)
        if ni_yoy_growth > 0:
            score += 0.40

    # 2. Net income QoQ growth (+0.20)
    ni_qoq_growth: float | None = None
    if net_income_prior_qtr != 0:
        ni_qoq_growth = (net_income_latest - net_income_prior_qtr) / abs(net_income_prior_qtr)
        if ni_qoq_growth > 0:
            score += 0.20

    # 3. Revenue YoY growth (+0.20)
    rev_yoy_growth: float | None = None
    if revenue_yoy is not None and revenue_yoy != 0:
        rev_yoy_growth = (revenue_latest - revenue_yoy) / abs(revenue_yoy)
        if rev_yoy_growth > 0:
            score += 0.20

    # 4. Acceleration: most recent growth rate > prior quarter growth rate (+0.10)
    if len(valid) >= 3 and ni_qoq_growth is not None:
        prev_qtr = valid[-3]
        if net_income_prior_qtr != 0 and prev_qtr["net_income"] != 0:
            prior_growth = (net_income_prior_qtr - prev_qtr["net_income"]) / abs(prev_qtr["net_income"])
            if ni_qoq_growth > prior_growth:
                score += 0.10

    # 5. Three or more consecutive quarters of positive net income (+0.10)
    consec_positive = 0
    for q in reversed(valid):
        if q["net_income"] and q["net_income"] > 0:
            consec_positive += 1
        else:
            break
    if consec_positive >= 3:
        score += 0.10

    conviction = round(min(score, 1.0), 3)

    rationale_parts = []
    if ni_yoy_growth is not None:
        rationale_parts.append(f"NI YoY {ni_yoy_growth:+.1%}")
    if ni_qoq_growth is not None:
        rationale_parts.append(f"NI QoQ {ni_qoq_growth:+.1%}")
    if rev_yoy_growth is not None:
        rationale_parts.append(f"Rev YoY {rev_yoy_growth:+.1%}")
    rationale_parts.append(f"{consec_positive} consec profitable qtrs")
    rationale = "; ".join(rationale_parts)

    meta = {
        "net_income_latest": net_income_latest,
        "net_income_yoy_growth": round(ni_yoy_growth, 4) if ni_yoy_growth is not None else None,
        "net_income_qoq_growth": round(ni_qoq_growth, 4) if ni_qoq_growth is not None else None,
        "revenue_latest": revenue_latest,
        "revenue_yoy_growth": round(rev_yoy_growth, 4) if rev_yoy_growth is not None else None,
        "consecutive_profitable_quarters": consec_positive,
        "quarters_available": len(valid),
    }

    return conviction, rationale, meta


class EarningsModel(SelectionModel):
    def run(self, config: dict, dal: DataAccessLayer) -> list[HoldingRecord]:
        cfg = {**_DEFAULT_CONFIG, **config}
        eval_date = cfg.get("eval_date", "")
        min_conviction = cfg["min_conviction"]
        max_holdings = cfg["max_holdings"]

        tickers = dal.get_all_tickers()
        log.info("earnings.running", universe=len(tickers))

        scored: list[HoldingRecord] = []

        for ticker in tickers:
            try:
                conviction, rationale, metadata = _score_ticker(ticker, dal, cfg)
            except Exception as exc:
                log.debug("earnings.score_error", ticker=ticker, error=str(exc))
                continue

            if conviction >= min_conviction:
                scored.append(HoldingRecord(
                    model="earnings",
                    eval_date=eval_date,
                    ticker=ticker,
                    conviction=conviction,
                    rationale=rationale,
                    metadata=metadata,
                ))

        scored.sort(key=lambda h: h.conviction, reverse=True)
        result = scored[:max_holdings]

        log.info("earnings.complete", candidates=len(scored), selected=len(result))
        return result
