"""
src/selection/momentum.py

Momentum-based stock selection model.

Strategy:
  - Universe: S&P 500 only
  - Rank all tickers by trailing 12-month return (252 trading days)
  - Rank stability filter: ticker must not have dropped in rank vs. the
    most recent prior run (rank must be same or better)
  - Select top 5 from the passing set

Persistence:
  - Standard HoldingRecord output → data.nosync/models/{eval_date}/momentum.json
  - Full ranking sidecar → data.nosync/models/{eval_date}/momentum_ranks.json
    (used by the next run's stability check)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import structlog

from src.selection.base import DataAccessLayer, HoldingRecord, SelectionModel

log = structlog.get_logger()

_MODELS_DIR = Path("data.nosync/models")
_LOOKBACK = 252  # trading days ≈ 12 months

_DEFAULT_CONFIG = {
    "max_holdings": 5,
}


def _compute_ranks(dal: DataAccessLayer) -> list[dict]:
    """
    Returns a list of dicts sorted by rank (best first):
        [{"ticker": ..., "rank": 1, "return_12m": 0.42, "total": 500}, ...]

    Tickers missing either the most-recent or the _LOOKBACK-th-from-last close are excluded.
    """
    tickers_sp500 = set(dal.get_all_tickers(tier="sp500"))

    # Wide-format: dates as index, tickers as columns
    close_wide = dal.get_bulk_close_prices()

    # Filter columns to S&P 500 universe
    sp500_cols = [c for c in close_wide.columns if c in tickers_sp500]
    close_wide = close_wide[sp500_cols]

    if len(close_wide) < _LOOKBACK + 1:
        log.warning("momentum.insufficient_history", rows=len(close_wide), required=_LOOKBACK + 1)
        return []

    # Keep only tickers where both the last and the _LOOKBACK-th-from-last rows are non-null
    last_row = close_wide.iloc[-1]
    prior_row = close_wide.iloc[-(1 + _LOOKBACK)]
    valid_mask = last_row.notna() & prior_row.notna()
    close_valid = close_wide.loc[:, valid_mask]

    log_returns = np.log(close_valid.iloc[-1] / close_valid.iloc[-(1 + _LOOKBACK)])
    log_returns = log_returns.dropna().sort_values(ascending=False)
    total = len(log_returns)

    log.info("momentum.universe", sp500=len(tickers_sp500), valid=total)

    return [
        {
            "ticker": str(ticker),
            "rank": i + 1,
            "return_12m": round(float(ret), 6),
            "total": total,
        }
        for i, (ticker, ret) in enumerate(log_returns.items())
    ]


def _load_prior_ranks(eval_date: str) -> dict[str, int]:
    """
    Find the most recent momentum_ranks.json before eval_date.
    Returns a dict of {ticker: rank}.
    """
    if not _MODELS_DIR.exists():
        return {}

    prior_dirs = sorted(
        [d for d in _MODELS_DIR.iterdir() if d.is_dir() and d.name < eval_date],
        reverse=True,
    )
    for d in prior_dirs:
        path = d / "momentum_ranks.json"
        if path.exists():
            with open(path) as f:
                records = json.load(f)
            log.info("momentum.prior_ranks_loaded", from_date=d.name, count=len(records))
            return {r["ticker"]: r["rank"] for r in records}

    log.info("momentum.no_prior_ranks")
    return {}


def _save_ranks(ranks: list[dict], eval_date: str) -> None:
    out_dir = _MODELS_DIR / eval_date
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "momentum_ranks.json"
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(ranks, f, indent=2)
    tmp.replace(path)
    log.info("momentum.ranks_saved", path=str(path), count=len(ranks))


class MomentumModel(SelectionModel):
    def run(self, config: dict, dal: DataAccessLayer) -> list[HoldingRecord]:
        cfg = {**_DEFAULT_CONFIG, **config}
        eval_date = cfg.get("eval_date", "")
        max_holdings = cfg["max_holdings"]

        # Step 1: Rank all S&P 500 tickers by 12-month return
        ranks = _compute_ranks(dal)
        log.info("momentum.ranked", total=len(ranks))

        # Step 2: Persist full ranking for future stability checks
        if eval_date:
            _save_ranks(ranks, eval_date)

        # Step 3: Load prior ranks for stability filter
        prior_ranks = _load_prior_ranks(eval_date) if eval_date else {}

        # Step 4: Filter to tickers that didn't drop in rank
        # Tickers with no prior rank (first run, or new to dataset) are allowed through.
        stable = [
            r for r in ranks
            if r["ticker"] not in prior_ranks or r["rank"] <= prior_ranks[r["ticker"]]
        ]
        log.info("momentum.after_stability_filter", stable=len(stable), dropped=len(ranks) - len(stable))

        # Step 5: Take top N and build HoldingRecords
        selected = stable[:max_holdings]
        total = ranks[0]["total"] if ranks else 1

        holdings: list[HoldingRecord] = []
        for r in selected:
            # Conviction: rank 1 → 1.0, rank `total` → approaches 0
            conviction = round(1.0 - (r["rank"] - 1) / max(total - 1, 1), 3)
            prior_rank = prior_ranks.get(r["ticker"])
            rank_change = (
                "new"
                if prior_rank is None
                else ("+" + str(prior_rank - r["rank"]) if prior_rank > r["rank"] else "0")
            )
            rationale = (
                f"12m log return {r['return_12m']*100:.1f}%, "
                f"rank {r['rank']}/{total} "
                f"(prior rank: {prior_rank if prior_rank is not None else 'n/a'}, change: {rank_change})"
            )
            holdings.append(HoldingRecord(
                model="momentum",
                eval_date=eval_date,
                ticker=r["ticker"],
                conviction=conviction,
                rationale=rationale,
                metadata={
                    "return_12m": r["return_12m"],
                    "rank": r["rank"],
                    "total": total,
                    "prior_rank": prior_rank,
                    "rank_change": rank_change,
                },
            ))

        log.info("momentum.complete", selected=len(holdings))
        return holdings
