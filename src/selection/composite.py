"""
src/selection/composite.py

Composite meta-model: flags tickers where 2+ enabled models agree.
Reads other models' output files for the same eval_date from data/models/{eval_date}/.
Runs after all other models in the same workflow execution.
"""

from __future__ import annotations

import structlog

from src.selection.base import DataAccessLayer, HoldingRecord, SelectionModel

log = structlog.get_logger()

_DEFAULT_CONFIG = {
    "min_models_agreeing": 2,
    "conviction_weights": {
        "momentum": 1.0,
        "buyback": 1.0,
        "watchlist": 0.5,
    },
}


class CompositeModel(SelectionModel):
    def run(self, config: dict, dal: DataAccessLayer) -> list[HoldingRecord]:
        cfg = {**_DEFAULT_CONFIG, **config}
        eval_date = cfg.get("eval_date", "")
        min_agreeing = cfg["min_models_agreeing"]
        weights = cfg.get("conviction_weights", {})

        # Load all other models' outputs for this eval date
        source_models = [m for m in weights.keys()]
        per_model: dict[str, list[HoldingRecord]] = {}
        for model_name in source_models:
            holdings = dal.load_model_output(model_name, eval_date)
            if holdings:
                per_model[model_name] = holdings
                log.debug("composite.loaded", model=model_name, holdings=len(holdings))

        if not per_model:
            log.warning("composite.no_model_outputs", eval_date=eval_date)
            return []

        # Build per-ticker view: {ticker: {model: conviction}}
        ticker_models: dict[str, dict[str, float]] = {}
        for model_name, holdings in per_model.items():
            for h in holdings:
                if h.ticker not in ticker_models:
                    ticker_models[h.ticker] = {}
                ticker_models[h.ticker][model_name] = h.conviction

        results: list[HoldingRecord] = []
        for ticker, model_convictions in ticker_models.items():
            agreeing_models = list(model_convictions.keys())
            if len(agreeing_models) < min_agreeing:
                continue

            # Weighted average conviction
            total_weight = sum(weights.get(m, 1.0) for m in agreeing_models)
            weighted_conviction = sum(
                model_convictions[m] * weights.get(m, 1.0)
                for m in agreeing_models
            ) / total_weight if total_weight > 0 else 0.0

            rationale = " + ".join(
                f"{m} ({model_convictions[m]:.2f})" for m in sorted(agreeing_models)
            )

            results.append(HoldingRecord(
                model="composite",
                eval_date=eval_date,
                ticker=ticker,
                conviction=round(weighted_conviction, 3),
                rationale=f"Multi-model agreement: {rationale}",
                metadata={
                    "agreeing_models": agreeing_models,
                    "per_model_conviction": model_convictions,
                    "models_count": len(agreeing_models),
                },
            ))

        results.sort(key=lambda h: (h.metadata.get("models_count", 0), h.conviction), reverse=True)
        log.info("composite.complete", tickers_with_agreement=len(results))
        return results
