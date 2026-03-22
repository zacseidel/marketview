"""
src/strategy/covered_call.py

Covered call strategy evaluation.

Phase 3 implementation will:
- Accept a ticker, stock entry price, and options chain data
- Select optimal short call strike/expiration based on:
    target delta (configurable, default 0.30)
    DTE range (21–45 days per settings.yaml)
- Compute: premium collected, max profit (premium + upside to strike), breakeven,
    annualized yield, downside protection provided by premium
- Return a StrategyRecord with full strategy_params and evaluation metrics
- Requires options chain data in data/options/{ticker}/

Uses: src/strategy/options_math.py
"""
