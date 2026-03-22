"""
src/strategy/csp.py

Cash-secured put (CSP) strategy evaluation.

Phase 3 implementation will:
- Accept a ticker, current price, and options chain data
- Select short put strike/expiration based on:
    target delta (configurable, default 0.30)
    DTE range (21–45 days per settings.yaml)
- Compute: premium collected, effective purchase price (strike - premium),
    annualized yield on cash reserved, assignment probability, max loss
- Return a StrategyRecord with full strategy_params and evaluation metrics
- Flag tickers where CSP yield > threshold as attractive entry alternatives to market buy

Uses: src/strategy/options_math.py
"""
