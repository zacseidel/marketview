"""
src/strategy/diagonal.py

Diagonal spread (poor man's covered call) strategy evaluation.

Phase 3 implementation will:
- Accept a ticker, current price, and options chain data
- Construct a diagonal: long LEAP call + short near-term call at higher strike
- Compute: net debit, max profit at short strike, breakeven, theta capture rate,
    roll schedule for the short leg
- Evaluate multiple combinations of long/short strikes and expirations
- Return a StrategyRecord for the highest-scoring combination
- Compare favorably vs. covered call when stock ownership capital is limiting

Uses: src/strategy/options_math.py
"""
