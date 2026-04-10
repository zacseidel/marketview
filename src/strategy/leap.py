"""
src/strategy/leap.py

LEAP call option strategy evaluation.

Phase 3 implementation will:
- Accept a ticker, current price, and options chain data
- Select LEAP call options with 6–24 month expirations
- For each candidate: compute cost vs. stock ownership, effective leverage,
    breakeven price, theta decay rate, delta
- Compare capital efficiency: LEAP premium vs. 100 shares of stock
- Return a StrategyRecord per candidate with full metrics
- Flag high-IV environments where LEAP is expensive relative to expected move

Uses: src/strategy/options_math.py
"""

from __future__ import annotations
