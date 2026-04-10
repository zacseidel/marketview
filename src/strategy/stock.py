"""
src/strategy/stock.py

Plain stock ownership strategy evaluation.

Phase 3 implementation will:
- Accept a ticker, entry price, and optional stop-loss level
- Model P&L at various price scenarios (+10%, +20%, -10%, -20%, etc.)
- Compute capital required (per-share or 100-share lot)
- Return a StrategyRecord with expected_return, max_risk, breakeven, capital_required
- No options math required; uses only price data

Used by: evaluate-strategies.yml workflow and src/tracking/pnl.py
"""

from __future__ import annotations
