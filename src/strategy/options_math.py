"""
src/strategy/options_math.py

Shared options math utilities used by all strategy evaluation modules.

Phase 3 implementation will provide:
- Black-Scholes pricing for calls and puts
- Delta, gamma, theta, vega calculations
- Implied volatility calculation via Newton-Raphson or bisection
- Annualized return calculation given premium, capital, and DTE
- Breakeven price calculation for various strategy structures
- Assignment probability estimation from delta
- P&L scenarios at expiration for given strike/premium combinations

All functions are pure (no I/O) and operate on numeric inputs.
Used by: covered_call.py, leap.py, diagonal.py, csp.py
"""
