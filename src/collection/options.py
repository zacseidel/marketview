"""
src/collection/options.py

Options chain fetching and storage.

Phase 3 implementation will:
- Accept a ticker and fetch the full options chain via PolygonClient.get_options_chain()
- Options chain = ~15 API calls per ticker (paginated snapshot endpoint)
- Parse each contract: strike, expiration, type (call/put), bid, ask, mid, IV, delta, gamma, theta, vega, OI
- Store results in data/options/{ticker}/{date}.json
- Idempotent: skip if today's chain already exists
- Track fetch cost (number of API calls) for rate limit planning
- Called by the queue processor when task_type='options_chain'

Entry point:
    fetch_options_chain(ticker: str, date: str | None = None) -> str  # returns data path
"""

from __future__ import annotations
