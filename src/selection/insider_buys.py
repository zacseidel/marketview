"""
src/selection/insider_buys.py

Insider buying model (placeholder).

Will implement:
- Fetch SEC Form 4 filings for open-market purchases by insiders
- Filter for significant purchase size relative to insider's existing holdings
- Score by recency, cluster of multiple insiders buying, and purchase size
- Universe: S&P 500 + S&P 400

Currently a stub. Set enabled: false in config/models.yaml.
"""

from __future__ import annotations

from src.selection.base import DataAccessLayer, HoldingRecord, SelectionModel


class InsiderBuysModel(SelectionModel):
    def run(self, config: dict, dal: DataAccessLayer) -> list[HoldingRecord]:
        return []
