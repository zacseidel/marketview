"""
src/selection/thirteen_f.py

13F institutional filing analysis model (placeholder).

Will implement:
- Fetch and parse SEC EDGAR 13F filings for target fund managers
- Track new positions, position increases, position exits
- Generate consensus signals when multiple tracked filers hold the same stock
- Assign conviction based on number of filers and position size changes

Currently a stub. Set enabled: false in config/models.yaml.
"""

from __future__ import annotations

from src.selection.base import DataAccessLayer, HoldingRecord, SelectionModel


class ThirteenFModel(SelectionModel):
    def run(self, config: dict, dal: DataAccessLayer) -> list[HoldingRecord]:
        return []
