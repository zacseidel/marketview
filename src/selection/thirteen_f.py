"""
src/selection/thirteen_f.py

13F institutional filing analysis model (Phase 5 placeholder).

Will implement:
- Fetch and parse SEC EDGAR 13F filings for target fund managers
- Track new positions, position increases, position exits
- Generate consensus signals when multiple tracked filers hold the same stock
- Assign conviction based on number of filers and position size changes

Currently a stub. Set enabled: false in config/models.yaml.

Implements: SelectionModel from src/selection/base.py
"""

from __future__ import annotations
