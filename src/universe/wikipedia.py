"""
src/universe/wikipedia.py

Scrapes S&P 500 and S&P 400 constituent lists from Wikipedia.
"""

from __future__ import annotations

import re

import pandas as pd
import structlog

log = structlog.get_logger()

_SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
_SP400_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_400_companies"

_HEADERS = {
    "User-Agent": "marketview/1.0 (github.com/marketview; educational project)"
}


def _clean_ticker(raw: str) -> str:
    """Normalize ticker symbols (Wikipedia sometimes uses dots instead of dashes)."""
    return str(raw).strip().replace(".", "-").upper()


def _scrape_index(url: str, tier: str) -> dict[str, str]:
    """
    Parse the first table on a Wikipedia index page.
    Returns {ticker: tier}.
    """
    log.info("wikipedia.scraping", url=url, tier=tier)
    tables = pd.read_html(url, header=0, storage_options={"User-Agent": _HEADERS["User-Agent"]})

    # The constituent table is always the first table on both pages
    df = tables[0]

    # Column name varies slightly between pages; find the ticker column
    ticker_col = None
    for col in df.columns:
        if re.search(r"symbol|ticker", str(col), re.IGNORECASE):
            ticker_col = col
            break

    if ticker_col is None:
        raise ValueError(f"Could not find ticker column in {url}. Columns: {list(df.columns)}")

    tickers = {_clean_ticker(t): tier for t in df[ticker_col].dropna()}
    log.info("wikipedia.scraped", tier=tier, count=len(tickers))
    return tickers


def scrape_sp500() -> dict[str, str]:
    """Returns {ticker: 'sp500'} for all current S&P 500 constituents."""
    return _scrape_index(_SP500_URL, "sp500")


def scrape_sp400() -> dict[str, str]:
    """Returns {ticker: 'sp400'} for all current S&P 400 constituents."""
    return _scrape_index(_SP400_URL, "sp400")


def get_index_constituents() -> dict[str, str]:
    """
    Merged S&P 500 + S&P 400 constituents.
    S&P 500 takes precedence if a ticker appears in both.
    Returns {ticker: tier}.
    """
    sp400 = scrape_sp400()
    sp500 = scrape_sp500()
    return {**sp400, **sp500}  # sp500 overwrites sp400 on collision


def diff_against_universe(
    scraped: dict[str, str],
    current_universe: dict[str, dict],
) -> tuple[list[str], list[str]]:
    """
    Compare freshly scraped index members against the stored universe.

    Returns:
        added   — tickers in scraped but not in universe (or tier changed to sp500/sp400)
        removed — tickers in universe at sp500/sp400 tier but no longer in scraped
    """
    index_tickers = set(scraped.keys())
    universe_index = {
        t for t, rec in current_universe.items()
        if rec.get("tier") in ("sp500", "sp400") and rec.get("status") == "active"
    }

    added = [t for t in index_tickers if t not in universe_index]
    removed = [t for t in universe_index if t not in index_tickers]

    log.info(
        "wikipedia.diff",
        scraped=len(index_tickers),
        in_universe=len(universe_index),
        added=len(added),
        removed=len(removed),
    )
    return added, removed
