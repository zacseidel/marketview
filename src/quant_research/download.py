"""
src/quant_research/download.py

One-time download of 12 years of daily OHLC data for all SP500/SP400 tickers
using yfinance. Output: data/quant/raw_prices.parquet

Resume-safe: if the parquet already exists, skips tickers already present.

Usage:
    python -m src.quant_research.download
"""

from __future__ import annotations

import json
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import structlog

log = structlog.get_logger()

_UNIVERSE_FILE = Path("data/universe/constituents.json")
_OUTPUT_FILE = Path("data/quant/raw_prices.parquet")
_YEARS = 12


def _get_universe_tickers() -> list[str]:
    with open(_UNIVERSE_FILE) as f:
        constituents = json.load(f)
    return [
        v["ticker"]
        for v in constituents.values()
        if v.get("status") == "active"
        and v.get("tier") in ("sp500", "sp400")
    ]


def _load_existing() -> set[str]:
    if not _OUTPUT_FILE.exists():
        return set()
    df = pd.read_parquet(_OUTPUT_FILE, columns=["ticker"])
    return set(df["ticker"].unique())


def download(tickers: list[str] | None = None) -> None:
    import yfinance as yf

    _OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    if tickers is None:
        tickers = _get_universe_tickers()
        for benchmark in ("SPY", "QQQ"):
            if benchmark not in tickers:
                tickers = [benchmark] + tickers  # always include benchmarks

    already_done = _load_existing()
    remaining = [t for t in tickers if t not in already_done]

    end_date = date.today()
    start_date = end_date - timedelta(days=int(_YEARS * 365.25))

    log.info(
        "download.starting",
        total=len(tickers),
        already_done=len(already_done),
        remaining=len(remaining),
        start=start_date.isoformat(),
        end=end_date.isoformat(),
    )

    if not remaining:
        log.info("download.complete", msg="All tickers already downloaded")
        return

    # Download in batches to avoid yfinance rate limits
    batch_size = 50
    all_frames: list[pd.DataFrame] = []

    # Load existing data to append to
    if _OUTPUT_FILE.exists():
        all_frames.append(pd.read_parquet(_OUTPUT_FILE))

    for batch_start in range(0, len(remaining), batch_size):
        batch = remaining[batch_start: batch_start + batch_size]
        log.info(
            "download.batch",
            start=batch_start,
            end=batch_start + len(batch),
            total=len(remaining),
        )

        try:
            raw = yf.download(
                batch,
                start=start_date.isoformat(),
                end=end_date.isoformat(),
                auto_adjust=True,
                progress=False,
                group_by="ticker",
            )
        except Exception as exc:
            log.error("download.batch_error", error=str(exc))
            time.sleep(5)
            continue

        # yfinance always returns MultiIndex columns (Price, Ticker) regardless of batch size
        for ticker in batch:
            try:
                df = raw[ticker][["Open", "High", "Low", "Close", "Volume"]].copy()
                df.columns = ["open", "high", "low", "close", "volume"]
                df = df.dropna(subset=["close"])
                df["ticker"] = ticker
                df.index.name = "date"
                df = df.reset_index()
                all_frames.append(df)
            except (KeyError, Exception) as exc:
                log.debug("download.ticker_error", ticker=ticker, error=str(exc))

        # Write checkpoint after each batch
        if all_frames:
            combined = pd.concat(all_frames, ignore_index=True)
            combined["date"] = pd.to_datetime(combined["date"])
            combined = combined.sort_values(["ticker", "date"]).reset_index(drop=True)
            combined.to_parquet(_OUTPUT_FILE, index=False)
            log.info("download.checkpoint", tickers_done=combined["ticker"].nunique())

        time.sleep(1)  # be polite to yfinance

    # Final stats
    if _OUTPUT_FILE.exists():
        final = pd.read_parquet(_OUTPUT_FILE)
        log.info(
            "download.done",
            tickers=final["ticker"].nunique(),
            rows=len(final),
            date_range=f"{final['date'].min().date()} → {final['date'].max().date()}",
        )


if __name__ == "__main__":
    download()
