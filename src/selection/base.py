"""
src/selection/base.py

Base interface and shared data structures for all selection models.
"""

from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import structlog

log = structlog.get_logger()

_PRICES_DIR = Path("data.nosync/prices")
_FUNDAMENTALS_DIR = Path("data.nosync/fundamentals")
_UNIVERSE_FILE = Path("data.nosync/universe/constituents.json")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class HoldingRecord:
    model: str
    eval_date: str
    ticker: str
    conviction: float        # 0.0–1.0
    rationale: str
    metadata: dict = field(default_factory=dict)
    status: str = "hold"     # 'hold' | 'new_buy' | 'sell'

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> HoldingRecord:
        return cls(**d)


# ---------------------------------------------------------------------------
# Data Access Layer
# ---------------------------------------------------------------------------

class DataAccessLayer:
    """
    Provides models with read access to prices, fundamentals, and universe data.
    Loads the most recent max_lookback price files and caches them in memory.
    """

    def __init__(
        self,
        prices_dir: str | Path = _PRICES_DIR,
        fundamentals_dir: str | Path = _FUNDAMENTALS_DIR,
        universe_file: str | Path = _UNIVERSE_FILE,
        max_lookback: int = 280,
    ):
        self._prices_dir = Path(prices_dir)
        self._prices_parquet = self._prices_dir / "prices.parquet"
        self._fundamentals_dir = Path(fundamentals_dir)
        self._universe_file = Path(universe_file)
        self._max_lookback = max_lookback
        self._price_df: pd.DataFrame | None = None
        self._prices_by_ticker: dict[str, pd.DataFrame] = {}
        self._universe_cache: dict | None = None
        self._fundamentals_cache: dict[str, list] = {}
        self._fundamentals_loaded: bool = False

    def _ensure_prices(self) -> pd.DataFrame:
        if self._price_df is not None:
            return self._price_df

        if self._prices_parquet.exists():
            df = pd.read_parquet(self._prices_parquet)
            df["date"] = pd.to_datetime(df["date"])
            # Apply lookback window using the most recent max_lookback unique trading dates
            recent_dates = sorted(df["date"].unique())[-self._max_lookback:]
            df = df[df["date"].isin(recent_dates)]
            source = "parquet"
            source_count = len(recent_dates)
        else:
            files = sorted(self._prices_dir.glob("*.json"))
            files = files[-self._max_lookback:]

            if not files:
                log.warning("dal.no_price_files", dir=str(self._prices_dir))
                self._price_df = pd.DataFrame(columns=["date", "ticker", "open", "high", "low", "close", "volume", "vwap", "ohlc_avg"])
                return self._price_df

            records: list[dict] = []
            for f in files:
                with open(f) as fp:
                    records.extend(json.load(fp))

            df = pd.DataFrame(records)
            df["date"] = pd.to_datetime(df["date"])
            source = "json"
            source_count = len(files)

        df = df.drop_duplicates(subset=["date", "ticker"], keep="last")
        df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
        self._price_df = df
        self._prices_by_ticker = {
            ticker: grp.reset_index(drop=True)
            for ticker, grp in df.groupby("ticker", sort=False)
        }
        log.info("dal.prices_loaded", source=source, dates=source_count, tickers=len(self._prices_by_ticker), rows=len(df))
        return df

    def get_prices(self, ticker: str, lookback_days: int | None = None) -> pd.DataFrame:
        """
        Returns a DataFrame for one ticker: [date, open, high, low, close, volume, vwap, ohlc_avg].
        Indexed by date. Most recent `lookback_days` rows only.
        """
        self._ensure_prices()
        result = self._prices_by_ticker.get(ticker, pd.DataFrame())
        if lookback_days:
            result = result.tail(lookback_days)
        return result.set_index("date")

    def get_bulk_close_prices(self) -> pd.DataFrame:
        """
        Returns a wide-format DataFrame: dates as index, tickers as columns, close prices as values.
        Used by vectorized models (e.g. MomentumModel) to avoid per-ticker loops.
        """
        df = self._ensure_prices()
        return df.pivot(index="date", columns="ticker", values="close")

    def get_spy_prices(self, lookback_days: int | None = None) -> pd.DataFrame:
        return self.get_prices("SPY", lookback_days=lookback_days)

    def _ensure_fundamentals(self) -> None:
        """Bulk-load all fundamentals files into cache in one pass."""
        if self._fundamentals_loaded:
            return
        t0 = time.perf_counter()
        files = list(self._fundamentals_dir.glob("*.json"))
        log.debug("dal.fundamentals_glob", files=len(files), elapsed_ms=round((time.perf_counter() - t0) * 1000))
        for i, f in enumerate(files):
            ticker = f.stem
            if ticker in self._fundamentals_cache:
                continue
            t_file = time.perf_counter()
            log.debug("dal.fundamentals_reading", ticker=ticker, n=i)
            try:
                with open(f) as fp:
                    data = json.load(fp)
                if isinstance(data, list):
                    self._fundamentals_cache[ticker] = sorted(
                        data, key=lambda q: q.get("filing_date", ""), reverse=True
                    )
                else:
                    self._fundamentals_cache[ticker] = []
            except Exception as exc:
                log.warning("dal.fundamentals_read_error", ticker=ticker, error=str(exc))
                self._fundamentals_cache[ticker] = []
            file_ms = round((time.perf_counter() - t_file) * 1000)
            if file_ms > 200:
                log.warning("dal.fundamentals_slow_file", ticker=ticker, elapsed_ms=file_ms)
        self._fundamentals_loaded = True
        log.info("dal.fundamentals_loaded", tickers=len(self._fundamentals_cache), elapsed_s=round(time.perf_counter() - t0, 2))

    def get_fundamentals(self, ticker: str) -> list[dict]:
        """Returns quarterly fundamentals for a ticker, most recent first."""
        if not self._fundamentals_loaded:
            self._ensure_fundamentals()
        return self._fundamentals_cache.get(ticker, [])

    def get_universe(self, tier: str | None = None) -> list[dict]:
        """Returns active universe records, optionally filtered by tier."""
        if self._universe_cache is None:
            if not self._universe_file.exists():
                log.warning("dal.no_universe_file")
                self._universe_cache = {}
            else:
                with open(self._universe_file) as f:
                    self._universe_cache = json.load(f)

        records = [r for r in self._universe_cache.values() if r.get("status") == "active"]
        if tier:
            records = [r for r in records if r.get("tier") == tier]
        return records

    def get_all_tickers(self, tier: str | None = None) -> list[str]:
        return [r["ticker"] for r in self.get_universe(tier=tier)]

    def load_model_output(self, model: str, eval_date: str) -> list[HoldingRecord]:
        """Load a previously saved model holdings list from data.nosync/models/{eval_date}/{model}.json."""
        path = Path("data.nosync/models") / eval_date / f"{model}.json"
        if not path.exists():
            return []
        with open(path) as f:
            return [HoldingRecord.from_dict(r) for r in json.load(f)]

    def save_model_output(self, holdings: list[HoldingRecord], eval_date: str, model: str) -> str:
        """Persist a model's holdings list. Returns the file path."""
        out_dir = Path("data.nosync/models") / eval_date
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{model}.json"
        with open(path, "w") as f:
            json.dump([h.to_dict() for h in holdings], f, indent=2)
        log.info("dal.model_output_saved", model=model, eval_date=eval_date, holdings=len(holdings))
        return str(path)


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------

class SelectionModel(ABC):
    @abstractmethod
    def run(self, config: dict, dal: DataAccessLayer) -> list[HoldingRecord]:
        """
        Run the model. Returns a holdings list — the tickers this model
        believes should be owned right now, each with a conviction score.
        """
        ...
