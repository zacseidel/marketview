# CLAUDE.md — marketview

Developer reference for Claude Code. Covers conventions, data paths, how to run things, and what's built vs. not.

---

## Python Conventions

- **All `src/` files must start with `from __future__ import annotations`** — local Python is 3.9, GitHub Actions uses 3.11. This is mandatory.
- Use `structlog` for all logging: `log = structlog.get_logger()` at module level.
- Use `dataclasses` for structured data; `asdict()` for JSON serialization.
- Entry points use `if __name__ == "__main__"` with `from dotenv import load_dotenv; load_dotenv()`.
- All file I/O uses `pathlib.Path`, never string concatenation.
- Use **log returns** (`math.log(exit/entry)`) as the standard metric everywhere — model scorecards, quant features, and strategy evaluation all use log returns.

---

## Key Data Paths

```
data.nosync/universe/constituents.json       # ~901 active tickers: {ticker: {status, tier, name, ...}}
data.nosync/prices/{YYYY-MM-DD}.json         # Daily OHLCV for all universe tickers (2024-03-19 → present)
data.nosync/fundamentals/{ticker}.json       # Quarterly: {filing_date, shares_outstanding, revenue, net_income, market_cap}
data.nosync/models/{YYYY-MM-DD}/{model}.json # Model holdings outputs per eval date
data.nosync/queue/pending.json               # Work queue tasks
data.nosync/splits/                          # Split detection results per ticker/date
data.nosync/quant/raw_prices.parquet         # 12yr yfinance price history for quant research (gitignored)
data.nosync/quant/features_v4.parquet        # v4 feature matrix: all daily rows, Thursday-filtered in training (gitignored)
data.nosync/quant/artifacts/gbm_v7/          # v7 XGBoost artifacts: model.pkl, sector_categories.json (committed)
data.nosync/quant/recent_prices.parquet      # Live price cache for quant inference (gitignored)
config/models.yaml                    # Model registry: enabled flag, module, class, params
config/watchlist.yaml                 # User-curated tickers with conviction and notes
docs/index.html                       # Daily dashboard (GitHub Pages)
docs/reports/{YYYY-MM-DD}.html        # Dated dashboard snapshots written each pipeline run
docs/weekly.md                        # Weekly digest markdown
notebooks/gbm_walkthrough.ipynb       # Interactive GBM model walkthrough
```

---

## Module Map

### `src/collection/`
- `polygon_client.py` — PolygonClient with rate limiter, retry, pagination. Methods: `get_grouped_daily`, `get_ticker_details`, `get_stock_financials`, `get_options_chain`, `get_agg_bars`, `get_splits`
- `rate_limiter.py` — Token bucket, 5 calls/min, thread-safe
- `queue.py` — WorkQueue with JSON persistence, dedup, lifecycle (pending → completed/failed)
- `process_queue.py` — Queue processor entry point; handles split_correction, price_fetch, options_chain tasks
- `fundamentals.py` — Fetch + store quarterly financials; `bulk_fetch()` is resumable via `.init_state.json`. Writes `constituents.json` atomically via `.tmp` + `.replace()`.
- `earnings.py` — Per-ticker earnings records (EPS surprise, NI growth, revenue growth, days to next earnings)
- `earnings_refresh.py` — Bulk refresh of earnings data for the universe
- `options.py` — Options chain fetch + storage

### `src/universe/`
- `wikipedia.py` — S&P 500/400 scraper via `pandas.read_html`
- `ingestion.py` — Daily grouped bars ingestion, split detection (±40% single-day move flag)
- `ticker_details.py` — Ticker Details fetch + `bulk_init()` (resumable)
- `splits.py` — Split confirmation against Polygon splits endpoint; price history correction
- `reconcile.py` — Weekly Wikipedia diff → add/remove tickers
- `init.py` — Local init orchestrator: `--step wikipedia|details|prices`

### `src/selection/`
All models implement `SelectionModel.run(config, dal) -> list[HoldingRecord]`.

- `base.py` — `HoldingRecord`, `DataAccessLayer`, `SelectionModel` ABC. `HoldingRecord` fields include `entry_eval_date` (last date the model recommended this ticker — used as the time-based exit clock start).
- `momentum.py` — `MomentumModel`: top 5 S&P 500 by trailing 252-day log return; rank-stability filter (rank must not drop week-over-week). Saves full ranking sidecar to `data.nosync/models/{date}/momentum_ranks.json`.
- `munger.py` — `MungerModel`: top 100 S&P 500 by market cap; buy if price touched ≤ SMA200 in last 21 days AND currently above EMA15.
- `repurchase.py` — `RepurchaseModel`: top 5 by trailing-12-month share buyback % (shares repurchased / shares outstanding); must be above 21-day EMA.
- `buyback.py` — `BuybackModel`: 2+ consecutive quarters of declining share count ≥1%/quarter (disabled)
- `watchlist.py` — `WatchlistModel`: reads `config/watchlist.yaml`
- `quant.py` — `QuantModel`: LightGBM on 15 technical factors; predicts 20d forward log return. Wired in as `quant_gbm`.
- `quant_v3.py` — `QuantModelV3`: LightGBM v3 with 28 features (technical + buyback + earnings + SPY state + sector); predicts 10d forward log return. Wired in as `quant_gbm_v3`.
- `thirteen_f.py` — stub (enabled: false)
- `runner.py` — Loads enabled models from `config/models.yaml`, runs them, assigns statuses, saves outputs. All models use **time-based exits**: a ticker is held until 10 trading days after the last run that recommended it. Key functions: `_prev_holdings_with_entry` (finds each ticker's last recommendation date by replaying model history), `_assign_statuses_time_based` (resets the clock when model re-picks; generates sell when 10 days elapsed since last rec).

`DataAccessLayer` methods: `get_prices(ticker, lookback_days)`, `get_spy_prices()`, `get_fundamentals(ticker)`, `get_universe(tier)`, `get_all_tickers(tier)`, `load_model_output(model, eval_date)`, `save_model_output(holdings, eval_date, model)`

### `src/quant_research/`
Standalone research pipeline — runs locally, not wired into GitHub Actions.

- `download.py` — Downloads 12yr daily OHLCV via yfinance for all SP500/SP400 tickers + SPY. Resume-safe.
- `features.py` — Builds 15-feature matrix from raw prices. Train/val split: last 2 years = val.
- `features_v2.py` — V2 feature set with sector encoding.
- `features_v3.py` — V3 feature set (28 features: v2 + `log_ret_756d`).
- `train.py` — Trains GBM and KNN models on 15-feature set.
- `train_v2.py` — Trains GBM v2 with sector features.
- `train_v3.py` — Trains GBM v3 (28 features, 10d target).
- `features_v4.py` — V4 feature set (35 features: slope/R² regressions, dollar volume, earnings timing, sector 126d, in_sp500). All daily rows output; Thursday filter applied in training.
- `train_v4.py` — Trains XGBoost v4 (Thursday-only, cross-sectional rank-percentile target, 5d forward window). No StandardScaler. Saves `model.pkl` + `sector_categories.json`.
- `evaluate.py` — Shared evaluation: simulates top-N equal-weight portfolio; computes hit rate, avg log return, excess vs SPY, annualized Sharpe.
- `compare.py` — Side-by-side comparison of model versions.

### `src/reports/`
- `daily.py` — `generate_daily_dashboard()` → `docs/index.html` + `docs/reports/{eval_date}.html`. Sections: Universe, Market (SPY/QQQ), Past Reports, Model Overview, Multi-Model Confluence, Model Signals (one card per model). No JS dependencies.
- `weekly.py` — `generate_weekly_digest()` → `docs/weekly.md`. Shows all model holdings for each eval date in the trailing 7 days, with 1W price returns per ticker and a multi-model confluence section.

### `src/decisions/`, `src/tracking/`, `src/strategy/`
These modules exist in the codebase but are **not part of the active pipeline**. They handle portfolio position tracking, decision files, and options strategy evaluation — functionality that was intentionally removed from the pipeline to simplify scope.

### Root scripts
- `action.py` — Full pipeline: prices → queue → earnings → models → dashboard. Run Tue/Fri (or anytime locally). Prints a recommendations summary to the console on completion showing each model's holdings sorted by conviction, with 1W price returns and a confluence section.
- `review.py` — Optional interactive model review: walks through each model's new buys and exits, lets you type `?TICKER` for detail, press Enter to accept all or `y` to add exceptions. Not part of the automated pipeline.
- `finish.py`, `trades.py` — Portfolio tracking and options trade CLI. Not part of the active pipeline.

---

## Enabled Models (as of 2026-04-21)

| Model key | Class | Strategy | Val Sharpe / ICIR | Exit rule |
|---|---|---|---|---|
| `momentum` | `MomentumModel` | Top 5 S&P 500 by 252d log return, rank-stable | — | 10d from last rec |
| `munger` | `MungerModel` | Top 100 by mkt cap; touched SMA200, above EMA15 | — | 10d from last rec |
| `repurchase` | `RepurchaseModel` | Top 5 by 12mo buyback %; above 21d EMA | — | 10d from last rec |
| `watchlist` | `WatchlistModel` | User-curated tickers | — | 10d from last rec |
| `quant_gbm_v7` | `QuantModelV7` | XGBoost v7: 47 features — v6 + ni_qoq_growth + ni_acceleration (±3) + earn_ret_5d_to_20d; 5d raw target; **Tue/Fri** | ICIR 2.983 | 10d from last rec |

Disabled: `thirteen_f` (stub), `buyback`, `quant_gbm` (v1), `quant_gbm_v3`, `quant_gbm_v4`, `quant_gbm_v5`, `quant_gbm_v6` (superseded).

**Time-based exit rule (all models):** A ticker stays in the portfolio as long as the model keeps recommending it. Each time it's recommended, `entry_eval_date` is reset to the current eval date. Once the model stops recommending it, it sells 10 trading days after the last recommendation.

---

## GitHub Actions Workflows

| Workflow | Schedule | Entry Point |
|---|---|---|
| `run-models.yml` | Tue/Fri 11:00 AM MT | prices → queue → earnings → models → dashboard |
| `daily-prices.yml` | Weekdays 11:00 AM MT | `src/universe/ingestion.py` |
| `process-queue.yml` | Weekdays 11:30 AM MT + Tue/Fri 4:00 PM MT | `src/collection/process_queue.py` |
| `daily-dashboard.yml` | Weekdays 1:15 PM MT + Tue/Fri 6:00 PM MT | `src/reports/daily.py` |
| `weekly-digest.yml` | Saturday 10:00 AM ET | `src/reports/weekly.py` |
| `universe-refresh.yml` | Sunday 12:00 PM ET | `src/universe/reconcile.py` |
| `quarterly-fundamentals.yml` | Quarterly | `src/collection/fundamentals.py` |

---

## Normal Workflow

### Automated (Tue/Fri trade cycle)

The **11:00 AM MT run** (`run-models.yml`) is the core twice-weekly pipeline:

1. Fetches prices for the previous trading day
2. Processes queue (splits, price fetches)
3. Refreshes earnings data
4. Runs all models → saves to `data.nosync/models/{date}/`
5. Regenerates `docs/index.html` + `docs/reports/{date}.html`

To run the full cycle locally:
```bash
python action.py
```
Prints a recommendations summary on completion.

### Running Things Locally

```bash
# Run selection models for today
python -m src.selection.runner

# Regenerate daily dashboard
python -m src.reports.daily

# Generate weekly digest
python -m src.reports.weekly

# Process the work queue (split corrections, price fetches)
python -m src.collection.process_queue

# Fetch fundamentals for entire universe (slow — 5 calls/min, ~3 hrs, resumable)
python -m src.collection.fundamentals
```

---

## Data Status (as of 2026-04-21)

| Layer | Status |
|---|---|
| Universe | 901 active tickers (S&P 500 + 400) |
| Prices | ~530 daily files, 2024-03-19 → 2026-04-20, ~901 tickers/day |
| Fundamentals | Bulk fetch run for S&P 400 only (~360 files). S&P 500 not yet fetched. |
| Quant artifacts | v7 (quant_gbm_v7) trained and deployed; v1–v6 disabled/superseded |
| Model history | Live runs since 2026-04-10 |

---

## Quant Research Pipeline

One-time research pipeline — trains GBM models on 12 years of price history, validates on the most recent 2 years. Must be run locally. Output lands in `data.nosync/quant/` (gitignored except `artifacts/`).

### First-time setup

```bash
pip install yfinance scikit-learn lightgbm pyarrow
```

### Step 1 — Download price history (~30 min)

```bash
python -m src.quant_research.download
# Output: data.nosync/quant/raw_prices.parquet
# Resume-safe: re-running skips already-downloaded tickers
```

### Step 2 — Build feature matrix (~10 min)

```bash
python -m src.quant_research.features_v7   # builds on features_v4 + v6 earnings signals
```

### Step 3 — Train

```bash
python -m src.quant_research.train_v7      # saves artifacts to data.nosync/quant/artifacts/gbm_v7/
```

### Validation results — quant_gbm_v7 (last trained: 2026-03)

| Metric | Value |
|---|---|
| ICIR | **2.983** |
| Target | 5d raw return |
| Features | 47 (v6 base + ni_qoq_growth + ni_acceleration ±3 clip + earn_ret_5d_to_20d) |

**v7 XGBoost dtype requirements:** All feature columns must be `float64`; sector must be `pd.Categorical` (not `CategoricalDtype`) for XGBoost 2.x `enable_categorical` support. `days_since_earnings` may arrive as object dtype from v6 features — cast via `pd.to_numeric(..., errors="coerce")` before building DMatrix.

---

## Adding a New Selection Model

1. Create `src/selection/{name}.py` implementing `SelectionModel`
2. Add an entry to `config/models.yaml` with `enabled: true/false`, `module`, `class`, `params`
3. Add `time_based_exit_days: 10` to params (or another window if appropriate)
4. That's it — `runner.py` picks it up automatically

## Adding a New Workflow Step

All workflows use `POLYGON_API_KEY: ${{ secrets.POLYGON_API_KEY }}` as env. Follow the pattern in existing yml files.
