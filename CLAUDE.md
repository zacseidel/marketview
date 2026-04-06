# CLAUDE.md — marketview

Developer reference for Claude Code. Covers conventions, data paths, how to run things, and what's built vs. not.

---

## Python Conventions

- **All `src/` files must start with `from __future__ import annotations`** — local Python is 3.9, GitHub Actions uses 3.11. This is mandatory.
- Use `structlog` for all logging: `log = structlog.get_logger()` at module level.
- Use `dataclasses` for structured data; `asdict()` for JSON serialization.
- Entry points use `if __name__ == "__main__"` with `from dotenv import load_dotenv; load_dotenv()`.
- All file I/O uses `pathlib.Path`, never string concatenation.
- Use **log returns** (`math.log(exit/entry)`) as the standard metric everywhere — model scorecards, filtering analysis, quant features, and evaluation all use log returns.

---

## Key Data Paths

```
data.nosync/universe/constituents.json       # ~901 active tickers: {ticker: {status, tier, name, ...}}
data.nosync/prices/{YYYY-MM-DD}.json         # Daily OHLCV for all universe tickers (501 files, 2024-03-19 → 2026-03-18)
data.nosync/fundamentals/{ticker}.json       # Quarterly: {filing_date, shares_outstanding, revenue, net_income, market_cap}
data.nosync/models/{YYYY-MM-DD}/{model}.json # Model holdings outputs per eval date
data.nosync/models/scorecards/{model}.json   # Model theoretical performance scorecards
data.nosync/decisions/{YYYY-MM-DD}.json      # Processed decision records (buy/sell/hold with user_approved)
data.nosync/positions/positions.json         # All positions: open + closed, with P&L fields
data.nosync/positions/portfolio_history.json # Daily portfolio snapshots
data.nosync/positions/filtering_analysis.json# Per-model filtering alpha vs. user positions
data.nosync/queue/pending.json               # Work queue tasks
data.nosync/splits/                          # Split detection results per ticker/date
data.nosync/strategy_observations/           # Strategy snapshot lifecycle + returns.json
data.nosync/quant/raw_prices.parquet         # 12yr yfinance price history for quant research (gitignored)
data.nosync/quant/features.parquet           # Feature matrix: ~1.6M rows (gitignored)
data.nosync/quant/artifacts/{gbm,knn}/       # Trained model artifacts — committed to repo
data.nosync/quant/recent_prices.parquet      # Live price cache for quant inference (gitignored)
config/models.yaml                    # Model registry: enabled flag, module, class, params
config/watchlist.yaml                 # User-curated tickers with conviction and notes
decisions/pending/{YYYY-MM-DD}.md     # Markdown decision files for user review via GitHub mobile
docs/index.html                       # Daily dashboard (GitHub Pages)
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

- `base.py` — `HoldingRecord`, `DataAccessLayer`, `SelectionModel` ABC
- `momentum.py` — `MomentumModel`: top 10 S&P 500 by trailing 252-day log return; rank-stability filter (rank must not drop week-over-week). Saves full ranking sidecar to `data.nosync/models/{date}/momentum_ranks.json`.
- `munger.py` — `MungerModel`: top 100 S&P 500 by market cap; buy if price touched ≤ SMA200 in last 21 days AND currently above EMA15; sell if drops below EMA15.
- `repurchase.py` — `RepurchaseModel`: top 5 by trailing-12-month share buyback % (shares repurchased / shares outstanding); must be above 21-day EMA; sell if drops out of top 5 or below EMA.
- `buyback.py` — `BuybackModel`: 2+ consecutive quarters of declining share count ≥1%/quarter
- `watchlist.py` — `WatchlistModel`: reads `config/watchlist.yaml`
- `composite.py` — `CompositeModel`: flags tickers where 2+ enabled models agree
- `earnings.py` — `EarningsModel`: YoY/QoQ net income growth, revenue growth, acceleration (removed — earnings signals subsumed by `quant_gbm_v3`)
- `quant.py` — `QuantModel`: loads trained artifacts from `data.nosync/quant/artifacts/`, downloads recent prices via yfinance, returns top-N picks with predicted 20d log return. Supports `model: gbm`. Wired in as `quant_gbm`.
- `thirteen_f.py` — stub (enabled: false)
- `runner.py` — Loads enabled models from `config/models.yaml`, runs them, assigns new_buy/hold/sell statuses, calls `generate_decision_file`. `_prev_holdings_tickers` excludes sell-status records to prevent duplicate sell signals.

`DataAccessLayer` methods: `get_prices(ticker, lookback_days)`, `get_spy_prices()`, `get_fundamentals(ticker)`, `get_universe(tier)`, `get_all_tickers(tier)`, `load_model_output(model, eval_date)`, `save_model_output(holdings, eval_date, model)`

### `src/quant_research/`
Standalone research pipeline — runs locally, not wired into GitHub Actions.

- `download.py` — Downloads 12yr daily OHLCV via yfinance for all SP500/SP400 tickers + SPY. Resume-safe. Uses `group_by="ticker"` for yfinance MultiIndex compatibility.
- `features.py` — Builds 15-feature matrix from raw prices. Drops outliers (e.g. `log_ret_5d > 1.0`). Train/val split: last 2 years = val.
- `train.py` — Trains GBM and KNN models; evaluates each on val set; prints comparison table. `CLUSTER_SCORE_THRESHOLD` constant used by both training eval and live inference.
- `evaluate.py` — Shared evaluation: simulates top-N equal-weight portfolio every 20 days; computes hit rate, avg log return, excess vs SPY, annualized Sharpe. Supports `min_score_threshold` for filter-style models.

### `src/decisions/`
- `generate.py` — Generates `decisions/pending/{date}.md` with checkboxes. New buys unchecked (opt-in), holds/sells pre-checked.
- `process.py` — Parses the markdown, writes `data.nosync/decisions/{date}.json`, queues execution-day price fetches
- `execute.py` — Records fills (OHLC avg) for approved decisions; opens/closes positions

### `src/strategy/`
- `templates.py` — Strategy evaluation templates
- `snapshot.py` — Strategy observation lifecycle: open → closed; `check_expirations`, `reopen_expired_strategies`
- `returns.py` — Aggregates log returns across closed observations → `data.nosync/strategy_observations/returns.json`
- `runner.py` — Evaluates all strategies for a position
- `stock.py`, `covered_call.py`, `leap.py`, `diagonal.py`, `csp.py` — Per-strategy evaluation logic
- `options_math.py` — Shared Black-Scholes utilities

### `src/tracking/`
- `pnl.py` — Mark-to-market open positions using latest close prices; updates `unrealized_pnl`, `current_value` in positions.json
- `positions.py` — `open_position()`, `close_position()`, `get_open_positions()`, `get_closed_positions()`; position_id = `{ticker}_{strategy}_{entry_date}`
- `portfolio.py` — `compute_portfolio_performance()`: aggregates positions, appends to `portfolio_history.json`
- `model_scorecard.py` — Replays eval dirs to compute each model's theoretical return series (log returns); stores to `data.nosync/models/scorecards/{model}.json`
- `filtering.py` — Per-model filtering alpha = user avg log return − model avg log return

### `src/reports/`
- `daily.py` — `generate_daily_dashboard()` → `docs/index.html` (dark-mode HTML, no JS dependencies)
- `weekly.py` — `generate_weekly_digest()` → `docs/weekly.md` (markdown, trailing 7 days)

---

## Enabled Models (as of 2026-03-21)

| Model key | Class | Strategy | Val Sharpe |
|---|---|---|---|
| `momentum` | `MomentumModel` | Top 10 S&P 500 by 252d log return, rank-stable | — |
| `munger` | `MungerModel` | Top 100 by mkt cap; touched SMA200, above EMA15 | — |
| `repurchase` | `RepurchaseModel` | Top 5 by 12mo buyback %; above 21d EMA | — |
| `buyback` | `BuybackModel` | 2+ consecutive quarters of share count decline | — |
| `watchlist` | `WatchlistModel` | User-curated tickers | — |
| `composite` | `CompositeModel` | 2+ models agree | — |
| `quant_gbm` | `QuantModel` | LightGBM on 15 technical factors | 0.794 |
| `quant_gbm_v3` | `QuantModelV3` | LightGBM v3: 28 features incl. earnings + sector | 1.125 |

Disabled: `thirteen_f` (stub), `buyback` (disabled).

---

## GitHub Actions Workflows

| Workflow | Schedule | Entry Point |
|---|---|---|
| `daily-prices.yml` | Weekdays 6:30 PM ET | `src/universe/ingestion.py` |
| `process-queue.yml` | Weekdays 7:00 PM ET | `src/collection/process_queue.py` |
| `run-models.yml` | Mon/Thu 7:30 PM ET | `src/selection/runner.py` |
| `process-decisions.yml` | On push to `decisions/pending/` | `src/decisions/process.py` |
| `record-executions.yml` | Tue/Fri 7:00 PM ET | `src/decisions/execute.py` |
| `evaluate-strategies.yml` | Weekdays 8:00 PM ET | `src/strategy/runner.py` |
| `update-positions.yml` | Weekdays 8:30 PM ET | `src/tracking/pnl.py` |
| `daily-dashboard.yml` | Weekdays 9:00 PM ET | `src/reports/daily.py` |
| `weekly-digest.yml` | Saturday 10:00 AM ET | `src/tracking/model_scorecard.py` → `src/reports/weekly.py` |
| `universe-refresh.yml` | Sunday 12:00 PM ET | `src/universe/reconcile.py` |
| `quarterly-fundamentals.yml` | Quarterly | `src/collection/fundamentals.py` |

---

## Normal Workflow

### Automated (Mon/Thu cycle)
1. **6:30 PM** — prices downloaded
2. **7:30 PM** — all models run, `decisions/pending/YYYY-MM-DD.md` created
3. **Review on GitHub mobile** — new buys are unchecked (opt-in); holds/sells pre-checked
4. **Check boxes for buys you want, push** — triggers `process-decisions.yml`
5. **Tue/Fri 7:00 PM** — fills recorded from that day's OHLC prices
6. **9:00 PM** — dashboard updated at GitHub Pages

### Running Things Locally

```bash
# Process the work queue (split corrections, price fetches)
python -m src.collection.process_queue

# Fetch fundamentals for entire universe (slow — 5 calls/min, ~3 hrs, resumable)
python -m src.collection.fundamentals

# Run selection models for today
python -m src.selection.runner

# Generate decision file for a specific date
python -m src.decisions.generate 2026-03-20

# Update position marks to market
python -m src.tracking.pnl

# Update model scorecards
python -m src.tracking.model_scorecard

# Generate weekly digest
python -m src.reports.weekly

# Regenerate daily dashboard
python -m src.reports.daily
```

---

## Data Status (as of 2026-03-21)

| Layer | Status |
|---|---|
| Universe | 901 active tickers (S&P 500 + 400) |
| Prices | 501 daily files, 2024-03-19 → 2026-03-18, ~901 tickers/day |
| Fundamentals | Bulk fetch run for S&P 400 only (~360 files). S&P 500 not yet fetched. |
| Quant artifacts | GBM and KNN trained and deployed; cluster dropped (no val-set edge) |
| Scorecards | Populated for enabled models |

---

## Quant Research Pipeline

One-time research pipeline — trains GBM and KNN on 12 years of price history, validates on the most recent 2 years. Must be run locally. Output lands in `data.nosync/quant/` (gitignored except `artifacts/`).

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

### Step 2 — Build feature matrix (~5 min)

```bash
python -m src.quant_research.features
# Output: data.nosync/quant/features.parquet
# ~1.6M rows: (ticker, date, 15 features, fwd_log_ret_20d, split)
# split=train: first 10 years; split=val: last 2 years
```

### Step 3 — Train and evaluate (~20–60 min depending on hardware)

```bash
python -m src.quant_research.train --model gbm      # recommended primary model
python -m src.quant_research.train --model knn      # secondary model
# Artifacts saved to data.nosync/quant/artifacts/{gbm,knn}/
```

### Validation results (last run: 2026-03-21)

| Model | Periods | AvgRet | Excess vs SPY | HitRate | Sharpe |
|---|---|---|---|---|---|
| GBM | 26 | +2.50% | +1.46% | 57.1% | **0.794** |
| KNN | 26 | +1.69% | +0.64% | 56.7% | 0.373 |

Cluster model was evaluated and dropped — no edge in the 2024–2026 validation period despite strong training stats. The top clusters (23, 12, 13) captured high-vol crash/recovery stocks; pattern did not generalize.

### Retraining

Re-run Steps 1–3 as more data accumulates. Step 1 is resume-safe. Steps 2–3 rebuild from scratch.

### Feature vector (15 features)

| Feature | Formula | Form |
|---|---|---|
| `log_price` | `log(close)` | absolute (intentional exception — size/growth proxy) |
| `pct_sma10/50/200` | `(price/SMAn - 1) × 100` | % |
| `pct_ath` | `(price/ATH - 1) × 100` | % (negative = below ATH) |
| `pct_time_since_ath` | `(days_since_ath / 1260) × 100` | % of 5yr window elapsed |
| `pct_52w_low` | `(price/52w-low - 1) × 100` | % |
| `log_ret_5/20/60/126/252/756d` | `log(price / price_Nd_ago)` | log return |
| `vol_20d / vol_60d` | `std(daily log returns) × √252` | annualized vol |

Target: `fwd_log_ret_20d` — 20-day forward log return.

GBM top features by importance: `pct_time_since_ath`, `log_price`, `vol_60d`, `log_ret_756d`, `log_ret_252d`.

### Exploration

```bash
jupyter notebook notebooks/gbm_walkthrough.ipynb
```

Walks through raw data → features → model → evaluation → live picks with inline charts and explanations at each step.

---

## Adding a New Selection Model

1. Create `src/selection/{name}.py` implementing `SelectionModel`
2. Add an entry to `config/models.yaml` with `enabled: true/false`, `module`, `class`, `params`
3. That's it — `runner.py` picks it up automatically

## Adding a New Workflow Step

All workflows use `POLYGON_API_KEY: ${{ secrets.POLYGON_API_KEY }}` as env. Follow the pattern in existing yml files.
