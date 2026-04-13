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
data.nosync/prices/{YYYY-MM-DD}.json         # Daily OHLCV for all universe tickers (515 files, 2024-03-19 → present)
data.nosync/fundamentals/{ticker}.json       # Quarterly: {filing_date, shares_outstanding, revenue, net_income, market_cap}
data.nosync/models/{YYYY-MM-DD}/{model}.json # Model holdings outputs per eval date
data.nosync/models/scorecards/{model}.json   # Model theoretical performance scorecards
data.nosync/decisions/{YYYY-MM-DD}.json      # Processed decision records (buy/sell/hold with user_approved)
data.nosync/positions/positions.json         # All positions: open + closed, with P&L fields
data.nosync/positions/portfolio_history.json # Daily portfolio snapshots
data.nosync/queue/pending.json               # Work queue tasks
data.nosync/splits/                          # Split detection results per ticker/date
data.nosync/strategy_observations/           # Strategy snapshot lifecycle + returns.json
data.nosync/quant/raw_prices.parquet         # 12yr yfinance price history for quant research (gitignored)
data.nosync/quant/features.parquet           # Feature matrix: ~1.6M rows (gitignored)
data.nosync/quant/features_v4.parquet        # v4 feature matrix: all daily rows, Thursday-filtered in training (gitignored)
data.nosync/quant/artifacts/{gbm,gbm_v3}/    # Trained model artifacts — committed to repo
data.nosync/quant/artifacts/gbm_v4/          # v4 XGBoost artifacts: model.pkl, sector_categories.json (committed)
data.nosync/quant/recent_prices.parquet      # Live price cache for quant inference (gitignored)
config/models.yaml                    # Model registry: enabled flag, module, class, params
config/watchlist.yaml                 # User-curated tickers with conviction and notes
decisions/pending/{YYYY-MM-DD}.md     # Markdown decision files for user review via GitHub mobile
docs/index.html                       # Daily dashboard (GitHub Pages)
docs/weekly.md                        # Weekly digest markdown
notebooks/gbm_walkthrough.ipynb       # Interactive GBM model walkthrough
trades/                               # trades.py local data: accounts.json, positions.json, strategy_evals.json
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
- `runner.py` — Loads enabled models from `config/models.yaml`, runs them, assigns statuses, saves outputs, calls `generate_decision_file`. All models use **time-based exits**: a ticker is held until 10 trading days after the last run that recommended it. Key functions: `_prev_holdings_with_entry` (finds each ticker's last recommendation date by replaying model history), `_assign_statuses_time_based` (resets the clock when model re-picks; generates sell when 10 days elapsed since last rec).

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

### `src/decisions/`
- `generate.py` — Generates `decisions/pending/{date}.md` with checkboxes. All items (new buys, holds, sells) are pre-checked — uncheck to veto a buy or override a sell.
- `process.py` — Parses the markdown, writes `data.nosync/decisions/{date}.json`, queues execution-day price fetches
- `execute.py` — Records fills (OHLC avg) for approved decisions; opens/closes positions. Auto-fetches execution-day prices from Polygon if not cached locally.

### `src/strategy/`
Options strategy evaluation — used by `trades.py` (local CLI), not automated.

- `templates.py` — Strategy evaluation templates
- `snapshot.py` — Strategy observation lifecycle: open → closed
- `returns.py` — Aggregates log returns across closed observations → `data.nosync/strategy_observations/returns.json`
- `runner.py` — Evaluates all strategies for a position
- `stock.py`, `covered_call.py`, `leap.py`, `diagonal.py`, `csp.py` — Per-strategy evaluation logic
- `options_math.py` — Shared Black-Scholes utilities

### `src/tracking/`
- `pnl.py` — Mark-to-market open positions using latest close prices; updates `unrealized_pnl`, `current_value` in positions.json
- `positions.py` — `open_position()`, `close_position()`, `get_open_positions()`, `get_closed_positions()`; position_id = `{ticker}_{strategy}_{entry_date}`
- `portfolio.py` — `compute_portfolio_performance()`: aggregates positions, appends to `portfolio_history.json`
- `model_scorecard.py` — Replays all eval dirs to compute each model's theoretical return series. Opens positions on `new_buy`, closes on `sell`. Stores per-model metrics (hit rate, avg return, SPY alpha, beat-SPY rate) and per-position detail to `data.nosync/models/scorecards/{model}.json`.

### `src/reports/`
- `daily.py` — `generate_daily_dashboard()` → `docs/index.html` (dark-mode HTML, no JS dependencies)
- `weekly.py` — `generate_weekly_digest()` → `docs/weekly.md` (markdown, trailing 7 days)

### Root scripts
- `action.py` — Pre-decision pipeline: prices → queue → earnings → models → scorecards → dashboard. Run Tue/Fri before reviewing decisions, or manually when running the cycle locally.
- `review.py` — Interactive model review: walks through each model's picks, lets you type `?TICKER` for detail, press Enter to accept all or `y` to add exceptions (veto buys / keep sells). Rewrites decision file checkboxes on exit.
- `finish.py` — Post-execution pipeline: process decisions → record fills → update P&L → portfolio history → dashboard. Run after trades execute (Monday/Wednesday), or Tuesday/Friday locally after GitHub Actions fill recording.
- `trades.py` — Standalone options/stock trade tracker CLI. Manages accounts, positions, and options strategy evaluations (covered call, LEAP, CSP, diagonal) with live Polygon pricing. Data stored in `trades/`.

---

## Enabled Models (as of 2026-04-10)

| Model key | Class | Strategy | Val Sharpe | Exit rule |
|---|---|---|---|---|
| `momentum` | `MomentumModel` | Top 5 S&P 500 by 252d log return, rank-stable | — | 10d from last rec |
| `munger` | `MungerModel` | Top 100 by mkt cap; touched SMA200, above EMA15 | — | 10d from last rec |
| `repurchase` | `RepurchaseModel` | Top 5 by 12mo buyback %; above 21d EMA | — | 10d from last rec |
| `watchlist` | `WatchlistModel` | User-curated tickers | — | 10d from last rec |
| `quant_gbm` | `QuantModel` | LightGBM on 15 technical factors; 20d target | 0.794 | 10d from last rec |
| `quant_gbm_v3` | `QuantModelV3` | LightGBM v3: 28 features incl. earnings + sector; 10d target | 1.125 | 10d from last rec |
| `quant_gbm_v4` | `QuantModelV4` | XGBoost v4: 35 features (slope/R², dollar vol, earnings timing, sector 126d); Thursday cross-sectional rank target; 5d window; **Friday only** | — | 10d from last rec |

Disabled: `thirteen_f` (stub), `buyback` (disabled).

**Time-based exit rule (all models):** A ticker stays in the portfolio as long as the model keeps recommending it. Each time it's recommended, `entry_eval_date` is reset to the current eval date. Once the model stops recommending it, it sells 10 trading days after the last recommendation.

---

## GitHub Actions Workflows

| Workflow | Schedule | Entry Point |
|---|---|---|
| `run-models.yml` | Tue/Fri 11:00 AM MT | prices → queue → earnings → models → scorecards → dashboard |
| `process-decisions.yml` | On push to `decisions/pending/` | `src/decisions/process.py` |
| `record-executions.yml` | Tue/Fri 5:00 PM MT | `src/decisions/execute.py` |
| `daily-prices.yml` | Weekdays 11:00 AM MT | `src/universe/ingestion.py` |
| `process-queue.yml` | Weekdays 11:30 AM MT + Tue/Fri 4:00 PM MT | `src/collection/process_queue.py` |
| `update-positions.yml` | Weekdays 1:00 PM MT | `src/tracking/pnl.py` |
| `daily-dashboard.yml` | Weekdays 1:15 PM MT + Tue/Fri 6:00 PM MT | `src/reports/daily.py` |
| `weekly-digest.yml` | Saturday 10:00 AM ET | `src/tracking/model_scorecard.py` → `src/reports/weekly.py` |
| `universe-refresh.yml` | Sunday 12:00 PM ET | `src/universe/reconcile.py` |
| `quarterly-fundamentals.yml` | Quarterly | `src/collection/fundamentals.py` |

---

## Normal Workflow

### Automated (Tue/Fri trade cycle)

The **11:00 AM MT mid-day run** (`run-models.yml`) is the core twice-weekly pipeline:

1. **Tue/Fri 11:00 AM MT** — trade cycle fires automatically:
   - Fetches Mon/Thu EOD prices (skips if already present — safe to pre-run manually)
   - Processes queue (splits, options chains)
   - Runs all models → creates `decisions/pending/YYYY-MM-DD.md`
   - Updates model scorecards
   - Regenerates `docs/index.html` dashboard
2. **Run `python review.py`** — walk through each model, press Enter to accept or `y` to veto specific picks
3. **Push decision file** — triggers `process-decisions.yml`
4. **Next trading day** — place trades at brokerage
5. **Tue/Fri 4:00 PM MT** — process-queue fetches execution-day prices from Polygon
6. **Tue/Fri 5:00 PM MT** — fills recorded from execution-day OHLC prices

Or run the full cycle locally:
```bash
python action.py    # prices + queue + earnings + models + scorecards + dashboard
python review.py    # interactive review → rewrites decision file checkboxes
# commit and push decisions/pending/YYYY-MM-DD.md
python finish.py    # (Monday/Wednesday) fills + P&L + portfolio history + dashboard
```

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

## Data Status (as of 2026-04-10)

| Layer | Status |
|---|---|
| Universe | 901 active tickers (S&P 500 + 400) |
| Prices | 515 daily files, 2024-03-19 → 2026-04-09, ~901 tickers/day |
| Fundamentals | Bulk fetch run for S&P 400 only (~360 files). S&P 500 not yet fetched. |
| Quant artifacts | GBM (quant_gbm) and GBM v3 (quant_gbm_v3) trained and deployed |
| Scorecards | Fresh start 2026-04-10 — accumulating from first live run |
| Positions | Fresh start 2026-04-10 |

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

### Step 2 — Build feature matrix (~5 min)

```bash
python -m src.quant_research.features    # 15-feature set (quant_gbm)
python -m src.quant_research.features_v3 # 28-feature set (quant_gbm_v3)
```

### Step 3 — Train and evaluate

```bash
python -m src.quant_research.train --model gbm   # quant_gbm artifacts
python -m src.quant_research.train_v3            # quant_gbm_v3 artifacts
python -m src.quant_research.features_v4        # v4 feature matrix (~10 min)
python -m src.quant_research.train_v4            # quant_gbm_v4 artifacts (XGBoost)
```

### Validation results (last run: 2026-03-21)

| Model | Periods | AvgRet | Excess vs SPY | HitRate | Sharpe |
|---|---|---|---|---|---|
| GBM (quant_gbm) | 26 | +2.50% | +1.46% | 57.1% | **0.794** |
| GBM v3 (quant_gbm_v3) | — | — | +0.84%/period | — | **1.125** |

### Feature vector — quant_gbm (15 features)

| Feature | Formula |
|---|---|
| `log_price` | `log(close)` |
| `pct_sma10/50/200` | `(price/SMAn - 1) × 100` |
| `pct_ath` | `(price/ATH - 1) × 100` |
| `pct_time_since_ath` | `(days_since_ath / 1260) × 100` |
| `pct_52w_low` | `(price/52w-low - 1) × 100` |
| `log_ret_5/20/60/126/252/756d` | `log(price / price_Nd_ago)` |
| `vol_20d / vol_60d` | `std(daily log returns) × √252` |

quant_gbm_v3 adds: `log_ret_756d`, buyback %, earnings signals (EPS surprise, NI/rev growth, days to next earnings), SPY market state (20d ret, vol, pct above SMA200), sector encoding.

---

## Adding a New Selection Model

1. Create `src/selection/{name}.py` implementing `SelectionModel`
2. Add an entry to `config/models.yaml` with `enabled: true/false`, `module`, `class`, `params`
3. Add `time_based_exit_days: 10` to params (or another window if appropriate)
4. That's it — `runner.py` picks it up automatically

## Adding a New Workflow Step

All workflows use `POLYGON_API_KEY: ${{ secrets.POLYGON_API_KEY }}` as env. Follow the pattern in existing yml files.
