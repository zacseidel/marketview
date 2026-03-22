# Market Tracker — Design Document

**Automated Stock Analysis & Strategy Platform**
*March 2026*

---

## 1. Executive Summary

Market Tracker is a GitHub-hosted, GitHub Actions-driven platform for automated stock analysis, selection model evaluation, investment strategy comparison, and performance tracking. It tracks a universe of approximately 3,000 publicly traded stocks sourced from Polygon.io's grouped daily endpoint, with S&P 500 and S&P 400 constituents as the priority tier. The system uses the Polygon.io free tier as its primary market data source and publishes a GitHub Pages dashboard. The repository is public, creating a transparent and auditable track record of model recommendations.

The core workflow operates on a Tuesday/Friday evaluation cadence. Models run on Monday/Thursday evenings using end-of-day data and produce holdings recommendations. The user reviews recommendations via a markdown decision file (editable on mobile via the GitHub app), approving or rejecting each suggestion. Approved actions execute at the next trading day's OHLC average price. The system tracks three performance layers: each model's theoretical performance, the user's curated portfolio, and the value added (or lost) by the user's filtering of model recommendations.

The system is designed as a modular pipeline with flat-file data storage, idempotent operations, and a work queue for deferred data collection.

---

## 2. Architecture Overview

### 2.1 Pipeline Stages

| Stage | Function | Frequency |
|-------|----------|-----------|
| 1. Universe Management | Maintain ~3,000 stock universe from grouped daily data, S&P index compositions (Wikipedia), and Polygon ticker details | Daily prices; weekly index reconciliation; quarterly fundamentals |
| 2. Stock Selection | Run enabled models to produce holdings recommendations on evaluation days | Tuesday/Friday cadence (models run Mon/Thu evening) |
| 3. Decision Processing | Generate decision markdown, process user approvals, record portfolio transitions | After model run; after user commits decisions |
| 4. Strategy Evaluation | For each held stock, evaluate all investment approaches (stock, covered call, LEAP, diagonal, CSP) | On demand; GitHub Actions handles queue, local run for bulk |
| 5. Tracking & Dashboard | Track model performance, user portfolio, filtering value-add; publish GitHub Pages dashboard | Daily dashboard rebuild; weekly digest |

### 2.2 Technology Stack

- **Language:** Python 3.11+
- **Orchestration:** GitHub Actions
- **Storage:** JSON and CSV flat files (no database)
- **Market Data:** Polygon.io REST API (free tier, 5 calls/min)
- **Fundamentals:** Polygon Stock Financials endpoint (revenue, net income, share count, market cap)
- **Index Composition:** Wikipedia scraping (S&P 500 and S&P 400)
- **Dashboard:** GitHub Pages, served from `docs/`
- **Local/Remote Parity:** All scripts run both in Actions and locally; heavy tasks run locally

### 2.3 Repository Structure

```
market-tracker/
├── config/
│   ├── settings.yaml          # API keys reference, universe rules, strategy parameters
│   ├── models.yaml            # Model registry: enabled flags, parameters, module paths
│   └── watchlist.yaml         # Manual watchlist with conviction scores and notes
├── src/
│   ├── universe/              # Wikipedia scraping, grouped daily ingestion, ticker details, split detection
│   ├── selection/             # Selection models (each implements standard run() → holdings list interface)
│   ├── decisions/             # Decision markdown generation, parsing, transition recording
│   ├── strategy/              # Covered call, LEAP, diagonal, CSP analysis + shared options math
│   ├── collection/            # Polygon API client (rate limiter), work queue, options chain fetcher
│   ├── tracking/              # Model scorecards, user portfolio P&L, filtering value-add analysis
│   └── reports/               # Dashboard page generation (daily, weekly, model comparison, strategy)
├── data/
│   ├── universe/              # ~3,000 stocks with tier tags (sp500, sp400, broad), company info
│   ├── prices/                # Daily OHLCV by date, filtered to universe, split-adjusted
│   ├── fundamentals/          # Per-ticker JSON: quarterly shares, revenue, net income, market cap
│   ├── options/               # Options chain snapshots by ticker and date
│   ├── queue/                 # Pending data collection tasks
│   ├── models/                # Per-evaluation-date model outputs (holdings lists + conviction)
│   ├── decisions/             # User approval/rejection records, transition history
│   ├── positions/             # Active and closed positions with strategy details and P&L
│   └── splits/                # Detected splits and correction status
├── decisions/
│   └── pending/               # Generated markdown decision files for mobile review
├── docs/                      # GitHub Pages static dashboard
└── .github/
    └── workflows/             # GitHub Actions workflow definitions
```

---

## 3. Data Structures

All persistent data is stored as JSON or CSV files. Each record type includes metadata fields (`created_at`, `updated_at`, `source`) for auditability.

### 3.1 Universe

#### 3.1.1 Stock Record

Stocks enter the universe by appearing in the Polygon grouped daily endpoint with valid company information (description and company details available from Ticker Details v3). Stocks are tiered based on S&P index membership.

| Field | Type | Description |
|-------|------|-------------|
| `ticker` | string | Primary identifier (e.g. AAPL) |
| `name` | string | Company name |
| `description` | string | Company description from Polygon Ticker Details |
| `tier` | string | `sp500`, `sp400`, or `broad` |
| `sector` | string | GICS sector |
| `industry` | string | GICS industry |
| `sic_code` | string | SIC code from Polygon |
| `market_cap` | number | Latest market cap in USD |
| `shares_outstanding` | number | Weighted shares outstanding (for buyback tracking) |
| `avg_volume_30d` | number | 30-day average daily volume |
| `added_date` | date | When added to our universe |
| `removed_date` | date \| null | When removed (null if active) |
| `removal_reason` | string \| null | Why removed: `no_company_info`, `delisted`, `manual` |
| `status` | string | `active` or `removed` |
| `last_details_fetch` | date | When Ticker Details was last fetched |
| `last_financials_fetch` | date | When Stock Financials was last fetched |

#### 3.1.2 Three-Tier Universe Model

All tiers receive daily OHLCV price storage and quarterly fundamental updates. The tier determines priority for analysis and rate-limited API calls.

| Tier | Size | Source | Treatment |
|------|------|--------|-----------|
| S&P 500 | ~500 stocks | Wikipedia scrape (weekly) | Highest priority for options data, strategy evaluation, and model analysis |
| S&P 400 | ~400 stocks | Wikipedia scrape (weekly) | Same as S&P 500; mid-cap universe |
| Broad | ~2,100 stocks | Polygon grouped daily, filtered by valid company info in Ticker Details | Daily prices stored; included in model screening; lower priority for options fetches |

#### 3.1.3 Daily Price Record

Sourced from Polygon's grouped daily endpoint (1 API call per day for the entire universe). On ingestion, records are filtered to universe members only. All prices are stored split-adjusted; when a split is detected, the affected ticker's full history is re-downloaded with `adjusted=true`.

| Field | Type | Description |
|-------|------|-------------|
| `date` | date | Trading date |
| `ticker` | string | Stock ticker |
| `open` | number | Opening price |
| `high` | number | Intraday high |
| `low` | number | Intraday low |
| `close` | number | Closing price |
| `volume` | number | Total volume |
| `vwap` | number | Volume-weighted average price |
| `ohlc_avg` | number | Computed: (open + high + low + close) / 4, used as simulated fill price |

#### 3.1.4 Fundamentals Record

Per-ticker quarterly data stored in `data/fundamentals/{ticker}.json`. Sourced from Polygon's Stock Financials endpoint. Used for buyback screening, basic valuation, and future quantitative models.

| Field | Type | Description |
|-------|------|-------------|
| `ticker` | string | Stock ticker |
| `period` | string | Fiscal quarter (e.g. 2026-Q1) |
| `filing_date` | date | When the filing was published |
| `shares_outstanding` | number | Weighted shares outstanding |
| `revenue` | number | Total revenue for the quarter |
| `net_income` | number | Net income for the quarter |
| `market_cap` | number | Market cap at time of fetch |

#### 3.1.5 Split Detection and Correction

When processing a new grouped daily file, the system compares each ticker's close to its previous close. Any single-day move beyond ±40% is flagged as a potential split/reverse split. The system cross-references Polygon's stock splits endpoint to confirm. For confirmed splits, the ticker's full price history is re-downloaded with `adjusted=true` and all historical daily files are updated. `data/splits/` tracks detected splits and correction status.

### 3.2 Selection Models

#### 3.2.1 Model Interface

Every selection model implements the same interface: a `run()` function that accepts a config object and a data access layer, and returns a **holdings list** — the set of tickers the model believes should be owned right now, each with a conviction score. The data access layer reads prices, fundamentals, and universe data from flat files; models never touch file paths directly. Models are registered in `config/models.yaml` with an `enabled` flag, and a single workflow iterates through enabled models on each evaluation run.

#### 3.2.2 Holdings List (Model Output)

On each evaluation day (Monday/Thursday end-of-day), every enabled model produces a holdings list. The system diffs this against the previous evaluation's list to identify transitions. Holdings in both lists carry forward with no action required.

| Field | Type | Description |
|-------|------|-------------|
| `model` | string | Which model produced this list |
| `eval_date` | date | Evaluation date (the Monday or Thursday) |
| `ticker` | string | Stock ticker |
| `conviction` | number | 0.0–1.0 confidence score |
| `rationale` | string | Human-readable explanation |
| `metadata` | object | Model-specific data (e.g. momentum lookback values, quarters of buyback) |
| `status` | string | `hold` (in both lists), `new_buy` (just entered), `sell` (exited) |

#### 3.2.3 Model Registry

Each model is defined in `config/models.yaml`. Adding a new model = write the Python module + add a config entry. No workflow changes needed.

| Model | Input Data | Initial Scope | Future Enhancements |
|-------|-----------|---------------|---------------------|
| Momentum | Daily prices | Price vs. 50/200 DMA, ROC over 1/3/6 month, relative strength vs. SPY | Sector-relative momentum, volume confirmation, mean-reversion filters |
| Buyback | Quarterly fundamentals | Flag tickers with 2+ consecutive quarters of declining share count (≥1%/quarter) | Revenue/income overlay, insider buying correlation |
| Manual Watchlist | User-curated YAML | Ticker list with notes and conviction levels | Tag-based filtering, thesis tracking with expiry dates |
| Composite | Other models' outputs | Meta-model: flags tickers where 2+ models agree, weighted by conviction | Configurable weighting, performance-adjusted conviction |
| Quantitative | Prices + fundamentals | Placeholder — infrastructure only in early phases | Factor models, statistical screens |
| 13F Analysis | SEC EDGAR 13F filings | Placeholder — infrastructure only in early phases | Track top filers, detect new positions, consensus signals |
| Earnings Calendar | Earnings dates + prices | Placeholder — infrastructure only in early phases | Pre-earnings IV flagging, post-earnings drift |

### 3.3 Decision Workflow

#### 3.3.1 Evaluation Cadence

Evaluations happen twice per week. Models run on Monday and Thursday evenings using end-of-day data. Decisions execute on Tuesday and Friday at the OHLC average of that trading day (or next trading day if holiday). Models can also be triggered manually.

#### 3.3.2 Decision Markdown File

After models run, a GitHub Actions workflow generates `decisions/pending/{date}.md` with checkboxes:

```markdown
# Evaluation: 2026-03-19 (Thursday)
Execute: Friday 2026-03-20

## New Buy Recommendations
- [ ] NVDA — momentum (0.85) + buyback (0.72)
- [ ] COST — buyback (0.80)
- [ ] URI — momentum (0.68)

## Current Holdings — Confirm Continue
- [x] AAPL — momentum (0.90)
- [x] MSFT — momentum (0.78)

## Sell Recommendations (uncheck to override and keep)
- [x] META — momentum dropped (0.25)
```

New buys default **unchecked** (user opts in). Current holds default **checked** (carry forward automatically). Sells default **checked** (user confirms or overrides). User reviews on phone via GitHub mobile app, checks/unchecks, commits. A push-triggered Action processes the decisions.

#### 3.3.3 Decision Record

| Field | Type | Description |
|-------|------|-------------|
| `eval_date` | date | Evaluation date (Monday or Thursday) |
| `execution_date` | date | Trade execution date (Tuesday or Friday) |
| `ticker` | string | Stock ticker |
| `action` | string | `buy`, `sell`, or `hold` |
| `recommending_models` | string[] | Which models recommended this action |
| `user_approved` | boolean | Whether the user approved |
| `execution_price` | number \| null | OHLC average on execution date (filled after close) |
| `notes` | string \| null | Optional user notes from the markdown |

### 3.4 Strategy Evaluation

#### 3.4.1 Strategy Record

For each portfolio stock, the system evaluates all five investment strategies. Requires options chain data (~15 API calls per ticker). Runs via GitHub Actions but designed for local pre-computation — if local run already populated data, Actions is a no-op.

| Field | Type | Description |
|-------|------|-------------|
| `eval_id` | string | Unique: `{ticker}_{strategy}_{date}` |
| `ticker` | string | Stock ticker |
| `strategy` | string | `stock`, `covered_call`, `leap`, `diagonal`, `csp` |
| `eval_date` | date | Date of evaluation |
| `entry_price` | number | Stock price at evaluation (OHLC avg) |
| `strategy_params` | object | Strategy-specific parameters (see 3.4.2) |
| `expected_return` | number | Annualized expected return |
| `max_risk` | number | Maximum dollar risk per unit |
| `breakeven` | number | Breakeven price |
| `capital_required` | number | Capital needed per unit |
| `return_on_capital` | number | Expected return / capital required |
| `notes` | string | Additional context |

#### 3.4.2 Strategy Parameters

| Strategy | Key Parameters | Decision Factors |
|----------|---------------|-----------------|
| Own Stock | Entry price, position size, stop loss | Conviction, portfolio concentration, sector exposure |
| Covered Call | Stock entry, call strike, expiration, premium | IV rank, DTE, delta, premium yield vs. upside cap |
| LEAP | Call strike, expiration (6–24 mo), premium, delta | Cost vs. stock, leverage, theta decay, IV level |
| Diagonal Spread | Long LEAP strike/expiry, short call strike/expiry, net debit | LEAP delta vs. short call delta, roll schedule, net theta |
| Cash-Secured Put | Put strike, expiration, premium, cash reserved | Effective purchase price, annualized yield, assignment probability |

### 3.5 Work Queue

#### 3.5.1 Queue Item

The work queue coordinates deferred data needs. When any stage needs data not yet available, it creates a queue item. The processor runs on each Actions invocation, fetches what's ready, marks items complete.

| Field | Type | Description |
|-------|------|-------------|
| `task_id` | string | Unique identifier |
| `task_type` | string | `price_fetch`, `options_chain`, `ticker_details`, `financials`, `split_correction` |
| `ticker` | string | Target ticker |
| `requested_date` | date | Date data is needed for |
| `requested_by` | string | Which pipeline stage created this |
| `priority` | string | `high`, `normal`, `low` |
| `status` | string | `pending`, `ready`, `completed`, `failed`, `expired` |
| `created_at` | datetime | When queued |
| `completed_at` | datetime \| null | When completed |
| `retry_count` | number | Fetch attempts |
| `data_path` | string \| null | Where data was stored |

#### 3.5.2 Common Queue Patterns

| Pattern | Trigger | Queued Task | Downstream Action |
|---------|---------|-------------|-------------------|
| Decision → Execution Price | User approves buy/sell on day T | Fetch OHLC for T+1 (execution day) | Record entry/exit at OHLC average |
| Selection → Options Data | Stock enters portfolio | Fetch options chain (~15 calls/ticker) | All five strategy evaluations run |
| Split Detected | Daily price shows ±40%+ move | Re-download ticker history with adjusted=true | Historical prices corrected in-place |
| New Universe Member | Ticker appears with valid company info | Fetch Ticker Details + Stock Financials | Stock added with full metadata |
| Quarterly Refresh | Quarterly scheduled job | Batch fetch Financials for ~3,000 tickers | Updated share counts, revenue, income; buyback model re-runs |

### 3.6 Positions & Performance Tracking

#### 3.6.1 Position Record

| Field | Type | Description |
|-------|------|-------------|
| `position_id` | string | Unique identifier |
| `ticker` | string | Stock ticker |
| `strategy` | string | Strategy type (tracked separately per strategy) |
| `entry_date` | date | Date position opened (execution day) |
| `entry_price` | number | OHLC average on entry date |
| `entry_details` | object | Strategy-specific info (premiums, strikes, etc.) |
| `status` | string | `open`, `closed`, `expired`, `assigned` |
| `current_value` | number | Latest mark-to-market value |
| `unrealized_pnl` | number | Current unrealized P&L |
| `exit_date` | date \| null | When closed |
| `exit_price` | number \| null | OHLC average on exit date |
| `realized_pnl` | number \| null | Final realized P&L |
| `originating_models` | string[] | Which models recommended this stock |

#### 3.6.2 Three-Layer Performance Tracking

| Layer | What It Measures | How It Works |
|-------|-----------------|-------------|
| Model Performance | How well each model's recs perform if followed blindly | Track theoretical portfolio of each model's full holdings list. Hit rate, avg return, Sharpe, vs. SPY. |
| User Portfolio | How the user's curated portfolio performs | Track only approved positions. Includes strategy-level returns. |
| Filtering Value-Add | Does user filtering add or destroy value? | Compare user's picks vs. each model's full list. Per-model breakdown: did the user's subset outperform the model's full list? |

---

## 4. API Integration

### 4.1 Polygon.io (Massive.com) Endpoints

Free tier: 5 API calls per minute.

| Endpoint | Purpose | Frequency | Rate Consideration |
|----------|---------|-----------|-------------------|
| Grouped Daily Bars | All tickers' OHLCV for a single date | Daily (after close) | 1 call/day for entire universe |
| Ticker Details v3 | Company name, description, sector, SIC, market cap, shares | On new additions; periodic refresh | Per-ticker; ~3,000 for init (local) |
| Stock Financials (vX) | Quarterly: revenue, net income, shares outstanding | Quarterly refresh | Per-ticker; ~3,000/quarter (local) |
| Options Chain | Options contracts with Greeks | On demand (~15 calls/ticker) | Batched across workflow runs |
| Aggregate Bars (ticker) | Individual ticker history with adjusted=true | On split detection; new ticker backfill | Per-ticker; infrequent |
| Stock Splits | Corporate actions for split confirmation | Weekly | Low volume |
| Reference: Tickers | All active tickers with metadata | Weekly | Paginated |

### 4.2 Additional Data Sources

- **Wikipedia** — S&P 500 and S&P 400 index composition, scraped weekly, diffed against current universe
- **SEC EDGAR** — 13F filings (Phase 5)
- **Earnings Calendar** — Polygon, Financial Modeling Prep, or similar (Phase 5)

### 4.3 Rate Limiting Strategy

All calls go through a centralized rate limiter (token bucket, 5/min) with exponential backoff on 429s. Grouped daily (1 call/day) is the foundation. Options chain fetches (~15/ticker) are the primary concern; the queue processor batches across runs, handling ~250 tickers per run (5/min × ~50 min). Priority ordering ensures active signals get data first. Heavy initialization and quarterly refreshes run locally.

### 4.4 Initial Data Population

Local operation before Actions takes over:

1. Scrape Wikipedia for S&P 500/400 composition
2. Fetch Ticker Details for ~3,000 tickers (~10 hours at 5/min)
3. Fetch Stock Financials for ~3,000 tickers (~10 hours)
4. Backfill 2 years of grouped daily prices (~500 calls, ~2 hours)

All scripts are resumable via state file. Once complete, commit and push; Actions handles daily maintenance.

---

## 5. GitHub Actions Workflows

Staggered timing ensures each workflow runs after upstream dependencies complete.

| Workflow | Schedule | What It Does |
|----------|----------|-------------|
| `daily-prices.yml` | Weekdays 6:30 PM ET | Fetch grouped daily, filter to universe, detect splits, commit |
| `process-queue.yml` | Weekdays 7:00 PM ET | Process pending queue tasks (options chains, ticker details, split corrections) |
| `run-models.yml` | Mon/Thu 7:30 PM ET | Run enabled selection models, produce holdings lists, generate decision markdown |
| `process-decisions.yml` | On push to `decisions/pending/` | Parse committed markdown, record transitions, queue execution-day price fetches |
| `record-executions.yml` | Tue/Fri 7:00 PM ET | Fetch execution-day OHLC, record entry/exit prices, update positions |
| `evaluate-strategies.yml` | Weekdays 8:00 PM ET | Evaluate strategies for stocks with pending options data; no-op if already computed |
| `update-positions.yml` | Weekdays 8:30 PM ET | Update open position values, check stop losses, compute unrealized P&L |
| `daily-dashboard.yml` | Weekdays 9:00 PM ET | Regenerate GitHub Pages dashboard |
| `weekly-digest.yml` | Saturday 10:00 AM ET | Generate weekly digest with model performance and filtering analysis |
| `universe-refresh.yml` | Sunday 12:00 PM ET | Scrape Wikipedia for index changes, reconcile universe, queue new additions |

---

## 6. GitHub Pages Dashboard

Static site from `docs/`, regenerated nightly and weekly. Model recommendations are public; user portfolio decisions are not displayed.

### 6.1 Dashboard Landing Page
Daily view, scannable in under two minutes. Market overview (SPY/QQQ, sector heat map), model recommendations with conviction levels, strategy evaluations completed, queue status. Historical pages preserved and navigable.

### 6.2 Weekly Digest Page
Model performance scorecards, strategy comparison, universe changes, look-ahead for earnings and expiring options.

### 6.3 Model Scorecard Page
Per-model drill-down with historical charts. Signal count, hit rate, average return, holding period, Sharpe-like ratio, vs. SPY.

---

## 7. Key Design Principles

### 7.1 Flat File Architecture
No database. JSON and CSV committed to the repo. Free version control, easy debugging, zero infrastructure cost. Prices partitioned by date; fundamentals by ticker; positions and decisions in single files.

### 7.2 Idempotent Operations
Every stage is safely re-runnable. Deterministic IDs, existence checks, and queue status tracking prevent duplicates.

### 7.3 Local/Remote Parity
All scripts run identically on Actions and locally. Heavy tasks run locally; Actions handles daily routines. If local pre-populates data, Actions is a no-op.

### 7.4 Paper Trading with Realistic Fills
Entry/exit at OHLC average of execution day (Tuesday/Friday). Not connected to a brokerage — analytical tool for building model and strategy confidence.

### 7.5 Progressive Enhancement
Placeholder modules slot in without changing data structures or downstream pipeline. Each model produces a holdings list; adding one means writing code + a config entry.

---

## 8. Build Phases

### Phase 1: Foundation (Weeks 1–3)

**Goal:** Data infrastructure, universe population, basic dashboard. Significant local initialization.

| Task | Details | Output |
|------|---------|--------|
| Repository scaffolding | Directory structure, config files, shared utilities, `docs/` | Working repo |
| Polygon API client | 5 calls/min rate limiter, exponential backoff, logging, resumable state | `src/collection/polygon_client.py` |
| Universe initialization | Wikipedia scrape → Ticker Details for ~3,000 (local, ~10 hrs) → tiered universe | `data/universe/constituents.json` |
| Fundamentals initialization | Stock Financials for ~3,000 (local, ~10 hrs) → per-ticker quarterly series | `data/fundamentals/{ticker}.json` |
| Price backfill | 2 years grouped daily (~500 calls, ~2 hrs local), filtered, split-adjusted | `data/prices/{date}.json` |
| Split detection | Anomalous move detection, splits endpoint cross-ref, history correction | `data/splits/` + corrected prices |
| Daily price workflow | GitHub Actions: fetch, filter, detect splits, commit | Automated daily updates |
| Work queue system | Queue structure, processor, lifecycle management, retry logic | `src/collection/queue.py` |
| GitHub Pages dashboard | Static site: gainers/losers, index performance, sector heat, universe stats | `docs/` with deployment |

### Phase 2: Selection Models & Decision Workflow (Weeks 4–5)

**Goal:** First models on the Tuesday/Friday cadence with mobile decision workflow.

| Task | Details | Output |
|------|---------|--------|
| Model interface | Standard `run()` contract, data access layer, config-driven runner | `src/selection/base.py` |
| Momentum model | Holdings list: 50/200 DMA, ROC 1/3/6 month, relative strength vs. SPY | `src/selection/momentum.py` |
| Buyback model | Holdings list: 2+ quarters declining shares (≥1%/quarter) | `src/selection/buyback.py` |
| Manual watchlist model | Reads `config/watchlist.yaml`, produces holdings list | `src/selection/watchlist.py` |
| Composite model | Flags multi-model agreement, conviction-weighted | `src/selection/composite.py` |
| Decision markdown generator | `decisions/pending/{date}.md` with checkboxes | `src/decisions/generate.py` |
| Decision processor | Parse markdown, record transitions, queue execution-day fetches | `src/decisions/process.py` |
| Execution recorder | Fetch OHLC, compute fills, record entries/exits | `src/decisions/execute.py` |
| Workflows | Mon/Thu model runs + markdown; Tue/Fri execution recording | GitHub Actions |
| Enhanced dashboard | Model recommendations, decision history, basic scorecard | Updated `docs/` |

### Phase 3: Strategy Evaluation (Weeks 6–7)

**Goal:** Evaluate all strategies for portfolio stocks. Options-heavy; local pre-computation supported.

| Task | Details | Output |
|------|---------|--------|
| Options data integration | Fetch/store options chains (~15 calls/ticker), parse Greeks/IV | `src/collection/options.py` |
| Stock ownership eval | P&L modeling with stop loss scenarios | `src/strategy/stock.py` |
| Covered call eval | Premium analysis, upside cap, downside protection | `src/strategy/covered_call.py` |
| LEAP eval | Cost vs. stock, leverage, theta decay | `src/strategy/leap.py` |
| Diagonal spread eval | Net debit, roll schedule, theta capture | `src/strategy/diagonal.py` |
| Cash-secured put eval | Yield, effective purchase price, assignment probability | `src/strategy/csp.py` |
| Strategy dashboard | Side-by-side comparison for each portfolio stock | New `docs/` page |

### Phase 4: Position Tracking & Performance (Weeks 8–10)

**Goal:** Full paper trading lifecycle with three-layer performance.

| Task | Details | Output |
|------|---------|--------|
| Position manager | Open/close/update, handle options expiry and assignment | `src/tracking/positions.py` |
| P&L engine | Mark-to-market, realized P&L, including options strategies | `src/tracking/pnl.py` |
| Model scorecard | Each model's theoretical portfolio: hit rate, return, Sharpe, vs. SPY | `src/tracking/model_scorecard.py` |
| User portfolio tracker | Actual curated positions and returns | `src/tracking/portfolio.py` |
| Filtering value-add | User picks vs. model full list; per-model filtering alpha | `src/tracking/filtering.py` |
| Performance dashboard | Scorecards, portfolio, filtering analysis, strategy comparison | Updated `docs/` |

### Phase 5: Advanced Models & Backtest (Weeks 11+)

**Goal:** Sophisticated models and historical validation.

| Task | Details | Output |
|------|---------|--------|
| Earnings calendar | Track upcoming earnings, flag IV opportunities | `src/selection/earnings.py` |
| 13F analysis | Parse SEC EDGAR 13F, track fund positions | `src/selection/thirteen_f.py` |
| Quant screening | Factor-based screening from fundamentals | `src/selection/quant.py` |
| Historical backtest | Run models against 2 years of stored data | `src/backtest/` |
| Quarterly fundamentals refresh | Re-fetch Financials for ~3,000 tickers | Workflow + local script |

---

## 9. Decisions & Open Questions

### 9.1 Resolved Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Polygon tier | Free (5 calls/min) | Sufficient for grouped daily + queued fetches. Heavy init runs locally. |
| Report delivery | GitHub Pages dashboard | Richer than email. Model recs public; portfolio decisions private. |
| Index composition | Wikipedia scraping (weekly) | Free, well-structured tables for S&P 500 and 400. |
| Repo visibility | Public | Transparent model track record. API keys in GitHub Secrets. |
| Repo structure | Single repo (data + Pages) | `docs/` for Pages; data doesn't affect build. |
| Fill price | OHLC average of execution day | Realistic, easy to calculate, all values stored. |
| Universe scope | ~3,000 stocks (three tiers) | S&P 500 + 400 priority; ~2,100 broad with valid company info. |
| Price storage | All ~3,000, split-adjusted, filter on ingestion | Grouped daily gives everything in one call. Discard non-universe. Rewrite on splits. |
| Fundamentals | Shares, revenue, net income, market cap | Quarterly from Polygon Financials. Enables buyback + future quant. |
| Evaluation cadence | Tue/Fri (models Mon/Thu evening) | Twice-weekly. Holdings diffed for transitions. Manual trigger available. |
| Decision interface | Markdown checkboxes via GitHub mobile | Carry-forward defaults. New buys unchecked; holds/sells pre-checked. |
| Model architecture | Independent + composite layer | Each model produces holdings list. Composite flags multi-model agreement. |
| Model scheduling | Single workflow, enable/disable in config | `config/models.yaml` controls everything. |
| Strategy evaluation | Actions primary; local pre-computation | If local run populated data, Actions is a no-op. |
| Performance tracking | Three layers | Model quality, user returns, and per-model filtering alpha. |
| Initial population | Local (~22 hrs), push to GitHub | Ticker details + financials + price backfill. Resumable. |

### 9.2 Open Questions

| Question | Options | When to Decide |
|----------|---------|---------------|
| Options data granularity | Full chain vs. specific strikes/expirations | Phase 3 |
| Position sizing model | Fixed dollar, % of portfolio, Kelly, or conviction-weighted | Phase 4 |
| 13F data source | SEC EDGAR direct vs. aggregator API | Phase 5 |
| Backtest methodology | Walk-forward vs. simple historical | Phase 5 |
| Dashboard tech | Static HTML with minimal JS vs. lightweight framework | Phase 1 |

---

## 10. Getting Started

Phase 1 begins with repository scaffolding and the Polygon API client with rate limiting and resumable state. The first major milestone is universe initialization: scrape Wikipedia, then run the local Ticker Details fetch for ~3,000 tickers. Second is the fundamentals fetch. Third is the two-year price backfill. Fourth is the daily price workflow on GitHub Actions. Once data is flowing, the dashboard comes online. Phase 2 adds models and the decision workflow that make the system actionable.

This plan is a living document. It will be updated as open questions are resolved and the system evolves.
