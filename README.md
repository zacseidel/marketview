# marketview

Automated stock analysis and paper trading platform. Runs entirely on GitHub Actions + GitHub Pages with Polygon.io as the data source. No database, no servers — flat JSON files in the repo.

**Dashboard:** https://zacseidel.github.io/marketview/

---

## What It Does

Models evaluate the S&P 500 + 400 universe twice a week and produce a markdown checklist of buy/hold/sell recommendations. You review on your phone via the GitHub mobile app, check/uncheck items, and commit. The next trading day, approved trades execute at the OHLC average price. Everything is paper-traded and tracked across three performance layers.

**Evaluation cadence:** Models run Monday/Thursday evenings → decision file generated → you review → executes Tuesday/Friday.

---

## Current State

| Component | Status |
|---|---|
| Universe | 901 active tickers (S&P 500 + 400) |
| Price history | 501 trading days, 2024-03-19 → present, ~901 tickers/day, split-adjusted |
| Fundamentals | Not yet bulk-fetched — run `python -m src.collection.fundamentals` (~3 hrs) |
| Selection models | momentum, buyback, watchlist, composite — running; earnings ready but disabled |
| Decision workflow | Fully wired: generate → review → process → execute |
| Strategy evaluation | covered_call, LEAP, diagonal, CSP, stock — wired to Actions |
| Position tracking | P&L mark-to-market, open/close lifecycle — implemented |
| Model scorecards | Implemented — populated automatically each Saturday |
| Weekly digest | Implemented — generates `docs/weekly.md` each Saturday |
| Dashboard | Live at GitHub Pages — universe, market, models, positions, strategy returns |

---

## How the Decision Workflow Works

1. **Monday/Thursday evening** — `run-models.yml` evaluates all enabled models against the full universe and writes `decisions/pending/{date}.md` to the repo:

```
# Evaluation: 2026-03-20 (Thursday)
Execute: Friday 2026-03-21

## New Buy Recommendations
- [ ] NVDA — momentum (0.85) + composite (0.78)
- [ ] COST — buyback (0.80)

## Current Holdings — Confirm Continue
- [x] AAPL — momentum (0.90)

## Sell Recommendations (uncheck to override and keep)
- [x] META — momentum (0.25)
```

2. **You review on GitHub mobile** — check new buys you want, uncheck any sells you want to keep, commit.

3. **Push triggers `process-decisions.yml`** — parses the markdown, records decisions, queues execution-day price fetches.

4. **Tuesday/Friday evening** — `record-executions.yml` fills approved trades at OHLC average, opens/closes positions.

---

## Selection Models

Models are registered in `config/models.yaml`. Enable/disable without touching workflows.

| Model | Signal | Status |
|---|---|---|
| **momentum** | Price vs. SMA50/200, rate-of-change over 1/3/6 months, relative strength vs. SPY | Enabled |
| **buyback** | 2+ consecutive quarters of declining share count (≥1%/quarter) | Enabled |
| **watchlist** | User-curated tickers from `config/watchlist.yaml` | Enabled |
| **composite** | Meta-model: tickers where 2+ models agree, conviction-weighted | Enabled |
| **earnings** | YoY/QoQ net income growth, revenue growth, consecutive profitable quarters | Disabled — needs fundamentals data |
| **quant** | Factor-based screening | Stub |
| **thirteen_f** | SEC 13F institutional position tracking | Stub |

All models implement the same interface: `run(config, dal) -> list[HoldingRecord]`. Adding a model = write a Python file + add a config entry.

---

## Performance Tracking (Three Layers)

| Layer | What | Where |
|---|---|---|
| **Model scorecard** | Each model's theoretical returns if followed blindly — hit rate, avg return | `data/models/scorecards/` |
| **User portfolio** | Actual approved positions + P&L | `data/positions/positions.json` |
| **Filtering alpha** | Did your curation beat the model's full list? | `data/positions/filtering_analysis.json` |

---

## Strategy Evaluation

For each held stock, the system evaluates five approaches using live options chain data:

| Strategy | Description |
|---|---|
| **stock** | Direct ownership with stop-loss scenarios |
| **covered_call** | Stock + short call — premium yield vs. upside cap |
| **leap** | Long-dated call — leverage, theta decay, cost vs. stock |
| **diagonal** | Long LEAP + short near-term call — theta capture |
| **csp** | Cash-secured put — annualized yield, effective purchase price |

---

## GitHub Actions Workflows

| Workflow | When | What |
|---|---|---|
| `daily-prices.yml` | Weekdays 6:30 PM ET | Fetch all OHLCV, detect splits |
| `process-queue.yml` | Weekdays 7:00 PM ET | Process deferred tasks (options chains, split corrections) |
| `run-models.yml` | Mon/Thu 7:30 PM ET | Run models, generate decision markdown |
| `process-decisions.yml` | On push to `decisions/pending/` | Parse approvals, queue fills |
| `record-executions.yml` | Tue/Fri 7:00 PM ET | Record fills, open/close positions |
| `evaluate-strategies.yml` | Weekdays 8:00 PM ET | Strategy evaluation for held stocks |
| `update-positions.yml` | Weekdays 8:30 PM ET | Mark positions to market |
| `daily-dashboard.yml` | Weekdays 9:00 PM ET | Rebuild `docs/index.html` |
| `weekly-digest.yml` | Saturday 10 AM ET | Update scorecards, generate `docs/weekly.md` |
| `universe-refresh.yml` | Sunday 12 PM ET | Reconcile S&P composition changes |
| `quarterly-fundamentals.yml` | Quarterly | Refresh fundamentals for ~900 tickers |

---

## Repository Structure

```
config/
  models.yaml          # Model registry — enable/disable, params
  watchlist.yaml        # Your manually curated tickers
  settings.yaml         # Universe rules, strategy parameters

src/
  collection/           # Polygon API client, rate limiter, work queue, fundamentals
  universe/             # Wikipedia scraping, daily ingestion, split detection, reconciliation
  selection/            # All selection models + runner
  decisions/            # Markdown generation, decision parsing, execution recording
  strategy/             # Strategy evaluation (stock, covered_call, LEAP, diagonal, CSP)
  tracking/             # P&L, positions, model scorecards, portfolio, filtering analysis
  reports/              # daily.py (dashboard) and weekly.py (digest)

data/
  universe/             # constituents.json (~901 tickers)
  prices/               # {date}.json — daily OHLCV per trading day
  fundamentals/         # {ticker}.json — quarterly financials (needs bulk fetch)
  models/               # {date}/{model}.json — model outputs per eval run
  models/scorecards/    # {model}.json — theoretical performance metrics
  decisions/            # {date}.json — processed decision records
  positions/            # positions.json, portfolio_history.json, filtering_analysis.json
  queue/                # pending.json — deferred data tasks
  strategy_observations/# Strategy snapshots + returns.json

decisions/pending/      # Markdown files for mobile review
docs/                   # GitHub Pages — index.html (daily), weekly.md
```

---

## Setup

```bash
pip install -r requirements.txt
pip install -e .
cp .env.example .env     # add POLYGON_API_KEY
```

Required secret in GitHub repository settings: `POLYGON_API_KEY`.

## Running Locally

```bash
python -m src.collection.process_queue      # clear work queue
python -m src.collection.fundamentals       # bulk fetch fundamentals (~3 hrs, resumable)
python -m src.selection.runner              # run all enabled models
python -m src.tracking.pnl                  # mark positions to market
python -m src.tracking.model_scorecard      # update scorecards
python -m src.reports.daily                 # rebuild dashboard
python -m src.reports.weekly                # generate weekly digest
```

---

## Design Principles

- **Flat files only** — JSON committed to the repo. No database, no infrastructure.
- **Idempotent** — every step is safely re-runnable. Deterministic IDs, existence checks.
- **Local/remote parity** — every script runs identically locally and in Actions. Heavy initialization (fundamentals bulk fetch) runs locally; Actions handles daily maintenance.
- **Paper trading** — OHLC average fills. Not connected to a brokerage. The goal is building confidence in models and strategies before committing real capital.
- **Progressive** — new models slot in by writing one Python file and adding a config entry. No workflow changes.

See [DESIGN.md](DESIGN.md) for full architecture detail and data structure specs.
