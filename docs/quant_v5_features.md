# Quant Model v5 — Feature Reference

XGBoost regression trained on Thursday-only rows, predicting raw `fwd_log_ret_5d`
(Thursday EOD → following Thursday close). 45 numeric features + 1 categorical = 46 total.

---

## Model Architecture

| Property | Value |
|---|---|
| Algorithm | XGBoost (`XGBRegressor`) |
| Target | `fwd_log_ret_5d` — raw 5-day log return (Thursday → Thursday) |
| Training cadence | Thursday-only rows (non-overlapping 5-day windows, no target leakage) |
| Inference cadence | Friday only (uses prior Thursday EOD prices) |
| Hyperparameters | n_estimators=600, max_depth=5, lr=0.03, subsample=0.8, colsample=0.7, min_child_weight=20, reg_alpha=0.1, reg_lambda=1.0 |
| NaN handling | XGBoost native (learns default split direction per feature — no imputation) |
| Sector encoding | `pd.CategoricalDtype` with fixed categories saved as artifact |
| Artifacts | `data.nosync/quant/artifacts/gbm_v5/model.pkl`, `sector_categories.json` |

---

## Val Performance (last trained)

| Metric | Value | Notes |
|---|---|---|
| ICIR | **2.792** | Primary cross-model comparable — IC / IC.std() × √(periods/yr) |
| IC (mean) | 0.0612 | Spearman rank corr: predicted score vs actual returns, per Thursday |
| Decile Hit Rate | 31.6% | Fraction of top-20 picks landing in actual top 10% of returns |
| Sharpe | 4.369 | Excess-over-universe, annualized (5d cadence, 52 periods/yr) |
| Excess/Period | +1.96% | Avg return of top-20 picks minus equal-weight universe avg |
| Val periods | 95 Thursdays | ~2 years |

---

## Feature Set

### Trend — SMA Position (3 features)

| Feature | Formula |
|---|---|
| `pct_sma10` | `(close / SMA10 − 1) × 100` |
| `pct_sma50` | `(close / SMA50 − 1) × 100` |
| `pct_sma200` | `(close / SMA200 − 1) × 100` |

### Trend — Regression Slope & Fit (6 features)

Vectorized rolling linear regression of log-price over N-day windows (centered x-axis for stability).
`slope_Nd` is in log-return units (directly comparable to `log_ret_Nd`).

| Feature | Formula |
|---|---|
| `slope_10d` | `raw_slope × 10` — 10-day regression-estimated trend |
| `slope_50d` | `raw_slope × 50` |
| `slope_200d` | `raw_slope × 200` |
| `r2_10d` | R² of 10-day log-price regression [0, 1] |
| `r2_50d` | R² of 50-day log-price regression |
| `r2_200d` | R² of 200-day log-price regression |

### Momentum — Log Returns (6 features)

| Feature | Formula |
|---|---|
| `log_ret_5d` | `log(close_t / close_{t−5})` |
| `log_ret_20d` | `log(close_t / close_{t−20})` |
| `log_ret_60d` | `log(close_t / close_{t−60})` |
| `log_ret_126d` | `log(close_t / close_{t−126})` — ~6 months |
| `log_ret_252d` | `log(close_t / close_{t−252})` — ~1 year |
| `log_ret_756d` | `log(close_t / close_{t−756})` — ~3 years |

### Momentum — Derived (6 features)

| Feature | Formula | Notes |
|---|---|---|
| `momentum_sharpe` | `log_ret_252d / vol_252d` | Return-to-risk ratio over trailing year |
| `trend_accel` | `log_ret_20d − log_ret_60d / 3` | Recent acceleration vs. 3-month trend |
| `pct_52w_high` | `(close / rolling_252_max − 1) × 100` | Proximity to 52-week high (≤ 0) |
| `pct_52w_low` | `(close / rolling_252_min − 1) × 100` | Distance from 52-week low (≥ 0) |
| `pct_ath` | `(close / expanding_max − 1) × 100` | Distance below all-time high (≤ 0) |
| `pct_time_since_ath` | `(days_since_ath / 1260) × 100` | % of 5-year window elapsed since ATH |

### Volatility (2 features)

| Feature | Formula |
|---|---|
| `vol_20d` | `std(daily log returns, 20d) × √252` — annualized |
| `vol_60d` | `std(daily log returns, 60d) × √252` |

### Liquidity (2 features)

| Feature | Formula | Notes |
|---|---|---|
| `log_dollar_vol_20d` | `log(close × rolling_20d_avg_volume)` | Size-adjusted trading activity |
| `dollar_vol_rel_20d` | `today_dollar_vol / rolling_20d_avg_dollar_vol` | Relative volume vs. recent norm |

### Earnings Timing — v4 Style (2 features)

Short window (0–20 days). Outside the window → `NaN` (XGBoost treats as "no signal").

| Feature | Definition |
|---|---|
| `days_since_earnings` | Days since last earnings event; `NaN` if > 20 |
| `days_until_earnings` | Days until next earnings event; `NaN` if > 20 |

### Market State — SPY (3 features)

Broadcast to all tickers on the same date.

| Feature | Formula |
|---|---|
| `spy_ret_20d` | SPY 20-day log return |
| `spy_vol_20d` | SPY 20-day annualized realized volatility |
| `spy_pct_above_sma200` | `(SPY_close / SPY_SMA200 − 1) × 100` |

### Sector — v4 Style, 126-day lookback (2 features)

Requires ≥ 3 tickers in sector per date; else `NaN`.

| Feature | Formula |
|---|---|
| `sector_ret_126d` | Mean `log_ret_126d` for all tickers in same sector |
| `stock_vs_sector_126d` | `log_ret_126d − sector_ret_126d` |

### Universe Membership (1 feature)

| Feature | Values | Notes |
|---|---|---|
| `in_sp500` | 1 / 0 | S&P 500 vs. S&P 400; from `constituents.json` |

### Earnings Fundamentals (4 features)

Sourced from per-ticker earnings files. Joined as-of-date to the training row — uses the most recent past earnings event.

| Feature | Definition |
|---|---|
| `eps_surprise_pct` | `(actual_EPS − consensus_EPS) / |consensus_EPS| × 100` |
| `earn_ret_5d` | Stock log return in the 5 trading days following the earnings announcement |
| `ni_yoy_growth` | Net income YoY growth (TTM vs. prior-year TTM) |
| `rev_yoy_growth` | Revenue YoY growth (TTM vs. prior-year TTM) |

### Price Level (1 feature)

| Feature | Formula | Notes |
|---|---|---|
| `log_price` | `log(close)` | Proxy for firm size and options availability |

### Buyback (2 features)

Sourced from quarterly fundamentals files. Joined as-of filing date.

| Feature | Formula |
|---|---|
| `buyback_pct_12m` | `(shares_4q_ago − shares_now) / shares_4q_ago × 100` |
| `buyback_pct_1q` | `(shares_1q_ago − shares_now) / shares_1q_ago × 100` |

### Earnings Timing — v3 Style (1 feature)

Broader window (0–180 days), normalized to 0–100 scale. Complements the v4-style binary window.

| Feature | Formula |
|---|---|
| `days_to_next_earnings` | `clip(days_until_next, 0, 180) / 180 × 100` |

### Sector — v3 Style, 20-day lookback (4 features)

Requires ≥ 3 tickers in sector per date; else `NaN`.

| Feature | Formula |
|---|---|
| `sector_ret_20d` | Mean `log_ret_20d` for all tickers in same sector |
| `sector_vs_spy_20d` | `sector_ret_20d − spy_ret_20d` |
| `sector_ret_rank` | Within-sector percentile rank of `log_ret_20d` [0, 1] |
| `sector_size` | `log(count of active tickers in sector)` |

### Categorical (1 feature)

| Feature | Values | Encoding |
|---|---|---|
| `sector` | 11 GICS-style sectors | `pd.CategoricalDtype` with fixed categories; XGBoost `enable_categorical=True` |

Sector mapping uses SIC codes → 11 broad sectors: Technology, Healthcare, Financials, Consumer Discretionary, Consumer Staples, Industrials, Energy, Materials, Utilities, Real Estate, Communication Services.

---

## Feature Selection History

The v5 feature set was arrived at through a controlled comparison experiment (`train_compare.py`). All variants used identical XGBoost hyperparameters, the same `fwd_log_ret_5d` target, and Thursday-only training rows — only the feature set varied.

| Model | Features | ICIR | IC | DclHit | Sharpe |
|---|---|---|---|---|---|
| xgb_v1feat | 15 pure technical | 0.693 | 0.0112 | 23.4% | 0.766 |
| xgb_v3feat | 30 (v1 + buyback + earnings + sector 20d) | 2.439 | 0.0522 | 30.2% | 4.041 |
| xgb_v4feat | 34 (slope/R² + dollar vol + earnings timing + sector 126d) | 0.879 | 0.0177 | 23.1% | 1.239 |
| xgb_v4bfeat | 38 (v4 + earnings fundamentals) | 2.390 | 0.0549 | 31.9% | 4.015 |
| ensemble v3+v4b | post-hoc score average | 2.598 | 0.0615 | 31.8% | 3.986 |
| **xgb_v5feat** | **46 (full union)** | **2.792** | **0.0612** | **31.6%** | **4.369** |

Key findings:
- The earnings fundamental features (`eps_surprise_pct`, `earn_ret_5d`, `ni_yoy_growth`, `rev_yoy_growth`) are the dominant signal source — their absence explains why v4 (ICIR 0.879) underperforms v3 (ICIR 2.439) despite more technical features.
- v3 and v4b have low pick overlap (Jaccard 0.296, rank corr 0.413) despite similar ICIR, indicating partially independent signal paths.
- A joint model (v5) trained on the full feature union outperforms both individual models and the post-hoc ensemble, confirming that a single model can learn cross-feature interactions that score averaging cannot capture.

---

## Data Dependencies

| Source | Path | Provides |
|---|---|---|
| Raw OHLCV prices | `data.nosync/quant/raw_prices.parquet` | All price-derived features |
| Fundamentals | `data.nosync/fundamentals/{ticker}.json` | `buyback_pct_12m`, `buyback_pct_1q` |
| Earnings events | `data.nosync/earnings/{ticker}.json` | All earnings features |
| Next earnings calendar | `data.nosync/earnings/next_dates.json` | `days_until_earnings`, `days_to_next_earnings` |
| Universe constituents | `data.nosync/universe/constituents.json` | `in_sp500`, sector SIC codes |
