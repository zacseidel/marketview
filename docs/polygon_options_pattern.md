# Polygon Free-Plan Options Chain Pattern

Reference for fetching options data efficiently on Polygon's free tier, and the
strategy templates each contract selection is meant to serve.

---

## Options Strategies

### 1. Stock (baseline)
Own the stock outright. No options legs. Included as a benchmark to isolate
the contribution of each options overlay.

**Capital deployed:** full stock price per share  
**Exit:** sell stock at market price

---

### 2. Covered Call
Own the stock and sell a short-dated call against it. The premium received
reduces cost basis. Upside is capped at the strike if the call is assigned.

**Short call target:** DTE 21–45, delta ~0.20–0.25 (roughly 5% OTM)  
**Capital deployed:** stock price − call premium received  
**Exit:** buy back call at market; sell stock at market  
**Why delta 0.225:** balances premium income against probability of cap — at
~22% delta the call expires worthless ~78% of the time, capturing most of the
premium while leaving meaningful upside.

---

### 3. LEAP OTM Call
Buy a deep out-of-the-money long-dated call. High leverage, defined risk (max
loss = premium paid). Used to express a bullish conviction without full capital
commitment.

**Long call target:** DTE ~450–550 (typically the January expiry ~18 months out),
strike 10% OTM  
**Capital deployed:** call premium only  
**Exit:** sell call at market price

---

### 4. Diagonal Spread
Buy a deep ITM LEAP call (high delta, acts like synthetic stock) and sell a
short-dated OTM call against it — the same short call structure as a covered
call, but capital-efficient because the LEAP replaces stock ownership.

**Long call target (back leg):** same LEAP expiry as above, strike 25% ITM
(75% of stock price) — delta ~0.75–0.85, behaves like owning ~80 shares per
contract  
**Short call target (front leg):** same spec as covered call (DTE 21–45,
~5% OTM delta 0.225)  
**Capital deployed:** long call premium − short call premium received  
**Exit:** sell long call; buy back short call  
**Important:** both legs must share the same LEAP expiry date so pricing is
consistent and the spread can be evaluated as a unit.

---

### 5. Cash-Secured Put (CSP)
Sell a short-dated ATM put while holding cash equal to the strike price as
collateral. If assigned, you acquire the stock at the strike (effectively
buying at a discount equal to the premium). If it expires worthless, you keep
the premium.

**Short put target:** DTE 18–25, ATM (strike closest to current stock price)  
**Capital deployed:** strike price − put premium received (cash reserved)  
**Exit:** buy back put at market (or let expire); release cash collateral  
**Why ATM:** maximises premium relative to capital while still having a
reasonable chance of expiring worthless (~50% at exactly ATM).

---

## Polygon Free-Plan Contract Selection Pattern

### The constraint

Polygon's real-time snapshot endpoint is **not available on the free plan**:

```
GET /v3/snapshot/options/{ticker}  →  403 NOT_AUTHORIZED
```

The free plan includes:
- `/v3/reference/options/contracts` — contract metadata (what actually exists)
- `/v2/aggs/ticker/{optionTicker}/range/1/day/{from}/{to}` — EOD OHLCV per contract

---

### Step 1 — Discover what actually exists (3 reference calls)

Query the reference API per DTE band and contract type. Results come back sorted
by `expiration_date` ascending.

```
GET /v3/reference/options/contracts
  ?underlying_ticker=AAPL
  &contract_type=call
  &expiration_date.gte={eval_date + 21d}
  &expiration_date.lte={eval_date + 45d}
  &expired=false
  &sort=expiration_date
  &order=asc
  &limit=250
```

Run once per DTE band / type needed. For the five strategies above, that is three
calls:

| Call | contract_type | DTE window | Serves |
|------|--------------|------------|--------|
| 1 | `call` | 21–45 DTE | covered_call, diagonal front leg |
| 2 | `call` | 450–550 DTE | leap_otm, diagonal back leg |
| 3 | `put`  | 18–25 DTE | csp |

**Why reference first:** strike increments vary by stock (e.g. $0.50, $1, $2.50,
$5, $10, $25) and are not always predictable. Probing guessed tickers wastes API
calls and silently misses the nearest available strike.

---

### Step 2 — Pick best match per target (no API calls)

```python
def best_match(contracts, target_price):
    # contracts already sorted by expiration_date asc
    nearest_exp = contracts[0]["expiration_date"]
    same_exp = [c for c in contracts if c["expiration_date"] == nearest_exp]
    return min(same_exp, key=lambda c: abs(c["strike_price"] - target_price))
```

**Selection rule: nearest available expiry first, then closest strike to target.**

| Strategy target | target_price |
|----------------|-------------|
| covered_call / diagonal front | `stock_price × 1.05` |
| leap_otm | `stock_price × 1.10` |
| diagonal back | `stock_price × 0.75` |
| csp | `stock_price` |

For the diagonal, pin **both legs to the same LEAP expiry** before selecting
strikes — otherwise the two legs may reference different expirations and can't
be evaluated as a spread.

Deduplicate by contract ticker before fetching prices (the OTM and ITM LEAP
picks may occasionally resolve to the same contract for low-priced stocks).

---

### Step 3 — Fetch EOD close for the specific eval date (3–4 price calls)

```
GET /v2/aggs/ticker/O:AAPL260515C00210000/range/1/day/2026-04-01/2026-04-01
  ?adjusted=true
```

**Always use the explicit eval date — not `/prev`.**

`/prev` returns "yesterday relative to when the HTTP call is made." If a queued
task is processed a day late, `/prev` silently returns the wrong date. The
`/range` endpoint is stable: it always returns data for the exact date requested,
regardless of when you call it.

If the range call returns no results, the data is not yet finalized (task
processed same day before market close). Let the task fail and retry the next
day — do not fall back to the wrong date's price.

---

### Scheduling: queue on eval day, process the next day

EOD options data on Polygon is finalized after market close. The recommended
pattern:

1. **Eval day (e.g. Monday 7:30 PM):** run models, queue options chain tasks
   with `requested_date = eval_date`
2. **Next day (Tuesday 7:00 PM):** queue processor picks up tasks, calls
   `/range/1/day/{eval_date}/{eval_date}` — data is finalized ✓

If the processor runs before close on the same day, the range call returns empty.
Correct behavior: fail the task, retry next run.

---

### API call budget

| Step | Calls per ticker |
|------|-----------------|
| 3× reference queries | 3 |
| 3–4× EOD price fetches | 3–4 |
| **Total** | **~7** |

At Polygon's free-plan rate limit of 5 calls/min: 5 tickers ≈ 7 minutes.

---

### Output format (snapshot-compatible)

Structure each result to match the `/v3/snapshot/options/{ticker}` shape so
downstream contract-selection and pricing code works unchanged:

```python
{
    "details": {
        "ticker": "O:AAPL260515C00210000",
        "contract_type": "call",       # "call" | "put"
        "strike_price": 210.0,
        "expiration_date": "2026-05-15",
    },
    "greeks": None,             # not available on free plan
    "implied_volatility": None,
    "last_quote": {"midpoint": 3.45},  # EOD close price
}
```

Downstream selection code must handle `greeks=None` and fall back to
strike-percentage selection. This is correct for all strategy templates whose
targets are expressed as a percentage of the stock price (which covers all five
strategies above).
