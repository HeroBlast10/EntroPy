# EntroPy — Trading & Cost Layer Dictionary

> Portfolio → Trades → PnL: execution simulation with full cost attribution.

---

## Architecture

```
Daily Weights (from portfolio layer)
    │
    ├── execution.py
    │   ├── generate_trades()     — diff weights → trade list (side, shares, notional)
    │   └── simulate_execution()  — loop over all dates, attach costs
    │
    ├── costs.py
    │   ├── CostModel             — all parameters in one dataclass
    │   ├── estimate_trade_cost() — single-trade cost breakdown
    │   └── estimate_batch_costs()— vectorised over trade DataFrame
    │
    ├── pnl.py
    │   ├── compute_daily_returns()  — gross/net NAV, drawdown
    │   ├── cost_attribution()       — component-level cost breakdown
    │   └── performance_summary()    — Sharpe, Sortino, Calmar, MaxDD, etc.
    │
    └── pipeline.py
        └── run_trading_pipeline()   — orchestrator: load → execute → PnL → save
```

---

## Cost Model (`CostModel`)

All parameters are stored in a single dataclass with US large-cap defaults.

### Commission

| Parameter | Default | Description |
|-----------|---------|-------------|
| `commission_per_share` | $0.005 | Per-share broker fee (IB tiered) |
| `commission_pct` | 0.0 | Alternative: percentage of notional |
| `commission_min` | $1.00 | Minimum commission per order |

If `commission_pct > 0`, it overrides the per-share model.

### Slippage

| Parameter | Default | Description |
|-----------|---------|-------------|
| `slippage_bps` | 5.0 | One-way half-spread in basis points |

Modelled as a fixed fraction of notional. 5 bps ≈ 1¢ on a $20 stock.

### Market Impact (Square-Root Model)

```
impact = impact_coeff × σ_daily × (shares / ADV)^impact_exponent × notional
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `impact_coeff` | 0.1 | Calibration constant |
| `impact_exponent` | 0.5 | 0.5 = square-root (Almgren-Chriss); 1.0 = linear |
| `adv_lookback` | 20 | Trading days for average daily volume |

The square-root model captures the empirical observation that impact grows
sub-linearly with order size — consistent with Kyle (1985) and
Almgren & Chriss (2000).

### Regulatory Fees (US-specific)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `sec_fee_rate` | $8 / $1M | SEC Transaction Fee (sells only) |
| `finra_taf_per_share` | $0.000119 | FINRA Trading Activity Fee (sells only) |

### Stamp Duty (Non-US Placeholder)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `stamp_duty_pct` | 0.0 | Applied on buys only (e.g. 10 bps for UK) |

### Borrow / Financing Cost

| Parameter | Default | Description |
|-----------|---------|-------------|
| `borrow_rate_annual` | 0.5% | Annualised borrow cost for short positions |

Applied daily: `|short_notional| × borrow_rate / 252`.

---

## Execution Simulation

### Trade Generation (`generate_trades()`)

For each date where weights change:

1. Compute `delta_weight = w_new − w_old` per ticker.
2. Determine `side` (buy if delta > 0, sell if delta < 0).
3. Convert to shares: `notional_trade / price`.
4. Attach ADV (20-day trailing mean volume) and daily volatility.

### Fill Model

Trades are filled at the **close price** (decision price = close of signal date).
Slippage and impact are added as explicit cost components rather than
adjusting the fill price, keeping gross vs net returns cleanly separated.

---

## PnL Engine

### Daily Returns

```
gross_ret(t) = Σ_i [ w_i(t) × stock_return_i(t) ]
net_ret(t)   = gross_ret(t) − trading_cost(t) / NAV(t-1) − borrow_cost(t) / NAV(t-1)
```

### NAV Tracking

```
NAV_gross(t) = NAV_gross(t-1) × (1 + gross_ret(t))
NAV_net(t)   = NAV_net(t-1) × (1 + gross_ret(t)) − dollar_cost(t)
```

### Drawdown

```
drawdown(t) = NAV(t) / max(NAV(0..t)) − 1
```

---

## Cost Attribution

`cost_attribution()` produces a table breaking down total cost by component:

| Component | Description |
|-----------|-------------|
| `commission` | Broker fees |
| `slippage` | Bid-ask spread cost |
| `impact` | Market impact (price movement from trade) |
| `sec_fee` | SEC regulatory fee |
| `finra_taf` | FINRA trading activity fee |
| `stamp_duty` | Stamp duty (non-US) |

Each row shows: `total_dollar`, `pct_of_total`, `bps_of_notional`.

---

## Performance Summary

`performance_summary()` computes both **gross** and **net** versions of:

| Metric | Formula |
|--------|---------|
| Annualised Return | `(cumulative)^(252/N) − 1` |
| Annualised Volatility | `daily_std × √252` |
| Sharpe Ratio | `ann_return / ann_vol` |
| Sortino Ratio | `ann_return / downside_vol` |
| Calmar Ratio | `ann_return / |max_drawdown|` |
| Max Drawdown | `min(drawdown series)` |

Plus cost-specific metrics:
- `total_trading_cost_bps` — cumulative trading cost in bps
- `total_borrow_cost_bps` — cumulative borrow cost in bps
- `avg_daily_cost_bps` — average daily cost burden

---

## Output Files

| File | Format | Description |
|------|--------|-------------|
| `data/backtest/trades.parquet` | Parquet | All trades with cost breakdown |
| `data/backtest/daily_pnl.parquet` | Parquet | Daily gross/net returns, NAV, drawdown |
| `data/backtest/cost_attribution.csv` | CSV | Component-level cost summary |
| `data/backtest/performance_summary.csv` | CSV | Headline performance metrics |

---

## CLI Usage

```bash
# Default cost model
python scripts/run_backtest.py

# Custom costs
python scripts/run_backtest.py --slippage-bps 10 --impact-coeff 0.15

# Zero-cost backtest (gross-only analysis)
python scripts/run_backtest.py --slippage-bps 0 --impact-coeff 0 --commission-pct 0

# Specific weights file + higher capital
python scripts/run_backtest.py --weights data/portfolio/weights_quantile_long_only_M.parquet \
                               --capital 5000000

# Sensitivity: high-cost scenario
python scripts/run_backtest.py --slippage-bps 15 --impact-coeff 0.2 --borrow-rate 0.02
```

---

## Cost Sensitivity Analysis

By varying parameters via CLI, you can produce a cost sensitivity table:

```bash
# Low cost (institutional)
python scripts/run_backtest.py --slippage-bps 2 --impact-coeff 0.05

# Medium cost (default)
python scripts/run_backtest.py

# High cost (retail / small cap)
python scripts/run_backtest.py --slippage-bps 15 --impact-coeff 0.20 --commission-per-share 0.01
```

Comparing the `performance_summary.csv` across runs reveals how sensitive
the strategy's edge is to execution quality — a key aspect of
translating paper alpha into live P&L.
