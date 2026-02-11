# EntroPy — Portfolio Construction Dictionary

> Signal → Portfolio: two construction methods with full constraint support.

---

## Architecture

```
Signal (factor DataFrame)
    │
    ├── rebalance.py ──→ Rebalance schedule (D / W / M)
    │
    ├── QuantilePortfolio (baseline)
    │       ├── Rank stocks by signal
    │       ├── Select top/bottom quantile (or top-N)
    │       └── Assign weights (equal / market-cap / signal-prop)
    │
    ├── OptimizedPortfolio (advanced)
    │       ├── Estimate shrunk covariance (Ledoit-Wolf)
    │       ├── Solve max α'w − (λ/2)w'Σw
    │       └── Box + sector + turnover constraints via SLSQP
    │
    ├── constraints.py ──→ Post-processing: stock clip → sector clip → turnover clip
    │
    └── pipeline.py ──→ Orchestrator: load → build → carry-forward → save
```

---

## Method 1: Quantile Portfolio (Baseline)

**File:** `entropy/portfolio/quantile.py`

### Logic
1. On each rebalance date, rank all tradable stocks by the alpha signal.
2. Assign stocks to quantiles (default: quintiles, Q1–Q5).
3. **Long-only:** buy stocks in `long_quantile` (default Q5 = top 20%).
4. **Long-short:** buy Q5, sell Q1.
5. Assign weights per chosen scheme.

### Weighting Schemes

| Scheme | Config Value | Description |
|--------|-------------|-------------|
| **Equal** | `equal` | 1/N across all selected stocks |
| **Market-Cap** | `market_cap` | Weight ∝ market capitalisation |
| **Signal** | `signal` | Weight ∝ |signal strength| |
| **Inverse-Vol** | `inverse_vol` | Weight ∝ 1/σ (risk parity flavour) |

### Top-N Override
Set `top_n=50` to skip quantile logic and simply pick the 50 highest-signal stocks (useful for concentrated portfolios).

---

## Method 2: Optimised Portfolio (Advanced)

**File:** `entropy/portfolio/optimize.py`

### Objective

```
maximise   α'w − (λ/2) w'Σw
```

Where:
- **α** = cross-sectional signal (z-scored factor values)
- **Σ** = Ledoit-Wolf shrunk covariance matrix (configurable lookback & shrinkage)
- **λ** = risk aversion parameter

### Constraints

| Constraint | Type | Description |
|-----------|------|-------------|
| Budget | Equality | Σw = 1 (long-only) or Σw = 0 (dollar-neutral long-short) |
| Box | Bounds | 0 ≤ w_i ≤ max_stock_weight (long-only) |
| Sector | Inequality | Σw_sector ≤ max_sector_weight |
| Turnover | Inequality | Σ|w_new − w_old| / 2 ≤ max_turnover |

### Covariance Estimation
- **Lookback:** 120 trading days (configurable)
- **Shrinkage:** Ledoit-Wolf linear shrinkage toward diagonal target
  - `shrinkage=0.0` → pure sample covariance
  - `shrinkage=1.0` → diagonal (variance-only, single-factor model)
  - Default: `0.5` (balanced)

### Solver
- `scipy.optimize.minimize` with **SLSQP** method
- Fallback: equal weight if optimisation fails (ensures pipeline never breaks)

---

## Constraints Module

**File:** `entropy/portfolio/constraints.py`

Applied as post-processing in this order:

| Step | Function | Description |
|------|----------|-------------|
| 1 | `clip_stock_weight()` | Cap individual stock weight at `max_stock_weight` |
| 2 | `clip_sector_weight()` | Scale down sector if aggregate exceeds `max_sector_weight` |
| 3 | `clip_turnover()` | Blend toward previous weights if turnover > `max_turnover` |
| 4 | `normalise_weights()` | Re-normalise to sum=1 (long-only) or balanced (long-short) |

---

## Rebalance Schedule

**File:** `entropy/portfolio/rebalance.py`

| Frequency | Config | Logic |
|-----------|--------|-------|
| **Daily** | `D` | Every NYSE trading day |
| **Weekly** | `W` | Last trading day of each ISO week |
| **Monthly** | `M` | Last trading day of each calendar month |

Between rebalance dates, weights are **carried forward** unchanged (`carry_forward_weights()`).

---

## Portfolio Modes

### Long-Only
- All weights ≥ 0, sum to 1.0
- Suitable for most institutional mandates and retail accounts
- Default mode

### Long-Short
- Long side sums to +1.0, short side sums to −1.0
- Dollar-neutral (net exposure = 0) in the optimised method
- Quantile method: long Q5, short Q1
- Requires margin / short-selling capability

---

## Configuration Reference

| Parameter | CLI Flag | Default | Description |
|-----------|----------|---------|-------------|
| `mode` | `--mode` | `long_only` | `long_only` or `long_short` |
| `weight_scheme` | `--weight` | `equal` | `equal`, `market_cap`, `signal` |
| `rebalance_freq` | `--freq` | `M` | `D`, `W`, `M` |
| `n_quantiles` | `--n-quantiles` | 5 | Number of quantile bins |
| `top_n` | `--top-n` | None | Override: pick top N stocks |
| `max_stock_weight` | `--max-stock-weight` | 0.05 | 5% per stock |
| `max_sector_weight` | `--max-sector-weight` | 0.30 | 30% per sector |
| `max_turnover` | `--max-turnover` | None | Max one-way turnover |
| `risk_aversion` | `--risk-aversion` | 1.0 | λ for optimised method |

---

## CLI Usage

```bash
# Baseline: quantile long-only, monthly, equal weight
python scripts/build_portfolio.py

# Long-short, weekly rebalance
python scripts/build_portfolio.py --mode long_short --freq W

# Use a specific factor
python scripts/build_portfolio.py --signal MOM_12_1M

# Top-30 concentrated portfolio
python scripts/build_portfolio.py --top-n 30

# Optimised with constraints
python scripts/build_portfolio.py --method optimize --risk-aversion 2.0 \
    --max-stock-weight 0.03 --max-turnover 0.20

# Market-cap weighted, daily rebalance
python scripts/build_portfolio.py --weight market_cap --freq D
```

## Output

| File | Format | Key | Description |
|------|--------|-----|-------------|
| `data/portfolio/weights_*.parquet` | Parquet | `(date, ticker)` | Daily position weights (carried forward between rebalances) |
