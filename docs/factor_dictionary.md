# EntroPy — Factor Dictionary

> 24 alpha factors across 3 categories, all sharing a unified interface.

---

## Architecture

```
FactorBase (ABC)
├── meta: FactorMeta         # name, category, lookback, lag, direction
├── _compute() → raw signal  # subclass implements this
└── compute() → full pipeline:
      raw → lag → handle_missing → winsorize → zscore → neutralize → output
```

### Pipeline Steps (applied automatically by `FactorBase.compute()`)

| Step | Default | Description |
|------|---------|-------------|
| **Lag** | `meta.lag` (1–21 days) | Shift signal forward to prevent T+0 look-ahead |
| **Missing values** | `"drop"` | `drop` / `zero` / `median` (cross-sectional) |
| **Winsorize** | `[1%, 99%]` | Cross-sectional quantile clipping per date |
| **Z-score** | `True` | Cross-sectional standardisation (mean=0, std=1) per date |
| **Neutralize** | optional | Demean by sector (categorical) or OLS-residualize vs. log_mcap (continuous) |

---

## Category 1: Momentum & Reversal (7 factors)

| Factor | Lookback | Lag | Dir | Description | Reference |
|--------|----------|-----|-----|-------------|-----------|
| `MOM_1M` | 22d | 1 | +1 | 1-month (21-day) price momentum | Jegadeesh & Titman (1993) |
| `MOM_3M` | 64d | 1 | +1 | 3-month (63-day) price momentum | Jegadeesh & Titman (1993) |
| `MOM_6M` | 127d | 1 | +1 | 6-month (126-day) price momentum | Jegadeesh & Titman (1993) |
| `MOM_12_1M` | 252d | 21 | +1 | 12-month momentum, skip last 1 month | Jegadeesh & Titman (1993) |
| `STR_1W` | 6d | 1 | −1 | 1-week short-term reversal | Jegadeesh (1990) |
| `STR_1M` | 22d | 1 | −1 | 1-month short-term reversal | Jegadeesh (1990) |
| `MOM_PATH` | 127d | 1 | +1 | Path-dependent momentum: return / \|max drawdown\| | Daniel & Moskowitz (2016) |

### Notes
- **`MOM_12_1M`** uses `lag=21` to skip the most recent month (short-term reversal contaminates raw 12-month momentum).
- **`STR_*`** factors have `direction=−1`: *lower* past returns predict *higher* future returns.
- **`MOM_PATH`** rewards smooth trends over volatile ones with the same total return.

---

## Category 2: Volatility, Risk & Tail (9 factors)

| Factor | Lookback | Lag | Dir | Description | Reference |
|--------|----------|-----|-----|-------------|-----------|
| `VOL_20D` | 21d | 1 | −1 | 20-day annualised realised volatility | Ang et al. (2006) |
| `VOL_60D` | 61d | 1 | −1 | 60-day annualised realised volatility | — |
| `IDIOVOL` | 61d | 1 | −1 | Idiosyncratic vol (CAPM residual std, 60d) | Ang et al. (2006) |
| `SKEW_60D` | 61d | 1 | +1 | 60-day return skewness (negative = crash-prone) | — |
| `KURT_60D` | 61d | 1 | −1 | 60-day return excess kurtosis (heavy tails) | — |
| `DOWNVOL_60D` | 61d | 1 | −1 | Downside semi-deviation (annualised, 60d) | Sortino & van der Meer (1991) |
| `TAIL_RISK` | 61d | 1 | −1 | CVaR / Expected Shortfall at 5% (60d) | — |
| `VOL_OF_VOL` | 80d | 1 | −1 | Std of 20-day vol over 60d (stochastic-vol ν proxy) | Heston (1993); Baltussen et al. (2018) |
| `REALIZED_JUMP` | 61d | 1 | −1 | Jump variance ratio via bi-power variation | Barndorff-Nielsen & Shephard (2004) |

### Stochastic-Process Perspective
- **`VOL_OF_VOL`** proxies the **ν** parameter in the Heston (1993) model — stocks with high vol-of-vol exhibit regime-switching and are harder to hedge.
- **`REALIZED_JUMP`** separates the **continuous** (diffusion) component from the **jump** component of total variance. A high jump ratio indicates event-driven, discontinuous returns.
- **`TAIL_RISK`** (CVaR) captures the expected loss *given* that you're in the worst 5% of the return distribution — relevant for portfolio risk budgeting.

---

## Category 3: Liquidity, Volume & Turnover (8 factors)

| Factor | Lookback | Lag | Dir | Description | Reference |
|--------|----------|-----|-----|-------------|-----------|
| `TURNOVER_20D` | 21d | 1 | −1 | 20-day avg relative turnover (vol / 60d median vol) | — |
| `TURNOVER_60D` | 61d | 1 | −1 | 60-day avg relative turnover | — |
| `ILLIQ_AMIHUD` | 61d | 1 | −1 | Amihud illiquidity: mean(\|ret\| / dollar volume) | Amihud (2002) |
| `VOLUME_CV` | 61d | 1 | −1 | Coefficient of variation of daily volume (60d) | — |
| `PRICE_IMPACT` | 61d | 1 | −1 | Kyle's λ proxy: β of \|ret\| on signed volume | Kyle (1985) |
| `TURNOVER_ACCEL` | 61d | 1 | −1 | Turnover acceleration: 20d avg / 60d avg − 1 | — |
| `ABNORMAL_VOLUME` | 61d | 1 | −1 | Volume z-score vs 60-day trailing distribution | — |
| `SPREAD_HL` | 22d | 1 | −1 | High-low spread estimator (Corwin-Schultz) | Corwin & Schultz (2012) |

### Notes
- Turnover factors use **relative turnover** (volume / trailing median volume) since absolute shares outstanding is not always available from free data sources.
- **`ILLIQ_AMIHUD`** is scaled by 1e6 for numerical stability.
- **`SPREAD_HL`** decomposes the daily range into volatility + bid-ask spread components — a microstructure-based liquidity measure that doesn't require tick data.

---

## Evaluation Metrics

Each factor is evaluated via `factor_tearsheet()`:

| Metric | Description |
|--------|-------------|
| **IC** (Pearson) | Daily cross-sectional correlation(factor, forward return) |
| **Rank IC** (Spearman) | Rank correlation — more robust to outliers |
| **ICIR** | IC / std(IC) × √252 — signal-to-noise ratio (annualised) |
| **t-statistic** | Mean IC / (std IC / √N) — statistical significance |
| **Hit rate** | Fraction of days with IC > 0 |
| **IC decay** | IC at horizons 1d … 20d — measures signal persistence |
| **Quantile returns** | Mean forward return per factor quintile |
| **Long−short return** | Q5 − Q1 daily return |
| **Factor turnover** | 1 − rank_corr(t, t−1) — implementation cost proxy |

### Comparison Table

`compare_factors()` produces a single DataFrame with one row per factor and all key metrics, enabling quick screening and ablation.

---

## Quick Start

```python
from entropy.factors.registry import FactorRegistry
from entropy.factors.evaluation import add_forward_returns, factor_tearsheet
from entropy.utils.io import load_parquet

# 1. Load prices
prices = load_parquet("data/prices/prices.parquet")

# 2. Compute all factors
reg = FactorRegistry()
reg.discover()
factor_df = reg.compute_all(prices, winsorize_limits=(0.01, 0.99), zscore=True)

# 3. Evaluate
prices_fwd = add_forward_returns(prices)
merged = factor_df.merge(prices_fwd[["date", "ticker", "fwd_ret_1d"]], on=["date", "ticker"])
ts = factor_tearsheet(merged, "MOM_12_1M", return_col="fwd_ret_1d")
print(ts["rank_ic_stats"])
```

## CLI

```bash
# Compute all factors
python scripts/build_factors.py

# Compute + evaluate
python scripts/build_factors.py --evaluate

# List available factors
python scripts/build_factors.py --list

# Specific factors only
python scripts/build_factors.py -f MOM_12_1M -f VOL_20D -f ILLIQ_AMIHUD --evaluate
```
