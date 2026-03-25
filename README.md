<div align="center">

# EntroPy

**Do signal-processing and state-space features add value to equity factor portfolios?**

A quantitative research platform that tests whether Kalman filters, regime detection, spectral entropy, and OU-process signals improve upon standard cross-sectional factors—under realistic transaction costs, point-in-time data, and rolling out-of-sample validation.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

</div>

---

## Research Question

> **Do state-space / regime / entropy features provide verifiable incremental alpha over standard cross-sectional factors, after transaction costs?**

Most multi-factor platforms stop at momentum + value + quality. This project asks whether signals from statistical physics and signal processing—Kalman velocity, spectral entropy, Hurst exponent, HMM regime detection—can improve risk-adjusted returns when combined with conventional factors.

## Data

| Item | Detail |
|------|--------|
| **Universe** | Liquidity-filtered large-cap US equities (top 500 by market cap, ADV ≥ $5M, dynamically reconstituted monthly) |
| **Period** | 2010-01-01 to 2024-12-31 |
| **Price data** | yfinance daily OHLCV, split-adjusted (point-in-time) |
| **Fundamentals** | SimFin (income, balance, cashflow), 45-day publication lag to avoid look-ahead |
| **Calendar** | NYSE (XNYS) via `exchange_calendars` |
| **PIT discipline** | All signals use only data known as of each rebalance date; fundamentals lagged by reporting delay |

## Methodology

### Signal Library (30+ signals, four types)

| Type | Examples | Evaluation Scorecard | Purpose |
|------|----------|---------------------|---------|
| **Cross-sectional** (28) | MOM_12_1M, ILLIQ_AMIHUD, BOOK_TO_MARKET | IC / RankIC / monotonicity / turnover | Stock ranking |
| **Time-series** (8) | KF_VELOCITY, SPECTRAL_ENTROPY, HURST | Hit rate / directional accuracy / directional Sharpe | Per-asset latent-state |
| **Regime overlay** (1) | HMM_TURBULENCE_PROB | Overlay Sharpe improvement / drawdown reduction | Exposure modulation |
| **Mean-reversion** (1) | OU_ZSCORE (single-stock) | Half-life / ADF stationarity / spread Sharpe | Mean-reversion quality |

Each signal type is evaluated with its own scorecard (see `signals/evaluation/`), not forced through cross-sectional IC analysis.

### Cross-Sectional Factors

| Category | Count | Key Factors |
|----------|-------|-------------|
| **Momentum / Reversal** | 7 | `MOM_12_1M` (Jegadeesh-Titman), `MOM_PATH`, `STR_1W` |
| **Volatility / Risk / Tail** | 9 | `IDIOVOL`, `REALIZED_JUMP` (BPV), `VOL_OF_VOL` (Heston proxy) |
| **Liquidity / Volume** | 8 | `ILLIQ_AMIHUD`, `SPREAD_HL` (Corwin-Schultz), `PRICE_IMPACT` |
| **Value / Quality** | 4 | `BOOK_TO_MARKET`, `EARNINGS_YIELD`, `ROE`, `GROSS_PROFITABILITY` |

### Time-Series Features (Numba-accelerated)

| Signal | Method | What it captures |
|--------|--------|------------------|
| `KF_VELOCITY` | Kalman filter (constant-acceleration model) | Denoised trend rate of change |
| `KF_TREND_STRENGTH` | Kalman filter | Dimensionless trend strength |
| `SPECTRAL_ENTROPY_60D` | FFT + Shannon entropy | Market randomness (1 = noise, 0 = periodic) |
| `HURST_60D` | R/S analysis | Trending (>0.5) vs. mean-reverting (<0.5) |
| `OU_ZSCORE` | Ornstein-Uhlenbeck MLE | Single-stock mean-reversion z-score |

### Alpha Models

| Model | Method | Description |
|-------|--------|-------------|
| **Linear ensemble** | Rank-weighted average | Equal- or fixed-weight combination of factor z-scores |
| **Ridge/Lasso regression** | Regularized cross-sectional regression | ML-based alpha: Fama-MacBeth + L1/L2, walk-forward monthly retraining |
| **Regime overlay** | HMM modulation | Scale portfolio exposure based on turbulence probability |

The ML alpha model uses **Purged K-Fold CV** (Lopez de Prado) to prevent look-ahead bias, and **expanding-window walk-forward** retraining every 21 trading days.

### Portfolio Construction

| Component | Implementation |
|-----------|---------------|
| **Stock selection** | Quantile-based (top/bottom quintile) |
| **Weighting** | Equal weight, market-cap weight, signal-proportional, or **inverse-volatility** (naive risk parity) |
| **Rebalance** | Monthly (last trading day), with explicit position zeroing to prevent stale holdings |
| **Risk model** | Barra-style factor decomposition (Market / Size / Value betas, EWMA + Ledoit-Wolf shrinkage, specific risk from regression residuals) |

### Risk Management

| Metric | Method |
|--------|--------|
| **VaR** | Historical, Parametric (Gaussian), Cornish-Fisher (skew/kurtosis adjusted) |
| **CVaR** | Conditional VaR (Expected Shortfall) |
| **Risk decomposition** | Factor risk vs. specific risk breakdown; per-factor contribution |
| **Benchmark analytics** | Alpha, Beta, Information Ratio, Treynor Ratio vs. equal-weight benchmark |

### Transaction Cost Model (US Equity)

| Component | Model | Default |
|-----------|-------|---------|
| Commission | Per-share | $0.005/share |
| Slippage | Fixed half-spread | 5 bps |
| Market Impact | √-model (Almgren-Chriss) | `0.1 × σ × √(participation)` |
| SEC Fee | Sells only | $8 / $1M |

### Validation

| Technique | Description |
|-----------|-------------|
| **Rolling OOS check** | 36-month train / 12-month test / 12-month step; reports in-sample vs. OOS IC and Sharpe |
| **Ablation study** | Incrementally add signal groups to isolate marginal contribution |
| **Cost stress test** | Sweep transaction costs from 0 to 20 bps to find break-even point |

> **Note:** The rolling OOS check is a simplified train/test sanity check. It does not perform per-fold factor selection or hyperparameter tuning; portfolio construction is fixed (top quintile, equal weight).

## Ablation Design

| Scenario | Factors | Regime Overlay |
|----------|---------|----------------|
| `baseline_only` | MOM + STR + VOL + ILLIQ | No |
| `baseline_plus_value` | + BOOK_TO_MARKET + EARNINGS_YIELD + ROE + GP | No |
| `baseline_plus_kalman` | + KF_VELOCITY + KF_TREND | No |
| `baseline_plus_noise` | + ENTROPY + HURST | No |
| `baseline_plus_regime` | baseline | Yes (HMM) |
| `ml_alpha` | Ridge regression on all factors | No |
| `full_ensemble` | All signals | Yes |

Evaluated on: IC, net Sharpe, max drawdown, turnover, cost stress test (0–20 bps).

## Known Limitations

1. **Universe is approximate.** We use a dynamically reconstituted top-500-by-market-cap universe with a $5M ADV filter, not official S&P 500 historical constituents. This introduces mild survivorship bias.
2. **Rolling OOS is simplified.** The walk-forward validation does not perform per-fold factor selection, parameter tuning, or model retraining. It is a sanity check, not a production walk-forward framework.
3. **OU z-score is single-stock.** `OU_ZSCORE` fits an Ornstein-Uhlenbeck process to individual log-prices, not pair spreads. It is a single-stock mean-reversion signal, not pairs trading.
4. **No intraday data.** All signals are computed on daily OHLCV. Intraday microstructure effects are not captured.
5. **Factor covariance is simplified.** The Barra-style risk model uses time-series regression with 3 factors (Market, Size, Value). Production Barra models use 40+ factors with cross-sectional regression.
6. **ML alpha is linear.** The current ML model uses Ridge/Lasso regression. Non-linear models (XGBoost, neural networks) are not yet implemented.

## Architecture

```
quant_platform/
├── core/
│   ├── data/                          # Data layer (PIT-aware, multi-market)
│   │   ├── calendar.py                #   Trading calendars (XNYS, XSHG, XSHE)
│   │   ├── universe.py                #   Dynamic universe: top-N by mcap + ADV filter
│   │   ├── adapters/                  #   Market-specific data pipelines
│   │   └── market_rules/              #   Market-specific constraints
│   │
│   ├── signals/                       # 4-tier signal library (30+ signals)
│   │   ├── base.py                    #   FactorBase + FactorMeta (with signal_type)
│   │   ├── registry.py                #   Auto-discovery & batch compute
│   │   ├── cross_sectional/           #   28 CS factors (momentum, vol, liquidity, value/quality)
│   │   ├── time_series/               #   Kalman, spectral entropy, Hurst, higher moments
│   │   ├── regime/                    #   HMM turbulence probability
│   │   ├── relative_value/            #   OU process z-score (single-stock mean reversion)
│   │   └── evaluation/                #   Type-specific scorecards (CS / TS / regime / RV)
│   │
│   ├── alpha_models/                  # Signal → alpha score
│   │   ├── cross_sectional_ranker.py  #   CS factor composite ranking
│   │   ├── ml_alpha.py                #   Ridge/Lasso + Purged K-Fold CV + walk-forward
│   │   ├── regime_overlay.py          #   HMM-based weight modulation
│   │   └── ensemble.py                #   Multi-model combination
│   │
│   ├── portfolio/                     # Portfolio construction & risk
│   │   ├── quantile.py                #   Quantile selection + 4 weighting schemes
│   │   ├── rebalance.py               #   Weight carry-forward with explicit position zeroing
│   │   └── risk_model.py              #   Barra-style factor risk decomposition
│   │
│   ├── execution/
│   │   ├── backtest/                  #   Vectorized daily backtest + PnL
│   │   └── cost_models/               #   US equity + CN A-share transaction costs
│   │
│   └── evaluation/
│       ├── walkforward.py             #   Rolling OOS check (train/test split)
│       ├── risk_metrics.py            #   VaR / CVaR (Historical, Parametric, Cornish-Fisher)
│       ├── risk_plots.py              #   Risk visualization (rolling VaR, distributions)
│       ├── benchmark_analytics.py     #   Alpha, Beta, IR, Treynor vs. benchmark
│       └── report.py                  #   12-section HTML research report
│
└── tests/                             # Unit tests for all modules
```

## Quick Start

```bash
pip install -r requirements.txt

# End-to-end pipeline
python scripts/build_dataset.py                    # 1. Download & align price data
python scripts/build_factors.py --evaluate         # 2. Compute & evaluate all signals
python scripts/build_portfolio.py --signal MOM_12_1M  # 3. Build portfolio weights
python scripts/run_backtest.py                     # 4. Simulate execution + PnL
python scripts/generate_report.py --signal MOM_12_1M  # 5. Generate HTML research report

# One-command pipeline (compute → portfolio → backtest → report)
python scripts/run_factor_pipeline.py --factors MOM_12_1M
python scripts/run_factor_pipeline.py --all-factors --quick

# Factor parameter tuning
python scripts/tune_factors.py --objective ric_icir --top 5
```

**Output:** Self-contained HTML report with 12 sections: executive summary, all-factor comparison, NAV & drawdown, monthly heatmap, rolling Sharpe, turnover, IC/RankIC, factor correlation, cost attribution, rolling OOS check, VaR/CVaR risk metrics, and ablation study.

## Tests

```bash
pytest tests/ -v
```

## Tech Stack

`pandas` · `numpy` · `scipy` · `numba` · `scikit-learn` · `pyarrow` · `matplotlib` · `plotly` · `yfinance` · `simfin` · `exchange_calendars` · `loguru`

## License

[MIT](LICENSE)
