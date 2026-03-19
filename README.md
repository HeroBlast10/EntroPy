<div align="center">

# EntroPy

**A modular quantitative research and paper-trading platform for equities, combining conventional cross-sectional factors with state-space / regime / entropy-based signals, under bias-aware data handling and realistic market-specific execution constraints.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

*Signal-processing / state-space / regime-based alpha platform with full research pipeline.*

</div>

---

## What is this?

EntroPy is a **multi-market equity research platform** that goes beyond standard multi-factor backtesting.  It provides a complete closed loop — from bias-aware data handling to walk-forward validation, ablation studies, and IB paper-trading — while integrating **state-space models (Kalman filter), regime detection (HMM), noise spectroscopy (spectral entropy, Hurst), and mean-reversion signals (OU process)** alongside conventional cross-sectional factors.

The platform supports both **US equities** (NYSE/NASDAQ) and **CN A-shares** (SSE/SZSE), with market-specific adapters for data loading, trading rules, and transaction cost modelling.

> **Background:** Many of the advanced signal modules draw on techniques from statistical physics and signal processing — Kalman filtering for price denoising, Hidden Markov Models for regime identification, Ornstein-Uhlenbeck processes for mean-reversion quantification, and spectral analysis for noise structure characterisation.  These are unified under a common factor registry with a **4-tier signal taxonomy**.

## Four-Tier Signal Taxonomy

| Type | Examples | Evaluation | Purpose |
|------|----------|------------|---------|
| **Cross-sectional factors** | Momentum, Volatility, Liquidity | IC / RankIC / quantile returns | Stock ranking |
| **Time-series features** | Kalman velocity, Spectral entropy, Hurst, Skewness | Directional accuracy / hit rate | Per-asset latent-state description |
| **Regime overlays** | HMM turbulence probability | Regime-split Sharpe | Risk modulation (when to reduce) |
| **Relative-value strategies** | OU pairs z-score | Half-life / cointegration test | Spread trading |

## Architecture

```
quant_platform/
├── core/
│   ├── data/                          # Multi-market data layer
│   │   ├── calendar.py                #   Trading calendars (XNYS, XSHG, XSHE)
│   │   ├── adapters/
│   │   │   ├── us_equity.py           #   US: yfinance + SimFin pipeline
│   │   │   └── cn_a_share.py          #   CN: Baostock/Tushare + T+1/ST/price-limit handling
│   │   └── market_rules/              #   Market-specific constraints
│   │
│   ├── signals/                       # 4-tier signal library (30+ signals)
│   │   ├── base.py                    #   FactorBase + FactorMeta (with signal_type field)
│   │   ├── registry.py                #   Auto-discovery & batch compute
│   │   ├── transforms.py              #   Lag, winsorize, z-score, neutralize
│   │   ├── cross_sectional/           #   24 CS factors (momentum, vol, liquidity, value)
│   │   ├── time_series/               #   Kalman, spectral entropy, Hurst, higher moments
│   │   ├── regime/                    #   HMM turbulence probability
│   │   ├── relative_value/            #   OU process z-score
│   │   └── diagnostics/               #   Phase space reconstruction (visualization)
│   │
│   ├── alpha_models/                  # Signal → alpha score
│   │   ├── cross_sectional_ranker.py  #   CS factor composite ranking
│   │   ├── ts_forecaster.py           #   TS feature ensemble + momentum z-score
│   │   ├── regime_overlay.py          #   HMM-based weight modulation
│   │   └── ensemble.py                #   w_cs * CS + w_ts * TS × regime_scalar
│   │
│   ├── portfolio/                     # Quantile + mean-variance optimisation
│   ├── execution/
│   │   ├── backtest/                  #   Vectorized daily (US) + event sim (CN A-share)
│   │   ├── cost_models/               #   US equity + CN A-share transaction costs
│   │   └── paper/ibkr/                #   IB TWS paper trading with risk controls
│   │
│   └── evaluation/                    # Walk-forward, ablation, overfit detection, HTML report
│
├── apps/dashboard/                    # Signal Lab: Streamlit interactive dashboard
└── experiments/                       # YAML experiment configs
```

## Quick Start

```bash
pip install -r requirements.txt

# Full US equity pipeline
python scripts/build_dataset.py          # 1. Download & align price data
python scripts/build_factors.py --evaluate  # 2. Compute & evaluate all signals → factor_comparison.csv
python scripts/build_portfolio.py        # 3. Build portfolio weights
python scripts/run_backtest.py           # 4. Simulate execution + PnL
python scripts/generate_report.py --auto-best  # 5. Generate HTML report (auto-select best factor)

# Factor parameter tuning (grid-search)
python scripts/tune_factors.py           # Search period/lag space, rank by IC/ICIR/Sharpe
python scripts/tune_factors.py --objective ric_icir --top 5

# Interactive dashboard
streamlit run quant_platform/apps/dashboard/app.py
```

## Signal Library

### Cross-Sectional Factors (24)

| Category | # | Highlights |
|----------|---|-----------|
| **Momentum / Reversal** | 7 | `MOM_12_1M` (Jegadeesh-Titman), `MOM_PATH`, `STR_1W` |
| **Volatility / Risk / Tail** | 9 | `IDIOVOL`, `REALIZED_JUMP` (BPV), `VOL_OF_VOL` (Heston proxy) |
| **Liquidity / Volume** | 8 | `ILLIQ_AMIHUD`, `SPREAD_HL` (Corwin-Schultz), `PRICE_IMPACT` |

### Time-Series Features (Numba-accelerated)

| Signal | Method | What it captures |
|--------|--------|------------------|
| `KF_VELOCITY` | Kalman filter | Trend rate of change (denoised) |
| `KF_TREND_STRENGTH` | Kalman filter | Dimensionless trend strength |
| `KF_NOISE_RATIO` | Kalman filter | Microstructure noise proxy |
| `SPECTRAL_ENTROPY_60D` | FFT + Shannon entropy | Market randomness (1 = noise, 0 = periodic) |
| `HURST_60D` | R/S analysis | Trending (>0.5) vs. mean-reverting (<0.5) |
| `ROLLING_SKEW_60D` | Higher moments | Asymmetric return distribution |
| `ROLLING_KURT_60D` | Higher moments | Fat-tail risk indicator |
| `ACF_DECAY_60D` | Autocorrelation decay | Serial dependence persistence |

### Regime Overlay

| Signal | Method | Usage |
|--------|--------|-------|
| `HMM_TURBULENCE_PROB` | 2-state Gaussian HMM | Scale down exposure when P(turbulent) > threshold |

### Relative-Value

| Signal | Method | Usage |
|--------|--------|-------|
| `OU_ZSCORE` | Ornstein-Uhlenbeck MLE | Mean-reversion z-score for pairs/spread trading |

## Hero Experiment

The platform's core research question:

> **Do state-space / regime / entropy features provide verifiable incremental value over standard cross-sectional factors?**

Designed as a single ablation study with walk-forward OOS validation:

| Scenario | Factors | Regime Overlay |
|----------|---------|----------------|
| `baseline_only` | MOM + STR + VOL + ILLIQ | No |
| `baseline_plus_kalman` | + KF_VELOCITY + KF_TREND | No |
| `baseline_plus_noise` | + ENTROPY + HURST | No |
| `baseline_plus_regime` | baseline | Yes (HMM) |
| `full_ensemble` | All signals | Yes |

Evaluated on: IC, net Sharpe, max drawdown, turnover, cost stress test (0-20 bps), regime-split performance.

See: `quant_platform/experiments/us_signal_lab.yaml`

## Multi-Market Support

| Feature | US Equities | CN A-Shares |
|---------|-------------|-------------|
| Calendar | NYSE (XNYS) | SSE/SZSE (XSHG/XSHE) |
| Data Source | yfinance + SimFin | Baostock / Tushare |
| Settlement | T+2 | T+1 |
| Short Selling | Yes | No |
| Price Limits | None | ±10% main / ±20% STAR |
| Special Rules | — | ST filtering, suspension handling |
| Cost Model | Commission + slippage + impact | Commission + stamp tax + transfer fee |

## Transaction Cost Models

### US Equity

| Component | Model | Default |
|-----------|-------|---------|
| Commission | Per-share | $0.005/share |
| Slippage | Fixed half-spread | 5 bps |
| Market Impact | √-model (Almgren-Chriss) | `0.1 × σ × √(participation)` |
| SEC Fee | Sells only | $8 / $1M |

### CN A-Share

| Component | Model | Default |
|-----------|-------|---------|
| Commission | Bilateral | 3 bps (min ¥5) |
| Stamp Tax | Sell-only | 5 bps |
| Transfer Fee | Bilateral | 0.1 bps |
| Slippage | Fixed / volume / volatility | 5 bps |

## Research Report

Auto-generated self-contained HTML with **11 sections**: executive summary, **all-factor comparison table** (with auto-best selection), NAV & drawdown, monthly heatmap, rolling Sharpe, turnover, IC/RankIC for selected factor, factor correlation, cost attribution, walk-forward OOS, ablation.

**New features:**
- `--auto-best` flag: automatically selects the top-ranked factor from `factor_comparison.csv`
- `--optimize-by ric_mean_ic | ric_icir | ls_sharpe`: choose ranking metric
- Section 2 displays full factor comparison table with metrics for all computed factors

## IB Paper Trading

Full integration with Interactive Brokers TWS/Gateway paper account:

```bash
python scripts/paper_trade.py                     # Default: 5 tickers, MKT orders
python scripts/paper_trade.py -t AAPL -t MSFT     # Custom tickers
python scripts/paper_trade.py --dry-run            # Log orders without submitting
```

Risk controls: kill switch, max order notional, position count limits, daily loss limits.

## Tests

```bash
pytest tests/ -v
```

## Tech Stack

`pandas` · `numpy` · `scipy` · `numba` · `pyarrow` · `matplotlib` · `plotly` · `streamlit` · `yfinance` · `simfin` · `baostock` · `exchange_calendars` · `ib_insync` · `loguru`

## License

[MIT](LICENSE)
