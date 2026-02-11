<div align="center">

# EntroPy

**Research-Grade Multi-Factor Backtest Framework for US Equities**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

*Data → Factors → Portfolio → Execution → Report → Live Trading — six layers, from research to paper trading.*

</div>

---

## What is this?

EntroPy is an **end-to-end quantitative research pipeline** that takes raw
price data and produces a publication-quality backtest report — complete with
walk-forward validation, cost attribution, ablation studies, and an **IB paper-trading demo**.

```
build_dataset → build_factors → build_portfolio → run_backtest → generate_report → paper_trade
     (data)        (24 alphas)      (weights)       (trades+PnL)     (HTML report)    (IB live)
```

**Key differentiators:**

- **Bias-aware** — point-in-time adjustment factors, publication-lag alignment,
  NYSE calendar enforcement, no look-ahead at any stage
- **Realistic costs** — commission, slippage, square-root impact (Almgren-Chriss),
  SEC/FINRA fees, borrow cost — all separately parameterised
- **Robustness-first** — walk-forward OOS validation, multi-scenario cost ablation,
  factor crowding diagnostics
- **Research-to-live** — IB paper-trading integration with real-time data,
  risk-gated order flow, position tracking, and structured logging

## Quick Start

```bash
pip install -r requirements.txt

# Full pipeline (5 commands)
python scripts/build_dataset.py          # 1. Download & align price data
python scripts/build_factors.py          # 2. Compute 24 alpha factors
python scripts/build_portfolio.py        # 3. Build portfolio weights
python scripts/run_backtest.py           # 4. Simulate execution + PnL
python scripts/generate_report.py        # 5. Generate HTML research report

# Or run the quick demo (~5 min, 10 tickers)
python examples/quick_start.py
```

## Architecture

```
EntroPy/
├── config/settings.yaml          # Pipeline configuration
├── configs/                      # Experiment presets (baseline / conservative / aggressive)
│
├── entropy/
│   ├── data/                     # Step 1 — Data Layer
│   │   ├── calendar.py           #   NYSE trading calendar (exchange_calendars)
│   │   ├── prices.py             #   OHLCV + point-in-time adj_factor (yfinance)
│   │   ├── universe.py           #   Tradable pool (IPO age, price, market cap filters)
│   │   ├── fundamentals.py       #   Financials with publication lag (SimFin / yfinance)
│   │   ├── schema.py             #   PyArrow schema enforcement
│   │   ├── manifest.py           #   xxHash checksums + git tagging
│   │   └── pipeline.py           #   Orchestrator
│   │
│   ├── factors/                  # Step 2 — Factor Library (24 factors)
│   │   ├── base.py               #   FactorBase ABC: lag → clean → winsorize → zscore → neutralize
│   │   ├── transforms.py         #   Cross-sectional transforms
│   │   ├── evaluation.py         #   IC / RankIC / ICIR / quantile returns / tearsheet
│   │   ├── registry.py           #   Auto-discovery & batch compute
│   │   ├── momentum.py           #   7 momentum / reversal factors
│   │   ├── volatility.py         #   9 vol / risk / tail factors
│   │   └── liquidity.py          #   8 liquidity / volume factors
│   │
│   ├── portfolio/                # Step 3 — Portfolio Construction
│   │   ├── construction.py       #   Base class + PortfolioConfig
│   │   ├── quantile.py           #   Baseline: quantile selection + 4 weighting schemes
│   │   ├── optimize.py           #   Advanced: mean-variance (Ledoit-Wolf + SLSQP)
│   │   ├── constraints.py        #   Stock / sector / turnover constraints
│   │   ├── rebalance.py          #   D / W / M schedule + carry-forward
│   │   └── pipeline.py           #   Orchestrator
│   │
│   ├── trading/                  # Step 4 — Execution & Cost Model
│   │   ├── costs.py              #   CostModel: commission, slippage, impact, fees, borrow
│   │   ├── execution.py          #   Weight-diff → trades → fill simulation
│   │   ├── pnl.py                #   Daily gross/net PnL, NAV, drawdown, Sharpe/Sortino/Calmar
│   │   └── pipeline.py           #   Orchestrator
│   │
│   ├── evaluation/               # Step 5 — Research Report & Robustness
│   │   ├── analytics.py          #   Sector exposure, factor correlation, crowding proxy
│   │   ├── plots.py              #   11 publication-quality matplotlib charts
│   │   ├── walkforward.py        #   Rolling train/validate OOS framework
│   │   ├── ablation.py           #   Cost / neutralize / universe sensitivity
│   │   └── report.py             #   Self-contained HTML report generator
│   │
│   ├── live/                     # Step 6 — IB Paper Trading
│   │   ├── config.py             #   IBConfig, RiskLimits, StrategyConfig
│   │   ├── gateway.py            #   IB connection lifecycle (connect/reconnect/shutdown)
│   │   ├── market_data.py        #   Real-time quotes, bars, tick subscriptions
│   │   ├── execution.py          #   Risk-gated order placement, fill tracking
│   │   ├── risk.py               #   Pre-trade checks, kill switch, session limits
│   │   ├── portfolio.py          #   Live position & PnL tracking, rebalance diff
│   │   └── strategy.py           #   Signal → target weights → orders loop
│   │
│   └── utils/io.py               # Config loader, Parquet I/O
│
├── scripts/                      # CLI entry points (one per step)
├── examples/quick_start.py       # One-click full demo
├── tests/                        # Unit tests (pytest)
├── configs/                      # Experiment YAML presets
├── docs/                         # Field-level documentation for each layer
├── CHANGELOG.md                  # Iterative development log
└── requirements.txt
```

## The Six Layers

### 1. Data Layer

| Output | Key | Description |
|--------|-----|-------------|
| `prices.parquet` | `(date, ticker)` | OHLCV + point-in-time `adj_factor` (no future leakage) |
| `universe.parquet` | `(date, ticker)` | Tradable pool after filters (60d listing, $1 min, $50M cap) |
| `fundamentals.parquet` | `(date, ticker)` | Financials with 45-day publication lag |
| `manifest.json` | — | xxHash checksums + git tag for full reproducibility |

→ [docs/data_dictionary.md](docs/data_dictionary.md)

### 2. Factor Library — 24 Alpha Factors

| Category | # | Highlights |
|----------|---|-----------|
| **Momentum / Reversal** | 7 | `MOM_12_1M` (Jegadeesh-Titman), `MOM_PATH`, `STR_1W` |
| **Volatility / Risk / Tail** | 9 | `IDIOVOL`, `REALIZED_JUMP` (BPV), `VOL_OF_VOL` (Heston proxy) |
| **Liquidity / Volume** | 8 | `ILLIQ_AMIHUD`, `SPREAD_HL` (Corwin-Schultz), `PRICE_IMPACT` |

Every factor passes through: `raw → lag → clean → winsorize [1%,99%] → z-score → neutralize`

→ [docs/factor_dictionary.md](docs/factor_dictionary.md)

### 3. Portfolio Construction

| Method | Description |
|--------|-------------|
| **Quantile** (baseline) | Top/bottom quintile, equal / market-cap / signal weight |
| **Optimised** (advanced) | `max α'w − (λ/2)w'Σw` with box + sector + turnover constraints |

Modes: long-only · long-short · D/W/M rebalance · configurable position limits

→ [docs/portfolio_dictionary.md](docs/portfolio_dictionary.md)

### 4. Transaction Cost Model

| Component | Model | Default |
|-----------|-------|---------|
| Commission | Per-share | $0.005/share |
| Slippage | Fixed half-spread | 5 bps |
| Market Impact | √-model (Almgren-Chriss) | `0.1 × σ × √(participation)` |
| SEC Fee | Sells only | $8 / $1M |
| Borrow | Annual / 252 | 50 bps/yr |

→ [docs/trading_dictionary.md](docs/trading_dictionary.md)

### 5. Research Report

Auto-generated self-contained HTML with **10 sections**:

| | Section | Why it matters |
|-|---------|---------------|
| 1 | Executive Summary | Headline metrics at a glance |
| 2 | NAV & Drawdown | "Show me the equity curve" |
| 3 | Monthly Heatmap | Seasonal patterns, consistency check |
| 4 | Rolling Sharpe | Regime sensitivity |
| 5 | Turnover | Cost sustainability |
| 6 | IC / RankIC | Signal quality & decay |
| 7 | Factor Correlation | Redundancy & diversification |
| 8 | Cost Attribution | Where does the drag come from? |
| 9 | Walk-Forward | "Is this overfit?" — OOS Sharpe per fold |
| 10 | Ablation | Cost sensitivity: zero → low → high |

### 6. Live Trading (IB Paper)

Connect to IB TWS / Gateway **paper account** for real-time market data and
simulated execution with full risk controls.

| Module | Responsibility |
|--------|----------------|
| `gateway.py` | Connection lifecycle, account info, auto-reconnect |
| `market_data.py` | Streaming quotes, snapshots, historical bars |
| `execution.py` | MKT / LMT / Adaptive orders, fill callbacks, fill log |
| `risk.py` | Kill switch, per-order / position / session limits |
| `portfolio.py` | Live positions, rebalance diff (target vs actual) |
| `strategy.py` | Main loop: data → signal → rebalance → submit → log |

**Risk controls (all toggleable):**

| Check | Default |
|-------|---------|
| Kill switch | Off (instant halt) |
| Max order notional | $50k |
| Max position count | 20 |
| Max daily loss | $10k (auto kill-switch) |
| Max daily trades | 200 |
| Max daily notional | $1M cumulative |

```bash
# Prerequisites: IB TWS or Gateway running in paper mode (port 7497)

# Default: 5 tickers, MKT orders, 5-min rebalance
python scripts/paper_trade.py

# Custom tickers, faster cycle
python scripts/paper_trade.py -t AAPL -t MSFT -t NVDA --interval 60

# Dry run (log orders, don't submit)
python scripts/paper_trade.py --dry-run

# Limit orders with 10 bps offset
python scripts/paper_trade.py --order-type LMT --limit-offset 10

# Single cycle then exit
python scripts/paper_trade.py --run-once

# Strict risk
python scripts/paper_trade.py --max-order-notional 10000 --max-daily-loss 5000
```

Outputs: `data/live/logs/` (JSONL cycle logs + fill logs) · `data/live/state/` (session snapshots)

## Tests

```bash
pytest tests/ -v
```

Covers: calendar alignment, schema validation, adjustment factors, transaction costs, factor transforms (lag/winsorize/zscore/neutralize), rebalance schedule.

## Experiment Configs

Three pre-built YAML experiments in `configs/`:

| File | Strategy |
|------|----------|
| `experiment_baseline.yaml` | MOM_12_1M, long-only, equal weight, monthly |
| `experiment_conservative.yaml` | Market-cap weight, turnover cap 20%, tighter constraints |
| `experiment_aggressive.yaml` | Long-short optimised, weekly rebalance |

## Configuration

All pipeline settings in [`config/settings.yaml`](config/settings.yaml):
date range, universe filters, adjustment method, publication lag, git tagging.

## Tech Stack

`pandas` · `pyarrow` · `numpy` · `scipy` · `matplotlib` · `yfinance` · `simfin` · `exchange_calendars` · `ib_insync` · `loguru` · `click` · `xxhash`

## License

[MIT](LICENSE)
