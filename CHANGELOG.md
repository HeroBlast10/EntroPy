# Changelog

All notable changes to EntroPy are documented in this file.

## [0.5.0] — 2026-02-11

### Added — Engineering Polish (Step 6)
- `examples/quick_start.py` — one-click demo (10 tickers, 3 factors, full pipeline in ~5 min)
- `tests/` — 30+ unit tests covering calendar alignment, schema validation, adjustment factors, transaction costs, factor transforms, and rebalance logic
- `configs/` — three experiment YAML presets (baseline, conservative, aggressive)
- `CHANGELOG.md`, `LICENSE` (MIT)
- Polished README with 30-second pitch, architecture diagram, and full CLI reference

## [0.4.0] — 2026-02-11

### Added — Evaluation & Robustness (Step 5)
- `entropy/evaluation/report.py` — auto-generates a self-contained HTML research report with 10 sections (NAV, drawdown, monthly heatmap, rolling Sharpe, IC, factor correlation, cost attribution, walk-forward, ablation)
- `entropy/evaluation/walkforward.py` — rolling train/validate framework (configurable train/test/step windows)
- `entropy/evaluation/ablation.py` — cost sensitivity ablation (zero / low / baseline / high / very-high) + full pipeline ablation
- `entropy/evaluation/analytics.py` — sector exposure, factor correlation matrix, crowding proxy, rolling stats, monthly return table
- `entropy/evaluation/plots.py` — 11 publication-quality matplotlib chart generators
- `scripts/generate_report.py` — CLI entry point

## [0.3.0] — 2026-02-11

### Added — Trading & Cost Layer (Step 4)
- `entropy/trading/costs.py` — pluggable `CostModel` dataclass: commission (per-share / pct), slippage (bps), market impact (square-root Almgren-Chriss), SEC fee, FINRA TAF, stamp duty, borrow cost
- `entropy/trading/execution.py` — weight-diff → trade list, full-period execution simulation
- `entropy/trading/pnl.py` — daily mark-to-market PnL engine (gross/net NAV, drawdown, cost attribution, Sharpe/Sortino/Calmar)
- `scripts/run_backtest.py` — CLI with per-parameter cost overrides

## [0.2.0] — 2026-02-11

### Added — Portfolio Construction (Step 3)
- `entropy/portfolio/quantile.py` — baseline quantile stock selection with 4 weighting schemes (equal, market-cap, signal-proportional, inverse-vol)
- `entropy/portfolio/optimize.py` — mean-variance optimisation with Ledoit-Wolf shrinkage, SLSQP solver, box + sector + turnover constraints
- `entropy/portfolio/constraints.py` — stock clip, sector clip, turnover blending
- `entropy/portfolio/rebalance.py` — D/W/M rebalance schedule + carry-forward
- Long-only and long-short modes
- `scripts/build_portfolio.py` — CLI entry point

### Added — Factor Library (Step 2)
- 24 alpha factors across 3 categories:
  - **Momentum/Reversal** (7): MOM_1M, MOM_3M, MOM_6M, MOM_12_1M, STR_1W, STR_1M, MOM_PATH
  - **Volatility/Risk/Tail** (9): VOL_20D, VOL_60D, IDIOVOL, SKEW_60D, KURT_60D, DOWNVOL_60D, TAIL_RISK, VOL_OF_VOL, REALIZED_JUMP
  - **Liquidity/Volume** (8): TURNOVER_20D, TURNOVER_60D, ILLIQ_AMIHUD, VOLUME_CV, PRICE_IMPACT, TURNOVER_ACCEL, ABNORMAL_VOLUME, SPREAD_HL
- `FactorBase` ABC with unified pipeline: lag → missing → winsorize → zscore → neutralize
- `FactorRegistry` for auto-discovery and batch computation
- IC / RankIC / ICIR / IC-decay / quantile returns / long-short / turnover evaluation
- `scripts/build_factors.py` — CLI entry point

## [0.1.0] — 2026-02-11

### Added — Data Layer (Step 1)
- `entropy/data/prices.py` — OHLCV fetcher via yfinance with point-in-time adjustment factor (no look-ahead bias from future splits)
- `entropy/data/fundamentals.py` — financial statement fetcher (SimFin + yfinance fallback) with publication-lag alignment (default 45 days)
- `entropy/data/universe.py` — tradable pool builder (min listing days, min price, min market cap, trading status)
- `entropy/data/calendar.py` — NYSE trading calendar wrapper (exchange_calendars)
- `entropy/data/schema.py` — PyArrow schema definitions for all 3 tables
- `entropy/data/manifest.py` — data versioning with xxHash checksums + auto git tags
- `config/settings.yaml` — centralised configuration
- `scripts/build_dataset.py` — CLI entry point
- `docs/data_dictionary.md` — field-level documentation
