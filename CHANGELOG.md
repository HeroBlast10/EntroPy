# Changelog

All notable changes to EntroPy are documented in this file.

## [0.7.0] — 2026-03-19

### Added — All-Factor Comparison & Auto‑Best Selection
- `quant_platform/core/evaluation/report.py` — new Section 2 "Factor Comparison (All Factors)" that reads `data/factors/factor_comparison.csv` and renders a full comparison table; renumbered existing sections 2‑10 → 3‑11
- `select_best_factor(comparison, metric="ric_mean_ic")` helper that ranks factors by any metric (ric_mean_ic, ric_icir, ls_sharpe, …) and returns the winner
- `scripts/generate_report.py` — added `--auto-best` flag and `--optimize-by` option; when `--auto-best` is set the CLI reads factor_comparison.csv, calls `select_best_factor()`, and passes the winner as `signal_col`
- `scripts/generate_report.py` — extended CLI to support multi-factor batch reporting: `--factors` now accepts 1‑N factor names, `--all-factors` generates reports for every row in `factor_comparison.csv`, `--list` prints a ranked table of available factors, and `--output-dir` controls the destination directory for multiple HTML reports

### Added — Factor Parameter Overrides & Registry Support
- `quant_platform/core/signals/base.py` — `FactorBase.__init__(**param_overrides)`; meta-field keys (lookback, lag, …) replace the frozen FactorMeta via `dataclasses.replace`; all other keys land in `self._extra_params` for `_compute()` to consume; `compute()` now uses `self._meta` throughout
- `quant_platform/core/signals/registry.py` — `compute_all(…, factor_params: Dict[str, Dict] = None)` so callers can pass per‑factor constructor kwargs for one‑shot tuning
- Updated all 7 momentum/reversal factors (`MOM_1M`, `MOM_3M`, `MOM_6M`, `MOM_12_1M`, `STR_1W`, `STR_1M`, `MOM_PATH`) to read `period`/`window` from `self._extra_params` with fallback to original defaults, preserving existing behaviour

### Added — Grid‑Search Auto‑Tuner
- `scripts/tune_factors.py` (new) — grid‑search CLI that iterates the cartesian product of `period`/`window` × `lag` search space for each factor, evaluates `ric_mean_ic`, `ric_icir`, and `ls_sharpe`, saves `data/factors/tune_results.csv`, and prints top‑N results per factor and the overall best configuration
- Default search space covers all 7 momentum factors with practical period/lag ranges; supports custom objective (`--objective ric_mean_ic | ric_icir | ls_sharpe`), factor subset (`--factors`), and result count (`--top`)

### Added — End-to-End Factor Research Pipeline
- `scripts/run_factor_pipeline.py` (new) — orchestrates the full research loop (factor computation → `factor_comparison.csv` evaluation → portfolio construction → cost-aware backtest → HTML research report) behind a single CLI
- Pipeline supports single, multiple, or all factors via `--factors`, `--all-factors`, and `--auto-best` flags, plus `--quick` mode (skips walk-forward and ablation), `--skip-factor-compute`, and portfolio configuration options (`--mode`, `--freq`, `--method`, `--max-stock-weight`, `--max-sector-weight`, `--capital`)
- Batch runs produce one report per factor (e.g. `data/reports/report_MOM_12_1M.html`) and summarise results in a Sharpe‑sorted leaderboard at the end of the run

## [0.6.0] — 2026-02-11

### Added — IB Paper Trading (Step 7)
- `entropy/live/config.py` — `IBConfig`, `RiskLimits`, `StrategyConfig`, `PaperTradingConfig` dataclasses with all tuneable knobs
- `entropy/live/gateway.py` — IB TWS/Gateway connection manager with auto-reconnect, structured logging, context manager support
- `entropy/live/market_data.py` — real-time streaming quotes, snapshots, historical bars via `ib_insync`
- `entropy/live/execution.py` — risk-gated `OrderManager` supporting MKT / LMT / Adaptive orders, fill callbacks, fill log persistence
- `entropy/live/risk.py` — `RiskManager` with kill switch, per-order limits (notional, shares), position limits, session limits (daily loss auto-kill, trade count, cumulative notional)
- `entropy/live/portfolio.py` — live position tracking, target-vs-actual rebalance diff computation, account PnL queries
- `entropy/live/strategy.py` — `PaperTradingStrategy` main loop: subscribe → snapshot → signal → rebalance → submit → log → sleep → repeat
- `scripts/paper_trade.py` — full CLI with `--dry-run`, `--kill-switch`, `--order-type`, `--max-daily-loss`, `--run-once`, etc.
- Structured logging: JSONL cycle logs (`data/live/logs/`) + session state snapshots (`data/live/state/`)
- Added `ib_insync>=0.9.86` to requirements.txt

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
