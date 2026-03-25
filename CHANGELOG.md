# Changelog

All notable changes to EntroPy are documented in this file.

## [0.8.0] — 2026-03-25

### Added — ML-Based Alpha Model
- `quant_platform/core/alpha_models/ml_alpha.py` — Ridge/Lasso/ElasticNet cross-sectional regression with Purged K-Fold CV (Lopez de Prado) to prevent look-ahead bias in time-series data
- `PurgedKFold` class — time-series cross-validation with embargo period (default 1%) between train/test splits
- `MLAlphaModel` class — regularized linear regression with feature standardization, coefficient-based feature importance, and cross-validation metrics (R², IC)
- `WalkForwardMLAlpha` class — expanding-window walk-forward retraining (default: refit every 21 days, min 252 days training), tracks feature importance evolution across models
- `tests/test_ml_alpha.py` — comprehensive tests for Purged K-Fold splits, model training/prediction, cross-validation, walk-forward retraining, and edge cases (missing features, regularization strength, Lasso feature selection)

### Added — Barra-Style Factor Risk Model
- `quant_platform/core/portfolio/risk_model.py` — extended from simple covariance to full Barra-style factor decomposition
- Factor exposure estimation via time-series regression (Market, Size, Value betas) with configurable lookback (default 252 days) and minimum observations (default 20)
- Factor covariance matrix using EWMA (default halflife 60 days) + Ledoit-Wolf shrinkage (default 0.5)
- Specific risk estimation from regression residuals (annualized standard deviation of idiosyncratic returns)
- Portfolio risk decomposition: `decompose_risk()` breaks down total portfolio variance into factor risk (per-factor contributions) and specific risk components
- Getter methods for exposures, factor covariance, and specific risk with validation
- `tests/test_factor_risk_model.py` — tests for exposure estimation accuracy, covariance matrix properties, specific risk calculation, full model fit, risk decomposition (including market-neutral portfolios), and error handling

### Added — VaR/CVaR Risk Metrics
- `quant_platform/core/evaluation/risk_metrics.py` — three VaR methods (Historical, Parametric Gaussian, Cornish-Fisher with skew/kurtosis adjustment) plus CVaR (Conditional VaR / Expected Shortfall)
- `compute_var()` — supports 95%, 99%, 99.9% confidence levels with method selection
- `compute_cvar()` — expected loss beyond VaR threshold
- `compute_rolling_var()` — time-series of VaR estimates for visualization
- `quant_platform/core/evaluation/risk_plots.py` — visualization functions: rolling VaR chart, return distribution with VaR/CVaR markers, VaR method comparison
- `tests/test_risk_metrics.py` — tests for all VaR methods, CVaR calculation, rolling VaR, edge cases (insufficient data, extreme confidence levels)

### Added — Benchmark Analytics
- `quant_platform/core/data/benchmark.py` — equal-weight benchmark construction from universe
- `quant_platform/core/evaluation/benchmark_analytics.py` — CAPM-style analytics: Alpha, Beta, Information Ratio, Treynor Ratio, tracking error, active return
- `quant_platform/core/evaluation/benchmark_plots.py` — cumulative return comparison, rolling alpha/beta, scatter plot with regression line
- `tests/test_benchmark_analytics.py` — tests for all benchmark metrics and edge cases

### Added — Value/Quality Factors
- `quant_platform/core/signals/cross_sectional/value_quality.py` — four Fama-French style factors:
  - `BOOK_TO_MARKET` — book value / market cap (value factor)
  - `EARNINGS_YIELD` — trailing 12-month earnings / market cap
  - `ROE` — return on equity (net income / book equity)
  - `GROSS_PROFITABILITY` — (revenue - COGS) / total assets (Novy-Marx 2013)
- All factors use SimFin fundamentals with 45-day publication lag to avoid look-ahead bias
- `tests/test_value_quality_factors.py` — tests for all four factors with synthetic fundamental data

### Added — Inverse-Volatility Weighting
- `quant_platform/core/portfolio/quantile.py` — added inverse-volatility weighting (naive risk parity) as fourth weighting scheme
- `_compute_inverse_vol_weights()` — three-tier fallback: precomputed volatility from universe → computed rolling volatility from prices (63-day window, annualized) → equal weight
- `_compute_rolling_volatility()` — point-in-time rolling volatility calculation with proper date alignment
- `tests/test_inverse_vol_weighting.py` — tests for volatility-to-weight conversion, rolling volatility calculation, integration, fallbacks, and edge cases

### Added — Type-Aware Evaluation System
- `quant_platform/core/signals/evaluation/` — new module with four signal-type-specific scorecards
- `router.py` — dispatches signals to appropriate scorecard based on `FactorMeta.signal_type`
- `cross_sectional.py` — IC, RankIC, IC IR, hit rate, monotonicity (Q5-Q1 spread), turnover
- `time_series.py` — hit rate, directional accuracy, directional Sharpe, mean absolute error
- `regime.py` — baseline vs. overlay Sharpe/drawdown comparison, regime detection rate, exposure reduction
- `relative_value.py` — half-life, ADF stationarity test, mean reversion quality (R²), spread Sharpe, entry/exit ratio
- Each signal type now evaluated with metrics appropriate to its purpose (e.g., Kalman velocity evaluated on hit rate, not IC)

### Added — Dynamic Universe Filtering
- `quant_platform/core/data/universe.py` — replaced placeholder `in_index=True` with dynamic filtering
- `_apply_dynamic_universe_filter()` — selects top N stocks (default 500) by market cap that meet liquidity threshold (default $5M ADV over 30 days)
- Creates "liquidity-filtered large-cap universe" that approximates major indices using actual market data, not official constituent lists
- `config/settings.yaml` — added `top_n_by_mcap`, `min_avg_dollar_volume`, `adv_window_days` parameters

### Fixed — Position Forward-Fill Bug (CRITICAL)
- `quant_platform/core/portfolio/rebalance.py` — fixed `carry_forward_weights()` to explicitly zero out old positions on rebalance dates
- Previous implementation caused weights to accumulate (e.g., summing to 5.66 instead of 1.0) due to stale positions being carried forward indefinitely
- New implementation creates complete weight vector for all tickers on each rebalance date (selected stocks = actual weight, non-selected = 0.0) before forward-filling
- Added `validate_portfolio_weights()` function to check invariants: long-only sum=1.0, no negative weights, gross exposure ≤ 2.0 for long-short
- `tests/test_rebalance.py` — added tests for weight sum invariant, old position zeroing, and validation function

### Fixed — Signal Mismatch in Report Generation
- `quant_platform/core/evaluation/report.py` — changed from warning to error when backtest signal ≠ report signal
- Previously allowed generating inconsistent reports where NAV/performance reflected factor A but IC analysis used factor B
- Now raises `ValueError` with clear instructions to re-run backtest or use correct signal
- Prevents misleading reports that mix different factors' metrics

### Fixed — Documentation Accuracy
- `quant_platform/core/evaluation/walkforward.py` — downgraded claims from "walk-forward validation framework" to "simple rolling OOS check"
- Explicitly documented limitations: no per-fold factor selection, no parameter tuning, no model retraining, fixed portfolio construction
- `quant_platform/core/signals/relative_value/ou_pairs.py` — clarified that `OU_ZSCORE` is single-stock mean reversion, not pairs trading
- Added warning that true pairs trading requires cointegration testing, spread construction, and OU fitting on spreads (not individual prices)
- `README.md` — updated all references to reflect accurate capabilities

### Changed — README Rewrite (Research Focus)
- Shifted focus from dashboard/paper trading to research question and methodology
- Added **Research Question** section: "Do state-space / regime / entropy features provide verifiable incremental alpha?"
- Added **Data** section: universe definition, period, PIT discipline
- Added **Methodology** section: signal library (30+ signals, four types), alpha models, portfolio construction, risk management, transaction costs, validation
- Added **Ablation Design** table: 7 scenarios (baseline, +value, +kalman, +noise, +regime, ML alpha, full ensemble)
- Added **Known Limitations** section: 6 honest limitations (approximate universe, simplified OOS, single-stock OU, no intraday data, simplified risk model, linear ML)
- Updated **Architecture** diagram to reflect new modules (evaluation scorecards, ML alpha, risk decomposition)
- Removed IB paper trading and Streamlit dashboard from prominence (still available but not featured)

### Changed — .gitignore
- Added 11 implementation/interview prep `.md` files to `.gitignore`:
  - `RESUME_AND_INTERVIEW_PREP.md`, `VALUE_QUALITY_IMPLEMENTATION.md`, `ML_ALPHA_IMPLEMENTATION.md`, `FACTOR_RISK_MODEL_IMPLEMENTATION.md`, `VAR_CVAR_IMPLEMENTATION.md`, `BENCHMARK_IMPLEMENTATION.md`, `INVERSE_VOL_IMPLEMENTATION.md`, `IMPROVEMENT_ROADMAP.md`, `BUGFIX_SUMMARY.md`, `ACCURACY_FIXES_SUMMARY.md`, `EVALUATION_SYSTEM_DESIGN.md`
- These files remain local for personal interview preparation but are excluded from the public repository

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
