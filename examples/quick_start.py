#!/usr/bin/env python
"""EntroPy Quick Start — run the full pipeline on a small subset.

This script demonstrates the entire framework end-to-end in ~5 minutes:
1. Download price data for 10 tickers (2020–2023)
2. Compute 3 representative factors (MOM_12_1M, VOL_20D, ILLIQ_AMIHUD)
3. Build a quantile portfolio (top quintile, equal weight, monthly)
4. Simulate execution with realistic transaction costs
5. Generate an HTML research report

Usage::

    python examples/quick_start.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Setup project root
_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root))

from loguru import logger
logger.remove()
logger.add(sys.stderr, level="INFO", format=(
    "<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | <level>{message}</level>"
))

from entropy.utils.io import set_project_root, load_config
set_project_root(_root)


def main():
    # ── Config ──────────────────────────────────────────────────────
    TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "JPM",
               "JNJ", "XOM", "PG", "NEE", "AMT"]
    START = "2020-01-01"
    END = "2023-12-31"

    print("=" * 60)
    print("  EntroPy — Quick Start Demo")
    print("=" * 60)

    # ── Step 1: Data ────────────────────────────────────────────────
    print("\n▸ Step 1/5: Downloading price data...")
    from entropy.data.prices import build_prices
    prices_path = build_prices(tickers=TICKERS, start=START, end=END)
    print(f"  ✓ Prices saved → {prices_path}")

    # ── Step 2: Factors ─────────────────────────────────────────────
    print("\n▸ Step 2/5: Computing factors...")
    from entropy.utils.io import load_parquet
    import pandas as pd

    prices = load_parquet(prices_path)
    prices["date"] = pd.to_datetime(prices["date"])

    from entropy.factors.registry import FactorRegistry
    reg = FactorRegistry()
    reg.discover()

    factor_df = reg.compute_all(
        prices,
        factor_names=["MOM_12_1M", "VOL_20D", "ILLIQ_AMIHUD"],
    )
    factors_path = reg.save_factors(factor_df)
    print(f"  ✓ 3 factors computed → {factors_path}")

    # ── Step 3: Portfolio ───────────────────────────────────────────
    print("\n▸ Step 3/5: Building portfolio...")
    from entropy.portfolio.construction import PortfolioConfig, PortfolioMode, WeightScheme
    from entropy.portfolio.quantile import QuantilePortfolio
    from entropy.portfolio.rebalance import rebalance_dates, carry_forward_weights
    from entropy.data.calendar import trading_dates

    # Use all tickers as tradable universe
    universe = prices[["date", "ticker"]].drop_duplicates().copy()
    universe["pass_all_filters"] = True

    signal = factor_df[["date", "ticker", "MOM_12_1M"]].dropna()

    config = PortfolioConfig(
        mode=PortfolioMode.LONG_ONLY,
        weight_scheme=WeightScheme.EQUAL,
        rebalance_freq="M",
        n_quantiles=5,
        max_stock_weight=0.20,  # relaxed for 10 stocks
    )

    reb = rebalance_dates("M", start=START, end=END)
    constructor = QuantilePortfolio(config)
    weights = constructor.build(signal, universe, reb)

    all_dates = trading_dates(start=str(weights["date"].min().date()),
                              end=str(weights["date"].max().date()))
    daily_weights = carry_forward_weights(weights, all_dates)

    from entropy.utils.io import resolve_data_path, save_parquet
    wpath = resolve_data_path("portfolio", "weights_quickstart.parquet")
    save_parquet(daily_weights, wpath)
    print(f"  ✓ Portfolio weights saved → {wpath}")

    # ── Step 4: Backtest ────────────────────────────────────────────
    print("\n▸ Step 4/5: Simulating execution...")
    from entropy.trading.costs import CostModel
    from entropy.trading.execution import simulate_execution
    from entropy.trading.pnl import compute_daily_returns, performance_summary

    cm = CostModel()
    trades = simulate_execution(daily_weights, prices, cm, initial_capital=1_000_000)
    daily_pnl = compute_daily_returns(daily_weights, prices, trades, cm, 1_000_000)
    perf = performance_summary(daily_pnl)

    # Save for report
    backtest_dir = resolve_data_path("backtest")
    backtest_dir.mkdir(parents=True, exist_ok=True)
    save_parquet(trades, backtest_dir / "trades.parquet")
    save_parquet(daily_pnl.reset_index(), backtest_dir / "daily_pnl.parquet")
    pd.DataFrame([perf]).to_csv(backtest_dir / "performance_summary.csv", index=False)

    print(f"  ✓ Backtest complete")
    print(f"    Gross: {perf['gross_ann_return']:+.2%} ann | Sharpe {perf['gross_sharpe']:.2f}")
    print(f"    Net:   {perf['net_ann_return']:+.2%} ann | Sharpe {perf['net_sharpe']:.2f}")
    print(f"    MaxDD: {perf['net_max_drawdown']:.2%}")

    # ── Step 5: Report ──────────────────────────────────────────────
    print("\n▸ Step 5/5: Generating research report...")
    from entropy.evaluation.report import generate_report
    report_path = generate_report(
        signal_col="MOM_12_1M",
        run_walkforward=False,   # skip for speed
        run_ablation=True,
    )
    print(f"  ✓ Report saved → {report_path}")

    print("\n" + "=" * 60)
    print("  Done! Open the report in your browser:")
    print(f"  file:///{report_path.resolve()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
