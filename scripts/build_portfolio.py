#!/usr/bin/env python
"""CLI entry point for portfolio construction.

Usage examples::

    # Baseline: quantile long-only, monthly rebalance, equal weight
    python scripts/build_portfolio.py

    # Long-short, weekly rebalance, top/bottom quintile
    python scripts/build_portfolio.py --method quantile --mode long_short --freq W

    # Optimised portfolio with risk aversion = 2
    python scripts/build_portfolio.py --method optimize --risk-aversion 2.0

    # Use a specific factor as signal
    python scripts/build_portfolio.py --signal MOM_12_1M

    # Custom constraints
    python scripts/build_portfolio.py --max-stock-weight 0.03 --max-sector-weight 0.25
"""

from __future__ import annotations

import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

import click
import pandas as pd
from loguru import logger

from quant_platform.core.utils.io import set_project_root

logger.remove()
logger.add(sys.stderr, level="INFO", format=(
    "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
    "<level>{message}</level>"
))


@click.command()
@click.option("--method", "-m", type=click.Choice(["quantile", "optimize"]),
              default="quantile", help="Construction method.")
@click.option("--signal", "-s", type=str, default=None,
              help="Factor column to use as alpha signal.")
@click.option("--factors", multiple=True, default=None,
              help="Multiple factor columns to combine into alpha_multi.")
@click.option("--combiner", type=click.Choice(["rolling_icir", "mean_variance", "risk_parity", "orthogonal_incremental"]),
              default="rolling_icir", help="Multi-factor combiner for --factors.")
@click.option("--baseline-factor", multiple=True, default=None,
              help="Baseline factors for orthogonal_incremental combiner.")
@click.option("--regime-col", type=str, default=None,
              help="Regime column that modulates factor weights and exposure.")
@click.option("--mode", type=click.Choice(["long_only", "long_short"]),
              default="long_only", help="Portfolio mode.")
@click.option("--weight", "-w", type=click.Choice(["equal", "market_cap", "signal", "inverse_vol"]),
              default="equal", help="Weighting scheme.")
@click.option("--freq", "-f", type=click.Choice(["D", "W", "M"]),
              default="M", help="Rebalance frequency.")
@click.option("--n-quantiles", type=int, default=5, help="Number of quantiles.")
@click.option("--top-n", type=int, default=None,
              help="Pick top N stocks instead of quantile-based selection.")
@click.option("--max-stock-weight", type=float, default=0.05,
              help="Max weight per stock (default 5%%).")
@click.option("--max-sector-weight", type=float, default=0.30,
              help="Max weight per sector (default 30%%).")
@click.option("--max-turnover", type=float, default=None,
              help="Max one-way turnover per rebalance (e.g. 0.30).")
@click.option("--risk-aversion", type=float, default=1.0,
              help="Risk aversion λ for optimised method.")
@click.option("--no-factor-risk", is_flag=True, default=False,
              help="Disable FactorRiskModel covariance in optimized portfolios.")
@click.option("--turnover-penalty", type=float, default=0.0,
              help="Quadratic turnover penalty for optimized portfolios.")
def main(method, signal, factors, combiner, baseline_factor, regime_col,
         mode, weight, freq, n_quantiles, top_n,
         max_stock_weight, max_sector_weight, max_turnover, risk_aversion,
         no_factor_risk, turnover_penalty):
    """EntroPy — Build portfolio weights from factor signals."""
    set_project_root(_project_root)

    from quant_platform.core.portfolio.construction import PortfolioConfig, PortfolioMode, WeightScheme
    from quant_platform.core.portfolio.pipeline import run_portfolio_pipeline

    config = PortfolioConfig(
        mode=PortfolioMode(mode),
        weight_scheme=WeightScheme(weight),
        max_stock_weight=max_stock_weight,
        max_sector_weight=max_sector_weight,
        max_turnover=max_turnover,
        n_quantiles=n_quantiles,
        top_n=top_n,
        rebalance_freq=freq,
    )

    constructor_kwargs = {}
    if method == "optimize":
        constructor_kwargs["risk_aversion"] = risk_aversion
        constructor_kwargs["use_factor_risk"] = not no_factor_risk
        constructor_kwargs["turnover_penalty"] = turnover_penalty

    result = run_portfolio_pipeline(
        method=method,
        signal_col=signal,
        factor_cols=list(factors) if factors else None,
        config=config,
        multi_factor_method=combiner,
        baseline_factors=list(baseline_factor) if baseline_factor else None,
        regime_col=regime_col,
        **constructor_kwargs,
    )

    if result["output_path"]:
        click.echo(f"\n✓ Portfolio weights saved → {result['output_path']}")

        # Print summary
        w = result["weights"]
        if not w.empty:
            click.echo(f"  Rebalance dates : {w['date'].nunique()}")
            click.echo(f"  Avg holdings    : {w.groupby('date').size().mean():.0f}")
            click.echo(f"  Date range      : {w['date'].min().date()} – {w['date'].max().date()}")
            long_pct = (w["weight"] > 0).mean()
            click.echo(f"  Long positions  : {long_pct:.1%}")
            if config.mode == PortfolioMode.LONG_SHORT:
                short_pct = (w["weight"] < 0).mean()
                click.echo(f"  Short positions : {short_pct:.1%}")
    else:
        click.echo("ERROR: Portfolio construction produced no output.")
        sys.exit(1)


if __name__ == "__main__":
    main()
