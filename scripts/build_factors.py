#!/usr/bin/env python
"""CLI entry point for factor computation and evaluation.

Usage examples::

    # Compute all factors
    python scripts/build_factors.py

    # Compute specific factors
    python scripts/build_factors.py --factors MOM_12_1M VOL_20D ILLIQ_AMIHUD

    # Compute + evaluate (IC tearsheets)
    python scripts/build_factors.py --evaluate

    # List available factors
    python scripts/build_factors.py --list
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
@click.option("--factors", "-f", multiple=True, default=None,
              help="Factor names to compute. Omit to compute all.")
@click.option("--evaluate", "-e", is_flag=True, default=False,
              help="Run IC / RankIC evaluation after computing.")
@click.option("--list", "list_factors", is_flag=True, default=False,
              help="List all available factors and exit.")
@click.option("--periods", "-p", multiple=True, type=int, default=[1, 5, 10, 20],
              help="Forward return periods (trading days) for evaluation.")
def main(factors, evaluate, list_factors, periods):
    """EntroPy — Compute and evaluate alpha factors."""
    set_project_root(_project_root)

    from quant_platform.core.signals.registry import FactorRegistry

    reg = FactorRegistry()
    reg.discover()

    # --- List mode ---
    if list_factors:
        summary = reg.list_factors()
        click.echo(summary.to_string(index=False))
        return

    # --- Load prices ---
    from quant_platform.core.utils.io import load_config, resolve_data_path, load_parquet

    cfg = load_config()
    prices_path = resolve_data_path(cfg["paths"]["prices_dir"], "prices.parquet")
    if not prices_path.exists():
        click.echo(f"ERROR: prices not found at {prices_path}. Run build_dataset.py first.")
        sys.exit(1)

    prices = load_parquet(prices_path)
    prices["date"] = pd.to_datetime(prices["date"])

    # --- Compute ---
    factor_names = list(factors) if factors else None
    factor_df = reg.compute_all(prices, factor_names=factor_names)
    out_path = reg.save_factors(factor_df)
    click.echo(f"\n✓ Factors saved → {out_path}")

    # --- Evaluate ---
    if evaluate:
        from quant_platform.core.signals.cross_sectional.evaluation import (
            add_forward_returns,
            compare_factors,
            factor_tearsheet,
        )

        prices_with_fwd = add_forward_returns(prices, periods=list(periods))
        eval_df = factor_df.merge(
            prices_with_fwd[["date", "ticker"] + [f"fwd_ret_{p}d" for p in periods]],
            on=["date", "ticker"],
            how="inner",
        )

        # Evaluate each factor against 1-day forward return
        factor_cols = [c for c in factor_df.columns if c not in ("date", "ticker")]
        tearsheets = {}
        for fc in factor_cols:
            try:
                ts = factor_tearsheet(eval_df, fc, return_col="fwd_ret_1d")
                tearsheets[fc] = ts
            except Exception as exc:
                logger.error("Evaluation failed for {}: {}", fc, exc)

        if tearsheets:
            comparison = compare_factors(tearsheets)
            click.echo("\n" + "=" * 80)
            click.echo("FACTOR COMPARISON (1-day forward return)")
            click.echo("=" * 80)
            click.echo(comparison.to_string())

            # Save comparison
            comp_path = resolve_data_path("factors", "factor_comparison.csv")
            comp_path.parent.mkdir(parents=True, exist_ok=True)
            comparison.to_csv(comp_path)
            click.echo(f"\n✓ Comparison saved → {comp_path}")


if __name__ == "__main__":
    main()
