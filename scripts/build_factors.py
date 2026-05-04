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
@click.option("--use-cache/--no-cache", default=True,
              help="Use shared price feature cache during factor computation.")
def main(factors, evaluate, list_factors, periods, use_cache):
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

    # --- Load fundamentals (optional — needed for value/quality factors) ---
    fundamentals = None
    fund_path = resolve_data_path(cfg["paths"].get("fundamentals_dir", "fundamentals"), "fundamentals.parquet")
    if fund_path.exists():
        fundamentals = load_parquet(fund_path)
        fundamentals["date"] = pd.to_datetime(fundamentals["date"])
        click.echo(f"✓ Fundamentals loaded ({len(fundamentals)} rows)")
    else:
        click.echo("⚠ Fundamentals not found — value/quality factors will be skipped")

    # --- Compute ---
    factor_names = list(factors) if factors else None
    factor_df = reg.compute_all(
        prices,
        fundamentals=fundamentals,
        factor_names=factor_names,
        use_cache=use_cache,
    )
    out_path = reg.save_factors(factor_df)
    click.echo(f"\n✓ Factors saved → {out_path}")

    # --- Evaluate ---
    if evaluate:
        from quant_platform.core.signals.cross_sectional.evaluation import (
            add_forward_returns,
            compare_factors,
            factor_tearsheet,
        )
        from quant_platform.core.signals.catalog import save_factor_catalog
        from quant_platform.core.signals.evaluation.router import evaluate_signal
        from quant_platform.core.signals.factor_selection import (
            apply_deployability_filters,
            apply_multiple_testing_controls,
        )

        prices_with_fwd = add_forward_returns(prices, periods=list(periods))
        price_eval_cols = ["date", "ticker"] + [f"fwd_ret_{p}d" for p in periods]
        for extra_col in ("adj_close", "close", "volume", "amount"):
            if extra_col in prices_with_fwd.columns:
                price_eval_cols.append(extra_col)
        eval_df = factor_df.merge(
            prices_with_fwd[price_eval_cols],
            on=["date", "ticker"],
            how="inner",
        )

        # Route each factor to its type-specific scorecard
        factor_cols = [c for c in factor_df.columns if c not in ("date", "ticker")]
        tearsheets = {}          # CS factors — classic IC tearsheet
        typed_results = {}       # all factors — type-specific scorecard

        for fc in factor_cols:
            try:
                # Look up signal_type from registry
                factor_cls = reg.get(fc)
                meta = factor_cls.meta

                if meta.signal_type == "cross_sectional":
                    # Standard IC / RankIC tearsheet for cross-sectional factors
                    ts = factor_tearsheet(
                        eval_df,
                        fc,
                        return_col="fwd_ret_1d",
                        direction=meta.direction,
                        forward_periods=list(periods),
                    )
                    tearsheets[fc] = ts
                else:
                    # Type-specific scorecard (time_series / regime / relative_value)
                    result = evaluate_signal(
                        signal_df=eval_df,
                        signal_col=fc,
                        meta=meta,
                        prices=prices,
                    )
                    typed_results[fc] = result
                    logger.info("Evaluated {} ({}) → {}", fc, meta.signal_type, result)
            except Exception as exc:
                logger.error("Evaluation failed for {}: {}", fc, exc)

        comparison = None
        # Cross-sectional comparison table (IC-based)
        if tearsheets:
            comparison = compare_factors(tearsheets)
            comparison = apply_multiple_testing_controls(comparison, tearsheets)
            comparison = apply_deployability_filters(comparison)
            click.echo("\n" + "=" * 80)
            click.echo("FACTOR COMPARISON — Cross-Sectional (1-day forward return)")
            click.echo("=" * 80)
            click.echo(comparison.to_string())

            # Save comparison
            comp_path = resolve_data_path("factors", "factor_comparison.csv")
            comp_path.parent.mkdir(parents=True, exist_ok=True)
            comparison.to_csv(comp_path)
            click.echo(f"\n✓ Comparison saved → {comp_path}")

        # Type-specific results summary
        if typed_results:
            click.echo("\n" + "=" * 80)
            click.echo("TYPE-SPECIFIC EVALUATION RESULTS")
            click.echo("=" * 80)
            for fc, result in typed_results.items():
                factor_cls = reg.get(fc)
                click.echo(f"\n  {fc} ({factor_cls.meta.signal_type}):")
                for k, v in result.items():
                    if isinstance(v, float):
                        click.echo(f"    {k}: {v:.4f}")
                    else:
                        click.echo(f"    {k}: {v}")

            # Persist type-specific results
            import json
            import numpy as _np
            typed_path = resolve_data_path("factors", "typed_factor_evaluation.json")
            typed_path.parent.mkdir(parents=True, exist_ok=True)
            serializable = {}
            for fc, result in typed_results.items():
                factor_cls = reg.get(fc)
                entry = {"signal_type": factor_cls.meta.signal_type}
                for k, v in result.items():
                    entry[k] = float(v) if isinstance(v, (int, float, _np.floating)) else str(v)
                serializable[fc] = entry
            typed_path.write_text(json.dumps(serializable, indent=2), encoding="utf-8")
            click.echo(f"\n✓ Type-specific evaluation saved → {typed_path}")

        catalog_path = save_factor_catalog(reg, comparison=comparison, typed_results=typed_results)
        click.echo(f"\nUnified factor catalog saved -> {catalog_path}")


if __name__ == "__main__":
    main()
