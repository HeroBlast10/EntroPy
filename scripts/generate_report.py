#!/usr/bin/env python
"""CLI entry point for research report generation.

Usage examples::

    # Full report (walk-forward + ablation)
    python scripts/generate_report.py

    # Quick report (skip slow analyses)
    python scripts/generate_report.py --no-walkforward --no-ablation

    # Specify signal factor
    python scripts/generate_report.py --signal MOM_12_1M

    # Auto-select best factor from factor_comparison.csv (ranked by ric_mean_ic)
    python scripts/generate_report.py --auto-best

    # Auto-select best factor by a different metric
    python scripts/generate_report.py --auto-best --optimize-by ric_icir
    python scripts/generate_report.py --auto-best --optimize-by ls_sharpe

    # Custom walk-forward parameters
    python scripts/generate_report.py --wf-train 24 --wf-test 6 --wf-step 6

    # Custom output path
    python scripts/generate_report.py --output reports/my_report.html
"""

from __future__ import annotations

import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

import click
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
@click.option("--signal", "-s", type=str, default=None,
              help="Factor column to highlight in IC analysis.")
@click.option("--auto-best", is_flag=True, default=False,
              help="Auto-select best factor from data/factors/factor_comparison.csv.")
@click.option("--optimize-by", type=str, default="ric_mean_ic",
              show_default=True,
              help="Metric used to rank factors when --auto-best is set "
                   "(e.g. ric_mean_ic, ric_icir, ls_sharpe).")
@click.option("--output", "-o", type=str, default=None,
              help="Output HTML file path.")
@click.option("--no-walkforward", is_flag=True, default=False,
              help="Skip walk-forward validation (faster).")
@click.option("--no-ablation", is_flag=True, default=False,
              help="Skip ablation study (faster).")
@click.option("--wf-train", type=int, default=36,
              help="Walk-forward training window (months).")
@click.option("--wf-test", type=int, default=12,
              help="Walk-forward test window (months).")
@click.option("--wf-step", type=int, default=12,
              help="Walk-forward step size (months).")
def main(signal, auto_best, optimize_by, output, no_walkforward, no_ablation,
         wf_train, wf_test, wf_step):
    """EntroPy — Generate HTML research report."""
    set_project_root(_project_root)

    from quant_platform.core.evaluation.report import generate_report, select_best_factor
    from quant_platform.core.utils.io import resolve_data_path

    # --- Auto-select best factor ---
    if auto_best and signal is None:
        cmp_path = resolve_data_path("factors", "factor_comparison.csv")
        if cmp_path.exists():
            import pandas as pd
            comparison = pd.read_csv(cmp_path, index_col=0)
            signal = select_best_factor(comparison, metric=optimize_by)
            if signal:
                click.echo(f"  Auto-selected factor: {signal} (ranked by {optimize_by})")
            else:
                click.echo("  WARNING: could not determine best factor from comparison CSV.")
        else:
            click.echo(
                f"  WARNING: {cmp_path} not found. "
                "Run 'python scripts/build_factors.py --evaluate' first."
            )

    wf_kwargs = {
        "train_months": wf_train,
        "test_months": wf_test,
        "step_months": wf_step,
    }

    path = generate_report(
        output_path=output,
        signal_col=signal,
        run_walkforward=not no_walkforward,
        run_ablation=not no_ablation,
        walkforward_kwargs=wf_kwargs,
    )

    click.echo(f"\n✓ Report generated → {path}")
    click.echo(f"  Open in browser: file:///{path.resolve()}")


if __name__ == "__main__":
    main()
