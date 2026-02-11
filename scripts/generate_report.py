#!/usr/bin/env python
"""CLI entry point for research report generation.

Usage examples::

    # Full report (walk-forward + ablation)
    python scripts/generate_report.py

    # Quick report (skip slow analyses)
    python scripts/generate_report.py --no-walkforward --no-ablation

    # Specify signal factor
    python scripts/generate_report.py --signal MOM_12_1M

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

from entropy.utils.io import set_project_root

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
def main(signal, output, no_walkforward, no_ablation, wf_train, wf_test, wf_step):
    """EntroPy — Generate HTML research report."""
    set_project_root(_project_root)

    from entropy.evaluation.report import generate_report

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
