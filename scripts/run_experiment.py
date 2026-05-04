#!/usr/bin/env python
"""Run a YAML-configured EntroPy experiment."""

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
@click.option("--config", "-c", type=str, default=None,
              help="Experiment YAML path, e.g. quant_platform/experiments/us_baseline.yaml.")
@click.option("--output-dir", type=str, default=None,
              help="Override output directory. Default: data/experiments/<name>.")
@click.option("--list", "list_only", is_flag=True, default=False,
              help="List available experiments and exit.")
@click.option("--no-backtest", is_flag=True, default=False,
              help="Only build selected-factor portfolio weights.")
def main(config, output_dir, list_only, no_backtest):
    """Run configured factor research experiments."""
    set_project_root(_project_root)

    from quant_platform.core.experiments.runner import ExperimentRunner, list_experiments

    if list_only:
        table = list_experiments(_project_root / "quant_platform" / "experiments")
        click.echo(table.to_string(index=False))
        return

    if not config:
        raise click.UsageError("--config is required unless --list is used")

    runner = ExperimentRunner(config, output_dir=output_dir)
    result = runner.run(run_backtest=not no_backtest)

    click.echo(f"\nExperiment complete: {result.name}")
    click.echo(f"  Output dir       : {result.output_dir}")
    click.echo(f"  Selected factors : {', '.join(result.selected_factors)}")
    if result.performance:
        click.echo(f"  Net Sharpe       : {result.performance.get('net_sharpe', 0):.2f}")
        click.echo(f"  Net ann return   : {result.performance.get('net_ann_return', 0):+.2%}")


if __name__ == "__main__":
    main()

