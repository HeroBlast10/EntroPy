#!/usr/bin/env python
"""CLI entry point for the EntroPy data pipeline.

Usage examples::

    # Full build (all steps)
    python scripts/build_dataset.py

    # Only rebuild prices
    python scripts/build_dataset.py --steps prices

    # Rebuild prices + universe with custom date range
    python scripts/build_dataset.py --steps prices universe --start 2015-01-01 --end 2023-12-31

    # Verify existing data integrity
    python scripts/build_dataset.py --verify
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on sys.path
_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

import click
from loguru import logger

from entropy.utils.io import set_project_root

# Configure loguru
logger.remove()
logger.add(sys.stderr, level="INFO", format=(
    "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
    "<level>{message}</level>"
))


@click.command()
@click.option("--steps", "-s", multiple=True, default=None,
              help="Steps to run: prices, fundamentals, universe, manifest. "
                   "Omit to run all.")
@click.option("--start", type=str, default=None, help="Override start date (YYYY-MM-DD).")
@click.option("--end", type=str, default=None, help="Override end date (YYYY-MM-DD).")
@click.option("--tickers", "-t", type=str, default=None,
              help="Comma-separated ticker list (overrides config).")
@click.option("--verify", is_flag=True, default=False,
              help="Verify existing manifest checksums instead of building.")
def main(steps, start, end, tickers, verify):
    """EntroPy — Build or verify the standardised data layer."""
    set_project_root(_project_root)

    if verify:
        from entropy.data.manifest import verify_manifest
        ok = verify_manifest()
        sys.exit(0 if ok else 1)

    from entropy.data.pipeline import run_pipeline

    step_list = list(steps) if steps else None
    ticker_list = [t.strip() for t in tickers.split(",")] if tickers else None

    outputs = run_pipeline(steps=step_list, tickers=ticker_list, start=start, end=end)

    click.echo("\n✓ Build complete. Outputs:")
    for name, path in outputs.items():
        click.echo(f"  {name:15s} → {path}")


if __name__ == "__main__":
    main()
