#!/usr/bin/env python
"""CLI entry point for the trading / backtest simulation.

Usage examples::

    # Run backtest with default cost model
    python scripts/run_backtest.py

    # Custom cost parameters
    python scripts/run_backtest.py --slippage-bps 10 --impact-coeff 0.15

    # Specify weights file and initial capital
    python scripts/run_backtest.py --weights data/portfolio/weights_quantile_long_only_M.parquet \
                                   --capital 5000000

    # Sensitivity: zero-cost backtest (gross only)
    python scripts/run_backtest.py --slippage-bps 0 --impact-coeff 0 --commission-pct 0
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
@click.option("--weights", "-w", type=str, default=None,
              help="Path to portfolio weights Parquet. Auto-detects if omitted.")
@click.option("--capital", type=float, default=1_000_000.0,
              help="Initial capital ($).")
@click.option("--slippage-bps", type=float, default=5.0,
              help="One-way slippage in basis points.")
@click.option("--impact-coeff", type=float, default=0.1,
              help="Market impact coefficient (Almgren-Chriss).")
@click.option("--impact-exponent", type=float, default=0.5,
              help="Impact exponent (0.5 = sqrt, 1.0 = linear).")
@click.option("--commission-per-share", type=float, default=0.005,
              help="Commission per share ($).")
@click.option("--commission-pct", type=float, default=0.0,
              help="Commission as %% of notional (overrides per-share if > 0).")
@click.option("--borrow-rate", type=float, default=0.005,
              help="Annual borrow rate for short positions.")
@click.option("--sec-fee-rate", type=float, default=8e-6,
              help="SEC fee rate (sells only).")
def main(weights, capital, slippage_bps, impact_coeff, impact_exponent,
         commission_per_share, commission_pct, borrow_rate, sec_fee_rate):
    """EntroPy — Run backtest simulation with transaction costs."""
    set_project_root(_project_root)

    from entropy.trading.costs import CostModel
    from entropy.trading.pipeline import run_trading_pipeline

    cost_model = CostModel(
        slippage_bps=slippage_bps,
        impact_coeff=impact_coeff,
        impact_exponent=impact_exponent,
        commission_per_share=commission_per_share,
        commission_pct=commission_pct,
        borrow_rate_annual=borrow_rate,
        sec_fee_rate=sec_fee_rate,
    )

    result = run_trading_pipeline(
        weights_path=weights,
        cost_model=cost_model,
        initial_capital=capital,
    )

    out = result["output_dir"]
    perf = result["performance"]

    click.echo(f"\n✓ Backtest complete. Outputs → {out}/")
    click.echo(f"  trades.parquet         — {len(result['trades']):,} trades")
    click.echo(f"  daily_pnl.parquet      — {len(result['daily_pnl']):,} days")
    click.echo(f"  cost_attribution.csv")
    click.echo(f"  performance_summary.csv")
    click.echo("")
    click.echo(f"  Gross: {perf.get('gross_ann_return', 0):+.2%} ann  "
               f"Sharpe {perf.get('gross_sharpe', 0):.2f}  "
               f"MaxDD {perf.get('gross_max_drawdown', 0):.2%}")
    click.echo(f"  Net:   {perf.get('net_ann_return', 0):+.2%} ann  "
               f"Sharpe {perf.get('net_sharpe', 0):.2f}  "
               f"MaxDD {perf.get('net_max_drawdown', 0):.2%}")


if __name__ == "__main__":
    main()
