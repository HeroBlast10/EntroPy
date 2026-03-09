#!/usr/bin/env python
"""CLI entry point for IB paper-trading demo.

Prerequisites:
  1. Install: pip install ib_insync
  2. Start IB TWS or IB Gateway in **paper-trading** mode
  3. Enable API access: TWS → Edit → Global Config → API → Settings
     - Enable ActiveX and Socket Clients
     - Socket port = 7497 (TWS paper) or 4002 (Gateway paper)
     - Trusted IPs: 127.0.0.1

Usage examples::

    # Default: connect to TWS paper, equal-weight 5 tickers, rebalance every 5 min
    python scripts/paper_trade.py

    # Custom tickers and faster rebalance
    python scripts/paper_trade.py --tickers AAPL MSFT NVDA --interval 60

    # Dry run (log orders but don't submit)
    python scripts/paper_trade.py --dry-run

    # Connect to IB Gateway instead of TWS
    python scripts/paper_trade.py --port 4002

    # Limit order mode with 10 bps offset
    python scripts/paper_trade.py --order-type LMT --limit-offset 10

    # Strict risk limits
    python scripts/paper_trade.py --max-order-notional 10000 --max-daily-loss 5000

    # Kill switch test — activate immediately
    python scripts/paper_trade.py --kill-switch
"""

from __future__ import annotations

import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

import click
from loguru import logger

from quant_platform.core.utils.io import set_project_root

# Configure logger
logger.remove()
logger.add(sys.stderr, level="DEBUG", format=(
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
    "<level>{message}</level>"
))


@click.command()
@click.option("--host", type=str, default="127.0.0.1",
              help="IB TWS/Gateway host.")
@click.option("--port", type=int, default=7497,
              help="IB port (7497=TWS paper, 4002=Gateway paper).")
@click.option("--client-id", type=int, default=1,
              help="IB client ID.")
@click.option("--tickers", "-t", type=str, multiple=True,
              default=["AAPL", "MSFT", "GOOGL", "AMZN", "JPM"],
              help="Tickers to trade (repeat -t for multiple).")
@click.option("--interval", type=int, default=300,
              help="Rebalance interval in seconds.")
@click.option("--capital", type=float, default=100_000.0,
              help="Target portfolio capital (USD).")
@click.option("--order-type", type=click.Choice(["MKT", "LMT", "ADAPTIVE"],
              case_sensitive=False), default="MKT",
              help="Order type.")
@click.option("--limit-offset", type=float, default=5.0,
              help="Limit price offset (bps) for LMT orders.")
@click.option("--dry-run", is_flag=True, default=False,
              help="Log orders without submitting.")
@click.option("--readonly", is_flag=True, default=False,
              help="Market data only, no order submission.")
@click.option("--max-order-notional", type=float, default=50_000.0,
              help="Max single order notional (USD).")
@click.option("--max-daily-loss", type=float, default=10_000.0,
              help="Max daily loss before kill switch (USD).")
@click.option("--max-positions", type=int, default=20,
              help="Max number of concurrent positions.")
@click.option("--kill-switch", is_flag=True, default=False,
              help="Start with kill switch ON (no orders, data only).")
@click.option("--run-once", is_flag=True, default=False,
              help="Run a single rebalance cycle then exit.")
def main(
    host, port, client_id, tickers, interval, capital,
    order_type, limit_offset, dry_run, readonly,
    max_order_notional, max_daily_loss, max_positions,
    kill_switch, run_once,
):
    """EntroPy — IB Paper Trading Demo.

    Connects to IB TWS/Gateway paper account, subscribes to real-time
    market data, computes target weights, and submits rebalance orders
    through a risk-gated order manager.
    """
    set_project_root(_project_root)

    from quant_platform.core.execution.paper.ibkr.config import (
        IBConfig, PaperTradingConfig, RiskLimits, StrategyConfig,
    )
    from quant_platform.core.execution.paper.ibkr.strategy import PaperTradingStrategy

    # Build config from CLI args
    ib_cfg = IBConfig(
        host=host, port=port, client_id=client_id, readonly=readonly,
    )

    risk_cfg = RiskLimits(
        kill_switch=kill_switch,
        max_order_notional=max_order_notional,
        max_daily_loss=max_daily_loss,
        max_positions=max_positions,
    )

    strat_cfg = StrategyConfig(
        tickers=list(tickers),
        rebalance_interval_sec=interval,
        target_capital=capital,
        order_type=order_type.upper(),
        limit_offset_bps=limit_offset,
        dry_run=dry_run,
    )

    config = PaperTradingConfig(
        ib=ib_cfg, risk=risk_cfg, strategy=strat_cfg,
    )

    # Log file
    log_dir = Path(config.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    import datetime as dt
    log_file = log_dir / f"paper_trade_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logger.add(str(log_file), level="DEBUG", rotation="10 MB")
    logger.info("Log file: {}", log_file)

    # Print config summary
    click.echo("\n" + "=" * 60)
    click.echo("  EntroPy — IB Paper Trading Demo")
    click.echo("=" * 60)
    click.echo(f"  Host:       {host}:{port}")
    click.echo(f"  Tickers:    {', '.join(tickers)}")
    click.echo(f"  Capital:    ${capital:,.0f}")
    click.echo(f"  Order type: {order_type}")
    click.echo(f"  Interval:   {interval}s")
    click.echo(f"  Dry run:    {dry_run}")
    click.echo(f"  Kill switch:{kill_switch}")
    click.echo(f"  Log file:   {log_file}")
    click.echo("=" * 60 + "\n")

    # Run
    strategy = PaperTradingStrategy(config)

    if run_once:
        try:
            strategy.gateway.connect()
            strategy.md = __import__(
                "quant_platform.core.execution.paper.ibkr.market_data", fromlist=["MarketDataManager"]
            ).MarketDataManager(strategy.gateway)
            strategy.orders = __import__(
                "quant_platform.core.execution.paper.ibkr.execution", fromlist=["OrderManager"]
            ).OrderManager(strategy.gateway, strategy.risk_mgr, config)
            strategy.portfolio = __import__(
                "quant_platform.core.execution.paper.ibkr.portfolio", fromlist=["PortfolioTracker"]
            ).PortfolioTracker(strategy.gateway)

            strategy.md.subscribe(config.strategy.tickers)
            strategy.gateway.sleep(5)
            strategy.run_once()
        finally:
            strategy.shutdown()
    else:
        strategy.start()

    click.echo("\nDone.")


if __name__ == "__main__":
    main()
