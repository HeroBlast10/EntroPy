#!/usr/bin/env python
"""Explicit paper runner; defaults to one read-only, dry cycle.

For connectivity, use ibkr_diagnose.py. This remains a demo runner, not
an exchange-calendar scheduler or a production order management system.
"""
from __future__ import annotations

import datetime as dt
import json
import sys
from dataclasses import asdict
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

import click
from loguru import logger
from quant_platform.core.execution.paper.ibkr.config import load_paper_config
from quant_platform.core.utils.io import set_project_root


@click.command()
@click.option("--config", "config_path", type=click.Path(exists=True, path_type=Path),
              default=_project_root / "config/ibkr_paper.yaml", show_default=True)
@click.option("--host", default=None)
@click.option("--port", type=int, default=None)
@click.option("--client-id", type=int, default=None)
@click.option("--account", default=None, help="Explicit DU paper account.")
@click.option("--tickers", "-t", multiple=True, help="Repeat -t for each symbol.")
@click.option("--interval", type=int, default=None)
@click.option("--capital", type=float, default=None, help="Provisional USD cap, not HKD.")
@click.option("--order-type", type=click.Choice(["MKT", "LMT", "ADAPTIVE"]), default=None)
@click.option("--limit-offset", type=float, default=None)
@click.option("--weights-file", type=click.Path(exists=True, path_type=Path), default=None)
@click.option("--dry-run/--no-dry-run", default=None)
@click.option("--readonly/--readwrite", default=None)
@click.option("--submit-paper-orders", is_flag=True,
              help="Requires DU account, readwrite, no-dry-run and explicit weights.")
@click.option("--max-order-notional", type=float, default=None)
@click.option("--max-daily-loss", type=float, default=None)
@click.option("--max-positions", type=int, default=None)
@click.option("--kill-switch", is_flag=True)
@click.option("--run-once/--loop", default=True, help="Default: one cycle. Loop is a demo timer.")
@click.option("--show-config", is_flag=True, help="Print effective config without connecting.")
def main(config_path, host, port, client_id, account, tickers, interval, capital,
         order_type, limit_offset, weights_file, dry_run, readonly, submit_paper_orders,
         max_order_notional, max_daily_loss, max_positions, kill_switch, run_once, show_config):
    """EntroPy IB Gateway paper runner (orders disabled by default)."""
    set_project_root(_project_root)
    try:
        config = load_paper_config(config_path)
        for attr, value in [("host", host), ("port", port), ("client_id", client_id),
                            ("account", account), ("readonly", readonly)]:
            if value is not None:
                setattr(config.ib, attr, value)
        for attr, value in [("rebalance_interval_sec", interval), ("target_capital", capital),
                            ("order_type", order_type), ("limit_offset_bps", limit_offset),
                            ("dry_run", dry_run)]:
            if value is not None:
                setattr(config.strategy, attr, value)
        if tickers:
            config.strategy.tickers = list(tickers)
        if weights_file:
            config.strategy.weights_file = str(weights_file.resolve())
        elif config.strategy.weights_file:
            path = Path(config.strategy.weights_file)
            config.strategy.weights_file = str((path if path.is_absolute() else _project_root / path).resolve())
        config.strategy.enable_paper_orders = submit_paper_orders
        for attr, value in [("max_order_notional", max_order_notional),
                            ("max_daily_loss", max_daily_loss), ("max_positions", max_positions)]:
            if value is not None:
                setattr(config.risk, attr, value)
        if kill_switch:
            config.risk.kill_switch = True
        config.validate()
    except (ValueError, OSError) as exc:
        raise click.ClickException(str(exc)) from exc
    view = asdict(config)
    if view["ib"]["account"]:
        view["ib"]["account"] = "DU***"
    click.echo(json.dumps(view, indent=2, ensure_ascii=False))
    if show_config:
        return

    from quant_platform.core.execution.paper.ibkr.strategy import PaperTradingStrategy
    from quant_platform.core.execution.paper.ibkr.market_data import MarketDataManager
    from quant_platform.core.execution.paper.ibkr.execution import OrderManager
    from quant_platform.core.execution.paper.ibkr.portfolio import PortfolioTracker

    log_dir = _project_root / config.log_dir
    log_dir.mkdir(parents=True, exist_ok=True)
    sink = logger.add(str(log_dir / f"paper_trade_{dt.datetime.now():%Y%m%d_%H%M%S}.log"), rotation="10 MB")
    strategy = PaperTradingStrategy(config)
    try:
        if not run_once:
            strategy.start()
            return
        try:
            strategy.gateway.connect()
            strategy.md = MarketDataManager(strategy.gateway)
            strategy.orders = OrderManager(strategy.gateway, strategy.risk_mgr, config)
            strategy.portfolio = PortfolioTracker(strategy.gateway)
            strategy.portfolio.subscribe_pnl()
            strategy.md.subscribe(config.strategy.tickers)
            strategy.gateway.sleep(5)
            strategy.run_once()
        finally:
            strategy.shutdown()
    finally:
        logger.remove(sink)


if __name__ == "__main__":
    main()
