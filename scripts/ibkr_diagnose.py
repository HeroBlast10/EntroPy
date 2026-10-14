#!/usr/bin/env python
"""Read-only IBKR preflight: no order submission or cancellation.

Default: inspect config, SDK and socket. --connect checks a paper session.
--quotes and --history separately test market data permissions.
"""
from __future__ import annotations

import argparse
import datetime as dt
import importlib.metadata
import json
import re
import socket
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from quant_platform.core.execution.paper.ibkr.config import load_paper_config


def masked(text: str) -> str:
    return re.sub(r"\b(?:DU[A-Z]*|U)\d+\b", "[account]", text)


def diagnose(config_path: Path, connect=False, quotes=False, history=False) -> dict:
    cfg = load_paper_config(config_path)
    cfg.ib.readonly = True
    cfg.strategy.dry_run = True
    cfg.strategy.enable_paper_orders = False
    report = {"checked_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
              "config": str(config_path.resolve()), "host": cfg.ib.host,
              "port": cfg.ib.port, "client_id": cfg.ib.client_id,
              "readonly": True, "orders_enabled": False,
              "target_capital_usd": cfg.strategy.target_capital,
              "account_status": "not_checked", "data_status": "not_checked",
              "history_status": "not_checked", "warnings": [], "errors": []}
    try:
        report["ib_insync_version"] = importlib.metadata.version("ib_insync")
    except importlib.metadata.PackageNotFoundError:
        report["ib_insync_version"] = None
        report["errors"].append("ib_insync is not installed in this Python environment.")
    try:
        with socket.create_connection((cfg.ib.host, cfg.ib.port), timeout=1):
            report["tcp_open"] = True
    except OSError:
        report["tcp_open"] = False
    if not connect:
        return report
    if not report["tcp_open"] or not report["ib_insync_version"]:
        report["errors"].append("Start IB Gateway, log into Paper Trading and check port 4002.")
        return report

    from quant_platform.core.execution.paper.ibkr.gateway import IBGateway
    from quant_platform.core.execution.paper.ibkr.market_data import MarketDataManager
    from quant_platform.core.execution.paper.ibkr.portfolio import PortfolioTracker
    gateway = IBGateway(cfg.ib)
    md = portfolio = None
    try:
        gateway.connect()
        report["account_status"] = "paper_connected"
        report["account"] = "DU***"
        report["server_version"] = gateway.ib.client.serverVersion()
        report["server_time"] = gateway.ib.reqCurrentTime().isoformat()
        report["base_currency"] = gateway.base_currency()
        tags = {"NetLiquidation", "SettledCash", "TotalCashValue", "AvailableFunds"}
        report["account_values"] = [{"tag": av.tag, "value": av.value, "currency": av.currency}
                                    for av in gateway.ib.accountValues(gateway.account_id) if av.tag in tags]
        portfolio = PortfolioTracker(gateway)
        portfolio.subscribe_pnl()
        gateway.sleep(2)
        report["pnl"] = portfolio.pnl_summary()
        if quotes or history:
            md = MarketDataManager(gateway)
        if quotes:
            md.subscribe(cfg.strategy.tickers)
            gateway.sleep(8)
            report["quotes"] = [md.snapshot(symbol) for symbol in cfg.strategy.tickers]
            report["data_status"] = "quotes_received" if all(
                q.get("mid") is not None for q in report["quotes"]) else "quotes_unavailable_or_partial"
        if history:
            report["history"] = {}
            for symbol in cfg.strategy.tickers:
                bars = md.historical_bars(symbol, duration="1 M", bar_size="1 day")
                report["history"][symbol] = {"rows": len(bars), "available": not bars.empty}
            report["history_status"] = "available" if all(
                x["available"] for x in report["history"].values()) else "unavailable_or_partial"
    except Exception as exc:
        report["errors"].append(masked(f"{type(exc).__name__}: {exc}"))
    finally:
        try:
            if gateway.is_connected:
                if md:
                    md.unsubscribe_all()
                if portfolio:
                    portfolio.close()
        except Exception as exc:
            report["errors"].append(masked(f"Cleanup: {exc}"))
        finally:
            gateway.disconnect()
        # Include events emitted during the connection handshake, not only
        # events arriving after IB.connect has returned.
        for item in gateway.messages:
            code = item["code"]
            if code in {10167, 2172} or code < 2000 or code >= 3000:
                target = "warnings" if code in {10167, 2172} else "errors"
                report[target].append({"code": code, "message": masked(item["message"])})
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config/ibkr_paper.yaml")
    parser.add_argument("--connect", action="store_true")
    parser.add_argument("--quotes", action="store_true")
    parser.add_argument("--history", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / "data/live/ibkr_diagnostic.json")
    args = parser.parse_args()
    if (args.quotes or args.history) and not args.connect:
        parser.error("--quotes and --history require --connect")
    try:
        report = diagnose(args.config, args.connect, args.quotes, args.history)
    except (ValueError, OSError) as exc:
        parser.exit(2, masked(str(exc)) + "\n")
    payload = json.dumps(report, indent=2, ensure_ascii=False, default=str, allow_nan=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(payload, encoding="utf-8")
    print(payload)
    failed = (args.connect and (report["account_status"] != "paper_connected" or report["errors"]))
    failed = failed or (args.quotes and report["data_status"] != "quotes_received")
    failed = failed or (args.history and report["history_status"] != "available")
    return 2 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
