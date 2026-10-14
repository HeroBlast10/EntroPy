"""Broker-boundary regressions. All broker operations are replaced with fakes."""
from __future__ import annotations

import datetime as dt
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace as NS

import pandas as pd
import pytest
# IBKR is an optional surface in the existing CI environment.
pytest.importorskip("ib_insync")
from eventkit import Event
from ib_insync import MarketOrder, OrderStatus, Stock, Trade

from quant_platform.core.execution.paper.ibkr.config import IBConfig, PaperTradingConfig, RiskLimits, load_paper_config
from quant_platform.core.execution.paper.ibkr.gateway import IBGateway, readable_message
from quant_platform.core.execution.paper.ibkr.execution import OrderManager
from quant_platform.core.execution.paper.ibkr.portfolio import PortfolioTracker
from quant_platform.core.execution.paper.ibkr.risk import RiskManager, trading_date
from quant_platform.core.execution.paper.ibkr.strategy import PaperTradingStrategy
from scripts.ibkr_diagnose import diagnose
from scripts.paper_trade import main
from click.testing import CliRunner

ROOT = Path(__file__).resolve().parents[1]


def test_unicode_ib_message_keeps_other_backslashes():
    assert readable_message(r"\u4f1a\u8bdd C:\new") == "会话 C:\\new"


class FakeIB:
    def __init__(self, accounts=None, currency="USD"):
        self.accounts = accounts or ["DU123456"]
        self.currency = currency
        self.connectedEvent = Event()
        self.disconnectedEvent = Event()
        self.errorEvent = Event()
        self.orderStatusEvent = Event()
        self.execDetailsEvent = Event()
        self.client = NS(serverVersion=lambda: 176)
        self.disconnected = False
        self.sent = []
        self.cancelled = []
        self.pos = []
        self.trades = []
        self.pnl_value = NS(dailyPnL=0.0, unrealizedPnL=0.0, realizedPnL=0.0)

    def connect(self, **kwargs):
        self.connect_args = kwargs

    def disconnect(self):
        self.disconnected = True

    def isConnected(self):
        return not self.disconnected

    def managedAccounts(self):
        return self.accounts

    def accountValues(self, account):
        return [NS(tag="Currency", currency="BASE", value=self.currency),
                NS(tag="NetLiquidation", currency="BASE", value="100000"),
                NS(tag="SettledCash", currency="USD", value="12000")]

    def accountSummary(self, account):
        return []

    def positions(self, account):
        assert account in self.accounts
        return self.pos

    def qualifyContracts(self, contract):
        contract.conId = 42
        return [contract]

    def openTrades(self):
        return self.trades

    def placeOrder(self, contract, order):
        order.orderId = len(self.sent) + 1
        order.clientId = 17
        trade = Trade(contract, order, OrderStatus(status="Submitted", remaining=order.totalQuantity))
        self.sent.append(trade)
        self.trades.append(trade)
        return trade

    def cancelOrder(self, order):
        self.cancelled.append(order)

    def reqGlobalCancel(self):
        pytest.fail("Global cancellation must never be called")

    def reqPnL(self, account, model):
        return self.pnl_value

    def cancelPnL(self, account, model):
        pass

    def reqCurrentTime(self):
        return dt.datetime.now(dt.timezone.utc)

    def sleep(self, seconds=0):
        pass


def connected(fake=None, cfg=None):
    gw = IBGateway(cfg or IBConfig())
    gw._ib = fake or FakeIB()
    gw._connected = True
    return gw


def order_manager(tmp_path, *, readonly=False, dry_run=False, enabled=True):
    cfg = PaperTradingConfig()
    cfg.ib = IBConfig(readonly=readonly, account="DU123456")
    cfg.strategy.dry_run = dry_run
    cfg.strategy.enable_paper_orders = enabled
    cfg.strategy.weights_file = str(tmp_path / "reviewed.parquet")
    risk = RiskManager(cfg.risk)
    risk.update_daily_pnl(0.0)
    gw = connected(cfg=cfg.ib)
    return OrderManager(gw, risk, cfg), gw._ib


def test_default_profile_cannot_submit():
    cfg = load_paper_config(ROOT / "config/ibkr_paper.yaml")
    assert cfg.ib.port == 4002 and cfg.ib.readonly
    assert cfg.strategy.dry_run and not cfg.strategy.enable_paper_orders
    assert cfg.strategy.target_capital == 12000
    result = CliRunner().invoke(main, ["--show-config"])
    assert result.exit_code == 0
    assert '"readonly": true' in result.output


@pytest.mark.parametrize("text", ['ib:\n  readonly: "false"\n', 'ib:\n  readoonly: false\n',
                                  'ib:\n  port: 4001\n', 'strategy:\n  target_capital: .nan\n'])
def test_invalid_yaml_rejected(tmp_path, text):
    path = tmp_path / "invalid.yaml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError):
        load_paper_config(path)


@pytest.mark.parametrize("args", [["--port", "4001"], ["--account", "U123456"],
                                 ["--submit-paper-orders"], ["--capital", "-1"]])
def test_unsafe_cli_rejected_without_connection(args):
    result = CliRunner().invoke(main, args + ["--show-config"])
    assert result.exit_code != 0


def test_selected_account_and_ambiguity():
    fake = FakeIB(["DU_FIRST", "DU_SELECTED"])
    assert connected(fake, IBConfig(account="DU_SELECTED")).account_id == "DU_SELECTED"
    with pytest.raises(ValueError):
        connected(fake).account_id
    with pytest.raises(ValueError):
        connected(fake, IBConfig(account="DU_ABSENT")).account_id
    fake.accounts = []
    with pytest.raises(ValueError):
        connected(fake, IBConfig(account="DU_SELECTED")).account_id


def test_live_session_on_paper_port_is_disconnected(monkeypatch):
    fake = FakeIB(["U123456"])
    monkeypatch.setattr("ib_insync.IB", lambda: fake)
    gw = IBGateway(IBConfig())
    with pytest.raises(ValueError, match="Live/non-DU"):
        gw.connect()
    assert fake.disconnected and not gw.is_connected


def test_non_usd_base_is_not_mislabelled_as_usd():
    gw = connected(FakeIB(currency="HKD"))
    assert gw.base_currency() == "HKD"
    with pytest.raises(ValueError, match="unavailable"):
        gw.net_liquidation("USD")


def test_base_currency_placeholder_uses_summary_currency():
    fake = FakeIB(currency="BASE")
    fake.accountSummary = lambda account: [NS(tag="NetLiquidation", currency="HKD", value="1000000")]
    gw = connected(fake)
    assert gw.base_currency() == "HKD"
    with pytest.raises(ValueError):
        gw.net_liquidation("USD")


def test_diagnostic_retains_handshake_upgrade_notice(monkeypatch):
    fake = FakeIB()
    def connect(**kwargs):
        fake.connect_args = kwargs
        fake.errorEvent.emit(-1, 2172, "Upgrade required before 20261215", None)
    fake.connect = connect
    monkeypatch.setattr("ib_insync.IB", lambda: fake)
    monkeypatch.setattr("socket.create_connection", lambda *a, **k: nullcontext())
    report = diagnose(ROOT / "config/ibkr_paper.yaml", connect=True)
    assert report["warnings"] == [{"code": 2172, "message": "Upgrade required before 20261215"}]
    assert not report["errors"]


@pytest.mark.parametrize("readonly,dry_run,enabled", [(True, False, True), (False, True, False), (False, False, False)])
def test_non_trading_modes_never_submit_or_cancel(tmp_path, readonly, dry_run, enabled):
    om, fake = order_manager(tmp_path, readonly=readonly, dry_run=dry_run, enabled=enabled)
    assert om.place_limit_order("AAPL", "BUY", 1, 100.0) is None
    om.cancel_all()
    assert fake.sent == [] and fake.cancelled == []


def test_account_bound_submission_and_owned_cancel(tmp_path):
    om, fake = order_manager(tmp_path)
    own = om.place_limit_order("AAPL", "BUY", 1, 100.0)
    assert own.order.account == "DU123456" and not own.order.outsideRth
    foreign = Trade(Stock("MSFT", "SMART", "USD"), MarketOrder("BUY", 1), OrderStatus(status="Submitted"))
    om.cancel_order(foreign)
    om.cancel_all()
    assert fake.cancelled == [own.order]


def test_missing_price_and_pending_order_block_new_submission(tmp_path):
    om, fake = order_manager(tmp_path)
    assert om.place_market_order("AAPL", "BUY", 1) is None
    assert om.place_limit_order("AAPL", "BUY", 1, 100) is not None
    assert om.place_limit_order("AAPL", "BUY", 1, 100) is None
    assert len(fake.sent) == 1


def test_reduce_position_at_limit_and_missing_pnl():
    rm = RiskManager(RiskLimits(max_positions=1, max_position_notional=1000))
    assert rm.check_order("SELL", 1, 100, 1, 1000)[0]
    assert not rm.check_order("BUY", 1, 100, 1, 1000)[0]
    assert not rm.check_order("SELL", 11, 100, 1, 1000)[0]
    assert not rm.check_order("BUY", 1, 100)[0]  # unavailable PnL
    rm.update_daily_pnl(0)
    assert rm.check_order("BUY", 1, 100)[0]
    rm.update_daily_pnl(-120)
    assert not rm.check_order("BUY", 1, 100)[0]
    assert rm.check_order("SELL", 1, 100, 1, 1000)[0]


@pytest.mark.parametrize("value", [None, float("nan"), float("inf"), 1.7976931348623157e308])
def test_unavailable_pnl_remains_unknown(value):
    gw = connected()
    pt = PortfolioTracker(gw)
    assert pt.pnl_summary()["daily_pnl"] is None
    gw._ib.pnl_value.dailyPnL = value
    pt.subscribe_pnl()
    assert pt.pnl_summary()["daily_pnl"] is None


def test_unrelated_holdings_are_not_liquidated():
    pt = PortfolioTracker(connected())
    pt.position_dict = lambda: {"AAPL": 10, "OTHER": 100}
    orders = pt.compute_rebalance({"AAPL": 0.0}, {"AAPL": 100, "OTHER": 100}, 12000)
    assert [o["symbol"] for o in orders] == ["AAPL"]


def test_no_signal_fallback_and_preserve_cash_weight(tmp_path):
    cfg = PaperTradingConfig(log_dir=str(tmp_path / "logs"), state_dir=str(tmp_path / "state"))
    strategy = PaperTradingStrategy(cfg)
    assert strategy._compute_signal({"AAPL": 100}) == {}
    path = tmp_path / "weights.parquet"
    pd.DataFrame({"date": [trading_date()], "ticker": ["AAPL"], "weight": [0.5]}).to_parquet(path)
    cfg.strategy.weights_file = str(path)
    assert strategy._compute_signal({"AAPL": 100}) == {"AAPL": 0.5}
    pd.DataFrame({"date": [trading_date() + dt.timedelta(days=1)], "ticker": ["AAPL"], "weight": [0.5]}).to_parquet(path)
    assert strategy._compute_signal({"AAPL": 100}) == {}


def test_connected_diagnostic_has_no_trading_side_effects(monkeypatch):
    fake = FakeIB()
    monkeypatch.setattr("ib_insync.IB", lambda: fake)
    monkeypatch.setattr("socket.create_connection", lambda *a, **k: nullcontext())
    report = diagnose(ROOT / "config/ibkr_paper.yaml", connect=True)
    assert report["account_status"] == "paper_connected"
    assert fake.connect_args["readonly"] is True
    assert report["account"] == "DU***" and report["pnl"]["daily_pnl"] == 0
    assert fake.sent == [] and fake.cancelled == [] and fake.disconnected


def test_missing_settled_cash_blocks_buy(tmp_path):
    om, fake = order_manager(tmp_path)
    fake.accountValues = lambda account: []
    assert om.place_limit_order("AAPL", "BUY", 1, 100) is None
    assert fake.sent == []


def test_external_or_duplicate_fills_do_not_pollute_session(tmp_path):
    om, fake = order_manager(tmp_path)
    own = om.place_limit_order("AAPL", "BUY", 1, 100)
    execution = NS(orderId=own.order.orderId, acctNumber="DU_OTHER", clientId=17,
                   shares=1, price=100, side="BOT", execId="execution-1")
    fill = NS(execution=execution, commissionReport=None)
    om._on_exec_details(own, fill)
    assert om._fill_log == []
    execution.acctNumber = "DU123456"
    om._on_exec_details(own, fill)
    om._on_exec_details(own, fill)
    assert len(om._fill_log) == 1 and om._risk.session.trades_count == 1


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1.0])
def test_invalid_price_cannot_pass_risk(value):
    rm = RiskManager(RiskLimits())
    rm.update_daily_pnl(0)
    assert not rm.check_order("BUY", 1, value)[0]


def test_shutdown_disconnects_even_when_cancel_fails(tmp_path):
    cfg = PaperTradingConfig(log_dir=str(tmp_path / "logs"), state_dir=str(tmp_path / "state"))
    strategy = PaperTradingStrategy(cfg)
    strategy.gateway = connected()
    calls = []
    def fail():
        raise RuntimeError("cancel failed")
    strategy.orders = NS(cancel_all=fail, save_fill_log=lambda: calls.append("saved"))
    with pytest.raises(RuntimeError, match="cancel failed"):
        strategy.shutdown()
    assert strategy.gateway._ib.disconnected and calls == ["saved"]
