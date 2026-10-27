"""Financial-accounting and broker-boundary regressions; no network calls."""
import datetime as dt
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace as NS

import exchange_calendars as xc
import numpy as np
import pandas as pd
import pytest

from quant_platform.core.execution.paper.ibkr.config import IBConfig, PaperTradingConfig
from quant_platform.core.execution.paper.ibkr.workflow_research import (
    WorkflowSettings, features, last_complete_session, load_panel, metrics, regular_session,
    replay, target_weights, write_json,
)
from quant_platform.core.execution.paper.ibkr.workflow_broker import (
    PaperBroker, execution_gates, order_plan, usd_account_values,
)
from quant_platform.core.execution.paper.ibkr.workflow_state import Ledger, account_key
from scripts.ibkr_workflow import exclusive_run, main


def settings(**kwargs):
    return WorkflowSettings(strategy_id='test_v1', kind='etf', research_config='x.yaml', **kwargs)


def panel(n=310, symbols=('A', 'B')):
    sessions = xc.get_calendar('XNYS').sessions_in_range('2024-01-02', '2026-01-01')[:n]
    rows = []
    for i, symbol in enumerate(symbols):
        price = 100 * np.exp(np.arange(len(sessions)) * (.001 + i * .0001))
        rows.append(pd.DataFrame({'date': sessions, 'ticker': symbol, 'open': price,
            'high': price * 1.01, 'low': price * .99, 'close': price,
            'adjusted_open': price, 'adjusted_close': price, 'volume': 1e7,
            'momentum': .1, 'volatility': .2, 'reversal': -.01, 'adv20': 1e9}))
    return pd.concat(rows, ignore_index=True)


def test_calendar_close_buffer_early_close_and_dst():
    assert regular_session('2026-10-20T14:00:00Z')
    assert not regular_session('2026-10-20T11:00:00Z')
    assert not regular_session('2026-11-27T18:01:00Z')  # Thanksgiving Friday 13 ET
    assert last_complete_session('2026-11-27T18:05:00Z') == pd.Timestamp('2026-11-25')
    assert last_complete_session('2026-11-27T18:16:00Z') == pd.Timestamp('2026-11-27')
    assert regular_session('2026-12-01T14:31:00Z')
    with pytest.raises(ValueError, match='aware'):
        last_complete_session('2026-10-20')


@pytest.mark.parametrize('kwargs', [{'gross_weight': 1.1}, {'max_weight': .3},
    {'top_k': True}, {'cost_bps': float('nan')}, {'execution_window_seconds': 301}])
def test_invalid_settings_fail(kwargs):
    with pytest.raises(ValueError):
        settings(**kwargs)


def test_current_or_future_prices_cannot_change_prior_signals():
    original = panel()
    changed = original.copy()
    cutoff = sorted(original.date.unique())[-5]
    changed.loc[changed.date >= cutoff, 'adjusted_close'] *= 100
    a, b = features(original), features(changed)
    cols = ['momentum', 'volatility', 'reversal']
    pd.testing.assert_frame_equal(a[a.date < cutoff][cols], b[b.date < cutoff][cols])


def test_replay_does_not_capture_overnight_gap_before_first_purchase():
    data = panel(260, ('A',))
    data.loc[:, ['open', 'close', 'adjusted_open', 'adjusted_close']] = 100.
    date = data.date.iloc[253]
    data.loc[data.date >= date, ['open', 'close', 'adjusted_open', 'adjusted_close']] = 150.
    cfg = settings(capital_usd=10000, gross_weight=.2, max_weight=.2, top_k=1)
    daily, trades = replay(data, ['A'], cfg, 1e6, 5)
    assert daily.nav.iloc[0] == pytest.approx(9997)  # $2 bps + $1 fee, no gap profit
    assert (trades.signal_date < trades.date).all()
    assert daily.cash_usd.min() > 0


def test_replay_accounts_for_open_to_close_and_deducts_costs():
    data = panel(260, ('A',))
    data.loc[:, ['open', 'close', 'adjusted_open', 'adjusted_close']] = 100.
    data.loc[data.index == 253, 'adjusted_close'] = 110.
    cfg = settings(capital_usd=10000, gross_weight=.2, max_weight=.2, top_k=1)
    daily, _ = replay(data, ['A'], cfg, 1e6, 5)
    assert daily.nav.iloc[0] == pytest.approx(10197)
    stressed, _ = replay(data, ['A'], cfg, 1e6, 5, cost_bps=40)
    assert stressed.nav.iloc[0] == pytest.approx(10191)
    assert metrics(daily)['cost_usd'] >= 3


def test_weights_preserve_cash_and_explicit_zero_exits():
    day = panel(1, ('A', 'B')).copy()
    day.loc[day.ticker == 'B', 'momentum'] = -.2
    cfg = settings(top_k=1)
    weights = target_weights(day, ['A', 'B'], cfg, 1e6, 5)
    assert weights == {'A': .1, 'B': 0.0}
    assert sum(weights.values()) < cfg.gross_weight


def cache(tmp_path, frame):
    symbols = frame.ticker.unique().tolist()
    write_json(tmp_path / 'download_manifest.json',
               {'source': 'IBKR TWS API', 'downloaded_at_utc': '2026-01-01T00:00:00Z',
                'bar_size': '1 day', 'useRTH': True, 'failures': {}})
    for symbol in symbols:
        part = frame[frame.ticker == symbol].rename(columns={'ticker': 'symbol'})
        for kind in ('TRADES', 'ADJUSTED_LAST'):
            part.to_parquet(tmp_path / f'{symbol}_{kind}.parquet', index=False)
    return symbols


def test_missing_session_and_trailing_stale_symbol_fail(tmp_path):
    data = panel(310)
    symbols = cache(tmp_path, data)
    result, provenance = load_panel(tmp_path, symbols, now='2026-01-02T12:00:00Z')
    assert not provenance['data_current']
    assert len(result) == len(data)
    broken = data.drop(data[data.ticker == 'A'].index[-1])
    cache(tmp_path, broken)
    with pytest.raises(ValueError, match='Calendar gaps'):
        load_panel(tmp_path, symbols, now='2026-01-02T12:00:00Z')


def test_non_ibkr_or_duplicate_data_is_rejected(tmp_path):
    data = panel(310)
    symbols = cache(tmp_path, pd.concat([data, data.iloc[:1]]))
    with pytest.raises(ValueError, match='duplicate'):
        load_panel(tmp_path, symbols, now='2026-01-02T12:00:00Z')
    write_json(tmp_path / 'download_manifest.json', {'source': 'synthetic'})
    with pytest.raises(ValueError, match='actual IBKR'):
        load_panel(tmp_path, symbols)


def test_hkd_conversion_requires_broker_exchange_rate_and_never_invents_pnl():
    values = [NS(tag='NetLiquidation', currency='HKD', value='77800'),
              NS(tag='ExchangeRate', currency='USD', value='7.78'),
              NS(tag='SettledCash', currency='USD', value='1000')]
    result = usd_account_values(values, 'HKD', 77.8)
    assert result['net_liquidation_usd'] == pytest.approx(10000)
    assert result['daily_pnl_usd'] == pytest.approx(10)
    assert result['cash_usd'] is None
    assert usd_account_values(values[:1], 'HKD', 77.8)['daily_pnl_usd'] is None
    assert usd_account_values(values, 'HKD', float('nan'))['daily_pnl_usd'] is None


def test_new_ledger_prefix_and_dated_settled_cash():
    from zoneinfo import ZoneInfo
    today = dt.datetime.now(ZoneInfo('America/New_York')).strftime('%Y%m%d')
    values = [NS(tag='$LEDGER-CashBalance', currency='HKD', value='1000000'),
              NS(tag='$LEDGER-ExchangeRate', currency='USD', value='7.78'),
              NS(tag='SettledCashByDate', currency='USD', value=today + ':12000')]
    result = usd_account_values(values, 'HKD', None)
    assert result['cash_by_currency'] == {'HKD': 1000000.0}
    assert result['settled_cash_usd'] == 12000
    assert result['base_units_per_usd'] == 7.78
    values[-1].value = '20990101:12000'
    assert usd_account_values(values, 'HKD', None)['settled_cash_usd'] is None


def quote(now='2026-10-20T14:00:00Z', **kwargs):
    return {'bid': 99.99, 'ask': 100.01, 'conid': 42, 'min_tick': .01,
            'market_data_type': 1, 'bid_time': now, 'ask_time': now, **kwargs}


def test_whole_share_plan_limits_and_zero_weight_close():
    orders = order_plan({'A': .1, 'B': 0}, {'A': quote(), 'B': quote(conid=43)},
                       {'43': 5}, 12000, 1, 5)
    assert orders[0]['symbol'] == 'B' and orders[0]['side'] == 'SELL'
    buy = orders[1]
    assert buy['quantity'] == 11 and buy['notional_usd'] < 1200
    assert all(o['quantity'] == int(o['quantity']) for o in orders)
    with pytest.raises(ValueError, match='short/fractional'):
        order_plan({'A': 0}, {'A': quote()}, {'42': .5}, 12000, 1, 5)


def gate_inputs():
    summary = {'signal_session': '2026-10-19', 'provenance': {'data_current': True},
               'settings': {'capital_usd': 12000}}
    snap = {'positions': {}, 'daily_pnl_usd': 0, 'net_liquidation_usd': 12000,
            'settled_cash_usd': 12000, 'open_orders': []}
    quotes = {'A': quote()}
    plan = order_plan({'A': .1}, quotes, {}, 12000, 50, 5)
    return summary, snap, quotes, plan


def test_submission_gates_accept_only_fresh_realtime_and_valid_accounting():
    summary, snap, quotes, plan = gate_inputs()
    cfg = PaperTradingConfig()
    assert execution_gates(summary, snap, quotes, {'A': .1}, plan, cfg,
                           now='2026-10-20T14:00:00Z') == []
    quotes['A']['market_data_type'] = 3
    snap['daily_pnl_usd'] = None
    snap['settled_cash_usd'] = None
    gates = execution_gates(summary, snap, quotes, {'A': .1}, plan, cfg,
                            now='2026-10-20T14:00:00Z')
    assert 'A:real_time_market_data_required' in gates
    assert 'usd_daily_pnl_unavailable' in gates
    assert 'insufficient_or_unknown_settled_usd_cash' in gates


def test_gates_block_foreign_holdings_pending_orders_stale_quotes_and_reservations():
    summary, snap, quotes, plan = gate_inputs()
    snap['positions'] = {'42': 1}
    snap['open_orders'] = [{'remaining': 1}]
    quotes['A']['bid_time'] = '2026-10-20T13:55:00Z'
    gates = execution_gates(summary, snap, quotes, {'A': .1}, plan, PaperTradingConfig(),
        now='2026-10-20T14:00:00Z', usage={'n': 20, 'notional': 3600}, unresolved=[{}])
    for reason in ('A:holding_not_owned_by_strategy', 'A:stale_bid_time',
                   'account_has_pending_orders', 'unresolved_durable_order_intent',
                   'daily_order_count_limit', 'daily_notional_limit_including_reservations'):
        assert reason in gates


def snapshot(cash=10000, positions=None, time='2026-10-20T14:00:00+00:00'):
    return {'cash_usd': cash, 'positions': positions or {}, 'captured_at_utc': time}


def fill(commission=1):
    return {'exec_id': 'unique-1', 'ref': 'test', 'conid': 42, 'symbol': 'A', 'side': 'BUY',
            'quantity': 10, 'price': 100, 'commission': commission, 'commission_currency': 'USD',
            'currency': 'USD', 'timestamp': '2026-10-20T14:01:00+00:00'}


def test_reconciliation_is_durable_idempotent_and_handles_late_commission(tmp_path):
    path = tmp_path / 'ledger.sqlite3'
    key = account_key('DU123')
    ledger = Ledger(path)
    assert ledger.reconcile(key, snapshot(), [])['status'] == 'baseline_initialized'
    pending = ledger.reconcile(key, snapshot(9000, {'42': 10}), [fill(None)])
    assert 'commission_pending' in pending['reasons']
    assert ledger.reconcile(key, snapshot(8999, {'42': 10}), [fill()])['status'] == 'matched'
    ledger.close()
    ledger = Ledger(path)
    repeated = ledger.reconcile(key, snapshot(8999, {'42': 10}), [fill()])
    assert repeated['status'] == 'matched' and repeated['new_executions'] == 0
    assert len(ledger.executions(key)) == 1
    ledger.close()


def test_initial_existing_commission_is_not_charged_twice(tmp_path):
    ledger = Ledger(tmp_path / 'state.sqlite3')
    ledger.reconcile('paper', snapshot(8999, {'42': 10}), [fill()])
    assert ledger.reconcile('paper', snapshot(8999, {'42': 10}), [fill()])['status'] == 'matched'
    ledger.close()


def test_hkd_cash_can_reconcile_without_inventing_a_usd_balance(tmp_path):
    ledger = Ledger(tmp_path / 'state.sqlite3')
    snap = {**snapshot(None), 'cash_by_currency': {'HKD': 1000000}}
    ledger.reconcile('paper', snap, [])
    result = ledger.reconcile('paper', snap, [])
    assert result['status'] == 'matched' and result['cash_residual_by_currency'] == {'HKD': 0}
    assert result['cash_residual_usd'] is None
    ledger.close()


def test_late_commission_in_another_currency_is_accounted_for(tmp_path):
    ledger = Ledger(tmp_path / 'state.sqlite3')
    snap = {**snapshot(), 'cash_by_currency': {'USD': 10000, 'HKD': 100}}
    ledger.reconcile('paper', snap, [])
    actual = {**snapshot(9000, {'42': 10}), 'cash_by_currency': {'USD': 9000, 'HKD': 92.2}}
    execution = {**fill(7.8), 'commission_currency': 'HKD'}
    assert ledger.reconcile('paper', actual, [execution])['status'] == 'matched'
    assert ledger.reconcile('paper', actual, [execution])['status'] == 'matched'
    ledger.close()


def test_reconciliation_never_erases_mismatch_or_assumes_missing_cash_zero(tmp_path):
    ledger = Ledger(tmp_path / 'state.sqlite3')
    ledger.reconcile('paper', snapshot(), [])
    for _ in range(2):
        report = ledger.reconcile('paper', snapshot(10000, {'42': 1}), [])
        assert report['status'] == 'review_required' and not report['positions_match']
    assert ledger.reconcile('paper', snapshot(None), [])['cash_match'] is None
    ledger.close()


def test_restart_preserves_uncertain_send_and_cancelled_daily_budget(tmp_path):
    path = tmp_path / 'state.sqlite3'
    order = {'symbol': 'A', 'conid': 42, 'side': 'BUY', 'quantity': 10, 'limit_price': 100}
    ledger = Ledger(path)
    ref = ledger.reserve('paper', 'strategy', '2026-10-20', order, 'token')
    ledger.close()
    ledger = Ledger(path)
    assert ledger.unresolved('paper')[0]['status'] == 'send_uncertain'
    assert ledger.reserve('paper', 'strategy', '2026-10-20', order, 'token') is None
    ledger.update_intent(ref, 'Cancelled')
    assert ledger.unresolved('paper') == []
    assert ledger.daily_usage('paper', '2026-10-20') == {'n': 1, 'notional': 1000.0}
    ledger.close()


@pytest.mark.parametrize('argv', [['submit'], ['transport-check'],
    ['submit', '--submit-paper-orders', '--account', 'U123'],
    ['preview', '--submit-paper-orders'], ['preview', '--port', '4001']])
def test_cli_rejects_unsafe_order_modes_before_any_connection(argv):
    with pytest.raises(SystemExit):
        main(argv)


def test_cross_process_lock_prevents_overlapping_strategies(tmp_path):
    with exclusive_run(tmp_path / 'run.lock'):
        with pytest.raises(RuntimeError, match='already running'):
            with exclusive_run(tmp_path / 'run.lock'):
                pytest.fail('Concurrent workflow acquired the same lock')


def fake_broker(fail_send=False):
    ib = pytest.importorskip('ib_insync')
    from eventkit import Event
    broker = PaperBroker.__new__(PaperBroker)
    broker.account, broker.key = 'DU123', account_key('DU123')
    contract = ib.Stock('A', 'SMART', 'USD', conId=42)
    sent, cancelled = [], []
    def place(contract, order):
        if fail_send:
            raise ConnectionError('socket disconnected after intent commit')
        order.orderId, order.clientId, order.permId = 1, 29, 100
        trade = ib.Trade(contract, order, ib.OrderStatus(status='Submitted', filled=3, remaining=7))
        sent.append(trade)
        return trade
    def cancel(order):
        cancelled.append(order)
        sent[0].orderStatus.status = 'Cancelled'
    broker.gateway = NS(cfg=NS(readonly=False, client_id=29), is_connected=True,
        ib=NS(placeOrder=place, cancelOrder=cancel), sleep=lambda seconds: None)
    broker.md = NS(get_contract=lambda symbol: contract)
    broker.fills = lambda: []
    return broker, sent, cancelled


def test_partial_fill_is_cancelled_confirmed_and_not_automatically_resubmitted(tmp_path):
    broker, sent, cancelled = fake_broker()
    ledger = Ledger(tmp_path / 'state.sqlite3')
    plan = [{'symbol': 'A', 'conid': 42, 'side': 'BUY', 'quantity': 10, 'limit_price': 100}]
    result = broker.execute(plan, ledger, 'strategy', 'token', 0)
    assert result[0]['filled'] == 3 and result[0]['remaining'] == 7
    assert result[0]['status'] == 'Cancelled' and len(cancelled) == 1
    repeated = broker.execute(plan, ledger, 'strategy', 'token', 0)
    assert repeated[0]['status'] == 'duplicate_intent_blocked' and len(sent) == 1
    ledger.close()


def test_socket_failure_preserves_uncertain_intent_for_manual_reconciliation(tmp_path):
    broker, sent, cancelled = fake_broker(fail_send=True)
    ledger = Ledger(tmp_path / 'state.sqlite3')
    plan = [{'symbol': 'A', 'conid': 42, 'side': 'BUY', 'quantity': 10, 'limit_price': 100}]
    with pytest.raises(ConnectionError, match='disconnected'):
        broker.execute(plan, ledger, 'strategy', 'token', 0)
    assert ledger.unresolved(broker.key)[0]['status'] == 'send_uncertain'
    assert not sent and not cancelled
    ledger.close()


def test_pending_cancel_is_awaited_until_a_real_terminal_confirmation(tmp_path):
    broker, sent, _ = fake_broker()
    calls = []
    def cancel(order):
        sent[0].orderStatus.status = 'PendingCancel'
    def sleep(seconds):
        calls.append(seconds)
        if sent and sent[0].orderStatus.status == 'PendingCancel':
            sent[0].orderStatus.status = 'Cancelled'
    broker.gateway.ib.cancelOrder = cancel
    broker.gateway.sleep = sleep
    ledger = Ledger(tmp_path / 'state.sqlite3')
    plan = [{'symbol': 'A', 'conid': 42, 'side': 'BUY', 'quantity': 10, 'limit_price': 100}]
    result = broker.execute(plan, ledger, 'strategy', 'token', 0)
    assert result[0]['status'] == 'Cancelled'
    assert .5 in calls
    ledger.close()


def test_completed_order_recovers_by_perm_id_without_overwriting_order_id(tmp_path):
    broker, _, _ = fake_broker()
    ledger = Ledger(tmp_path / 'state.sqlite3')
    order = {'symbol': 'A', 'conid': 42, 'side': 'BUY', 'quantity': 1, 'limit_price': 100}
    ref = ledger.reserve(broker.key, 'strategy', '2026-10-20', order, 'token')
    ledger.update_intent(ref, 'PendingCancel', order_id=10, perm_id=123)
    completed = NS(order=NS(orderRef=ref, account=broker.account, clientId=0, orderId=0,
                             permId=123, filledQuantity=0), orderStatus=NS(status='Cancelled'))
    broker.gateway.ib.trades = lambda: []
    broker.gateway.ib.reqCompletedOrders = lambda apiOnly: [completed]
    broker.sync_intents(ledger)
    assert ledger.unresolved(broker.key) == []
    recovered = ledger.intent(ref)
    assert recovered['order_id'] == 10 and recovered['filled_quantity'] == 0
    ledger.close()
