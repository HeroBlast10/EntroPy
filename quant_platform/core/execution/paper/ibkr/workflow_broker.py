"""Paper-only broker boundary. No live accounts, implicit FX loans or retries."""
from __future__ import annotations

import datetime as dt
import math
import time
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR

import pandas as pd

from .config import IBConfig, PaperTradingConfig
from .gateway import IBGateway
from .market_data import MarketDataManager, _safe_float
from .workflow_research import last_complete_session, regular_session
from .workflow_state import Ledger, account_key


def terminal(trade):
    # ib_insync.isActive() excludes PendingCancel. A cancellation request is
    # not a broker-confirmed terminal state and must still be awaited.
    return trade.orderStatus.status in {'Filled', 'Cancelled', 'ApiCancelled', 'Inactive'}


def usd_account_values(values: list, base_currency: str, daily_pnl) -> dict:
    """IB ExchangeRate is units of account base currency per ledger currency."""
    lookup = {(v.tag.removeprefix('$LEDGER-'), v.currency): _safe_float(v.value) for v in values}
    fx = 1.0 if base_currency == 'USD' else lookup.get(('ExchangeRate', 'USD'))
    if fx is not None and fx <= 0:
        fx = None
    net = lookup.get(('NetLiquidation', base_currency))
    if net is None:
        net = lookup.get(('NetLiquidation', 'BASE'))
    settled = lookup.get(('SettledCash', 'USD'))
    if settled is None:
        # New Gateway builds can emit a dated value instead of SettledCash.
        # Accept the observed single-date format only for today's NY date;
        # unknown/future schedules never become spendable cash implicitly.
        import re
        from zoneinfo import ZoneInfo
        today = dt.datetime.now(ZoneInfo('America/New_York')).strftime('%Y%m%d')
        dated = next((v.value for v in values if v.tag == 'SettledCashByDate' and v.currency == 'USD'), '')
        match = re.fullmatch(r'(\d{8}):([+-]?\d+(?:\.\d+)?)', dated)
        if match and match.group(1) == today:
            settled = _safe_float(match.group(2))
    cash = lookup.get(('CashBalance', 'USD'))
    if cash is None:
        cash = lookup.get(('TotalCashBalance', 'USD'))
    currencies = {currency for tag, currency in lookup if currency not in {'', 'BASE'}}
    cash_by_currency = {}
    for currency in currencies:
        amount = lookup.get(('CashBalance', currency))
        if amount is None:
            amount = lookup.get(('TotalCashBalance', currency))
        if amount is not None:
            cash_by_currency[currency] = amount
    return {'base_currency': base_currency, 'base_units_per_usd': fx, 'cash_by_currency': cash_by_currency,
            'cash_usd': cash, 'settled_cash_usd': settled,
            'net_liquidation_usd': net / fx if net is not None and fx else None,
            'daily_pnl_usd': _safe_float(daily_pnl) / fx if _safe_float(daily_pnl) is not None and fx else None,
            'fx_source': 'IBKR per-currency account ledger ExchangeRate' if fx else 'unavailable'}


def order_plan(weights: dict, quotes: dict, positions: dict, capital: float,
               min_trade: float, offset_bps: float) -> list[dict]:
    if (not weights or any(not math.isfinite(w) or w < 0 for w in weights.values())
            or sum(weights.values()) > 1 + 1e-8 or not math.isfinite(capital) or capital <= 0):
        raise ValueError('Invalid unlevered target weights/capital')
    orders = []
    for symbol, weight in sorted(weights.items()):
        quote = quotes.get(symbol, {})
        bid, ask = quote.get('bid'), quote.get('ask')
        if bid is None or ask is None or not 0 < bid <= ask:
            continue
        mid = (bid + ask) / 2
        current = float(positions.get(str(quote['conid']), 0))
        if current < 0 or not current.is_integer():
            raise ValueError('Pilot cannot manage existing short/fractional holdings')
        # Size conservatively at a buy limit, so a spread/offset cannot push a
        # nominal 10% target beyond its USD position budget.
        target = math.floor(weight * capital / (ask * (1 + offset_bps / 10000)))
        delta = target - int(current)
        if not delta or abs(delta) * mid < min_trade:
            continue
        side = 'BUY' if delta > 0 else 'SELL'
        raw_limit = (ask * (1 + offset_bps / 10000) if side == 'BUY'
                     else bid * (1 - offset_bps / 10000))
        tick = Decimal(str(quote.get('min_tick', 0.01)))
        if tick <= 0:
            raise ValueError('Invalid contract tick size')
        limit = float((Decimal(str(raw_limit)) / tick).to_integral_value(
            rounding=ROUND_CEILING if side == 'BUY' else ROUND_FLOOR) * tick)
        orders.append({'symbol': symbol, 'conid': quote['conid'], 'side': side,
                       'quantity': abs(delta), 'current_shares': int(current), 'target_shares': target,
                       'limit_price': limit, 'notional_usd': abs(delta) * limit,
                       'market_data_type': quote.get('market_data_type')})
    return sorted(orders, key=lambda o: (o['side'] != 'SELL', o['symbol']))


def execution_gates(summary: dict, snapshot: dict, quotes: dict, weights: dict,
                    plan: list, cfg: PaperTradingConfig, *, now=None, owned=None,
                    reconciliation=None, unresolved=None, usage=None) -> list[str]:
    now = pd.Timestamp(now if now is not None else dt.datetime.now(dt.timezone.utc))
    reasons = []
    if not regular_session(now):
        reasons.append('outside_regular_trading_session')
    if pd.Timestamp(summary['signal_session']) != last_complete_session(now):
        reasons.append('stale_or_future_signal')
    # A just-completed close cannot be executed earlier in the same session.
    if pd.Timestamp(summary['signal_session']).date() >= now.tz_convert('America/New_York').date():
        reasons.append('signal_requires_next_session_execution')
    if not summary['provenance']['data_current']:
        reasons.append('research_data_stale')
    if cfg.risk.kill_switch:
        reasons.append('kill_switch_active')
    if snapshot.get('daily_pnl_usd') is None:
        reasons.append('usd_daily_pnl_unavailable')
    elif snapshot['daily_pnl_usd'] <= -cfg.risk.max_daily_loss:
        reasons.append('daily_loss_limit')
    if snapshot.get('net_liquidation_usd') is None:
        reasons.append('usd_net_liquidation_or_fx_unavailable')
    elif snapshot['net_liquidation_usd'] < summary['settings']['capital_usd']:
        reasons.append('strategy_capital_exceeds_account_equity')
    if unresolved:
        reasons.append('unresolved_durable_order_intent')
    if snapshot.get('open_orders'):
        reasons.append('account_has_pending_orders')
    if reconciliation and reconciliation['status'] == 'review_required':
        reasons.append('account_reconciliation_requires_review')
    planned_buy = 0.0
    for symbol in weights:
        quote = quotes.get(symbol, {})
        bid, ask = quote.get('bid'), quote.get('ask')
        if bid is None or ask is None or not 0 < bid <= ask:
            reasons.append(f'{symbol}:missing_or_crossed_quote')
            continue
        if quote.get('market_data_type') != 1:
            reasons.append(f'{symbol}:real_time_market_data_required')
        for field in ('bid_time', 'ask_time'):
            timestamp = pd.Timestamp(quote[field]) if quote.get(field) else None
            age = (now - timestamp).total_seconds() if timestamp is not None and timestamp.tzinfo else float('inf')
            if not 0 <= age <= 120:
                reasons.append(f'{symbol}:stale_{field}')
        if (ask - bid) / ((ask + bid) / 2) > 0.005:
            reasons.append(f'{symbol}:spread_exceeds_50bps')
        conid = str(quote['conid'])
        actual = snapshot['positions'].get(conid, 0)
        expected_owned = (owned or {}).get(quote['conid'], 0)
        if abs(actual - expected_owned) > 1e-8:
            reasons.append(f'{symbol}:holding_not_owned_by_strategy')
    for order in plan:
        if order['notional_usd'] > cfg.risk.max_order_notional or order['quantity'] > cfg.risk.max_order_shares:
            reasons.append(f'{order["symbol"]}:order_limit')
        quote = quotes[order['symbol']]
        projected = order['target_shares'] * (quote['ask'] if order['side'] == 'BUY' else quote['bid'])
        if projected > cfg.risk.max_position_notional or order['target_shares'] > cfg.risk.max_position_shares:
            reasons.append(f'{order["symbol"]}:position_limit')
        if order['side'] == 'BUY':
            planned_buy += order['notional_usd'] * 1.01
    existing = sum(float(v) != 0 for v in snapshot['positions'].values())
    new = sum(o['side'] == 'BUY' and o['current_shares'] == 0 for o in plan)
    if existing + new > cfg.risk.max_positions:
        reasons.append('position_count_limit')
    cash = snapshot.get('settled_cash_usd')
    if planned_buy and (cash is None or cash < planned_buy):
        reasons.append('insufficient_or_unknown_settled_usd_cash')
    usage = usage or {'n': 0, 'notional': 0.0}
    if usage['n'] + len(plan) > cfg.risk.max_daily_trades:
        reasons.append('daily_order_count_limit')
    if usage['notional'] + sum(o['notional_usd'] for o in plan) > cfg.risk.max_daily_notional:
        reasons.append('daily_notional_limit_including_reservations')
    return sorted(set(reasons))


class PaperBroker:
    def __init__(self, config: IBConfig):
        self.gateway = IBGateway(config)
        self.md = MarketDataManager(self.gateway)
        self.price_times = {}
        self.details = {}
        self.pnl = None

    def __enter__(self):
        self.gateway.connect()
        try:
            self.account = self.gateway.assert_paper_account()
            self.key = account_key(self.account)
            self.pnl = self.gateway.ib.reqPnL(self.account, '')
            self.gateway.sleep(2)
        except BaseException:
            self.gateway.disconnect()
            raise
        return self

    def __exit__(self, *exc):
        try:
            if self.gateway.is_connected:
                self.md.unsubscribe_all()
                if self.pnl is not None:
                    self.gateway.ib.cancelPnL(self.account, '')
        finally:
            self.gateway.disconnect()

    def quotes(self, symbols):
        self.md.subscribe(symbols)
        for symbol in symbols:
            ticker = self.md._tickers.get(symbol)
            if ticker is None:
                continue
            def capture(ticker, symbol=symbol):
                for tick in ticker.ticks:
                    side = {1: 'bid_time', 2: 'ask_time', 66: 'bid_time', 67: 'ask_time'}.get(tick.tickType)
                    if side and tick.price > 0:
                        self.price_times.setdefault(symbol, {})[side] = tick.time.isoformat()
            ticker.updateEvent += capture
            contract = self.md.get_contract(symbol)
            details = self.gateway.ib.reqContractDetails(contract)
            if len(details) != 1 or contract.secType != 'STK' or contract.currency != 'USD':
                raise ValueError(f'Ambiguous or non-USD stock/ETF contract: {symbol}')
            self.details[symbol] = details[0]
        self.gateway.sleep(8)
        result = {}
        for symbol in symbols:
            quote = self.md.snapshot(symbol)
            if not quote:
                continue
            contract = self.md.get_contract(symbol)
            quote.update(self.price_times.get(symbol, {}))
            quote.update({'conid': contract.conId, 'min_tick': self.details[symbol].minTick})
            result[symbol] = quote
        return result

    def snapshot(self):
        ib = self.gateway.ib
        # Account summary includes $LEDGER:ALL, unlike the narrow account view
        # returned by some Gateway configurations. Missing USD stays unknown.
        values = list(ib.accountSummary(self.account)) + list(ib.accountValues(self.account))
        account = usd_account_values(values, self.gateway.base_currency(), self.pnl.dailyPnL)
        account.update({'captured_at_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
                        'account_key': self.key,
                        'positions': {str(p.contract.conId): float(p.position)
                                      for p in ib.reqPositions() if p.account == self.account},
                        'open_orders': [{'ref': t.order.orderRef, 'symbol': t.contract.symbol,
                                         'client_id': t.order.clientId, 'order_id': t.order.orderId,
                                         'remaining': float(t.orderStatus.remaining)}
                                        for t in ib.reqAllOpenOrders() if t.order.account == self.account
                                        and not terminal(t)]})
        return account

    def fills(self):
        from ib_insync import ExecutionFilter
        result = []
        for fill in self.gateway.ib.reqExecutions(ExecutionFilter(acctCode=self.account)):
            execution, commission = fill.execution, fill.commissionReport
            if execution.acctNumber != self.account:
                continue
            known = bool(commission and commission.execId == execution.execId and commission.currency)
            result.append({'exec_id': execution.execId, 'ref': execution.orderRef,
                           'conid': fill.contract.conId, 'symbol': fill.contract.symbol,
                           'side': {'BOT': 'BUY', 'SLD': 'SELL'}.get(execution.side, execution.side),
                           'quantity': float(execution.shares), 'price': float(execution.price),
                           'currency': fill.contract.currency,
                           'timestamp': execution.time.isoformat(),
                           'commission': _safe_float(commission.commission) if known else None,
                           'commission_currency': commission.currency if known else None})
        return result

    def sync_intents(self, ledger: Ledger):
        # Own-client orders only. Reading all orders does not bind other clients.
        trades = list(self.gateway.ib.trades())
        if ledger.unresolved(self.key):
            trades.extend(self.gateway.ib.reqCompletedOrders(apiOnly=True))
        for trade in trades:
            intent = ledger.intent(trade.order.orderRef)
            # Completed-order responses omit clientId/orderId in some builds.
            # The durable ref AND broker permanent ID identify that receipt;
            # this is for reading state, never for cancelling another client.
            same_perm = intent and intent['perm_id'] and intent['perm_id'] == trade.order.permId
            if (intent and trade.order.account == self.account
                    and (trade.order.clientId == self.gateway.cfg.client_id or same_perm)):
                filled = (_safe_float(getattr(trade.order, 'filledQuantity', None)) if trade.order.clientId == 0
                          else float(trade.orderStatus.filled))
                ledger.update_intent(trade.order.orderRef, trade.orderStatus.status,
                                     trade.order.orderId or None, trade.order.permId or None, filled)
        # Executions can prove a complete fill even after an offline restart.
        for intent in ledger.unresolved(self.key):
            done = sum(f['quantity'] for f in ledger.executions(self.key) if f['ref'] == intent['ref'])
            if done >= intent['quantity']:
                ledger.update_intent(intent['ref'], 'Filled', filled_quantity=done)

    def execute(self, plan, ledger: Ledger, strategy: str, token: str, seconds: int):
        from ib_insync import LimitOrder
        if self.gateway.cfg.readonly:
            raise ValueError('A read-only client cannot submit orders')
        # Actual NY timezone, including DST, determines the durable risk day.
        from zoneinfo import ZoneInfo
        session = dt.datetime.now(ZoneInfo('America/New_York')).date().isoformat()
        submitted, result = [], []
        try:
            for order in plan:
                ref = ledger.reserve(self.key, strategy, session, order, token)
                if ref is None:
                    result.append({'symbol': order['symbol'], 'status': 'duplicate_intent_blocked'})
                    continue
                # The uncertain intent survives any exception between reserve
                # and socket acknowledgement. Never reset it to "not sent".
                contract = self.md.get_contract(order['symbol'])
                if contract is None or contract.conId != order['conid']:
                    raise ValueError('Contract identity changed before send')
                ib_order = LimitOrder(order['side'], order['quantity'], order['limit_price'],
                                      account=self.account, orderRef=ref, tif='DAY', outsideRth=False)
                trade = self.gateway.ib.placeOrder(contract, ib_order)
                submitted.append((ref, trade))
                ledger.update_intent(ref, 'PendingSubmit', trade.order.orderId)
                self.gateway.sleep(1)
            deadline = time.monotonic() + seconds
            while any(not terminal(t) for _, t in submitted) and time.monotonic() < deadline:
                self.gateway.sleep(0.5)
        finally:
            # Always reconcile confirmed cancellation; disconnecting alone does
            # not cancel an exchange order. Never issue reqGlobalCancel.
            for ref, trade in submitted:
                if not terminal(trade) and self.gateway.is_connected:
                    self.gateway.ib.cancelOrder(trade.order)
            deadline = time.monotonic() + 30
            while (self.gateway.is_connected and any(not terminal(t) for _, t in submitted)
                   and time.monotonic() < deadline):
                self.gateway.sleep(0.5)
            for ref, trade in submitted:
                status = trade.orderStatus.status or 'send_uncertain'
                ledger.update_intent(ref, status, trade.order.orderId, trade.order.permId or None,
                                     float(trade.orderStatus.filled))
                result.append({'ref': ref, 'symbol': trade.contract.symbol, 'status': status,
                               'order_id': trade.order.orderId, 'perm_id': trade.order.permId,
                               'filled': float(trade.orderStatus.filled),
                               'remaining': float(trade.orderStatus.remaining),
                               'events': [{'status': event.status, 'message': event.message.replace(self.account, 'DU***'),
                                           'error_code': event.errorCode} for event in trade.log]})
            ledger.ingest(self.key, self.fills())
        return result
