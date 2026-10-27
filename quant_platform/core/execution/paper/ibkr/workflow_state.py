"""Durable paper order intents, executions and account reconciliation.

Intent is committed BEFORE touching the socket. An uncertain send is never
automatically retried. Broker execution IDs make repeated reconciliation safe.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import sqlite3
from pathlib import Path


def account_key(account: str) -> str:
    if not account.startswith('DU'):
        raise ValueError('Only a DU paper account may use this ledger')
    return hashlib.sha256(account.encode()).hexdigest()[:20]


class Ledger:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.row_factory = sqlite3.Row
        self.db.executescript('''
            CREATE TABLE IF NOT EXISTS intents (
                ref TEXT PRIMARY KEY, account TEXT NOT NULL, strategy TEXT NOT NULL,
                session TEXT NOT NULL, symbol TEXT NOT NULL, conid INTEGER NOT NULL,
                side TEXT NOT NULL, quantity INTEGER NOT NULL, limit_price REAL NOT NULL,
                status TEXT NOT NULL, order_id INTEGER, perm_id INTEGER,
                created_at TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS executions (
                exec_id TEXT PRIMARY KEY, account TEXT NOT NULL, ref TEXT NOT NULL,
                conid INTEGER NOT NULL, symbol TEXT NOT NULL, side TEXT NOT NULL,
                quantity REAL NOT NULL, price REAL NOT NULL, commission REAL,
                commission_currency TEXT, currency TEXT NOT NULL, timestamp TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS checkpoints (
                account TEXT PRIMARY KEY, snapshot TEXT NOT NULL, seen_ids TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS baseline (
                account TEXT PRIMARY KEY, snapshot TEXT NOT NULL, seen_ids TEXT NOT NULL);
        ''')
        if 'filled_quantity' not in {r[1] for r in self.db.execute('PRAGMA table_info(intents)')}:
            self.db.execute('ALTER TABLE intents ADD COLUMN filled_quantity REAL')
        self.db.commit()

    def close(self):
        self.db.close()

    def reserve(self, account: str, strategy: str, session: str, order: dict, token: str) -> str | None:
        identity = f'{account}|{strategy}|{session}|{token}|{order["symbol"]}|{order["side"]}'
        ref = 'EP-' + hashlib.sha256(identity.encode()).hexdigest()[:24]
        with self.db:
            count = self.db.execute('''INSERT OR IGNORE INTO intents
                (ref,account,strategy,session,symbol,conid,side,quantity,limit_price,status,created_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?)''',
                (ref, account, strategy, session, order['symbol'], order['conid'], order['side'],
                 order['quantity'], order['limit_price'], 'send_uncertain',
                 dt.datetime.now(dt.timezone.utc).isoformat())).rowcount
        return ref if count else None

    def update_intent(self, ref, status, order_id=None, perm_id=None, filled_quantity=None):
        with self.db:
            self.db.execute('UPDATE intents SET status=?, order_id=COALESCE(?,order_id), '
                            'perm_id=COALESCE(?,perm_id), filled_quantity=COALESCE(?,filled_quantity) WHERE ref=?',
                            (status, order_id, perm_id, filled_quantity, ref))

    def intent(self, ref):
        row = self.db.execute('SELECT * FROM intents WHERE ref=?', (ref,)).fetchone()
        return dict(row) if row else None

    def unresolved(self, account):
        return [dict(r) for r in self.db.execute(
            "SELECT * FROM intents WHERE account=? AND status NOT IN ('Filled','Cancelled','ApiCancelled','Inactive')",
            (account,))]

    def executions(self, account):
        return [dict(r) for r in self.db.execute('SELECT * FROM executions WHERE account=?', (account,))]

    def ingest(self, account, executions):
        with self.db:
            for fill in executions:
                if fill['side'] not in {'BUY', 'SELL'} or fill['quantity'] <= 0 or fill['price'] <= 0:
                    raise ValueError('Invalid broker execution')
                self.db.execute('''INSERT INTO executions VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
                    ON CONFLICT(exec_id) DO UPDATE SET
                    commission=COALESCE(excluded.commission, executions.commission),
                    commission_currency=COALESCE(excluded.commission_currency, executions.commission_currency)''',
                    (fill['exec_id'], account, fill['ref'], fill['conid'], fill['symbol'], fill['side'],
                     fill['quantity'], fill['price'], fill.get('commission'),
                     fill.get('commission_currency'), fill['currency'], fill['timestamp']))

    def owned_positions(self, account, strategy):
        result = {}
        rows = self.db.execute('''SELECT e.* FROM executions e JOIN intents i ON e.ref=i.ref
            WHERE e.account=? AND i.strategy=?''', (account, strategy))
        for row in rows:
            result[row['conid']] = result.get(row['conid'], 0) + row['quantity'] * (1 if row['side'] == 'BUY' else -1)
        return result

    def daily_usage(self, account, session):
        # Count the full original order budget, including cancelled orders. This
        # deliberately over-reserves; restarts cannot recycle a daily allowance.
        row = self.db.execute('''SELECT COUNT(*) AS n, COALESCE(SUM(quantity*limit_price),0) AS notional
            FROM intents WHERE account=? AND session=?''', (account, session)).fetchone()
        return dict(row)

    def reconcile(self, account: str, snapshot: dict, executions: list[dict]) -> dict:
        self.ingest(account, executions)
        current_ids = {f['exec_id'] for f in self.executions(account)}
        row = self.db.execute('SELECT * FROM checkpoints WHERE account=?', (account,)).fetchone()
        if row is None:
            snapshot = dict(snapshot)
            snapshot['known_commissions'] = {f['exec_id']: f['commission'] for f in self.executions(account)
                                               if f['commission'] is not None}
            with self.db:
                payload = (account, json.dumps(snapshot), json.dumps(sorted(current_ids)))
                self.db.execute('INSERT INTO baseline VALUES (?,?,?)', payload)
                self.db.execute('INSERT INTO checkpoints VALUES (?,?,?)', payload)
            return {'status': 'baseline_initialized', 'positions_match': None, 'cash_match': None,
                    'new_executions': 0, 'coverage': 'First broker observation, not an independent historical audit.'}
        previous = json.loads(row['snapshot'])
        seen = set(json.loads(row['seen_ids']))
        expected = {str(k): float(v) for k, v in previous['positions'].items()}
        def cash_map(observation):
            if observation.get('cash_by_currency'):
                return dict(observation['cash_by_currency'])
            return {'USD': observation['cash_usd']} if observation.get('cash_usd') is not None else {}
        cash = cash_map(previous)
        actual_cash = cash_map(snapshot)
        missing_cash_currencies = set()
        missing_commission = False
        # Late commissions on fills already seen must still affect the delta.
        previous_commissions = previous.get('known_commissions', {})
        for fill in self.executions(account):
            sign = 1 if fill['side'] == 'BUY' else -1
            if fill['exec_id'] not in seen:
                key = str(fill['conid'])
                expected[key] = expected.get(key, 0.0) + sign * fill['quantity']
                currency = fill['currency']
                if currency in cash:
                    cash[currency] -= sign * fill['quantity'] * fill['price']
                else:
                    missing_cash_currencies.add(currency)
            if fill['commission'] is None or not fill['commission_currency']:
                if fill['exec_id'] not in seen:
                    missing_commission = True
            else:
                difference = fill['commission'] - previous_commissions.get(fill['exec_id'], 0.0)
                currency = fill['commission_currency']
                if difference and currency in cash:
                    cash[currency] -= difference
                elif difference:
                    missing_cash_currencies.add(currency)
        actual = snapshot['positions']
        differences = {k: float(actual.get(k, 0)) - expected.get(k, 0)
                       for k in set(expected) | set(actual)
                       if abs(float(actual.get(k, 0)) - expected.get(k, 0)) > 1e-8}
        residuals = {currency: actual_cash[currency] - cash[currency]
                     for currency in set(actual_cash) & set(cash)}
        missing_cash_currencies.update(set(actual_cash) ^ set(cash))
        residual = residuals.get('USD')
        # An older checkpoint may predate support for a new broker cash prefix.
        # Establish that field's first observation explicitly, without claiming
        # its earlier cash history was audited or erasing any position mismatch.
        if not cash and actual_cash and not differences and current_ids == seen:
            with self.db:
                self.db.execute('UPDATE checkpoints SET snapshot=? WHERE account=?',
                                (json.dumps(snapshot), account))
            return {'status': 'cash_baseline_initialized', 'positions_match': True,
                    'cash_match': None, 'new_executions': 0,
                    'coverage': 'First observation of per-currency cash; earlier cash history is unverified.'}
        prior = dt.datetime.fromisoformat(previous['captured_at_utc'])
        current = dt.datetime.fromisoformat(snapshot['captured_at_utc'])
        # A default API execution query may cover only the current day. Flag
        # any regular trading interval between the checkpoint and UTC midnight
        # which that query could have missed. A post-close daily checkpoint
        # remains valid across nights, weekends and holidays.
        import exchange_calendars as xc
        import pandas as pd
        cal = xc.get_calendar('XNYS')
        midnight = pd.Timestamp(current).tz_convert('UTC').normalize()
        sessions = cal.sessions_in_range(prior.date() - dt.timedelta(days=1), current.date())
        coverage_gap = any(cal.session_close(s) > pd.Timestamp(prior)
                           and cal.session_open(s) < midnight for s in sessions)
        reasons = []
        if differences:
            reasons.append('position_mismatch')
        if not residuals or missing_cash_currencies:
            reasons.append('cash_currency_unavailable_or_external_cash_flow')
        if any(abs(v) > 0.05 for v in residuals.values()):
            reasons.append('cash_mismatch_or_unrecorded_cash_flow')
        if missing_commission:
            reasons.append('commission_pending')
        if coverage_gap:
            reasons.append('execution_history_coverage_requires_review')
        # Do not roll a failing checkpoint forward and silently erase a mismatch.
        if not reasons:
            snapshot = dict(snapshot)
            snapshot['known_commissions'] = {f['exec_id']: f['commission'] for f in self.executions(account)
                                               if f['commission'] is not None}
            with self.db:
                self.db.execute('UPDATE checkpoints SET snapshot=?, seen_ids=? WHERE account=?',
                                (json.dumps(snapshot), json.dumps(sorted(current_ids)), account))
        return {'status': 'matched' if not reasons else 'review_required',
                'positions_match': not differences, 'position_differences': differences,
                'cash_match': (all(abs(v) <= 0.05 for v in residuals.values())
                               if residuals and not missing_cash_currencies else None),
                'cash_residual_usd': residual, 'cash_residual_by_currency': residuals,
                'new_executions': len(current_ids - seen), 'reasons': reasons,
                'coverage': 'API executions since checkpoint; external cash flows and missing days require review.'}
