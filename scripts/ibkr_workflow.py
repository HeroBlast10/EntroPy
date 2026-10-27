#!/usr/bin/env python
"""Daily IBKR workflow: research, preview, submit, reconcile and transport-check.

Default commands are read-only. Actual paper orders require the explicit
--submit-paper-orders flag and --account DU... on this invocation.
"""
from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import html
import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
import yaml
from loguru import logger

from quant_platform.core.execution.paper.ibkr.config import load_paper_config
from quant_platform.core.execution.paper.ibkr.workflow_research import (
    digest, last_complete_session, load_settings, regular_session, run_research, write_json,
)
from quant_platform.core.execution.paper.ibkr.workflow_state import Ledger
from quant_platform.core.execution.paper.ibkr.workflow_broker import PaperBroker, execution_gates, order_plan


@contextlib.contextmanager
def exclusive_run(path):
    """Cross-process lock shared by both strategies and reconciliation."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+b') as handle:
        handle.seek(0, 2)
        if handle.tell() == 0:
            handle.write(b'0')
            handle.flush()
        handle.seek(0)
        try:
            if sys.platform == 'win32':
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError('Another IBKR workflow is already running; wait for it to finish') from exc
        try:
            yield
        finally:
            handle.seek(0)
            if sys.platform == 'win32':
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def render_report(output, summary=None, broker_report=None):
    summary = summary or {}
    broker_report = broker_report or {}
    plot = ''
    if (output / 'backtest_daily.csv').exists():
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 4))
        for name, label in [('backtest_daily.csv', 'Strategy'), ('baseline_daily.csv', 'Equal-weight baseline'),
                            ('market_benchmark_daily.csv', 'SPY matched-exposure benchmark')]:
            if not (output / name).exists():
                continue
            frame = pd.read_csv(output / name, parse_dates=['date'])
            ax.plot(frame.date, (1 + frame['return']).cumprod(), label=label)
        if summary.get('holdout_start'):
            ax.axvline(pd.Timestamp(summary['holdout_start']), color='gray', linestyle='--', label='Holdout starts')
        ax.set(title='IBKR fixed-universe pilot - net modeled costs', ylabel='Growth of initial capital')
        ax.legend()
        ax.grid(alpha=.2)
        fig.tight_layout()
        fig.savefig(output / 'backtest_equity.png', dpi=140)
        plt.close(fig)
        plot = '<img src="backtest_equity.png" alt="Backtest growth curves">'
    rows = []
    for key, label in [('full_sample', '全部回测'), ('development', '开发区间'),
                       ('holdout', '最近252交易日留出区间'),
                       ('holdout_equal_weight_baseline', '相近敞口等权基准'),
                       ('holdout_market_benchmark', '相近敞口SPY基准')]:
        if key in summary:
            m = summary[key]
            rows.append({'区间': label, '交易日': m['sessions'], '总收益': f"{m['total_return']:.2%}",
                         '年化波动': f"{m['annual_volatility']:.2%}",
                         '最大回撤': f"{m['max_drawdown']:.2%}", '模型成本USD': f"{m['cost_usd']:.2f}"})
    metrics_html = pd.DataFrame(rows).to_html(index=False) if rows else ''
    weights = summary.get('target_weights', {})
    weights_html = pd.DataFrame([{'ticker': k, '目标权重': f'{v:.2%}'} for k, v in weights.items()]).to_html(index=False)
    limitations = ''.join('<li>' + html.escape(s) + '</li>' for s in summary.get('limitations', []))
    receipt = output / 'transport_receipt.json'
    receipt_html = ('<h2>券商接口测试回执</h2><pre>' + html.escape(receipt.read_text(encoding='utf-8')) + '</pre>'
                    if receipt.exists() else '')
    content = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<title>IBKR {html.escape(summary.get('strategy_id', output.name))} 运行报告</title>
<style>body{{font:15px/1.65 system-ui;max-width:1150px;margin:32px auto;padding:0 20px;color:#172333}}
table{{border-collapse:collapse;width:100%;margin:15px 0}}td,th{{border:1px solid #d6dee7;padding:8px;text-align:left}}
pre{{white-space:pre-wrap;overflow-wrap:anywhere;background:#f3f6fa;padding:16px}}img{{max-width:100%}}</style>
<h1>IBKR {html.escape(summary.get('strategy_id', output.name))} 运行报告</h1>
<p>信号交易日：{html.escape(summary.get('signal_session', '尚未产生'))}。
数据来源：IBKR TWS API。模拟运行与历史研究均不构成实盘验证。</p>
<h2>研究与成本回测</h2>{metrics_html}{plot}<h2>最新目标权重</h2>{weights_html}
{receipt_html}<h2>券商预检、委托与对账</h2><pre>{html.escape(json.dumps(broker_report, ensure_ascii=False, indent=2, default=str))}</pre>
<h2>研究边界</h2><ul>{limitations}</ul></html>'''
    (output / 'workflow_report.html').write_text(content, encoding='utf-8')


def transport_check(broker, ledger, cfg, directory, symbol, settings):
    """One paper share parked below reference, outside RTH, then cancelled.

This tests the order transport only. It never counts as a strategy fill test.
Delayed data is expressly permitted here because outsideRth=False and the
probe must end at least five minutes before the next exchange opening.
"""
    import exchange_calendars as xc
    from ib_insync import LimitOrder
    now = pd.Timestamp.now(tz='UTC')
    cal = xc.get_calendar('XNYS')
    day = now.tz_convert('America/New_York').date()
    prior = [dict(r) for r in ledger.db.execute('''SELECT * FROM intents
        WHERE account=? AND strategy=? AND session=?''',
        (broker.key, settings.strategy_id + '_transport', day.isoformat()))]
    if prior:
        passed = all(r['status'] in {'Cancelled', 'ApiCancelled'} and r['perm_id'] and r['filled_quantity'] == 0
                     for r in prior)
        return {'purpose': 'transport_only_not_strategy_execution',
                'status': 'accepted_and_cancelled' if passed else 'review_required',
                'existing_probe_reused': True, 'orders': prior}
    sessions = cal.sessions_in_range(day, day + dt.timedelta(days=10))
    future_opens = [cal.session_open(s) for s in sessions if cal.session_open(s) > now]
    if regular_session(now) or not future_opens or (future_opens[0] - now).total_seconds() < 300:
        raise ValueError('Transport check requires a closed regular session and >5 minutes until next open')
    raw = pd.read_parquet(directory / f'{symbol}_TRADES.parquet')
    raw['date'] = pd.to_datetime(raw.date)
    raw = raw[raw.date <= last_complete_session(now)].sort_values('date')
    if raw.empty or raw.date.iloc[-1] != last_complete_session(now):
        raise ValueError('Transport probe needs a current completed-session reference')
    quotes = broker.quotes([symbol])
    quote = quotes[symbol]
    reference = float(raw.close.iloc[-1])
    if quote.get('bid'):
        reference = min(reference, quote['bid'])
    # US ETF prices >$1 have $0.01 minimum increments in this pilot.
    from decimal import Decimal, ROUND_FLOOR
    tick = Decimal(str(quote['min_tick']))
    limit = float((Decimal(str(reference * .98)) / tick).to_integral_value(rounding=ROUND_FLOOR) * tick)
    if not 0 < limit <= cfg.risk.max_order_notional:
        raise ValueError('Transport probe exceeds one-order paper cap')
    before = broker.snapshot()
    if cfg.risk.kill_switch:
        raise ValueError('Kill switch blocks the transport probe')
    usage = ledger.daily_usage(broker.key, day.isoformat())
    if usage['n'] >= cfg.risk.max_daily_trades or usage['notional'] + limit > cfg.risk.max_daily_notional:
        raise ValueError('Transport probe exceeds the durable daily order budget')
    if before['positions'].get(str(quote['conid']), 0) or before['open_orders'] or ledger.unresolved(broker.key):
        raise ValueError('Transport probe requires no position in the symbol and no unresolved orders')
    order = {'symbol': symbol, 'conid': quote['conid'], 'side': 'BUY', 'quantity': 1, 'limit_price': limit}
    order_state = broker.gateway.ib.whatIfOrder(broker.md.get_contract(symbol),
        LimitOrder('BUY', 1, limit, account=broker.account, tif='DAY', outsideRth=False))
    if order_state is None:
        raise ValueError('Broker did not acknowledge the what-if order (check Read-Only API)')
    result = broker.execute([order], ledger, settings.strategy_id + '_transport',
                            'one_transport_probe_per_session', 3)
    good = bool(result and result[0].get('perm_id') and
                result[0]['status'] in {'Cancelled', 'ApiCancelled'} and result[0]['filled'] == 0)
    return {'purpose': 'transport_only_not_strategy_execution', 'reference_price': reference,
            'plan': order, 'what_if_status': order_state.status,
            'orders': result, 'status': 'accepted_and_cancelled' if good else 'review_required'}


def save_order_evidence(output, broker, ledger, strategy_id):
    rows = [dict(r) for r in ledger.db.execute('''SELECT ref,symbol,side,quantity,limit_price,status,
        order_id,perm_id,filled_quantity FROM intents WHERE account=? AND strategy=?''',
        (broker.key, strategy_id + '_transport'))]
    if rows:
        passed = all(r['perm_id'] and r['status'] in {'Cancelled', 'ApiCancelled'}
                     and r['filled_quantity'] == 0 for r in rows)
        write_json(output / 'transport_receipt.json', {
            'purpose': 'transport_only_not_strategy_execution',
            'status': 'accepted_and_cancelled' if passed else 'review_required',
            'verified_at_utc': dt.datetime.now(dt.timezone.utc).isoformat(), 'orders': rows})
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['research', 'preview', 'submit', 'reconcile', 'daily', 'transport-check'])
    parser.add_argument('--profile', choices=['etf', 'equity'], default='etf')
    parser.add_argument('--refresh', action='store_true', help='Download paired IBKR daily bars before research.')
    parser.add_argument('--submit-paper-orders', action='store_true')
    parser.add_argument('--account', help='Explicit DU account, required for any order submission.')
    parser.add_argument('--port', type=int, choices=[4002, 7497])
    parser.add_argument('--probe-symbol', default='IEF')
    args = parser.parse_args(argv)
    committing = args.command in {'submit', 'transport-check'} or (args.command == 'daily' and args.submit_paper_orders)
    if committing and (not args.submit_paper_orders or not args.account or not args.account.startswith('DU')):
        parser.error('Submitting requires --submit-paper-orders and --account DU...')
    if args.submit_paper_orders and not committing:
        parser.error('Order flag is only valid for submit, daily or transport-check')
    if args.command == 'transport-check' and args.profile != 'etf':
        parser.error('The transport check is restricted to the ETF pilot')
    settings = load_settings(ROOT / f'config/ibkr_{args.profile}_workflow.yaml')
    config = yaml.safe_load((ROOT / settings.research_config).read_text(encoding='utf-8'))
    if args.probe_symbol not in config['symbols'] and args.command == 'transport-check':
        parser.error('Probe symbol must belong to the ETF research universe')
    directory = ROOT / config['output_dir']
    output = directory / settings.strategy_id
    output.mkdir(parents=True, exist_ok=True)
    cfg = load_paper_config(ROOT / 'config/ibkr_paper.yaml')
    cfg.ib.client_id = 29
    cfg.ib.readonly = not committing
    cfg.ib.account = args.account or ''
    if args.port:
        cfg.ib.port = args.port
    cfg.ib.validate()
    logger.remove()
    logger.add(sys.stderr, level='WARNING')
    summary, report = None, {'command': args.command, 'paper_orders_requested': committing,
                            'started_at_utc': dt.datetime.now(dt.timezone.utc).isoformat()}
    exit_code = 0
    try:
        with exclusive_run(ROOT / 'data/live/ibkr_workflow/workflow.lock'):
            if args.refresh:
                # This downloader only reads IBKR and writes paired vendor caches.
                completed = subprocess.run([sys.executable, str(ROOT / 'scripts/ibkr_daily_research.py'),
                    '--research-config', str(ROOT / settings.research_config), '--port', str(cfg.ib.port)], cwd=ROOT)
                if completed.returncode:
                    raise ValueError('IBKR download/data-quality step failed; inspect its manifest')
            if args.command in {'research', 'daily'}:
                summary = run_research(ROOT, settings)
                report['status'] = 'research_complete'
            else:
                summary = json.loads((output / 'research_summary.json').read_text(encoding='utf-8'))
                if summary['settings'] != asdict(settings):
                    raise ValueError('Research settings changed; rerun research before brokerage operations')
                if digest(output / 'target_weights.parquet') != summary['weights_sha256']:
                    raise ValueError('Target weights changed after research')
                frame = pd.read_parquet(output / 'target_weights.parquet')
                if (frame.ticker.duplicated().any() or not pd.to_datetime(frame.date).eq(pd.Timestamp(summary['signal_session'])).all()
                        or dict(zip(frame.ticker, frame.weight)) != summary['target_weights']):
                    raise ValueError('Signal summary and dated weight artifact disagree')
                for name, expected in summary['provenance']['source_sha256'].items():
                    if digest(directory / name) != expected:
                        raise ValueError('Source cache changed; rerun research before brokerage operations')
            if args.command != 'research':
                ledger = Ledger(ROOT / 'data/live/ibkr_workflow/ledger.sqlite3')
                try:
                    with PaperBroker(cfg.ib) as broker:
                        broker.sync_intents(ledger)
                        before = broker.snapshot()
                        report['account'] = before
                        reconciliation = ledger.reconcile(broker.key, before, broker.fills())
                        report['reconciliation_before'] = reconciliation
                        broker.sync_intents(ledger)
                        if args.command == 'transport-check':
                            try:
                                report['transport_check'] = transport_check(broker, ledger, cfg, directory,
                                                                           args.probe_symbol, settings)
                            finally:
                                report['reconciliation_after'] = ledger.reconcile(broker.key, broker.snapshot(), broker.fills())
                            if report['transport_check']['status'] != 'accepted_and_cancelled':
                                exit_code = 2
                        elif args.command == 'reconcile':
                            report['reconciliation_after'] = reconciliation
                            report['status'] = 'reconciled' if reconciliation['status'] == 'matched' else 'review_required'
                            exit_code = 0 if reconciliation['status'] == 'matched' else 2
                        else:
                            weights = summary['target_weights']
                            quotes = broker.quotes(list(weights))
                            report['quotes'] = quotes
                            # Refresh cash/PnL after market-data subscriptions have populated.
                            before = broker.snapshot()
                            plan = order_plan(weights, quotes, before['positions'], settings.capital_usd,
                                              settings.min_trade_usd, cfg.strategy.limit_offset_bps)
                            report['plan'] = plan
                            session = dt.datetime.now(ZoneInfo('America/New_York')).date().isoformat()
                            gates = execution_gates(summary, before, quotes, weights, plan, cfg,
                                owned=ledger.owned_positions(broker.key, settings.strategy_id),
                                reconciliation=reconciliation, unresolved=ledger.unresolved(broker.key),
                                usage=ledger.daily_usage(broker.key, session))
                            prior = ledger.db.execute('SELECT 1 FROM intents WHERE account=? AND strategy=? LIMIT 1',
                                                      (broker.key, settings.strategy_id)).fetchone()
                            import exchange_calendars as xc
                            cal = xc.get_calendar('XNYS')
                            day = pd.Timestamp(session)
                            if prior and cal.is_session(day) and cal.previous_session(day).to_period('W-FRI') == day.to_period('W-FRI'):
                                gates.append('weekly_rebalance_not_due')
                            report['submission_blockers'] = sorted(set(gates))
                            report['strategy_orders'] = []
                            if committing and not gates:
                                try:
                                    report['strategy_orders'] = broker.execute(plan, ledger, settings.strategy_id,
                                        summary['weights_sha256'], settings.execution_window_seconds)
                                finally:
                                    report['reconciliation_after'] = ledger.reconcile(broker.key, broker.snapshot(), broker.fills())
                            else:
                                report['reconciliation_after'] = ledger.reconcile(broker.key, broker.snapshot(), broker.fills())
                            report['status'] = 'blocked' if gates else ('submitted' if committing else 'preview_ready')
                            if gates:
                                exit_code = 2
                        report['durable_transport_orders'] = save_order_evidence(output, broker, ledger, settings.strategy_id)
                        if ledger.unresolved(broker.key):
                            report['status'] = 'unresolved_order_requires_review'
                            exit_code = 2
                finally:
                    ledger.close()
    except Exception as exc:
        # Account IDs never belong in shareable reports or console output.
        message = f'{type(exc).__name__}: {exc}'
        message = message.replace(args.account, 'DU***') if args.account else message
        import re
        report['error'] = re.sub(r'\bDU\d+\b', 'DU***', message)
        report['status'] = 'failed'
        exit_code = 2
    report['finished_at_utc'] = dt.datetime.now(dt.timezone.utc).isoformat()
    stamp = dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    write_json(output / f'{args.command}_{stamp}.json', report)
    write_json(output / 'latest_broker_report.json', report)
    render_report(output, summary, report)
    print(json.dumps({'profile': args.profile, 'command': args.command, 'exit_code': exit_code,
                      'report': str(output / 'workflow_report.html'),
                      'status': report.get('status', report.get('transport_check', {}).get('status', 'research_complete')),
                      'blockers': report.get('submission_blockers', []), 'error': report.get('error')},
                     indent=2, ensure_ascii=False))
    return exit_code


if __name__ == '__main__':
    raise SystemExit(main())
