#!/usr/bin/env python
"""Small read-only IBKR daily pilot: cache bars, validate and inspect features.

No orders or target-weight output. The fixed current-symbol sample contains
survivorship/selection bias. Vendor adjustments are not a PIT adjustment series.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import html
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import yaml
from quant_platform.core.execution.paper.ibkr.config import load_paper_config
from quant_platform.core.execution.paper.ibkr.gateway import IBGateway
from quant_platform.core.execution.paper.ibkr.market_data import MarketDataManager
from quant_platform.core.execution.paper.ibkr.risk import trading_date


def prepare_panel(trades: pd.DataFrame, adjusted: pd.DataFrame, cutoff: dt.date) -> pd.DataFrame:
    """Validate vendor data, exclude the current session and keep provenance."""
    def clean(frame, label):
        frame = frame.copy()
        frame['date'] = pd.to_datetime(frame['date']).dt.normalize()
        frame = frame[frame.date.dt.date < cutoff].sort_values(['symbol', 'date'])
        if frame.duplicated(['symbol', 'date']).any():
            raise ValueError(f'Duplicate daily bars in {label}')
        for col in ['open', 'high', 'low', 'close', 'volume']:
            frame[col] = pd.to_numeric(frame[col], errors='raise')
            if not np.isfinite(frame[col]).all():
                raise ValueError(f'Nonfinite {label} {col}')
        if ((frame[['open','high','low','close']] <= 0).any().any()
                or (frame.volume < 0).any()
                or (frame.high < frame[['open','low','close']].max(axis=1)).any()
                or (frame.low > frame[['open','high','close']].min(axis=1)).any()):
            raise ValueError(f'Invalid OHLCV in {label}')
        return frame
    trades = clean(trades, 'TRADES')
    adjusted = clean(adjusted, 'ADJUSTED_LAST')
    adjusted = adjusted[['date','symbol','close']].rename(columns={'close':'adjusted_close'})
    panel = trades.merge(adjusted, on=['date','symbol'], how='left', validate='one_to_one')
    if panel.adjusted_close.isna().any():
        raise ValueError('Missing corresponding ADJUSTED_LAST close')
    panel = panel.rename(columns={'symbol':'ticker'}).sort_values(['ticker','date']).reset_index(drop=True)
    g = panel.groupby('ticker', sort=False)
    panel['mom_12_1'] = g.adjusted_close.transform(lambda x: x.shift(21) / x.shift(252) - 1)
    panel['vol_20'] = g.adjusted_close.transform(lambda x: x.pct_change(fill_method=None).rolling(20).std() * np.sqrt(252))
    # Last-close features are available only AFTER that session has closed.
    panel['adv_20_usd_proxy'] = (panel.close * panel.volume).groupby(panel.ticker).transform(lambda x:x.rolling(20).mean())
    return panel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ib-config', type=Path, default=ROOT/'config/ibkr_paper.yaml')
    parser.add_argument('--research-config', type=Path, default=ROOT/'config/ibkr_daily_research.yaml')
    parser.add_argument('--offline', action='store_true', help='Re-analyse the saved vendor caches.')
    args = parser.parse_args()
    research = yaml.safe_load(args.research_config.read_text(encoding='utf-8'))
    cfg = load_paper_config(args.ib_config)
    cfg.ib.readonly = True
    cfg.strategy.dry_run = True
    cfg.strategy.enable_paper_orders = False
    output = ROOT / research['output_dir']
    output.mkdir(parents=True, exist_ok=True)
    requested = research['symbols'] + research['benchmarks']
    cutoff = trading_date()
    manifest = {'downloaded_at_utc':dt.datetime.now(dt.timezone.utc).isoformat(),
                'source':'IBKR TWS API', 'duration':research['duration'], 'bar_size':'1 day',
                'useRTH':True, 'excluded_on_or_after':str(cutoff),
                'universe_kind':'fixed_current_symbol_pilot_not_PIT',
                'orders_submitted':0, 'contracts':{}, 'failures':{}}
    if not args.offline:
        gw = IBGateway(cfg.ib)
        try:
            gw.connect()
            md = MarketDataManager(gw)
            for symbol in requested:
                try:
                    contract = md.get_contract(symbol)
                    if contract is None:
                        raise ValueError('No qualified contract')
                    manifest['contracts'][symbol] = {'conId':contract.conId,
                        'primaryExchange':contract.primaryExchange, 'currency':contract.currency}
                    frames = {}
                    for kind in ['TRADES', 'ADJUSTED_LAST']:
                        frame = md.historical_bars(symbol, duration=research['duration'], what_to_show=kind)
                        if frame.empty:
                            raise ValueError(f'No {kind} bars')
                        frames[kind] = frame
                        gw.sleep(1)
                    # Do not overwrite a symbol cache with a partial pair.
                    for kind, frame in frames.items():
                        frame.to_parquet(output/f'{symbol}_{kind}.parquet', index=False)
                    print(f'{symbol}: {len(frames["TRADES"])} TRADES / {len(frames["ADJUSTED_LAST"])} adjusted bars', flush=True)
                except Exception as exc:
                    manifest['failures'][symbol] = str(exc)
                    print(f'{symbol}: FAILED {exc}', flush=True)
            (output/'download_manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf-8')
        finally:
            gw.disconnect()
    else:
        manifest = json.loads((output/'download_manifest.json').read_text(encoding='utf-8'))
    valid = [s for s in requested if s not in manifest['failures'] and
             all((output/f'{s}_{k}.parquet').exists() for k in ['TRADES','ADJUSTED_LAST'])]
    if not valid:
        raise RuntimeError('No complete paired symbol data')
    trades = pd.concat([pd.read_parquet(output/f'{s}_TRADES.parquet') for s in valid])
    adjusted = pd.concat([pd.read_parquet(output/f'{s}_ADJUSTED_LAST.parquet') for s in valid])
    panel = prepare_panel(trades, adjusted, cutoff)
    panel.to_parquet(output/'daily_features.parquet', index=False)
    latest_date = panel.date.max()
    latest = panel[panel.date == latest_date].copy()
    latest['role'] = latest.ticker.map(lambda t:'benchmark' if t in research['benchmarks'] else 'research_seed')
    latest['passes_price_liquidity'] = ((latest.close >= research['min_price_usd']) &
        (latest.adv_20_usd_proxy >= research['min_adv20_usd']))
    latest['momentum_rank_seed'] = latest.loc[latest.role=='research_seed','mom_12_1'].rank(ascending=False,method='min')
    latest = latest.sort_values(['role','momentum_rank_seed','ticker'])
    columns = ['date','ticker','role','close','adjusted_close','mom_12_1','vol_20',
               'adv_20_usd_proxy','passes_price_liquidity','momentum_rank_seed']
    latest[columns].to_csv(output/'latest_features.csv', index=False)
    qa = panel.groupby('ticker').agg(rows=('date','size'),start=('date','min'),end=('date','max'))
    import exchange_calendars as xc
    cal = xc.get_calendar('XNYS')
    qa['missing_sessions'] = [len(set(cal.sessions_in_range(row.start,row.end).tz_localize(None)) -
                                 set(panel.loc[panel.ticker==ticker,'date'])) for ticker,row in qa.iterrows()]
    qa.to_csv(output/'data_quality.csv')
    summary = {'symbols_requested':len(requested),'symbols_downloaded':len(valid),
               'rows':len(panel),'latest_complete_session':str(latest_date.date()),
               'failures':manifest['failures'],'missing_sessions':int(qa.missing_sessions.sum()),
               'fundamentals_available':False,'earnings_events_available':False,
               'strategy_validated':False,'orders_enabled':False,
               'limitations':['Fixed surviving current symbols; no historical membership or market-cap screen.',
                 'Vendor-adjusted prices are not point-in-time corporate-action data.',
                 'No PEAD or fundamental-quality signal yet; no cost-aware backtest or trade recommendation.']}
    summary['cache_sha256'] = {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in output.glob('*_*.parquet')}
    (output/'research_summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(11,5))
    for ticker, frame in panel.groupby('ticker'):
        ax.plot(frame.date, frame.adjusted_close/frame.adjusted_close.iloc[0]*100,
                label=ticker, linewidth=2 if ticker in research['benchmarks'] else 1)
    ax.set(title='IBKR daily pilot - vendor-adjusted prices (start=100)',ylabel='Index')
    ax.legend(ncol=5,fontsize=8)
    ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(output/'price_history.png',dpi=150)
    plt.close(fig)
    notes = '<ul>' + ''.join(f'<li>{html.escape(x)}</li>' for x in summary['limitations']) + '</ul>'
    report = f'''<!doctype html><html lang="zh"><meta charset="utf-8"><title>IBKR 日频研究样本</title>
<style>body{{font-family:system-ui;max-width:1300px;margin:32px auto;padding:16px;line-height:1.6}}table{{border-collapse:collapse;font-size:13px}}td,th{{padding:7px;border:1px solid #ddd}}img{{max-width:100%}}</style>
<h1>IBKR 日频研究样本</h1><p>最近完整交易日：{latest_date.date()}；{len(valid)} 个标的，{len(panel)} 条日线。</p>
<p>只读研究，自动下单关闭。固定样本用于数据与因子检查，尚未按历史市值验证为中小盘股票池。</p>
<p>动量 = P[t-21] / P[t-252] - 1；波动率为20日年化波动率；成交额为收盘价×股数的20日均值代理。特征在当日收盘后可用。</p>
<img src="price_history.png"><h2>最新特征（研究排序，不是交易建议）</h2>{latest[columns].to_html(index=False,float_format=lambda x:f'{x:,.4f}')}
<h2>数据完整性</h2>{qa.to_html()}<h2>研究边界</h2>{notes}</html>'''
    (output/'report.html').write_text(report,encoding='utf-8')
    print(json.dumps(summary,indent=2,ensure_ascii=False),flush=True)
    return 2 if manifest['failures'] or summary['missing_sessions'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
