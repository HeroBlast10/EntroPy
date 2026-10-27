"""Reproducible daily IBKR pilots; next-session-open, self-financing research.

Research holds fractional units of a vendor total-return adjusted series. It
does not claim a historical whole-share execution replay or PIT membership.
The broker boundary separately sizes whole shares from current bid/ask.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path

import exchange_calendars as xc
import numpy as np
import pandas as pd
import yaml


@dataclass(frozen=True)
class WorkflowSettings:
    strategy_id: str
    kind: str
    research_config: str
    capital_usd: float = 12000.0
    gross_weight: float = 0.20
    max_weight: float = 0.10
    top_k: int = 3
    cost_bps: float = 10.0
    minimum_fee_usd: float = 1.0
    min_trade_usd: float = 50.0
    holdout_sessions: int = 252
    execution_window_seconds: int = 45

    def __post_init__(self):
        if self.kind not in {'etf', 'equity'} or not self.strategy_id.replace('_', '').isalnum():
            raise ValueError('Invalid strategy kind/id')
        for name in ('capital_usd', 'gross_weight', 'max_weight', 'cost_bps',
                     'minimum_fee_usd', 'min_trade_usd'):
            value = getattr(self, name)
            if isinstance(value, bool) or not np.isfinite(value) or value <= 0:
                raise ValueError(f'{name} must be finite and positive')
        if not 0 < self.max_weight <= self.gross_weight <= 1:
            raise ValueError('Weights must obey 0 < max_weight <= gross_weight <= 1')
        for name in ('top_k', 'holdout_sessions', 'execution_window_seconds'):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f'{name} must be a positive integer')
        if self.execution_window_seconds > 300:
            raise ValueError('Execution window is limited to 300 seconds')


def load_settings(path: Path) -> WorkflowSettings:
    raw = yaml.safe_load(path.read_text(encoding='utf-8'))
    if not isinstance(raw, dict) or set(raw) - {f.name for f in fields(WorkflowSettings)}:
        raise ValueError('Unknown workflow settings')
    return WorkflowSettings(**raw)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value) -> None:
    def clean(obj):
        if isinstance(obj, dict):
            return {str(k): clean(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [clean(v) for v in obj]
        if isinstance(obj, (float, np.floating)):
            return float(obj) if np.isfinite(obj) else None
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (dt.datetime, dt.date, pd.Timestamp, Path)):
            return str(obj)
        return obj
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(clean(value), indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    temp.replace(path)


def last_complete_session(now=None) -> pd.Timestamp:
    now = pd.Timestamp(now if now is not None else dt.datetime.now(dt.timezone.utc))
    if now.tzinfo is None:
        raise ValueError('An aware timestamp is required')
    cal = xc.get_calendar('XNYS')
    day = now.tz_convert('America/New_York').date()
    sessions = cal.sessions_in_range(day - dt.timedelta(days=14), day)
    complete = [s for s in sessions if cal.session_close(s) + pd.Timedelta(minutes=15) <= now]
    if not complete:
        raise ValueError('No completed session in calendar')
    return complete[-1].tz_localize(None).normalize()


def regular_session(now=None) -> bool:
    now = pd.Timestamp(now if now is not None else dt.datetime.now(dt.timezone.utc))
    if now.tzinfo is None:
        raise ValueError('An aware timestamp is required')
    cal = xc.get_calendar('XNYS')
    day = pd.Timestamp(now.tz_convert('America/New_York').date())
    return bool(cal.is_session(day) and cal.session_open(day) <= now < cal.session_close(day))


def load_panel(directory: Path, symbols: list[str], now=None) -> tuple[pd.DataFrame, dict]:
    """Require complete paired tapes on the common calendar; never forward-fill."""
    manifest = json.loads((directory / 'download_manifest.json').read_text(encoding='utf-8'))
    if manifest.get('source') != 'IBKR TWS API':
        raise ValueError('This workflow requires actual IBKR historical data')
    if manifest.get('bar_size') != '1 day' or manifest.get('useRTH') is not True:
        raise ValueError('Research requires regular-session daily bars')
    if set(symbols) & set(manifest.get('failures', {})):
        raise ValueError('Requested symbols have failed IBKR downloads')
    frames, hashes = [], {}
    cutoff = last_complete_session(now)
    for symbol in dict.fromkeys(symbols):
        pair = {}
        for kind in ('TRADES', 'ADJUSTED_LAST'):
            path = directory / f'{symbol}_{kind}.parquet'
            hashes[path.name] = digest(path)
            frame = pd.read_parquet(path)[['date', 'symbol', 'open', 'high', 'low', 'close', 'volume']].copy()
            if not frame.symbol.eq(symbol).all():
                raise ValueError(f'Wrong contract data for {symbol}')
            frame['date'] = pd.to_datetime(frame.date).dt.normalize()
            frame = frame[frame.date <= cutoff].sort_values('date')
            if frame.empty or frame.date.duplicated().any():
                raise ValueError(f'Empty or duplicate {symbol} {kind} bars')
            values = frame[['open', 'high', 'low', 'close', 'volume']].to_numpy(dtype=float)
            if (not np.isfinite(values).all() or (values[:, :4] <= 0).any()
                    or (values[:, 4] < 0).any()
                    or (frame.high < frame[['open', 'close', 'low']].max(axis=1)).any()
                    or (frame.low > frame[['open', 'close', 'high']].min(axis=1)).any()):
                raise ValueError(f'Invalid {symbol} {kind} OHLCV')
            pair[kind] = frame
        raw = pair['TRADES'].rename(columns={'symbol': 'ticker'})
        adj = pair['ADJUSTED_LAST'][['date', 'close']].rename(columns={'close': 'adjusted_close'})
        raw = raw.merge(adj, on='date', validate='one_to_one', how='outer')
        if raw[['close', 'adjusted_close']].isna().any().any():
            raise ValueError(f'Mismatched paired bars: {symbol}')
        raw['adjusted_open'] = raw.open * raw.adjusted_close / raw.close
        frames.append(raw)
    # IB may provide shorter history for a listing. Trim only the shared start;
    # do not erase a trailing stale/missing session by intersecting dates.
    start = max(f.date.min() for f in frames)
    end = max(f.date.max() for f in frames)
    expected = xc.get_calendar('XNYS').sessions_in_range(start, end).tz_localize(None)
    for frame in frames:
        observed = pd.DatetimeIndex(frame.loc[frame.date >= start, 'date'])
        missing, extra = expected.difference(observed), observed.difference(expected)
        if len(missing) or len(extra):
            raise ValueError(f'Calendar gaps for {frame.ticker.iloc[0]}: missing={len(missing)}, extra={len(extra)}')
    panel = pd.concat([f[f.date >= start] for f in frames], ignore_index=True)
    provenance = {'source': 'IBKR TWS API', 'downloaded_at_utc': manifest['downloaded_at_utc'],
                  'source_sha256': hashes, 'common_start': str(start.date()),
                  'last_session': str(end.date()), 'expected_last_session': str(cutoff.date()),
                  'data_current': bool(end == cutoff), 'point_in_time_universe': False,
                  'vendor_adjustments_are_point_in_time': False}
    return panel.sort_values(['ticker', 'date']).reset_index(drop=True), provenance


def features(panel: pd.DataFrame) -> pd.DataFrame:
    result = panel.sort_values(['ticker', 'date']).copy()
    g = result.groupby('ticker', sort=False)
    result['momentum'] = g.adjusted_close.transform(lambda s: s.shift(21) / s.shift(252) - 1)
    result['volatility'] = g.adjusted_close.transform(
        lambda s: s.pct_change(fill_method=None).rolling(20, min_periods=20).std() * np.sqrt(252))
    result['reversal'] = g.adjusted_close.transform(lambda s: -(s / s.shift(21) - 1))
    result['adv20'] = (result.close * result.volume).groupby(result.ticker).transform(
        lambda s: s.rolling(20, min_periods=20).mean())
    return result


def target_weights(day: pd.DataFrame, symbols: list[str], settings: WorkflowSettings,
                   min_adv: float, min_price: float, baseline=False) -> dict[str, float]:
    valid = day[day.ticker.isin(symbols)].copy()
    if valid.ticker.duplicated().any():
        raise ValueError('Duplicate signal row')
    valid = valid[np.isfinite(valid[['momentum', 'volatility', 'reversal', 'adv20']]).all(axis=1)
                  & valid.volatility.gt(0) & valid.adv20.ge(min_adv) & valid.close.ge(min_price)]
    weights = dict.fromkeys(symbols, 0.0)
    if baseline:
        for symbol in valid.ticker:
            weights[symbol] = min(settings.max_weight, settings.gross_weight / len(valid))
        return weights
    if settings.kind == 'etf':
        valid = valid[valid.momentum > 0]
        valid['score'] = valid.momentum
    else:
        # Fixed ex ante directions and equal ranks: no full-sample fitting.
        valid['score'] = (valid.momentum.rank(pct=True) +
                          (-valid.volatility).rank(pct=True) + valid.reversal.rank(pct=True)) / 3
    selected = valid.sort_values(['score', 'ticker'], ascending=[False, True]).head(settings.top_k)
    if not selected.empty:
        inverse = 1 / selected.volatility
        allocated = settings.gross_weight * inverse / inverse.sum()
        for symbol, weight in zip(selected.ticker, allocated):
            weights[symbol] = min(settings.max_weight, float(weight))
    return weights


def replay(panel: pd.DataFrame, symbols: list[str], settings: WorkflowSettings,
           min_adv: float, min_price: float, *, cost_bps=None, baseline=False):
    """T-close decisions -> T+1 open; actual drift, cash and deducted costs."""
    dates = sorted(panel.date.unique())
    grouped = {pd.Timestamp(d): f for d, f in panel.groupby('date')}
    units = pd.Series(0.0, index=symbols)
    cash, previous_nav = settings.capital_usd, settings.capital_usd
    daily, trades, initialized = [], [], False
    rate = (settings.cost_bps if cost_bps is None else cost_bps) / 10000
    for i in range(253, len(dates)):
        date, signal_date = pd.Timestamp(dates[i]), pd.Timestamp(dates[i - 1])
        day = grouped[date].set_index('ticker').loc[symbols]
        opening, closing = day.adjusted_open, day.adjusted_close
        nav_open = cash + float((units * opening).sum())
        cost, turnover = 0.0, 0.0
        week_changed = date.to_period('W-FRI') != signal_date.to_period('W-FRI')
        if not initialized or week_changed:
            weights = pd.Series(target_weights(grouped[signal_date], symbols, settings, min_adv, min_price, baseline))
            target = nav_open * weights / opening
            delta = target - units
            notionals = delta * opening
            active = notionals.abs() >= settings.min_trade_usd
            delta = delta.where(active, 0.0)
            notionals = delta * opening
            cost = float(notionals.abs().sum() * rate + active.sum() * settings.minimum_fee_usd)
            cash -= float(notionals.sum()) + cost
            if cash < -1e-6:
                raise ValueError('Research target would borrow cash after costs')
            turnover = float(notionals.abs().sum() / nav_open)
            units = units + delta
            for symbol in notionals.index[active]:
                trades.append({'date': date, 'signal_date': signal_date, 'ticker': symbol,
                               'signed_notional_usd': float(notionals[symbol])})
            initialized = True
        nav = cash + float((units * closing).sum())
        daily.append({'date': date, 'nav': nav, 'return': nav / previous_nav - 1,
                      'cost_usd': cost, 'two_way_turnover': turnover,
                      'gross_exposure': float((units * closing).sum() / nav), 'cash_usd': cash})
        previous_nav = nav
    if not daily:
        raise ValueError('Need at least 254 complete common sessions')
    return pd.DataFrame(daily), pd.DataFrame(trades)


def metrics(daily: pd.DataFrame) -> dict:
    returns = daily['return']
    growth = (1 + returns).cumprod()
    peak = growth.cummax().clip(lower=1.0)
    std = float(returns.std(ddof=1))
    return {'sessions': len(daily), 'total_return': float(growth.iloc[-1] - 1),
            'annual_return': float(growth.iloc[-1] ** (252 / len(daily)) - 1),
            'annual_volatility': std * np.sqrt(252),
            'sharpe_zero_cash_rate': float(returns.mean() / std * np.sqrt(252)) if std > 0 else None,
            'max_drawdown': float((growth / peak - 1).min()),
            'cost_usd': float(daily.cost_usd.sum()),
            'mean_gross_exposure': float(daily.gross_exposure.mean())}


def run_research(root: Path, settings: WorkflowSettings) -> dict:
    config = yaml.safe_load((root / settings.research_config).read_text(encoding='utf-8'))
    directory = root / config['output_dir']
    symbols = config['symbols']
    if len(set(symbols)) != len(symbols) or settings.top_k > len(symbols):
        raise ValueError('Invalid research universe')
    requested = list(dict.fromkeys(symbols + config['benchmarks']))
    panel, provenance = load_panel(directory, requested)
    panel = features(panel)
    daily, trades = replay(panel, symbols, settings, config['min_adv20_usd'], config['min_price_usd'])
    if len(daily) <= settings.holdout_sessions + 252:
        raise ValueError('Insufficient development and holdout history')
    baseline, _ = replay(panel, symbols, settings, config['min_adv20_usd'], config['min_price_usd'], baseline=True)
    benchmark_settings = replace(settings, max_weight=settings.gross_weight, top_k=1)
    benchmark, _ = replay(panel, [config['benchmarks'][0]], benchmark_settings,
                          config['min_adv20_usd'], config['min_price_usd'], baseline=True)
    holdout = daily.tail(settings.holdout_sessions)
    stress = {}
    for bps in sorted(set([settings.cost_bps, 20.0, 40.0])):
        replayed, _ = replay(panel, symbols, settings, config['min_adv20_usd'], config['min_price_usd'], cost_bps=bps)
        stress[str(bps)] = metrics(replayed.tail(settings.holdout_sessions))
    latest = panel.date.max()
    weights = target_weights(panel[panel.date == latest], symbols, settings,
                             config['min_adv20_usd'], config['min_price_usd'])
    output = directory / settings.strategy_id
    output.mkdir(parents=True, exist_ok=True)
    daily.to_csv(output / 'backtest_daily.csv', index=False)
    baseline.to_csv(output / 'baseline_daily.csv', index=False)
    benchmark.to_csv(output / 'market_benchmark_daily.csv', index=False)
    trades.to_csv(output / 'backtest_trades.csv', index=False)
    panel[panel.date == latest].to_csv(output / 'latest_signals.csv', index=False)
    pd.DataFrame({'date': latest, 'ticker': list(weights), 'weight': list(weights.values())}).to_parquet(
        output / 'target_weights.parquet', index=False)
    summary = {'strategy_id': settings.strategy_id, 'kind': settings.kind, 'settings': asdict(settings),
               'provenance': provenance, 'signal_session': str(latest.date()),
               'full_sample': metrics(daily), 'development': metrics(daily.iloc[:-settings.holdout_sessions]),
               'holdout_start': str(holdout.date.iloc[0].date()), 'holdout': metrics(holdout),
               'holdout_equal_weight_baseline': metrics(baseline.tail(settings.holdout_sessions)),
               'holdout_market_benchmark': metrics(benchmark.tail(settings.holdout_sessions)),
               'cost_stress_holdout': stress, 'target_weights': weights,
               'weights_sha256': digest(output / 'target_weights.parquet'),
               'created_at_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
               'strategy_validated_for_live': False,
               'limitations': ['Fixed surviving current-symbol universe; no PIT membership claim.',
                 'Fractional adjusted total-return research units, not historical physical whole-share fills.',
                 'Synthetic adjusted opens inherit vendor adjustment revisions.',
                 'Costs are modeled bps plus per-symbol minimum fees; cash interest is zero.',
                 'Holdout is time-separated but has not been observed prospectively.',
                 'Equity factors are momentum, low volatility and short-term reversal; no fundamental/earnings data.']}
    write_json(output / 'research_summary.json', summary)
    return summary
