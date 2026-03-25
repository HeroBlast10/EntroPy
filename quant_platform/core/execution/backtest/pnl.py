"""PnL engine: daily mark-to-market, gross/net returns, cost attribution.

Produces two core outputs:

1. **Portfolio-level daily series** — NAV, gross return, net return,
   cumulative cost, drawdown.
2. **Cost attribution table** — how much of the drag comes from
   commission vs slippage vs impact vs regulatory fees.

The engine is purely vectorised (no per-date loop) for speed.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from quant_platform.core.execution.cost_models.us_equity import CostModel, daily_borrow_cost


# ===================================================================
# Daily portfolio return computation
# ===================================================================

def compute_daily_returns(
    daily_weights: pd.DataFrame,
    prices: pd.DataFrame,
    trades: Optional[pd.DataFrame] = None,
    cost_model: Optional[CostModel] = None,
    initial_capital: float = 1_000_000.0,
) -> pd.DataFrame:
    """Compute daily gross and net portfolio returns.

    Parameters
    ----------
    daily_weights : ``[date, ticker, weight]`` — target weight on each day.
    prices : ``[date, ticker, adj_close, close, volume]``.
    trades : output from ``simulate_execution()`` (with cost columns).
        If ``None``, only gross returns are computed.
    cost_model : for overnight borrow cost calculation.
    initial_capital : starting NAV.

    Returns
    -------
    DataFrame indexed by date with columns:
        ``gross_ret, net_ret, trading_cost, borrow_cost,
        total_cost, nav_gross, nav_net, drawdown_gross, drawdown_net``
    """
    if cost_model is None:
        cost_model = CostModel()

    dw = daily_weights.copy()
    px = prices.copy()
    dw["date"] = pd.to_datetime(dw["date"])
    px["date"] = pd.to_datetime(px["date"])

    # --- Stock-level daily returns ---
    px = px.sort_values(["ticker", "date"])
    px["stock_ret"] = px.groupby("ticker")["adj_close"].pct_change()

    # Merge weights with returns
    merged = dw.merge(px[["date", "ticker", "stock_ret"]], on=["date", "ticker"], how="left")
    merged["stock_ret"] = merged["stock_ret"].fillna(0.0)

    # Weighted return per stock
    merged["contrib"] = merged["weight"] * merged["stock_ret"]

    # --- Gross return ---
    daily_gross = merged.groupby("date")["contrib"].sum()
    daily_gross.name = "gross_ret"

    # --- Trading cost per day ---
    dates = sorted(daily_gross.index)
    date_idx = pd.DatetimeIndex(dates, name="date")

    if trades is not None and not trades.empty:
        trades["date"] = pd.to_datetime(trades["date"])
        daily_trade_cost = trades.groupby("date")["total_cost"].sum()
    else:
        daily_trade_cost = pd.Series(0.0, index=date_idx, name="total_cost")

    daily_trade_cost = daily_trade_cost.reindex(date_idx, fill_value=0.0)

    # --- Borrow cost (short positions) ---
    short_notional_per_day = pd.Series(0.0, index=date_idx)
    if (dw["weight"] < 0).any():
        short_w = dw[dw["weight"] < 0].copy()
        short_w["abs_weight"] = short_w["weight"].abs()
        short_agg = short_w.groupby("date")["abs_weight"].sum()
        short_notional_per_day = short_agg.reindex(date_idx, fill_value=0.0) * initial_capital

    daily_borrow = short_notional_per_day.apply(
        lambda n: daily_borrow_cost(n, cost_model)
    )

    # --- Assemble ---
    result = pd.DataFrame(index=date_idx)
    result["gross_ret"] = daily_gross.reindex(date_idx, fill_value=0.0)

    # Express costs as fraction of NAV (not dollar)
    # For the first day, use initial_capital; thereafter use running NAV
    result["trading_cost_dollar"] = daily_trade_cost.values
    result["borrow_cost_dollar"] = daily_borrow.values
    result["total_cost_dollar"] = result["trading_cost_dollar"] + result["borrow_cost_dollar"]

    # Build NAV series
    nav_gross = [initial_capital]
    nav_net = [initial_capital]
    for i, dt in enumerate(result.index):
        g = result["gross_ret"].iloc[i]
        tc = result["total_cost_dollar"].iloc[i]
        # Gross NAV
        new_nav_g = nav_gross[-1] * (1 + g)
        nav_gross.append(new_nav_g)
        # Net NAV (subtract dollar cost)
        new_nav_n = nav_net[-1] * (1 + g) - tc
        nav_net.append(new_nav_n)

    result["nav_gross"] = nav_gross[1:]
    result["nav_net"] = nav_net[1:]

    # Net return (including cost drag)
    result["net_ret"] = result["nav_net"].pct_change()
    result.loc[result.index[0], "net_ret"] = (
        result["nav_net"].iloc[0] / initial_capital - 1.0
    )

    # Cost as fraction of NAV
    result["trading_cost_bps"] = (
        result["trading_cost_dollar"] / result["nav_net"].shift(1).fillna(initial_capital) * 10_000
    )
    result["borrow_cost_bps"] = (
        result["borrow_cost_dollar"] / result["nav_net"].shift(1).fillna(initial_capital) * 10_000
    )

    # Drawdown
    result["peak_gross"] = result["nav_gross"].cummax()
    result["peak_net"] = result["nav_net"].cummax()
    result["drawdown_gross"] = result["nav_gross"] / result["peak_gross"] - 1.0
    result["drawdown_net"] = result["nav_net"] / result["peak_net"] - 1.0

    # Clean up helper columns
    result.drop(columns=["peak_gross", "peak_net"], inplace=True)

    logger.info(
        "PnL computed: {} days, gross={:.2%} cum, net={:.2%} cum, max_dd={:.2%}",
        len(result),
        result["nav_gross"].iloc[-1] / initial_capital - 1,
        result["nav_net"].iloc[-1] / initial_capital - 1,
        result["drawdown_net"].min(),
    )

    return result


# ===================================================================
# Cost attribution
# ===================================================================

def cost_attribution(trades: pd.DataFrame) -> pd.DataFrame:
    """Break down total trading cost by component.

    Returns a DataFrame with one row per component:
    commission, slippage, impact, sec_fee, finra_taf, stamp_duty.
    """
    cost_cols = ["commission", "slippage", "impact", "sec_fee", "finra_taf", "stamp_duty"]
    available = [c for c in cost_cols if c in trades.columns]

    total = trades[available].sum()
    total_notional = trades["notional"].sum() if "notional" in trades.columns else 1.0

    attr = pd.DataFrame({
        "component": total.index,
        "total_dollar": total.values,
        "pct_of_total": total.values / max(total.sum(), 1e-10),
        "bps_of_notional": total.values / max(total_notional, 1e-10) * 10_000,
    })

    # Add summary row
    summary = pd.DataFrame([{
        "component": "TOTAL",
        "total_dollar": total.sum(),
        "pct_of_total": 1.0,
        "bps_of_notional": total.sum() / max(total_notional, 1e-10) * 10_000,
    }])
    attr = pd.concat([attr, summary], ignore_index=True)

    return attr


# ===================================================================
# Performance summary statistics
# ===================================================================

def performance_summary(
    daily_returns: pd.DataFrame,
    annualisation: int = 252,
    benchmark_returns: Optional[pd.Series] = None,
    risk_free_rate: float = 0.0,
) -> Dict[str, float]:
    """Compute standard performance metrics from daily PnL.

    Parameters
    ----------
    daily_returns : output from :func:`compute_daily_returns`.
    annualisation : number of periods per year (252 for daily)
    benchmark_returns : optional benchmark returns for alpha/beta analysis
    risk_free_rate : annualized risk-free rate (e.g., 0.03 for 3%)

    Returns
    -------
    Dict of headline numbers, including benchmark-relative metrics if benchmark provided.
    """
    dr = daily_returns
    n = len(dr)
    if n == 0:
        return {}

    def _ann_ret(series):
        cum = (1 + series).prod()
        years = n / annualisation
        return cum ** (1 / max(years, 1e-6)) - 1

    def _ann_vol(series):
        return series.std() * np.sqrt(annualisation)

    def _sharpe(series):
        vol = _ann_vol(series)
        return _ann_ret(series) / vol if vol > 0 else np.nan

    def _sortino(series):
        downside = series[series < 0].std() * np.sqrt(annualisation)
        return _ann_ret(series) / downside if downside > 0 else np.nan

    def _calmar(series, dd_series):
        max_dd = dd_series.min()
        return _ann_ret(series) / abs(max_dd) if max_dd < 0 else np.nan

    gross = dr["gross_ret"]
    net = dr["net_ret"]

    summary = {
        # Gross
        "gross_ann_return": round(_ann_ret(gross), 6),
        "gross_ann_vol": round(_ann_vol(gross), 6),
        "gross_sharpe": round(_sharpe(gross), 4),
        "gross_sortino": round(_sortino(gross), 4),
        "gross_calmar": round(_calmar(gross, dr["drawdown_gross"]), 4),
        "gross_max_drawdown": round(dr["drawdown_gross"].min(), 6),
        # Net
        "net_ann_return": round(_ann_ret(net), 6),
        "net_ann_vol": round(_ann_vol(net), 6),
        "net_sharpe": round(_sharpe(net), 4),
        "net_sortino": round(_sortino(net), 4),
        "net_calmar": round(_calmar(net, dr["drawdown_net"]), 4),
        "net_max_drawdown": round(dr["drawdown_net"].min(), 6),
        # Cost drag
        "total_trading_cost_bps": round(dr["trading_cost_bps"].sum(), 2),
        "total_borrow_cost_bps": round(dr["borrow_cost_bps"].sum(), 2),
        "avg_daily_cost_bps": round(
            (dr["trading_cost_bps"] + dr["borrow_cost_bps"]).mean(), 4
        ),
        # General
        "total_days": n,
        "start_date": str(dr.index[0].date()),
        "end_date": str(dr.index[-1].date()),
    }
    
    # Add VaR/CVaR risk metrics
    from quant_platform.core.evaluation.risk_metrics import risk_metrics_summary
    
    risk_metrics = risk_metrics_summary(net, confidence=0.95)
    summary.update({
        "var_95_daily": risk_metrics["var_historical"],
        "cvar_95_daily": risk_metrics["cvar_historical"],
        "var_95_parametric": risk_metrics["var_parametric"],
        "var_95_cornish_fisher": risk_metrics["var_cornish_fisher"],
        "return_skewness": risk_metrics["skewness"],
        "return_kurtosis": risk_metrics["kurtosis"],
    })
    
    # Add benchmark-relative metrics if benchmark provided
    if benchmark_returns is not None and len(benchmark_returns) > 0:
        from quant_platform.core.evaluation.benchmark_analytics import benchmark_analysis
        
        # Align benchmark to portfolio dates
        bench_aligned = benchmark_returns.reindex(dr.index)
        bench_aligned = bench_aligned.dropna()
        
        if len(bench_aligned) >= 10:
            # Use net returns for benchmark comparison
            port_aligned = net.reindex(bench_aligned.index)
            
            bench_metrics = benchmark_analysis(
                port_aligned,
                bench_aligned,
                risk_free_rate=risk_free_rate,
                annualization=annualisation,
            )
            
            # Add benchmark metrics to summary
            summary.update(bench_metrics)
        else:
            logger.warning("Insufficient overlapping dates for benchmark analysis")

    return summary
