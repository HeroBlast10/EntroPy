"""PnL engine: daily mark-to-market, gross/net returns, cost attribution.

Produces two core outputs:

1. **Portfolio-level daily series** — NAV, gross return, net return,
   cumulative cost, drawdown.
2. **Cost attribution table** — how much of the drag comes from
   commission vs slippage vs impact vs regulatory fees.

The engine is purely vectorised (no per-date loop) for speed.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Dict, Optional

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
    execution_ledger: Optional[pd.DataFrame] = None,
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

    ledger = execution_ledger
    if ledger is None and trades is not None:
        candidate = trades.attrs.get("daily_ledger")
        if isinstance(candidate, pd.DataFrame):
            ledger = candidate

    if ledger is None:
        # Keep the public gross-only behaviour when no trade list is supplied:
        # execute the target path without trading charges, but retain borrow and
        # the same mark-then-trade timing used by the main simulator.
        simulation_model = cost_model
        if trades is None:
            simulation_model = replace(
                cost_model,
                commission_per_share=0.0,
                commission_pct=0.0,
                commission_min=0.0,
                slippage_bps=0.0,
                impact_coeff=0.0,
                sec_fee_rate=0.0,
                finra_taf_per_share=0.0,
                stamp_duty_pct=0.0,
                max_participation_rate=None,
            )
        from quant_platform.core.execution.backtest.vectorized_daily import (
            simulate_execution,
        )

        _, ledger = simulate_execution(
            daily_weights=daily_weights,
            prices=prices,
            cost_model=simulation_model,
            initial_capital=initial_capital,
            return_daily_ledger=True,
        )

    result = ledger.copy()
    result.index = pd.to_datetime(result.index)
    result.index.name = "date"
    required = {
        "gross_ret",
        "net_ret",
        "trading_cost_dollar",
        "borrow_cost_dollar",
        "total_cost_dollar",
        "trading_cost_bps",
        "borrow_cost_bps",
        "nav_gross",
        "nav_net",
        "drawdown_gross",
        "drawdown_net",
    }
    if missing := required.difference(result.columns):
        raise ValueError(f"execution_ledger missing columns: {sorted(missing)}")

    logger.info(
        "PnL computed: {} days, gross={:.2%} cum, net={:.2%} cum, max_dd={:.2%}",
        len(result),
        result["nav_gross"].iloc[-1] / initial_capital - 1 if len(result) else 0.0,
        result["nav_net"].iloc[-1] / initial_capital - 1 if len(result) else 0.0,
        result["drawdown_net"].min() if len(result) else 0.0,
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

    rf_per_period = (1.0 + risk_free_rate) ** (1.0 / annualisation) - 1.0

    def _sharpe(series):
        excess = series.dropna() - rf_per_period
        volatility = excess.std(ddof=1)
        return (
            excess.mean() / volatility * np.sqrt(annualisation)
            if volatility > 0
            else np.nan
        )

    def _sortino(series):
        excess = series.dropna() - rf_per_period
        downside = np.sqrt(np.mean(np.minimum(excess, 0.0) ** 2))
        return (
            excess.mean() / downside * np.sqrt(annualisation)
            if downside > 0
            else np.nan
        )

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
