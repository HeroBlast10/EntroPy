"""Auditable turnover and fixed-fill-path cost diagnostics.

Source buckets describe observable instructions. The boundary bucket is a
rank-band proxy; it cannot establish that a rank move was noise. Causal
effects must be measured by separate, matched execution replays.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from quant_platform.core.execution.backtest.vectorized_daily import _attach_order_sources
from quant_platform.core.portfolio.low_turnover import transition_sources

SOURCES = ("signal", "boundary", "drift_resize", "lot_residual")
FEES = ("commission", "slippage", "impact", "sec_fee", "finra_taf", "stamp_duty")


def annotate_recorded_orders(trades, targets, scores, score):
    """Recover source fractions from old artifacts without rerunning fills."""
    ranks = scores[["date", "ticker"]].copy()
    ranks["rank"] = scores.groupby("date")[score].rank(pct=True)
    rank_groups = {date: day.set_index("ticker")["rank"] for date, day in ranks.groupby("date")}
    target_groups = {date: day.set_index("ticker").weight for date, day in targets.groupby("date")}
    previous = pd.Series(dtype=float)
    previous_rank = pd.Series(dtype=float)
    frames = []
    # Iterate target dates too: a rebalance with no orders still supersedes a
    # prior instruction and updates the reference used by the next trade.
    dates = sorted(set(target_groups) | set(pd.to_datetime(trades.date)))
    order_groups = {date: day for date, day in trades.groupby("date")}
    for date in dates:
        day = order_groups.get(date)
        current = target_groups.get(date)
        if current is not None:
            idx = current.index.union(previous.index)
            rank = rank_groups.get(date, pd.Series(dtype=float)).reindex(idx)
            source = transition_sources(previous.reindex(idx, fill_value=0),
                                        current.reindex(idx, fill_value=0),
                                        previous_rank.reindex(idx), rank)
        else:
            source = pd.Series(dtype=object)
        if day is not None:
            rebalance = bool(day.is_rebalance_order.any())
            frames.append(_attach_order_sources(day.copy(), previous, source, rebalance, 1))
        if current is not None:
            previous, previous_rank = current, rank
    return pd.concat(frames, ignore_index=True) if frames else trades.copy()


def break_even_trade_bps(gross, borrow_rate, traded_fraction):
    """Uniform bps per buy/sell dollar giving terminal wealth 1, borrow fixed.

    This replaces trading fees on the recorded gross/fill path. It is not a
    capacity claim or a counterfactual execution replay. Negative values mean
    even zero trading fees would require a subsidy to break even.
    """
    base = 1 + np.asarray(gross, dtype=float) - np.asarray(borrow_rate, dtype=float)
    intensity = np.asarray(traded_fraction, dtype=float) / 10_000
    if (base <= 0).any() or not np.isfinite(base).all() or not np.isfinite(intensity).all():
        raise ValueError("Invalid fixed-path returns or turnover")
    if (intensity < 0).any():
        raise ValueError("Turnover cannot be negative")
    active = intensity > 0
    if not active.any():
        return np.nan
    upper = float(np.min(base[active] / intensity[active])) * (1 - 1e-12)

    def log_wealth(bps):
        return float(np.log(base - bps * intensity).sum())

    lower = -1.0
    while log_wealth(lower) < 0:
        lower *= 2
    return float(brentq(log_wealth, lower, upper, xtol=1e-10))


def turnover_cost_diagnostics(trades, ledger, initial_capital=1_000_000.0, commission_min=1.0):
    """Daily metrics, reconciled source/fee tables and per-strategy summary.

    One-way turnover = 0.5 * (buy + sell notional) / pre-trade NAV. Unit cost
    and break-even bps use *buy + sell* dollars, avoiding a factor-of-two error.
    Cash-only dates remain in all daily means; opening trades remain included.
    """
    p = ledger.sort_index()
    prior_nav = p.nav_net.shift().fillna(initial_capital)
    t = trades.copy()
    if len(t) and not all(f"source_{s}_fraction" in t for s in SOURCES):
        raise ValueError("Missing instruction-source attribution")
    daily = pd.DataFrame(index=p.index)
    daily["gross_ret"], daily["net_ret"] = p.gross_ret, p.net_ret
    for field in ("executed_notional", "total_cost", *FEES):
        daily[field] = t.groupby("date")[field].sum().reindex(p.index, fill_value=0) if len(t) else 0.0
    np.testing.assert_allclose(daily.total_cost, p.trading_cost_dollar, atol=1e-7)
    daily["borrow_cost"] = p.borrow_cost_dollar
    daily["total_cost_including_borrow"] = daily.total_cost + daily.borrow_cost
    daily["one_way_turnover"] = p.executed_turnover / 2
    daily["traded_fraction_prior_nav"] = daily.executed_notional / prior_nav
    daily["unit_trading_cost_bps"] = daily.total_cost.div(daily.executed_notional.replace(0, np.nan)) * 10_000
    daily["trading_cost_nav_bps"] = daily.total_cost / prior_nav * 10_000
    daily["borrow_cost_nav_bps"] = daily.borrow_cost / prior_nav * 10_000
    source_rows = []
    source_total = pd.Series(0.0, index=p.index)
    if len(t):
        fractions = t[[f"source_{s}_fraction" for s in SOURCES]]
        np.testing.assert_allclose(fractions.sum(axis=1), 1.0, atol=1e-12)
        if (fractions.to_numpy() < 0).any():
            raise ValueError("Negative source allocation")
    for source in SOURCES:
        fraction = t[f"source_{source}_fraction"] if len(t) else pd.Series(dtype=float)
        costs = (t.total_cost * fraction).groupby(t.date).sum() if len(t) else pd.Series(dtype=float)
        daily[f"{source}_cost"] = costs.reindex(p.index, fill_value=0)
        source_total += daily[f"{source}_cost"]
        row = {"source": source,
               "total_cost_dollar": float(daily[f"{source}_cost"].sum()),
               "annual_cost_nav_bps": float((daily[f"{source}_cost"] / prior_nav).mean() * 252 * 10_000),
               "executed_notional": float((t.executed_notional * fraction).sum()) if len(t) else 0.0}
        for fee in FEES:
            row[fee] = float((t[fee] * fraction).sum()) if len(t) else 0.0
        source_rows.append(row)
    np.testing.assert_allclose(source_total, p.trading_cost_dollar, atol=1e-7)
    source_rows.append({"source": "borrow", "total_cost_dollar": float(daily.borrow_cost.sum()),
                        "annual_cost_nav_bps": float(daily.borrow_cost_nav_bps.mean() * 252),
                        "executed_notional": 0.0, **{fee: 0.0 for fee in FEES}})
    notional = daily.executed_notional.sum()
    filled = t.shares.gt(0) if len(t) else pd.Series(dtype=bool)
    summary = {
        "days": len(p),
        "daily_one_way_turnover": float(daily.one_way_turnover.mean()),
        "annual_one_way_turnover": float(daily.one_way_turnover.mean() * 252),
        "unit_trading_cost_bps": float(daily.total_cost.sum() / notional * 10_000) if notional else np.nan,
        "unit_total_cost_bps": float(daily.total_cost_including_borrow.sum() / notional * 10_000) if notional else np.nan,
        "break_even_trade_bps": break_even_trade_bps(p.gross_ret, daily.borrow_cost / prior_nav,
                                                     daily.traded_fraction_prior_nav),
        "annual_trading_drag_bps": float(daily.trading_cost_nav_bps.mean() * 252),
        "annual_borrow_drag_bps": float(daily.borrow_cost_nav_bps.mean() * 252),
        "executed_orders": int(filled.sum()),
        "median_fill_dollar": float(t.loc[filled, "executed_notional"].median()) if filled.any() else np.nan,
        "rounding_rejects": int(t.fill_status.eq("rejected_rounding").sum()) if len(t) else 0,
        "suppressed_candidates": int(t.fill_status.eq("suppressed_tolerance").sum()) if len(t) else 0,
        "mean_gross_exposure": float(p.gross_exposure.mean()),
        "max_gross_exposure": float(p.gross_exposure.max()),
        "mean_absolute_net_exposure": float(p.net_exposure.abs().mean()),
        "max_absolute_net_exposure": float(p.net_exposure.abs().max()),
        "minimum_commission_fraction": float(t.loc[t.commission.eq(commission_min), "commission"].sum() / t.commission.sum())
            if len(t) and t.commission.sum() else np.nan,
    }
    return daily, pd.DataFrame(source_rows), summary
