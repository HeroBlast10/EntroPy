"""Capacity and participation analytics for executed backtests."""

from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
import pandas as pd

from quant_platform.core.execution.cost_models.us_equity import CostModel, estimate_batch_costs


def capacity_analysis(
    trades: pd.DataFrame,
    daily_pnl: Optional[pd.DataFrame] = None,
    *,
    cost_model: Optional[CostModel] = None,
    capital_grid: Iterable[float] = (1_000_000, 5_000_000, 10_000_000, 50_000_000),
    annualisation: int = 252,
) -> dict[str, pd.DataFrame]:
    """Compute capacity summary and cost elasticity across capital levels."""
    if trades is None or trades.empty:
        return {
            "summary": pd.DataFrame(),
            "capital_curve": pd.DataFrame(),
        }
    if cost_model is None:
        cost_model = CostModel()

    tr = trades.copy()
    tr["date"] = pd.to_datetime(tr["date"])
    tr["shares"] = pd.to_numeric(tr["shares"], errors="coerce").abs().fillna(0.0)
    tr["capacity_shares"] = pd.to_numeric(
        tr.get("requested_shares", tr["shares"]), errors="coerce"
    ).abs().fillna(0.0)
    tr["adv_shares"] = pd.to_numeric(tr.get("adv_shares", pd.Series(np.nan, index=tr.index)), errors="coerce").replace(0, np.nan)
    tr["price"] = pd.to_numeric(tr.get("price", np.nan), errors="coerce")
    tr["notional"] = pd.to_numeric(tr.get("notional", tr.get("notional_trade", np.nan)), errors="coerce")
    tr["capacity_notional"] = pd.to_numeric(
        tr.get("notional_trade", tr["notional"]), errors="coerce"
    ).fillna(0.0)
    tr["portfolio_value_before_trade"] = pd.to_numeric(
        tr.get("portfolio_value_before_trade", np.nan),
        errors="coerce",
    )

    # Execution diagnostics describe fills; capacity describes desired orders.
    valid_adv = tr["adv_shares"].gt(0) & tr["adv_shares"].ge(cost_model.min_adv_shares) & np.isfinite(tr["adv_shares"])
    tr["adv_shares"] = tr["adv_shares"].where(valid_adv)
    tr["requested_participation"] = tr["capacity_shares"] / tr["adv_shares"]
    tr["executed_participation"] = tr["shares"] / tr["adv_shares"]
    tr["participation_rate"] = tr["executed_participation"]
    tr["adv_dollar"] = tr["adv_shares"] * tr["price"]
    tr["adv_notional_pct"] = (
        (tr["capacity_shares"] * tr["price"]) / tr["adv_dollar"].replace(0, np.nan)
    )
    base_capital = _base_capital(tr)

    # Sub-share rounding rejects are not executable order requests. Including
    # their zero-only retry days lets monthly rebalances fall outside p95 and
    # spuriously inflates capacity by orders of magnitude.
    all_active = tr.loc[tr["capacity_shares"].gt(0)].copy()
    # Demand is sampled at original rebalances. Even positive whole-share
    # retries must not turn a monthly strategy into a daily low-demand sample.
    active = (all_active.loc[all_active["is_rebalance_order"].eq(True)].copy()
              if "is_rebalance_order" in all_active else all_active)
    daily_requested = active.groupby("date")["requested_participation"].max()
    p95_requested = float(daily_requested.quantile(0.95))
    daily_participation = active.groupby("date")["executed_participation"].max()
    p95_participation = float(daily_participation.quantile(0.95))
    # Normalise out NAV growth: reported capacity is initial capital at which
    # 95% of daily worst requested orders fit inside 10% ADV.
    nav_scale = base_capital / active["portfolio_value_before_trade"].where(active["portfolio_value_before_trade"].gt(0)).fillna(base_capital)
    normalized_daily = (active["requested_participation"] * nav_scale).groupby(active["date"]).max()
    all_nav_scale = base_capital / all_active["portfolio_value_before_trade"].where(all_active["portfolio_value_before_trade"].gt(0)).fillna(base_capital)
    normalized_all = (all_active["requested_participation"] * all_nav_scale).groupby(all_active["date"]).max()
    normalized_p95 = normalized_daily.quantile(0.95)
    capacity_10pct_adv = (
        base_capital * 0.10 / normalized_p95
        if pd.notna(normalized_p95) and normalized_p95 > 0
        else np.nan
    )

    summary = pd.DataFrame([{
        "base_capital": base_capital,
        "avg_participation_rate": float(tr["participation_rate"].mean()),
        "p95_participation_rate": p95_participation,
        "max_participation_rate": float(tr["participation_rate"].max()),
        "avg_requested_participation": float(tr["requested_participation"].mean()),
        "p95_requested_participation": p95_requested,
        "max_requested_participation": float(tr["requested_participation"].max()),
        "p95_executed_participation": p95_participation,
        "max_executed_participation": float(tr["executed_participation"].max()),
        "p95_aggregation": "95th percentile of daily maximum",
        "p95_sample": ("original rebalance requests only; residual retries excluded"
                       if "is_rebalance_order" in tr else "positive whole-share requests; rebalance provenance unavailable"),
        "p95_sample_days": int(active["date"].nunique()),
        "capacity_basis": "requested orders, normalized to initial capital; not a full strategy replay",
        "capacity_at_max_10pct_adv": float(base_capital * 0.10 / normalized_all.max()) if normalized_all.max() > 0 else np.nan,
        "capacity_liquidity_coverage": float(tr.loc[tr["capacity_notional"].gt(0), "adv_shares"].notna().mean()),
        "avg_adv_notional_pct": float(tr["adv_notional_pct"].mean()),
        "p95_adv_notional_pct": float(active.groupby("date")["adv_notional_pct"].max().quantile(0.95)),
        "max_adv_notional_pct": float(tr["adv_notional_pct"].max()),
        "capacity_at_10pct_adv": float(capacity_10pct_adv) if pd.notna(capacity_10pct_adv) else np.nan,
        "total_traded_notional": float(tr["notional"].sum()),
        "total_requested_notional": float(tr["capacity_notional"].sum()),
        "total_cost": float(pd.to_numeric(tr.get("total_cost", pd.Series(0.0, index=tr.index)), errors="coerce").fillna(0.0).sum()),
    }])

    curve = _capital_curve(
        tr,
        daily_pnl,
        base_capital=base_capital,
        capital_grid=capital_grid,
        cost_model=cost_model,
        annualisation=annualisation,
    )
    return {"summary": summary, "capital_curve": curve}


def _capital_curve(
    trades: pd.DataFrame,
    daily_pnl: Optional[pd.DataFrame],
    *,
    base_capital: float,
    capital_grid: Iterable[float],
    cost_model: CostModel,
    annualisation: int,
) -> pd.DataFrame:
    rows = []
    for capital in capital_grid:
        scale = float(capital) / max(base_capital, 1.0)
        scaled = trades.copy()
        scaled["requested_shares"] = np.floor(scaled["capacity_shares"] * scale)
        scaled["shares"] = scaled["requested_shares"]
        eligible = scaled["price"].gt(0) & scaled["adv_shares"].ge(cost_model.min_adv_shares) & scaled["adv_shares"].gt(0)
        if "fill_status" in scaled:
            rejected = scaled["fill_status"].isin(["rejected_not_tradable", "rejected_missing_price"])
            eligible &= ~rejected
            scaled.loc[rejected, "shares"] = 0.0
        if cost_model.max_participation_rate is not None:
            # Respect the ticker/day budget even if input has several orders.
            requested_daily = scaled.groupby(["date", "ticker"])["requested_shares"].transform("sum")
            allowance = scaled["adv_shares"] * cost_model.max_participation_rate
            ratio = (allowance / requested_daily.replace(0, np.nan)).clip(upper=1).fillna(0)
            scaled["shares"] = np.floor(scaled["shares"] * ratio)
            scaled.loc[~eligible, "shares"] = 0.0
        scaled["notional_trade"] = scaled["shares"] * scaled["price"].fillna(0)
        executed_participation = scaled["shares"] / scaled["adv_shares"]
        # A different fill path changes holdings and PnL. Reusing gross_ret
        # would be misleading; keep cost estimates, suppress return estimates.
        same_fill_path = bool(np.allclose(scaled["shares"], trades["shares"] * scale, atol=1e-8, rtol=0))
        scaled["daily_vol"] = pd.to_numeric(
            scaled.get("daily_vol", pd.Series(0.02, index=scaled.index)), errors="coerce"
        ).fillna(0.02)
        estimable = (
            scaled["shares"].gt(0)
            & scaled["price"].gt(0)
            & scaled["adv_shares"].gt(0)
        )
        if estimable.any():
            costed = estimate_batch_costs(
                scaled.loc[
                    estimable,
                    [
                        "date",
                        "ticker",
                        "side",
                        "shares",
                        "price",
                        "adv_shares",
                        "daily_vol",
                        "notional_trade",
                    ],
                ].copy(),
                cost_model,
            )
        else:
            costed = pd.DataFrame(columns=["date", "total_cost", "notional"])
        total_cost = float(costed["total_cost"].sum())
        total_notional = float(costed["notional"].sum())
        daily_cost = costed.groupby("date")["total_cost"].sum()
        net_sharpe = np.nan
        net_ann_return = np.nan
        avg_daily_cost_bps = np.nan
        if same_fill_path and daily_pnl is not None and not daily_pnl.empty and "gross_ret" in daily_pnl.columns:
            pnl = daily_pnl.copy()
            if not isinstance(pnl.index, pd.DatetimeIndex):
                pnl["date"] = pd.to_datetime(pnl["date"])
                pnl = pnl.set_index("date")
            daily_cost_aligned = daily_cost.reindex(pnl.index, fill_value=0.0)
            borrow_bps = pd.to_numeric(
                pnl.get("borrow_cost_bps", 0.0), errors="coerce"
            )
            if not isinstance(borrow_bps, pd.Series):
                borrow_bps = pd.Series(float(borrow_bps), index=pnl.index)
            borrow_frac = borrow_bps.reindex(pnl.index, fill_value=0.0) / 10_000
            nav = float(capital)
            net_values = []
            cost_fracs = []
            total_borrow_cost = 0.0
            for dt, gross_ret in pnl["gross_ret"].fillna(0.0).items():
                prev_nav = nav
                borrow_cost = prev_nav * float(borrow_frac.loc[dt])
                trading_cost = float(daily_cost_aligned.loc[dt])
                nav = prev_nav * (1.0 + float(gross_ret)) - borrow_cost - trading_cost
                net_values.append(nav / prev_nav - 1.0)
                cost_fracs.append((borrow_cost + trading_cost) / prev_nav)
                total_borrow_cost += borrow_cost
            net = pd.Series(net_values, index=pnl.index)
            net_sharpe = _sharpe(net, annualisation)
            net_ann_return = _ann_return(net, annualisation)
            avg_daily_cost_bps = float(np.mean(cost_fracs) * 10_000)
        else:
            total_borrow_cost = np.nan

        rows.append({
            "capital": float(capital),
            "scale_vs_base": scale,
            "total_cost": total_cost,
            "total_borrow_cost": total_borrow_cost,
            "total_notional": total_notional,
            "requested_notional": float((scaled["requested_shares"] * scaled["price"].fillna(0)).sum()),
            "max_requested_participation": float((scaled["requested_shares"] / scaled["adv_shares"]).max()),
            "max_executed_participation": float(executed_participation.max()),
            "fill_ratio": float(scaled["shares"].sum() / max(scaled["requested_shares"].sum(), 1)),
            "return_estimate_valid": same_fill_path,
            "return_estimate_note": "same scaled fills as base" if same_fill_path else "fill path changed; full execution replay required",
            "total_cost_bps": total_cost / max(total_notional, 1.0) * 10_000,
            "avg_daily_cost_bps": avg_daily_cost_bps,
            "net_ann_return_est": net_ann_return,
            "net_sharpe_est": net_sharpe,
        })
    return pd.DataFrame(rows)


def _base_capital(trades: pd.DataFrame) -> float:
    vals = trades["portfolio_value_before_trade"].replace([np.inf, -np.inf], np.nan).dropna()
    if not vals.empty and vals.iloc[0] > 0:
        return float(vals.iloc[0])
    notional = pd.to_numeric(trades.get("notional", 0.0), errors="coerce").fillna(0.0)
    delta = pd.to_numeric(trades.get("delta_weight", pd.Series(np.nan, index=trades.index)), errors="coerce").abs().replace(0, np.nan)
    implied = (notional / delta).replace([np.inf, -np.inf], np.nan).dropna()
    return float(implied.median()) if not implied.empty else 1_000_000.0


def _ann_return(series: pd.Series, annualisation: int) -> float:
    ret = series.dropna()
    if ret.empty:
        return np.nan
    years = len(ret) / annualisation
    return float((1.0 + ret).prod() ** (1.0 / max(years, 1e-9)) - 1.0)


def _sharpe(series: pd.Series, annualisation: int) -> float:
    ret = series.dropna()
    if len(ret) < 2 or ret.std() <= 0:
        return np.nan
    return float(ret.mean() / ret.std() * np.sqrt(annualisation))
