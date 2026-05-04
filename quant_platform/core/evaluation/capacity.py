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
    tr["adv_shares"] = pd.to_numeric(tr.get("adv_shares", np.nan), errors="coerce").replace(0, np.nan)
    tr["price"] = pd.to_numeric(tr.get("price", np.nan), errors="coerce")
    tr["notional"] = pd.to_numeric(tr.get("notional", tr.get("notional_trade", np.nan)), errors="coerce")
    tr["portfolio_value_before_trade"] = pd.to_numeric(
        tr.get("portfolio_value_before_trade", np.nan),
        errors="coerce",
    )

    tr["participation_rate"] = tr["shares"] / tr["adv_shares"]
    tr["adv_dollar"] = tr["adv_shares"] * tr["price"]
    tr["adv_notional_pct"] = tr["notional"] / tr["adv_dollar"].replace(0, np.nan)
    base_capital = _base_capital(tr)

    daily_participation = tr.groupby("date")["participation_rate"].max()
    p95_participation = float(daily_participation.quantile(0.95)) if len(daily_participation) else np.nan
    capacity_10pct_adv = (
        base_capital * 0.10 / p95_participation
        if pd.notna(p95_participation) and p95_participation > 0
        else np.nan
    )

    summary = pd.DataFrame([{
        "base_capital": base_capital,
        "avg_participation_rate": float(tr["participation_rate"].mean()),
        "p95_participation_rate": p95_participation,
        "max_participation_rate": float(tr["participation_rate"].max()),
        "avg_adv_notional_pct": float(tr["adv_notional_pct"].mean()),
        "p95_adv_notional_pct": float(tr["adv_notional_pct"].quantile(0.95)),
        "max_adv_notional_pct": float(tr["adv_notional_pct"].max()),
        "capacity_at_10pct_adv": float(capacity_10pct_adv) if pd.notna(capacity_10pct_adv) else np.nan,
        "total_traded_notional": float(tr["notional"].sum()),
        "total_cost": float(pd.to_numeric(tr.get("total_cost", 0.0), errors="coerce").fillna(0.0).sum()),
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
        scaled["shares"] = scaled["shares"] * scale
        scaled["notional_trade"] = pd.to_numeric(
            scaled.get("notional_trade", scaled.get("notional", 0.0)),
            errors="coerce",
        ).fillna(0.0) * scale
        costed = estimate_batch_costs(
            scaled[["date", "ticker", "side", "shares", "price", "adv_shares", "daily_vol", "notional_trade"]].copy(),
            cost_model,
        )
        total_cost = float(costed["total_cost"].sum())
        total_notional = float(costed["notional"].sum())
        daily_cost = costed.groupby("date")["total_cost"].sum()
        net_sharpe = np.nan
        net_ann_return = np.nan
        avg_daily_cost_bps = np.nan
        if daily_pnl is not None and not daily_pnl.empty and "gross_ret" in daily_pnl.columns:
            pnl = daily_pnl.copy()
            if not isinstance(pnl.index, pd.DatetimeIndex):
                pnl["date"] = pd.to_datetime(pnl["date"])
                pnl = pnl.set_index("date")
            cost_frac = daily_cost.reindex(pnl.index, fill_value=0.0) / max(float(capital), 1.0)
            net = pnl["gross_ret"].fillna(0.0) - cost_frac
            net_sharpe = _sharpe(net, annualisation)
            net_ann_return = _ann_return(net, annualisation)
            avg_daily_cost_bps = float(cost_frac.mean() * 10_000)

        rows.append({
            "capital": float(capital),
            "scale_vs_base": scale,
            "total_cost": total_cost,
            "total_notional": total_notional,
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
    delta = pd.to_numeric(trades.get("delta_weight", np.nan), errors="coerce").abs().replace(0, np.nan)
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

