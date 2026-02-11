"""Trade generation and fill simulation.

Converts daily weight changes into a concrete trade list with:
- Side (buy / sell)
- Share quantities (from notional + price)
- Fill prices (decision price ± slippage)
- ADV and volatility for cost estimation

The execution model is intentionally simple (close-to-close) but
realistic enough to demonstrate awareness of market microstructure.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

from entropy.trading.costs import CostModel, estimate_batch_costs


# ===================================================================
# Weight-diff → trade list
# ===================================================================

def generate_trades(
    weights_today: pd.Series,
    weights_yesterday: Optional[pd.Series],
    prices_today: pd.Series,
    adv_today: pd.Series,
    vol_today: pd.Series,
    capital: float,
) -> pd.DataFrame:
    """Convert weight changes into a trade list for one date.

    Parameters
    ----------
    weights_today : target weights indexed by ticker.
    weights_yesterday : previous weights (``None`` on the first day).
    prices_today : close prices indexed by ticker.
    adv_today : average daily volume (shares) indexed by ticker.
    vol_today : annualised daily volatility indexed by ticker.
    capital : current portfolio value in dollars.

    Returns
    -------
    DataFrame with columns:
        ``ticker, side, weight_from, weight_to, delta_weight,
        notional_trade, shares, price, adv_shares, daily_vol``
    """
    if weights_yesterday is None:
        weights_yesterday = pd.Series(dtype=float)

    # Align to common ticker set
    all_tickers = weights_today.index.union(weights_yesterday.index)
    w_new = weights_today.reindex(all_tickers, fill_value=0.0)
    w_old = weights_yesterday.reindex(all_tickers, fill_value=0.0)
    delta = w_new - w_old

    # Drop zero changes
    delta = delta[delta.abs() > 1e-10]
    if delta.empty:
        return pd.DataFrame(columns=[
            "ticker", "side", "weight_from", "weight_to", "delta_weight",
            "notional_trade", "shares", "price", "adv_shares", "daily_vol",
        ])

    trades = pd.DataFrame({
        "ticker": delta.index,
        "weight_from": w_old.reindex(delta.index, fill_value=0.0).values,
        "weight_to": w_new.reindex(delta.index, fill_value=0.0).values,
        "delta_weight": delta.values,
    })

    trades["side"] = np.where(trades["delta_weight"] > 0, "buy", "sell")
    trades["notional_trade"] = (trades["delta_weight"].abs() * capital)
    trades["price"] = prices_today.reindex(trades["ticker"], fill_value=np.nan).values
    trades["shares"] = np.where(
        trades["price"] > 0,
        (trades["notional_trade"] / trades["price"]).round(0),
        0.0,
    )
    trades["adv_shares"] = adv_today.reindex(trades["ticker"], fill_value=1e6).values
    trades["daily_vol"] = vol_today.reindex(trades["ticker"], fill_value=0.02).values

    return trades.reset_index(drop=True)


# ===================================================================
# Full execution simulation across all dates
# ===================================================================

def simulate_execution(
    daily_weights: pd.DataFrame,
    prices: pd.DataFrame,
    cost_model: Optional[CostModel] = None,
    initial_capital: float = 1_000_000.0,
    adv_lookback: int = 20,
    vol_lookback: int = 20,
) -> pd.DataFrame:
    """Simulate trade execution across the entire backtest period.

    Parameters
    ----------
    daily_weights : ``[date, ticker, weight]`` — one row per holding per day.
    prices : ``[date, ticker, close, adj_close, volume]`` from the data layer.
    cost_model : transaction cost parameters.
    initial_capital : starting portfolio value.
    adv_lookback : days for ADV computation.
    vol_lookback : days for volatility computation.

    Returns
    -------
    DataFrame of all trades with cost breakdowns, one row per trade.
    Columns include everything from :func:`generate_trades` plus
    cost columns from :func:`estimate_batch_costs`.
    """
    if cost_model is None:
        cost_model = CostModel()

    daily_weights = daily_weights.copy()
    prices = prices.copy()
    daily_weights["date"] = pd.to_datetime(daily_weights["date"])
    prices["date"] = pd.to_datetime(prices["date"])

    # Pre-compute ADV and rolling vol per (ticker, date)
    prices_sorted = prices.sort_values(["ticker", "date"])

    adv = (
        prices_sorted.groupby("ticker")["volume"]
        .transform(lambda s: s.rolling(adv_lookback, min_periods=5).mean())
    )
    prices_sorted["adv_shares"] = adv

    ret = prices_sorted.groupby("ticker")["adj_close"].transform(lambda s: s.pct_change())
    daily_vol = ret.groupby(prices_sorted["ticker"]).transform(
        lambda s: s.rolling(vol_lookback, min_periods=5).std()
    )
    prices_sorted["daily_vol"] = daily_vol

    # Build lookup: (date, ticker) → price / adv / vol
    px_lookup = prices_sorted.set_index(["date", "ticker"])

    dates = sorted(daily_weights["date"].unique())
    all_trades: list[pd.DataFrame] = []
    prev_weights: Optional[pd.Series] = None
    capital = initial_capital

    for dt in dates:
        w_today = daily_weights.loc[daily_weights["date"] == dt].set_index("ticker")["weight"]

        # Get prices for today
        if dt not in px_lookup.index.get_level_values(0):
            prev_weights = w_today
            continue

        px_today = px_lookup.loc[dt]

        price_series = px_today["close"] if "close" in px_today.columns else pd.Series(dtype=float)
        adv_series = px_today["adv_shares"] if "adv_shares" in px_today.columns else pd.Series(dtype=float)
        vol_series = px_today["daily_vol"] if "daily_vol" in px_today.columns else pd.Series(dtype=float)

        trades = generate_trades(
            weights_today=w_today,
            weights_yesterday=prev_weights,
            prices_today=price_series,
            adv_today=adv_series,
            vol_today=vol_series,
            capital=capital,
        )

        if not trades.empty:
            trades_with_costs = estimate_batch_costs(trades, cost_model)
            trades_with_costs.insert(0, "date", dt)
            all_trades.append(trades_with_costs)

            # Update capital by subtracting trading costs
            total_cost = trades_with_costs["total_cost"].sum()
            capital -= total_cost

        prev_weights = w_today

    if not all_trades:
        logger.warning("No trades generated across the backtest period")
        return pd.DataFrame()

    result = pd.concat(all_trades, ignore_index=True)
    logger.info(
        "Execution: {} trades across {} days, total cost = ${:,.0f} ({:.1f} bps avg)",
        len(result),
        result["date"].nunique(),
        result["total_cost"].sum(),
        result["total_cost_bps"].mean(),
    )
    return result
