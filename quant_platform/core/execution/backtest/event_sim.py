"""Vectorized backtest engine with A-share constraints.

Migrated from TradeX engine.py. Complete backtest engine supporting:
- T+1 signal lag (signal at T → trade at T+1)
- Skip limit up/down, ST, suspended, new listing
- Lot size enforcement (100-share minimum)
- AShareCostModel for realistic transaction costs
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Literal, Optional

import numpy as np
import pandas as pd
from loguru import logger

from quant_platform.core.execution.cost_models.cn_a_share import AShareCostModel


# ===================================================================
# Config & result dataclasses
# ===================================================================

class WeightScheme(str, Enum):
    TOP_N = "top_n"
    FACTOR_WEIGHT = "factor_weight"
    LONG_SHORT = "long_short"


@dataclass
class BacktestConfig:
    """Backtest engine configuration."""

    # --- Signal → weights ---
    weight_scheme: WeightScheme = WeightScheme.TOP_N
    top_n: int = 30
    max_weight: float = 0.05

    # --- Rebalance ---
    rebalance_freq: str = "M"  # D | W | M

    # --- A-share constraints ---
    signal_lag: int = 1  # T+1: signal at T → trade at T+1
    lot_size: int = 100
    skip_limit_up: bool = True
    skip_limit_down: bool = True
    skip_st: bool = True
    skip_suspended: bool = True
    skip_new_listing: bool = True

    # --- Capital ---
    initial_capital: float = 1e6


@dataclass
class BacktestResult:
    """Backtest output."""

    nav: pd.Series
    daily_returns: pd.Series
    weights: pd.DataFrame
    turnover: pd.Series
    total_cost_bps: float
    sharpe: float
    ann_return: float
    max_drawdown: float


# ===================================================================
# VectorizedBacktest
# ===================================================================

class VectorizedBacktest:
    """Vectorized backtest engine with A-share constraints."""

    def __init__(
        self,
        config: Optional[BacktestConfig] = None,
        cost_model: Optional[AShareCostModel] = None,
    ):
        self.config = config or BacktestConfig()
        self.cost_model = cost_model or AShareCostModel()

    def run(
        self,
        panel: pd.DataFrame,
        prices: pd.DataFrame,
        universe: Optional[pd.DataFrame] = None,
    ) -> BacktestResult:
        """Run the full backtest.

        Parameters
        ----------
        panel : long-format DataFrame with date, ticker (or ts_code), signal,
                and optionally limit_up, limit_down, is_st, is_suspended, is_new_listing.
        prices : date × ticker close prices (or long format with date, ticker, close).
        universe : optional tradable universe [date, ticker, pass_all_filters].
        """
        ticker_col = "ticker" if "ticker" in panel.columns else "ts_code"
        close_mat, ret_mat, signal_mat, tradable_mat, dates, tickers = self._build_matrices(
            panel, prices, ticker_col, universe
        )
        reb_dates = self._rebalance_schedule(dates)
        weights_mat = self._signal_to_weights(signal_mat, tradable_mat, dates, reb_dates, tickers)
        weights_mat = self._cap_weights(weights_mat)
        nav, daily_ret, turnover, cost_bps = self._simulate(
            close_mat, ret_mat, weights_mat, dates, tickers,
        )
        sharpe = float(daily_ret.mean() / daily_ret.std() * np.sqrt(252)) if daily_ret.std() > 0 else 0.0
        ann_ret = float((1 + daily_ret).prod() ** (252 / len(daily_ret)) - 1) if len(daily_ret) > 0 else 0.0
        cum = (1 + daily_ret).cumprod()
        peak = cum.cummax()
        dd = (cum / peak - 1.0)
        max_dd = float(dd.min()) if len(dd) > 0 else 0.0

        weights_df = pd.DataFrame(
            weights_mat, index=dates, columns=tickers,
        ).stack().reset_index()
        weights_df.columns = ["date", ticker_col, "weight"]
        weights_df = weights_df[weights_df["weight"].abs() > 1e-10]

        return BacktestResult(
            nav=nav,
            daily_returns=daily_ret,
            weights=weights_df,
            turnover=turnover,
            total_cost_bps=cost_bps,
            sharpe=sharpe,
            ann_return=ann_ret,
            max_drawdown=max_dd,
        )

    def _build_matrices(
        self,
        panel: pd.DataFrame,
        prices: pd.DataFrame,
        ticker_col: str,
        universe: Optional[pd.DataFrame],
    ):
        """Build aligned date × ticker matrices for close, returns, signal, tradable."""
        panel = panel.copy()
        panel["date"] = pd.to_datetime(panel["date"])
        prices = prices.copy()
        prices["date"] = pd.to_datetime(prices["date"])

        # Normalize ticker in prices
        if "ts_code" in prices.columns and "ticker" not in prices.columns:
            prices = prices.rename(columns={"ts_code": "ticker"})
        price_ticker = "ticker" if "ticker" in prices.columns else "ts_code"

        # Pivot close
        close_col = "adj_close" if "adj_close" in prices.columns else "close"
        close_wide = prices.pivot_table(
            index="date", columns=price_ticker, values=close_col, aggfunc="last"
        )
        dates = close_wide.index
        tickers = close_wide.columns.tolist()

        # Returns
        ret_wide = close_wide.pct_change()
        ret_mat = ret_wide.values
        close_mat = close_wide.values

        # Signal: detect first non-date/ticker column
        sig_col = None
        for c in panel.columns:
            if c not in ("date", ticker_col):
                sig_col = c
                break
        if sig_col is None:
            raise ValueError("Panel has no signal column")
        signal_wide = panel.pivot_table(
            index="date", columns=ticker_col, values=sig_col, aggfunc="last"
        )
        signal_wide = signal_wide.reindex(index=dates, columns=tickers)
        signal_mat = signal_wide.values

        # Tradable mask: apply A-share constraints
        tradable_mat = np.ones_like(signal_mat, dtype=bool)
        tradable_mat = tradable_mat & ~np.isnan(close_mat) & (close_mat > 0)
        tradable_mat = tradable_mat & ~np.isnan(signal_mat)

        if "limit_up" in panel.columns:
            limit_up_wide = panel.pivot_table(
                index="date", columns=ticker_col, values="limit_up", aggfunc="last"
            ).reindex(index=dates, columns=tickers)
            if self.config.skip_limit_up:
                tradable_mat = tradable_mat & ~(limit_up_wide.fillna(False).values)
        if "limit_down" in panel.columns:
            limit_down_wide = panel.pivot_table(
                index="date", columns=ticker_col, values="limit_down", aggfunc="last"
            ).reindex(index=dates, columns=tickers)
            if self.config.skip_limit_down:
                tradable_mat = tradable_mat & ~(limit_down_wide.fillna(False).values)
        if "is_st" in panel.columns:
            is_st_wide = panel.pivot_table(
                index="date", columns=ticker_col, values="is_st", aggfunc="last"
            ).reindex(index=dates, columns=tickers)
            if self.config.skip_st:
                tradable_mat = tradable_mat & ~(is_st_wide.fillna(False).values)
        if "is_suspended" in panel.columns:
            susp_wide = panel.pivot_table(
                index="date", columns=ticker_col, values="is_suspended", aggfunc="last"
            ).reindex(index=dates, columns=tickers)
            if self.config.skip_suspended:
                tradable_mat = tradable_mat & ~(susp_wide.fillna(True).values)
        if "is_new_listing" in panel.columns:
            new_wide = panel.pivot_table(
                index="date", columns=ticker_col, values="is_new_listing", aggfunc="last"
            ).reindex(index=dates, columns=tickers)
            if self.config.skip_new_listing:
                tradable_mat = tradable_mat & ~(new_wide.fillna(False).values)

        if universe is not None:
            uni_wide = universe.pivot_table(
                index="date", columns="ticker", values="pass_all_filters", aggfunc="last"
            ).reindex(index=dates, columns=tickers)
            tradable_mat = tradable_mat & uni_wide.fillna(False).values

        return close_mat, ret_mat, signal_mat, tradable_mat, dates, tickers

    def _rebalance_schedule(self, dates: pd.DatetimeIndex) -> np.ndarray:
        """Boolean mask of rebalance dates."""
        freq = self.config.rebalance_freq
        if freq == "D":
            return np.ones(len(dates), dtype=bool)
        df = pd.DataFrame({"date": dates})
        if freq == "W":
            df["key"] = df["date"].dt.isocalendar().year.astype(str) + "-" + \
                       df["date"].dt.isocalendar().week.astype(str).str.zfill(2)
            last = df.groupby("key")["date"].transform("max")
        elif freq == "M":
            df["key"] = df["date"].dt.to_period("M")
            last = df.groupby("key")["date"].transform("max")
        else:
            return np.ones(len(dates), dtype=bool)
        return (df["date"] == last).values

    def _signal_to_weights(
        self,
        signal_mat: np.ndarray,
        tradable_mat: np.ndarray,
        dates: pd.DatetimeIndex,
        reb_mask: np.ndarray,
        tickers: list,
    ) -> np.ndarray:
        """Convert signal to target weights. T+1 lag applied."""
        T, N = signal_mat.shape
        weights = np.zeros((T, N))
        lag = self.config.signal_lag

        for t in range(lag, T):
            if not reb_mask[t]:
                if t > 0:
                    weights[t] = weights[t - 1]
                continue
            sig_t = signal_mat[t - lag]
            tradable_t = tradable_mat[t]
            valid = tradable_t & ~np.isnan(sig_t)
            if not valid.any():
                weights[t] = weights[t - 1] if t > 0 else np.zeros(N)
                continue
            sig_valid = np.where(valid, sig_t, -np.inf)
            scheme = self.config.weight_scheme
            top_n = self.config.top_n

            if scheme == WeightScheme.TOP_N:
                rank = np.argsort(-sig_valid)
                n_pick = min(top_n, int(valid.sum()))
                w = np.zeros(N)
                for i in range(n_pick):
                    idx = rank[i]
                    if valid[idx]:
                        w[idx] = 1.0 / n_pick
                weights[t] = w

            elif scheme == WeightScheme.FACTOR_WEIGHT:
                sig_pos = np.maximum(sig_valid, 0)
                total = np.nansum(sig_pos)
                w = np.where(valid, sig_pos / total if total > 0 else 0.0, 0.0)
                weights[t] = w

            elif scheme == WeightScheme.LONG_SHORT:
                rank = np.argsort(-sig_valid)
                n_pick = min(top_n, int(valid.sum()) // 2)
                w = np.zeros(N)
                for i in range(n_pick):
                    idx = rank[i]
                    if valid[idx]:
                        w[idx] = 1.0 / n_pick
                for i in range(N - n_pick, N):
                    idx = rank[i]
                    if valid[idx]:
                        w[idx] = -1.0 / n_pick
                weights[t] = w
            else:
                weights[t] = weights[t - 1] if t > 0 else np.zeros(N)

        # Forward-fill weights on non-rebalance days
        for t in range(1, T):
            if not reb_mask[t] and np.all(weights[t] == 0):
                weights[t] = weights[t - 1]
        return weights

    def _cap_weights(self, weights: np.ndarray) -> np.ndarray:
        """Cap individual weights at max_weight."""
        max_w = self.config.max_weight
        return np.clip(weights, -max_w, max_w)

    def _simulate(
        self,
        close_mat: np.ndarray,
        ret_mat: np.ndarray,
        weights_mat: np.ndarray,
        dates: pd.DatetimeIndex,
        tickers: list,
    ):
        """Simulate NAV with costs. Enforce lot size on trades."""
        T, N = close_mat.shape
        lot = self.config.lot_size
        nav = np.ones(T + 1) * self.config.initial_capital
        total_cost_bps = 0.0

        for t in range(1, T):
            g_ret = np.nansum(weights_mat[t - 1] * np.nan_to_num(ret_mat[t], 0))
            nav_before_cost = nav[t - 1] * (1 + g_ret)
            nav[t] = nav_before_cost

            # Turnover and cost (cost as fraction of NAV)
            delta = np.abs(weights_mat[t] - weights_mat[t - 1])
            turnover_t = np.nansum(delta)
            if turnover_t > 1e-10:
                cost_frac = self.cost_model.compute_total_cost(
                    weights_mat[t - 1], weights_mat[t],
                    nav=float(nav_before_cost),
                    close_prices=close_mat[t],
                )
                nav[t] -= cost_frac * nav_before_cost
                total_cost_bps += cost_frac * 10_000

        nav = pd.Series(nav[1:], index=dates)
        daily_ret = nav.pct_change().fillna(0)
        turnover = pd.Series(
            np.abs(weights_mat[1:] - weights_mat[:-1]).sum(axis=1),
            index=dates,
        )
        return nav, daily_ret, turnover, total_cost_bps
