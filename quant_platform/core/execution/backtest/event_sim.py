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

    def __post_init__(self) -> None:
        if self.rebalance_freq not in {"D", "W", "M"}:
            raise ValueError("rebalance_freq must be D, W, or M")
        if self.signal_lag < 0:
            raise ValueError("signal_lag cannot be negative")
        if self.top_n <= 0:
            raise ValueError("top_n must be positive")
        if not 0 < self.max_weight <= 1:
            raise ValueError("max_weight must be in (0, 1]")
        if self.lot_size <= 0:
            raise ValueError("lot_size must be positive")
        if self.initial_capital <= 0:
            raise ValueError("initial_capital must be positive")


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
        (
            close_mat,
            ret_mat,
            signal_mat,
            eligible_mat,
            buyable_mat,
            sellable_mat,
            dates,
            tickers,
        ) = self._build_matrices(
            panel, prices, ticker_col, universe
        )
        reb_dates = self._rebalance_schedule(dates)
        weights_mat = self._signal_to_weights(
            signal_mat, eligible_mat, dates, reb_dates, tickers
        )
        weights_mat = self._cap_weights(weights_mat)
        nav, daily_ret, turnover, cost_bps, actual_weights = self._simulate(
            close_mat,
            ret_mat,
            weights_mat,
            buyable_mat,
            sellable_mat,
            reb_dates,
            dates,
            tickers,
        )
        sharpe = float(daily_ret.mean() / daily_ret.std() * np.sqrt(252)) if daily_ret.std() > 0 else 0.0
        ann_ret = float((1 + daily_ret).prod() ** (252 / len(daily_ret)) - 1) if len(daily_ret) > 0 else 0.0
        cum = (1 + daily_ret).cumprod()
        peak = cum.cummax()
        dd = (cum / peak - 1.0)
        max_dd = float(dd.min()) if len(dd) > 0 else 0.0

        weights_df = pd.DataFrame(
            actual_weights, index=dates, columns=tickers,
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
        # Raw close is required for share/lot sizing.  Adjusted close is used
        # only for total-return marking.
        close_col = "close" if "close" in prices.columns else "adj_close"
        close_wide = prices.pivot_table(
            index="date", columns=price_ticker, values=close_col, aggfunc="last"
        )
        dates = close_wide.index
        tickers = close_wide.columns.tolist()

        return_col = "adj_close" if "adj_close" in prices.columns else close_col
        return_wide = prices.pivot_table(
            index="date", columns=price_ticker, values=return_col, aggfunc="last"
        ).reindex(index=dates, columns=tickers)
        ret_wide = return_wide.pct_change()
        ret_mat = ret_wide.values
        close_mat = close_wide.values

        # Prefer the canonical signal column and never mistake an execution
        # status flag for alpha merely because of input column order.
        status_columns = {
            "limit_up",
            "limit_down",
            "is_st",
            "is_suspended",
            "is_new_listing",
            "pass_all_filters",
        }
        candidates = [
            c
            for c in panel.columns
            if c not in {"date", ticker_col, *status_columns}
        ]
        if "signal" in candidates:
            sig_col = "signal"
        elif len(candidates) == 1:
            sig_col = candidates[0]
        else:
            raise ValueError(
                "Panel must contain a 'signal' column or exactly one alpha "
                f"column; found {candidates}"
            )
        signal_wide = panel.pivot_table(
            index="date", columns=ticker_col, values=sig_col, aggfunc="last"
        )
        signal_wide = signal_wide.reindex(index=dates, columns=tickers)
        signal_mat = signal_wide.values

        # Eligibility filters block both directions.  Price limits are applied
        # later by order direction: limit-up blocks buys only, limit-down sells.
        eligible_mat = np.ones_like(signal_mat, dtype=bool)
        eligible_mat &= ~np.isnan(close_mat) & (close_mat > 0)
        limit_up_mat = np.zeros_like(signal_mat, dtype=bool)
        limit_down_mat = np.zeros_like(signal_mat, dtype=bool)

        if "limit_up" in panel.columns:
            limit_up_wide = panel.pivot_table(
                index="date", columns=ticker_col, values="limit_up", aggfunc="last"
            ).reindex(index=dates, columns=tickers)
            if self.config.skip_limit_up:
                limit_up_mat = limit_up_wide.fillna(False).values.astype(bool)
        if "limit_down" in panel.columns:
            limit_down_wide = panel.pivot_table(
                index="date", columns=ticker_col, values="limit_down", aggfunc="last"
            ).reindex(index=dates, columns=tickers)
            if self.config.skip_limit_down:
                limit_down_mat = limit_down_wide.fillna(False).values.astype(bool)
        if "is_st" in panel.columns:
            is_st_wide = panel.pivot_table(
                index="date", columns=ticker_col, values="is_st", aggfunc="last"
            ).reindex(index=dates, columns=tickers)
            if self.config.skip_st:
                eligible_mat &= ~is_st_wide.fillna(False).values.astype(bool)
        if "is_suspended" in panel.columns:
            susp_wide = panel.pivot_table(
                index="date", columns=ticker_col, values="is_suspended", aggfunc="last"
            ).reindex(index=dates, columns=tickers)
            if self.config.skip_suspended:
                eligible_mat &= ~susp_wide.fillna(True).values.astype(bool)
        if "is_new_listing" in panel.columns:
            new_wide = panel.pivot_table(
                index="date", columns=ticker_col, values="is_new_listing", aggfunc="last"
            ).reindex(index=dates, columns=tickers)
            if self.config.skip_new_listing:
                eligible_mat &= ~new_wide.fillna(False).values.astype(bool)

        if universe is not None:
            uni_wide = universe.pivot_table(
                index="date", columns="ticker", values="pass_all_filters", aggfunc="last"
            ).reindex(index=dates, columns=tickers)
            eligible_mat &= uni_wide.fillna(False).values.astype(bool)

        buyable_mat = eligible_mat & ~limit_up_mat
        sellable_mat = eligible_mat & ~limit_down_mat
        return (
            close_mat,
            ret_mat,
            signal_mat,
            eligible_mat,
            buyable_mat,
            sellable_mat,
            dates,
            tickers,
        )

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
        else:  # guarded by BacktestConfig.__post_init__
            raise ValueError(f"Unknown rebalance frequency: {freq}")
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
                valid_idx = np.flatnonzero(valid)
                rank = valid_idx[np.argsort(-sig_t[valid_idx])]
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
                valid_idx = np.flatnonzero(valid)
                ordered = valid_idx[np.argsort(sig_t[valid_idx])]
                n_pick = min(top_n, int(valid.sum()) // 2)
                w = np.zeros(N)
                if n_pick > 0:
                    w[ordered[-n_pick:]] = 1.0 / n_pick
                    w[ordered[:n_pick]] = -1.0 / n_pick
                weights[t] = w
            else:
                weights[t] = weights[t - 1] if t > 0 else np.zeros(N)

        # Forward-fill weights on non-rebalance days
        for t in range(1, T):
            if not reb_mask[t] and np.all(weights[t] == 0):
                weights[t] = weights[t - 1]
        return weights

    def _cap_weights(self, weights: np.ndarray) -> np.ndarray:
        """Cap positions while preserving each sleeve's intended notional."""
        max_w = self.config.max_weight
        capped = np.zeros_like(weights, dtype=float)
        for row_idx, row in enumerate(weights):
            positive = row > 0
            negative = row < 0
            if positive.any():
                capped[row_idx, positive] = self._cap_sleeve(
                    row[positive], max_w, float(row[positive].sum())
                )
            if negative.any():
                capped[row_idx, negative] = -self._cap_sleeve(
                    -row[negative], max_w, float(-row[negative].sum())
                )
        return capped

    @staticmethod
    def _cap_sleeve(values: np.ndarray, cap: float, target_sum: float) -> np.ndarray:
        if len(values) * cap < target_sum - 1e-12:
            raise ValueError(
                "max_weight is infeasible for the number of selected names"
            )
        result = np.zeros(len(values), dtype=float)
        remaining = target_sum
        active = np.ones(len(values), dtype=bool)
        scores = np.maximum(np.asarray(values, dtype=float), 0.0)
        while remaining > 1e-12 and active.any():
            idx = np.flatnonzero(active)
            active_scores = scores[idx]
            proposal = (
                np.full(len(idx), remaining / len(idx))
                if active_scores.sum() <= 0
                else remaining * active_scores / active_scores.sum()
            )
            room = cap - result[idx]
            hit = proposal >= room - 1e-12
            if not hit.any():
                result[idx] += proposal
                break
            result[idx[hit]] += room[hit]
            remaining -= float(room[hit].sum())
            active[idx[hit]] = False
        return result

    def _simulate(
        self,
        close_mat: np.ndarray,
        ret_mat: np.ndarray,
        weights_mat: np.ndarray,
        buyable_mat: np.ndarray,
        sellable_mat: np.ndarray,
        rebalance_mask: np.ndarray,
        dates: pd.DatetimeIndex,
        tickers: list,
    ):
        """Simulate NAV with costs. Enforce lot size on trades."""
        T, N = close_mat.shape
        lot = max(int(self.config.lot_size), 1)
        nav_values = np.zeros(T, dtype=float)
        actual_weights = np.zeros((T, N), dtype=float)
        turnover_values = np.zeros(T, dtype=float)
        position_values = np.zeros(N, dtype=float)
        pending = np.zeros(N, dtype=bool)
        previous_nav = float(self.config.initial_capital)
        total_cost_bps = 0.0

        for t in range(T):
            close_today = close_mat[t]
            day_return = np.nan_to_num(ret_mat[t], nan=0.0)
            pnl = float(np.sum(position_values * day_return))
            position_values = position_values * (1.0 + day_return)
            nav_before_cost = previous_nav + pnl
            if nav_before_cost <= 0:
                raise RuntimeError(f"NAV became non-positive on {dates[t]}")

            weights_before = position_values / nav_before_cost
            desired_values = weights_mat[t] * nav_before_cost
            desired_shares = np.zeros(N, dtype=float)
            valid_price = np.isfinite(close_today) & (close_today > 0)
            desired_shares[valid_price] = (
                np.sign(desired_values[valid_price])
                * np.floor(
                    np.abs(desired_values[valid_price])
                    / close_today[valid_price]
                    / lot
                )
                * lot
            )
            rounded_values = desired_shares * np.where(valid_price, close_today, 0.0)
            direction = desired_values - position_values
            direction_ok = (
                ((direction > 1e-10) & buyable_mat[t])
                | ((direction < -1e-10) & sellable_mat[t])
            )
            if rebalance_mask[t]:
                attempted = valid_price & direction_ok
                needs_trade = np.abs(direction) > 1e-10
                pending = needs_trade & ~attempted
            else:
                attempted = pending & valid_price & direction_ok
                pending[attempted] = False

            executed_values = position_values.copy()
            executed_values[attempted] = rounded_values[attempted]
            weights_after = executed_values / nav_before_cost
            turnover_t = float(np.abs(weights_after - weights_before).sum())
            cost_frac = 0.0
            if turnover_t > 1e-10:
                cost_frac = self.cost_model.compute_total_cost(
                    weights_before,
                    weights_after,
                    nav=float(nav_before_cost),
                    close_prices=close_today,
                )
                total_cost_bps += cost_frac * 10_000
            nav_after_cost = nav_before_cost * (1.0 - cost_frac)
            position_values = executed_values
            actual_weights[t] = executed_values / nav_after_cost
            turnover_values[t] = turnover_t
            nav_values[t] = nav_after_cost
            previous_nav = nav_after_cost

        nav = pd.Series(nav_values, index=dates, name="nav")
        daily_ret = nav.pct_change()
        if len(daily_ret):
            daily_ret.iloc[0] = nav.iloc[0] / self.config.initial_capital - 1.0
        turnover = pd.Series(turnover_values, index=dates, name="turnover")
        return nav, daily_ret.fillna(0.0), turnover, total_cost_bps, actual_weights
