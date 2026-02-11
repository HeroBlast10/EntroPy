"""Baseline portfolio construction: quantile-based stock selection.

Strategy logic
--------------
1. On each rebalance date, rank all tradable stocks by the alpha signal.
2. Select stocks in the top quantile (long) and optionally the bottom
   quantile (short).
3. Assign weights via the chosen scheme (equal, market-cap, signal-prop,
   or inverse-volatility).

Supports both **long-only** and **long-short** modes.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

from entropy.portfolio.construction import (
    PortfolioConfig,
    PortfolioConstructor,
    PortfolioMode,
    WeightScheme,
)


class QuantilePortfolio(PortfolioConstructor):
    """Quantile-based stock selection with flexible weighting."""

    def __init__(self, config: Optional[PortfolioConfig] = None) -> None:
        super().__init__(config)

    # ------------------------------------------------------------------
    # Core
    # ------------------------------------------------------------------

    def _generate_weights(
        self,
        signal: pd.DataFrame,
        universe: pd.DataFrame,
        date: pd.Timestamp,
        prev_weights: Optional[pd.Series] = None,
    ) -> pd.Series:
        cfg = self.config
        sig_col = self._detect_signal_col(signal)

        # --- Today's cross-section ---
        sig_today = signal.loc[signal["date"] == date, ["ticker", sig_col]].copy()
        uni_today = universe.loc[universe["date"] == date, "ticker"]
        tradable = set(uni_today)
        sig_today = sig_today[sig_today["ticker"].isin(tradable)].dropna(subset=[sig_col])

        if sig_today.empty:
            return pd.Series(dtype=float)

        sig_today = sig_today.set_index("ticker")[sig_col]

        # --- Top-N override ---
        if cfg.top_n is not None:
            long_tickers = sig_today.nlargest(cfg.top_n).index
            short_tickers = sig_today.nsmallest(cfg.top_n).index if \
                cfg.mode == PortfolioMode.LONG_SHORT else pd.Index([])
        else:
            # --- Quantile assignment ---
            try:
                q_labels = pd.qcut(sig_today.rank(method="first"), cfg.n_quantiles,
                                   labels=False, duplicates="drop") + 1
            except ValueError:
                return pd.Series(dtype=float)

            long_tickers = q_labels[q_labels == cfg.long_quantile].index
            short_tickers = q_labels[q_labels == cfg.short_quantile].index if \
                cfg.mode == PortfolioMode.LONG_SHORT else pd.Index([])

        if long_tickers.empty:
            return pd.Series(dtype=float)

        # --- Weight assignment ---
        long_w = self._assign_weights(
            sig_today.loc[long_tickers], signal, universe, date, side="long")
        short_w = pd.Series(dtype=float)
        if not short_tickers.empty:
            short_w = self._assign_weights(
                sig_today.loc[short_tickers], signal, universe, date, side="short")

        # --- Combine ---
        weights = pd.concat([long_w, short_w])

        # --- Normalise ---
        weights = self.normalise_weights(weights, cfg.mode)

        return weights

    # ------------------------------------------------------------------
    # Weighting schemes
    # ------------------------------------------------------------------

    def _assign_weights(
        self,
        selected_signal: pd.Series,
        full_signal: pd.DataFrame,
        universe: pd.DataFrame,
        date: pd.Timestamp,
        side: str = "long",
    ) -> pd.Series:
        """Assign weights to selected tickers based on the weight scheme."""
        cfg = self.config
        tickers = selected_signal.index
        n = len(tickers)
        if n == 0:
            return pd.Series(dtype=float)

        if cfg.weight_scheme == WeightScheme.EQUAL:
            w = pd.Series(1.0 / n, index=tickers)

        elif cfg.weight_scheme == WeightScheme.MARKET_CAP:
            # Try to get market_cap from universe
            uni_today = universe.loc[universe["date"] == date]
            if "market_cap" in uni_today.columns:
                mcap = uni_today.set_index("ticker")["market_cap"].reindex(tickers)
                mcap = mcap.fillna(mcap.median())  # fallback for missing
                mcap = mcap.clip(lower=1.0)
                w = mcap / mcap.sum()
            else:
                w = pd.Series(1.0 / n, index=tickers)
                logger.debug("market_cap not available, falling back to equal weight")

        elif cfg.weight_scheme == WeightScheme.SIGNAL:
            # Weight proportional to signal strength
            sig_abs = selected_signal.abs()
            total = sig_abs.sum()
            if total > 0:
                w = sig_abs / total
            else:
                w = pd.Series(1.0 / n, index=tickers)

        elif cfg.weight_scheme == WeightScheme.INVERSE_VOL:
            # Requires volatility data — attempt to compute from signal df
            # For now, use equal weight as fallback
            w = pd.Series(1.0 / n, index=tickers)
            logger.debug("inverse_vol weighting: using equal weight fallback")

        else:
            w = pd.Series(1.0 / n, index=tickers)

        # Flip sign for short side
        if side == "short":
            w = -w

        return w

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_signal_col(signal: pd.DataFrame) -> str:
        """Find the signal column (first column that isn't date/ticker)."""
        for c in signal.columns:
            if c not in ("date", "ticker"):
                return c
        raise ValueError("Signal DataFrame has no factor column (only date & ticker found)")
