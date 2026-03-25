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

from quant_platform.core.portfolio.construction import (
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
            # Inverse-volatility weighting: weight_i = (1/σ_i) / Σ(1/σ_j)
            # This is naive risk parity — equal risk contribution
            w = self._compute_inverse_vol_weights(
                tickers, date, full_signal, universe
            )
            if w.empty:
                # Fallback to equal weight if volatility computation fails
                w = pd.Series(1.0 / n, index=tickers)
                logger.debug("inverse_vol weighting: volatility data unavailable, using equal weight")

        else:
            w = pd.Series(1.0 / n, index=tickers)

        # Flip sign for short side
        if side == "short":
            w = -w

        return w

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _compute_inverse_vol_weights(
        self,
        tickers: pd.Index,
        date: pd.Timestamp,
        signal: pd.DataFrame,
        universe: pd.DataFrame,
    ) -> pd.Series:
        """Compute inverse-volatility weights for selected tickers.
        
        Volatility is computed as rolling std of returns over a lookback window.
        Weight_i = (1 / σ_i) / Σ(1 / σ_j)
        
        This implements naive risk parity: each stock contributes equal risk.
        
        Parameters
        ----------
        tickers : selected tickers to weight
        date : current rebalance date
        signal : full signal DataFrame (may contain prices if available)
        universe : universe DataFrame (may contain volatility data)
        
        Returns
        -------
        Series of weights indexed by ticker, or empty Series if computation fails
        """
        # Try to get volatility from universe first (pre-computed)
        uni_today = universe.loc[universe["date"] == date]
        if "volatility" in uni_today.columns:
            vol = uni_today.set_index("ticker")["volatility"].reindex(tickers)
            vol = vol.dropna()
            if len(vol) >= len(tickers) * 0.5:  # At least 50% coverage
                return self._weights_from_volatility(vol)
        
        # Otherwise, try to compute from prices in signal DataFrame
        if "adj_close" in signal.columns or "close" in signal.columns:
            price_col = "adj_close" if "adj_close" in signal.columns else "close"
            vol = self._compute_rolling_volatility(
                signal, tickers, date, price_col, window=63
            )
            if not vol.empty:
                return self._weights_from_volatility(vol)
        
        # If all else fails, return empty (caller will use equal weight fallback)
        return pd.Series(dtype=float)
    
    @staticmethod
    def _weights_from_volatility(vol: pd.Series) -> pd.Series:
        """Convert volatility series to inverse-vol weights.
        
        weight_i = (1 / σ_i) / Σ(1 / σ_j)
        """
        # Handle zero or near-zero volatility
        vol = vol.clip(lower=1e-6)
        
        # Inverse volatility
        inv_vol = 1.0 / vol
        
        # Normalize to sum to 1
        weights = inv_vol / inv_vol.sum()
        
        return weights
    
    @staticmethod
    def _compute_rolling_volatility(
        signal: pd.DataFrame,
        tickers: pd.Index,
        date: pd.Timestamp,
        price_col: str = "adj_close",
        window: int = 63,
    ) -> pd.Series:
        """Compute rolling volatility from price data.
        
        Parameters
        ----------
        signal : DataFrame with [date, ticker, price_col]
        tickers : tickers to compute volatility for
        date : current date
        price_col : column name for prices
        window : lookback window in trading days (default 63 = ~3 months)
        
        Returns
        -------
        Series of volatility indexed by ticker
        """
        # Filter to relevant tickers and dates up to current date
        sig = signal[["date", "ticker", price_col]].copy()
        sig["date"] = pd.to_datetime(sig["date"])
        sig = sig[sig["ticker"].isin(tickers)]
        sig = sig[sig["date"] <= date]
        
        if sig.empty:
            return pd.Series(dtype=float)
        
        # Compute returns and rolling volatility for each ticker
        vol_list = []
        for ticker in tickers:
            ticker_data = sig[sig["ticker"] == ticker].sort_values("date")
            
            if len(ticker_data) < window:
                continue  # Not enough history
            
            # Compute returns
            ticker_data = ticker_data.copy()
            ticker_data["return"] = ticker_data[price_col].pct_change()
            
            # Rolling volatility (annualized)
            recent_returns = ticker_data["return"].iloc[-window:]
            vol = recent_returns.std() * np.sqrt(252)  # Annualize
            
            if pd.notna(vol) and vol > 0:
                vol_list.append({"ticker": ticker, "volatility": vol})
        
        if not vol_list:
            return pd.Series(dtype=float)
        
        vol_df = pd.DataFrame(vol_list)
        return vol_df.set_index("ticker")["volatility"]

    @staticmethod
    def _detect_signal_col(signal: pd.DataFrame) -> str:
        """Find the signal column (first column that isn't date/ticker)."""
        for c in signal.columns:
            if c not in ("date", "ticker"):
                return c
        raise ValueError("Signal DataFrame has no factor column (only date & ticker found)")
