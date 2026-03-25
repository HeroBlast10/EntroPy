"""Regime signal scorecard.

Metrics for regime overlay signals (HMM turbulence):
- Baseline metrics: Performance without regime overlay
- Overlay metrics: Performance with regime overlay
- Comparison: Improvement in Sharpe, drawdown, turnover
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd
from loguru import logger

from quant_platform.core.signals.base import FactorMeta


class RegimeScorecard:
    """Scorecard for regime overlay signals."""
    
    def evaluate(
        self,
        signal_df: pd.DataFrame,
        signal_col: str,
        meta: FactorMeta,
        prices: pd.DataFrame,
        baseline_returns: pd.Series = None,
        overlay_threshold: float = 0.5,
        **kwargs,
    ) -> Dict[str, Any]:
        """Evaluate regime overlay signal.
        
        Parameters
        ----------
        signal_df : DataFrame [date, ticker, signal_col]
            Regime signal (e.g., turbulence probability)
        signal_col : str
        meta : FactorMeta
        prices : DataFrame [date, ticker, close, ...]
        baseline_returns : Series, optional
            Baseline portfolio returns (without overlay)
            If None, uses equal-weight market returns
        overlay_threshold : float
            Threshold for regime signal (default 0.5)
            When signal > threshold, reduce exposure
        
        Returns
        -------
        Dict with:
            - baseline_sharpe: Sharpe without overlay
            - overlay_sharpe: Sharpe with overlay
            - sharpe_improvement: Difference
            - baseline_max_dd: Max drawdown without overlay
            - overlay_max_dd: Max drawdown with overlay
            - dd_improvement: Improvement in max drawdown
            - regime_detection_rate: % of time in high-risk regime
        """
        # If no baseline returns provided, compute equal-weight market returns
        if baseline_returns is None:
            prices = prices.copy()
            prices = prices.sort_values(["ticker", "date"])
            prices["ret"] = prices.groupby("ticker")["close"].pct_change()
            baseline_returns = prices.groupby("date")["ret"].mean()
        
        # Get regime signal (aggregate across tickers if needed)
        if "ticker" in signal_df.columns:
            # Average regime signal across all tickers for each date
            regime_signal = signal_df.groupby("date")[signal_col].mean()
        else:
            regime_signal = signal_df.set_index("date")[signal_col]
        
        # Align returns and regime signal
        common_dates = baseline_returns.index.intersection(regime_signal.index)
        baseline_ret = baseline_returns.loc[common_dates]
        regime_sig = regime_signal.loc[common_dates]
        
        if len(common_dates) == 0:
            logger.warning("No overlapping dates for regime evaluation")
            return self._empty_result()
        
        # Apply overlay: reduce exposure when regime signal > threshold
        # Simple approach: scale returns by (1 - regime_signal) when signal > threshold
        overlay_multiplier = pd.Series(1.0, index=common_dates)
        high_risk = regime_sig > overlay_threshold
        overlay_multiplier[high_risk] = 1.0 - regime_sig[high_risk]
        
        overlay_ret = baseline_ret * overlay_multiplier
        
        # Compute metrics
        baseline_sharpe = self._compute_sharpe(baseline_ret)
        overlay_sharpe = self._compute_sharpe(overlay_ret)
        
        baseline_max_dd = self._compute_max_drawdown(baseline_ret)
        overlay_max_dd = self._compute_max_drawdown(overlay_ret)
        
        # Regime detection rate
        regime_detection_rate = high_risk.mean()
        
        # Average exposure reduction
        avg_exposure_reduction = (1.0 - overlay_multiplier).mean()
        
        return {
            "baseline_sharpe": float(baseline_sharpe),
            "overlay_sharpe": float(overlay_sharpe),
            "sharpe_improvement": float(overlay_sharpe - baseline_sharpe),
            "baseline_max_dd": float(baseline_max_dd),
            "overlay_max_dd": float(overlay_max_dd),
            "dd_improvement": float(baseline_max_dd - overlay_max_dd),  # Positive = better
            "regime_detection_rate": float(regime_detection_rate),
            "avg_exposure_reduction": float(avg_exposure_reduction),
            "n_periods": len(common_dates),
        }
    
    def _compute_sharpe(self, returns: pd.Series) -> float:
        """Compute annualized Sharpe ratio."""
        if len(returns) < 2 or returns.std() == 0:
            return 0.0
        return returns.mean() / returns.std() * np.sqrt(252)
    
    def _compute_max_drawdown(self, returns: pd.Series) -> float:
        """Compute maximum drawdown."""
        cum_ret = (1 + returns).cumprod()
        running_max = cum_ret.expanding().max()
        drawdown = (cum_ret - running_max) / running_max
        return abs(drawdown.min())
    
    def _empty_result(self) -> Dict[str, Any]:
        """Return empty result dict."""
        return {
            "baseline_sharpe": 0.0,
            "overlay_sharpe": 0.0,
            "sharpe_improvement": 0.0,
            "baseline_max_dd": 0.0,
            "overlay_max_dd": 0.0,
            "dd_improvement": 0.0,
            "regime_detection_rate": 0.0,
            "avg_exposure_reduction": 0.0,
            "n_periods": 0,
        }
