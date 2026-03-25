"""Time-series signal scorecard.

Metrics for per-asset latent-state features (Kalman, entropy, Hurst):
- Hit rate: % of correct directional predictions
- Directional accuracy: Sign agreement between signal and future return
- Directional Sharpe: Sharpe ratio of signal-weighted returns
- Mean absolute error: Average prediction error
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd
from loguru import logger

from quant_platform.core.signals.base import FactorMeta


class TimeSeriesScorecard:
    """Scorecard for time-series features."""
    
    def evaluate(
        self,
        signal_df: pd.DataFrame,
        signal_col: str,
        meta: FactorMeta,
        prices: pd.DataFrame,
        forward_periods: int = 1,
        **kwargs,
    ) -> Dict[str, Any]:
        """Evaluate time-series signal.
        
        Parameters
        ----------
        signal_df : DataFrame [date, ticker, signal_col]
        signal_col : str
        meta : FactorMeta
        prices : DataFrame [date, ticker, close, ...]
        forward_periods : int
            Number of days for forward returns (default 1)
        
        Returns
        -------
        Dict with:
            - hit_rate: % of correct directional predictions
            - directional_accuracy: Sign agreement
            - directional_sharpe: Sharpe of signal-weighted returns
            - mean_absolute_error: Average prediction error
        """
        # Add forward returns
        prices = prices.copy()
        prices = prices.sort_values(["ticker", "date"])
        prices["fwd_ret"] = prices.groupby("ticker")["close"].pct_change(forward_periods).shift(-forward_periods)
        
        # Merge signal with forward returns
        eval_df = signal_df.merge(
            prices[["date", "ticker", "fwd_ret"]],
            on=["date", "ticker"],
            how="inner",
        )
        eval_df = eval_df.dropna(subset=[signal_col, "fwd_ret"])
        
        if len(eval_df) == 0:
            logger.warning("No valid data for time-series evaluation")
            return self._empty_result()
        
        # Directional accuracy: sign agreement
        signal_sign = np.sign(eval_df[signal_col])
        return_sign = np.sign(eval_df["fwd_ret"])
        
        # Hit rate: % of correct predictions
        correct = (signal_sign == return_sign) & (signal_sign != 0)
        hit_rate = correct.mean()
        
        # Directional accuracy (same as hit rate but clearer name)
        directional_accuracy = hit_rate
        
        # Directional Sharpe: Sharpe of signal-weighted returns
        # Weight each return by signal strength
        weighted_returns = eval_df[signal_col] * eval_df["fwd_ret"]
        
        # Group by date to get portfolio-level returns
        daily_returns = weighted_returns.groupby(eval_df["date"]).mean()
        
        if len(daily_returns) > 1 and daily_returns.std() > 0:
            directional_sharpe = daily_returns.mean() / daily_returns.std() * np.sqrt(252)
        else:
            directional_sharpe = 0.0
        
        # Mean absolute error (if signal is meant to predict returns)
        mae = np.abs(eval_df[signal_col] - eval_df["fwd_ret"]).mean()
        
        # Per-ticker hit rate (for diagnostics)
        ticker_hit_rates = {}
        for ticker, group in eval_df.groupby("ticker"):
            ticker_signal_sign = np.sign(group[signal_col])
            ticker_return_sign = np.sign(group["fwd_ret"])
            ticker_correct = (ticker_signal_sign == ticker_return_sign) & (ticker_signal_sign != 0)
            ticker_hit_rates[ticker] = ticker_correct.mean()
        
        avg_ticker_hit_rate = np.mean(list(ticker_hit_rates.values()))
        
        return {
            "hit_rate": float(hit_rate),
            "directional_accuracy": float(directional_accuracy),
            "directional_sharpe": float(directional_sharpe),
            "mean_absolute_error": float(mae),
            "avg_ticker_hit_rate": float(avg_ticker_hit_rate),
            "n_observations": len(eval_df),
            "n_tickers": eval_df["ticker"].nunique(),
        }
    
    def _empty_result(self) -> Dict[str, Any]:
        """Return empty result dict."""
        return {
            "hit_rate": 0.0,
            "directional_accuracy": 0.0,
            "directional_sharpe": 0.0,
            "mean_absolute_error": 0.0,
            "avg_ticker_hit_rate": 0.0,
            "n_observations": 0,
            "n_tickers": 0,
        }
