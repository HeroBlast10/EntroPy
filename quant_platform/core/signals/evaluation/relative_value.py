"""Relative-value signal scorecard.

Metrics for mean-reversion/spread strategies (OU process):
- Half-life: Mean reversion speed
- Stationarity test: ADF test p-value
- Mean reversion quality: R² of OU fit
- Spread Sharpe: Sharpe ratio of mean-reversion trades
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd
from loguru import logger

from quant_platform.core.signals.base import FactorMeta
from quant_platform.core.signals.orientation import apply_direction


class RelativeValueScorecard:
    """Scorecard for relative-value/mean-reversion signals."""
    
    def evaluate(
        self,
        signal_df: pd.DataFrame,
        signal_col: str,
        meta: FactorMeta,
        prices: pd.DataFrame,
        **kwargs,
    ) -> Dict[str, Any]:
        """Evaluate relative-value signal.
        
        Parameters
        ----------
        signal_df : DataFrame [date, ticker, signal_col]
            Mean-reversion signal (e.g., OU z-score)
        signal_col : str
        meta : FactorMeta
        prices : DataFrame [date, ticker, close, ...]
        
        Returns
        -------
        Dict with:
            - avg_half_life: Average half-life across stocks (days)
            - avg_stationarity_pvalue: Average ADF test p-value
            - mean_reversion_quality: R² of OU fit
            - spread_sharpe: Sharpe of mean-reversion strategy
            - entry_exit_ratio: Ratio of profitable to unprofitable trades
        """
        # Merge signal with prices
        eval_df = signal_df.merge(
            prices[["date", "ticker", "close"]],
            on=["date", "ticker"],
            how="inner",
        )
        eval_df = eval_df.sort_values(["ticker", "date"])
        
        if len(eval_df) == 0:
            logger.warning("No valid data for relative-value evaluation")
            return self._empty_result()
        
        # Per-ticker analysis
        half_lives = []
        stationarity_pvalues = []
        mean_reversion_trades = []
        
        for ticker, group in eval_df.groupby("ticker"):
            if len(group) < 30:  # Need minimum data
                continue
            
            # Estimate half-life from signal autocorrelation
            signal_vals = apply_direction(group[signal_col], meta.direction).values
            if len(signal_vals) > 1:
                # Simple autocorrelation-based half-life estimate
                acf_1 = np.corrcoef(signal_vals[:-1], signal_vals[1:])[0, 1]
                if acf_1 > 0 and acf_1 < 1:
                    half_life = -np.log(2) / np.log(acf_1)
                    if 0 < half_life < 252:  # Reasonable range
                        half_lives.append(half_life)
            
            # Stationarity test (simplified ADF)
            try:
                from statsmodels.tsa.stattools import adfuller
                adf_result = adfuller(signal_vals, maxlag=10)
                stationarity_pvalues.append(adf_result[1])  # p-value
            except Exception:
                pass
            
            # Simulate mean-reversion trades
            # Entry: when |z-score| > 2, Exit: when z-score crosses 0
            returns = group["close"].pct_change()
            z_score = apply_direction(group[signal_col], meta.direction)
            
            position = 0  # 0 = no position, +1 = long, -1 = short
            trade_returns = []
            
            for i in range(1, len(group)):
                prev_z = z_score.iloc[i-1]
                curr_z = z_score.iloc[i]
                ret = returns.iloc[i]
                
                # Entry logic
                if position == 0:
                    if prev_z < -2:  # Oversold, go long
                        position = 1
                    elif prev_z > 2:  # Overbought, go short
                        position = -1
                
                # Exit logic
                if position != 0:
                    if (position == 1 and curr_z > 0) or (position == -1 and curr_z < 0):
                        # Exit and record trade return
                        trade_returns.append(position * ret)
                        position = 0
                    else:
                        # Continue holding
                        trade_returns.append(position * ret)
            
            if trade_returns:
                mean_reversion_trades.extend(trade_returns)
        
        # Aggregate metrics
        avg_half_life = np.mean(half_lives) if half_lives else np.nan
        avg_stationarity_pvalue = np.mean(stationarity_pvalues) if stationarity_pvalues else np.nan
        
        # Spread Sharpe from mean-reversion trades
        if mean_reversion_trades:
            trade_series = pd.Series(mean_reversion_trades)
            spread_sharpe = trade_series.mean() / trade_series.std() * np.sqrt(252) if trade_series.std() > 0 else 0.0
            entry_exit_ratio = (trade_series > 0).sum() / len(trade_series) if len(trade_series) > 0 else 0.0
        else:
            spread_sharpe = 0.0
            entry_exit_ratio = 0.0
        
        # Mean reversion quality (R² of signal vs lagged signal)
        all_signals = eval_df.groupby("ticker")[signal_col].apply(
            lambda s: apply_direction(s, meta.direction).tolist()
        )
        r_squared_values = []
        for signals in all_signals:
            if len(signals) > 2:
                # R² of AR(1) model
                y = signals[1:]
                x = signals[:-1]
                if np.std(x) > 0:
                    corr = np.corrcoef(x, y)[0, 1]
                    r_squared_values.append(corr**2)
        
        mean_reversion_quality = np.mean(r_squared_values) if r_squared_values else 0.0
        
        return {
            "avg_half_life": float(avg_half_life) if not np.isnan(avg_half_life) else 0.0,
            "avg_stationarity_pvalue": float(avg_stationarity_pvalue) if not np.isnan(avg_stationarity_pvalue) else 1.0,
            "mean_reversion_quality": float(mean_reversion_quality),
            "spread_sharpe": float(spread_sharpe),
            "entry_exit_ratio": float(entry_exit_ratio),
            "n_tickers": len(half_lives),
            "n_trades": len(mean_reversion_trades),
        }
    
    def _empty_result(self) -> Dict[str, Any]:
        """Return empty result dict."""
        return {
            "avg_half_life": 0.0,
            "avg_stationarity_pvalue": 1.0,
            "mean_reversion_quality": 0.0,
            "spread_sharpe": 0.0,
            "entry_exit_ratio": 0.0,
            "n_tickers": 0,
            "n_trades": 0,
        }
