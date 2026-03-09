"""Time-series feature forecaster alpha model.

Combines Kalman velocity, spectral entropy, Hurst exponent, and higher-moment
features into a single per-stock alpha score.

Also includes the MomentumZScore strategy logic (velocity z-score with
overbought/oversold thresholds) from TradeX.
"""
from __future__ import annotations
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from numba import njit
from loguru import logger


@njit(cache=True)
def _rolling_zscore(arr, window):
    """Rolling Z-score computation."""
    n = len(arr)
    out = np.full(n, np.nan)
    for i in range(window - 1, n):
        sub = arr[i - window + 1: i + 1]
        m = 0.0; cnt = 0
        for j in range(window):
            if not np.isnan(sub[j]):
                m += sub[j]; cnt += 1
        if cnt < 5:
            continue
        m /= cnt
        var = 0.0
        for j in range(window):
            if not np.isnan(sub[j]):
                var += (sub[j] - m) ** 2
        std = np.sqrt(var / cnt)
        if std > 1e-15:
            out[i] = (arr[i] - m) / std
    return out


class TSForecaster:
    """Combine time-series features into a single alpha score per stock.
    
    Parameters
    ----------
    ts_factor_names : list of time-series factor column names
    weights : optional weight for each factor
    """
    
    def __init__(
        self,
        ts_factor_names: Optional[List[str]] = None,
        weights: Optional[List[float]] = None,
    ):
        self.ts_factor_names = ts_factor_names or [
            "KF_VELOCITY", "KF_TREND_STRENGTH",
            "SPECTRAL_ENTROPY_60D", "HURST_60D",
            "ROLLING_SKEW_60D", "ROLLING_KURT_60D",
        ]
        self.weights = weights
    
    def score(self, factor_df: pd.DataFrame) -> pd.DataFrame:
        """Compute per-stock time-series alpha score."""
        df = factor_df.copy()
        available = [f for f in self.ts_factor_names if f in df.columns]
        if not available:
            df["alpha_ts"] = 0.0
            return df[["date", "ticker", "alpha_ts"]]
        
        w = self.weights or [1.0 / len(available)] * len(available)
        w = np.array(w[:len(available)])
        w = w / w.sum()
        
        # Cross-sectional z-score each TS feature, then weighted average
        for fname in available:
            grp = df.groupby("date")[fname]
            mu = grp.transform("mean")
            std = grp.transform("std").replace(0, np.nan)
            df[f"_z_{fname}"] = (df[fname] - mu) / std
        
        z_cols = [f"_z_{f}" for f in available]
        df["alpha_ts"] = np.nanmean(df[z_cols].values * w, axis=1)
        return df[["date", "ticker", "alpha_ts"]]


class MomentumZScoreSignal:
    """Velocity Z-score strategy signal (from TradeX).
    
    Z-score of Kalman velocity: overbought/oversold indicator.
    This is a strategy-level signal, not a base factor.
    """
    
    def __init__(
        self,
        zscore_window: int = 60,
        overbought: float = 2.0,
        oversold: float = -2.0,
    ):
        self.zscore_window = zscore_window
        self.overbought = overbought
        self.oversold = oversold
    
    def compute(self, df: pd.DataFrame, velocity_col: str = "KF_VELOCITY") -> pd.DataFrame:
        """Compute velocity z-score and momentum signal per stock."""
        df = df.copy()
        results = []
        ticker_col = "ticker" if "ticker" in df.columns else "ts_code"
        
        for _, grp in df.groupby(ticker_col):
            g = grp.copy()
            vel = g[velocity_col].values.astype(np.float64)
            z = _rolling_zscore(vel, self.zscore_window)
            g["vel_zscore"] = z
            
            signal = np.zeros(len(z))
            for i in range(len(z)):
                if np.isnan(z[i]):
                    continue
                if z[i] > self.overbought:
                    signal[i] = -1.0
                elif z[i] < self.oversold:
                    signal[i] = 1.0
            g["momentum_signal"] = signal
            results.append(g)
        
        return pd.concat(results, ignore_index=True)
