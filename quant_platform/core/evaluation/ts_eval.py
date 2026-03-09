"""Evaluation metrics for time-series signals.

Unlike cross-sectional IC, time-series signals are evaluated on:
- Directional accuracy (did the signal predict return direction?)
- Hit rate conditioned on signal strength
- Signal-return correlation per stock
"""
from __future__ import annotations
from typing import Dict, Optional
import numpy as np
import pandas as pd
from loguru import logger


def directional_accuracy(
    signal: pd.Series,
    returns: pd.Series,
    lag: int = 1,
) -> float:
    """Fraction of times signal correctly predicted return direction."""
    sig = signal.shift(lag)
    aligned = pd.DataFrame({"signal": sig, "ret": returns}).dropna()
    if len(aligned) < 10:
        return np.nan
    correct = (np.sign(aligned["signal"]) == np.sign(aligned["ret"])).mean()
    return float(correct)


def conditional_hit_rate(
    signal: pd.Series,
    returns: pd.Series,
    threshold: float = 1.0,
    lag: int = 1,
) -> Dict[str, float]:
    """Hit rate when signal exceeds threshold."""
    sig = signal.shift(lag)
    aligned = pd.DataFrame({"signal": sig, "ret": returns}).dropna()

    strong_long = aligned[aligned["signal"] > threshold]
    strong_short = aligned[aligned["signal"] < -threshold]

    return {
        "strong_long_hit": float((strong_long["ret"] > 0).mean()) if len(strong_long) > 5 else np.nan,
        "strong_short_hit": float((strong_short["ret"] < 0).mean()) if len(strong_short) > 5 else np.nan,
        "n_strong_long": len(strong_long),
        "n_strong_short": len(strong_short),
    }


def per_stock_signal_corr(
    factor_df: pd.DataFrame,
    signal_col: str,
    return_col: str = "fwd_ret_1d",
    ticker_col: str = "ticker",
) -> pd.Series:
    """Compute signal-return correlation per stock (time-series correlation)."""
    corrs = {}
    for ticker, grp in factor_df.groupby(ticker_col):
        valid = grp[[signal_col, return_col]].dropna()
        if len(valid) < 30:
            continue
        corrs[ticker] = valid[signal_col].corr(valid[return_col])
    return pd.Series(corrs, name=f"ts_corr_{signal_col}")


def ts_signal_summary(
    factor_df: pd.DataFrame,
    signal_col: str,
    return_col: str = "fwd_ret_1d",
    ticker_col: str = "ticker",
) -> Dict[str, float]:
    """Comprehensive time-series signal evaluation."""
    corrs = per_stock_signal_corr(factor_df, signal_col, return_col, ticker_col)

    return {
        "mean_ts_corr": float(corrs.mean()) if len(corrs) > 0 else np.nan,
        "median_ts_corr": float(corrs.median()) if len(corrs) > 0 else np.nan,
        "pct_positive_corr": float((corrs > 0).mean()) if len(corrs) > 0 else np.nan,
        "n_stocks": len(corrs),
    }
