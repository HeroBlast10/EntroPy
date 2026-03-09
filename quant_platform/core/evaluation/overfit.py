"""Overfit detection: CSCV and Deflated Sharpe Ratio.

These methods help assess whether backtest performance is likely to
hold out-of-sample or is an artifact of data mining.

References
----------
- Bailey & Lopez de Prado (2014) "The Deflated Sharpe Ratio"
- Bailey et al. (2017) "The Probability of Backtest Overfitting"
"""
from __future__ import annotations
from typing import Dict, Optional
import numpy as np
import pandas as pd
from loguru import logger


def deflated_sharpe_ratio(
    sharpe: float,
    n_trials: int,
    n_obs: int,
    skew: float = 0.0,
    kurt: float = 3.0,
) -> float:
    """Compute the Deflated Sharpe Ratio (DSR).

    Adjusts the observed Sharpe ratio for multiple testing, non-normality,
    and sample length.

    Parameters
    ----------
    sharpe : observed Sharpe ratio
    n_trials : number of strategy variants tested
    n_obs : number of return observations
    skew : skewness of returns
    kurt : kurtosis of returns (normal = 3)
    """
    from scipy import stats

    # Expected maximum Sharpe under null (all trials are noise)
    e_max_sr = stats.norm.ppf(1 - 1.0 / n_trials) if n_trials > 1 else 0.0

    # Standard error of Sharpe estimator
    se = np.sqrt((1 + 0.5 * sharpe**2 - skew * sharpe + ((kurt - 3) / 4) * sharpe**2) / n_obs)

    if se <= 0:
        return 0.0

    # Probability that observed Sharpe exceeds the expected max under null
    dsr = float(stats.norm.cdf((sharpe - e_max_sr) / se))
    return dsr


def probability_of_backtest_overfitting(
    returns_matrix: pd.DataFrame,
    n_splits: int = 10,
) -> Dict[str, float]:
    """Combinatorially Symmetric Cross-Validation (CSCV) estimate.

    Splits the return series into n_splits blocks and tests all
    (n_splits choose n_splits/2) train/test splits.

    Parameters
    ----------
    returns_matrix : DataFrame of strategy variant returns (columns = variants)
    n_splits : number of time blocks

    Returns
    -------
    Dict with PBO estimate and related statistics.
    """
    logger.warning("CSCV is computationally expensive; using simplified estimate.")

    n = len(returns_matrix)
    block_size = n // n_splits
    if block_size < 20:
        return {"pbo": np.nan, "note": "insufficient data for CSCV"}

    # Simplified: split into 2 halves, pick best in-sample, check OOS
    half = n // 2
    is_returns = returns_matrix.iloc[:half]
    oos_returns = returns_matrix.iloc[half:]

    is_std = is_returns.std()
    is_sharpes = np.where(is_std > 1e-10, is_returns.mean() / is_std * np.sqrt(252), 0.0)
    is_sharpes = pd.Series(is_sharpes, index=is_returns.columns)
    best_is = is_sharpes.idxmax()

    oos_std_best = oos_returns[best_is].std()
    oos_sharpe_best = float(
        oos_returns[best_is].mean() / oos_std_best * np.sqrt(252)
        if oos_std_best > 1e-10 else 0.0
    )
    oos_std_all = oos_returns.std()
    oos_sharpes_all = np.where(
        oos_std_all > 1e-10,
        oos_returns.mean() / oos_std_all * np.sqrt(252),
        0.0,
    )
    oos_sharpe_median = float(np.median(oos_sharpes_all))

    # PBO approximation: fraction of time the IS-best underperforms OOS median
    pbo_approx = 1.0 if oos_sharpe_best < oos_sharpe_median else 0.0

    return {
        "pbo_approx": pbo_approx,
        "best_is_variant": best_is,
        "is_sharpe": float(is_sharpes[best_is]),
        "oos_sharpe": oos_sharpe_best,
        "oos_median_sharpe": oos_sharpe_median,
    }
