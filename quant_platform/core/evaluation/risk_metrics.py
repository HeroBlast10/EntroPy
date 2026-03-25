"""Advanced risk metrics: VaR, CVaR (Expected Shortfall), and related measures.

This module implements multiple VaR methodologies:
- **Historical VaR**: empirical quantile from historical returns
- **Parametric VaR**: assumes normal distribution
- **Cornish-Fisher VaR**: adjusts for skewness and kurtosis
- **CVaR (Expected Shortfall)**: mean of losses beyond VaR threshold

VaR and CVaR are widely used in risk management and regulatory frameworks
(e.g., Basel III mandates Expected Shortfall over VaR for market risk).
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from loguru import logger


# ===================================================================
# Historical VaR and CVaR
# ===================================================================

def compute_var(
    returns: pd.Series | np.ndarray,
    confidence: float = 0.95,
) -> float:
    """Compute Historical Value-at-Risk at given confidence level.
    
    VaR is the maximum expected loss over a given time horizon at a given
    confidence level. For example, 95% VaR = 2% means "there is a 5% chance
    of losing more than 2% in a single day."
    
    Parameters
    ----------
    returns : daily returns (can be negative)
    confidence : confidence level (e.g., 0.95 for 95%)
    
    Returns
    -------
    VaR as a positive number (e.g., 0.02 for 2% loss)
    
    Examples
    --------
    >>> returns = pd.Series([-0.03, -0.01, 0.01, 0.02, 0.00])
    >>> var_95 = compute_var(returns, confidence=0.95)
    >>> # 95% VaR is the 5th percentile loss (as positive number)
    """
    if isinstance(returns, pd.Series):
        returns = returns.dropna().values
    
    if len(returns) == 0:
        return np.nan
    
    # VaR is the negative of the (1-confidence) percentile
    # e.g., 95% VaR = -percentile(5%)
    percentile = (1 - confidence) * 100
    var = -np.percentile(returns, percentile)
    
    # Floor at 0: if even the worst-case return is positive, there is no
    # loss risk, so VaR = 0.
    return float(max(0.0, var))


def compute_cvar(
    returns: pd.Series | np.ndarray,
    confidence: float = 0.95,
) -> float:
    """Compute Conditional VaR (CVaR), also known as Expected Shortfall (ES).
    
    CVaR is the expected loss given that the loss exceeds VaR. It is a
    coherent risk measure (unlike VaR) and is preferred by Basel III.
    
    Parameters
    ----------
    returns : daily returns
    confidence : confidence level
    
    Returns
    -------
    CVaR as a positive number
    
    Examples
    --------
    >>> returns = pd.Series([-0.05, -0.03, -0.01, 0.01, 0.02])
    >>> cvar_95 = compute_cvar(returns, confidence=0.95)
    >>> # CVaR is the mean of the worst 5% losses
    """
    if isinstance(returns, pd.Series):
        returns = returns.dropna().values
    
    if len(returns) == 0:
        return np.nan
    
    # Compute VaR threshold
    var = compute_var(returns, confidence)
    
    # CVaR = mean of losses beyond VaR
    # (returns <= -var) selects the tail losses
    tail_losses = returns[returns <= -var]
    
    if len(tail_losses) == 0:
        # No losses beyond VaR (all returns positive)
        return 0.0
    
    cvar = -tail_losses.mean()
    return float(cvar)


# ===================================================================
# Parametric VaR (assumes normal distribution)
# ===================================================================

def compute_parametric_var(
    returns: pd.Series | np.ndarray,
    confidence: float = 0.95,
) -> float:
    """Compute Parametric VaR assuming normal distribution.
    
    Parametric VaR = μ + σ × z_α
    where z_α is the (1-confidence) quantile of standard normal.
    
    This is faster than historical VaR but assumes normality, which often
    fails for financial returns (fat tails, skewness).
    
    Parameters
    ----------
    returns : daily returns
    confidence : confidence level
    
    Returns
    -------
    Parametric VaR as a positive number
    """
    if isinstance(returns, pd.Series):
        returns = returns.dropna().values
    
    if len(returns) == 0:
        return np.nan
    
    mu = np.mean(returns)
    sigma = np.std(returns, ddof=1)
    
    # z-score for (1-confidence) quantile
    # e.g., 95% confidence → 5th percentile → z = -1.645
    z = stats.norm.ppf(1 - confidence)
    
    # VaR = -(μ + σ × z)
    var = -(mu + sigma * z)
    
    return float(var)


# ===================================================================
# Cornish-Fisher VaR (adjusts for skewness and kurtosis)
# ===================================================================

def compute_cornish_fisher_var(
    returns: pd.Series | np.ndarray,
    confidence: float = 0.95,
) -> float:
    """Compute Cornish-Fisher VaR with skewness and kurtosis adjustment.
    
    Cornish-Fisher expansion adjusts the normal quantile for higher moments:
    z_CF = z + (z² - 1) × S/6 + (z³ - 3z) × K/24 - (2z³ - 5z) × S²/36
    
    where S = skewness, K = excess kurtosis.
    
    This is more accurate than parametric VaR for non-normal returns.
    
    Parameters
    ----------
    returns : daily returns
    confidence : confidence level
    
    Returns
    -------
    Cornish-Fisher VaR as a positive number
    
    References
    ----------
    - Cornish & Fisher (1937) "Moments and Cumulants in the Specification of Distributions"
    - Favre & Galeano (2002) "Mean-Modified Value-at-Risk Optimization with Hedge Funds"
    """
    if isinstance(returns, pd.Series):
        returns = returns.dropna().values
    
    if len(returns) == 0:
        return np.nan
    
    mu = np.mean(returns)
    sigma = np.std(returns, ddof=1)
    skew = stats.skew(returns, bias=False)
    kurt = stats.kurtosis(returns, bias=False)  # excess kurtosis
    
    # Normal quantile
    z = stats.norm.ppf(1 - confidence)
    
    # Cornish-Fisher adjustment
    z_cf = (z +
            (z**2 - 1) * skew / 6 +
            (z**3 - 3*z) * kurt / 24 -
            (2*z**3 - 5*z) * skew**2 / 36)
    
    # VaR = -(μ + σ × z_CF)
    var = -(mu + sigma * z_cf)
    
    return float(var)


# ===================================================================
# Rolling VaR and CVaR
# ===================================================================

def compute_rolling_var(
    returns: pd.Series,
    window: int = 252,
    confidence: float = 0.95,
    method: str = "historical",
) -> pd.Series:
    """Compute rolling VaR over time.
    
    Parameters
    ----------
    returns : daily returns (indexed by date)
    window : rolling window size (trading days)
    confidence : confidence level
    method : "historical", "parametric", or "cornish_fisher"
    
    Returns
    -------
    Series of rolling VaR values (indexed by date)
    """
    if method == "historical":
        var_func = compute_var
    elif method == "parametric":
        var_func = compute_parametric_var
    elif method == "cornish_fisher":
        var_func = compute_cornish_fisher_var
    else:
        raise ValueError(f"Unknown method: {method}")
    
    rolling_var = returns.rolling(window, min_periods=window // 2).apply(
        lambda x: var_func(x.values, confidence), raw=False
    )
    
    rolling_var.name = f"VaR_{int(confidence*100)}"
    return rolling_var


def compute_rolling_cvar(
    returns: pd.Series,
    window: int = 252,
    confidence: float = 0.95,
) -> pd.Series:
    """Compute rolling CVaR over time.
    
    Parameters
    ----------
    returns : daily returns (indexed by date)
    window : rolling window size
    confidence : confidence level
    
    Returns
    -------
    Series of rolling CVaR values
    """
    rolling_cvar = returns.rolling(window, min_periods=window // 2).apply(
        lambda x: compute_cvar(x.values, confidence), raw=False
    )
    
    rolling_cvar.name = f"CVaR_{int(confidence*100)}"
    return rolling_cvar


# ===================================================================
# Comprehensive risk summary
# ===================================================================

def risk_metrics_summary(
    returns: pd.Series | np.ndarray,
    confidence: float = 0.95,
) -> Dict[str, float]:
    """Compute all VaR/CVaR metrics in one call.
    
    Parameters
    ----------
    returns : daily returns
    confidence : confidence level
    
    Returns
    -------
    Dict with:
        - var_historical: historical VaR
        - cvar_historical: historical CVaR
        - var_parametric: parametric VaR (normal)
        - var_cornish_fisher: Cornish-Fisher VaR
        - skewness: return distribution skewness
        - kurtosis: return distribution excess kurtosis
    """
    if isinstance(returns, pd.Series):
        returns_clean = returns.dropna()
    else:
        returns_clean = returns[~np.isnan(returns)]
    
    if len(returns_clean) < 10:
        return {
            "var_historical": np.nan,
            "cvar_historical": np.nan,
            "var_parametric": np.nan,
            "var_cornish_fisher": np.nan,
            "skewness": np.nan,
            "kurtosis": np.nan,
        }
    
    var_hist = compute_var(returns_clean, confidence)
    cvar_hist = compute_cvar(returns_clean, confidence)
    var_param = compute_parametric_var(returns_clean, confidence)
    var_cf = compute_cornish_fisher_var(returns_clean, confidence)
    
    skew = float(stats.skew(returns_clean, bias=False))
    kurt = float(stats.kurtosis(returns_clean, bias=False))
    
    return {
        "var_historical": round(var_hist, 6),
        "cvar_historical": round(cvar_hist, 6),
        "var_parametric": round(var_param, 6),
        "var_cornish_fisher": round(var_cf, 6),
        "skewness": round(skew, 4),
        "kurtosis": round(kurt, 4),
    }


# ===================================================================
# VaR backtesting (violations)
# ===================================================================

def var_backtest(
    returns: pd.Series,
    var_series: pd.Series,
    confidence: float = 0.95,
) -> Dict[str, float]:
    """Backtest VaR model by counting violations.
    
    A violation occurs when the actual loss exceeds VaR. Under a correct
    model, violations should occur (1-confidence)% of the time.
    
    Parameters
    ----------
    returns : actual daily returns
    var_series : VaR estimates (aligned to same dates)
    confidence : confidence level used for VaR
    
    Returns
    -------
    Dict with:
        - n_violations: number of times loss > VaR
        - violation_rate: fraction of violations
        - expected_rate: theoretical violation rate (1-confidence)
        - kupiec_pvalue: p-value from Kupiec test (null: model is correct)
    
    References
    ----------
    - Kupiec (1995) "Techniques for Verifying the Accuracy of Risk Measurement Models"
    """
    # Align
    df = pd.DataFrame({"ret": returns, "var": var_series}).dropna()
    
    if len(df) < 10:
        return {
            "n_violations": np.nan,
            "violation_rate": np.nan,
            "expected_rate": 1 - confidence,
            "kupiec_pvalue": np.nan,
        }
    
    # Violation = actual loss > VaR
    violations = (df["ret"] < -df["var"]).sum()
    n = len(df)
    violation_rate = violations / n
    expected_rate = 1 - confidence
    
    # Kupiec test (likelihood ratio test)
    # H0: violation_rate = expected_rate
    if violations == 0 or violations == n:
        kupiec_pvalue = np.nan
    else:
        lr = -2 * (
            violations * np.log(expected_rate) +
            (n - violations) * np.log(1 - expected_rate) -
            violations * np.log(violation_rate) -
            (n - violations) * np.log(1 - violation_rate)
        )
        kupiec_pvalue = 1 - stats.chi2.cdf(lr, df=1)
    
    return {
        "n_violations": int(violations),
        "violation_rate": round(violation_rate, 4),
        "expected_rate": round(expected_rate, 4),
        "kupiec_pvalue": round(kupiec_pvalue, 4) if not np.isnan(kupiec_pvalue) else np.nan,
    }
