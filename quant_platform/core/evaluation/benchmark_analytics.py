"""Benchmark comparison and alpha/beta decomposition analytics.

This module provides functions to analyze portfolio performance relative to
a benchmark, including:

- Active return and tracking error
- Information Ratio
- CAPM alpha and beta (via linear regression)
- Treynor Ratio
- Rolling alpha and beta
- Alpha/beta/residual return decomposition

All functions assume excess returns (portfolio - risk_free_rate) for proper
CAPM analysis. If risk_free_rate is not provided, it defaults to 0.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger
from scipy import stats


# ===================================================================
# Active return metrics
# ===================================================================

def compute_active_return(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
) -> pd.Series:
    """Compute daily active return (portfolio - benchmark).
    
    Parameters
    ----------
    portfolio_returns : daily portfolio returns
    benchmark_returns : daily benchmark returns (aligned to same dates)
    
    Returns
    -------
    Series of active returns
    """
    return portfolio_returns - benchmark_returns


def compute_tracking_error(
    active_returns: pd.Series,
    annualization: int = 252,
) -> float:
    """Compute annualized tracking error (std of active returns).
    
    Parameters
    ----------
    active_returns : daily active returns
    annualization : number of periods per year (252 for daily)
    
    Returns
    -------
    Annualized tracking error
    """
    return float(active_returns.std() * np.sqrt(annualization))


def compute_information_ratio(
    active_returns: pd.Series,
    annualization: int = 252,
) -> float:
    """Compute Information Ratio = annualized(active return) / tracking error.
    
    IR measures risk-adjusted active return. Higher is better.
    
    Parameters
    ----------
    active_returns : daily active returns
    annualization : number of periods per year
    
    Returns
    -------
    Information Ratio
    """
    mean_active = active_returns.mean()
    std_active = active_returns.std()
    
    if std_active == 0:
        return np.nan
    
    # Annualize mean and std
    ann_mean = mean_active * annualization
    ann_std = std_active * np.sqrt(annualization)
    
    return float(ann_mean / ann_std)


# ===================================================================
# CAPM alpha and beta
# ===================================================================

def compute_capm_alpha_beta(
    portfolio_excess_returns: pd.Series,
    benchmark_excess_returns: pd.Series,
    annualization: int = 252,
) -> Dict[str, float]:
    """Compute CAPM alpha and beta via linear regression.
    
    Model: R_p - R_f = alpha + beta * (R_m - R_f) + epsilon
    
    Parameters
    ----------
    portfolio_excess_returns : portfolio return - risk_free_rate
    benchmark_excess_returns : benchmark return - risk_free_rate
    annualization : periods per year (for annualizing alpha)
    
    Returns
    -------
    Dict with keys:
        - alpha: annualized CAPM alpha
        - beta: CAPM beta
        - r_squared: R² of regression
        - alpha_tstat: t-statistic for alpha
        - alpha_pvalue: p-value for alpha
        - residual_vol: annualized volatility of residuals
    """
    # Align and drop NaNs
    df = pd.DataFrame({
        "port": portfolio_excess_returns,
        "bench": benchmark_excess_returns,
    }).dropna()
    
    if len(df) < 10:
        return {
            "alpha": np.nan,
            "beta": np.nan,
            "r_squared": np.nan,
            "alpha_tstat": np.nan,
            "alpha_pvalue": np.nan,
            "residual_vol": np.nan,
        }
    
    # Linear regression: port = alpha + beta * bench + epsilon
    X = df["bench"].values
    y = df["port"].values
    
    # Add intercept
    X_with_const = np.column_stack([np.ones(len(X)), X])
    
    # OLS via normal equation: (X'X)^-1 X'y
    try:
        params = np.linalg.lstsq(X_with_const, y, rcond=None)[0]
    except np.linalg.LinAlgError:
        logger.warning("CAPM regression failed (singular matrix)")
        return {
            "alpha": np.nan,
            "beta": np.nan,
            "r_squared": np.nan,
            "alpha_tstat": np.nan,
            "alpha_pvalue": np.nan,
            "residual_vol": np.nan,
        }
    
    alpha_daily = params[0]
    beta = params[1]
    
    # Residuals
    y_pred = X_with_const @ params
    residuals = y - y_pred
    
    # R-squared
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    
    # Standard error of alpha (for t-stat)
    n = len(y)
    residual_var = ss_res / (n - 2)  # degrees of freedom = n - 2
    X_var = np.var(X, ddof=1)
    if X_var == 0:
        se_alpha = np.nan
    else:
        se_alpha = np.sqrt(residual_var * (1/n + X.mean()**2 / (n * X_var)))
    
    alpha_tstat = alpha_daily / se_alpha if se_alpha > 0 else np.nan
    alpha_pvalue = 2 * (1 - stats.t.cdf(abs(alpha_tstat), df=n-2)) if not np.isnan(alpha_tstat) else np.nan
    
    # Annualize alpha
    alpha_annual = alpha_daily * annualization
    
    # Residual volatility (annualized)
    residual_vol = np.std(residuals) * np.sqrt(annualization)
    
    return {
        "alpha": round(alpha_annual, 6),
        "beta": round(beta, 4),
        "r_squared": round(r_squared, 4),
        "alpha_tstat": round(alpha_tstat, 4),
        "alpha_pvalue": round(alpha_pvalue, 6),
        "residual_vol": round(residual_vol, 6),
    }


# ===================================================================
# Treynor Ratio
# ===================================================================

def compute_treynor_ratio(
    portfolio_excess_returns: pd.Series,
    beta: float,
    annualization: int = 252,
) -> float:
    """Compute Treynor Ratio = annualized(excess return) / beta.
    
    Measures return per unit of systematic risk (beta).
    
    Parameters
    ----------
    portfolio_excess_returns : portfolio return - risk_free_rate
    beta : CAPM beta
    annualization : periods per year
    
    Returns
    -------
    Treynor Ratio
    """
    if beta == 0 or np.isnan(beta):
        return np.nan
    
    mean_excess = portfolio_excess_returns.mean() * annualization
    return float(mean_excess / beta)


# ===================================================================
# Rolling alpha and beta
# ===================================================================

def compute_rolling_alpha_beta(
    portfolio_excess_returns: pd.Series,
    benchmark_excess_returns: pd.Series,
    window: int = 252,
    annualization: int = 252,
) -> pd.DataFrame:
    """Compute rolling CAPM alpha and beta.
    
    Parameters
    ----------
    portfolio_excess_returns : portfolio return - risk_free_rate
    benchmark_excess_returns : benchmark return - risk_free_rate
    window : rolling window size (trading days)
    annualization : periods per year
    
    Returns
    -------
    DataFrame with columns [date, alpha, beta, r_squared]
    """
    df = pd.DataFrame({
        "port": portfolio_excess_returns,
        "bench": benchmark_excess_returns,
    }).dropna()
    
    if len(df) < window:
        logger.warning("Insufficient data for rolling alpha/beta (need {} days, have {})", window, len(df))
        return pd.DataFrame(columns=["date", "alpha", "beta", "r_squared"])
    
    results = []
    
    for i in range(window - 1, len(df)):
        window_data = df.iloc[i - window + 1:i + 1]
        date = df.index[i]
        
        X = window_data["bench"].values
        y = window_data["port"].values
        
        # Regression
        X_with_const = np.column_stack([np.ones(len(X)), X])
        try:
            params = np.linalg.lstsq(X_with_const, y, rcond=None)[0]
            alpha_daily = params[0]
            beta = params[1]
            
            # R-squared
            y_pred = X_with_const @ params
            residuals = y - y_pred
            ss_res = np.sum(residuals ** 2)
            ss_tot = np.sum((y - y.mean()) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
            
            results.append({
                "date": date,
                "alpha": alpha_daily * annualization,
                "beta": beta,
                "r_squared": r_squared,
            })
        except np.linalg.LinAlgError:
            continue
    
    return pd.DataFrame(results)


# ===================================================================
# Alpha/beta/residual decomposition
# ===================================================================

def decompose_return(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
    risk_free_rate: float = 0.0,
    annualization: int = 252,
) -> Dict[str, float]:
    """Decompose portfolio return into alpha, beta, and residual components.
    
    Total Return = Risk-Free + Beta * (Benchmark - Risk-Free) + Alpha + Residual
    
    Parameters
    ----------
    portfolio_returns : daily portfolio returns
    benchmark_returns : daily benchmark returns
    risk_free_rate : annualized risk-free rate (e.g., 0.03 for 3%)
    annualization : periods per year
    
    Returns
    -------
    Dict with:
        - total_return: annualized portfolio return
        - benchmark_return: annualized benchmark return
        - risk_free_return: risk-free rate
        - beta_contribution: beta * (benchmark - rf)
        - alpha_contribution: CAPM alpha
        - residual_contribution: unexplained component
        - active_return: total - benchmark
    """
    # Convert annual risk-free rate to daily
    rf_daily = (1 + risk_free_rate) ** (1 / annualization) - 1
    
    # Excess returns
    port_excess = portfolio_returns - rf_daily
    bench_excess = benchmark_returns - rf_daily
    
    # CAPM regression
    capm = compute_capm_alpha_beta(port_excess, bench_excess, annualization)
    alpha = capm["alpha"]
    beta = capm["beta"]
    
    # Annualized returns
    total_ret = (1 + portfolio_returns).prod() ** (annualization / len(portfolio_returns)) - 1
    bench_ret = (1 + benchmark_returns).prod() ** (annualization / len(benchmark_returns)) - 1
    
    # Beta contribution = beta * (benchmark - rf)
    beta_contrib = beta * (bench_ret - risk_free_rate)
    
    # Residual = total - rf - beta_contrib - alpha
    residual = total_ret - risk_free_rate - beta_contrib - alpha
    
    return {
        "total_return": round(total_ret, 6),
        "benchmark_return": round(bench_ret, 6),
        "risk_free_return": round(risk_free_rate, 6),
        "beta_contribution": round(beta_contrib, 6),
        "alpha_contribution": round(alpha, 6),
        "residual_contribution": round(residual, 6),
        "active_return": round(total_ret - bench_ret, 6),
    }


# ===================================================================
# Comprehensive benchmark analysis
# ===================================================================

def benchmark_analysis(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
    risk_free_rate: float = 0.0,
    annualization: int = 252,
) -> Dict[str, float]:
    """Comprehensive benchmark comparison analysis.
    
    Returns all key metrics in one call.
    
    Parameters
    ----------
    portfolio_returns : daily portfolio returns
    benchmark_returns : daily benchmark returns (aligned)
    risk_free_rate : annualized risk-free rate
    annualization : periods per year
    
    Returns
    -------
    Dict with all benchmark-relative metrics
    """
    # Align
    df = pd.DataFrame({
        "port": portfolio_returns,
        "bench": benchmark_returns,
    }).dropna()
    
    if len(df) < 10:
        logger.warning("Insufficient data for benchmark analysis")
        return {}
    
    port = df["port"]
    bench = df["bench"]
    
    # Active return
    active = compute_active_return(port, bench)
    
    # Excess returns
    rf_daily = (1 + risk_free_rate) ** (1 / annualization) - 1
    port_excess = port - rf_daily
    bench_excess = bench - rf_daily
    
    # CAPM
    capm = compute_capm_alpha_beta(port_excess, bench_excess, annualization)
    
    # Treynor
    treynor = compute_treynor_ratio(port_excess, capm["beta"], annualization)
    
    # Tracking error and IR
    tracking_error = compute_tracking_error(active, annualization)
    info_ratio = compute_information_ratio(active, annualization)
    
    # Decomposition
    decomp = decompose_return(port, bench, risk_free_rate, annualization)
    
    # Combine all
    result = {
        # Active metrics
        "active_return_ann": round(active.mean() * annualization, 6),
        "tracking_error": round(tracking_error, 6),
        "information_ratio": round(info_ratio, 4),
        # CAPM
        "alpha": capm["alpha"],
        "beta": capm["beta"],
        "r_squared": capm["r_squared"],
        "alpha_tstat": capm["alpha_tstat"],
        "alpha_pvalue": capm["alpha_pvalue"],
        "residual_vol": capm["residual_vol"],
        # Treynor
        "treynor_ratio": round(treynor, 4),
        # Decomposition
        "total_return": decomp["total_return"],
        "benchmark_return": decomp["benchmark_return"],
        "beta_contribution": decomp["beta_contribution"],
        "alpha_contribution": decomp["alpha_contribution"],
        "residual_contribution": decomp["residual_contribution"],
    }
    
    logger.info(
        "Benchmark analysis: IR={:.2f}, Alpha={:.2%}, Beta={:.2f}, R²={:.2%}",
        result["information_ratio"],
        result["alpha"],
        result["beta"],
        result["r_squared"],
    )
    
    return result
