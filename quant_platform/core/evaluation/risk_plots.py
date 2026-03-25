"""Plotting functions for VaR/CVaR risk metrics.

Generates charts for:
- Rolling VaR and CVaR over time
- VaR comparison (Historical vs Parametric vs Cornish-Fisher)
- VaR violations (backtest)
- Return distribution with VaR/CVaR thresholds
"""

from __future__ import annotations

from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from quant_platform.core.evaluation.risk_metrics import (
    compute_rolling_var,
    compute_rolling_cvar,
    compute_var,
    compute_cvar,
    var_backtest,
)


# ===================================================================
# Rolling VaR and CVaR
# ===================================================================

def plot_rolling_var_cvar(
    returns: pd.Series,
    window: int = 252,
    confidence: float = 0.95,
    figsize: Tuple[int, int] = (12, 6),
) -> plt.Figure:
    """Plot rolling VaR and CVaR over time.
    
    Parameters
    ----------
    returns : daily returns (indexed by date)
    window : rolling window size (trading days)
    confidence : confidence level (e.g., 0.95 for 95%)
    figsize : figure size
    
    Returns
    -------
    matplotlib Figure
    """
    # Compute rolling metrics
    rolling_var = compute_rolling_var(returns, window, confidence, method="historical")
    rolling_cvar = compute_rolling_cvar(returns, window, confidence)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot
    ax.plot(rolling_var.index, rolling_var.values, 
            label=f"VaR ({int(confidence*100)}%)", linewidth=2, color="#e74c3c")
    ax.plot(rolling_cvar.index, rolling_cvar.values, 
            label=f"CVaR ({int(confidence*100)}%)", linewidth=2, color="#c0392b")
    
    # Fill between
    ax.fill_between(rolling_var.index, rolling_var.values, rolling_cvar.values,
                     alpha=0.2, color="#e74c3c")
    
    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("Risk (daily)", fontsize=11)
    ax.set_title(f"Rolling {window}-Day VaR and CVaR", fontsize=13, fontweight="bold")
    ax.legend(loc="upper left", framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x:.1%}"))
    
    fig.tight_layout()
    return fig


# ===================================================================
# VaR method comparison
# ===================================================================

def plot_var_comparison(
    returns: pd.Series,
    window: int = 252,
    confidence: float = 0.95,
    figsize: Tuple[int, int] = (12, 6),
) -> plt.Figure:
    """Compare Historical, Parametric, and Cornish-Fisher VaR.
    
    Parameters
    ----------
    returns : daily returns
    window : rolling window size
    confidence : confidence level
    figsize : figure size
    
    Returns
    -------
    matplotlib Figure
    """
    # Compute rolling VaR with different methods
    var_hist = compute_rolling_var(returns, window, confidence, method="historical")
    var_param = compute_rolling_var(returns, window, confidence, method="parametric")
    var_cf = compute_rolling_var(returns, window, confidence, method="cornish_fisher")
    
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.plot(var_hist.index, var_hist.values, 
            label="Historical VaR", linewidth=2, color="#3498db")
    ax.plot(var_param.index, var_param.values, 
            label="Parametric VaR (Normal)", linewidth=2, color="#95a5a6", linestyle="--")
    ax.plot(var_cf.index, var_cf.values, 
            label="Cornish-Fisher VaR", linewidth=2, color="#9b59b6", linestyle="-.")
    
    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("VaR (daily)", fontsize=11)
    ax.set_title(f"VaR Comparison ({int(confidence*100)}% Confidence)", 
                 fontsize=13, fontweight="bold")
    ax.legend(loc="upper left", framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x:.1%}"))
    
    fig.tight_layout()
    return fig


# ===================================================================
# Return distribution with VaR/CVaR thresholds
# ===================================================================

def plot_return_distribution(
    returns: pd.Series,
    confidence: float = 0.95,
    bins: int = 50,
    figsize: Tuple[int, int] = (10, 6),
) -> plt.Figure:
    """Plot return distribution histogram with VaR and CVaR thresholds.
    
    Parameters
    ----------
    returns : daily returns
    confidence : confidence level
    bins : number of histogram bins
    figsize : figure size
    
    Returns
    -------
    matplotlib Figure
    """
    returns_clean = returns.dropna()
    
    # Compute VaR and CVaR
    var = compute_var(returns_clean, confidence)
    cvar = compute_cvar(returns_clean, confidence)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Histogram
    n, bins_edges, patches = ax.hist(returns_clean, bins=bins, 
                                      color="#3498db", alpha=0.6, edgecolor="black")
    
    # Color the tail (losses beyond VaR) in red
    for i, patch in enumerate(patches):
        if bins_edges[i] < -var:
            patch.set_facecolor("#e74c3c")
            patch.set_alpha(0.8)
    
    # Add VaR and CVaR lines
    ax.axvline(-var, color="#e74c3c", linewidth=2, linestyle="--", 
               label=f"VaR ({int(confidence*100)}%): {var:.2%}")
    ax.axvline(-cvar, color="#c0392b", linewidth=2, linestyle="--", 
               label=f"CVaR ({int(confidence*100)}%): {cvar:.2%}")
    
    ax.set_xlabel("Daily Return", fontsize=11)
    ax.set_ylabel("Frequency", fontsize=11)
    ax.set_title("Return Distribution with VaR/CVaR Thresholds", 
                 fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(True, alpha=0.3, axis="y")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x:.1%}"))
    
    fig.tight_layout()
    return fig


# ===================================================================
# VaR violations (backtest)
# ===================================================================

def plot_var_violations(
    returns: pd.Series,
    window: int = 252,
    confidence: float = 0.95,
    figsize: Tuple[int, int] = (12, 6),
) -> plt.Figure:
    """Plot actual returns vs VaR threshold, highlighting violations.
    
    Parameters
    ----------
    returns : daily returns
    window : rolling window for VaR calculation
    confidence : confidence level
    figsize : figure size
    
    Returns
    -------
    matplotlib Figure
    """
    # Compute rolling VaR
    rolling_var = compute_rolling_var(returns, window, confidence, method="historical")
    
    # Align
    df = pd.DataFrame({
        "ret": returns,
        "var": rolling_var,
    }).dropna()
    
    # Identify violations
    df["violation"] = df["ret"] < -df["var"]
    
    # Backtest statistics
    backtest_stats = var_backtest(df["ret"], df["var"], confidence)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot returns
    ax.plot(df.index, df["ret"], linewidth=1, color="#3498db", alpha=0.6, label="Daily Return")
    
    # Plot VaR threshold
    ax.plot(df.index, -df["var"], linewidth=2, color="#e74c3c", linestyle="--", 
            label=f"VaR ({int(confidence*100)}%)")
    
    # Highlight violations
    violations = df[df["violation"]]
    if len(violations) > 0:
        ax.scatter(violations.index, violations["ret"], color="#c0392b", 
                   s=50, zorder=5, label=f"Violations ({len(violations)})")
    
    ax.axhline(0, color="black", linewidth=0.8, alpha=0.5)
    
    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("Return", fontsize=11)
    ax.set_title(f"VaR Violations (Expected: {backtest_stats['expected_rate']:.1%}, "
                 f"Actual: {backtest_stats['violation_rate']:.1%})", 
                 fontsize=13, fontweight="bold")
    ax.legend(loc="lower left", framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x:.1%}"))
    
    fig.tight_layout()
    return fig


# ===================================================================
# VaR/CVaR summary panel
# ===================================================================

def plot_var_summary_panel(
    returns: pd.Series,
    window: int = 252,
    confidence: float = 0.95,
    figsize: Tuple[int, int] = (14, 10),
) -> plt.Figure:
    """Create a 2x2 panel of VaR/CVaR visualizations.
    
    Parameters
    ----------
    returns : daily returns
    window : rolling window size
    confidence : confidence level
    figsize : figure size
    
    Returns
    -------
    matplotlib Figure with 4 subplots
    """
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    # 1. Rolling VaR/CVaR
    ax1 = fig.add_subplot(gs[0, 0])
    rolling_var = compute_rolling_var(returns, window, confidence, method="historical")
    rolling_cvar = compute_rolling_cvar(returns, window, confidence)
    ax1.plot(rolling_var.index, rolling_var.values, label="VaR", linewidth=2, color="#e74c3c")
    ax1.plot(rolling_cvar.index, rolling_cvar.values, label="CVaR", linewidth=2, color="#c0392b")
    ax1.fill_between(rolling_var.index, rolling_var.values, rolling_cvar.values, alpha=0.2, color="#e74c3c")
    ax1.set_title(f"Rolling {window}-Day VaR/CVaR", fontweight="bold")
    ax1.set_ylabel("Risk (daily)")
    ax1.legend(loc="upper left", framealpha=0.9)
    ax1.grid(True, alpha=0.3)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x:.1%}"))
    
    # 2. VaR method comparison
    ax2 = fig.add_subplot(gs[0, 1])
    var_hist = compute_rolling_var(returns, window, confidence, method="historical")
    var_param = compute_rolling_var(returns, window, confidence, method="parametric")
    var_cf = compute_rolling_var(returns, window, confidence, method="cornish_fisher")
    ax2.plot(var_hist.index, var_hist.values, label="Historical", linewidth=2, color="#3498db")
    ax2.plot(var_param.index, var_param.values, label="Parametric", linewidth=2, 
             color="#95a5a6", linestyle="--")
    ax2.plot(var_cf.index, var_cf.values, label="Cornish-Fisher", linewidth=2, 
             color="#9b59b6", linestyle="-.")
    ax2.set_title("VaR Method Comparison", fontweight="bold")
    ax2.set_ylabel("VaR (daily)")
    ax2.legend(loc="upper left", framealpha=0.9)
    ax2.grid(True, alpha=0.3)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x:.1%}"))
    
    # 3. Return distribution
    ax3 = fig.add_subplot(gs[1, 0])
    returns_clean = returns.dropna()
    var = compute_var(returns_clean, confidence)
    cvar = compute_cvar(returns_clean, confidence)
    n, bins_edges, patches = ax3.hist(returns_clean, bins=50, color="#3498db", 
                                       alpha=0.6, edgecolor="black")
    for i, patch in enumerate(patches):
        if bins_edges[i] < -var:
            patch.set_facecolor("#e74c3c")
            patch.set_alpha(0.8)
    ax3.axvline(-var, color="#e74c3c", linewidth=2, linestyle="--", label=f"VaR: {var:.2%}")
    ax3.axvline(-cvar, color="#c0392b", linewidth=2, linestyle="--", label=f"CVaR: {cvar:.2%}")
    ax3.set_title("Return Distribution", fontweight="bold")
    ax3.set_xlabel("Daily Return")
    ax3.set_ylabel("Frequency")
    ax3.legend(loc="upper right", framealpha=0.9)
    ax3.grid(True, alpha=0.3, axis="y")
    ax3.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x:.1%}"))
    
    # 4. VaR violations
    ax4 = fig.add_subplot(gs[1, 1])
    df = pd.DataFrame({"ret": returns, "var": rolling_var}).dropna()
    df["violation"] = df["ret"] < -df["var"]
    ax4.plot(df.index, df["ret"], linewidth=1, color="#3498db", alpha=0.6, label="Return")
    ax4.plot(df.index, -df["var"], linewidth=2, color="#e74c3c", linestyle="--", label="VaR")
    violations = df[df["violation"]]
    if len(violations) > 0:
        ax4.scatter(violations.index, violations["ret"], color="#c0392b", s=30, 
                    zorder=5, label=f"Violations ({len(violations)})")
    ax4.axhline(0, color="black", linewidth=0.8, alpha=0.5)
    backtest_stats = var_backtest(df["ret"], df["var"], confidence)
    ax4.set_title(f"VaR Violations (Actual: {backtest_stats['violation_rate']:.1%})", 
                  fontweight="bold")
    ax4.set_xlabel("Date")
    ax4.set_ylabel("Return")
    ax4.legend(loc="lower left", framealpha=0.9, fontsize=9)
    ax4.grid(True, alpha=0.3)
    ax4.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x:.1%}"))
    
    return fig
