"""Plotting functions for benchmark comparison analysis.

Generates charts for:
- NAV curve with benchmark overlay
- Rolling alpha and beta
- Active return decomposition
- Tracking error over time
"""

from __future__ import annotations

from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from quant_platform.core.evaluation.benchmark_analytics import (
    compute_active_return,
    compute_rolling_alpha_beta,
)


# ===================================================================
# NAV with benchmark overlay
# ===================================================================

def plot_nav_with_benchmark(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
    initial_capital: float = 1_000_000.0,
    figsize: Tuple[int, int] = (12, 6),
) -> plt.Figure:
    """Plot portfolio NAV with benchmark overlay.
    
    Parameters
    ----------
    portfolio_returns : daily portfolio returns (indexed by date)
    benchmark_returns : daily benchmark returns (indexed by date)
    initial_capital : starting NAV
    figsize : figure size
    
    Returns
    -------
    matplotlib Figure
    """
    # Align
    df = pd.DataFrame({
        "port": portfolio_returns,
        "bench": benchmark_returns,
    }).dropna()
    
    # Compute NAV
    port_nav = initial_capital * (1 + df["port"]).cumprod()
    bench_nav = initial_capital * (1 + df["bench"]).cumprod()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.plot(port_nav.index, port_nav.values, label="Portfolio", linewidth=2, color="#2c3e50")
    ax.plot(bench_nav.index, bench_nav.values, label="Benchmark", linewidth=2, 
            color="#95a5a6", linestyle="--", alpha=0.8)
    
    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("NAV ($)", fontsize=11)
    ax.set_title("Portfolio vs Benchmark NAV", fontsize=13, fontweight="bold")
    ax.legend(loc="upper left", framealpha=0.9)
    ax.grid(True, alpha=0.3)
    
    # Format y-axis as currency
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"${x:,.0f}"))
    
    fig.tight_layout()
    return fig


# ===================================================================
# Rolling alpha and beta
# ===================================================================

def plot_rolling_alpha_beta(
    portfolio_excess_returns: pd.Series,
    benchmark_excess_returns: pd.Series,
    window: int = 252,
    annualization: int = 252,
    figsize: Tuple[int, int] = (12, 8),
) -> plt.Figure:
    """Plot rolling CAPM alpha and beta.
    
    Parameters
    ----------
    portfolio_excess_returns : portfolio return - risk_free_rate
    benchmark_excess_returns : benchmark return - risk_free_rate
    window : rolling window size (trading days)
    annualization : periods per year
    figsize : figure size
    
    Returns
    -------
    matplotlib Figure
    """
    rolling = compute_rolling_alpha_beta(
        portfolio_excess_returns,
        benchmark_excess_returns,
        window=window,
        annualization=annualization,
    )
    
    if rolling.empty:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "Insufficient data for rolling alpha/beta",
                ha="center", va="center", fontsize=12)
        return fig
    
    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)
    
    # Alpha
    axes[0].plot(rolling["date"], rolling["alpha"], linewidth=2, color="#e74c3c")
    axes[0].axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    axes[0].set_ylabel("Alpha (ann.)", fontsize=11)
    axes[0].set_title(f"Rolling {window}-Day Alpha", fontsize=12, fontweight="bold")
    axes[0].grid(True, alpha=0.3)
    axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x:.1%}"))
    
    # Beta
    axes[1].plot(rolling["date"], rolling["beta"], linewidth=2, color="#3498db")
    axes[1].axhline(1.0, color="black", linewidth=0.8, linestyle="--", alpha=0.5, label="Beta = 1")
    axes[1].set_ylabel("Beta", fontsize=11)
    axes[1].set_title(f"Rolling {window}-Day Beta", fontsize=12, fontweight="bold")
    axes[1].legend(loc="upper right", framealpha=0.9)
    axes[1].grid(True, alpha=0.3)
    
    # R-squared
    axes[2].plot(rolling["date"], rolling["r_squared"], linewidth=2, color="#9b59b6")
    axes[2].set_ylabel("R²", fontsize=11)
    axes[2].set_xlabel("Date", fontsize=11)
    axes[2].set_title(f"Rolling {window}-Day R²", fontsize=12, fontweight="bold")
    axes[2].grid(True, alpha=0.3)
    axes[2].set_ylim([0, 1])
    
    fig.tight_layout()
    return fig


# ===================================================================
# Active return decomposition
# ===================================================================

def plot_return_decomposition(
    decomposition: dict,
    figsize: Tuple[int, int] = (10, 6),
) -> plt.Figure:
    """Plot alpha/beta/residual return decomposition as a waterfall chart.
    
    Parameters
    ----------
    decomposition : output from decompose_return()
    figsize : figure size
    
    Returns
    -------
    matplotlib Figure
    """
    # Extract components
    rf = decomposition.get("risk_free_return", 0.0)
    beta_contrib = decomposition.get("beta_contribution", 0.0)
    alpha_contrib = decomposition.get("alpha_contribution", 0.0)
    residual = decomposition.get("residual_contribution", 0.0)
    total = decomposition.get("total_return", 0.0)
    
    # Build waterfall
    labels = ["Risk-Free", "Beta × (Mkt - RF)", "Alpha", "Residual", "Total"]
    values = [rf, beta_contrib, alpha_contrib, residual, 0]  # Total is cumulative
    cumulative = np.cumsum([rf, beta_contrib, alpha_contrib, residual])
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot bars
    colors = ["#95a5a6", "#3498db", "#27ae60", "#e67e22", "#2c3e50"]
    
    for i, (label, val) in enumerate(zip(labels[:-1], values[:-1])):
        if i == 0:
            ax.bar(i, val, color=colors[i], edgecolor="black", linewidth=0.5)
        else:
            ax.bar(i, val, bottom=cumulative[i-1], color=colors[i], 
                   edgecolor="black", linewidth=0.5)
    
    # Total bar
    ax.bar(len(labels)-1, total, color=colors[-1], edgecolor="black", linewidth=1.5)
    
    # Connecting lines
    for i in range(len(labels)-2):
        ax.plot([i+0.4, i+1-0.4], [cumulative[i], cumulative[i]], 
                color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Return (annualized)", fontsize=11)
    ax.set_title("Return Decomposition: Alpha vs Beta", fontsize=13, fontweight="bold")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x:.1%}"))
    ax.grid(True, axis="y", alpha=0.3)
    ax.axhline(0, color="black", linewidth=0.8)
    
    fig.tight_layout()
    return fig


# ===================================================================
# Tracking error over time
# ===================================================================

def plot_tracking_error(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
    window: int = 63,  # ~3 months
    annualization: int = 252,
    figsize: Tuple[int, int] = (12, 5),
) -> plt.Figure:
    """Plot rolling tracking error (std of active returns).
    
    Parameters
    ----------
    portfolio_returns : daily portfolio returns
    benchmark_returns : daily benchmark returns
    window : rolling window size
    annualization : periods per year
    figsize : figure size
    
    Returns
    -------
    matplotlib Figure
    """
    # Align
    df = pd.DataFrame({
        "port": portfolio_returns,
        "bench": benchmark_returns,
    }).dropna()
    
    # Active return
    active = compute_active_return(df["port"], df["bench"])
    
    # Rolling tracking error
    rolling_te = active.rolling(window).std() * np.sqrt(annualization)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.plot(rolling_te.index, rolling_te.values, linewidth=2, color="#e74c3c")
    ax.fill_between(rolling_te.index, 0, rolling_te.values, alpha=0.2, color="#e74c3c")
    
    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("Tracking Error (ann.)", fontsize=11)
    ax.set_title(f"Rolling {window}-Day Tracking Error", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x:.1%}"))
    
    # Add mean line
    mean_te = rolling_te.mean()
    ax.axhline(mean_te, color="black", linewidth=1, linestyle="--", 
               alpha=0.6, label=f"Mean: {mean_te:.2%}")
    ax.legend(loc="upper right", framealpha=0.9)
    
    fig.tight_layout()
    return fig
