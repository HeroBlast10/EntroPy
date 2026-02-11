"""Chart generators for the research report.

All functions return ``matplotlib.figure.Figure`` objects so they can be
embedded in HTML reports or Jupyter notebooks.  Charts follow a consistent
dark/professional style suitable for research presentations.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for server/CI
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd


# ===================================================================
# Style
# ===================================================================

_STYLE = {
    "figure.figsize": (12, 5),
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 10,
}


def _apply_style():
    plt.rcParams.update(_STYLE)


# ===================================================================
# NAV / equity curve
# ===================================================================

def plot_nav(
    daily_pnl: pd.DataFrame,
    title: str = "Portfolio NAV",
) -> plt.Figure:
    """Gross and net NAV on the same chart."""
    _apply_style()
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(daily_pnl.index, daily_pnl["nav_gross"], label="Gross", linewidth=1.2)
    ax.plot(daily_pnl.index, daily_pnl["nav_net"], label="Net", linewidth=1.2)
    ax.set_title(title)
    ax.set_ylabel("NAV ($)")
    ax.legend()
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    fig.tight_layout()
    return fig


# ===================================================================
# Drawdown
# ===================================================================

def plot_drawdown(
    daily_pnl: pd.DataFrame,
    title: str = "Drawdown",
) -> plt.Figure:
    _apply_style()
    fig, ax = plt.subplots(figsize=(14, 3.5))
    ax.fill_between(daily_pnl.index, daily_pnl["drawdown_net"], 0,
                    color="salmon", alpha=0.5, label="Net DD")
    ax.plot(daily_pnl.index, daily_pnl["drawdown_gross"],
            color="steelblue", linewidth=0.8, alpha=0.7, label="Gross DD")
    ax.set_title(title)
    ax.set_ylabel("Drawdown")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.legend()
    fig.tight_layout()
    return fig


# ===================================================================
# Rolling Sharpe
# ===================================================================

def plot_rolling_sharpe(
    returns: pd.Series,
    window: int = 252,
    title: str = "Rolling 1-Year Sharpe Ratio",
) -> plt.Figure:
    from entropy.evaluation.analytics import rolling_sharpe
    rs = rolling_sharpe(returns, window)

    _apply_style()
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(rs.index, rs, linewidth=1.0)
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.axhline(rs.mean(), color="orange", linewidth=0.8, linestyle=":", label=f"Mean={rs.mean():.2f}")
    ax.set_title(title)
    ax.set_ylabel("Sharpe")
    ax.legend()
    fig.tight_layout()
    return fig


# ===================================================================
# IC time series
# ===================================================================

def plot_ic_series(
    ic: pd.Series,
    title: str = "Daily IC (Rank)",
    ma_window: int = 21,
) -> plt.Figure:
    _apply_style()
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.bar(ic.index, ic, width=1.0, color="steelblue", alpha=0.3, label="Daily IC")
    ma = ic.rolling(ma_window, min_periods=1).mean()
    ax.plot(ma.index, ma, color="orange", linewidth=1.5, label=f"{ma_window}-day MA")
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_title(title)
    ax.set_ylabel("IC")
    ax.legend()
    fig.tight_layout()
    return fig


# ===================================================================
# Quantile return bar chart
# ===================================================================

def plot_quantile_returns(
    quantile_returns: pd.DataFrame,
    title: str = "Mean Forward Return by Factor Quantile",
) -> plt.Figure:
    """Bar chart of average return per quantile across all dates."""
    _apply_style()
    avg = quantile_returns.groupby("quantile")["mean_ret"].mean()

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#d9534f" if v < 0 else "#5cb85c" for v in avg]
    ax.bar(avg.index, avg, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Quantile")
    ax.set_ylabel("Mean Return")
    ax.set_title(title)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=2))
    fig.tight_layout()
    return fig


# ===================================================================
# Factor correlation heatmap
# ===================================================================

def plot_correlation_heatmap(
    corr: pd.DataFrame,
    title: str = "Average Cross-Sectional Factor Correlation",
) -> plt.Figure:
    _apply_style()
    n = len(corr)
    fig, ax = plt.subplots(figsize=(max(8, n * 0.6), max(6, n * 0.5)))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(corr.index, fontsize=8)
    # Annotate
    for i in range(n):
        for j in range(n):
            val = corr.iloc[i, j]
            color = "white" if abs(val) > 0.5 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7, color=color)
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title(title)
    fig.tight_layout()
    return fig


# ===================================================================
# Sector exposure stacked area
# ===================================================================

def plot_sector_exposure(
    exposure: pd.DataFrame,
    title: str = "Sector Exposure Over Time",
) -> plt.Figure:
    _apply_style()
    pivot = exposure.pivot_table(index="date", columns="sector", values="weight_net", aggfunc="sum").fillna(0)
    fig, ax = plt.subplots(figsize=(14, 5))
    pivot.plot.area(ax=ax, alpha=0.7, linewidth=0.5)
    ax.set_title(title)
    ax.set_ylabel("Net Weight")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    fig.tight_layout()
    return fig


# ===================================================================
# Monthly return heatmap
# ===================================================================

def plot_monthly_heatmap(
    monthly_table: pd.DataFrame,
    title: str = "Monthly Returns (%)",
) -> plt.Figure:
    _apply_style()
    data = monthly_table * 100  # to percent
    n_rows, n_cols = data.shape
    fig, ax = plt.subplots(figsize=(max(10, n_cols), max(4, n_rows * 0.5)))
    im = ax.imshow(data.values, cmap="RdYlGn", aspect="auto",
                   vmin=-5, vmax=5)
    ax.set_xticks(range(n_cols))
    ax.set_yticks(range(n_rows))
    month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    ax.set_xticklabels(month_labels[:n_cols], fontsize=9)
    ax.set_yticklabels(data.index, fontsize=9)
    for i in range(n_rows):
        for j in range(n_cols):
            val = data.iloc[i, j]
            if np.isfinite(val):
                color = "white" if abs(val) > 3 else "black"
                ax.text(j, i, f"{val:.1f}", ha="center", va="center", fontsize=7, color=color)
    fig.colorbar(im, ax=ax, shrink=0.8, label="%")
    ax.set_title(title)
    fig.tight_layout()
    return fig


# ===================================================================
# Walk-forward out-of-sample Sharpe
# ===================================================================

def plot_walkforward_sharpe(
    wf_results: pd.DataFrame,
    title: str = "Walk-Forward Out-of-Sample Sharpe",
) -> plt.Figure:
    """Bar chart of OOS Sharpe per fold."""
    _apply_style()
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#5cb85c" if v > 0 else "#d9534f" for v in wf_results["oos_sharpe"]]
    ax.bar(range(len(wf_results)), wf_results["oos_sharpe"], color=colors, edgecolor="white")
    ax.axhline(wf_results["oos_sharpe"].mean(), color="orange", linestyle=":", linewidth=1.5,
               label=f"Mean={wf_results['oos_sharpe'].mean():.2f}")
    ax.set_xlabel("Fold")
    ax.set_ylabel("OOS Sharpe")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig


# ===================================================================
# Ablation comparison bar chart
# ===================================================================

def plot_ablation(
    ablation_results: pd.DataFrame,
    metric: str = "net_sharpe",
    title: str = "Ablation Study",
) -> plt.Figure:
    _apply_style()
    fig, ax = plt.subplots(figsize=(10, 5))
    scenarios = ablation_results["scenario"]
    values = ablation_results[metric]
    colors = plt.cm.Set2(np.linspace(0, 1, len(scenarios)))
    bars = ax.barh(range(len(scenarios)), values, color=colors, edgecolor="white")
    ax.set_yticks(range(len(scenarios)))
    ax.set_yticklabels(scenarios, fontsize=9)
    ax.set_xlabel(metric.replace("_", " ").title())
    ax.set_title(title)
    # Value labels
    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=9)
    fig.tight_layout()
    return fig
