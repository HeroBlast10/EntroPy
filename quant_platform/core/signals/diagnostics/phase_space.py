"""Phase-space reconstruction and divergence analysis — migrated from TradeX.

Embeds a price time-series into a 2-D phase space (x = price change,
y = lagged price change) and derives polar coordinates plus a discrete
divergence proxy.  Useful for visual regime diagnostics — **not** a
FactorBase signal.

Classes
-------
- ``PhaseSpaceAnalyzer`` — compute metrics, plot phase portrait, plot
  divergence time series.
"""

from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class PhaseSpaceAnalyzer:
    """2-D delay-embedding phase-space analyser for price series.

    Parameters
    ----------
    price_col : str — column name for the price series (default ``"adj_close"``).
    tau : int — embedding delay in rows (default 1).
    """

    def __init__(self, price_col: str = "adj_close", tau: int = 1) -> None:
        self.price_col = price_col
        self.tau = tau

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def compute_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add phase-space columns to *df* (in-place copy returned).

        New columns
        -----------
        ps_x           — price change  Δp(t)
        ps_y           — lagged change Δp(t − τ)
        ps_radius      — √(x² + y²)
        ps_angle       — arctan2(y, x)
        ps_angular_vel — Δangle / Δt
        ps_radial_vel  — Δradius / Δt
        ps_divergence  — discrete proxy: sign(x·Δx + y·Δy)
        """
        df = df.copy()
        p = df[self.price_col].values.astype(np.float64)

        dx = np.empty_like(p)
        dx[0] = np.nan
        dx[1:] = np.diff(p)

        dy = np.full_like(p, np.nan)
        if self.tau < len(p):
            dy[self.tau:] = dx[:-self.tau] if self.tau > 0 else dx

        df["ps_x"] = dx
        df["ps_y"] = dy

        x = df["ps_x"].values
        y = df["ps_y"].values

        radius = np.sqrt(x ** 2 + y ** 2)
        angle = np.arctan2(y, x)

        df["ps_radius"] = radius
        df["ps_angle"] = angle

        # Angular velocity (handle wrap-around via np.unwrap)
        unwrapped = np.unwrap(np.where(np.isnan(angle), 0, angle))
        ang_vel = np.empty_like(unwrapped)
        ang_vel[0] = np.nan
        ang_vel[1:] = np.diff(unwrapped)
        ang_vel[np.isnan(angle)] = np.nan
        df["ps_angular_vel"] = ang_vel

        # Radial velocity
        rad_vel = np.empty_like(radius)
        rad_vel[0] = np.nan
        rad_vel[1:] = np.diff(radius)
        df["ps_radial_vel"] = rad_vel

        # Divergence proxy: sign( x·Δx + y·Δy )
        d_x = np.empty_like(x)
        d_x[0] = np.nan
        d_x[1:] = np.diff(x)

        d_y = np.empty_like(y)
        d_y[0] = np.nan
        d_y[1:] = np.diff(y)

        div = np.sign(x * d_x + y * d_y)
        df["ps_divergence"] = div

        return df

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def plot_phase_portrait(
        self,
        df: pd.DataFrame,
        *,
        color_by: Optional[str] = None,
        cmap: str = "coolwarm",
        title: str = "Phase Portrait",
        figsize: tuple = (8, 8),
        alpha: float = 0.5,
        s: float = 10,
        ax: Optional[plt.Axes] = None,
    ) -> plt.Axes:
        """Scatter plot of (ps_x, ps_y) with optional colour mapping.

        Parameters
        ----------
        color_by : column name to colour points by (e.g. ``"ps_divergence"``).
        """
        if "ps_x" not in df.columns:
            df = self.compute_metrics(df)

        mask = df[["ps_x", "ps_y"]].notna().all(axis=1)
        plot_df = df.loc[mask]

        if ax is None:
            _, ax = plt.subplots(figsize=figsize)

        c = plot_df[color_by].values if color_by and color_by in plot_df.columns else None
        scatter = ax.scatter(
            plot_df["ps_x"], plot_df["ps_y"],
            c=c, cmap=cmap, alpha=alpha, s=s, edgecolors="none",
        )
        if c is not None:
            plt.colorbar(scatter, ax=ax, label=color_by)

        ax.axhline(0, color="grey", linewidth=0.5, linestyle="--")
        ax.axvline(0, color="grey", linewidth=0.5, linestyle="--")
        ax.set_xlabel("Δp(t)")
        ax.set_ylabel(f"Δp(t−{self.tau})")
        ax.set_title(title)
        ax.set_aspect("equal", adjustable="datalim")
        return ax

    def plot_divergence(
        self,
        df: pd.DataFrame,
        *,
        window: int = 21,
        title: str = "Phase-Space Divergence",
        figsize: tuple = (12, 4),
        ax: Optional[plt.Axes] = None,
    ) -> plt.Axes:
        """Rolling mean of divergence proxy over time.

        Parameters
        ----------
        window : int — smoothing window (trading days).
        """
        if "ps_divergence" not in df.columns:
            df = self.compute_metrics(df)

        if ax is None:
            _, ax = plt.subplots(figsize=figsize)

        date_col = "date" if "date" in df.columns else df.index
        div_smooth = df["ps_divergence"].rolling(window, min_periods=1).mean()

        ax.fill_between(
            df["date"] if "date" in df.columns else df.index,
            div_smooth, 0,
            where=div_smooth >= 0, color="salmon", alpha=0.4, label="expanding",
        )
        ax.fill_between(
            df["date"] if "date" in df.columns else df.index,
            div_smooth, 0,
            where=div_smooth < 0, color="steelblue", alpha=0.4, label="contracting",
        )
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_ylabel("Divergence (smoothed)")
        ax.set_title(title)
        ax.legend(loc="upper right", framealpha=0.7)
        return ax
