"""Kalman-filter state-space factors wrapping TradeX Numba kernels.

Factors extract latent velocity, trend strength, and microstructure noise
from a constant-velocity Kalman model applied per ticker.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from numba import njit

from quant_platform.core.signals.base import FactorBase, FactorMeta


# ===================================================================
# Numba kernels (ported from TradeX kalman.py)
# ===================================================================

@njit(cache=True)
def _kalman_filter_1d(prices, q00, q01, q10, q11, r, a00, a01, a10, a11, h0, h1):
    """Single-stock Kalman recursion with flattened 2×2 matrices.

    State vector x = [position, velocity].
    Returns (filtered, velocity, kalman_gain_0).
    """
    n = len(prices)
    filtered = np.empty(n)
    velocity = np.empty(n)
    kg = np.empty(n)

    x0 = prices[0]
    x1 = 0.0
    p00 = 1.0; p01 = 0.0; p10 = 0.0; p11 = 1.0

    for i in range(n):
        z = prices[i]

        # Predict state
        xp0 = a00 * x0 + a01 * x1
        xp1 = a10 * x0 + a11 * x1

        # Predict covariance: P_pred = A @ P @ A^T + Q
        t00 = a00 * p00 + a01 * p10; t01 = a00 * p01 + a01 * p11
        t10 = a10 * p00 + a11 * p10; t11 = a10 * p01 + a11 * p11
        pp00 = t00 * a00 + t01 * a10 + q00; pp01 = t00 * a01 + t01 * a11 + q01
        pp10 = t10 * a00 + t11 * a10 + q10; pp11 = t10 * a01 + t11 * a11 + q11

        # Innovation covariance: S = H @ P_pred @ H^T + R
        hp0 = h0 * pp00 + h1 * pp10; hp1 = h0 * pp01 + h1 * pp11
        s = hp0 * h0 + hp1 * h1 + r

        # Kalman gain: K = P_pred @ H^T / S
        k0 = (pp00 * h0 + pp01 * h1) / s
        k1 = (pp10 * h0 + pp11 * h1) / s

        # Update state
        innov = z - (h0 * xp0 + h1 * xp1)
        x0 = xp0 + k0 * innov
        x1 = xp1 + k1 * innov

        # Update covariance: P = (I - K @ H) @ P_pred
        ikh00 = 1.0 - k0 * h0; ikh01 = -k0 * h1
        ikh10 = -k1 * h0;       ikh11 = 1.0 - k1 * h1
        p00 = ikh00 * pp00 + ikh01 * pp10; p01 = ikh00 * pp01 + ikh01 * pp11
        p10 = ikh10 * pp00 + ikh11 * pp10; p11 = ikh10 * pp01 + ikh11 * pp11

        filtered[i] = x0
        velocity[i] = x1
        kg[i] = k0

    return filtered, velocity, kg


@njit(cache=True)
def _batch_kalman_filter(all_prices, lengths, q00, q01, q10, q11, r,
                         a00, a01, a10, a11, h0, h1):
    """Run Kalman filter in parallel over multiple concatenated price series.

    Parameters
    ----------
    all_prices : 1-D array of concatenated (normalised) prices.
    lengths : 1-D int array — length of each individual series.
    """
    total = all_prices.shape[0]
    filtered = np.empty(total)
    velocity = np.empty(total)
    kg = np.empty(total)

    offset = 0
    for s in range(len(lengths)):
        n = lengths[s]
        seg = all_prices[offset:offset + n]
        f, v, g = _kalman_filter_1d(seg, q00, q01, q10, q11, r,
                                     a00, a01, a10, a11, h0, h1)
        filtered[offset:offset + n] = f
        velocity[offset:offset + n] = v
        kg[offset:offset + n] = g
        offset += n

    return filtered, velocity, kg


# ===================================================================
# Helper: run Kalman on a panel DataFrame
# ===================================================================

def _run_kalman_on_panel(
    prices: pd.DataFrame,
    Q: float = 1e-5,
    R: float = 1e-3,
    dt: float = 1.0,
):
    """Apply the Kalman filter per ticker and return aligned arrays.

    Returns (filtered, velocity, kg) as 1-D arrays aligned to *prices* index.
    """
    df = prices.sort_values(["ticker", "date"]).copy()

    all_prices_list = []
    lengths = []

    # per-ticker normalisation ranges for rescaling
    mins = []
    ranges = []

    for _ticker, grp in df.groupby("ticker", sort=False):
        p = grp["adj_close"].values.astype(np.float64)
        pmin = p.min()
        prange = p.max() - pmin
        if prange == 0.0:
            prange = 1.0
        p_norm = (p - pmin) / prange
        all_prices_list.append(p_norm)
        lengths.append(len(p_norm))
        mins.append(pmin)
        ranges.append(prange)

    all_prices = np.concatenate(all_prices_list)
    lengths_arr = np.array(lengths, dtype=np.int64)

    # Constant-velocity transition matrix A and observation H
    a00 = 1.0; a01 = dt; a10 = 0.0; a11 = 1.0
    h0 = 1.0; h1 = 0.0

    # Process noise Q (scaled)
    q00 = Q; q01 = 0.0; q10 = 0.0; q11 = Q

    filtered_norm, velocity_norm, kg = _batch_kalman_filter(
        all_prices, lengths_arr, q00, q01, q10, q11, R,
        a00, a01, a10, a11, h0, h1,
    )

    # Rescale back to original price domain
    filtered = np.empty_like(filtered_norm)
    velocity = np.empty_like(velocity_norm)
    raw_prices = np.empty_like(filtered_norm)

    offset = 0
    for idx, n in enumerate(lengths):
        sl = slice(offset, offset + n)
        filtered[sl] = filtered_norm[sl] * ranges[idx] + mins[idx]
        velocity[sl] = velocity_norm[sl] * ranges[idx]
        raw_prices[sl] = all_prices[sl] * ranges[idx] + mins[idx]
        offset += n

    return df, filtered, velocity, kg, raw_prices


# ===================================================================
# Factor classes
# ===================================================================

class KalmanVelocity(FactorBase):
    meta = FactorMeta(
        name="KF_VELOCITY",
        category="state_space",
        signal_type="time_series",
        description="Kalman-filtered velocity (trend rate of change)",
        lookback=60,
        lag=1,
        direction=1,
        references=["Kalman (1960)"],
    )

    def _compute(
        self,
        prices: pd.DataFrame,
        fundamentals: Optional[pd.DataFrame] = None,
    ) -> pd.Series:
        df, _filtered, velocity, _kg, _raw = _run_kalman_on_panel(prices)
        return pd.Series(velocity, index=df.index, name=self.meta.name).reindex(prices.index)


class KalmanTrendStrength(FactorBase):
    meta = FactorMeta(
        name="KF_TREND_STRENGTH",
        category="state_space",
        signal_type="time_series",
        description="Kalman velocity normalized by filtered price (dimensionless trend)",
        lookback=60,
        lag=1,
        direction=1,
    )

    def _compute(
        self,
        prices: pd.DataFrame,
        fundamentals: Optional[pd.DataFrame] = None,
    ) -> pd.Series:
        df, filtered, velocity, _kg, _raw = _run_kalman_on_panel(prices)
        safe_filt = np.where(np.abs(filtered) < 1e-12, np.nan, filtered)
        trend_strength = velocity / safe_filt
        return pd.Series(trend_strength, index=df.index, name=self.meta.name).reindex(prices.index)


class KalmanNoiseRatio(FactorBase):
    meta = FactorMeta(
        name="KF_NOISE_RATIO",
        category="state_space",
        signal_type="time_series",
        description="|market - filtered| / filtered (microstructure noise proxy)",
        lookback=60,
        lag=1,
        direction=-1,
    )

    def _compute(
        self,
        prices: pd.DataFrame,
        fundamentals: Optional[pd.DataFrame] = None,
    ) -> pd.Series:
        df, filtered, _velocity, _kg, _raw = _run_kalman_on_panel(prices)
        raw = df["adj_close"].values.astype(np.float64)
        safe_filt = np.where(np.abs(filtered) < 1e-12, np.nan, filtered)
        noise_ratio = np.abs(raw - filtered) / np.abs(safe_filt)
        return pd.Series(noise_ratio, index=df.index, name=self.meta.name).reindex(prices.index)


ALL_KALMAN_FACTORS = [KalmanVelocity, KalmanTrendStrength, KalmanNoiseRatio]
