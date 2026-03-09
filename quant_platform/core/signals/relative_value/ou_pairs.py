"""Ornstein–Uhlenbeck process for pairs / mean-reversion — wrapped from TradeX.

Provides MLE-based OU parameter estimation with Numba-accelerated kernels
and a rolling z-score factor for portfolio construction.

Numba kernels
-------------
- ``_ou_mle_fast``      — closed-form MLE for (θ, μ, σ, half-life)
- ``_rolling_ou_params`` — rolling-window OU fit + z-score

Classes
-------
- ``OUProcess``  — fit / rolling_fit / generate_signals / cointegration
- ``OUZScore``   — FactorBase subclass
"""

from __future__ import annotations

from typing import Optional, Tuple

import numba as nb
import numpy as np
import pandas as pd
from loguru import logger

from quant_platform.core.signals.base import FactorBase, FactorMeta

# ===================================================================
# Numba kernels
# ===================================================================

@nb.njit(cache=True)
def _ou_mle_fast(
    spread: np.ndarray,
    dt: float,
) -> Tuple[float, float, float, float]:
    """Closed-form MLE for discrete OU: dX = θ(μ − X)dt + σ dW.

    Returns (theta, mu, sigma, half_life).
    """
    n = spread.shape[0]
    if n < 3:
        return 0.0, 0.0, 0.0, np.inf

    sx = 0.0
    sy = 0.0
    sxx = 0.0
    sxy = 0.0
    syy = 0.0
    for i in range(n - 1):
        x = spread[i]
        y = spread[i + 1]
        sx += x
        sy += y
        sxx += x * x
        sxy += x * y
        syy += y * y

    m = float(n - 1)
    denom = m * sxx - sx * sx
    if abs(denom) < 1e-300:
        return 0.0, 0.0, 0.0, np.inf

    a = (m * sxy - sx * sy) / denom
    b = (sy * sxx - sx * sxy) / denom

    if a <= 0.0 or a >= 1.0:
        return 0.0, 0.0, 0.0, np.inf

    theta = -np.log(a) / dt
    mu = b / (1.0 - a)

    # Residual variance
    sigma_sq = 0.0
    for i in range(n - 1):
        resid = spread[i + 1] - a * spread[i] - b
        sigma_sq += resid * resid
    sigma_sq /= m

    sigma_ou = np.sqrt(sigma_sq * 2.0 * theta / (1.0 - a * a) + 1e-30)
    half_life = np.log(2.0) / theta if theta > 0 else np.inf

    return theta, mu, sigma_ou, half_life


@nb.njit(cache=True)
def _rolling_ou_params(
    spread: np.ndarray,
    window: int,
    dt: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Rolling-window OU MLE + z-score.

    Returns arrays of length len(spread):
        (theta, mu, sigma, half_life, zscore)
    with NaN for the initial ``window-1`` observations.
    """
    n = spread.shape[0]
    theta_arr = np.full(n, np.nan)
    mu_arr = np.full(n, np.nan)
    sigma_arr = np.full(n, np.nan)
    hl_arr = np.full(n, np.nan)
    zs_arr = np.full(n, np.nan)

    for i in range(window - 1, n):
        chunk = spread[i - window + 1 : i + 1]
        th, m, s, hl = _ou_mle_fast(chunk, dt)
        theta_arr[i] = th
        mu_arr[i] = m
        sigma_arr[i] = s
        hl_arr[i] = hl
        if s > 1e-10 and th > 0:
            eq_std = s / np.sqrt(2.0 * th)
            zs_arr[i] = (spread[i] - m) / eq_std
        else:
            zs_arr[i] = 0.0

    return theta_arr, mu_arr, sigma_arr, hl_arr, zs_arr


# ===================================================================
# OUProcess
# ===================================================================

class OUProcess:
    """Ornstein–Uhlenbeck process estimator and signal generator.

    Parameters
    ----------
    window : int — rolling window for parameter estimation (default 60).
    dt : float — time step in years (default 1/252 for daily data).
    """

    def __init__(self, window: int = 60, dt: float = 1.0 / 252) -> None:
        self.window = window
        self.dt = dt
        self.theta: float = 0.0
        self.mu: float = 0.0
        self.sigma: float = 0.0
        self.half_life: float = np.inf

    def fit(self, spread: np.ndarray) -> "OUProcess":
        """Full-sample MLE fit."""
        spread = np.asarray(spread, dtype=np.float64)
        self.theta, self.mu, self.sigma, self.half_life = _ou_mle_fast(
            spread, self.dt
        )
        logger.debug(
            "OU fit: θ={:.4f}, μ={:.4f}, σ={:.4f}, HL={:.1f}d",
            self.theta, self.mu, self.sigma, self.half_life,
        )
        return self

    def rolling_fit(
        self, spread: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Rolling-window MLE returning (theta, mu, sigma, half_life, zscore)."""
        spread = np.asarray(spread, dtype=np.float64)
        return _rolling_ou_params(spread, self.window, self.dt)

    def generate_signals(self, spread: np.ndarray) -> pd.DataFrame:
        """Return a DataFrame with rolling OU params and z-score."""
        spread = np.asarray(spread, dtype=np.float64)
        th, mu, sig, hl, zs = self.rolling_fit(spread)
        return pd.DataFrame({
            "ou_theta": th,
            "ou_mu": mu,
            "ou_sigma": sig,
            "ou_half_life": hl,
            "ou_zscore": zs,
        })

    @staticmethod
    def engle_granger_coint(
        y: np.ndarray, x: np.ndarray, trend: str = "c"
    ) -> Tuple[float, float, float]:
        """Engle-Granger two-step cointegration test.

        Returns (adf_stat, p_value, hedge_ratio).
        Requires ``statsmodels``.
        """
        from statsmodels.tsa.stattools import adfuller
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)

        # Step 1: OLS regression  y = β·x + ε
        X = np.column_stack([x, np.ones(len(x))])
        beta, _ = np.linalg.lstsq(X, y, rcond=None)[:2]
        hedge_ratio = beta[0]

        # Step 2: ADF on residuals
        residuals = y - X @ beta
        adf_result = adfuller(residuals, maxlag=None, regression=trend)
        return adf_result[0], adf_result[1], hedge_ratio

    @staticmethod
    def compute_spread(
        y: np.ndarray, x: np.ndarray, hedge_ratio: float
    ) -> np.ndarray:
        """Compute hedge-ratio-weighted spread: y − β·x."""
        return np.asarray(y, dtype=np.float64) - hedge_ratio * np.asarray(
            x, dtype=np.float64
        )


# ===================================================================
# Factor
# ===================================================================

class OUZScore(FactorBase):
    """Rolling OU z-score: deviation from equilibrium in stationary std units.

    Positive z-score ⇒ spread is above equilibrium (sell the spread);
    negative ⇒ below equilibrium (buy the spread).  ``direction=+1``
    because higher absolute deviation means stronger mean-reversion
    opportunity, but the sign carries the trade direction.
    """

    meta = FactorMeta(
        name="OU_ZSCORE",
        category="mean_reversion",
        signal_type="relative_value",
        description="OU process z-score: deviation from equilibrium in stationary std units",
        lookback=61,
        lag=1,
        direction=1,
    )

    def _compute(
        self,
        prices: pd.DataFrame,
        fundamentals: Optional[pd.DataFrame] = None,
    ) -> pd.Series:
        df = prices.sort_values(["ticker", "date"]).copy()
        df["log_price"] = np.log(df["adj_close"])

        ou = OUProcess(window=60, dt=1.0 / 252)
        results = []
        for _ticker, grp in df.groupby("ticker"):
            log_p = grp["log_price"].values
            _, _, _, _, zscore = ou.rolling_fit(log_p)
            results.append(pd.Series(zscore, index=grp.index, name=self.meta.name))

        return pd.concat(results)


# ===================================================================
# Convenience list
# ===================================================================

ALL_OU_FACTORS = [OUZScore]
