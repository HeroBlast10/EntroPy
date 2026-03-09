"""HMM regime detection — wrapped from TradeX regime.py.

Provides a 2-state Gaussian HMM via the Baum-Welch algorithm with
Numba-accelerated kernels.  The :class:`HMMTurbulenceProbability` factor
exposes the posterior probability of the high-volatility ("turbulent")
state for downstream portfolio construction.

Numba kernels
-------------
- ``_gaussian_pdf``  — univariate Gaussian density
- ``_forward``       — scaled forward pass (α, scaling factors)
- ``_backward``      — scaled backward pass (β)
- ``_baum_welch_step`` — single EM iteration

Classes
-------
- ``RegimeDetector`` — fit / predict / fit_predict convenience wrapper
- ``HMMTurbulenceProbability`` — FactorBase subclass
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
def _gaussian_pdf(x: float, mu: float, sigma: float) -> float:
    """Univariate Gaussian probability density."""
    z = (x - mu) / sigma
    return np.exp(-0.5 * z * z) / (sigma * np.sqrt(2.0 * np.pi))


@nb.njit(cache=True)
def _forward(
    obs: np.ndarray,
    pi: np.ndarray,
    A: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Scaled forward algorithm.

    Returns
    -------
    alpha : (T, K) forward probabilities
    c     : (T,)  scaling factors  (product = data likelihood)
    """
    T = obs.shape[0]
    K = pi.shape[0]
    alpha = np.zeros((T, K))
    c = np.zeros(T)

    # t = 0
    for j in range(K):
        alpha[0, j] = pi[j] * _gaussian_pdf(obs[0], mu[j], sigma[j])
    c[0] = alpha[0].sum()
    if c[0] == 0.0:
        c[0] = 1e-300
    alpha[0] /= c[0]

    # t = 1 … T-1
    for t in range(1, T):
        for j in range(K):
            s = 0.0
            for i in range(K):
                s += alpha[t - 1, i] * A[i, j]
            alpha[t, j] = s * _gaussian_pdf(obs[t], mu[j], sigma[j])
        c[t] = alpha[t].sum()
        if c[t] == 0.0:
            c[t] = 1e-300
        alpha[t] /= c[t]

    return alpha, c


@nb.njit(cache=True)
def _backward(
    obs: np.ndarray,
    A: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    c: np.ndarray,
) -> np.ndarray:
    """Scaled backward algorithm.

    Returns
    -------
    beta : (T, K) backward probabilities
    """
    T = obs.shape[0]
    K = A.shape[0]
    beta = np.zeros((T, K))

    # t = T-1
    for j in range(K):
        beta[T - 1, j] = 1.0

    # t = T-2 … 0
    for t in range(T - 2, -1, -1):
        for i in range(K):
            s = 0.0
            for j in range(K):
                s += A[i, j] * _gaussian_pdf(obs[t + 1], mu[j], sigma[j]) * beta[t + 1, j]
            beta[t, i] = s
        if c[t + 1] == 0.0:
            beta[t] /= 1e-300
        else:
            beta[t] /= c[t + 1]

    return beta


@nb.njit(cache=True)
def _baum_welch_step(
    obs: np.ndarray,
    pi: np.ndarray,
    A: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Single Baum-Welch (EM) iteration.

    Returns
    -------
    new_pi, new_A, new_mu, new_sigma, log_likelihood
    """
    T = obs.shape[0]
    K = pi.shape[0]

    alpha, c = _forward(obs, pi, A, mu, sigma)
    beta = _backward(obs, A, mu, sigma, c)

    # Log-likelihood from scaling factors
    ll = 0.0
    for t in range(T):
        ll += np.log(c[t] + 1e-300)

    # Posterior state probabilities  γ(t, j)
    gamma = np.zeros((T, K))
    for t in range(T):
        denom = 0.0
        for j in range(K):
            gamma[t, j] = alpha[t, j] * beta[t, j]
            denom += gamma[t, j]
        if denom == 0.0:
            denom = 1e-300
        for j in range(K):
            gamma[t, j] /= denom

    # Transition posteriors  ξ(t, i, j) — summed over t
    xi_sum = np.zeros((K, K))
    for t in range(T - 1):
        denom = 0.0
        for i in range(K):
            for j in range(K):
                v = (
                    alpha[t, i]
                    * A[i, j]
                    * _gaussian_pdf(obs[t + 1], mu[j], sigma[j])
                    * beta[t + 1, j]
                )
                xi_sum[i, j] += v
                denom += v
        if denom == 0.0:
            denom = 1e-300
        # (normalisation applied below via gamma sums)

    # --- M-step ---
    new_pi = np.empty(K)
    new_mu = np.empty(K)
    new_sigma = np.empty(K)
    new_A = np.empty((K, K))

    for j in range(K):
        new_pi[j] = gamma[0, j]

        gamma_sum_j = 0.0
        for t in range(T):
            gamma_sum_j += gamma[t, j]

        # Mean
        w_sum = 0.0
        for t in range(T):
            w_sum += gamma[t, j] * obs[t]
        new_mu[j] = w_sum / (gamma_sum_j + 1e-300)

        # Variance
        v_sum = 0.0
        for t in range(T):
            diff = obs[t] - new_mu[j]
            v_sum += gamma[t, j] * diff * diff
        new_sigma[j] = np.sqrt(v_sum / (gamma_sum_j + 1e-300) + 1e-10)

        # Transition row
        gamma_sum_no_last = 0.0
        for t in range(T - 1):
            gamma_sum_no_last += gamma[t, j]
        for k in range(K):
            new_A[j, k] = xi_sum[j, k] / (gamma_sum_no_last + 1e-300)

        # Normalise transition row
        row_sum = 0.0
        for k in range(K):
            row_sum += new_A[j, k]
        if row_sum > 0:
            for k in range(K):
                new_A[j, k] /= row_sum

    return new_pi, new_A, new_mu, new_sigma, ll


# ===================================================================
# RegimeDetector
# ===================================================================

class RegimeDetector:
    """2-state Gaussian HMM trained with Baum-Welch.

    States are labelled after fitting so that state 0 = low-vol ("calm")
    and state 1 = high-vol ("turbulent").

    Parameters
    ----------
    n_states : int (default 2)
    max_iter : int (default 100)
    tol : float — convergence threshold for log-likelihood change.
    """

    def __init__(
        self,
        n_states: int = 2,
        max_iter: int = 100,
        tol: float = 1e-4,
    ) -> None:
        self.n_states = n_states
        self.max_iter = max_iter
        self.tol = tol

        self.pi: Optional[np.ndarray] = None
        self.A: Optional[np.ndarray] = None
        self.mu: Optional[np.ndarray] = None
        self.sigma: Optional[np.ndarray] = None
        self.converged: bool = False

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(self, obs: np.ndarray) -> "RegimeDetector":
        """Fit the HMM on a 1-D observation array via Baum-Welch."""
        obs = np.asarray(obs, dtype=np.float64)
        K = self.n_states

        # Initialise with K-means-ish heuristic
        sorted_obs = np.sort(obs)
        splits = np.array_split(sorted_obs, K)
        mu = np.array([s.mean() for s in splits])
        sigma = np.array([max(s.std(), 1e-6) for s in splits])
        pi = np.ones(K) / K
        A = np.full((K, K), 1.0 / K)

        prev_ll = -np.inf
        for it in range(self.max_iter):
            pi, A, mu, sigma, ll = _baum_welch_step(obs, pi, A, mu, sigma)
            if abs(ll - prev_ll) < self.tol:
                self.converged = True
                logger.debug("HMM converged at iteration {}", it + 1)
                break
            prev_ll = ll

        # Label so state 0 = low-vol, state 1 = high-vol
        order = np.argsort(sigma)
        self.mu = mu[order]
        self.sigma = sigma[order]
        self.pi = pi[order]
        self.A = A[np.ix_(order, order)]
        return self

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------

    def predict(self, obs: np.ndarray) -> np.ndarray:
        """Return posterior state probabilities (T, K)."""
        obs = np.asarray(obs, dtype=np.float64)
        alpha, c = _forward(obs, self.pi, self.A, self.mu, self.sigma)
        beta = _backward(obs, self.A, self.mu, self.sigma, c)
        gamma = alpha * beta
        gamma /= gamma.sum(axis=1, keepdims=True) + 1e-300
        return gamma

    def fit_predict(self, obs: np.ndarray) -> np.ndarray:
        """Fit then return posterior probabilities."""
        self.fit(obs)
        return self.predict(obs)


# ===================================================================
# Factor
# ===================================================================

class HMMTurbulenceProbability(FactorBase):
    """Posterior probability of the high-vol (turbulent) HMM state.

    A value near 1.0 means the market is very likely in a turbulent
    regime; use ``direction=-1`` so the portfolio construction layer
    reduces exposure when turbulence is high.
    """

    meta = FactorMeta(
        name="HMM_TURBULENCE_PROB",
        category="regime",
        signal_type="regime",
        description="HMM posterior probability of turbulent (high-vol) state",
        lookback=120,
        lag=1,
        direction=-1,
    )

    def _compute(
        self,
        prices: pd.DataFrame,
        fundamentals: Optional[pd.DataFrame] = None,
    ) -> pd.Series:
        df = prices.sort_values(["ticker", "date"]).copy()
        df["ret"] = df.groupby("ticker")["adj_close"].pct_change()

        results = []
        for _ticker, grp in df.groupby("ticker"):
            rets = grp["ret"].dropna().values
            if len(rets) < self.meta.lookback:
                results.append(pd.Series(np.nan, index=grp.index, name=self.meta.name))
                continue
            hmm = RegimeDetector(n_states=2, max_iter=100, tol=1e-4)
            gamma = hmm.fit_predict(rets)
            prob_turb = np.full(len(grp), np.nan)
            # gamma is aligned to non-NaN returns (offset by 1 for pct_change)
            valid_idx = grp["ret"].dropna().index
            prob_turb_vals = gamma[:, 1]  # state 1 = high-vol
            s = pd.Series(np.nan, index=grp.index, name=self.meta.name)
            s.loc[valid_idx] = prob_turb_vals
            s = s.ffill()
            results.append(s)

        return pd.concat(results)


# ===================================================================
# Convenience list
# ===================================================================

ALL_REGIME_FACTORS = [HMMTurbulenceProbability]
