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
def _forward_step(
    previous: np.ndarray,
    observation: float,
    A: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
) -> np.ndarray:
    """Advance one causal filtering step under fixed HMM parameters."""
    K = previous.shape[0]
    current = np.zeros(K)
    for j in range(K):
        predicted = 0.0
        for i in range(K):
            predicted += previous[i] * A[i, j]
        current[j] = predicted * _gaussian_pdf(
            observation,
            mu[j],
            sigma[j],
        )
    normalizer = current.sum()
    if normalizer <= 0.0 or not np.isfinite(normalizer):
        for j in range(K):
            current[j] = 1.0 / K
    else:
        current /= normalizer
    return current


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
                denom += v
        if denom == 0.0:
            denom = 1e-300
        # Each time slice must be normalised before accumulation.  Forward
        # scaling constants vary by t, so summing raw numerators would give
        # arbitrary weights to dates with larger observation likelihood.
        for i in range(K):
            for j in range(K):
                v = (
                    alpha[t, i]
                    * A[i, j]
                    * _gaussian_pdf(obs[t + 1], mu[j], sigma[j])
                    * beta[t + 1, j]
                )
                xi_sum[i, j] += v / denom

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
        """Return smoothed posterior probabilities (offline diagnostics only).

        Smoothing uses the backward pass and therefore conditions historical
        states on future observations.  Live factors must use
        :meth:`predict_filtered`.
        """
        obs = np.asarray(obs, dtype=np.float64)
        alpha, c = _forward(obs, self.pi, self.A, self.mu, self.sigma)
        beta = _backward(obs, self.A, self.mu, self.sigma, c)
        gamma = alpha * beta
        gamma /= gamma.sum(axis=1, keepdims=True) + 1e-300
        return gamma

    def predict_filtered(self, obs: np.ndarray) -> np.ndarray:
        """Return causal filtered probabilities ``P(state_t | obs_0..t)``."""
        if any(value is None for value in (self.pi, self.A, self.mu, self.sigma)):
            raise RuntimeError("RegimeDetector must be fitted before prediction")
        obs = np.asarray(obs, dtype=np.float64)
        if obs.ndim != 1 or len(obs) == 0:
            raise ValueError("obs must be a non-empty one-dimensional array")
        alpha, _ = _forward(obs, self.pi, self.A, self.mu, self.sigma)
        return alpha

    @property
    def turbulent_state(self) -> int:
        """Index of the state with the largest fitted volatility."""
        if self.sigma is None:
            raise RuntimeError("RegimeDetector must be fitted first")
        return int(np.argmax(self.sigma))

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
        from quant_platform.core.data.returns import trailing_total_return
        df["ret"] = trailing_total_return(df)

        min_history = int(self._extra_params.get("min_history", self._meta.lookback))
        train_window = int(
            self._extra_params.get("train_window", max(252, min_history))
        )
        refit_interval = int(self._extra_params.get("refit_interval", 126))
        max_iter = int(self._extra_params.get("max_iter", 10))
        tol = float(self._extra_params.get("tol", 1e-4))
        if min_history < 20:
            raise ValueError("HMM min_history must be at least 20 observations")
        if train_window < min_history:
            raise ValueError("HMM train_window must be >= min_history")
        if refit_interval < 1:
            raise ValueError("HMM refit_interval must be positive")

        results = []
        for _ticker, grp in df.groupby("ticker", sort=False):
            valid = grp["ret"].replace([np.inf, -np.inf], np.nan).dropna()
            rets = valid.to_numpy(dtype=np.float64)
            if len(rets) <= min_history:
                results.append(pd.Series(np.nan, index=grp.index, name=self.meta.name))
                continue

            prob_turb = np.full(len(rets), np.nan, dtype=float)
            hmm: Optional[RegimeDetector] = None
            last_refit = -refit_interval
            filtered_state: Optional[np.ndarray] = None

            # Fit only on observations strictly before the scored return.
            # At a refit, initialise the causal state from the training
            # context.  Between refits, advance that state one observation at
            # a time instead of recomputing a 252-day forward pass for every
            # row.  This preserves prefix invariance and reduces the scoring
            # stage from O(T * train_window) to O(T).
            for pos in range(min_history, len(rets)):
                if hmm is None or pos - last_refit >= refit_interval:
                    train_start = max(0, pos - train_window)
                    train_obs = rets[train_start:pos]
                    candidate: Optional[RegimeDetector] = None
                    try:
                        candidate = RegimeDetector(
                            n_states=2,
                            max_iter=max_iter,
                            tol=tol,
                        ).fit(train_obs)
                    except Exception as exc:
                        logger.debug("{} HMM refit failed at {}: {}", _ticker, pos, exc)
                    if (
                        candidate is not None
                        and candidate.sigma is not None
                        and np.isfinite(candidate.sigma).all()
                    ):
                        hmm = candidate
                        last_refit = pos
                        filtered_state = hmm.predict_filtered(
                            rets[train_start:pos + 1]
                        )[-1]
                    elif hmm is None:
                        continue
                    elif filtered_state is not None:
                        # Keep the last valid parameter set if a scheduled
                        # refit is numerically unstable.
                        filtered_state = _forward_step(
                            filtered_state,
                            rets[pos],
                            hmm.A,
                            hmm.mu,
                            hmm.sigma,
                        )
                elif filtered_state is not None:
                    filtered_state = _forward_step(
                        filtered_state,
                        rets[pos],
                        hmm.A,
                        hmm.mu,
                        hmm.sigma,
                    )

                if filtered_state is not None:
                    prob_turb[pos] = filtered_state[hmm.turbulent_state]

            s = pd.Series(np.nan, index=grp.index, name=self.meta.name)
            s.loc[valid.index] = prob_turb
            results.append(s)

        return pd.concat(results)


# ===================================================================
# Convenience list
# ===================================================================

ALL_REGIME_FACTORS = [HMMTurbulenceProbability]
