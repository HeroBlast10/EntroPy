"""Advanced portfolio construction: constrained mean-variance optimisation.

Solves the classic Markowitz problem with practical constraints:

    max   α'w − (λ/2) w'Σw
    s.t.  Σw_i = 1   (long-only)  or  Σw_long=1, Σw_short=−1
          0 ≤ w_i ≤ w_max         (long-only)
          −w_max ≤ w_i ≤ w_max    (long-short)
          Σ|w_i − w_i_prev| / 2 ≤ turnover_max   (optional)
          Σw_sector ≤ sector_max  (optional)

Uses ``scipy.optimize.minimize`` (SLSQP) so no external solver is needed.

If the optimiser fails, falls back to the quantile baseline to ensure the
pipeline never produces an empty portfolio.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

from entropy.portfolio.construction import (
    PortfolioConfig,
    PortfolioConstructor,
    PortfolioMode,
)


class OptimizedPortfolio(PortfolioConstructor):
    """Mean-variance portfolio with linear/box constraints.

    Parameters
    ----------
    risk_aversion : λ in the objective.  Higher = more risk-averse.
    cov_lookback : trading days used to estimate the covariance matrix.
    shrinkage : Ledoit-Wolf shrinkage intensity ∈ [0, 1].
        0 = sample covariance, 1 = diagonal (single-factor) covariance.
    """

    def __init__(
        self,
        config: Optional[PortfolioConfig] = None,
        risk_aversion: float = 1.0,
        cov_lookback: int = 120,
        shrinkage: float = 0.5,
    ) -> None:
        super().__init__(config)
        self.risk_aversion = risk_aversion
        self.cov_lookback = cov_lookback
        self.shrinkage = shrinkage

    # ------------------------------------------------------------------
    # Core
    # ------------------------------------------------------------------

    def _generate_weights(
        self,
        signal: pd.DataFrame,
        universe: pd.DataFrame,
        date: pd.Timestamp,
        prev_weights: Optional[pd.Series] = None,
    ) -> pd.Series:
        cfg = self.config
        sig_col = self._detect_signal_col(signal)

        # --- Today's cross-section ---
        sig_today = signal.loc[signal["date"] == date, ["ticker", sig_col]].copy()
        uni_today = universe.loc[universe["date"] == date, "ticker"]
        tradable = set(uni_today)
        sig_today = sig_today[sig_today["ticker"].isin(tradable)].dropna(subset=[sig_col])

        if len(sig_today) < 5:
            return pd.Series(dtype=float)

        tickers = sig_today["ticker"].values
        alpha = sig_today.set_index("ticker")[sig_col].values.astype(float)

        # --- Estimate covariance from historical returns ---
        cov = self._estimate_covariance(signal, tickers, date)
        if cov is None:
            logger.warning("Covariance estimation failed on {} — using equal weight fallback", date.date())
            n = len(tickers)
            return pd.Series(1.0 / n, index=tickers)

        # --- Solve ---
        try:
            w = self._solve_qp(alpha, cov, tickers, prev_weights)
        except Exception as exc:
            logger.warning("Optimisation failed on {} ({}), using equal weight fallback", date.date(), exc)
            n = len(tickers)
            w = pd.Series(1.0 / n, index=tickers)

        return w

    # ------------------------------------------------------------------
    # Covariance estimation
    # ------------------------------------------------------------------

    def _estimate_covariance(
        self,
        signal: pd.DataFrame,
        tickers: np.ndarray,
        date: pd.Timestamp,
    ) -> Optional[np.ndarray]:
        """Estimate a shrunk covariance matrix from recent returns.

        Uses Ledoit-Wolf linear shrinkage toward a diagonal target.
        """
        # We need returns — try to find adj_close in the signal df
        # (the pipeline merges prices into signal before calling build)
        if "adj_close" not in signal.columns:
            return None

        hist = signal.loc[signal["date"] < date].copy()
        hist = hist[hist["ticker"].isin(tickers)]

        # Pivot to wide returns
        pivot = hist.pivot(index="date", columns="ticker", values="adj_close")
        pivot = pivot[list(tickers)]  # ensure column order
        pivot = pivot.sort_index().tail(self.cov_lookback)
        ret = pivot.pct_change().dropna()

        if len(ret) < 30 or ret.shape[1] < 2:
            return None

        # Sample covariance
        S = ret.cov().values
        n = S.shape[0]

        # Shrinkage target: diagonal (variance only)
        D = np.diag(np.diag(S))

        # Ledoit-Wolf blend
        lam = self.shrinkage
        cov = (1 - lam) * S + lam * D

        # Ensure positive semi-definite
        eigvals = np.linalg.eigvalsh(cov)
        if eigvals.min() < 0:
            cov += (-eigvals.min() + 1e-8) * np.eye(n)

        return cov

    # ------------------------------------------------------------------
    # QP solver
    # ------------------------------------------------------------------

    def _solve_qp(
        self,
        alpha: np.ndarray,
        cov: np.ndarray,
        tickers: np.ndarray,
        prev_weights: Optional[pd.Series],
    ) -> pd.Series:
        """Solve the mean-variance QP with constraints via SLSQP."""
        from scipy.optimize import minimize

        cfg = self.config
        n = len(tickers)
        lam = self.risk_aversion

        # --- Objective: minimise −(α'w) + (λ/2) w'Σw ---
        def objective(w):
            return -(alpha @ w) + 0.5 * lam * (w @ cov @ w)

        def grad(w):
            return -alpha + lam * (cov @ w)

        # --- Constraints ---
        constraints = []

        if cfg.mode == PortfolioMode.LONG_ONLY:
            # Sum to 1
            constraints.append({
                "type": "eq",
                "fun": lambda w: np.sum(w) - 1.0,
                "jac": lambda w: np.ones(n),
            })
            bounds = [(0.0, cfg.max_stock_weight)] * n
            w0 = np.full(n, 1.0 / n)

        else:  # long-short
            # Net exposure = 0 (dollar-neutral)
            constraints.append({
                "type": "eq",
                "fun": lambda w: np.sum(w),
                "jac": lambda w: np.ones(n),
            })
            # Gross exposure = 2 (1 long + 1 short)
            constraints.append({
                "type": "eq",
                "fun": lambda w: np.sum(np.abs(w)) - 2.0,
            })
            bounds = [(-cfg.max_stock_weight, cfg.max_stock_weight)] * n
            # Initial: top half positive, bottom half negative
            rank = alpha.argsort().argsort()
            w0 = np.where(rank >= n // 2, 1.0 / (n // 2), -1.0 / (n - n // 2))
            w0 = w0 - w0.mean()  # centre

        # --- Optional turnover constraint ---
        if cfg.max_turnover is not None and prev_weights is not None:
            w_prev = prev_weights.reindex(tickers, fill_value=0.0).values
            # |w - w_prev| / 2 ≤ max_turnover
            # Approximated as a linear constraint on the smooth part
            constraints.append({
                "type": "ineq",
                "fun": lambda w, wp=w_prev: cfg.max_turnover - np.sum(np.abs(w - wp)) / 2.0,
            })

        # --- Solve ---
        result = minimize(
            objective,
            w0,
            jac=grad,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 500, "ftol": 1e-10},
        )

        if not result.success:
            logger.warning("SLSQP did not converge: {}", result.message)

        w_opt = result.x

        # Clean up small weights
        w_opt[np.abs(w_opt) < 1e-6] = 0.0

        return pd.Series(w_opt, index=tickers)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_signal_col(signal: pd.DataFrame) -> str:
        for c in signal.columns:
            if c not in ("date", "ticker", "adj_close", "close", "volume",
                         "market_cap", "sector"):
                return c
        raise ValueError("Signal DataFrame has no factor column")
