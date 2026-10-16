"""Advanced portfolio construction: constrained mean-variance optimisation.

Solves the classic Markowitz problem with practical constraints:

    max   α'w − (λ/2) w'Σw
    s.t.  Σw_i = 1   (long-only)  or  Σw_long=1, Σw_short=−1
          0 ≤ w_i ≤ w_max         (long-only)
          −w_max ≤ w_i ≤ w_max    (long-short)
          Σ|w_i − w_i_prev| / 2 ≤ turnover_max   (optional)
          Σw_sector ≤ sector_max  (optional)

Uses ``scipy.optimize.minimize`` (SLSQP) so no external solver is needed.

Alpha and covariance use daily return units. Estimation or solver failures
raise PortfolioOptimizationError; only the calling pipeline may opt into a
recorded fallback.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

from quant_platform.core.portfolio.construction import (
    PortfolioConfig,
    PortfolioConstructor,
    PortfolioMode,
)


class PortfolioOptimizationError(RuntimeError):
    """Risk estimation or constrained portfolio optimisation failed."""


class OptimizedPortfolio(PortfolioConstructor):
    """Mean-variance portfolio with linear/box constraints.

    Parameters
    ----------
    risk_aversion : λ in the objective.  Higher = more risk-averse.
    cov_lookback : trading days used to estimate the covariance matrix.
    shrinkage : Ledoit-Wolf shrinkage intensity ∈ [0, 1].
        0 = sample covariance, 1 = diagonal (single-factor) covariance.
    expected_alpha_annual : assumed annual top-to-bottom cross-sectional
        expected-return spread. Scores are converted to centred percentile
        ranks and multiplied by this spread / 252. This is a configurable
        prior, not a return forecast estimated from the backtest.
    """

    def __init__(
        self,
        config: Optional[PortfolioConfig] = None,
        risk_aversion: float = 1.0,
        cov_lookback: int = 120,
        shrinkage: float = 0.5,
        use_factor_risk: bool = True,
        factor_risk_halflife: int = 60,
        factor_risk_shrinkage: float = 0.5,
        turnover_penalty: float = 0.0,
        expected_alpha_annual: float = 0.10,
    ) -> None:
        super().__init__(config)
        self.risk_aversion = risk_aversion
        self.cov_lookback = cov_lookback
        self.shrinkage = shrinkage
        self.use_factor_risk = use_factor_risk
        self.factor_risk_halflife = factor_risk_halflife
        self.factor_risk_shrinkage = factor_risk_shrinkage
        self.turnover_penalty = turnover_penalty
        if not np.isfinite(expected_alpha_annual) or expected_alpha_annual <= 0:
            raise ValueError("expected_alpha_annual must be finite and positive")
        if not np.isfinite(risk_aversion) or risk_aversion < 0:
            raise ValueError("risk_aversion must be finite and non-negative")
        self.expected_alpha_annual = expected_alpha_annual
        self.sector_map = None
        self.risk_factor_names_ = []

    def build(self, signal, universe, rebalance_dates, sector_map=None):
        self.sector_map = sector_map
        return super().build(signal, universe, rebalance_dates, sector_map=sector_map)

    def _daily_expected_returns(self, scores: np.ndarray) -> np.ndarray:
        """Map scores to a daily return prior with the configured full spread."""
        ranks = pd.Series(scores).rank(method="average").to_numpy()
        return (ranks - (len(ranks) + 1) / 2) / max(len(ranks) - 1, 1) * self.expected_alpha_annual / 252.0

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
        uni_today = universe.loc[universe["date"] == date].copy()
        if "pass_all_filters" in uni_today.columns:
            uni_today = uni_today.loc[
                uni_today["pass_all_filters"].fillna(False).astype(bool)
            ]
        tradable = set(uni_today["ticker"])
        sig_today = sig_today[sig_today["ticker"].isin(tradable)].dropna(subset=[sig_col])

        if len(sig_today) < 5:
            return pd.Series(dtype=float)

        tickers = sig_today["ticker"].values
        alpha = self._daily_expected_returns(sig_today[sig_col].to_numpy(dtype=float))

        # --- Estimate covariance from historical returns ---
        try:
            cov = self._estimate_covariance(signal, tickers, date, sig_col=sig_col)
            if cov is None:
                raise PortfolioOptimizationError(f"Insufficient covariance history on {date.date()}")
            return self._solve_qp(alpha, cov, tickers, prev_weights)
        except PortfolioOptimizationError:
            raise
        except Exception as exc:
            raise PortfolioOptimizationError(f"Optimisation failed on {date.date()}: {exc}") from exc

    # ------------------------------------------------------------------
    # Covariance estimation
    # ------------------------------------------------------------------

    def _estimate_covariance(
        self,
        signal: pd.DataFrame,
        tickers: np.ndarray,
        date: pd.Timestamp,
        sig_col: Optional[str] = None,
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
        if "is_tradable" in hist:
            hist.loc[~hist["is_tradable"].fillna(False).astype(bool), "adj_close"] = np.nan

        # Pivot to wide returns
        pivot = hist.pivot(index="date", columns="ticker", values="adj_close")
        pivot = pivot.reindex(columns=tickers)  # ensure column order
        pivot = pivot.sort_index().tail(self.cov_lookback)
        ret = pivot.pct_change(fill_method=None).dropna()

        if len(ret) < 30 or ret.shape[1] < 2:
            return None

        if self.use_factor_risk:
            return self._estimate_factor_risk_covariance(hist, ret, tickers, sig_col)

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

    def _estimate_factor_risk_covariance(
        self,
        hist: pd.DataFrame,
        stock_returns: pd.DataFrame,
        tickers: np.ndarray,
        sig_col: Optional[str],
    ) -> Optional[np.ndarray]:
        """Build a stock covariance matrix from the factor risk model.

        The optimizer consumes a stock-level covariance matrix, so this method
        fits ``FactorRiskModel`` on recent stock returns and reconstructs
        ``B F B' + diag(specific_var)``. Missing model inputs or incomplete
        estimates are errors, never an implicit switch to another risk model.
        """
        from quant_platform.core.portfolio.risk_model import FactorRiskModel

        if "market_cap" not in hist or self.sector_map is None or self.sector_map.empty:
            raise PortfolioOptimizationError(
                "Structural factor risk requires historical market_cap and sector_map; "
                "supply them or explicitly set use_factor_risk=False"
            )
        caps = hist.pivot(index="date", columns="ticker", values="market_cap")
        factor_returns = FactorRiskModel.build_structural_factors(stock_returns, caps)

        model = FactorRiskModel(
            halflife=self.factor_risk_halflife,
            shrinkage=self.factor_risk_shrinkage,
        )
        sector_map = self.sector_map
        if self.config.missing_sector_policy == "unknown_bucket":
            sector_map = sector_map.set_index("ticker").reindex(tickers).fillna({"sector": "Unknown"}).reset_index(names="ticker")
        model.fit(stock_returns, factor_returns, sector_map=sector_map)
        self.risk_factor_names_ = model.factor_names_
        if model.exposures_ is None or model.exposures_.empty or model.cov_matrix_ is None:
            return None

        exposures = model.exposures_.reindex(tickers)
        if exposures.isna().any().any() or model.specific_risk_.reindex(tickers).isna().any():
            raise PortfolioOptimizationError("Incomplete structural factor risk estimates")
        if exposures.empty:
            return None

        b = exposures.to_numpy(dtype=float)
        # FactorRiskModel exposes annualized risk; this optimizer otherwise
        # uses daily stock covariance, so convert both components to daily.
        factor_cov = model.cov_matrix_ / 252.0
        spec = model.specific_risk_.reindex(tickers)
        spec_daily_var = (spec.to_numpy(dtype=float) / np.sqrt(252.0)) ** 2

        cov = b @ factor_cov @ b.T + np.diag(spec_daily_var)
        eigvals = np.linalg.eigvalsh(cov)
        if eigvals.min() < 0:
            cov += (-eigvals.min() + 1e-8) * np.eye(cov.shape[0])
        logger.debug("Using factor risk covariance with {} factors", len(model.factor_names_ or []))
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
        """Solve with alpha in expected daily returns and cov in daily variance.

        Increasing lambda reduces optimal variance for a fixed convex feasible
        set; maximum single-name weight need not be monotone for general cov.
        """
        from scipy.optimize import minimize

        cfg = self.config
        n = len(tickers)
        lam = self.risk_aversion

        # --- Objective: minimise −(α'w) + (λ/2) w'Σw ---
        def objective(w):
            obj = -(alpha @ w) + 0.5 * lam * (w @ cov @ w)
            if self.turnover_penalty > 0 and prev_weights is not None:
                w_prev = prev_weights.reindex(tickers, fill_value=0.0).values
                obj += self.turnover_penalty * np.sum((w - w_prev) ** 2)
            return obj

        def grad(w):
            g = -alpha + lam * (cov @ w)
            if self.turnover_penalty > 0 and prev_weights is not None:
                w_prev = prev_weights.reindex(tickers, fill_value=0.0).values
                g += 2.0 * self.turnover_penalty * (w - w_prev)
            return g

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
            raise PortfolioOptimizationError(f"SLSQP did not converge: {result.message}")

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
                         "market_cap", "sector", "is_tradable"):
                return c
        raise ValueError("Signal DataFrame has no factor column")
