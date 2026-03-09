"""Simple factor covariance risk model."""
from __future__ import annotations
from typing import Optional
import numpy as np
import pandas as pd
from loguru import logger


class FactorRiskModel:
    """Estimate factor covariance matrix from historical returns.

    Uses exponentially-weighted sample covariance with Ledoit-Wolf shrinkage.
    """

    def __init__(self, halflife: int = 60, shrinkage: float = 0.5):
        self.halflife = halflife
        self.shrinkage = shrinkage
        self.cov_matrix_: Optional[np.ndarray] = None
        self.factor_names_: Optional[list] = None

    def fit(self, factor_returns: pd.DataFrame) -> "FactorRiskModel":
        """Estimate factor covariance from a returns DataFrame.

        Parameters
        ----------
        factor_returns : DataFrame where each column is a factor return series
        """
        self.factor_names_ = list(factor_returns.columns)

        # Exponential weights
        n = len(factor_returns)
        alpha = 1 - np.exp(-np.log(2) / self.halflife)
        weights = np.array([(1 - alpha) ** (n - 1 - i) for i in range(n)])
        weights /= weights.sum()

        # Weighted sample covariance
        data = factor_returns.values
        mean = np.average(data, axis=0, weights=weights)
        centered = data - mean
        sample_cov = (centered.T * weights) @ centered

        # Ledoit-Wolf shrinkage toward diagonal
        diag = np.diag(np.diag(sample_cov))
        self.cov_matrix_ = (1 - self.shrinkage) * sample_cov + self.shrinkage * diag

        logger.info("Risk model fit: %d factors, condition number %.1f",
                     len(self.factor_names_), np.linalg.cond(self.cov_matrix_))
        return self

    def predict_risk(self, weights: np.ndarray) -> float:
        """Predict portfolio variance given weight vector."""
        if self.cov_matrix_ is None:
            raise RuntimeError("Must call fit() first.")
        return float(weights @ self.cov_matrix_ @ weights)
