"""Barra-style factor risk model with decomposition.

Implements a multi-factor risk model that decomposes portfolio risk into:
- Factor risk: systematic risk from market, size, sector exposures
- Specific risk: idiosyncratic risk from stock-specific returns

Model structure:
    Stock Return = β_mkt × R_mkt + β_size × R_size + β_sector × R_sector + ε
    Portfolio Risk = Σ_factor(exposure² × variance) + Σ_stock(weight² × specific_risk²)
"""
from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import stats
from loguru import logger


class FactorRiskModel:
    """Barra-style factor risk model with exposure estimation and decomposition.

    This model estimates:
    1. Factor exposures (betas) for each stock via regression
    2. Factor covariance matrix (systematic risk)
    3. Specific risk (idiosyncratic risk) from regression residuals
    4. Portfolio risk decomposition into factor and specific components

    Parameters
    ----------
    halflife : int
        Half-life for exponential weighting (default 60 days)
    shrinkage : float
        Ledoit-Wolf shrinkage intensity [0, 1] (default 0.5)
    """

    def __init__(self, halflife: int = 60, shrinkage: float = 0.5):
        self.halflife = halflife
        self.shrinkage = shrinkage
        
        # Factor covariance
        self.cov_matrix_: Optional[np.ndarray] = None
        self.factor_names_: Optional[List[str]] = None
        
        # Factor exposures (betas)
        self.exposures_: Optional[pd.DataFrame] = None  # [ticker, factor] → beta
        
        # Specific risk
        self.specific_risk_: Optional[pd.Series] = None  # ticker → σ_specific

    def fit(
        self,
        stock_returns: pd.DataFrame,
        factor_returns: pd.DataFrame,
        sector_map: Optional[pd.DataFrame] = None,
    ) -> "FactorRiskModel":
        """Fit the factor risk model.

        Parameters
        ----------
        stock_returns : DataFrame [date, ticker] → return
            Stock returns in wide format (columns = tickers)
        factor_returns : DataFrame [date, factor] → return
            Factor returns (e.g., market, size, value)
        sector_map : DataFrame [ticker, sector], optional
            Sector classification for sector factors

        Returns
        -------
        self
        """
        # 1. Estimate factor exposures (betas) for each stock
        self.exposures_ = self._estimate_exposures(
            stock_returns, factor_returns, sector_map
        )
        
        # 2. Estimate factor covariance matrix
        self.factor_names_ = list(factor_returns.columns)
        self.cov_matrix_ = self._estimate_factor_covariance(factor_returns)
        
        # 3. Estimate specific risk from regression residuals
        self.specific_risk_ = self._estimate_specific_risk(
            stock_returns, factor_returns, self.exposures_
        )
        
        logger.info(
            "Factor risk model fit: %d factors, %d stocks, condition number %.1f",
            len(self.factor_names_),
            len(self.exposures_),
            np.linalg.cond(self.cov_matrix_),
        )
        return self

    def _estimate_exposures(
        self,
        stock_returns: pd.DataFrame,
        factor_returns: pd.DataFrame,
        sector_map: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Estimate factor exposures (betas) via time-series regression.

        For each stock, run regression:
            R_stock = α + β_mkt × R_mkt + β_size × R_size + ... + ε

        Returns
        -------
        DataFrame [ticker, factor] → beta
        """
        # Align dates
        common_dates = stock_returns.index.intersection(factor_returns.index)
        stock_ret = stock_returns.loc[common_dates]
        factor_ret = factor_returns.loc[common_dates]
        
        exposures = []
        
        for ticker in stock_ret.columns:
            y = stock_ret[ticker].dropna()
            
            # Align factor returns to stock dates
            X = factor_ret.loc[y.index]
            
            if len(y) < 20:  # Need minimum observations
                continue
            
            # OLS regression: y = X @ beta + epsilon
            try:
                # Add intercept
                X_with_const = np.column_stack([np.ones(len(X)), X.values])
                beta = np.linalg.lstsq(X_with_const, y.values, rcond=None)[0]
                
                # Store betas (exclude intercept)
                exposure_dict = {"ticker": ticker}
                for i, factor_name in enumerate(factor_ret.columns):
                    exposure_dict[factor_name] = beta[i + 1]
                
                exposures.append(exposure_dict)
            except np.linalg.LinAlgError:
                logger.warning("Regression failed for %s", ticker)
                continue
        
        if not exposures:
            return pd.DataFrame()
        
        exp_df = pd.DataFrame(exposures).set_index("ticker")
        return exp_df
    
    def _estimate_factor_covariance(self, factor_returns: pd.DataFrame) -> np.ndarray:
        """Estimate factor covariance matrix with EWMA and shrinkage.

        Returns
        -------
        Covariance matrix (K × K) where K = number of factors
        """
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
        cov = (1 - self.shrinkage) * sample_cov + self.shrinkage * diag
        
        return cov
    
    def _estimate_specific_risk(
        self,
        stock_returns: pd.DataFrame,
        factor_returns: pd.DataFrame,
        exposures: pd.DataFrame,
    ) -> pd.Series:
        """Estimate specific risk (idiosyncratic volatility) from regression residuals.

        Specific risk = std(ε) where ε = R_stock - (β × R_factors)

        Returns
        -------
        Series [ticker] → specific_risk (annualized)
        """
        common_dates = stock_returns.index.intersection(factor_returns.index)
        stock_ret = stock_returns.loc[common_dates]
        factor_ret = factor_returns.loc[common_dates]
        
        specific_risks = {}
        
        for ticker in exposures.index:
            if ticker not in stock_ret.columns:
                continue
            
            y = stock_ret[ticker].dropna()
            X = factor_ret.loc[y.index]
            
            # Get exposures for this stock
            betas = exposures.loc[ticker].values
            
            # Predicted return from factors
            y_pred = X.values @ betas
            
            # Residuals
            residuals = y.values - y_pred
            
            # Specific risk = std(residuals), annualized
            specific_vol = np.std(residuals, ddof=1) * np.sqrt(252)
            specific_risks[ticker] = specific_vol
        
        return pd.Series(specific_risks)
    
    def predict_risk(self, weights: pd.Series) -> float:
        """Predict portfolio variance given weight vector.

        Uses simple stock-level covariance (legacy method).
        For factor decomposition, use decompose_risk().

        Parameters
        ----------
        weights : Series [ticker] → weight

        Returns
        -------
        Portfolio variance
        """
        if self.cov_matrix_ is None:
            raise RuntimeError("Must call fit() first.")
        
        # This is a simplified version - for full decomposition use decompose_risk()
        w = weights.values if isinstance(weights, pd.Series) else weights
        return float(w @ self.cov_matrix_ @ w)
    
    def decompose_risk(
        self,
        weights: pd.Series,
    ) -> Dict[str, float]:
        """Decompose portfolio risk into factor and specific components.

        Portfolio Variance = Factor Risk + Specific Risk
        Factor Risk = Σ_k (portfolio_exposure_k² × factor_variance_k)
        Specific Risk = Σ_i (weight_i² × specific_risk_i²)

        Parameters
        ----------
        weights : Series [ticker] → weight
            Portfolio weights

        Returns
        -------
        Dict with:
            - total_risk: portfolio volatility (annualized)
            - factor_risk: systematic risk contribution
            - specific_risk: idiosyncratic risk contribution
            - factor_contributions: dict of {factor_name: contribution}
        """
        if self.exposures_ is None or self.specific_risk_ is None:
            raise RuntimeError("Must call fit() first.")
        
        # Align weights to stocks with exposures
        common_tickers = weights.index.intersection(self.exposures_.index)
        w = weights.loc[common_tickers]
        
        if len(w) == 0:
            return {
                "total_risk": 0.0,
                "factor_risk": 0.0,
                "specific_risk": 0.0,
                "factor_contributions": {},
            }
        
        # 1. Portfolio factor exposures = weighted average of stock exposures
        portfolio_exposures = (w.values[:, None] * self.exposures_.loc[common_tickers].values).sum(axis=0)
        
        # 2. Factor risk = portfolio_exposures' @ factor_cov @ portfolio_exposures
        factor_variance = portfolio_exposures @ self.cov_matrix_ @ portfolio_exposures
        
        # 3. Specific risk = Σ(weight_i² × specific_risk_i²)
        specific_variance = 0.0
        for ticker in common_tickers:
            if ticker in self.specific_risk_.index:
                specific_variance += w[ticker]**2 * self.specific_risk_[ticker]**2
        
        # 4. Total risk
        total_variance = factor_variance + specific_variance
        total_risk = np.sqrt(total_variance)
        
        # 5. Factor contributions (marginal contribution of each factor)
        # contribution_k = b_k × (F @ b)_k  so that Σ contributions = b' F b
        Fb = self.cov_matrix_ @ portfolio_exposures
        factor_contributions = {}
        for i, factor_name in enumerate(self.factor_names_):
            contribution = portfolio_exposures[i] * Fb[i]
            factor_contributions[factor_name] = float(contribution)
        
        return {
            "total_risk": float(total_risk),
            "factor_risk": float(np.sqrt(factor_variance)),
            "specific_risk": float(np.sqrt(specific_variance)),
            "factor_variance": float(factor_variance),
            "specific_variance": float(specific_variance),
            "factor_contributions": factor_contributions,
            "portfolio_exposures": dict(zip(self.factor_names_, portfolio_exposures)),
        }
    
    def get_exposures(self, tickers: Optional[List[str]] = None) -> pd.DataFrame:
        """Get factor exposures for specified tickers.

        Parameters
        ----------
        tickers : list of str, optional
            Tickers to retrieve. If None, return all.

        Returns
        -------
        DataFrame [ticker, factor] → beta
        """
        if self.exposures_ is None:
            raise RuntimeError("Must call fit() first.")
        
        if tickers is None:
            return self.exposures_
        
        return self.exposures_.loc[self.exposures_.index.intersection(tickers)]
    
    def get_specific_risk(self, tickers: Optional[List[str]] = None) -> pd.Series:
        """Get specific risk for specified tickers.

        Parameters
        ----------
        tickers : list of str, optional
            Tickers to retrieve. If None, return all.

        Returns
        -------
        Series [ticker] → specific_risk (annualized)
        """
        if self.specific_risk_ is None:
            raise RuntimeError("Must call fit() first.")
        
        if tickers is None:
            return self.specific_risk_
        
        return self.specific_risk_.loc[self.specific_risk_.index.intersection(tickers)]
