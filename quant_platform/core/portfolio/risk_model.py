"""Barra-style factor risk model with decomposition.

Implements a multi-factor risk model that decomposes portfolio risk into:
- Factor risk: systematic risk from market, size, sector exposures
- Specific risk: idiosyncratic risk from stock-specific returns

Model structure:
    Stock Return = β_mkt × R_mkt + β_size × R_size + β_sector × R_sector + ε
    Portfolio Variance = b' F b + Σ_stock(weight² × specific_risk²)
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
        self.intercepts_: Optional[pd.Series] = None
        
        # Factor exposures (betas)
        self.exposures_: Optional[pd.DataFrame] = None  # [ticker, factor] → beta
        
        # Specific risk
        self.specific_risk_: Optional[pd.Series] = None  # ticker → σ_specific
        self.factor_returns_: Optional[pd.DataFrame] = None

    @staticmethod
    def build_structural_factors(stock_returns: pd.DataFrame, market_caps: pd.DataFrame) -> pd.DataFrame:
        """Market and SMB returns independent of the candidate alpha.

        Size is equal-weight small minus big, split at the lagged cross-section
        median log market cap. Caps are shifted before aligning with returns,
        so today's cap (or a future cap) cannot choose today's size sleeves.
        Sector excess returns are added by fit(). No value factor is invented
        when point-in-time fundamentals are unavailable.
        """
        returns = stock_returns.replace([np.inf, -np.inf], np.nan).sort_index()
        caps = market_caps.sort_index().reindex(columns=returns.columns)
        log_caps = np.log(caps.where(caps > 0)).shift(1).reindex(returns.index)
        ranks = log_caps.rank(axis=1, method="average", pct=True)
        factors = pd.DataFrame(index=returns.index)
        factors["market"] = returns.mean(axis=1)
        factors["size"] = returns.where(ranks <= 0.5).mean(axis=1) - returns.where(ranks > 0.5).mean(axis=1)
        return factors

    @staticmethod
    def _sector_factors(stock_returns, factor_returns, sector_map):
        """Append K-1 sector excess returns, omitting a reference sector.

        A static map is assumed valid throughout the estimation window. Users
        must supply dated historical classifications for point-in-time claims.
        """
        factors = factor_returns.copy()
        if sector_map is None or sector_map.empty:
            return factors
        if sector_map["ticker"].duplicated().any():
            raise ValueError("sector_map must have one row per ticker")
        mapping = sector_map.set_index("ticker")["sector"].reindex(stock_returns.columns)
        if mapping.isna().any():
            raise ValueError("sector_map is incomplete for the fitted stocks")
        sectors = sorted(mapping.unique(), key=str)
        market = factors.get("market", stock_returns.mean(axis=1))
        for sector in sectors[1:]:
            members = mapping.index[mapping.eq(sector)]
            factors[f"sector:{sector}"] = stock_returns[members].mean(axis=1) - market
        return factors

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
        if "alpha_spread" in factor_returns:
            raise ValueError("Candidate alpha_spread must not be a risk factor")
        factor_returns = self._sector_factors(stock_returns, factor_returns, sector_map)
        self.factor_returns_ = factor_returns
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
            "Factor risk model fit: {} factors, {} stocks, condition number {:.1f}",
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
        intercepts = {}
        
        for ticker in stock_ret.columns:
            aligned = pd.concat(
                [stock_ret[ticker].rename("_stock"), factor_ret], axis=1
            ).replace([np.inf, -np.inf], np.nan).dropna()
            if len(aligned) < 20:
                continue
            y = aligned["_stock"]
            X = aligned[factor_ret.columns]
            
            # OLS regression: y = X @ beta + epsilon
            try:
                # Add intercept
                X_with_const = np.column_stack([np.ones(len(X)), X.values])
                beta = np.linalg.lstsq(X_with_const, y.values, rcond=None)[0]
                intercepts[ticker] = float(beta[0])
                
                # Store betas (exclude intercept)
                exposure_dict = {"ticker": ticker}
                for i, factor_name in enumerate(factor_ret.columns):
                    exposure_dict[factor_name] = beta[i + 1]
                
                exposures.append(exposure_dict)
            except np.linalg.LinAlgError:
                logger.warning("Regression failed for {}", ticker)
                continue
        
        self.intercepts_ = pd.Series(intercepts, dtype=float)
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
        clean_returns = factor_returns.replace(
            [np.inf, -np.inf], np.nan
        ).dropna()
        n = len(clean_returns)
        if n < 2:
            raise ValueError("Need at least two complete factor-return rows")
        alpha = 1 - np.exp(-np.log(2) / self.halflife)
        weights = np.array([(1 - alpha) ** (n - 1 - i) for i in range(n)])
        weights /= weights.sum()

        # Weighted sample covariance
        data = clean_returns.values
        mean = np.average(data, axis=0, weights=weights)
        centered = data - mean
        sample_cov = (centered.T * weights) @ centered

        # Ledoit-Wolf shrinkage toward diagonal
        diag = np.diag(np.diag(sample_cov))
        cov = (1 - self.shrinkage) * sample_cov + self.shrinkage * diag
        
        # Inputs are daily returns; store annualized covariance so it is in
        # the same units as ``specific_risk_`` below.
        return cov * 252.0
    
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
            
            aligned = pd.concat(
                [stock_ret[ticker].rename("_stock"), factor_ret], axis=1
            ).replace([np.inf, -np.inf], np.nan).dropna()
            if len(aligned) < 2:
                continue
            y = aligned["_stock"]
            X = aligned[factor_ret.columns]
            
            # Get exposures for this stock
            betas = exposures.loc[ticker].values
            
            # Predicted return from factors
            intercept = (
                float(self.intercepts_.get(ticker, 0.0))
                if self.intercepts_ is not None
                else 0.0
            )
            y_pred = intercept + X.values @ betas
            
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
        
        if not isinstance(weights, pd.Series):
            if self.exposures_ is None or len(weights) != len(self.exposures_):
                raise ValueError(
                    "Array weights must match fitted exposures; prefer a ticker-indexed Series"
                )
            weights = pd.Series(weights, index=self.exposures_.index)
        decomposition = self.decompose_risk(weights)
        return float(
            decomposition["factor_variance"]
            + decomposition["specific_variance"]
        )
    
    def decompose_risk(
        self,
        weights: pd.Series,
    ) -> Dict[str, float]:
        """Decompose portfolio risk into factor and specific components.

        Portfolio Variance = Factor Risk + Specific Risk
        Factor Risk = b' F b (including factor cross-covariances)
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
        
        active_weights = weights[weights.abs() > 1e-12]
        missing = active_weights.index.difference(
            self.exposures_.index.intersection(self.specific_risk_.index)
        )
        if len(missing):
            raise ValueError(
                "Risk model has no complete exposure/specific-risk estimate for: "
                + ", ".join(map(str, missing[:10]))
            )
        common_tickers = active_weights.index
        w = active_weights
        
        if len(w) == 0:
            return {
                "total_risk": 0.0,
                "factor_risk": 0.0,
                "specific_risk": 0.0,
                "factor_contributions": {},
                "factor_variance": 0.0,
                "specific_variance": 0.0,
                "specific_contributions": {},
                "volatility_contributions": {},
                "portfolio_exposures": {},
                "euler_variance_error": 0.0,
                "euler_volatility_error": 0.0,
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
        
        specific_contributions = (w**2 * self.specific_risk_.loc[common_tickers]**2).to_dict()
        # Euler contributions to volatility add to total volatility; standalone
        # factor/specific volatilities above do not add linearly.
        volatility_contributions = {
            **{f"factor:{k}": v / total_risk if total_risk > 0 else 0.0 for k, v in factor_contributions.items()},
            **{f"specific:{k}": v / total_risk if total_risk > 0 else 0.0 for k, v in specific_contributions.items()},
        }
        variance_error = sum(factor_contributions.values()) + sum(specific_contributions.values()) - total_variance
        volatility_error = sum(volatility_contributions.values()) - total_risk
        if not np.isclose(variance_error, 0, atol=1e-10) or not np.isclose(volatility_error, 0, atol=1e-10):
            raise RuntimeError("Euler risk contributions do not add to total risk")
        return {
            "total_risk": float(total_risk),
            "factor_risk": float(np.sqrt(factor_variance)),
            "specific_risk": float(np.sqrt(specific_variance)),
            "factor_variance": float(factor_variance),
            "specific_variance": float(specific_variance),
            "factor_contributions": factor_contributions,
            "specific_contributions": specific_contributions,
            "volatility_contributions": volatility_contributions,
            "euler_variance_error": float(variance_error),
            "euler_volatility_error": float(volatility_error),
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
