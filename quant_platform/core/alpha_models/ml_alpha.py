"""ML-based alpha models with proper time-series cross-validation.

Implements machine learning approaches to alpha generation with rigorous
safeguards against look-ahead bias and overfitting:

1. **Ridge/Lasso Cross-Sectional Regression**
   - Monthly cross-sectional regression: factors → forward returns
   - L1/L2 regularization to prevent overfitting
   - Walk-forward retraining with expanding window

2. **Purged K-Fold CV**
   - Proper time-series cross-validation (Lopez de Prado)
   - Purging: remove samples close to test set
   - Embargo: gap between train and test to prevent leakage

3. **Feature Importance**
   - Permutation importance to identify key factors
   - Regularization path analysis

Key design principles:
- No look-ahead bias: strict train/test separation
- Walk-forward: retrain every period with expanding window
- Regularization: prevent overfitting to noise
- Interpretability: linear models with feature importance
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit

from quant_platform.core.signals.effective import build_effective_signal


def _apply_effective_ml_features(factors: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    """Convert raw ML feature columns to the shared effective signal scale."""
    if "date" not in factors.columns or "ticker" not in factors.columns:
        return factors

    result = factors.copy()
    directions = _infer_feature_directions(feature_cols)
    base_index = result.set_index(["date", "ticker"]).index
    for col in feature_cols:
        if col not in result.columns:
            continue
        eff = build_effective_signal(
            result[["date", "ticker", col]].copy(),
            col,
            direction=directions.get(col, 1),
        )
        result[col] = eff.set_index(["date", "ticker"])[col].reindex(base_index).values
    return result


def _infer_feature_directions(feature_cols: List[str]) -> Dict[str, int]:
    try:
        from quant_platform.core.signals.registry import FactorRegistry

        reg = FactorRegistry()
        reg.discover()
        return {col: reg.get(col).meta.direction for col in feature_cols if col in reg}
    except Exception:
        return {}


def _align_factors_and_returns(
    factors: pd.DataFrame,
    forward_returns: pd.Series,
) -> pd.DataFrame:
    """Attach targets without allowing ambiguous positional alignment."""

    result = factors.copy()
    if isinstance(forward_returns.index, pd.MultiIndex):
        required_names = {"date", "ticker"}
        if not required_names.issubset(set(forward_returns.index.names)):
            raise ValueError("forward_returns MultiIndex must be named date and ticker")
        target = forward_returns.rename("forward_return").reset_index()
        if target.duplicated(["date", "ticker"]).any():
            raise ValueError("forward_returns contains duplicate date/ticker keys")
        if result.duplicated(["date", "ticker"]).any():
            raise ValueError("factors contains duplicate date/ticker keys")
        return result.merge(
            target[["date", "ticker", "forward_return"]],
            on=["date", "ticker"],
            how="inner",
            validate="one_to_one",
        )

    if not forward_returns.index.equals(result.index):
        raise ValueError(
            "Non-MultiIndex forward_returns must have exactly the same index as factors"
        )
    if result.index.has_duplicates:
        raise ValueError("factors/forward_returns row index must be unique")
    result["forward_return"] = forward_returns.reindex(result.index)
    return result


# ===================================================================
# Purged K-Fold Cross-Validation (Lopez de Prado)
# ===================================================================

class PurgedKFold:
    """Purged K-Fold cross-validation for time-series data.
    
    Implements the purging and embargo logic from Lopez de Prado's
    "Advances in Financial Machine Learning" to prevent leakage in
    time-series cross-validation.
    
    Purging: Remove training samples that are too close to test set
    Embargo: Add gap between train and test sets
    
    Parameters
    ----------
    n_splits : int
        Number of folds
    embargo_pct : float
        Percentage of samples to embargo after each test set (default 0.01 = 1%)
    """
    
    def __init__(
        self,
        n_splits: int = 5,
        embargo_pct: float = 0.01,
        *,
        horizon: int = 1,
        purge_days: Optional[int] = None,
    ):
        if horizon <= 0:
            raise ValueError("horizon must be positive")
        if purge_days is not None and purge_days < horizon:
            raise ValueError("purge_days must be at least horizon")
        self.n_splits = n_splits
        self.embargo_pct = embargo_pct
        self.horizon = horizon
        self.purge_days = horizon if purge_days is None else purge_days
    
    def split(self, X: pd.DataFrame, y: pd.Series = None) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Generate train/test indices with purging and embargo.
        
        Uses n_splits+1 equal segments: the first segment is always
        training-only, giving exactly n_splits folds where each fold
        trains on all data before the test block (minus embargo).
        
        Parameters
        ----------
        X : DataFrame with DatetimeIndex
        y : Series (optional)
        
        Yields
        ------
        train_idx, test_idx : arrays of indices
        """
        if self.n_splits <= 0:
            raise ValueError("n_splits must be positive")
        if not 0.0 <= self.embargo_pct < 1.0:
            raise ValueError("embargo_pct must be in [0, 1)")

        if isinstance(X.index, pd.MultiIndex) and "date" in X.index.names:
            row_dates = pd.DatetimeIndex(X.index.get_level_values("date"))
        elif isinstance(X.index, pd.DatetimeIndex):
            row_dates = pd.DatetimeIndex(X.index)
        elif "date" in X.columns:
            row_dates = pd.DatetimeIndex(pd.to_datetime(X["date"]))
        else:
            raise ValueError("PurgedKFold requires a DatetimeIndex or a date column")

        unique_dates = row_dates.unique().sort_values()
        # Divide unique dates, never flattened security rows, into segments.
        test_size = len(unique_dates) // (self.n_splits + 1)
        if test_size <= 0:
            raise ValueError("Not enough unique dates for the requested n_splits")
        embargo_size = max(
            self.purge_days,
            int(np.ceil(len(unique_dates) * self.embargo_pct)),
        )
        
        for i in range(self.n_splits):
            # Test set: segments 1 .. n_splits
            test_start = (i + 1) * test_size
            test_end = (
                test_start + test_size
                if i < self.n_splits - 1
                else len(unique_dates)
            )
            test_dates = unique_dates[test_start:test_end]
            
            # Train set: all data before test set, minus embargo gap
            train_end = max(0, test_start - embargo_size)
            if train_end <= 0:
                continue  # Skip if no training data
            
            train_dates = unique_dates[:train_end]
            train_idx = np.flatnonzero(row_dates.isin(train_dates))
            test_idx = np.flatnonzero(row_dates.isin(test_dates))
            
            yield train_idx, test_idx


# ===================================================================
# Ridge/Lasso Cross-Sectional Regression
# ===================================================================

class MLAlphaModel:
    """ML-based alpha model using regularized linear regression.
    
    This implements a Fama-MacBeth style cross-sectional regression with
    regularization. Each period, we regress stock returns on factor values
    using Ridge/Lasso to prevent overfitting.
    
    Model: R_stock = α + Σ(β_k × factor_k) + ε
    
    Parameters
    ----------
    model_type : str
        "ridge", "lasso", or "elastic_net"
    alpha : float
        Regularization strength (higher = more regularization)
    l1_ratio : float
        ElasticNet mixing parameter (0 = Ridge, 1 = Lasso)
    lookback : int
        Number of periods for training window (default 252 = 1 year)
    refit_freq : int
        Refit model every N periods (default 21 = monthly)
    """
    
    def __init__(
        self,
        model_type: str = "ridge",
        alpha: float = 1.0,
        l1_ratio: float = 0.5,
        lookback: int = 252,
        refit_freq: int = 21,
    ):
        self.model_type = model_type.lower()
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.lookback = lookback
        self.refit_freq = refit_freq
        
        # Model and scaler
        self.model_ = None
        self.scaler_ = StandardScaler()
        self.feature_fill_values_: Optional[pd.Series] = None
        
        # Feature importance
        self.feature_importance_: Optional[pd.Series] = None
        self.feature_names_: Optional[List[str]] = None
        
        # Training history
        self.training_history_: List[Dict] = []
    
    def _create_model(self):
        """Create sklearn model based on model_type."""
        if self.model_type == "ridge":
            return Ridge(alpha=self.alpha, fit_intercept=True)
        elif self.model_type == "lasso":
            return Lasso(alpha=self.alpha, fit_intercept=True, max_iter=2000)
        elif self.model_type == "elastic_net":
            return ElasticNet(
                alpha=self.alpha,
                l1_ratio=self.l1_ratio,
                fit_intercept=True,
                max_iter=2000,
            )
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")
    
    def fit(
        self,
        factors: pd.DataFrame,
        forward_returns: pd.Series,
        dates: Optional[pd.DatetimeIndex] = None,
    ) -> "MLAlphaModel":
        """Fit the model using walk-forward training.
        
        Parameters
        ----------
        factors : DataFrame [date, ticker, factor1, factor2, ...]
            Factor values (features)
        forward_returns : Series [date, ticker] → return
            Forward returns (target)
        dates : DatetimeIndex, optional
            Training dates. If None, use all dates in factors.
        
        Returns
        -------
        self
        """
        # Ensure factors has date column
        if "date" not in factors.columns:
            if isinstance(factors.index, pd.MultiIndex):
                factors = factors.reset_index()
        
        # Align by explicit keys (or an exactly matching row index).  Assigning
        # Series.values here would silently shuffle labels after reindexing.
        factors = _align_factors_and_returns(factors, forward_returns)
        factors = factors.dropna(subset=["forward_return"])
        
        if len(factors) == 0:
            logger.warning("No valid training data after alignment")
            return self
        
        # Get feature columns (exclude date, ticker, forward_return)
        feature_cols = [
            c for c in factors.columns
            if c not in ("date", "ticker", "forward_return")
        ]
        factors = _apply_effective_ml_features(factors, feature_cols)
        factors = factors.dropna(subset=feature_cols)
        if len(factors) == 0:
            logger.warning("No valid training data after effective feature transform")
            return self
        self.feature_names_ = feature_cols
        
        # Convert date to datetime
        factors["date"] = pd.to_datetime(factors["date"])
        
        # Sort by date
        factors = factors.sort_values("date")
        if dates is not None:
            requested_dates = pd.DatetimeIndex(pd.to_datetime(dates))
            factors = factors[factors["date"].isin(requested_dates)]
        if self.lookback is not None and self.lookback > 0 and not factors.empty:
            available_dates = pd.DatetimeIndex(factors["date"].unique()).sort_values()
            keep_dates = available_dates[-self.lookback :]
            factors = factors[factors["date"].isin(keep_dates)]
        if factors.empty:
            logger.warning("No valid training data after date/lookback filtering")
            return self
        
        # Use all data for training (expanding window)
        X = factors[feature_cols].values
        y = factors["forward_return"].values
        self.feature_fill_values_ = factors[feature_cols].median().fillna(0.5)
        
        # Standardize features
        X_scaled = self.scaler_.fit_transform(X)
        
        # Fit model
        self.model_ = self._create_model()
        self.model_.fit(X_scaled, y)
        
        # Store feature importance (coefficients)
        self.feature_importance_ = pd.Series(
            self.model_.coef_,
            index=feature_cols,
        ).abs().sort_values(ascending=False)
        
        logger.info(
            "ML alpha model fit: {}, {} samples, {} features, R²={:.3f}",
            self.model_type,
            len(X),
            len(feature_cols),
            self.model_.score(X_scaled, y),
        )
        
        return self
    
    def predict(self, factors: pd.DataFrame) -> pd.Series:
        """Predict alpha scores for given factors.
        
        Parameters
        ----------
        factors : DataFrame [date, ticker, factor1, factor2, ...]
        
        Returns
        -------
        Series [date, ticker] → alpha_score
        """
        if self.model_ is None:
            raise RuntimeError("Must call fit() first")
        
        # Get feature columns
        feature_cols = self.feature_names_
        
        # Check if all features are present
        missing = set(feature_cols) - set(factors.columns)
        if missing:
            raise ValueError(f"Missing features: {missing}")
        
        # Extract features
        factors = _apply_effective_ml_features(factors.copy(), feature_cols)
        fill_values = (
            self.feature_fill_values_
            if self.feature_fill_values_ is not None
            else pd.Series(0.5, index=feature_cols)
        )
        factors[feature_cols] = factors[feature_cols].fillna(fill_values)
        X = factors[feature_cols].values
        
        # Standardize
        X_scaled = self.scaler_.transform(X)
        
        # Predict
        predictions = self.model_.predict(X_scaled)
        
        # Return as Series with same index as factors
        if isinstance(factors.index, pd.MultiIndex):
            return pd.Series(predictions, index=factors.index, name="ml_alpha")
        else:
            # If factors has date and ticker columns, create MultiIndex
            if "date" in factors.columns and "ticker" in factors.columns:
                idx = pd.MultiIndex.from_arrays(
                    [factors["date"], factors["ticker"]],
                    names=["date", "ticker"],
                )
                return pd.Series(predictions, index=idx, name="ml_alpha")
            else:
                return pd.Series(predictions, index=factors.index, name="ml_alpha")
    
    def get_feature_importance(self, top_n: int = 10) -> pd.Series:
        """Get top N most important features.
        
        Parameters
        ----------
        top_n : int
            Number of top features to return
        
        Returns
        -------
        Series of feature importance (absolute coefficients)
        """
        if self.feature_importance_ is None:
            raise RuntimeError("Must call fit() first")
        
        return self.feature_importance_.head(top_n)
    
    def cross_validate(
        self,
        factors: pd.DataFrame,
        forward_returns: pd.Series,
        n_splits: int = 5,
        embargo_pct: float = 0.01,
        horizon: int = 1,
        purge_days: Optional[int] = None,
    ) -> Dict[str, float]:
        """Perform purged K-fold cross-validation.
        
        Parameters
        ----------
        factors : DataFrame [date, ticker, factor1, factor2, ...]
        forward_returns : Series [date, ticker] → return
        n_splits : int
            Number of CV folds
        embargo_pct : float
            Embargo percentage
        
        Returns
        -------
        Dict with CV metrics (mean_r2, std_r2, mean_ic, std_ic)
        """
        # Prepare data — merge via date/ticker to handle MultiIndex returns
        factors = factors.copy()
        if "date" not in factors.columns and isinstance(factors.index, pd.MultiIndex):
            factors = factors.reset_index()
        factors = _align_factors_and_returns(factors, forward_returns)
        factors = factors.dropna(subset=["forward_return"])
        
        feature_cols = [
            c for c in factors.columns
            if c not in ("date", "ticker", "forward_return")
        ]
        factors = _apply_effective_ml_features(factors, feature_cols)
        factors = factors.dropna(subset=feature_cols)
        
        # Ensure date index
        if "date" in factors.columns:
            factors["date"] = pd.to_datetime(factors["date"])
            factors = factors.set_index("date").sort_index()
        
        X = factors[feature_cols]
        y = factors["forward_return"]
        
        # Purged K-Fold
        pkf = PurgedKFold(
            n_splits=n_splits,
            embargo_pct=embargo_pct,
            horizon=horizon,
            purge_days=purge_days,
        )
        
        r2_scores = []
        ic_scores = []
        
        for train_idx, test_idx in pkf.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            # Fit model
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            model = self._create_model()
            model.fit(X_train_scaled, y_train)
            
            # Evaluate
            r2 = model.score(X_test_scaled, y_test)
            r2_scores.append(r2)
            
            # Average daily cross-sectional RankIC; a pooled correlation over
            # security-date rows is not a cross-sectional alpha metric.
            y_pred = model.predict(X_test_scaled)
            evaluation = pd.DataFrame({
                "date": pd.to_datetime(X_test.index),
                "actual": y_test.to_numpy(),
                "prediction": y_pred,
            })
            daily_values = []
            for _, group in evaluation.groupby("date", sort=True):
                valid = group[["actual", "prediction"]].dropna()
                if len(valid) >= 5:
                    daily_values.append(
                        valid["actual"].corr(valid["prediction"], method="spearman")
                    )
            ic = float(pd.Series(daily_values, dtype=float).mean()) if daily_values else np.nan
            ic_scores.append(ic)
        
        return {
            "mean_r2": np.mean(r2_scores),
            "std_r2": np.std(r2_scores),
            "mean_ic": np.mean(ic_scores),
            "std_ic": np.std(ic_scores),
            "n_splits": len(r2_scores),
        }


# ===================================================================
# Walk-Forward Alpha Generator
# ===================================================================

class WalkForwardMLAlpha:
    """Walk-forward ML alpha generation with periodic retraining.
    
    This implements a production-ready ML alpha pipeline:
    1. Expanding window training (use all historical data)
    2. Periodic retraining (e.g., monthly)
    3. Strict train/test separation (no look-ahead)
    
    Parameters
    ----------
    model_type : str
        "ridge", "lasso", or "elastic_net"
    alpha : float
        Regularization strength
    refit_freq : int
        Refit every N days (default 21 = monthly)
    min_train_periods : int
        Minimum training periods before first prediction (default 252 = 1 year)
    """
    
    def __init__(
        self,
        model_type: str = "ridge",
        alpha: float = 1.0,
        refit_freq: int = 21,
        min_train_periods: int = 252,
        horizon: int = 1,
        purge_days: Optional[int] = None,
    ):
        if refit_freq <= 0 or min_train_periods <= 0:
            raise ValueError("refit_freq and min_train_periods must be positive")
        if horizon <= 0:
            raise ValueError("horizon must be positive")
        if purge_days is not None and purge_days < horizon:
            raise ValueError("purge_days must be at least horizon")
        self.model_type = model_type
        self.alpha = alpha
        self.refit_freq = refit_freq
        self.min_train_periods = min_train_periods
        self.horizon = horizon
        self.purge_days = horizon if purge_days is None else purge_days
        
        self.models_: Dict[pd.Timestamp, MLAlphaModel] = {}
        self.feature_importance_history_: List[pd.Series] = []
    
    def generate_alpha(
        self,
        factors: pd.DataFrame,
        forward_returns: pd.Series,
    ) -> pd.Series:
        """Generate alpha scores using walk-forward training.
        
        Parameters
        ----------
        factors : DataFrame [date, ticker, factor1, factor2, ...]
        forward_returns : Series [date, ticker] → return
            Forward returns for training (not used for prediction)
        
        Returns
        -------
        Series [date, ticker] → alpha_score
        """
        # Prepare data
        factors = factors.copy()
        if "date" not in factors.columns:
            if isinstance(factors.index, pd.MultiIndex):
                factors = factors.reset_index()
        
        factors["date"] = pd.to_datetime(factors["date"])
        if not isinstance(forward_returns.index, pd.MultiIndex):
            raise ValueError(
                "WalkForwardMLAlpha requires date/ticker MultiIndex forward_returns"
            )
        
        # Get unique dates
        dates = sorted(factors["date"].unique())
        
        predictions = []
        last_refit_date = None
        current_model = None
        
        for i, date in enumerate(dates):
            # Check if we have enough training data
            if i < self.min_train_periods + self.purge_days:
                continue
            
            # Check if we need to refit
            should_refit = (
                current_model is None or
                last_refit_date is None or
                (i - dates.index(last_refit_date)) >= self.refit_freq
            )
            
            if should_refit:
                # A training label is usable only when its complete horizon
                # precedes the current prediction date.
                train_dates = dates[: i - self.purge_days]
                train_data = factors[factors["date"].isin(train_dates)]
                train_returns = forward_returns[
                    forward_returns.index.get_level_values("date").isin(train_dates)
                ]
                
                # Fit model
                current_model = MLAlphaModel(
                    model_type=self.model_type,
                    alpha=self.alpha,
                    lookback=len(train_dates),
                )
                current_model.fit(train_data, train_returns)
                if current_model.model_ is None:
                    raise RuntimeError("ML alpha refit produced no usable model")
                
                # Store model and feature importance
                self.models_[date] = current_model
                self.feature_importance_history_.append(
                    current_model.feature_importance_
                )
                
                last_refit_date = date
                
                logger.info(
                    "Retrained ML alpha model on {} (train size: {})",
                    date.strftime("%Y-%m-%d"),
                    len(train_data),
                )
            
            # Predict for current date
            if current_model is not None:
                current_data = factors[factors["date"] == date]
                pred = current_model.predict(current_data)
                predictions.append(pred)
        
        # Combine all predictions
        if not predictions:
            return pd.Series(dtype=float)
        
        return pd.concat(predictions)
    
    def get_average_feature_importance(self, top_n: int = 10) -> pd.Series:
        """Get average feature importance across all trained models.
        
        Parameters
        ----------
        top_n : int
            Number of top features to return
        
        Returns
        -------
        Series of average feature importance
        """
        if not self.feature_importance_history_:
            raise RuntimeError("No models trained yet")
        
        # Average across all models
        avg_importance = pd.concat(
            self.feature_importance_history_, axis=1
        ).mean(axis=1).sort_values(ascending=False)
        
        return avg_importance.head(top_n)
