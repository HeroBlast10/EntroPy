"""Unit tests for ML-based alpha models."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_platform.core.alpha_models.ml_alpha import (
    MLAlphaModel,
    PurgedKFold,
    WalkForwardMLAlpha,
)


# ===================================================================
# Test data fixtures
# ===================================================================

@pytest.fixture
def sample_factor_data():
    """Generate sample factor data with known relationships."""
    np.random.seed(42)
    
    dates = pd.date_range("2023-01-01", periods=252, freq="D")
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
    
    data = []
    
    for date in dates:
        for ticker in tickers:
            # Generate factors with some predictive power
            momentum = np.random.randn()
            value = np.random.randn()
            quality = np.random.randn()
            
            # Forward return has relationship with factors + noise
            forward_return = (
                0.3 * momentum +
                0.2 * value +
                0.1 * quality +
                np.random.randn() * 0.5
            ) * 0.01  # Scale to realistic returns
            
            data.append({
                "date": date,
                "ticker": ticker,
                "momentum": momentum,
                "value": value,
                "quality": quality,
                "forward_return": forward_return,
            })
    
    df = pd.DataFrame(data)
    return df


# ===================================================================
# Purged K-Fold tests
# ===================================================================

def test_purged_kfold_basic():
    """Test basic Purged K-Fold functionality."""
    dates = pd.date_range("2023-01-01", periods=100, freq="D")
    X = pd.DataFrame({"feature": np.random.randn(100)}, index=dates)
    
    pkf = PurgedKFold(n_splits=5, embargo_pct=0.01)
    
    splits = list(pkf.split(X))
    
    # Should have 5 splits
    assert len(splits) == 5
    
    # Each split should have train and test indices
    for train_idx, test_idx in splits:
        assert len(train_idx) > 0
        assert len(test_idx) > 0
        
        # Train should come before test (time-series)
        assert train_idx.max() < test_idx.min()


def test_purged_kfold_embargo():
    """Test that embargo creates gap between train and test."""
    dates = pd.date_range("2023-01-01", periods=100, freq="D")
    X = pd.DataFrame({"feature": np.random.randn(100)}, index=dates)
    
    pkf = PurgedKFold(n_splits=5, embargo_pct=0.05)  # 5% embargo
    
    for train_idx, test_idx in pkf.split(X):
        # Gap between train and test should be at least embargo_size
        gap = test_idx.min() - train_idx.max()
        assert gap > 0  # There should be a gap


# ===================================================================
# MLAlphaModel tests
# ===================================================================

def test_ml_alpha_model_fit(sample_factor_data):
    """Test fitting ML alpha model."""
    df = sample_factor_data
    
    factors = df[["date", "ticker", "momentum", "value", "quality"]]
    returns = df.set_index(["date", "ticker"])["forward_return"]
    
    model = MLAlphaModel(model_type="ridge", alpha=1.0)
    model.fit(factors, returns)
    
    # Check that model is fitted
    assert model.model_ is not None
    assert model.scaler_ is not None
    assert model.feature_importance_ is not None
    assert model.feature_names_ == ["momentum", "value", "quality"]


def test_ml_alpha_model_predict(sample_factor_data):
    """Test prediction from ML alpha model."""
    df = sample_factor_data
    
    # Split into train and test (252 calendar days: Jan 1 – Sep 9)
    train = df[df["date"] < "2023-07-01"]
    test = df[df["date"] >= "2023-07-01"]
    
    factors_train = train[["date", "ticker", "momentum", "value", "quality"]]
    returns_train = train.set_index(["date", "ticker"])["forward_return"]
    
    factors_test = test[["date", "ticker", "momentum", "value", "quality"]]
    
    # Fit model
    model = MLAlphaModel(model_type="ridge", alpha=1.0)
    model.fit(factors_train, returns_train)
    
    # Predict
    predictions = model.predict(factors_test)
    
    assert isinstance(predictions, pd.Series)
    assert len(predictions) == len(test)
    assert predictions.notna().all()


def test_ml_alpha_model_types(sample_factor_data):
    """Test different model types (Ridge, Lasso, ElasticNet)."""
    df = sample_factor_data
    
    factors = df[["date", "ticker", "momentum", "value", "quality"]]
    returns = df.set_index(["date", "ticker"])["forward_return"]
    
    for model_type in ["ridge", "lasso", "elastic_net"]:
        model = MLAlphaModel(model_type=model_type, alpha=0.1)
        model.fit(factors, returns)
        
        # Should fit successfully
        assert model.model_ is not None
        
        # Predict
        pred = model.predict(factors)
        assert len(pred) == len(df)


def test_feature_importance(sample_factor_data):
    """Test feature importance extraction."""
    df = sample_factor_data
    
    factors = df[["date", "ticker", "momentum", "value", "quality"]]
    returns = df.set_index(["date", "ticker"])["forward_return"]
    
    model = MLAlphaModel(model_type="ridge", alpha=1.0)
    model.fit(factors, returns)
    
    importance = model.get_feature_importance(top_n=3)
    
    assert isinstance(importance, pd.Series)
    assert len(importance) == 3
    assert all(importance >= 0)  # Absolute values
    
    # Momentum should be most important (highest coefficient in data generation)
    assert importance.index[0] == "momentum"


def test_cross_validate(sample_factor_data):
    """Test cross-validation with Purged K-Fold."""
    df = sample_factor_data
    
    factors = df[["date", "ticker", "momentum", "value", "quality"]]
    returns = df.set_index(["date", "ticker"])["forward_return"]
    
    model = MLAlphaModel(model_type="ridge", alpha=1.0)
    cv_results = model.cross_validate(factors, returns, n_splits=3)
    
    assert "mean_r2" in cv_results
    assert "std_r2" in cv_results
    assert "mean_ic" in cv_results
    assert "std_ic" in cv_results
    
    # R² should be positive (model has some predictive power)
    assert cv_results["mean_r2"] > 0
    
    # IC should be positive (factors have predictive power)
    assert cv_results["mean_ic"] > 0


# ===================================================================
# Walk-Forward tests
# ===================================================================

def test_walk_forward_basic(sample_factor_data):
    """Test basic walk-forward alpha generation."""
    df = sample_factor_data
    
    factors = df[["date", "ticker", "momentum", "value", "quality"]]
    returns = df.set_index(["date", "ticker"])["forward_return"]
    
    wf = WalkForwardMLAlpha(
        model_type="ridge",
        alpha=1.0,
        refit_freq=21,
        min_train_periods=60,  # Shorter for testing
    )
    
    alpha_scores = wf.generate_alpha(factors, returns)
    
    assert isinstance(alpha_scores, pd.Series)
    assert len(alpha_scores) > 0
    
    # Should have predictions for dates after min_train_periods
    assert len(alpha_scores) < len(df)  # Not all dates (need training data)


def test_walk_forward_retraining(sample_factor_data):
    """Test that walk-forward retrains periodically."""
    df = sample_factor_data
    
    factors = df[["date", "ticker", "momentum", "value", "quality"]]
    returns = df.set_index(["date", "ticker"])["forward_return"]
    
    wf = WalkForwardMLAlpha(
        model_type="ridge",
        alpha=1.0,
        refit_freq=30,
        min_train_periods=60,
    )
    
    alpha_scores = wf.generate_alpha(factors, returns)
    
    # Should have multiple models (retraining happened)
    assert len(wf.models_) > 1
    
    # Should have feature importance history
    assert len(wf.feature_importance_history_) > 0


def test_walk_forward_feature_importance(sample_factor_data):
    """Test average feature importance across walk-forward models."""
    df = sample_factor_data
    
    factors = df[["date", "ticker", "momentum", "value", "quality"]]
    returns = df.set_index(["date", "ticker"])["forward_return"]
    
    wf = WalkForwardMLAlpha(
        model_type="ridge",
        alpha=1.0,
        refit_freq=30,
        min_train_periods=60,
    )
    
    wf.generate_alpha(factors, returns)
    
    avg_importance = wf.get_average_feature_importance(top_n=3)
    
    assert isinstance(avg_importance, pd.Series)
    assert len(avg_importance) == 3
    
    # Momentum should be most important on average
    assert avg_importance.index[0] == "momentum"


def test_walk_forward_no_lookahead(sample_factor_data):
    """Test that walk-forward doesn't use future data."""
    df = sample_factor_data
    
    factors = df[["date", "ticker", "momentum", "value", "quality"]]
    returns = df.set_index(["date", "ticker"])["forward_return"]
    
    wf = WalkForwardMLAlpha(
        model_type="ridge",
        alpha=1.0,
        refit_freq=21,
        min_train_periods=60,
    )
    
    alpha_scores = wf.generate_alpha(factors, returns)
    
    # Check that predictions only exist for dates after min_train_periods
    if not alpha_scores.empty:
        first_pred_date = alpha_scores.index.get_level_values("date").min()
        all_dates = sorted(df["date"].unique())
        
        # First prediction should be after min_train_periods
        assert all_dates.index(first_pred_date) >= wf.min_train_periods


# ===================================================================
# Edge cases
# ===================================================================

def test_ml_alpha_with_missing_features():
    """Test that model raises error when features are missing."""
    dates = pd.date_range("2023-01-01", periods=100, freq="D")
    
    # Train with 3 features
    train_data = pd.DataFrame({
        "date": dates[:50].repeat(5),
        "ticker": ["A", "B", "C", "D", "E"] * 50,
        "f1": np.random.randn(250),
        "f2": np.random.randn(250),
        "f3": np.random.randn(250),
    })
    train_returns = pd.Series(
        np.random.randn(250) * 0.01,
        index=pd.MultiIndex.from_arrays(
            [train_data["date"], train_data["ticker"]],
            names=["date", "ticker"],
        ),
    )
    
    model = MLAlphaModel(model_type="ridge")
    model.fit(train_data, train_returns)
    
    # Test with only 2 features (missing f3)
    test_data = pd.DataFrame({
        "date": dates[50:60].repeat(5),
        "ticker": ["A", "B", "C", "D", "E"] * 10,
        "f1": np.random.randn(50),
        "f2": np.random.randn(50),
    })
    
    with pytest.raises(ValueError, match="Missing features"):
        model.predict(test_data)


def test_ml_alpha_with_no_data():
    """Test that model handles empty data gracefully."""
    empty_factors = pd.DataFrame(columns=["date", "ticker", "f1", "f2"])
    empty_returns = pd.Series(dtype=float)
    
    model = MLAlphaModel(model_type="ridge")
    model.fit(empty_factors, empty_returns)
    
    # Should not crash, but model won't be fitted
    # (fit() logs warning and returns self)


def test_regularization_strength():
    """Test that higher regularization leads to smaller coefficients."""
    np.random.seed(123)
    dates = pd.date_range("2023-01-01", periods=100, freq="D")
    
    data = pd.DataFrame({
        "date": dates.repeat(10),
        "ticker": ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"] * 100,
        "f1": np.random.randn(1000),
        "f2": np.random.randn(1000),
    })
    returns = pd.Series(
        np.random.randn(1000) * 0.01,
        index=pd.MultiIndex.from_arrays(
            [data["date"], data["ticker"]],
            names=["date", "ticker"],
        ),
    )
    
    # Low regularization
    model_low = MLAlphaModel(model_type="ridge", alpha=0.01)
    model_low.fit(data, returns)
    
    # High regularization
    model_high = MLAlphaModel(model_type="ridge", alpha=10.0)
    model_high.fit(data, returns)
    
    # High regularization should have smaller coefficients
    coef_low = np.abs(model_low.model_.coef_).sum()
    coef_high = np.abs(model_high.model_.coef_).sum()
    
    assert coef_high < coef_low


def test_lasso_feature_selection():
    """Test that Lasso can set some coefficients to zero (feature selection)."""
    np.random.seed(456)
    dates = pd.date_range("2023-01-01", periods=100, freq="D")
    
    # Create data where only f1 is predictive
    data = pd.DataFrame({
        "date": dates.repeat(10),
        "ticker": ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"] * 100,
        "f1": np.random.randn(1000),
        "f2": np.random.randn(1000),  # Noise
        "f3": np.random.randn(1000),  # Noise
    })
    
    # Only f1 is predictive
    returns = pd.Series(
        0.5 * data["f1"].values + np.random.randn(1000) * 0.1,
        index=pd.MultiIndex.from_arrays(
            [data["date"], data["ticker"]],
            names=["date", "ticker"],
        ),
    )
    
    # Lasso with strong regularization
    model = MLAlphaModel(model_type="lasso", alpha=0.1)
    model.fit(data, returns)
    
    # Some coefficients should be exactly zero
    assert (model.model_.coef_ == 0).any()
    
    # f1 should have non-zero coefficient
    assert model.model_.coef_[0] != 0
