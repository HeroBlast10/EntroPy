"""Unit tests for Barra-style factor risk model."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_platform.core.portfolio.risk_model import FactorRiskModel


# ===================================================================
# Test data fixtures
# ===================================================================

@pytest.fixture
def sample_factor_returns():
    """Generate sample factor returns (market, size, value)."""
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=252, freq="D")
    
    # Market factor: mean=0.05% daily, vol=1%
    market = np.random.normal(0.0005, 0.01, len(dates))
    
    # Size factor: mean=0.02% daily, vol=0.5%
    size = np.random.normal(0.0002, 0.005, len(dates))
    
    # Value factor: mean=0.03% daily, vol=0.6%
    value = np.random.normal(0.0003, 0.006, len(dates))
    
    return pd.DataFrame({
        "market": market,
        "size": size,
        "value": value,
    }, index=dates)


@pytest.fixture
def sample_stock_returns(sample_factor_returns):
    """Generate sample stock returns with known factor exposures."""
    np.random.seed(123)
    factor_ret = sample_factor_returns
    
    # Define true betas for 5 stocks
    true_betas = {
        "AAPL": {"market": 1.2, "size": -0.5, "value": 0.3},  # Large growth
        "MSFT": {"market": 1.1, "size": -0.3, "value": 0.2},  # Large growth
        "GOOGL": {"market": 1.3, "size": -0.4, "value": 0.1},  # Large growth
        "XOM": {"market": 0.8, "size": 0.2, "value": 0.8},    # Large value
        "F": {"market": 1.5, "size": 0.5, "value": 0.9},      # Small value
    }
    
    stock_returns = {}
    
    for ticker, betas in true_betas.items():
        # Stock return = beta @ factor_returns + specific_return
        factor_component = (
            betas["market"] * factor_ret["market"] +
            betas["size"] * factor_ret["size"] +
            betas["value"] * factor_ret["value"]
        )
        
        # Add specific (idiosyncratic) return
        specific = np.random.normal(0, 0.01, len(factor_ret))  # 1% specific vol
        
        stock_returns[ticker] = factor_component + specific
    
    return pd.DataFrame(stock_returns, index=factor_ret.index)


# ===================================================================
# Factor exposure estimation tests
# ===================================================================

def test_estimate_exposures(sample_stock_returns, sample_factor_returns):
    """Test factor exposure estimation via regression."""
    model = FactorRiskModel()
    
    exposures = model._estimate_exposures(
        sample_stock_returns, sample_factor_returns
    )
    
    assert isinstance(exposures, pd.DataFrame)
    assert len(exposures) == 5  # 5 stocks
    assert list(exposures.columns) == ["market", "size", "value"]
    
    # Check that estimated betas are close to true betas
    # AAPL should have market beta ~1.2
    assert 0.8 < exposures.loc["AAPL", "market"] < 1.6
    
    # F should have positive size beta (small cap)
    assert exposures.loc["F", "size"] > 0


def test_estimate_exposures_insufficient_data():
    """Test exposure estimation with insufficient data."""
    # Only 10 days of data
    dates = pd.date_range("2023-01-01", periods=10, freq="D")
    stock_ret = pd.DataFrame({
        "A": np.random.randn(10) * 0.01,
    }, index=dates)
    factor_ret = pd.DataFrame({
        "market": np.random.randn(10) * 0.01,
    }, index=dates)
    
    model = FactorRiskModel()
    exposures = model._estimate_exposures(stock_ret, factor_ret)
    
    # Should return empty (need at least 20 observations)
    assert exposures.empty


# ===================================================================
# Factor covariance tests
# ===================================================================

def test_estimate_factor_covariance(sample_factor_returns):
    """Test factor covariance matrix estimation."""
    model = FactorRiskModel(halflife=60, shrinkage=0.5)
    
    cov = model._estimate_factor_covariance(sample_factor_returns)
    
    assert isinstance(cov, np.ndarray)
    assert cov.shape == (3, 3)  # 3 factors
    
    # Should be symmetric
    assert np.allclose(cov, cov.T)
    
    # Should be positive semi-definite
    eigenvalues = np.linalg.eigvals(cov)
    assert all(eigenvalues >= -1e-10)


# ===================================================================
# Specific risk tests
# ===================================================================

def test_estimate_specific_risk(sample_stock_returns, sample_factor_returns):
    """Test specific risk estimation from residuals."""
    model = FactorRiskModel()
    
    # First estimate exposures
    exposures = model._estimate_exposures(
        sample_stock_returns, sample_factor_returns
    )
    
    # Then estimate specific risk
    specific_risk = model._estimate_specific_risk(
        sample_stock_returns, sample_factor_returns, exposures
    )
    
    assert isinstance(specific_risk, pd.Series)
    assert len(specific_risk) == 5
    assert all(specific_risk > 0)
    
    # Specific risk should be annualized (reasonable range 5-50%)
    assert all(specific_risk > 0.05)
    assert all(specific_risk < 0.50)


# ===================================================================
# Full model fit tests
# ===================================================================

def test_fit_model(sample_stock_returns, sample_factor_returns):
    """Test full model fitting."""
    model = FactorRiskModel(halflife=60, shrinkage=0.5)
    
    model.fit(sample_stock_returns, sample_factor_returns)
    
    # Check that all components are fitted
    assert model.exposures_ is not None
    assert model.cov_matrix_ is not None
    assert model.specific_risk_ is not None
    assert model.factor_names_ == ["market", "size", "value"]
    
    # Check shapes
    assert model.exposures_.shape == (5, 3)  # 5 stocks, 3 factors
    assert model.cov_matrix_.shape == (3, 3)
    assert len(model.specific_risk_) == 5


# ===================================================================
# Risk decomposition tests
# ===================================================================

def test_decompose_risk(sample_stock_returns, sample_factor_returns):
    """Test portfolio risk decomposition."""
    model = FactorRiskModel()
    model.fit(sample_stock_returns, sample_factor_returns)
    
    # Equal-weight portfolio
    weights = pd.Series(0.2, index=sample_stock_returns.columns)
    
    decomp = model.decompose_risk(weights)
    
    # Check all expected keys
    assert "total_risk" in decomp
    assert "factor_risk" in decomp
    assert "specific_risk" in decomp
    assert "factor_contributions" in decomp
    assert "portfolio_exposures" in decomp
    
    # Total risk should be positive
    assert decomp["total_risk"] > 0
    
    # Factor risk + specific risk should approximately equal total risk
    # (total_variance = factor_variance + specific_variance)
    total_var = decomp["total_risk"]**2
    factor_var = decomp["factor_risk"]**2
    specific_var = decomp["specific_risk"]**2
    
    assert abs(total_var - (factor_var + specific_var)) < 1e-6
    
    # Factor contributions should sum to factor variance
    contrib_sum = sum(decomp["factor_contributions"].values())
    assert abs(contrib_sum - decomp.get("factor_variance", factor_var)) < 1e-6


def test_decompose_risk_market_neutral():
    """Test risk decomposition for market-neutral portfolio."""
    np.random.seed(456)
    dates = pd.date_range("2023-01-01", periods=252, freq="D")
    
    # Market factor
    market = np.random.normal(0.0005, 0.01, len(dates))
    factor_ret = pd.DataFrame({"market": market}, index=dates)
    
    # Two stocks: one with beta=1.5, one with beta=-1.5
    stock_ret = pd.DataFrame({
        "A": 1.5 * market + np.random.normal(0, 0.005, len(dates)),
        "B": -1.5 * market + np.random.normal(0, 0.005, len(dates)),
    }, index=dates)
    
    model = FactorRiskModel()
    model.fit(stock_ret, factor_ret)
    
    # Market-neutral: 50% A, 50% B
    weights = pd.Series({"A": 0.5, "B": 0.5})
    
    decomp = model.decompose_risk(weights)
    
    # Portfolio market exposure should be close to 0
    market_exposure = decomp["portfolio_exposures"]["market"]
    assert abs(market_exposure) < 0.2  # Allow some estimation error
    
    # Factor risk should be low (market-neutral)
    # Specific risk should dominate
    assert decomp["specific_risk"] > decomp["factor_risk"]


def test_decompose_risk_empty_weights():
    """Test risk decomposition with empty weights."""
    dates = pd.date_range("2023-01-01", periods=100, freq="D")
    factor_ret = pd.DataFrame({"market": np.random.randn(100) * 0.01}, index=dates)
    stock_ret = pd.DataFrame({"A": np.random.randn(100) * 0.01}, index=dates)
    
    model = FactorRiskModel()
    model.fit(stock_ret, factor_ret)
    
    # Empty weights (no overlap)
    weights = pd.Series({"B": 1.0})  # Stock B not in model
    
    decomp = model.decompose_risk(weights)
    
    # Should return zeros
    assert decomp["total_risk"] == 0.0
    assert decomp["factor_risk"] == 0.0
    assert decomp["specific_risk"] == 0.0


# ===================================================================
# Getter method tests
# ===================================================================

def test_get_exposures(sample_stock_returns, sample_factor_returns):
    """Test get_exposures method."""
    model = FactorRiskModel()
    model.fit(sample_stock_returns, sample_factor_returns)
    
    # Get all exposures
    exp_all = model.get_exposures()
    assert len(exp_all) == 5
    
    # Get specific tickers
    exp_subset = model.get_exposures(["AAPL", "MSFT"])
    assert len(exp_subset) == 2
    assert "AAPL" in exp_subset.index
    assert "MSFT" in exp_subset.index


def test_get_specific_risk(sample_stock_returns, sample_factor_returns):
    """Test get_specific_risk method."""
    model = FactorRiskModel()
    model.fit(sample_stock_returns, sample_factor_returns)
    
    # Get all specific risks
    sr_all = model.get_specific_risk()
    assert len(sr_all) == 5
    
    # Get specific tickers
    sr_subset = model.get_specific_risk(["AAPL", "XOM"])
    assert len(sr_subset) == 2


def test_get_methods_before_fit():
    """Test that getter methods raise error before fit."""
    model = FactorRiskModel()
    
    with pytest.raises(RuntimeError, match="Must call fit"):
        model.get_exposures()
    
    with pytest.raises(RuntimeError, match="Must call fit"):
        model.get_specific_risk()
    
    with pytest.raises(RuntimeError, match="Must call fit"):
        weights = pd.Series({"A": 1.0})
        model.decompose_risk(weights)


# ===================================================================
# Edge cases
# ===================================================================

def test_model_with_single_factor():
    """Test model with only one factor (market)."""
    np.random.seed(789)
    dates = pd.date_range("2023-01-01", periods=252, freq="D")
    
    market = np.random.normal(0.0005, 0.01, len(dates))
    factor_ret = pd.DataFrame({"market": market}, index=dates)
    
    stock_ret = pd.DataFrame({
        "A": 1.2 * market + np.random.normal(0, 0.005, len(dates)),
        "B": 0.8 * market + np.random.normal(0, 0.005, len(dates)),
    }, index=dates)
    
    model = FactorRiskModel()
    model.fit(stock_ret, factor_ret)
    
    weights = pd.Series({"A": 0.6, "B": 0.4})
    decomp = model.decompose_risk(weights)
    
    # Should work with single factor
    assert decomp["total_risk"] > 0
    assert "market" in decomp["factor_contributions"]


def test_model_with_missing_stock_data():
    """Test model when some stocks have missing data."""
    np.random.seed(101)
    dates = pd.date_range("2023-01-01", periods=100, freq="D")
    
    market = np.random.normal(0.0005, 0.01, len(dates))
    factor_ret = pd.DataFrame({"market": market}, index=dates)
    
    # Stock A has full data, Stock B has missing data
    stock_ret = pd.DataFrame({
        "A": 1.0 * market + np.random.normal(0, 0.005, len(dates)),
        "B": [np.nan] * 50 + list(0.8 * market[50:] + np.random.normal(0, 0.005, 50)),
    }, index=dates)
    
    model = FactorRiskModel()
    model.fit(stock_ret, factor_ret)
    
    # Stock A should be in exposures
    assert "A" in model.exposures_.index
    
    # Stock B might be excluded (< 20 observations after dropna)
    # or included if enough data remains


def test_factor_contributions_sum():
    """Test that factor contributions sum correctly."""
    np.random.seed(202)
    dates = pd.date_range("2023-01-01", periods=252, freq="D")
    
    # Two uncorrelated factors
    factor_ret = pd.DataFrame({
        "f1": np.random.normal(0, 0.01, len(dates)),
        "f2": np.random.normal(0, 0.01, len(dates)),
    }, index=dates)
    
    stock_ret = pd.DataFrame({
        "A": 0.5 * factor_ret["f1"] + 0.5 * factor_ret["f2"] + np.random.normal(0, 0.002, len(dates)),
    }, index=dates)
    
    model = FactorRiskModel(shrinkage=0.0)  # No shrinkage for cleaner test
    model.fit(stock_ret, factor_ret)
    
    weights = pd.Series({"A": 1.0})
    decomp = model.decompose_risk(weights)
    
    # Factor contributions should sum to factor variance
    contrib_sum = sum(decomp["factor_contributions"].values())
    factor_var = decomp["factor_variance"]
    
    # Allow small numerical error
    assert abs(contrib_sum - factor_var) / factor_var < 0.01
