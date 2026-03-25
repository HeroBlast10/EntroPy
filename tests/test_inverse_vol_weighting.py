"""Unit tests for inverse-volatility weighting in quantile portfolio."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_platform.core.portfolio.quantile import QuantilePortfolio
from quant_platform.core.portfolio.construction import PortfolioConfig, WeightScheme


# ===================================================================
# Test data fixtures
# ===================================================================

@pytest.fixture
def sample_signal_with_prices():
    """Generate sample signal data with price history."""
    dates = pd.date_range("2023-01-01", periods=100, freq="D")
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN"]
    
    data = []
    np.random.seed(42)
    
    for ticker in tickers:
        # Different volatilities for each ticker
        if ticker == "AAPL":
            vol = 0.15  # Low vol
        elif ticker == "MSFT":
            vol = 0.20  # Medium vol
        elif ticker == "GOOGL":
            vol = 0.25  # High vol
        else:  # AMZN
            vol = 0.30  # Very high vol
        
        # Generate price series with different volatilities
        returns = np.random.normal(0.001, vol / np.sqrt(252), len(dates))
        prices = 100 * np.exp(np.cumsum(returns))
        
        for i, date in enumerate(dates):
            data.append({
                "date": date,
                "ticker": ticker,
                "signal": np.random.randn(),  # Random signal
                "adj_close": prices[i],
            })
    
    return pd.DataFrame(data)


@pytest.fixture
def sample_universe():
    """Generate sample universe data."""
    dates = pd.date_range("2023-01-01", periods=100, freq="D")
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN"]
    
    data = []
    for date in dates:
        for ticker in tickers:
            data.append({
                "date": date,
                "ticker": ticker,
                "pass_all_filters": True,
            })
    
    return pd.DataFrame(data)


# ===================================================================
# Helper function tests
# ===================================================================

def test_weights_from_volatility():
    """Test conversion of volatility to inverse-vol weights."""
    vol = pd.Series([0.10, 0.20, 0.30], index=["A", "B", "C"])
    
    weights = QuantilePortfolio._weights_from_volatility(vol)
    
    # Weights should sum to 1
    assert abs(weights.sum() - 1.0) < 1e-6
    
    # Lower vol should get higher weight
    assert weights["A"] > weights["B"] > weights["C"]
    
    # Check formula: weight_i = (1/σ_i) / Σ(1/σ_j)
    inv_vol = 1.0 / vol
    expected = inv_vol / inv_vol.sum()
    pd.testing.assert_series_equal(weights, expected)


def test_weights_from_volatility_zero_vol():
    """Test that zero volatility is handled gracefully."""
    vol = pd.Series([0.10, 0.0, 0.20], index=["A", "B", "C"])
    
    weights = QuantilePortfolio._weights_from_volatility(vol)
    
    # Should not crash, weights should sum to 1
    assert abs(weights.sum() - 1.0) < 1e-6
    assert all(weights >= 0)


def test_compute_rolling_volatility(sample_signal_with_prices):
    """Test rolling volatility calculation from prices."""
    signal = sample_signal_with_prices
    tickers = pd.Index(["AAPL", "MSFT", "GOOGL"])
    date = pd.Timestamp("2023-04-10")  # After 100 days
    
    vol = QuantilePortfolio._compute_rolling_volatility(
        signal, tickers, date, price_col="adj_close", window=63
    )
    
    assert isinstance(vol, pd.Series)
    assert len(vol) == len(tickers)
    assert all(vol > 0)
    
    # AAPL should have lowest vol, GOOGL highest (based on our fixture)
    assert vol["AAPL"] < vol["GOOGL"]


def test_compute_rolling_volatility_insufficient_data(sample_signal_with_prices):
    """Test volatility calculation with insufficient history."""
    signal = sample_signal_with_prices
    tickers = pd.Index(["AAPL", "MSFT"])
    date = pd.Timestamp("2023-01-10")  # Only 10 days of data
    
    vol = QuantilePortfolio._compute_rolling_volatility(
        signal, tickers, date, price_col="adj_close", window=63
    )
    
    # Should return empty (not enough data)
    assert vol.empty


# ===================================================================
# Integration tests
# ===================================================================

def test_inverse_vol_weighting_integration(sample_signal_with_prices, sample_universe):
    """Test inverse-vol weighting in full portfolio construction."""
    config = PortfolioConfig(
        weight_scheme=WeightScheme.INVERSE_VOL,
        n_quantiles=2,
        long_quantile=2,
    )
    
    portfolio = QuantilePortfolio(config)
    
    # Generate weights for a specific date
    date = pd.Timestamp("2023-04-10")
    weights = portfolio._generate_weights(
        sample_signal_with_prices,
        sample_universe,
        date,
        prev_weights=None,
    )
    
    assert isinstance(weights, pd.Series)
    assert len(weights) > 0
    
    # Weights should sum to 1
    assert abs(weights.sum() - 1.0) < 1e-6
    
    # All weights should be positive (long-only)
    assert all(weights > 0)


def test_inverse_vol_vs_equal_weight(sample_signal_with_prices, sample_universe):
    """Test that inverse-vol differs from equal weight."""
    # Equal weight
    config_eq = PortfolioConfig(
        weight_scheme=WeightScheme.EQUAL,
        n_quantiles=2,
        long_quantile=2,
    )
    portfolio_eq = QuantilePortfolio(config_eq)
    
    # Inverse-vol weight
    config_iv = PortfolioConfig(
        weight_scheme=WeightScheme.INVERSE_VOL,
        n_quantiles=2,
        long_quantile=2,
    )
    portfolio_iv = QuantilePortfolio(config_iv)
    
    date = pd.Timestamp("2023-04-10")
    
    weights_eq = portfolio_eq._generate_weights(
        sample_signal_with_prices, sample_universe, date
    )
    weights_iv = portfolio_iv._generate_weights(
        sample_signal_with_prices, sample_universe, date
    )
    
    # Weights should be different
    assert not weights_eq.equals(weights_iv)
    
    # Equal weight should have uniform weights
    assert weights_eq.std() < 1e-6
    
    # Inverse-vol should have non-uniform weights
    assert weights_iv.std() > 1e-6


def test_inverse_vol_fallback_to_equal_weight():
    """Test that inverse-vol falls back to equal weight when no price data."""
    # Signal without price columns
    signal = pd.DataFrame({
        "date": pd.date_range("2023-01-01", periods=10, freq="D").repeat(3),
        "ticker": ["A", "B", "C"] * 10,
        "signal": np.random.randn(30),
    })
    
    universe = pd.DataFrame({
        "date": pd.date_range("2023-01-01", periods=10, freq="D").repeat(3),
        "ticker": ["A", "B", "C"] * 10,
        "pass_all_filters": True,
    })
    
    config = PortfolioConfig(
        weight_scheme=WeightScheme.INVERSE_VOL,
        n_quantiles=2,
        long_quantile=2,
    )
    
    portfolio = QuantilePortfolio(config)
    date = pd.Timestamp("2023-01-05")
    
    weights = portfolio._generate_weights(signal, universe, date)
    
    # Should fall back to equal weight
    assert len(weights) > 0
    assert abs(weights.sum() - 1.0) < 1e-6
    # Equal weight → uniform distribution
    assert weights.std() < 1e-6


def test_inverse_vol_with_precomputed_volatility():
    """Test inverse-vol weighting when volatility is pre-computed in universe."""
    signal = pd.DataFrame({
        "date": pd.date_range("2023-01-01", periods=10, freq="D").repeat(3),
        "ticker": ["A", "B", "C"] * 10,
        "signal": np.random.randn(30),
    })
    
    # Universe with pre-computed volatility
    universe_data = []
    for date in pd.date_range("2023-01-01", periods=10, freq="D"):
        universe_data.append({"date": date, "ticker": "A", "pass_all_filters": True, "volatility": 0.10})
        universe_data.append({"date": date, "ticker": "B", "pass_all_filters": True, "volatility": 0.20})
        universe_data.append({"date": date, "ticker": "C", "pass_all_filters": True, "volatility": 0.30})
    
    universe = pd.DataFrame(universe_data)
    
    config = PortfolioConfig(
        weight_scheme=WeightScheme.INVERSE_VOL,
        n_quantiles=2,
        long_quantile=2,
    )
    
    portfolio = QuantilePortfolio(config)
    date = pd.Timestamp("2023-01-05")
    
    weights = portfolio._generate_weights(signal, universe, date)
    
    # Should use pre-computed volatility
    assert len(weights) > 0
    assert abs(weights.sum() - 1.0) < 1e-6
    
    # Lower vol (A) should get higher weight
    if "A" in weights.index and "C" in weights.index:
        assert weights["A"] > weights["C"]


# ===================================================================
# Edge cases
# ===================================================================

def test_inverse_vol_with_single_stock():
    """Test inverse-vol with only one stock selected."""
    signal = pd.DataFrame({
        "date": [pd.Timestamp("2023-01-05")] * 3,
        "ticker": ["A", "B", "C"],
        "signal": [10.0, 1.0, 1.0],  # A has very high signal
        "adj_close": [100.0, 100.0, 100.0],
    })
    
    universe = pd.DataFrame({
        "date": [pd.Timestamp("2023-01-05")] * 3,
        "ticker": ["A", "B", "C"],
        "pass_all_filters": True,
    })
    
    config = PortfolioConfig(
        weight_scheme=WeightScheme.INVERSE_VOL,
        top_n=1,  # Select only top 1
    )
    
    portfolio = QuantilePortfolio(config)
    date = pd.Timestamp("2023-01-05")
    
    weights = portfolio._generate_weights(signal, universe, date)
    
    # Single stock should get 100% weight
    assert len(weights) == 1
    assert abs(weights.iloc[0] - 1.0) < 1e-6


def test_inverse_vol_with_missing_tickers(sample_signal_with_prices, sample_universe):
    """Test inverse-vol when some tickers have missing price data."""
    signal = sample_signal_with_prices.copy()
    
    # Remove some price data for AMZN
    signal.loc[signal["ticker"] == "AMZN", "adj_close"] = np.nan
    
    config = PortfolioConfig(
        weight_scheme=WeightScheme.INVERSE_VOL,
        n_quantiles=2,
        long_quantile=2,
    )
    
    portfolio = QuantilePortfolio(config)
    date = pd.Timestamp("2023-04-10")
    
    weights = portfolio._generate_weights(signal, sample_universe, date)
    
    # Should still work (may exclude AMZN or use fallback)
    assert len(weights) > 0
    assert abs(weights.sum() - 1.0) < 1e-6
