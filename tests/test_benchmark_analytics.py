"""Unit tests for benchmark analytics and alpha/beta decomposition."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_platform.core.evaluation.benchmark_analytics import (
    compute_active_return,
    compute_tracking_error,
    compute_information_ratio,
    compute_capm_alpha_beta,
    compute_treynor_ratio,
    compute_rolling_alpha_beta,
    decompose_return,
    benchmark_analysis,
)


# ===================================================================
# Test data fixtures
# ===================================================================

@pytest.fixture
def sample_returns():
    """Generate sample portfolio and benchmark returns."""
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=252, freq="D")
    
    # Benchmark: mean=0.05% daily, vol=1%
    bench_ret = np.random.normal(0.0005, 0.01, len(dates))
    
    # Portfolio: beta=1.2, alpha=0.02% daily, plus noise
    port_ret = 1.2 * bench_ret + 0.0002 + np.random.normal(0, 0.005, len(dates))
    
    portfolio = pd.Series(port_ret, index=dates, name="portfolio")
    benchmark = pd.Series(bench_ret, index=dates, name="benchmark")
    
    return portfolio, benchmark


# ===================================================================
# Active return metrics tests
# ===================================================================

def test_compute_active_return(sample_returns):
    """Test active return calculation."""
    port, bench = sample_returns
    active = compute_active_return(port, bench)
    
    assert isinstance(active, pd.Series)
    assert len(active) == len(port)
    # Active return should be roughly port - bench
    assert np.allclose(active.values, (port - bench).values)


def test_compute_tracking_error(sample_returns):
    """Test tracking error calculation."""
    port, bench = sample_returns
    active = compute_active_return(port, bench)
    
    te = compute_tracking_error(active, annualization=252)
    
    assert isinstance(te, float)
    assert te > 0
    # Tracking error should be annualized std of active returns
    expected_te = active.std() * np.sqrt(252)
    assert abs(te - expected_te) < 1e-6


def test_compute_information_ratio(sample_returns):
    """Test Information Ratio calculation."""
    port, bench = sample_returns
    active = compute_active_return(port, bench)
    
    ir = compute_information_ratio(active, annualization=252)
    
    assert isinstance(ir, float)
    # IR should be finite for our sample data
    assert np.isfinite(ir)


# ===================================================================
# CAPM alpha/beta tests
# ===================================================================

def test_compute_capm_alpha_beta(sample_returns):
    """Test CAPM alpha and beta calculation."""
    port, bench = sample_returns
    
    # Use excess returns (assuming rf=0 for simplicity)
    result = compute_capm_alpha_beta(port, bench, annualization=252)
    
    assert isinstance(result, dict)
    assert "alpha" in result
    assert "beta" in result
    assert "r_squared" in result
    assert "alpha_tstat" in result
    assert "alpha_pvalue" in result
    
    # Beta should be close to 1.2 (our simulation parameter)
    assert 0.8 < result["beta"] < 1.6  # Allow some variance
    
    # R-squared should be between 0 and 1
    assert 0 <= result["r_squared"] <= 1
    
    # Alpha should be annualized
    assert np.isfinite(result["alpha"])


def test_compute_treynor_ratio(sample_returns):
    """Test Treynor Ratio calculation."""
    port, bench = sample_returns
    
    capm = compute_capm_alpha_beta(port, bench)
    beta = capm["beta"]
    
    treynor = compute_treynor_ratio(port, beta, annualization=252)
    
    assert isinstance(treynor, float)
    assert np.isfinite(treynor)


def test_compute_rolling_alpha_beta(sample_returns):
    """Test rolling alpha and beta calculation."""
    port, bench = sample_returns
    
    rolling = compute_rolling_alpha_beta(port, bench, window=63, annualization=252)
    
    assert isinstance(rolling, pd.DataFrame)
    assert "date" in rolling.columns
    assert "alpha" in rolling.columns
    assert "beta" in rolling.columns
    assert "r_squared" in rolling.columns
    
    # Should have data after window period
    assert len(rolling) > 0
    
    # Beta should be roughly stable around 1.2
    assert rolling["beta"].mean() > 0.5
    assert rolling["beta"].mean() < 2.0


# ===================================================================
# Return decomposition tests
# ===================================================================

def test_decompose_return(sample_returns):
    """Test return decomposition into alpha/beta/residual."""
    port, bench = sample_returns
    
    decomp = decompose_return(port, bench, risk_free_rate=0.03, annualization=252)
    
    assert isinstance(decomp, dict)
    assert "total_return" in decomp
    assert "benchmark_return" in decomp
    assert "beta_contribution" in decomp
    assert "alpha_contribution" in decomp
    assert "residual_contribution" in decomp
    assert "active_return" in decomp
    
    # Active return should equal total - benchmark
    assert abs(decomp["active_return"] - (decomp["total_return"] - decomp["benchmark_return"])) < 1e-4


# ===================================================================
# Comprehensive analysis tests
# ===================================================================

def test_benchmark_analysis(sample_returns):
    """Test comprehensive benchmark analysis."""
    port, bench = sample_returns
    
    result = benchmark_analysis(port, bench, risk_free_rate=0.03, annualization=252)
    
    assert isinstance(result, dict)
    
    # Check all expected keys
    expected_keys = [
        "active_return_ann", "tracking_error", "information_ratio",
        "alpha", "beta", "r_squared", "alpha_tstat", "alpha_pvalue",
        "treynor_ratio", "total_return", "benchmark_return",
        "beta_contribution", "alpha_contribution", "residual_contribution",
    ]
    
    for key in expected_keys:
        assert key in result, f"Missing key: {key}"
        assert np.isfinite(result[key]) or result[key] is not None


def test_benchmark_analysis_insufficient_data():
    """Test that benchmark_analysis handles insufficient data gracefully."""
    # Only 5 days of data
    dates = pd.date_range("2023-01-01", periods=5, freq="D")
    port = pd.Series(np.random.randn(5) * 0.01, index=dates)
    bench = pd.Series(np.random.randn(5) * 0.01, index=dates)
    
    result = benchmark_analysis(port, bench)
    
    # Should return empty dict or handle gracefully
    assert isinstance(result, dict)


def test_capm_with_zero_variance():
    """Test CAPM calculation when benchmark has zero variance."""
    dates = pd.date_range("2023-01-01", periods=100, freq="D")
    port = pd.Series(np.random.randn(100) * 0.01, index=dates)
    bench = pd.Series(np.zeros(100), index=dates)  # Zero variance
    
    result = compute_capm_alpha_beta(port, bench)
    
    # Should handle gracefully (likely return NaN for beta)
    assert isinstance(result, dict)
    assert "beta" in result
