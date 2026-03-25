"""Unit tests for VaR/CVaR risk metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_platform.core.evaluation.risk_metrics import (
    compute_var,
    compute_cvar,
    compute_parametric_var,
    compute_cornish_fisher_var,
    compute_rolling_var,
    compute_rolling_cvar,
    risk_metrics_summary,
    var_backtest,
)


# ===================================================================
# Test data fixtures
# ===================================================================

@pytest.fixture
def sample_returns():
    """Generate sample returns with known distribution."""
    np.random.seed(42)
    # Generate returns with slight negative skew and fat tails
    returns = np.random.normal(0.001, 0.015, 500)
    # Add some extreme losses
    returns[::50] = -0.05  # 10 extreme loss days
    return pd.Series(returns, index=pd.date_range("2023-01-01", periods=500, freq="D"))


@pytest.fixture
def normal_returns():
    """Generate perfectly normal returns for testing parametric VaR."""
    np.random.seed(123)
    returns = np.random.normal(0.0005, 0.01, 252)
    return pd.Series(returns, index=pd.date_range("2023-01-01", periods=252, freq="D"))


# ===================================================================
# Historical VaR and CVaR tests
# ===================================================================

def test_compute_var_basic(sample_returns):
    """Test basic VaR calculation."""
    var_95 = compute_var(sample_returns, confidence=0.95)
    
    assert isinstance(var_95, float)
    assert var_95 > 0  # VaR should be positive (representing a loss)
    assert np.isfinite(var_95)
    
    # VaR should be less than max loss
    max_loss = -sample_returns.min()
    assert var_95 <= max_loss


def test_compute_var_confidence_levels(sample_returns):
    """Test that higher confidence → higher VaR."""
    var_90 = compute_var(sample_returns, confidence=0.90)
    var_95 = compute_var(sample_returns, confidence=0.95)
    var_99 = compute_var(sample_returns, confidence=0.99)
    
    # Higher confidence should give higher VaR (more extreme quantile)
    assert var_90 < var_95 < var_99


def test_compute_cvar_basic(sample_returns):
    """Test basic CVaR calculation."""
    cvar_95 = compute_cvar(sample_returns, confidence=0.95)
    
    assert isinstance(cvar_95, float)
    assert cvar_95 > 0
    assert np.isfinite(cvar_95)


def test_cvar_greater_than_var(sample_returns):
    """Test that CVaR >= VaR (CVaR is mean of tail, VaR is threshold)."""
    var_95 = compute_var(sample_returns, confidence=0.95)
    cvar_95 = compute_cvar(sample_returns, confidence=0.95)
    
    # CVaR should be >= VaR (expected loss in tail >= threshold)
    assert cvar_95 >= var_95


def test_var_with_empty_returns():
    """Test VaR with empty returns."""
    empty = pd.Series([], dtype=float)
    var = compute_var(empty)
    
    assert np.isnan(var)


def test_var_with_all_positive_returns():
    """Test VaR when all returns are positive (no losses)."""
    positive = pd.Series([0.01, 0.02, 0.03, 0.01, 0.02])
    var = compute_var(positive, confidence=0.95)
    
    # VaR should be close to 0 or negative (no losses)
    assert var <= 0.01


# ===================================================================
# Parametric VaR tests
# ===================================================================

def test_parametric_var_normal_distribution(normal_returns):
    """Test parametric VaR on normal returns."""
    var_param = compute_parametric_var(normal_returns, confidence=0.95)
    
    assert isinstance(var_param, float)
    assert var_param > 0
    assert np.isfinite(var_param)


def test_parametric_vs_historical_var(normal_returns):
    """Test that parametric and historical VaR are similar for normal returns."""
    var_hist = compute_var(normal_returns, confidence=0.95)
    var_param = compute_parametric_var(normal_returns, confidence=0.95)
    
    # Should be close for normal distribution
    # Allow 50% relative difference due to sampling
    assert abs(var_hist - var_param) / var_param < 0.5


# ===================================================================
# Cornish-Fisher VaR tests
# ===================================================================

def test_cornish_fisher_var(sample_returns):
    """Test Cornish-Fisher VaR calculation."""
    var_cf = compute_cornish_fisher_var(sample_returns, confidence=0.95)
    
    assert isinstance(var_cf, float)
    assert var_cf > 0
    assert np.isfinite(var_cf)


def test_cornish_fisher_adjusts_for_skew(sample_returns):
    """Test that Cornish-Fisher differs from parametric for skewed returns."""
    var_param = compute_parametric_var(sample_returns, confidence=0.95)
    var_cf = compute_cornish_fisher_var(sample_returns, confidence=0.95)
    
    # Should differ for non-normal returns
    # (sample_returns has negative skew and fat tails)
    assert var_param != var_cf


# ===================================================================
# Rolling VaR tests
# ===================================================================

def test_rolling_var(sample_returns):
    """Test rolling VaR calculation."""
    rolling_var = compute_rolling_var(sample_returns, window=63, confidence=0.95)
    
    assert isinstance(rolling_var, pd.Series)
    assert len(rolling_var) == len(sample_returns)
    # Should have NaNs at the beginning
    assert rolling_var.isna().sum() > 0
    # Should have values after window period
    assert rolling_var.notna().sum() > 0


def test_rolling_cvar(sample_returns):
    """Test rolling CVaR calculation."""
    rolling_cvar = compute_rolling_cvar(sample_returns, window=63, confidence=0.95)
    
    assert isinstance(rolling_cvar, pd.Series)
    assert len(rolling_cvar) == len(sample_returns)
    assert rolling_cvar.notna().sum() > 0


def test_rolling_var_methods(sample_returns):
    """Test different rolling VaR methods."""
    methods = ["historical", "parametric", "cornish_fisher"]
    
    for method in methods:
        rolling_var = compute_rolling_var(sample_returns, window=63, 
                                          confidence=0.95, method=method)
        assert isinstance(rolling_var, pd.Series)
        assert rolling_var.notna().sum() > 0


# ===================================================================
# Risk metrics summary tests
# ===================================================================

def test_risk_metrics_summary(sample_returns):
    """Test comprehensive risk metrics summary."""
    summary = risk_metrics_summary(sample_returns, confidence=0.95)
    
    assert isinstance(summary, dict)
    
    expected_keys = [
        "var_historical", "cvar_historical", "var_parametric",
        "var_cornish_fisher", "skewness", "kurtosis"
    ]
    
    for key in expected_keys:
        assert key in summary
        assert np.isfinite(summary[key])


def test_risk_metrics_summary_insufficient_data():
    """Test risk metrics summary with insufficient data."""
    short_returns = pd.Series([0.01, -0.02, 0.01])
    summary = risk_metrics_summary(short_returns)
    
    # Should return NaNs
    assert all(np.isnan(v) for v in summary.values())


# ===================================================================
# VaR backtest tests
# ===================================================================

def test_var_backtest(sample_returns):
    """Test VaR backtesting."""
    # Compute rolling VaR
    rolling_var = compute_rolling_var(sample_returns, window=63, confidence=0.95)
    
    # Backtest
    backtest = var_backtest(sample_returns, rolling_var, confidence=0.95)
    
    assert isinstance(backtest, dict)
    assert "n_violations" in backtest
    assert "violation_rate" in backtest
    assert "expected_rate" in backtest
    assert "kupiec_pvalue" in backtest
    
    # Violation rate should be close to (1 - confidence)
    assert 0 <= backtest["violation_rate"] <= 1
    assert backtest["expected_rate"] == 0.05  # 1 - 0.95


def test_var_backtest_perfect_model():
    """Test VaR backtest with a perfect model."""
    np.random.seed(456)
    returns = pd.Series(np.random.normal(0, 0.01, 1000))
    
    # Use parametric VaR (should be accurate for normal returns)
    rolling_var = compute_rolling_var(returns, window=252, confidence=0.95, 
                                      method="parametric")
    
    backtest = var_backtest(returns, rolling_var, confidence=0.95)
    
    # Violation rate should be close to 5%
    # Allow 2-8% range due to sampling
    assert 0.02 <= backtest["violation_rate"] <= 0.08


# ===================================================================
# Edge cases
# ===================================================================

def test_var_with_nans():
    """Test VaR calculation with NaN values."""
    returns_with_nan = pd.Series([0.01, np.nan, -0.02, 0.01, np.nan, -0.01])
    var = compute_var(returns_with_nan, confidence=0.95)
    
    # Should handle NaNs gracefully
    assert np.isfinite(var)


def test_var_with_constant_returns():
    """Test VaR with constant returns (zero variance)."""
    constant = pd.Series([0.01] * 100)
    var = compute_var(constant, confidence=0.95)
    
    # VaR should be close to 0 (no volatility)
    assert abs(var) < 1e-6


def test_cvar_with_no_tail_losses():
    """Test CVaR when there are no losses beyond VaR."""
    # All returns positive except one small loss
    returns = pd.Series([0.01, 0.02, 0.01, -0.001, 0.02])
    cvar = compute_cvar(returns, confidence=0.95)
    
    # Should handle gracefully
    assert np.isfinite(cvar) or cvar == 0.0
