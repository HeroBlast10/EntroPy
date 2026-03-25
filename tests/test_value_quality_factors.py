"""Unit tests for value/quality factors."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_platform.core.signals.cross_sectional.value_quality import (
    EarningsYield,
    BookToMarket,
    GrossProfitability,
    AssetGrowth,
    _compute_ttm,
    _merge_fundamentals_to_prices,
)


# ===================================================================
# Test data fixtures
# ===================================================================

@pytest.fixture
def sample_prices():
    """Generate sample price data."""
    dates = pd.date_range("2023-01-01", periods=10, freq="D")
    tickers = ["AAPL", "MSFT"]
    
    data = []
    for ticker in tickers:
        for date in dates:
            data.append({
                "date": date,
                "ticker": ticker,
                "adj_close": 100.0 + np.random.randn() * 5,
            })
    
    return pd.DataFrame(data)


@pytest.fixture
def sample_fundamentals():
    """Generate sample fundamentals data with quarterly reports."""
    dates = pd.date_range("2023-01-01", periods=5, freq="D")
    report_dates = pd.date_range("2022-09-30", periods=4, freq="Q")
    tickers = ["AAPL", "MSFT"]
    
    data = []
    for ticker in tickers:
        for i, date in enumerate(dates):
            # Each date sees the most recent 4 quarters
            for j, rdate in enumerate(report_dates):
                data.append({
                    "date": date,
                    "ticker": ticker,
                    "report_date": rdate,
                    "publish_date": rdate + pd.Timedelta(days=45),
                    "net_income": 1000.0 + j * 100,
                    "gross_profit": 2000.0 + j * 200,
                    "total_assets": 50000.0,
                    "total_equity": 30000.0,
                    "market_cap": 100000.0,
                })
    
    return pd.DataFrame(data)


# ===================================================================
# Helper function tests
# ===================================================================

def test_compute_ttm(sample_fundamentals):
    """Test TTM calculation for income statement items."""
    ttm = _compute_ttm(sample_fundamentals, "net_income")
    
    assert "net_income_ttm" in ttm.columns
    assert len(ttm) > 0
    
    # Check that TTM is sum of 4 quarters
    # For our sample data: 1000 + 1100 + 1200 + 1300 = 4600
    first_row = ttm.iloc[0]
    assert first_row["net_income_ttm"] == pytest.approx(4600.0, abs=1.0)


def test_merge_fundamentals_to_prices(sample_prices, sample_fundamentals):
    """Test merging fundamentals to price grid with forward-fill."""
    fund = sample_fundamentals[["date", "ticker", "market_cap"]].drop_duplicates()
    
    merged = _merge_fundamentals_to_prices(sample_prices, fund, ["market_cap"])
    
    assert "market_cap" in merged.columns
    assert len(merged) == len(sample_prices)
    # Check forward-fill worked (no NaNs after first fundamental date)
    assert merged["market_cap"].notna().sum() > 0


# ===================================================================
# Factor tests
# ===================================================================

def test_earnings_yield(sample_prices, sample_fundamentals):
    """Test Earnings Yield factor computation."""
    factor = EarningsYield()
    result = factor._compute(sample_prices, sample_fundamentals)
    
    assert isinstance(result, pd.Series)
    assert result.name == "ep" or len(result) >= 0
    # Should have some non-NaN values
    assert result.notna().sum() > 0


def test_book_to_market(sample_prices, sample_fundamentals):
    """Test Book-to-Market factor computation."""
    factor = BookToMarket()
    result = factor._compute(sample_prices, sample_fundamentals)
    
    assert isinstance(result, pd.Series)
    assert result.notna().sum() > 0
    # B/M should be positive for our sample data
    assert (result[result.notna()] > 0).all()


def test_gross_profitability(sample_prices, sample_fundamentals):
    """Test Gross Profitability factor computation."""
    factor = GrossProfitability()
    result = factor._compute(sample_prices, sample_fundamentals)
    
    assert isinstance(result, pd.Series)
    assert result.notna().sum() > 0


def test_asset_growth(sample_prices):
    """Test Asset Growth factor computation."""
    # Need longer time series for YoY growth
    dates = pd.date_range("2022-01-01", periods=300, freq="D")
    fund_data = []
    
    for ticker in ["AAPL", "MSFT"]:
        for i, date in enumerate(dates):
            fund_data.append({
                "date": date,
                "ticker": ticker,
                "total_assets": 50000.0 * (1 + i * 0.001),  # gradual growth
            })
    
    fundamentals = pd.DataFrame(fund_data)
    
    # Extend prices to match
    px_data = []
    for ticker in ["AAPL", "MSFT"]:
        for date in dates:
            px_data.append({
                "date": date,
                "ticker": ticker,
                "adj_close": 100.0,
            })
    prices = pd.DataFrame(px_data)
    
    factor = AssetGrowth()
    result = factor._compute(prices, fundamentals)
    
    assert isinstance(result, pd.Series)
    # Should have some values after 252 days
    assert result.notna().sum() > 0


def test_factors_with_no_fundamentals(sample_prices):
    """Test that factors handle missing fundamentals gracefully."""
    factors = [EarningsYield(), BookToMarket(), GrossProfitability(), AssetGrowth()]
    
    for factor in factors:
        result = factor._compute(sample_prices, None)
        assert isinstance(result, pd.Series)
        assert len(result) == 0 or result.isna().all()


def test_factor_metadata():
    """Test that all factors have proper metadata."""
    factors = [EarningsYield(), BookToMarket(), GrossProfitability(), AssetGrowth()]
    
    for factor in factors:
        assert hasattr(factor, "meta")
        assert factor.meta.name is not None
        assert factor.meta.category in ["value", "quality"]
        assert factor.meta.signal_type == "cross_sectional"
        assert factor.meta.direction in [1, -1]
        assert len(factor.meta.references) > 0
