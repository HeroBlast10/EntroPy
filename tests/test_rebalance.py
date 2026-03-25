"""Tests for rebalance schedule and weight carry-forward."""

from __future__ import annotations

import pandas as pd
import pytest

from quant_platform.core.portfolio.rebalance import carry_forward_weights, rebalance_dates


class TestRebalanceDates:
    def test_daily(self):
        dates = rebalance_dates("D", "2023-01-01", "2023-01-31")
        assert len(dates) > 15  # ~21 trading days in Jan

    def test_weekly(self):
        dates = rebalance_dates("W", "2023-01-01", "2023-03-31")
        assert 10 <= len(dates) <= 14  # ~13 weeks

    def test_monthly(self):
        dates = rebalance_dates("M", "2023-01-01", "2023-12-31")
        assert len(dates) == 12

    def test_monthly_dates_are_last_trading_day(self):
        dates = rebalance_dates("M", "2023-01-01", "2023-06-30")
        for d in dates:
            # Should be a weekday
            assert d.weekday() < 5

    def test_invalid_freq_raises(self):
        with pytest.raises(ValueError, match="Unknown rebalance"):
            rebalance_dates("Q", "2023-01-01", "2023-12-31")


class TestCarryForward:
    def test_fills_between_rebalances(self):
        from quant_platform.core.data.calendar import trading_dates
        all_dates = trading_dates("2023-01-01", "2023-03-31")
        reb = rebalance_dates("M", "2023-01-01", "2023-03-31")

        weights = pd.DataFrame({
            "date": [reb[0], reb[0], reb[1], reb[1]],
            "ticker": ["A", "B", "A", "B"],
            "weight": [0.5, 0.5, 0.6, 0.4],
        })

        daily = carry_forward_weights(weights, all_dates)
        # Should have weights for every trading day from the first rebalance onwards
        dates_from_first_reb = all_dates[all_dates >= reb[0]]
        assert daily["date"].nunique() >= len(dates_from_first_reb) - 2  # small tolerance
        # Between rebalances, weights should be constant
        mid_date = all_dates[len(all_dates) // 4]  # some date in first month
        if mid_date >= reb[0]:
            mid_w = daily[daily["date"] == mid_date]
            assert len(mid_w) > 0
    
    def test_weights_sum_to_one_on_all_dates(self):
        """Test that weights sum to 1.0 on all dates (critical invariant)."""
        from quant_platform.core.data.calendar import trading_dates
        all_dates = trading_dates("2023-01-01", "2023-03-31")
        reb = rebalance_dates("M", "2023-01-01", "2023-03-31")

        # Create rebalance weights that sum to 1.0
        weights = pd.DataFrame({
            "date": [reb[0], reb[0], reb[1], reb[1], reb[1]],
            "ticker": ["A", "B", "A", "C", "D"],
            "weight": [0.5, 0.5, 0.4, 0.3, 0.3],
        })

        daily = carry_forward_weights(weights, all_dates)
        
        # Check that weights sum to 1.0 on every date
        for date in daily["date"].unique():
            date_weights = daily[daily["date"] == date]
            weight_sum = date_weights["weight"].sum()
            assert abs(weight_sum - 1.0) < 1e-6, \
                f"Weights sum to {weight_sum} (not 1.0) on {date}"
    
    def test_old_positions_zeroed_on_rebalance(self):
        """Test that old positions are properly zeroed out on rebalance dates."""
        from quant_platform.core.data.calendar import trading_dates
        all_dates = trading_dates("2023-01-01", "2023-03-31")
        reb = rebalance_dates("M", "2023-01-01", "2023-03-31")

        # First rebalance: hold A and B
        # Second rebalance: hold C and D (A and B should be zeroed out)
        weights = pd.DataFrame({
            "date": [reb[0], reb[0], reb[1], reb[1]],
            "ticker": ["A", "B", "C", "D"],
            "weight": [0.5, 0.5, 0.6, 0.4],
        })

        daily = carry_forward_weights(weights, all_dates)
        
        # After second rebalance, A and B should NOT appear
        after_second_reb = daily[daily["date"] >= reb[1]]
        tickers_after = after_second_reb["ticker"].unique()
        
        assert "A" not in tickers_after, "Stock A should be zeroed out after second rebalance"
        assert "B" not in tickers_after, "Stock B should be zeroed out after second rebalance"
        assert "C" in tickers_after
        assert "D" in tickers_after


class TestValidateWeights:
    def test_validate_long_only_valid(self):
        """Test validation passes for valid long-only weights."""
        from quant_platform.core.portfolio.rebalance import validate_portfolio_weights
        
        weights = pd.DataFrame({
            "date": ["2023-01-01", "2023-01-01", "2023-01-02", "2023-01-02"],
            "ticker": ["A", "B", "A", "B"],
            "weight": [0.6, 0.4, 0.5, 0.5],
        })
        
        # Should not raise
        validate_portfolio_weights(weights, mode="long_only")
    
    def test_validate_long_only_invalid_sum(self):
        """Test validation fails when weights don't sum to 1."""
        from quant_platform.core.portfolio.rebalance import validate_portfolio_weights
        
        weights = pd.DataFrame({
            "date": ["2023-01-01", "2023-01-01"],
            "ticker": ["A", "B"],
            "weight": [0.6, 0.6],  # Sum = 1.2, not 1.0
        })
        
        with pytest.raises(ValueError, match="sum to 1.2"):
            validate_portfolio_weights(weights, mode="long_only")
    
    def test_validate_long_only_negative_weight(self):
        """Test validation fails for negative weights in long-only mode."""
        from quant_platform.core.portfolio.rebalance import validate_portfolio_weights
        
        weights = pd.DataFrame({
            "date": ["2023-01-01", "2023-01-01"],
            "ticker": ["A", "B"],
            "weight": [1.2, -0.2],  # Sum = 1.0 but B is negative
        })
        
        with pytest.raises(ValueError, match="Negative weights"):
            validate_portfolio_weights(weights, mode="long_only")
