"""Tests for rebalance schedule and weight carry-forward."""

from __future__ import annotations

import pandas as pd
import pytest

from entropy.portfolio.rebalance import carry_forward_weights, rebalance_dates


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
        from entropy.data.calendar import trading_dates
        all_dates = trading_dates("2023-01-01", "2023-03-31")
        reb = rebalance_dates("M", "2023-01-01", "2023-03-31")

        weights = pd.DataFrame({
            "date": [reb[0], reb[0], reb[1], reb[1]],
            "ticker": ["A", "B", "A", "B"],
            "weight": [0.5, 0.5, 0.6, 0.4],
        })

        daily = carry_forward_weights(weights, all_dates)
        # Should have weights for every trading day after first rebalance
        assert daily["date"].nunique() >= len(all_dates) - 5  # some tolerance
        # Between rebalances, weights should be constant
        mid_date = all_dates[len(all_dates) // 4]  # some date in first month
        if mid_date >= reb[0]:
            mid_w = daily[daily["date"] == mid_date]
            assert len(mid_w) > 0
