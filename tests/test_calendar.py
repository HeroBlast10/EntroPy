"""Tests for trading calendar alignment."""

from __future__ import annotations

import pandas as pd
import pytest

from entropy.data.calendar import (
    align_to_calendar,
    is_trading_day,
    next_trading_day,
    prev_trading_day,
    trading_dates,
)


class TestTradingDates:
    def test_returns_datetimeindex(self):
        dates = trading_dates("2023-01-01", "2023-01-31")
        assert isinstance(dates, pd.DatetimeIndex)
        assert len(dates) > 0

    def test_no_weekends(self):
        dates = trading_dates("2023-01-01", "2023-12-31")
        weekdays = dates.weekday
        assert (weekdays < 5).all(), "Trading dates should not include weekends"

    def test_excludes_holidays(self):
        # 2023-01-02 is observed New Year's Day (Monday) — NYSE closed
        dates = trading_dates("2023-01-01", "2023-01-10")
        assert pd.Timestamp("2023-01-02") not in dates

    def test_monotonically_increasing(self):
        dates = trading_dates("2023-01-01", "2023-06-30")
        assert (dates[1:] > dates[:-1]).all()


class TestPointQueries:
    def test_is_trading_day_weekday(self):
        # 2023-01-03 is a Tuesday (first trading day of 2023)
        assert is_trading_day("2023-01-03") is True

    def test_is_not_trading_day_weekend(self):
        assert is_trading_day("2023-01-07") is False  # Saturday

    def test_next_trading_day(self):
        # After Friday 2023-01-06 → Monday 2023-01-09
        nxt = next_trading_day("2023-01-06")
        assert nxt == pd.Timestamp("2023-01-09")

    def test_prev_trading_day(self):
        # Before Monday 2023-01-09 → Friday 2023-01-06
        prv = prev_trading_day("2023-01-09")
        assert prv == pd.Timestamp("2023-01-06")


class TestAlignToCalendar:
    def test_inner_drops_weekends(self):
        df = pd.DataFrame({
            "date": pd.date_range("2023-01-01", "2023-01-10"),
            "value": range(10),
        })
        aligned = align_to_calendar(df, date_col="date",
                                     method="inner", start="2023-01-01", end="2023-01-10")
        weekdays = pd.to_datetime(aligned["date"]).dt.weekday
        assert (weekdays < 5).all()

    def test_inner_preserves_trading_days(self):
        dates = trading_dates("2023-01-01", "2023-01-31")
        df = pd.DataFrame({"date": dates, "value": range(len(dates))})
        aligned = align_to_calendar(df, date_col="date",
                                     method="inner", start="2023-01-01", end="2023-01-31")
        assert len(aligned) == len(dates)
