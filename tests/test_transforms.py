"""Tests for factor transforms: lag, winsorize, zscore, neutralize, missing."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_platform.core.signals.transforms import (
    apply_lag,
    cross_sectional_rank,
    cross_sectional_zscore,
    handle_missing,
    neutralize,
    winsorize,
)


def _make_factor_df(n_dates=20, n_tickers=5, seed=42):
    """Helper: create a (date, ticker, value) DataFrame."""
    np.random.seed(seed)
    dates = pd.bdate_range("2023-01-02", periods=n_dates, freq="B")
    tickers = [f"T{i}" for i in range(n_tickers)]
    rows = []
    for d in dates:
        for t in tickers:
            rows.append({"date": d, "ticker": t, "signal": np.random.randn()})
    return pd.DataFrame(rows)


class TestApplyLag:
    def test_lag_shifts_values(self):
        df = _make_factor_df()
        lagged = apply_lag(df, "signal", lag=1)
        # First row per ticker should be NaN
        for tkr in df["ticker"].unique():
            sub = lagged[lagged["ticker"] == tkr]
            assert pd.isna(sub.iloc[0]["signal"])

    def test_lag_zero_is_identity(self):
        df = _make_factor_df()
        lagged = apply_lag(df, "signal", lag=0)
        pd.testing.assert_series_equal(
            df.sort_values(["ticker", "date"])["signal"].reset_index(drop=True),
            lagged["signal"].reset_index(drop=True),
        )


class TestHandleMissing:
    def test_drop(self):
        df = _make_factor_df()
        df.loc[0, "signal"] = np.nan
        df.loc[5, "signal"] = np.inf
        result = handle_missing(df, "signal", method="drop")
        assert result["signal"].isna().sum() == 0
        assert np.isinf(result["signal"]).sum() == 0

    def test_zero_fill(self):
        df = _make_factor_df()
        df.loc[0, "signal"] = np.nan
        result = handle_missing(df, "signal", method="zero")
        assert result.loc[0, "signal"] == 0.0

    def test_median_fill(self):
        df = _make_factor_df()
        df.loc[0, "signal"] = np.nan
        result = handle_missing(df, "signal", method="median")
        assert not pd.isna(result.loc[0, "signal"])


class TestWinsorize:
    def test_clips_extremes(self):
        df = _make_factor_df()
        # Insert extreme value
        df.loc[0, "signal"] = 100.0
        result = winsorize(df, "signal", limits=(0.01, 0.99))
        assert result["signal"].max() < 100.0

    def test_no_change_within_limits(self):
        df = _make_factor_df()
        original = df["signal"].copy()
        result = winsorize(df, "signal", limits=(0.0, 1.0))
        pd.testing.assert_series_equal(original, result["signal"], check_names=False)


class TestCrossSectionalZscore:
    def test_zero_mean_per_date(self):
        df = _make_factor_df()
        result = cross_sectional_zscore(df, "signal")
        means = result.groupby("date")["signal"].mean()
        np.testing.assert_allclose(means, 0.0, atol=1e-10)

    def test_unit_std_per_date(self):
        df = _make_factor_df(n_tickers=20)  # need enough for stable std
        result = cross_sectional_zscore(df, "signal")
        stds = result.groupby("date")["signal"].std()
        np.testing.assert_allclose(stds, 1.0, atol=0.15)


class TestCrossSectionalRank:
    def test_rank_between_0_and_1(self):
        df = _make_factor_df()
        result = cross_sectional_rank(df, "signal")
        assert result["signal"].min() >= 0.0
        assert result["signal"].max() <= 1.0


class TestNeutralize:
    def test_categorical_demean(self):
        df = _make_factor_df()
        df["sector"] = df["ticker"].map({"T0": "A", "T1": "A", "T2": "B", "T3": "B", "T4": "B"})
        result = neutralize(df, "signal", group_cols=["sector"])
        # After demean, each (date, sector) group should have mean ≈ 0
        means = result.groupby(["date", "sector"])["signal"].mean()
        np.testing.assert_allclose(means, 0.0, atol=1e-10)
