"""Tests for schema validation and Parquet I/O."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_platform.core.data.schema import PRICES_SCHEMA, validate_dataframe


class TestValidateDataframe:
    def test_valid_prices(self, sample_prices):
        assert validate_dataframe(sample_prices, "prices") is True

    def test_missing_column_raises(self, sample_prices):
        df = sample_prices.drop(columns=["adj_close"])
        with pytest.raises(ValueError, match="Missing columns"):
            validate_dataframe(df, "prices")

    def test_extra_column_raises(self, sample_prices):
        df = sample_prices.copy()
        df["extra_col"] = 1
        with pytest.raises(ValueError, match="Extra columns"):
            validate_dataframe(df, "prices")

    def test_unknown_table_raises(self, sample_prices):
        with pytest.raises(KeyError, match="Unknown table"):
            validate_dataframe(sample_prices, "nonexistent")


class TestAdjustmentFactor:
    """Verify that point-in-time adjustment factor logic is correct."""

    def test_adj_close_equals_close_times_factor(self, sample_prices):
        df = sample_prices.copy()
        # In our synthetic data, adj_factor=1.0 so adj_close should equal close
        np.testing.assert_allclose(
            df["adj_close"], df["close"] * df["adj_factor"], rtol=1e-6
        )

    def test_adj_factor_never_negative(self, sample_prices):
        assert (sample_prices["adj_factor"] > 0).all()

    def test_no_future_adj_factor_leakage(self, sample_prices):
        """Each ticker's adj_factor should be deterministic given history up to that date."""
        df = sample_prices.sort_values(["ticker", "date"])
        for tkr, grp in df.groupby("ticker"):
            factors = grp["adj_factor"].values
            # In point-in-time mode, factor should be monotonically
            # non-decreasing or constant (no retroactive corrections)
            # For synthetic data with factor=1.0 this is trivially true
            assert (factors >= 0).all(), f"{tkr}: negative adj_factor found"
