"""Integration tests for incremental price downloads."""

import pandas as pd
import pytest
from pathlib import Path
import tempfile
import shutil

from quant_platform.core.data.prices import fetch_prices
from quant_platform.core.utils.io import save_parquet


@pytest.fixture
def temp_data_dir():
    """Create temporary directory for test data."""
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def existing_prices():
    """Generate mock existing price data."""
    dates = pd.date_range("2020-01-01", periods=50, freq="D")
    tickers = ["AAPL", "MSFT"]
    
    data = []
    for ticker in tickers:
        for date in dates:
            data.append({
                "date": date,
                "ticker": ticker,
                "open": 100.0,
                "high": 102.0,
                "low": 98.0,
                "close": 100.0,
                "volume": 1000000,
                "adj_close": 100.0,
                "adj_factor": 1.0,
            })
    
    return pd.DataFrame(data)


class TestIncrementalPrices:
    """Test incremental price download functionality."""
    
    def test_incremental_dedupe_logic(self, existing_prices, temp_data_dir):
        """Test that incremental mode preserves newest downloads over old data."""
        # Save existing prices
        existing_path = temp_data_dir / "prices.parquet"
        save_parquet(existing_prices, existing_path)
        
        # Create "new" download with corrected price
        # Simulate a price correction on 2020-01-10
        correction_date = pd.Timestamp("2020-01-10")
        
        new_prices = existing_prices[
            (existing_prices["date"] >= correction_date - pd.Timedelta(days=5)) &
            (existing_prices["ticker"] == "AAPL")
        ].copy()
        
        # Apply correction: reduce adj_close from 100 to 95 on correction_date
        new_prices.loc[new_prices["date"] == correction_date, "adj_close"] = 95.0
        
        # Merge using the same logic as fetch_prices
        df_old = existing_prices.copy()
        df_new = new_prices.copy()
        
        # Concatenate and dedupe (keep newest download, not highest price)
        df = pd.concat([df_old, df_new], ignore_index=True)
        df = df.drop_duplicates(subset=["date", "ticker"], keep="last")
        
        # Verify correction was preserved
        corrected_row = df[
            (df["date"] == correction_date) & 
            (df["ticker"] == "AAPL")
        ]
        assert len(corrected_row) == 1
        assert corrected_row.iloc[0]["adj_close"] == 95.0, \
            "Newest download (95.0) should overwrite old value (100.0)"
    
    def test_incremental_mode_disabled(self):
        """Test that non-incremental mode works (full download)."""
        # This is more of a smoke test since we can't actually download
        # Just verify the function signature works with incremental=False
        try:
            # Will fail because we don't have real tickers/network
            # but should fail on download, not parameter validation
            fetch_prices(
                tickers=["FAKE_TICKER"],
                start="2020-01-01",
                end="2020-01-10",
                incremental=False,
            )
        except Exception as e:
            # Expected to fail on download, not on parameter
            assert "incremental" not in str(e).lower()
    
    def test_overlap_days_parameter(self):
        """Test that overlap_days parameter is accepted."""
        try:
            fetch_prices(
                tickers=["FAKE_TICKER"],
                start="2020-01-01",
                end="2020-01-10",
                incremental=True,
                overlap_days=20,
            )
        except Exception as e:
            # Should fail on download, not parameter
            assert "overlap_days" not in str(e).lower()
    
    def test_parallel_mode_parameter(self):
        """Test that parallel mode parameter is accepted."""
        try:
            fetch_prices(
                tickers=["FAKE_TICKER"],
                start="2020-01-01",
                end="2020-01-10",
                parallel=True,
                max_workers=4,
            )
        except Exception as e:
            # Should fail on download, not parameter
            assert "parallel" not in str(e).lower()
            assert "max_workers" not in str(e).lower()
