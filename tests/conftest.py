"""Shared test fixtures for EntroPy test suite."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Ensure project root is importable
_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root))

from quant_platform.core.utils.io import set_project_root
set_project_root(_root)


@pytest.fixture
def sample_prices() -> pd.DataFrame:
    """Generate a small synthetic price DataFrame for testing."""
    np.random.seed(42)
    dates = pd.bdate_range("2023-01-03", periods=60, freq="B")  # ~3 months
    tickers = ["AAPL", "MSFT", "GOOGL", "JPM", "XOM"]

    rows = []
    for tkr in tickers:
        base_price = np.random.uniform(50, 300)
        cumret = np.cumprod(1 + np.random.normal(0.0005, 0.02, len(dates)))
        closes = base_price * cumret
        for i, dt in enumerate(dates):
            c = closes[i]
            rows.append({
                "date": dt,
                "ticker": tkr,
                "open": c * np.random.uniform(0.99, 1.01),
                "high": c * np.random.uniform(1.00, 1.03),
                "low": c * np.random.uniform(0.97, 1.00),
                "close": c,
                "volume": int(np.random.uniform(1e6, 1e7)),
                "amount": c * np.random.uniform(1e6, 1e7),
                "adj_factor": 1.0,
                "adj_close": c,
                "is_tradable": True,
            })

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df


@pytest.fixture
def sample_weights(sample_prices) -> pd.DataFrame:
    """Generate simple equal-weight portfolio weights."""
    tickers = sample_prices["ticker"].unique()
    dates = sorted(sample_prices["date"].unique())
    # Monthly rebalance: pick last date of each month
    monthly = pd.DataFrame({"date": dates})
    monthly["ym"] = pd.to_datetime(monthly["date"]).dt.to_period("M")
    reb_dates = monthly.groupby("ym")["date"].max().values

    rows = []
    w = 1.0 / len(tickers)
    for dt in reb_dates:
        for tkr in tickers:
            rows.append({"date": dt, "ticker": tkr, "weight": w})
    return pd.DataFrame(rows)
