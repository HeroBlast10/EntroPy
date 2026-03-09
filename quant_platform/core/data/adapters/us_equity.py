"""US equity data adapter.

Wraps the existing data layer (prices, fundamentals, universe, pipeline)
under a unified ``USEquityAdapter`` interface.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd
from loguru import logger

from quant_platform.core.utils.io import load_config, resolve_data_path, load_parquet, save_parquet
from quant_platform.core.data.calendar import trading_dates


class USEquityAdapter:
    """Data adapter for US equities (via yfinance / simfin / SEC EDGAR).

    Delegates to the existing ``quant_platform.core.data`` submodules for
    prices, fundamentals, and universe construction.
    """

    EXCHANGE = "XNYS"
    TICKER_COL = "ticker"

    def __init__(self, config: Optional[dict] = None):
        self.cfg = config or load_config()

    def run_pipeline(self, **kwargs) -> None:
        """Run the full US equity data pipeline (prices -> fundamentals -> universe)."""
        from quant_platform.core.data.pipeline import run_pipeline
        run_pipeline(**kwargs)

    def load_prices(self, path: Optional[str] = None) -> pd.DataFrame:
        if path is None:
            path = resolve_data_path("prices", "prices.parquet")
        return load_parquet(path)

    def load_fundamentals(self, path: Optional[str] = None) -> pd.DataFrame:
        if path is None:
            path = resolve_data_path("fundamentals", "fundamentals.parquet")
        return load_parquet(path)

    def load_universe(self, path: Optional[str] = None) -> pd.DataFrame:
        if path is None:
            path = resolve_data_path("universe", "universe.parquet")
        return load_parquet(path)

    def get_trading_dates(self, start=None, end=None) -> pd.DatetimeIndex:
        return trading_dates(start, end, exchange=self.EXCHANGE)
