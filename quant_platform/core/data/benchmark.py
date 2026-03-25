"""Benchmark data fetcher for US and CN equity markets.

Benchmarks
----------
- **SPY**: S&P 500 ETF (US equity market proxy)
- **CSI300** (000300.SS): CSI 300 Index (CN A-share market proxy)

Data is fetched via yfinance and cached locally. Returns are computed
as simple percentage returns on adjusted close prices.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
import yfinance as yf
from loguru import logger

from quant_platform.core.data.calendar import align_to_calendar, trading_dates
from quant_platform.core.utils.io import load_config, resolve_data_path, save_parquet, load_parquet


# ===================================================================
# Benchmark tickers
# ===================================================================

BENCHMARK_TICKERS = {
    "us": "SPY",           # S&P 500 ETF
    "cn": "000300.SS",     # CSI 300 Index
}


# ===================================================================
# Fetch benchmark data
# ===================================================================

def fetch_benchmark(
    market: str = "us",
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> pd.DataFrame:
    """Fetch benchmark price data from yfinance.
    
    Parameters
    ----------
    market : "us" or "cn"
    start : start date (YYYY-MM-DD), default = 5 years ago
    end : end date (YYYY-MM-DD), default = today
    
    Returns
    -------
    DataFrame with columns [date, ticker, adj_close, return]
    """
    ticker = BENCHMARK_TICKERS.get(market)
    if ticker is None:
        raise ValueError(f"Unknown market: {market!r}. Choose 'us' or 'cn'.")
    
    logger.info("Fetching benchmark {} for market {}", ticker, market)
    
    # Default date range: 5 years
    if start is None:
        start = (pd.Timestamp.now() - pd.DateOffset(years=5)).strftime("%Y-%m-%d")
    if end is None:
        end = pd.Timestamp.now().strftime("%Y-%m-%d")
    
    # Fetch from yfinance
    try:
        data = yf.download(ticker, start=start, end=end, progress=False)
    except Exception as exc:
        logger.error("Failed to fetch benchmark {}: {}", ticker, exc)
        return pd.DataFrame(columns=["date", "ticker", "adj_close", "return"])
    
    if data.empty:
        logger.warning("No data returned for benchmark {}", ticker)
        return pd.DataFrame(columns=["date", "ticker", "adj_close", "return"])
    
    # Normalize to our schema
    df = pd.DataFrame({
        "date": data.index,
        "ticker": ticker,
        "adj_close": data["Adj Close"].values,
    })
    
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    
    # Compute returns
    df["return"] = df["adj_close"].pct_change()
    
    logger.info("Fetched {} rows for benchmark {}", len(df), ticker)
    return df


# ===================================================================
# Build and cache benchmark data
# ===================================================================

def build_benchmark(
    market: str = "us",
    start: Optional[str] = None,
    end: Optional[str] = None,
    output_path: Optional[Path | str] = None,
) -> Path:
    """Fetch benchmark data and save to parquet.
    
    Returns the output path.
    """
    cfg = load_config()
    
    df = fetch_benchmark(market, start, end)
    
    if df.empty:
        logger.warning("No benchmark data to save")
        return None
    
    # Align to trading calendar
    df = align_to_calendar(df, date_col="date")
    
    # Save
    if output_path is None:
        bench_dir = resolve_data_path(cfg["paths"].get("benchmark_dir", "benchmark"))
        output_path = Path(bench_dir) / f"benchmark_{market}.parquet"
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    df.to_parquet(output_path, index=False)
    logger.info("Saved benchmark to {}", output_path)
    
    return output_path


# ===================================================================
# Load cached benchmark
# ===================================================================

def load_benchmark(
    market: str = "us",
    path: Optional[Path | str] = None,
) -> pd.DataFrame:
    """Load cached benchmark data.
    
    If path is None, uses default location from config.
    """
    if path is None:
        cfg = load_config()
        bench_dir = resolve_data_path(cfg["paths"].get("benchmark_dir", "benchmark"))
        path = Path(bench_dir) / f"benchmark_{market}.parquet"
    
    path = Path(path)
    if not path.exists():
        logger.warning("Benchmark file not found: {}. Run build_benchmark() first.", path)
        return pd.DataFrame(columns=["date", "ticker", "adj_close", "return"])
    
    df = load_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    return df


# ===================================================================
# Align benchmark to portfolio dates
# ===================================================================

def align_benchmark_to_portfolio(
    portfolio_returns: pd.DataFrame,
    benchmark_returns: pd.DataFrame,
    portfolio_date_col: str = "date",
    benchmark_date_col: str = "date",
    benchmark_return_col: str = "return",
) -> pd.DataFrame:
    """Align benchmark returns to portfolio dates.
    
    Parameters
    ----------
    portfolio_returns : DataFrame with portfolio daily returns (must have date index or column)
    benchmark_returns : DataFrame with benchmark data [date, return]
    
    Returns
    -------
    DataFrame with [date, portfolio_ret, benchmark_ret] aligned to same dates
    """
    # Ensure portfolio has date index
    if isinstance(portfolio_returns.index, pd.DatetimeIndex):
        port = portfolio_returns.copy()
    else:
        port = portfolio_returns.set_index(portfolio_date_col)
    
    # Ensure benchmark has date index
    bench = benchmark_returns.copy()
    bench[benchmark_date_col] = pd.to_datetime(bench[benchmark_date_col])
    bench = bench.set_index(benchmark_date_col)
    
    # Align on date index (inner join)
    aligned = pd.DataFrame({
        "portfolio_ret": port.get("net_ret", port.get("gross_ret", port.iloc[:, 0])),
        "benchmark_ret": bench[benchmark_return_col],
    })
    
    aligned = aligned.dropna()
    
    logger.info("Aligned {} days of portfolio vs benchmark returns", len(aligned))
    return aligned
