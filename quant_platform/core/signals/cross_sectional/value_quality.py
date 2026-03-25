"""Value & Quality factors — Fama-French + Novy-Marx + Cooper et al.

Factors
-------
1. **EARNINGS_YIELD (E/P)**   — Trailing 12M net income / market cap (Fama-French value)
2. **BOOK_TO_MARKET (B/M)**   — Book value / market cap (Fama-French HML)
3. **GROSS_PROFITABILITY**    — Gross profit / total assets (Novy-Marx 2013)
4. **ASSET_GROWTH**           — YoY % change in total assets (Cooper, Gulen & Schill 2008)

All factors use **point-in-time** fundamentals data with publication lag
already applied by the fundamentals pipeline (see ``quant_platform.core.data.fundamentals``).
This prevents look-ahead bias.

Trailing 12M (TTM) calculation
-------------------------------
For income statement items (net_income, gross_profit), we compute TTM by:
- Summing the most recent 4 quarters of data
- Falling back to the most recent annual value if quarterly data is incomplete

This is standard practice in fundamental factor research.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

from quant_platform.core.signals.base import FactorBase, FactorMeta


# ===================================================================
# Helpers
# ===================================================================

def _compute_ttm(
    fund: pd.DataFrame,
    metric_col: str,
    ticker_col: str = "ticker",
    date_col: str = "date",
    report_col: str = "report_date",
    n_quarters: int = 4,
) -> pd.DataFrame:
    """Compute trailing 12-month (TTM) sum for an income statement metric.
    
    For each (date, ticker), sum the most recent *n_quarters* of reported values.
    If fewer than *n_quarters* are available, use the most recent single value.
    
    Returns a DataFrame with [date, ticker, {metric_col}_ttm].
    """
    if metric_col not in fund.columns:
        return pd.DataFrame(columns=[date_col, ticker_col, f"{metric_col}_ttm"])
    
    fund = fund.copy()
    fund[date_col] = pd.to_datetime(fund[date_col])
    fund[report_col] = pd.to_datetime(fund[report_col])
    fund = fund.sort_values([ticker_col, date_col, report_col])
    
    results = []
    for (dt, tkr), grp in fund.groupby([date_col, ticker_col]):
        grp = grp.sort_values(report_col)
        vals = grp[metric_col].dropna()
        
        if len(vals) == 0:
            ttm = np.nan
        elif len(vals) >= n_quarters:
            # Sum most recent 4 quarters
            ttm = vals.iloc[-n_quarters:].sum()
        else:
            # Fallback: use most recent value (likely annual)
            ttm = vals.iloc[-1]
        
        results.append({date_col: dt, ticker_col: tkr, f"{metric_col}_ttm": ttm})
    
    return pd.DataFrame(results)


def _merge_fundamentals_to_prices(
    prices: pd.DataFrame,
    fundamentals: pd.DataFrame,
    fund_cols: list,
) -> pd.DataFrame:
    """Left-join fundamentals onto prices, forward-filling within each ticker.
    
    This ensures each trading day has the most recent available fundamental value.
    """
    if fundamentals is None or fundamentals.empty:
        return prices
    
    px = prices[["date", "ticker", "adj_close"]].copy()
    px["date"] = pd.to_datetime(px["date"])
    
    fund = fundamentals[["date", "ticker"] + fund_cols].copy()
    fund["date"] = pd.to_datetime(fund["date"])
    
    # Merge
    merged = px.merge(fund, on=["date", "ticker"], how="left")
    
    # Forward-fill fundamentals within each ticker
    merged = merged.sort_values(["ticker", "date"])
    for col in fund_cols:
        merged[col] = merged.groupby("ticker")[col].ffill()
    
    return merged


# ===================================================================
# Factors
# ===================================================================

class EarningsYield(FactorBase):
    """Earnings Yield = TTM Net Income / Market Cap.
    
    Classic value factor from Fama-French. Higher E/P indicates cheaper valuation.
    
    References
    ----------
    - Basu (1977) "Investment Performance of Common Stocks in Relation to Their
      Price-Earnings Ratios: A Test of the Efficient Market Hypothesis"
    - Fama & French (1992) "The Cross-Section of Expected Stock Returns"
    """
    meta = FactorMeta(
        name="EARNINGS_YIELD",
        category="value",
        signal_type="cross_sectional",
        description="TTM net income / market cap (E/P ratio)",
        lookback=1,
        lag=1,
        direction=1,
        references=["Basu (1977)", "Fama & French (1992)"],
    )

    def _compute(
        self,
        prices: pd.DataFrame,
        fundamentals: Optional[pd.DataFrame] = None,
    ) -> pd.Series:
        if fundamentals is None or fundamentals.empty:
            logger.warning("EARNINGS_YIELD: no fundamentals data provided")
            return pd.Series(dtype=float)
        
        # Compute TTM net income
        ttm = _compute_ttm(fundamentals, "net_income")
        
        # Merge with market cap
        fund_subset = fundamentals[["date", "ticker", "market_cap"]].copy()
        ttm = ttm.merge(fund_subset, on=["date", "ticker"], how="left")
        
        # E/P = net_income_ttm / market_cap
        ttm["ep"] = ttm["net_income_ttm"] / ttm["market_cap"]
        
        # Merge to prices grid
        merged = _merge_fundamentals_to_prices(prices, ttm, ["ep"])
        
        result = merged.set_index(["date", "ticker"])["ep"]
        return result


class BookToMarket(FactorBase):
    """Book-to-Market = Book Value / Market Cap.
    
    Core value factor in Fama-French 3-factor model (HML = High Minus Low B/M).
    
    References
    ----------
    - Fama & French (1993) "Common Risk Factors in the Returns on Stocks and Bonds"
    - Rosenberg, Reid & Lanstein (1985) "Persuasive Evidence of Market Inefficiency"
    """
    meta = FactorMeta(
        name="BOOK_TO_MARKET",
        category="value",
        signal_type="cross_sectional",
        description="Book value / market cap (Fama-French HML)",
        lookback=1,
        lag=1,
        direction=1,
        references=["Fama & French (1993)"],
    )

    def _compute(
        self,
        prices: pd.DataFrame,
        fundamentals: Optional[pd.DataFrame] = None,
    ) -> pd.Series:
        if fundamentals is None or fundamentals.empty:
            logger.warning("BOOK_TO_MARKET: no fundamentals data provided")
            return pd.Series(dtype=float)
        
        fund = fundamentals[["date", "ticker", "total_equity", "market_cap"]].copy()
        fund["bm"] = fund["total_equity"] / fund["market_cap"]
        
        merged = _merge_fundamentals_to_prices(prices, fund, ["bm"])
        result = merged.set_index(["date", "ticker"])["bm"]
        return result


class GrossProfitability(FactorBase):
    """Gross Profitability = TTM Gross Profit / Total Assets.
    
    Quality/profitability factor from Novy-Marx (2013). Firms with high gross
    profitability (relative to assets) tend to outperform.
    
    References
    ----------
    - Novy-Marx (2013) "The Other Side of Value: The Gross Profitability Premium"
    - Fama & French (2015) 5-factor model (RMW = Robust Minus Weak profitability)
    """
    meta = FactorMeta(
        name="GROSS_PROFITABILITY",
        category="quality",
        signal_type="cross_sectional",
        description="TTM gross profit / total assets (Novy-Marx 2013)",
        lookback=1,
        lag=1,
        direction=1,
        references=["Novy-Marx (2013)", "Fama & French (2015)"],
    )

    def _compute(
        self,
        prices: pd.DataFrame,
        fundamentals: Optional[pd.DataFrame] = None,
    ) -> pd.Series:
        if fundamentals is None or fundamentals.empty:
            logger.warning("GROSS_PROFITABILITY: no fundamentals data provided")
            return pd.Series(dtype=float)
        
        # Compute TTM gross profit
        ttm = _compute_ttm(fundamentals, "gross_profit")
        
        # Merge with total assets
        fund_subset = fundamentals[["date", "ticker", "total_assets"]].copy()
        ttm = ttm.merge(fund_subset, on=["date", "ticker"], how="left")
        
        # GP/A = gross_profit_ttm / total_assets
        ttm["gpa"] = ttm["gross_profit_ttm"] / ttm["total_assets"]
        
        merged = _merge_fundamentals_to_prices(prices, ttm, ["gpa"])
        result = merged.set_index(["date", "ticker"])["gpa"]
        return result


class AssetGrowth(FactorBase):
    """Asset Growth = YoY % change in Total Assets.
    
    Firms with high asset growth tend to underperform (Cooper, Gulen & Schill 2008).
    This is a negative predictor — low asset growth is better.
    
    References
    ----------
    - Cooper, Gulen & Schill (2008) "Asset Growth and the Cross-Section of Stock Returns"
    - Fama & French (2015) 5-factor model (CMA = Conservative Minus Aggressive investment)
    """
    meta = FactorMeta(
        name="ASSET_GROWTH",
        category="quality",
        signal_type="cross_sectional",
        description="YoY % change in total assets (lower is better)",
        lookback=252,  # need 1 year of history
        lag=1,
        direction=-1,  # negative: low growth is good
        references=["Cooper, Gulen & Schill (2008)", "Fama & French (2015)"],
    )

    def _compute(
        self,
        prices: pd.DataFrame,
        fundamentals: Optional[pd.DataFrame] = None,
    ) -> pd.Series:
        if fundamentals is None or fundamentals.empty:
            logger.warning("ASSET_GROWTH: no fundamentals data provided")
            return pd.Series(dtype=float)
        
        fund = fundamentals[["date", "ticker", "total_assets"]].copy()
        fund["date"] = pd.to_datetime(fund["date"])
        fund = fund.sort_values(["ticker", "date"])
        
        # Compute YoY growth: (assets_t - assets_t-1y) / assets_t-1y
        # Approximate 1 year = 252 trading days
        fund["assets_lag1y"] = fund.groupby("ticker")["total_assets"].shift(252)
        fund["asset_growth"] = (
            (fund["total_assets"] - fund["assets_lag1y"]) / fund["assets_lag1y"]
        )
        
        merged = _merge_fundamentals_to_prices(prices, fund, ["asset_growth"])
        result = merged.set_index(["date", "ticker"])["asset_growth"]
        return result


# ===================================================================
# Registry
# ===================================================================

ALL_VALUE_QUALITY_FACTORS = [
    EarningsYield,
    BookToMarket,
    GrossProfitability,
    AssetGrowth,
]
