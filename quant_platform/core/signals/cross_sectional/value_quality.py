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


def _compute_report_yoy_change(
    fund: pd.DataFrame,
    metric_col: str,
    *,
    ticker_col: str = "ticker",
    date_col: str = "date",
    report_col: str = "report_date",
    tolerance_days: int = 120,
) -> pd.DataFrame:
    """Compute year-over-year change using fiscal report dates, not row counts."""
    required_cols = [ticker_col, date_col, report_col, metric_col]
    if any(col not in fund.columns for col in required_cols):
        return pd.DataFrame(columns=[date_col, ticker_col, f"{metric_col}_yoy"])

    fund = fund[required_cols].copy()
    fund[date_col] = pd.to_datetime(fund[date_col])
    fund[report_col] = pd.to_datetime(fund[report_col])
    fund = fund.dropna(subset=[metric_col, report_col])

    results = []
    tolerance = pd.Timedelta(days=tolerance_days)

    for ticker, group in fund.groupby(ticker_col):
        reports = (
            group[[report_col, metric_col]]
            .sort_values(report_col)
            .drop_duplicates(subset=[report_col], keep="last")
            .reset_index(drop=True)
        )
        if reports.empty:
            continue

        targets = reports[[report_col]].copy()
        targets["_target_report_date"] = targets[report_col] - pd.DateOffset(years=1)

        history = reports.rename(columns={
            report_col: "_matched_report_date",
            metric_col: f"{metric_col}_lag1y",
        })

        matched = pd.merge_asof(
            targets.sort_values("_target_report_date"),
            history.sort_values("_matched_report_date"),
            left_on="_target_report_date",
            right_on="_matched_report_date",
            direction="nearest",
            tolerance=tolerance,
        )
        matched[metric_col] = reports[metric_col].values
        matched[f"{metric_col}_yoy"] = (
            (matched[metric_col] - matched[f"{metric_col}_lag1y"])
            / matched[f"{metric_col}_lag1y"]
        )

        report_yoy = matched[[report_col, f"{metric_col}_yoy"]]
        expanded = group.merge(report_yoy, on=report_col, how="left")
        results.append(expanded[[date_col, ticker_col, report_col, f"{metric_col}_yoy"]])

    if not results:
        return pd.DataFrame(columns=[date_col, ticker_col, report_col, f"{metric_col}_yoy"])

    return pd.concat(results, ignore_index=True)


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
        
        yoy = _compute_report_yoy_change(
            fundamentals,
            "total_assets",
            tolerance_days=120,
        )
        yoy = yoy.rename(columns={"total_assets_yoy": "asset_growth"})

        merged = _merge_fundamentals_to_prices(prices, yoy, ["asset_growth"])
        result = merged.set_index(["date", "ticker"])["asset_growth"]
        return result


# ===================================================================
# Accruals (Sloan 1996)
# ===================================================================

class Accruals(FactorBase):
    """Accruals = (Net Income - Cash from Operations) / Total Assets.

    References
    ----------
    Sloan (1996) "Do Stock Prices Fully Reflect Information in Accruals
    and Cash Flows about Future Earnings?", The Accounting Review.

    Intuition: high accruals = earnings driven by accounting estimates
    (working-capital changes, depreciation) rather than realised cash.
    These earnings are less persistent and the market under-reacts to
    that signal — high-accruals firms underperform low-accruals firms.

    Therefore the factor's ``direction`` is **-1**: lower (more negative)
    accruals are better.

    Implementation
    --------------
    We use the cash-flow-based definition (Hribar & Collins 2002), which
    is more reliable than the balance-sheet definition: ``accruals_i,t =
    NI_i,t - CFO_i,t``.  Both are TTM aggregates to smooth quarterly
    seasonality.  Scaling by total assets makes the metric cross-sectionally
    comparable.

    Look-ahead handling
    -------------------
    Both ``net_income`` and ``cash_from_operations`` arrive on the same
    publish_date in the fundamentals table; total assets is from the same
    report.  The PIT publish_date is already enforced in the fundamentals
    pipeline, so a simple TTM + cross-merge is bias-free.
    """

    meta = FactorMeta(
        name="ACCRUALS",
        category="quality",
        signal_type="cross_sectional",
        description="(NI_TTM - CFO_TTM) / total_assets — lower is better (Sloan 1996)",
        lookback=1,
        lag=1,
        direction=-1,
        references=[
            "Sloan (1996) Accounting Review",
            "Hribar & Collins (2002) JAR",
        ],
    )

    def _compute(
        self,
        prices: pd.DataFrame,
        fundamentals: Optional[pd.DataFrame] = None,
    ) -> pd.Series:
        if fundamentals is None or fundamentals.empty:
            logger.warning("ACCRUALS: no fundamentals data provided")
            return pd.Series(dtype=float)
        required = {"net_income", "cash_from_operations", "total_assets"}
        missing = required - set(fundamentals.columns)
        if missing:
            logger.warning("ACCRUALS: fundamentals missing columns: {}", sorted(missing))
            return pd.Series(dtype=float)

        ni_ttm = _compute_ttm(fundamentals, "net_income")
        cfo_ttm = _compute_ttm(fundamentals, "cash_from_operations")
        ta = (
            fundamentals[["date", "ticker", "total_assets"]]
            .drop_duplicates(["date", "ticker"], keep="last")
        )

        merged = ni_ttm.merge(cfo_ttm, on=["date", "ticker"], how="inner")
        merged = merged.merge(ta, on=["date", "ticker"], how="left")
        merged["accruals"] = (
            (merged["net_income_ttm"] - merged["cash_from_operations_ttm"])
            / merged["total_assets"].replace(0, np.nan)
        )
        merged = merged[["date", "ticker", "accruals"]]

        full = _merge_fundamentals_to_prices(prices, merged, ["accruals"])
        result = full.set_index(["date", "ticker"])["accruals"]
        return result


# ===================================================================
# Piotroski F-Score (Piotroski 2000) — 8 of 9 indicators (drops the
# current ratio leg, which we cannot compute from the available schema)
# ===================================================================

class PiotroskiFScore(FactorBase):
    """8-of-9 Piotroski Fundamental Score, integer 0..8 — higher is better.

    References
    ----------
    Piotroski (2000) "Value Investing: The Use of Historical Financial
    Statement Information to Separate Winners from Losers from High Book-
    to-Market Firms", Journal of Accounting Research.

    The full Piotroski (2000) F-Score has 9 indicators; we omit the
    "change in current ratio" indicator because the available SimFin /
    yfinance fundamentals schema does not split current vs. non-current
    assets.  The remaining 8 indicators capture profitability, leverage,
    and efficiency.  Empirically the truncated 8-leg score retains most of
    the original anomaly's predictive power (see Piotroski 2000 Table 5).

    The 8 indicators (each contributing 0 or 1):

    1. ROA > 0                                    — current profitability
    2. CFO > 0                                     — cash-flow profitability
    3. Delta ROA > 0                              — improving profitability
    4. CFO > Net Income                           — accruals quality
    5. Delta (total_debt / total_assets) < 0      — deleveraging
    6. No new equity issuance (Delta shares <= 0) — financing discipline
    7. Delta (gross_profit / revenue) > 0         — improving margins
    8. Delta (revenue / total_assets) > 0          — improving asset turnover

    Direction is +1 (higher score → better next-period return).
    """

    meta = FactorMeta(
        name="PIOTROSKI_F8",
        category="quality",
        signal_type="cross_sectional",
        description="Piotroski 8-of-9 fundamental score (drops current-ratio leg)",
        lookback=1,
        lag=1,
        direction=1,
        references=["Piotroski (2000) JAR"],
    )

    def _compute(
        self,
        prices: pd.DataFrame,
        fundamentals: Optional[pd.DataFrame] = None,
    ) -> pd.Series:
        if fundamentals is None or fundamentals.empty:
            logger.warning("PIOTROSKI_F8: no fundamentals data provided")
            return pd.Series(dtype=float)

        required = {
            "net_income", "cash_from_operations", "total_assets",
            "total_debt", "shares_outstanding",
            "gross_profit", "revenue",
        }
        missing = required - set(fundamentals.columns)
        if missing:
            logger.warning("PIOTROSKI_F8: fundamentals missing columns: {}", sorted(missing))
            return pd.Series(dtype=float)

        # TTM aggregates for income/cash flow items
        ni_ttm = _compute_ttm(fundamentals, "net_income")
        cfo_ttm = _compute_ttm(fundamentals, "cash_from_operations")
        gp_ttm = _compute_ttm(fundamentals, "gross_profit")
        rev_ttm = _compute_ttm(fundamentals, "revenue")

        # Balance-sheet snapshot at the latest report
        bs = (
            fundamentals[["date", "ticker", "total_assets", "total_debt", "shares_outstanding"]]
            .drop_duplicates(["date", "ticker"], keep="last")
        )

        merged = (
            ni_ttm
            .merge(cfo_ttm, on=["date", "ticker"], how="inner")
            .merge(gp_ttm, on=["date", "ticker"], how="left")
            .merge(rev_ttm, on=["date", "ticker"], how="left")
            .merge(bs, on=["date", "ticker"], how="left")
        )

        # YoY changes via report-date alignment
        roa_yoy = _piotroski_yoy_levels(
            fundamentals,
            metric_col="net_income",
            scale_col="total_assets",
        )
        leverage_yoy = _piotroski_yoy_levels(
            fundamentals,
            metric_col="total_debt",
            scale_col="total_assets",
        )
        margin_yoy = _piotroski_yoy_levels(
            fundamentals,
            metric_col="gross_profit",
            scale_col="revenue",
        )
        turnover_yoy = _piotroski_yoy_levels(
            fundamentals,
            metric_col="revenue",
            scale_col="total_assets",
        )
        shares_yoy = _compute_report_yoy_change(fundamentals, "shares_outstanding")

        for name, yoy_df in (
            ("roa_yoy", roa_yoy),
            ("leverage_yoy", leverage_yoy),
            ("margin_yoy", margin_yoy),
            ("turnover_yoy", turnover_yoy),
        ):
            if yoy_df.empty:
                merged[name] = np.nan
            else:
                merged = merged.merge(
                    yoy_df.rename(columns={f"{yoy_df.columns[-1]}": name}),
                    on=["date", "ticker"],
                    how="left",
                )
        if not shares_yoy.empty:
            merged = merged.merge(
                shares_yoy[["date", "ticker", "shares_outstanding_yoy"]].rename(
                    columns={"shares_outstanding_yoy": "shares_yoy"}
                ),
                on=["date", "ticker"],
                how="left",
            )
        else:
            merged["shares_yoy"] = np.nan

        ta = merged["total_assets"].replace(0, np.nan)
        # 8 binary indicators
        merged["F1_roa_pos"]      = (merged["net_income_ttm"] / ta > 0).astype(float)
        merged["F2_cfo_pos"]      = (merged["cash_from_operations_ttm"] > 0).astype(float)
        merged["F3_delta_roa"]    = (merged["roa_yoy"] > 0).astype(float)
        merged["F4_accrual_qual"] = (merged["cash_from_operations_ttm"] > merged["net_income_ttm"]).astype(float)
        merged["F5_delta_lev"]    = (merged["leverage_yoy"] < 0).astype(float)
        merged["F6_no_issuance"]  = (merged["shares_yoy"].fillna(0) <= 0).astype(float)
        merged["F7_delta_margin"] = (merged["margin_yoy"] > 0).astype(float)
        merged["F8_delta_turn"]   = (merged["turnover_yoy"] > 0).astype(float)

        leg_cols = [c for c in merged.columns if c.startswith("F") and "_" in c]
        # Where any leg is NaN, treat that leg as 0 (Piotroski's original
        # convention).  Aggregating with sum() handles this naturally.
        merged["piotroski_f8"] = merged[leg_cols].fillna(0).sum(axis=1)

        score = merged[["date", "ticker", "piotroski_f8"]]
        full = _merge_fundamentals_to_prices(prices, score, ["piotroski_f8"])
        return full.set_index(["date", "ticker"])["piotroski_f8"]


def _piotroski_yoy_levels(
    fund: pd.DataFrame,
    *,
    metric_col: str,
    scale_col: str,
    ticker_col: str = "ticker",
    date_col: str = "date",
    report_col: str = "report_date",
    tolerance_days: int = 120,
) -> pd.DataFrame:
    """YoY change of ``metric_col / scale_col`` aligned by report_date.

    Builds the same-quarter-prior-year ratio and returns the year-over-year
    delta in level form (so signs are directly interpretable as
    "improving / deteriorating").
    """
    required = [ticker_col, date_col, report_col, metric_col, scale_col]
    if any(col not in fund.columns for col in required):
        return pd.DataFrame()

    fund = fund[required].copy()
    fund[date_col] = pd.to_datetime(fund[date_col])
    fund[report_col] = pd.to_datetime(fund[report_col])
    fund["_ratio"] = fund[metric_col] / fund[scale_col].replace(0, np.nan)
    fund = fund.dropna(subset=["_ratio", report_col])

    out_col = f"{metric_col}_over_{scale_col}_yoy"
    results = []
    tolerance = pd.Timedelta(days=tolerance_days)

    for ticker, grp in fund.groupby(ticker_col):
        rep = (
            grp[[report_col, "_ratio"]]
            .sort_values(report_col)
            .drop_duplicates(report_col, keep="last")
            .reset_index(drop=True)
        )
        if rep.empty:
            continue
        target = rep[[report_col]].copy()
        target["_target_lag"] = target[report_col] - pd.DateOffset(years=1)

        history = rep.rename(columns={report_col: "_matched", "_ratio": "_ratio_lag"})
        matched = pd.merge_asof(
            target.sort_values("_target_lag"),
            history.sort_values("_matched"),
            left_on="_target_lag",
            right_on="_matched",
            direction="nearest",
            tolerance=tolerance,
        )
        matched["_ratio_now"] = rep["_ratio"].values
        matched[out_col] = matched["_ratio_now"] - matched["_ratio_lag"]

        sub = matched[[report_col, out_col]]
        expanded = grp.merge(sub, on=report_col, how="left")
        results.append(expanded[[date_col, ticker_col, out_col]])

    if not results:
        return pd.DataFrame()
    return pd.concat(results, ignore_index=True).drop_duplicates([date_col, ticker_col], keep="last")


# ===================================================================
# Registry
# ===================================================================

ALL_VALUE_QUALITY_FACTORS = [
    EarningsYield,
    BookToMarket,
    GrossProfitability,
    AssetGrowth,
    Accruals,
    PiotroskiFScore,
]
