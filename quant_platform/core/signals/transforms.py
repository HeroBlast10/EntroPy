"""Cross-sectional transforms: lag, winsorize, z-score, neutralize, missing values.

All functions operate on a DataFrame with columns ``[date, ticker, <value_col>]``
and return a DataFrame of the same shape (or fewer rows after dropping NaNs).
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger


# ===================================================================
# Lag
# ===================================================================

def apply_lag(
    df: pd.DataFrame,
    value_col: str,
    lag: int = 1,
    date_col: str = "date",
    ticker_col: str = "ticker",
) -> pd.DataFrame:
    """Shift factor values forward by *lag* trading days within each ticker.

    This ensures that on date *t* we only see information available up to
    *t − lag*, preventing any intra-day or T+0 leakage.

    The first *lag* rows per ticker will become NaN and are later handled
    by :func:`handle_missing`.
    """
    df = df.copy()
    df.sort_values([ticker_col, date_col], inplace=True)
    df[value_col] = df.groupby(ticker_col)[value_col].shift(lag)
    return df


# ===================================================================
# Missing-value handling
# ===================================================================

def handle_missing(
    df: pd.DataFrame,
    value_col: str,
    method: str = "drop",
    date_col: str = "date",
) -> pd.DataFrame:
    """Handle NaN / inf in the factor column.

    Parameters
    ----------
    method :
        ``"drop"``   — remove rows with NaN.
        ``"zero"``   — fill NaN with 0.
        ``"median"`` — fill NaN with cross-sectional median on each date.
    """
    df = df.copy()
    # Replace inf with NaN first
    df[value_col] = df[value_col].replace([np.inf, -np.inf], np.nan)

    n_missing = df[value_col].isna().sum()
    if n_missing == 0:
        return df

    if method == "drop":
        df = df.dropna(subset=[value_col]).reset_index(drop=True)
    elif method == "zero":
        df[value_col] = df[value_col].fillna(0.0)
    elif method == "median":
        medians = df.groupby(date_col)[value_col].transform("median")
        df[value_col] = df[value_col].fillna(medians)
    else:
        raise ValueError(f"Unknown fillna method: {method!r}")

    logger.debug("handle_missing({}): {} NaN handled via '{}'", value_col, n_missing, method)
    return df


# ===================================================================
# Winsorize
# ===================================================================

def winsorize(
    df: pd.DataFrame,
    value_col: str,
    limits: Tuple[float, float] = (0.01, 0.99),
    date_col: str = "date",
) -> pd.DataFrame:
    """Cross-sectional winsorization: clip to [q_lo, q_hi] each date.

    This is more robust than global winsorization because it adapts to
    the distribution on each cross-section.
    """
    df = df.copy()

    def _clip_group(s: pd.Series) -> pd.Series:
        lo = s.quantile(limits[0])
        hi = s.quantile(limits[1])
        return s.clip(lo, hi)

    df[value_col] = df.groupby(date_col)[value_col].transform(_clip_group)
    return df


# ===================================================================
# Z-score
# ===================================================================

def cross_sectional_zscore(
    df: pd.DataFrame,
    value_col: str,
    date_col: str = "date",
) -> pd.DataFrame:
    """Standardize *value_col* to zero mean, unit std on each date.

    Dates with std == 0 are left as 0 (constant cross-section).
    """
    df = df.copy()

    grp = df.groupby(date_col)[value_col]
    means = grp.transform("mean")
    stds = grp.transform("std")
    stds = stds.replace(0, np.nan)
    df[value_col] = (df[value_col] - means) / stds
    df[value_col] = df[value_col].fillna(0.0)
    return df


# ===================================================================
# Rank transform (useful for RankIC)
# ===================================================================

def cross_sectional_rank(
    df: pd.DataFrame,
    value_col: str,
    date_col: str = "date",
) -> pd.DataFrame:
    """Replace *value_col* with its cross-sectional percentile rank [0, 1]."""
    df = df.copy()
    df[value_col] = df.groupby(date_col)[value_col].rank(pct=True)
    return df


# ===================================================================
# Neutralize (industry / size)
# ===================================================================

def neutralize(
    df: pd.DataFrame,
    value_col: str,
    group_cols: List[str],
    date_col: str = "date",
) -> pd.DataFrame:
    """Neutralize *value_col* against *group_cols* within each date.

    For **categorical** group columns (e.g. ``sector``): demean within each
    (date, sector) group — equivalent to adding sector dummies in a
    cross-sectional regression and taking the residual.

    For **continuous** group columns (e.g. ``log_mcap``): OLS-residualize
    *value_col* against the continuous variable within each date.

    Mixed usage (e.g. ``["sector", "log_mcap"]``) performs sequential
    neutralization: first demean by sector, then residualize vs. log_mcap.
    """
    df = df.copy()

    for col in group_cols:
        if col not in df.columns:
            logger.warning("neutralize: column '{}' not in DataFrame, skipping", col)
            continue

        if df[col].dtype == "object" or pd.api.types.is_categorical_dtype(df[col]):
            # Categorical: demean within (date, group)
            group_mean = df.groupby([date_col, col])[value_col].transform("mean")
            df[value_col] = df[value_col] - group_mean
        else:
            # Continuous: OLS residual within each date
            def _residualize(sub: pd.DataFrame) -> pd.Series:
                y = sub[value_col].values
                x = sub[col].values
                mask = np.isfinite(y) & np.isfinite(x)
                if mask.sum() < 3:
                    return pd.Series(y, index=sub.index)
                xm = x[mask]
                ym = y[mask]
                xm_dm = xm - xm.mean()
                denom = (xm_dm ** 2).sum()
                if denom == 0:
                    return pd.Series(y - np.nanmean(y), index=sub.index)
                beta = (xm_dm * (ym - ym.mean())).sum() / denom
                alpha = ym.mean() - beta * xm.mean()
                resid = np.full_like(y, np.nan)
                resid[mask] = ym - (alpha + beta * xm)
                return pd.Series(resid, index=sub.index)

            df[value_col] = df.groupby(date_col, group_keys=False).apply(
                lambda g: _residualize(g)
            )[value_col]

    return df
