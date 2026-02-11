"""Factor evaluation: IC, RankIC, decay analysis, turnover, and summary statistics.

All evaluation functions expect a DataFrame with columns
``[date, ticker, <factor_col>, forward_ret]`` where ``forward_ret`` is the
next-period return used as the prediction target.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger


# ===================================================================
# Forward return computation
# ===================================================================

def add_forward_returns(
    prices: pd.DataFrame,
    periods: List[int] = [1, 5, 10, 20],
    price_col: str = "adj_close",
    date_col: str = "date",
    ticker_col: str = "ticker",
) -> pd.DataFrame:
    """Compute forward returns for multiple holding periods.

    Adds columns ``fwd_ret_1d``, ``fwd_ret_5d``, etc.
    Returns are simple (not log) percentage returns.
    """
    df = prices.copy()
    df.sort_values([ticker_col, date_col], inplace=True)

    for p in periods:
        col_name = f"fwd_ret_{p}d"
        df[col_name] = (
            df.groupby(ticker_col)[price_col]
            .pct_change(periods=p)
            .shift(-p)  # align to current date (look forward)
        )
    return df


# ===================================================================
# Information Coefficient (IC)
# ===================================================================

def _ic_single_date(
    factor_vals: pd.Series,
    return_vals: pd.Series,
    method: str = "pearson",
) -> float:
    """Compute IC for a single cross-section."""
    mask = factor_vals.notna() & return_vals.notna()
    if mask.sum() < 5:
        return np.nan
    if method == "pearson":
        return factor_vals[mask].corr(return_vals[mask])
    elif method == "spearman":
        return factor_vals[mask].corr(return_vals[mask], method="spearman")
    return np.nan


def compute_ic_series(
    df: pd.DataFrame,
    factor_col: str,
    return_col: str = "fwd_ret_1d",
    date_col: str = "date",
    method: str = "pearson",
) -> pd.Series:
    """Compute daily IC (cross-sectional correlation) between factor and forward returns.

    Returns a Series indexed by date.
    """
    ic = df.groupby(date_col).apply(
        lambda g: _ic_single_date(g[factor_col], g[return_col], method),
        include_groups=False,
    )
    ic.name = f"IC_{method}"
    return ic


def compute_rank_ic_series(
    df: pd.DataFrame,
    factor_col: str,
    return_col: str = "fwd_ret_1d",
    date_col: str = "date",
) -> pd.Series:
    """Compute daily Rank IC (Spearman correlation)."""
    return compute_ic_series(df, factor_col, return_col, date_col, method="spearman")


# ===================================================================
# IC summary statistics
# ===================================================================

def ic_summary(
    ic_series: pd.Series,
    annualization_factor: int = 252,
) -> Dict[str, float]:
    """Summary statistics for an IC time series.

    Returns
    -------
    Dict with keys: mean_ic, std_ic, icir, t_stat, hit_rate, skew, kurt.
    """
    ic = ic_series.dropna()
    n = len(ic)
    if n == 0:
        return {k: np.nan for k in
                ["mean_ic", "std_ic", "icir", "t_stat", "hit_rate", "skew", "kurt", "n_obs"]}

    mean = ic.mean()
    std = ic.std()
    icir = mean / std * np.sqrt(annualization_factor) if std > 0 else np.nan
    t_stat = mean / (std / np.sqrt(n)) if std > 0 else np.nan
    hit_rate = (ic > 0).mean()

    return {
        "mean_ic": round(mean, 6),
        "std_ic": round(std, 6),
        "icir": round(icir, 4),
        "t_stat": round(t_stat, 4),
        "hit_rate": round(hit_rate, 4),
        "skew": round(ic.skew(), 4),
        "kurt": round(ic.kurt(), 4),
        "n_obs": n,
    }


# ===================================================================
# IC decay
# ===================================================================

def ic_decay(
    df: pd.DataFrame,
    factor_col: str,
    max_lag: int = 20,
    date_col: str = "date",
    ticker_col: str = "ticker",
    price_col: str = "adj_close",
) -> pd.DataFrame:
    """Compute IC at multiple forward horizons to measure signal persistence.

    Returns a DataFrame with columns ``[horizon, mean_ic, icir]``.
    """
    prices_sub = df[[date_col, ticker_col, price_col, factor_col]].copy()
    prices_sub.sort_values([ticker_col, date_col], inplace=True)

    results = []
    for h in range(1, max_lag + 1):
        fwd_col = f"_fwd_{h}"
        prices_sub[fwd_col] = (
            prices_sub.groupby(ticker_col)[price_col]
            .pct_change(periods=h)
            .shift(-h)
        )
        ic = compute_ic_series(prices_sub, factor_col, fwd_col, date_col, method="spearman")
        stats = ic_summary(ic)
        results.append({"horizon": h, "mean_ic": stats["mean_ic"], "icir": stats["icir"]})
        prices_sub.drop(columns=[fwd_col], inplace=True)

    return pd.DataFrame(results)


# ===================================================================
# Quantile returns (long-short spread)
# ===================================================================

def quantile_returns(
    df: pd.DataFrame,
    factor_col: str,
    return_col: str = "fwd_ret_1d",
    n_quantiles: int = 5,
    date_col: str = "date",
) -> pd.DataFrame:
    """Compute mean forward return per factor quantile per date.

    Returns a DataFrame with columns ``[date, quantile, mean_ret]``.
    """
    tmp = df[[date_col, factor_col, return_col]].dropna().copy()

    def _assign_q(g: pd.DataFrame) -> pd.Series:
        try:
            return pd.qcut(g[factor_col], n_quantiles, labels=False, duplicates="drop") + 1
        except ValueError:
            return pd.Series(np.nan, index=g.index)

    tmp["quantile"] = tmp.groupby(date_col, group_keys=False).apply(_assign_q)
    tmp = tmp.dropna(subset=["quantile"])
    tmp["quantile"] = tmp["quantile"].astype(int)

    qr = tmp.groupby([date_col, "quantile"])[return_col].mean().reset_index()
    qr.columns = [date_col, "quantile", "mean_ret"]
    return qr


def long_short_returns(
    df: pd.DataFrame,
    factor_col: str,
    return_col: str = "fwd_ret_1d",
    n_quantiles: int = 5,
    date_col: str = "date",
) -> pd.Series:
    """Daily long-short return: Q5 (top) minus Q1 (bottom)."""
    qr = quantile_returns(df, factor_col, return_col, n_quantiles, date_col)
    top = qr[qr["quantile"] == n_quantiles].set_index(date_col)["mean_ret"]
    bottom = qr[qr["quantile"] == 1].set_index(date_col)["mean_ret"]
    ls = top - bottom
    ls.name = "long_short_ret"
    return ls


# ===================================================================
# Factor turnover
# ===================================================================

def factor_turnover(
    df: pd.DataFrame,
    factor_col: str,
    date_col: str = "date",
    ticker_col: str = "ticker",
) -> pd.Series:
    """Cross-sectional rank correlation of factor between consecutive dates.

    Turnover = 1 − rank_corr(t, t−1).  Higher = more turnover = higher
    implementation cost.
    """
    dates = sorted(df[date_col].unique())
    turnover = {}

    prev_ranks: Optional[pd.Series] = None
    for d in dates:
        cross = df.loc[df[date_col] == d].set_index(ticker_col)[factor_col].rank()
        if prev_ranks is not None:
            common = cross.index.intersection(prev_ranks.index)
            if len(common) >= 5:
                corr = cross.loc[common].corr(prev_ranks.loc[common], method="spearman")
                turnover[d] = 1.0 - corr
        prev_ranks = cross

    ts = pd.Series(turnover, name="turnover")
    ts.index.name = date_col
    return ts


# ===================================================================
# Factor tear sheet (all-in-one)
# ===================================================================

def factor_tearsheet(
    df: pd.DataFrame,
    factor_col: str,
    return_col: str = "fwd_ret_1d",
    date_col: str = "date",
    ticker_col: str = "ticker",
    n_quantiles: int = 5,
) -> Dict[str, object]:
    """Produce a comprehensive evaluation bundle for one factor.

    Returns
    -------
    Dict with keys:
        - ``ic_series`` — daily Pearson IC
        - ``rank_ic_series`` — daily Spearman RankIC
        - ``ic_stats`` — summary dict (mean, ICIR, t-stat, hit rate …)
        - ``rank_ic_stats`` — same for RankIC
        - ``quantile_returns`` — DataFrame of mean return per quantile
        - ``long_short`` — daily long−short return Series
        - ``turnover`` — daily factor turnover Series
    """
    ic = compute_ic_series(df, factor_col, return_col, date_col, method="pearson")
    ric = compute_rank_ic_series(df, factor_col, return_col, date_col)

    result = {
        "ic_series": ic,
        "rank_ic_series": ric,
        "ic_stats": ic_summary(ic),
        "rank_ic_stats": ic_summary(ric),
        "quantile_returns": quantile_returns(df, factor_col, return_col, n_quantiles, date_col),
        "long_short": long_short_returns(df, factor_col, return_col, n_quantiles, date_col),
        "turnover": factor_turnover(df, factor_col, date_col, ticker_col),
    }

    # Log headline numbers
    s = result["rank_ic_stats"]
    logger.info(
        "Factor {}: RankIC={:.4f}  ICIR={:.2f}  t={:.2f}  hit={:.1%}",
        factor_col, s["mean_ic"], s["icir"], s["t_stat"], s["hit_rate"],
    )
    return result


def compare_factors(
    tearsheets: Dict[str, Dict],
) -> pd.DataFrame:
    """Build a comparison table from multiple tearsheet results.

    Parameters
    ----------
    tearsheets : ``{factor_name: tearsheet_dict}`` as returned by
        :func:`factor_tearsheet`.

    Returns
    -------
    DataFrame with one row per factor and columns for key metrics.
    """
    rows = []
    for name, ts in tearsheets.items():
        row = {"factor": name}
        row.update({f"ic_{k}": v for k, v in ts["ic_stats"].items()})
        row.update({f"ric_{k}": v for k, v in ts["rank_ic_stats"].items()})
        row["mean_turnover"] = ts["turnover"].mean() if len(ts["turnover"]) else np.nan
        ls = ts["long_short"]
        if len(ls) > 0:
            row["ls_mean"] = ls.mean()
            row["ls_sharpe"] = ls.mean() / ls.std() * np.sqrt(252) if ls.std() > 0 else np.nan
        rows.append(row)
    return pd.DataFrame(rows).set_index("factor")
