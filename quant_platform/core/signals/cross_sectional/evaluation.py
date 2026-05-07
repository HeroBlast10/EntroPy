"""Factor evaluation: IC, RankIC, decay analysis, turnover, and summary statistics.

All evaluation functions expect a DataFrame with columns
``[date, ticker, <factor_col>, forward_ret]`` where ``forward_ret`` is the
next-period return used as the prediction target.
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from quant_platform.core.signals.effective import build_effective_signal


# ===================================================================
# Newey-West HAC standard error
# ===================================================================

def _andrews_lag(n: int) -> int:
    """Andrews (1991) automatic lag selection: floor(4 * (n/100)^(2/9))."""
    if n <= 1:
        return 0
    return max(1, int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0))))


def newey_west_se(series: pd.Series, lag: Optional[int] = None) -> float:
    """Newey-West HAC standard error of the sample mean.

    Daily IC time series usually exhibits positive autocorrelation, especially
    when forward returns overlap (e.g. fwd_ret_5d/10d/20d). Plain OLS
    standard errors then under-state uncertainty and t-stats become inflated.
    The Newey-West (1987) HAC estimator corrects for autocorrelation up to
    *lag* periods using Bartlett kernel weights ``w_k = 1 - k/(L+1)``.

    Parameters
    ----------
    series : Series whose mean is being tested (e.g. daily IC).
    lag : truncation lag.  Defaults to Andrews (1991) ``floor(4*(n/100)^(2/9))``.

    Returns
    -------
    HAC standard error of the sample mean (``np.nan`` if not estimable).
    """
    x = pd.to_numeric(series, errors="coerce").dropna().to_numpy(dtype=float)
    n = len(x)
    if n < 3:
        return float("nan")
    if lag is None:
        lag = _andrews_lag(n)
    lag = max(0, int(lag))

    x_dm = x - x.mean()
    s = float((x_dm ** 2).sum()) / n  # gamma_0
    for k in range(1, lag + 1):
        if k >= n:
            break
        cov_k = float((x_dm[k:] * x_dm[:-k]).sum()) / n
        weight = 1.0 - k / (lag + 1.0)
        s += 2.0 * weight * cov_k

    if s <= 0:
        return float("nan")
    return float(np.sqrt(s / n))


def newey_west_tstat(series: pd.Series, lag: Optional[int] = None) -> Tuple[float, float, int]:
    """Newey-West HAC t-stat and two-sided p-value for ``mean(series) == 0``.

    Returns
    -------
    (t_stat, p_value, lag_used) with NaNs if not estimable.
    """
    s = pd.to_numeric(series, errors="coerce").dropna()
    n = len(s)
    if n < 3:
        return float("nan"), float("nan"), 0
    if lag is None:
        lag = _andrews_lag(n)
    se = newey_west_se(s, lag=lag)
    if not np.isfinite(se) or se <= 0:
        return float("nan"), float("nan"), int(lag)
    t = float(s.mean() / se)
    p = float(math.erfc(abs(t) / math.sqrt(2.0)))
    return t, p, int(lag)


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
    nw_lag: Optional[int] = None,
) -> Dict[str, float]:
    """Summary statistics for an IC time series.

    Adds Newey-West HAC adjusted t-stat and p-value alongside the i.i.d.
    versions.  When forward returns overlap (5d/10d/20d) or IC has positive
    autocorrelation, the i.i.d. t-stat is biased upward; ``nw_t_stat`` is
    the production-grade significance test.

    Returns
    -------
    Dict with keys:
        - mean_ic, std_ic, icir, hit_rate, skew, kurt, n_obs
        - t_stat, p_value: i.i.d. standard error
        - nw_t_stat, nw_p_value, nw_lag: Newey-West HAC adjusted
    """
    ic = ic_series.dropna()
    n = len(ic)
    keys = [
        "mean_ic", "std_ic", "icir",
        "t_stat", "p_value",
        "nw_t_stat", "nw_p_value", "nw_lag",
        "hit_rate", "skew", "kurt", "n_obs",
    ]
    if n == 0:
        return {k: np.nan for k in keys}

    mean = ic.mean()
    std = ic.std()
    icir = mean / std * np.sqrt(annualization_factor) if std > 0 else np.nan
    t_stat = mean / (std / np.sqrt(n)) if std > 0 else np.nan
    p_value = math.erfc(abs(t_stat) / math.sqrt(2.0)) if np.isfinite(t_stat) else np.nan
    hit_rate = (ic > 0).mean()

    nw_t, nw_p, nw_l = newey_west_tstat(ic, lag=nw_lag)

    return {
        "mean_ic": round(mean, 6),
        "std_ic": round(std, 6),
        "icir": round(icir, 4),
        "t_stat": round(t_stat, 4),
        "p_value": round(p_value, 6),
        "nw_t_stat": round(nw_t, 4) if np.isfinite(nw_t) else np.nan,
        "nw_p_value": round(nw_p, 6) if np.isfinite(nw_p) else np.nan,
        "nw_lag": int(nw_l),
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


def estimate_alpha_half_life(
    ic_decay_df: pd.DataFrame,
    *,
    horizon_col: str = "horizon",
    ic_col: str = "mean_ic",
    min_points: int = 4,
) -> Dict[str, float]:
    """Fit IC(h) = ic0 * exp(-h / tau) and return half-life in trading days.

    Used to translate "how fast does this factor's predictive power decay"
    into a concrete recommended rebalance horizon.  Half-life is the most
    interpretable single number to drive rebalance frequency choice and
    factor-decay-aware portfolio construction.

    Method
    ------
    Linear regression of log(IC(h)) on h, restricted to horizons with
    positive IC.  Slope ``b < 0`` gives tau = -1/b and half-life = tau * ln 2.
    If fewer than *min_points* horizons have positive IC, returns NaN.

    Parameters
    ----------
    ic_decay_df : output of :func:`ic_decay` or any DataFrame with columns
        ``[horizon, mean_ic]``.

    Returns
    -------
    Dict with keys ``ic0``, ``tau``, ``half_life_days``, ``r_squared``,
    ``n_points``.  All values may be NaN when the fit is undefined (flat /
    increasing / sign-flipping decay curves).

    Examples
    --------
    >>> # Synthetic decay: IC(h) = 0.05 * exp(-h/10)
    >>> df = pd.DataFrame({'horizon': range(1, 21),
    ...                    'mean_ic': 0.05 * np.exp(-np.arange(1, 21) / 10.0)})
    >>> estimate_alpha_half_life(df)['half_life_days']  # ~ 6.93
    """
    nan_result = {
        "ic0": np.nan,
        "tau": np.nan,
        "half_life_days": np.nan,
        "r_squared": np.nan,
        "n_points": 0,
    }
    if ic_decay_df is None or ic_decay_df.empty:
        return nan_result
    if horizon_col not in ic_decay_df.columns or ic_col not in ic_decay_df.columns:
        return nan_result

    h = pd.to_numeric(ic_decay_df[horizon_col], errors="coerce").to_numpy(dtype=float)
    ic = pd.to_numeric(ic_decay_df[ic_col], errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(h) & np.isfinite(ic) & (ic > 0) & (h > 0)
    if mask.sum() < min_points:
        return {**nan_result, "n_points": int(mask.sum())}

    h_use = h[mask]
    log_ic = np.log(ic[mask])
    # OLS: log_ic = a + b * h
    h_mean = h_use.mean()
    log_ic_mean = log_ic.mean()
    h_var = float(((h_use - h_mean) ** 2).sum())
    if h_var <= 0:
        return {**nan_result, "n_points": int(mask.sum())}

    b = float(((h_use - h_mean) * (log_ic - log_ic_mean)).sum() / h_var)
    a = float(log_ic_mean - b * h_mean)
    log_ic_pred = a + b * h_use
    ss_res = float(((log_ic - log_ic_pred) ** 2).sum())
    ss_tot = float(((log_ic - log_ic_mean) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    if b >= 0:
        # IC is flat or increasing in horizon — no decay to fit
        return {
            "ic0": float(np.exp(a)),
            "tau": np.inf,
            "half_life_days": np.inf,
            "r_squared": float(r2) if np.isfinite(r2) else np.nan,
            "n_points": int(mask.sum()),
        }

    tau = -1.0 / b
    half_life = tau * math.log(2.0)
    return {
        "ic0": float(np.exp(a)),
        "tau": float(tau),
        "half_life_days": float(half_life),
        "r_squared": float(r2) if np.isfinite(r2) else np.nan,
        "n_points": int(mask.sum()),
    }


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
# Long-leg / short-leg attribution
# ===================================================================

def long_short_leg_attribution(
    df: pd.DataFrame,
    factor_col: str,
    return_col: str = "fwd_ret_1d",
    n_quantiles: int = 5,
    date_col: str = "date",
    annualisation: int = 252,
) -> Dict[str, float]:
    """Decompose long-short Sharpe into long-leg and short-leg contributions.

    Many factors derive most of their alpha from one side only (e.g. ASSET_GROWTH
    short, BAB long).  In long-only mandates, factors whose alpha is dominated
    by the short-leg are essentially unusable.  This metric is critical for
    deciding whether a factor that "looks great LS" can actually be deployed
    in a long-only product.

    Method
    ------
    Define market = cross-sectional mean return on each date.  Then::

        long_alpha[t]  = mean_return(top_quintile)    - market[t]
        short_alpha[t] = market[t]                    - mean_return(bottom_quintile)
        ls_return[t]   = long_alpha[t] + short_alpha[t]

    Sharpe of each leg is annualised separately.  ``long_share_of_ls`` is
    the fraction of total LS mean return contributed by the long leg.

    Returns
    -------
    Dict with:
        - long_leg_mean / short_leg_mean (daily)
        - long_leg_sharpe / short_leg_sharpe (annualised)
        - long_share_of_ls / short_share_of_ls (fractions, sum to 1)
        - long_only_compatible: bool — True iff long_leg has positive Sharpe
          AND contributes >= 30% of LS mean
    """
    nan_result = {
        "long_leg_mean": np.nan,
        "short_leg_mean": np.nan,
        "long_leg_sharpe": np.nan,
        "short_leg_sharpe": np.nan,
        "long_share_of_ls": np.nan,
        "short_share_of_ls": np.nan,
        "long_only_compatible": False,
    }
    if return_col not in df.columns or factor_col not in df.columns:
        return nan_result

    qr = quantile_returns(df, factor_col, return_col, n_quantiles, date_col)
    if qr.empty:
        return nan_result

    market = (
        df[[date_col, return_col]]
        .dropna(subset=[return_col])
        .groupby(date_col)[return_col]
        .mean()
    )
    top = qr[qr["quantile"] == n_quantiles].set_index(date_col)["mean_ret"]
    bot = qr[qr["quantile"] == 1].set_index(date_col)["mean_ret"]
    common = market.index.intersection(top.index).intersection(bot.index)
    if len(common) < 5:
        return nan_result

    market = market.loc[common]
    top = top.loc[common]
    bot = bot.loc[common]
    long_alpha = (top - market).dropna()
    short_alpha = (market - bot).dropna()

    def _sharpe(s: pd.Series) -> float:
        if len(s) < 2 or s.std(ddof=1) == 0:
            return np.nan
        return float(s.mean() / s.std(ddof=1) * np.sqrt(annualisation))

    long_mean = float(long_alpha.mean()) if len(long_alpha) else np.nan
    short_mean = float(short_alpha.mean()) if len(short_alpha) else np.nan
    ls_mean = (long_mean if pd.notna(long_mean) else 0.0) + (short_mean if pd.notna(short_mean) else 0.0)

    if pd.notna(long_mean) and pd.notna(short_mean) and ls_mean != 0:
        long_share = long_mean / ls_mean
        short_share = short_mean / ls_mean
    else:
        long_share = np.nan
        short_share = np.nan

    long_sharpe = _sharpe(long_alpha)
    long_only_ok = bool(
        pd.notna(long_sharpe) and long_sharpe > 0
        and pd.notna(long_share) and long_share >= 0.30
    )

    return {
        "long_leg_mean": round(long_mean, 6) if pd.notna(long_mean) else np.nan,
        "short_leg_mean": round(short_mean, 6) if pd.notna(short_mean) else np.nan,
        "long_leg_sharpe": round(long_sharpe, 4) if pd.notna(long_sharpe) else np.nan,
        "short_leg_sharpe": round(_sharpe(short_alpha), 4) if pd.notna(_sharpe(short_alpha)) else np.nan,
        "long_share_of_ls": round(long_share, 4) if pd.notna(long_share) else np.nan,
        "short_share_of_ls": round(short_share, 4) if pd.notna(short_share) else np.nan,
        "long_only_compatible": long_only_ok,
    }


# ===================================================================
# Cross-sectional stability across universe slices
# ===================================================================

def cross_sectional_stability(
    df: pd.DataFrame,
    factor_col: str,
    *,
    return_col: str = "fwd_ret_1d",
    segment_cols: Optional[Iterable[str]] = None,
    n_buckets: int = 4,
    date_col: str = "date",
) -> pd.DataFrame:
    """RankIC by universe slice: tests if factor works across size/vol/liquidity.

    Many factors look strong only in small caps, high-vol, or illiquid names —
    and become noise (or get fully eaten by costs) on the tradable universe.
    This function bucketises each named segment column on every date and
    reports RankIC per bucket, so production-readiness decisions account
    for *where* the alpha lives.

    Parameters
    ----------
    df : DataFrame containing factor_col, return_col, date_col, and any
        columns listed in *segment_cols*.
    segment_cols : columns used to bucketise (e.g. ``["market_cap", "amount"]``).
        Defaults to whichever of ``["market_cap", "amount", "volume"]`` are
        present — these are the most common production slices.
    n_buckets : number of equal-frequency buckets per segment (default 4).

    Returns
    -------
    DataFrame with columns ``[segment, bucket, rank_ic_mean, rank_ic_t,
    rank_ic_nw_t, n_dates]``.  Empty DataFrame if no segment column matches.
    """
    if segment_cols is None:
        segment_cols = [c for c in ("market_cap", "amount", "volume") if c in df.columns]
    else:
        segment_cols = [c for c in segment_cols if c in df.columns]

    if not segment_cols or factor_col not in df.columns or return_col not in df.columns:
        return pd.DataFrame(columns=["segment", "bucket", "rank_ic_mean", "rank_ic_t", "rank_ic_nw_t", "n_dates"])

    rows: List[Dict[str, object]] = []
    work_cols = [date_col, "ticker", factor_col, return_col]
    for seg in segment_cols:
        if seg not in df.columns:
            continue
        sub = df[work_cols + [seg]].dropna(subset=[seg])
        if sub.empty:
            continue
        # Bucketise within each date so buckets are comparable cross-sectionally
        def _assign_bucket(g: pd.DataFrame) -> pd.Series:
            try:
                return pd.qcut(g[seg], n_buckets, labels=False, duplicates="drop")
            except ValueError:
                return pd.Series(np.nan, index=g.index)

        sub = sub.copy()
        sub["_bucket"] = sub.groupby(date_col, group_keys=False).apply(_assign_bucket)
        sub = sub.dropna(subset=["_bucket"])
        if sub.empty:
            continue
        sub["_bucket"] = sub["_bucket"].astype(int)

        for bucket_id in sorted(sub["_bucket"].unique()):
            bucket_df = sub[sub["_bucket"] == bucket_id]
            ic = compute_rank_ic_series(bucket_df, factor_col, return_col, date_col)
            ic = ic.dropna()
            if ic.empty:
                continue
            n = len(ic)
            mean = float(ic.mean())
            std = float(ic.std(ddof=1)) if n > 1 else 0.0
            t = mean / (std / math.sqrt(n)) if std > 0 else np.nan
            nw_t, _, _ = newey_west_tstat(ic)
            rows.append({
                "segment": seg,
                "bucket": int(bucket_id),
                "rank_ic_mean": round(mean, 6),
                "rank_ic_t": round(t, 4) if np.isfinite(t) else np.nan,
                "rank_ic_nw_t": round(nw_t, 4) if np.isfinite(nw_t) else np.nan,
                "n_dates": int(n),
            })
    return pd.DataFrame(rows)


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
# Production-readiness metrics
# ===================================================================

def advanced_factor_metrics(
    df: pd.DataFrame,
    factor_col: str,
    periods: Optional[List[int]] = None,
    n_quantiles: int = 5,
    date_col: str = "date",
    ticker_col: str = "ticker",
    cost_bps_per_turnover: float = 10.0,
) -> Dict[str, float]:
    """Compute multi-horizon and stability metrics for one effective signal."""
    periods = periods or [1, 5, 10, 20]
    metrics: Dict[str, float] = {}

    horizon_signs = []
    primary_turnover = factor_turnover(df, factor_col, date_col, ticker_col)
    mean_turnover = float(primary_turnover.mean()) if len(primary_turnover) else np.nan
    metrics["mean_turnover"] = mean_turnover

    for p in periods:
        ret_col = f"fwd_ret_{p}d"
        if ret_col not in df.columns:
            continue

        ic = compute_ic_series(df, factor_col, ret_col, date_col, method="pearson")
        ric = compute_rank_ic_series(df, factor_col, ret_col, date_col)
        ic_stats = ic_summary(ic)
        ric_stats = ic_summary(ric)
        metrics[f"ic_mean_{p}d"] = ic_stats["mean_ic"]
        metrics[f"ic_tstat_{p}d"] = ic_stats["t_stat"]
        metrics[f"ic_nw_tstat_{p}d"] = ic_stats["nw_t_stat"]
        metrics[f"ric_mean_{p}d"] = ric_stats["mean_ic"]
        metrics[f"ric_icir_{p}d"] = ric_stats["icir"]
        metrics[f"ric_p_value_{p}d"] = ric_stats["p_value"]
        metrics[f"ric_nw_tstat_{p}d"] = ric_stats["nw_t_stat"]
        metrics[f"ric_nw_p_value_{p}d"] = ric_stats["nw_p_value"]

        qr = quantile_returns(df, factor_col, ret_col, n_quantiles, date_col)
        metrics[f"monotonicity_{p}d"] = _monotonicity_score(qr, n_quantiles)
        ls = long_short_returns(df, factor_col, ret_col, n_quantiles, date_col)
        if len(ls) > 1 and ls.std() > 0:
            metrics[f"ls_sharpe_{p}d"] = float(ls.mean() / ls.std() * np.sqrt(252 / p))
        else:
            metrics[f"ls_sharpe_{p}d"] = np.nan

        if pd.notna(ric_stats["mean_ic"]):
            horizon_signs.append(np.sign(ric_stats["mean_ic"]))

    if "fwd_ret_1d" in df.columns:
        ls_1d = long_short_returns(df, factor_col, "fwd_ret_1d", n_quantiles, date_col)
        if len(ls_1d) > 1 and ls_1d.std() > 0:
            cost_daily = (mean_turnover if pd.notna(mean_turnover) else 0.0) * cost_bps_per_turnover / 10_000.0
            metrics["cost_adj_ls_mean"] = float(ls_1d.mean() - cost_daily)
            metrics["cost_adj_ls_sharpe"] = float((ls_1d.mean() - cost_daily) / ls_1d.std() * np.sqrt(252))
            metrics["break_even_cost_bps"] = float(
                ls_1d.mean() / max(mean_turnover, 1e-12) * 10_000
            ) if pd.notna(mean_turnover) else np.nan
        else:
            metrics["cost_adj_ls_mean"] = np.nan
            metrics["cost_adj_ls_sharpe"] = np.nan
            metrics["break_even_cost_bps"] = np.nan

    metrics.update(_capacity_metrics(df, factor_col, n_quantiles, date_col))
    metrics.update(_market_regime_stability(df, factor_col, date_col))
    metrics.update(_subperiod_stability(df, factor_col, date_col))
    metrics.update(_rolling_oos_ic_stability(df, factor_col, date_col))

    # Long/short leg attribution on 1-day forward return
    if "fwd_ret_1d" in df.columns:
        leg = long_short_leg_attribution(df, factor_col, "fwd_ret_1d", n_quantiles, date_col)
        for k, v in leg.items():
            metrics[k] = v

    # Alpha decay half-life (fit IC(h) = ic0 * exp(-h / tau))
    try:
        decay_horizon = max(periods) if periods else 20
        if "adj_close" in df.columns:
            decay = ic_decay(df, factor_col, max_lag=decay_horizon,
                             date_col=date_col, ticker_col=ticker_col, price_col="adj_close")
            hl = estimate_alpha_half_life(decay)
            metrics["alpha_half_life_days"] = hl["half_life_days"] if np.isfinite(hl["half_life_days"]) else np.nan
            metrics["alpha_decay_r2"] = hl["r_squared"]
    except Exception as exc:
        logger.debug("alpha_half_life estimation failed for {}: {}", factor_col, exc)
        metrics["alpha_half_life_days"] = np.nan
        metrics["alpha_decay_r2"] = np.nan

    valid_horizon_signs = [s for s in horizon_signs if s != 0 and np.isfinite(s)]
    if valid_horizon_signs:
        positive = sum(s > 0 for s in valid_horizon_signs)
        negative = sum(s < 0 for s in valid_horizon_signs)
        metrics["horizon_sign_consistency"] = max(positive, negative) / len(valid_horizon_signs)
    else:
        metrics["horizon_sign_consistency"] = np.nan

    return metrics


def _monotonicity_score(qr: pd.DataFrame, n_quantiles: int) -> float:
    """Return Spearman-like monotonicity of average quantile returns."""
    if qr.empty or "quantile" not in qr.columns:
        return np.nan
    avg = qr.groupby("quantile")["mean_ret"].mean().reindex(range(1, n_quantiles + 1))
    avg = avg.dropna()
    if len(avg) < 3:
        return np.nan
    return float(pd.Series(avg.index, index=avg.index).corr(avg, method="spearman"))


def _capacity_metrics(
    df: pd.DataFrame,
    factor_col: str,
    n_quantiles: int,
    date_col: str,
) -> Dict[str, float]:
    """Estimate capacity from selected names' dollar volume when available."""
    price_col = "adj_close" if "adj_close" in df.columns else "close" if "close" in df.columns else None
    if "amount" in df.columns:
        dollar_volume = df["amount"]
    elif price_col is not None and "volume" in df.columns:
        dollar_volume = df[price_col] * df["volume"]
    else:
        return {"capacity_10pct_adv": np.nan, "median_selected_adv": np.nan}

    tmp = df[[date_col, factor_col]].copy()
    tmp["_dollar_volume"] = dollar_volume
    tmp = tmp.dropna(subset=[factor_col, "_dollar_volume"])
    if tmp.empty:
        return {"capacity_10pct_adv": np.nan, "median_selected_adv": np.nan}

    daily_capacity = []
    daily_median_adv = []
    for _, group in tmp.groupby(date_col):
        if len(group) < n_quantiles:
            continue
        cutoff = group[factor_col].quantile(1.0 - 1.0 / n_quantiles)
        selected = group[group[factor_col] >= cutoff]
        daily_capacity.append(selected["_dollar_volume"].clip(lower=0).sum() * 0.10)
        daily_median_adv.append(selected["_dollar_volume"].median())

    return {
        "capacity_10pct_adv": float(np.nanmedian(daily_capacity)) if daily_capacity else np.nan,
        "median_selected_adv": float(np.nanmedian(daily_median_adv)) if daily_median_adv else np.nan,
    }


def _market_regime_stability(
    df: pd.DataFrame,
    factor_col: str,
    date_col: str,
) -> Dict[str, float]:
    """Check whether IC sign survives up/down market regimes."""
    if "fwd_ret_1d" not in df.columns:
        return {"regime_sign_consistency": np.nan, "rank_ic_up_market": np.nan, "rank_ic_down_market": np.nan}

    market_ret = df.groupby(date_col)["fwd_ret_1d"].mean()
    ric = compute_rank_ic_series(df, factor_col, "fwd_ret_1d", date_col)
    common = ric.index.intersection(market_ret.index)
    if len(common) < 5:
        return {"regime_sign_consistency": np.nan, "rank_ic_up_market": np.nan, "rank_ic_down_market": np.nan}

    up = ric.loc[common][market_ret.loc[common] >= market_ret.loc[common].median()]
    down = ric.loc[common][market_ret.loc[common] < market_ret.loc[common].median()]
    up_mean = up.mean() if len(up) else np.nan
    down_mean = down.mean() if len(down) else np.nan
    signs = [np.sign(x) for x in (up_mean, down_mean) if pd.notna(x) and x != 0]
    consistency = max(sum(s > 0 for s in signs), sum(s < 0 for s in signs)) / len(signs) if signs else np.nan
    return {
        "regime_sign_consistency": float(consistency) if pd.notna(consistency) else np.nan,
        "rank_ic_up_market": float(up_mean) if pd.notna(up_mean) else np.nan,
        "rank_ic_down_market": float(down_mean) if pd.notna(down_mean) else np.nan,
    }


def _subperiod_stability(
    df: pd.DataFrame,
    factor_col: str,
    date_col: str,
    n_splits: int = 3,
) -> Dict[str, float]:
    """Compute IC sign consistency across chronological subperiods."""
    if "fwd_ret_1d" not in df.columns:
        return {"subperiod_sign_consistency": np.nan, "subperiod_min_rank_ic": np.nan}
    dates = pd.Series(sorted(pd.to_datetime(df[date_col].unique())))
    if len(dates) < n_splits:
        return {"subperiod_sign_consistency": np.nan, "subperiod_min_rank_ic": np.nan}

    means = []
    for chunk in np.array_split(dates, n_splits):
        sub = df[df[date_col].isin(chunk)]
        ric = compute_rank_ic_series(sub, factor_col, "fwd_ret_1d", date_col)
        means.append(ric.mean())

    signs = [np.sign(x) for x in means if pd.notna(x) and x != 0]
    consistency = max(sum(s > 0 for s in signs), sum(s < 0 for s in signs)) / len(signs) if signs else np.nan
    return {
        "subperiod_sign_consistency": float(consistency) if pd.notna(consistency) else np.nan,
        "subperiod_min_rank_ic": float(np.nanmin(means)) if any(pd.notna(x) for x in means) else np.nan,
    }


def _rolling_oos_ic_stability(
    df: pd.DataFrame,
    factor_col: str,
    date_col: str,
    train_months: int = 36,
    test_months: int = 12,
    step_months: int = 12,
) -> Dict[str, float]:
    """Rolling out-of-sample IC stability using train/test date splits."""
    if "fwd_ret_1d" not in df.columns:
        return {"oos_rank_ic_mean": np.nan, "oos_rank_ic_sign_consistency": np.nan}
    all_dates = pd.DatetimeIndex(sorted(pd.to_datetime(df[date_col].unique())))
    if all_dates.empty:
        return {"oos_rank_ic_mean": np.nan, "oos_rank_ic_sign_consistency": np.nan}

    cursor = all_dates.min() + pd.DateOffset(months=train_months)
    means = []
    while cursor + pd.DateOffset(months=test_months) <= all_dates.max() + pd.Timedelta(days=1):
        test_end = cursor + pd.DateOffset(months=test_months) - pd.Timedelta(days=1)
        test_dates = all_dates[(all_dates >= cursor) & (all_dates <= test_end)]
        if len(test_dates) > 0:
            sub = df[df[date_col].isin(test_dates)]
            means.append(compute_rank_ic_series(sub, factor_col, "fwd_ret_1d", date_col).mean())
        cursor += pd.DateOffset(months=step_months)

    signs = [np.sign(x) for x in means if pd.notna(x) and x != 0]
    consistency = max(sum(s > 0 for s in signs), sum(s < 0 for s in signs)) / len(signs) if signs else np.nan
    return {
        "oos_rank_ic_mean": float(np.nanmean(means)) if any(pd.notna(x) for x in means) else np.nan,
        "oos_rank_ic_sign_consistency": float(consistency) if pd.notna(consistency) else np.nan,
    }


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
    direction: int = 1,
    neutralize_by: Optional[List[str]] = None,
    forward_periods: Optional[List[int]] = None,
    cost_bps_per_turnover: float = 10.0,
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
    eff_df = build_effective_signal(
        df,
        factor_col,
        direction=direction,
        neutralize_by=neutralize_by,
        rank=True,
    )

    ic = compute_ic_series(eff_df, factor_col, return_col, date_col, method="pearson")
    ric = compute_rank_ic_series(eff_df, factor_col, return_col, date_col)
    qr = quantile_returns(eff_df, factor_col, return_col, n_quantiles, date_col)
    ls = long_short_returns(eff_df, factor_col, return_col, n_quantiles, date_col)
    turnover = factor_turnover(eff_df, factor_col, date_col, ticker_col)
    advanced = advanced_factor_metrics(
        eff_df,
        factor_col,
        periods=forward_periods,
        n_quantiles=n_quantiles,
        date_col=date_col,
        ticker_col=ticker_col,
        cost_bps_per_turnover=cost_bps_per_turnover,
    )

    # Cross-sectional stability: needs the original df with size/liquidity columns
    cs_stability = cross_sectional_stability(
        df.merge(
            eff_df[[date_col, ticker_col, factor_col]].rename(columns={factor_col: "_eff_signal_"}),
            on=[date_col, ticker_col],
            how="inner",
        ).drop(columns=[factor_col], errors="ignore").rename(columns={"_eff_signal_": factor_col}),
        factor_col=factor_col,
        return_col=return_col,
        date_col=date_col,
    )

    result = {
        "ic_series": ic,
        "rank_ic_series": ric,
        "ic_stats": ic_summary(ic),
        "rank_ic_stats": ic_summary(ric),
        "quantile_returns": qr,
        "long_short": ls,
        "turnover": turnover,
        "advanced_metrics": advanced,
        "cross_sectional_stability": cs_stability,
        "effective_df": eff_df[[date_col, ticker_col, factor_col]],
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
        row.update(ts.get("advanced_metrics", {}))
        row["mean_turnover"] = ts["turnover"].mean() if len(ts["turnover"]) else np.nan
        ls = ts["long_short"]
        if len(ls) > 0:
            row["ls_mean"] = ls.mean()
            row["ls_sharpe"] = ls.mean() / ls.std() * np.sqrt(252) if ls.std() > 0 else np.nan
        rows.append(row)
    return pd.DataFrame(rows).set_index("factor")
