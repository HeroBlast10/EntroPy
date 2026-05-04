"""Factor evaluation: IC, RankIC, decay analysis, turnover, and summary statistics.

All evaluation functions expect a DataFrame with columns
``[date, ticker, <factor_col>, forward_ret]`` where ``forward_ret`` is the
next-period return used as the prediction target.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from quant_platform.core.signals.effective import build_effective_signal


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
                ["mean_ic", "std_ic", "icir", "t_stat", "p_value", "hit_rate", "skew", "kurt", "n_obs"]}

    mean = ic.mean()
    std = ic.std()
    icir = mean / std * np.sqrt(annualization_factor) if std > 0 else np.nan
    t_stat = mean / (std / np.sqrt(n)) if std > 0 else np.nan
    p_value = math.erfc(abs(t_stat) / math.sqrt(2.0)) if np.isfinite(t_stat) else np.nan
    hit_rate = (ic > 0).mean()

    return {
        "mean_ic": round(mean, 6),
        "std_ic": round(std, 6),
        "icir": round(icir, 4),
        "t_stat": round(t_stat, 4),
        "p_value": round(p_value, 6),
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
        metrics[f"ric_mean_{p}d"] = ric_stats["mean_ic"]
        metrics[f"ric_icir_{p}d"] = ric_stats["icir"]
        metrics[f"ric_p_value_{p}d"] = ric_stats["p_value"]

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

    result = {
        "ic_series": ic,
        "rank_ic_series": ric,
        "ic_stats": ic_summary(ic),
        "rank_ic_stats": ic_summary(ric),
        "quantile_returns": qr,
        "long_short": ls,
        "turnover": turnover,
        "advanced_metrics": advanced,
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
