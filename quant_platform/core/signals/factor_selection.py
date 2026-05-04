"""Multi-metric factor screening and multiple-testing controls."""

from __future__ import annotations

from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd


def benjamini_hochberg(pvalues: pd.Series) -> pd.Series:
    """Return Benjamini-Hochberg q-values aligned to *pvalues*."""
    p = pd.to_numeric(pvalues, errors="coerce")
    valid = p.dropna().clip(lower=0.0, upper=1.0)
    q = pd.Series(np.nan, index=p.index, dtype=float)
    if valid.empty:
        return q

    ranked = valid.sort_values()
    m = float(len(ranked))
    raw_q = ranked * m / np.arange(1, len(ranked) + 1)
    monotone_q = raw_q.iloc[::-1].cummin().iloc[::-1].clip(upper=1.0)
    q.loc[monotone_q.index] = monotone_q
    return q


def white_reality_check(
    long_short_returns: Dict[str, pd.Series],
    *,
    n_boot: int = 500,
    random_state: int = 42,
) -> pd.Series:
    """Bootstrap p-value for each factor against the best strategy benchmark.

    This is a lightweight White Reality Check approximation using date-level
    bootstrap resampling of demeaned long-short returns.
    """
    cleaned = {
        name: ret.dropna().astype(float)
        for name, ret in long_short_returns.items()
        if ret is not None and len(ret.dropna()) > 5
    }
    result = pd.Series(np.nan, index=list(long_short_returns.keys()), dtype=float)
    if not cleaned:
        return result

    panel = pd.concat(cleaned, axis=1).dropna(how="all").fillna(0.0)
    if panel.empty:
        return result

    observed = panel.mean()
    demeaned = panel - observed
    rng = np.random.default_rng(random_state)
    max_boot = np.empty(n_boot)
    values = demeaned.to_numpy()
    n = len(panel)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        max_boot[i] = values[idx].mean(axis=0).max()

    for name in observed.index:
        result.loc[name] = float((max_boot >= observed.loc[name]).mean())
    return result


def apply_multiple_testing_controls(
    comparison: pd.DataFrame,
    tearsheets: Optional[Dict[str, Dict]] = None,
    *,
    alpha: float = 0.10,
) -> pd.DataFrame:
    """Add FDR, Bonferroni, deflated Sharpe, and WRC-style columns."""
    if comparison is None or comparison.empty:
        return comparison

    result = comparison.copy()
    p_col = "ric_p_value"
    if p_col not in result.columns and "ric_t_stat" in result.columns:
        result[p_col] = _normal_two_sided_pvalue(result["ric_t_stat"])

    if p_col in result.columns:
        result["fdr_q_value"] = benjamini_hochberg(result[p_col])
        result["fdr_pass_10pct"] = result["fdr_q_value"] <= alpha
        m = max(result[p_col].notna().sum(), 1)
        result["bonferroni_p_value"] = (result[p_col] * m).clip(upper=1.0)
        result["bonferroni_pass_5pct"] = result["bonferroni_p_value"] <= 0.05

    if "ls_sharpe" in result.columns:
        n_tests = max(len(result), 1)
        expected_noise_max = np.sqrt(2.0 * np.log(n_tests)) / np.sqrt(252.0)
        result["deflated_ls_sharpe"] = result["ls_sharpe"] - expected_noise_max

    if tearsheets:
        ls_rets = {
            name: ts.get("long_short")
            for name, ts in tearsheets.items()
            if isinstance(ts, dict) and "long_short" in ts
        }
        if ls_rets:
            result["white_reality_pvalue"] = white_reality_check(ls_rets).reindex(result.index)
            result["white_reality_pass_10pct"] = result["white_reality_pvalue"] <= alpha

    return result


def apply_deployability_filters(
    comparison: pd.DataFrame,
    *,
    min_rank_ic: float = 0.0,
    min_cost_adj_sharpe: float = 0.0,
    max_turnover: float = 2.0,
    min_capacity: float = 0.0,
    min_subperiod_consistency: float = 0.67,
    min_horizon_consistency: float = 0.75,
) -> pd.DataFrame:
    """Add hard production-readiness filters to a comparison table."""
    if comparison is None or comparison.empty:
        return comparison

    result = comparison.copy()

    checks = pd.DataFrame(index=result.index)
    checks["rank_ic_positive"] = _col(result, "ric_mean_ic", np.nan) > min_rank_ic
    checks["oos_ic_positive"] = _col(result, "oos_rank_ic_mean", 1.0) > 0.0
    checks["cost_adj_sharpe_positive"] = _col(result, "cost_adj_ls_sharpe", np.nan) > min_cost_adj_sharpe
    checks["turnover_ok"] = _col(result, "mean_turnover", np.inf) <= max_turnover
    checks["capacity_ok"] = _col(result, "capacity_10pct_adv", np.inf) >= min_capacity
    checks["subperiod_stable"] = (
        _col(result, "subperiod_sign_consistency", 0.0) >= min_subperiod_consistency
    )
    checks["horizon_stable"] = (
        _col(result, "horizon_sign_consistency", 0.0) >= min_horizon_consistency
    )

    result["deployability_pass_count"] = checks.sum(axis=1)
    result["deployability_total_checks"] = checks.shape[1]
    result["deployability_score"] = (
        result["deployability_pass_count"] / result["deployability_total_checks"]
    )
    result["deployable"] = checks.all(axis=1)

    for col in checks.columns:
        result[f"check_{col}"] = checks[col]

    return result


def _normal_two_sided_pvalue(t_stat: pd.Series) -> pd.Series:
    """Normal-approximation two-sided p-values without requiring scipy."""
    import math

    vals = pd.to_numeric(t_stat, errors="coerce")
    return vals.apply(lambda x: math.erfc(abs(x) / math.sqrt(2.0)) if pd.notna(x) else np.nan)


def _col(df: pd.DataFrame, name: str, default: float) -> pd.Series:
    if name in df.columns:
        return pd.to_numeric(df[name], errors="coerce")
    return pd.Series(default, index=df.index, dtype=float)
