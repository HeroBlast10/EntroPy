"""Multi-metric factor screening and multiple-testing controls."""

from __future__ import annotations

import math
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


def hansen_spa_test(
    long_short_returns: Dict[str, pd.Series],
    *,
    n_boot: int = 1000,
    block_length: Optional[int] = None,
    random_state: int = 42,
) -> Dict[str, object]:
    """Hansen (2005) Superior Predictive Ability (SPA) test.

    Tests the null hypothesis "no candidate strategy outperforms the
    benchmark of zero return" while correctly handling multiple comparisons.
    Hansen's improvement over White's Reality Check is the **studentisation**
    and the **threshold ``A`` truncation** that prevents poor strategies from
    biasing the limit distribution downward, which inflates the p-value.

    Three p-values are reported (Hansen 2005, Section 3); they obey the
    asymptotic ordering ``p_l <= p_c <= p_u``:

    - ``spa_pvalue_l`` (lower bound, most powerful):  recenters only
      strategies with **positive** sample mean.  Bad strategies stay bad in
      the bootstrap and contribute zero to the bootstrap max, giving the
      smallest bootstrap distribution and the smallest p-value.
    - ``spa_pvalue_c`` (consistent, recommended for practical use):
      recenters strategies whose mean exceeds the Hansen threshold
      ``-sqrt(2 * sigma^2 / n * log log n)``.  Asymptotically correct.
    - ``spa_pvalue_u`` (upper bound, equals White's Reality Check):
      recenters every strategy.  Most conservative — bad strategies are
      treated as if they had zero mean and inflate the bootstrap max.

    Stationary block bootstrap (Politis & Romano 1994) is used to preserve
    short-run dependence in the return series.

    Parameters
    ----------
    long_short_returns : ``{factor_name: daily LS return series}``.
    n_boot : number of bootstrap resamples (1000 is sufficient for 5%
        significance level testing).
    block_length : expected stationary block length (geometric distribution
        mean).  Defaults to ``floor(n^(1/3))`` per Politis-White.
    random_state : RNG seed.

    Returns
    -------
    Dict with:
        - ``spa_pvalue_l``, ``spa_pvalue_c``, ``spa_pvalue_u``: SPA p-values
        - ``best_strategy``: name with the highest studentised mean
        - ``test_statistic``: ``max(0, sqrt(n) * max_k(mean_k / sigma_k))``
        - ``n_strategies``, ``n_obs``, ``block_length``
    """
    cleaned = {
        name: ret.dropna().astype(float)
        for name, ret in long_short_returns.items()
        if ret is not None and len(ret.dropna()) > 10
    }
    if not cleaned:
        return {
            "spa_pvalue_l": np.nan,
            "spa_pvalue_c": np.nan,
            "spa_pvalue_u": np.nan,
            "best_strategy": None,
            "test_statistic": np.nan,
            "n_strategies": 0,
            "n_obs": 0,
        }

    panel = pd.concat(cleaned, axis=1).dropna(how="all").fillna(0.0)
    names = list(panel.columns)
    values = panel.to_numpy(dtype=float)  # (n, k)
    n, k = values.shape
    if n < 20 or k < 1:
        return {
            "spa_pvalue_l": np.nan,
            "spa_pvalue_c": np.nan,
            "spa_pvalue_u": np.nan,
            "best_strategy": None,
            "test_statistic": np.nan,
            "n_strategies": int(k),
            "n_obs": int(n),
        }

    if block_length is None:
        block_length = max(2, int(np.floor(n ** (1.0 / 3.0))))

    means = values.mean(axis=0)  # (k,)
    # Use a HAC-style sample variance to studentise: simple unbiased var here;
    # the bootstrap itself absorbs serial dependence via stationary blocks.
    variances = values.var(axis=0, ddof=1)
    variances = np.where(variances > 0, variances, 1e-12)
    omega = np.sqrt(variances)  # (k,)

    sqrt_n = math.sqrt(n)
    studentised = sqrt_n * means / omega
    test_stat = float(max(0.0, np.max(studentised)))
    best_idx = int(np.argmax(studentised))

    # Hansen (2005) threshold for "viable" strategies (eq. 8 of the paper).
    # Mean must exceed -A_n to be treated as a credible competitor.
    if n > 3 and math.log(n) > 1:
        A_n = -np.sqrt(omega ** 2 / n * 2.0 * math.log(math.log(n)))
    else:
        A_n = -np.full(k, np.inf)

    # Recentering vectors. Whether a strategy contributes to the bootstrap
    # max under the null is controlled by g_k:
    #   bootstrap statistic = sqrt(n) * (boot_mean_k - g_k) / sigma_k
    # If g_k = mean_k → bootstrap is centered at zero → strategy fully
    # recentered. If g_k = 0 → bootstrap retains the (negative) observed
    # mean → strategy contributes zero to max(0, ...) → effectively dropped.
    g_l = np.where(means > 0, means, 0.0)            # lower bound (most powerful)
    g_c = np.where(means >= A_n, means, 0.0)          # consistent
    g_u = means.copy()                                # upper bound = White's RC

    rng = np.random.default_rng(random_state)
    boot_stats_l = np.empty(n_boot)
    boot_stats_c = np.empty(n_boot)
    boot_stats_u = np.empty(n_boot)

    # Stationary bootstrap indices: avg block length = block_length
    p_geom = 1.0 / float(block_length)
    for b in range(n_boot):
        idx = _stationary_bootstrap_indices(n, p_geom, rng)
        sample = values[idx]                        # (n, k)
        boot_means = sample.mean(axis=0)
        boot_stud_l = sqrt_n * (boot_means - g_l) / omega
        boot_stud_c = sqrt_n * (boot_means - g_c) / omega
        boot_stud_u = sqrt_n * (boot_means - g_u) / omega
        boot_stats_l[b] = max(0.0, float(np.max(boot_stud_l)))
        boot_stats_c[b] = max(0.0, float(np.max(boot_stud_c)))
        boot_stats_u[b] = max(0.0, float(np.max(boot_stud_u)))

    return {
        "spa_pvalue_l": float((boot_stats_l >= test_stat).mean()),
        "spa_pvalue_c": float((boot_stats_c >= test_stat).mean()),
        "spa_pvalue_u": float((boot_stats_u >= test_stat).mean()),
        "best_strategy": names[best_idx],
        "test_statistic": test_stat,
        "n_strategies": int(k),
        "n_obs": int(n),
        "block_length": int(block_length),
    }


def _stationary_bootstrap_indices(n: int, p: float, rng: np.random.Generator) -> np.ndarray:
    """Politis-Romano (1994) stationary bootstrap index generator."""
    idx = np.empty(n, dtype=np.int64)
    idx[0] = int(rng.integers(0, n))
    new_block = rng.random(n) < p
    for t in range(1, n):
        if new_block[t]:
            idx[t] = int(rng.integers(0, n))
        else:
            idx[t] = (idx[t - 1] + 1) % n
    return idx


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

            try:
                spa = hansen_spa_test(ls_rets)
                # The SPA p-values are joint statistics (single value per universe of strategies);
                # we attach them as columns broadcast to all rows so they survive a CSV roundtrip.
                result["spa_pvalue_l"] = spa.get("spa_pvalue_l", np.nan)
                result["spa_pvalue_c"] = spa.get("spa_pvalue_c", np.nan)
                result["spa_pvalue_u"] = spa.get("spa_pvalue_u", np.nan)
                result["spa_best_strategy"] = spa.get("best_strategy")
                result["spa_pass_10pct"] = bool(
                    pd.notna(spa.get("spa_pvalue_c"))
                    and float(spa["spa_pvalue_c"]) <= alpha
                )
            except Exception:
                # SPA is optional; do not break the screening pipeline
                pass

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
