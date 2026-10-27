"""Paired score diagnostics without confusing overlapping labels with breadth."""
from __future__ import annotations
import numpy as np
import pandas as pd
from quant_platform.core.alpha_models.nonlinear_alpha import daily_rank_ic
from quant_platform.core.evaluation.factor_diagnostics import hac_summary


def adjacent_rank_correlation(rows, score, *, calendar=None):
    """Spearman on shared names in ADJACENT exchange sessions; never bridge gaps."""
    if rows.duplicated(["date", "ticker"]).any():
        raise ValueError("Duplicate score keys")
    calendar = pd.DatetimeIndex(sorted(rows.date.unique())) if calendar is None else pd.DatetimeIndex(calendar)
    prior = rows[["date", "ticker", score]].rename(columns={score: "_prior_score"}).copy()
    prior["date"] = prior.date.map(pd.Series(calendar[1:], index=calendar[:-1]))
    matched = rows[["date", "ticker", score]].merge(prior.dropna(subset=["date"]),
        on=["date", "ticker"], how="inner", validate="one_to_one")
    return daily_rank_ic(matched, score, return_col="_prior_score").reindex(calendar)


def threshold_rank_update(rows, score, threshold=.10):
    """Causal deadband on percentile ranks; absent names reset, no label input."""
    if not 0 < threshold < 1 or rows.duplicated(["date", "ticker"]).any():
        raise ValueError("Invalid threshold or duplicate keys")
    calendar = pd.DatetimeIndex(sorted(rows.date.unique()))
    values = rows.groupby("date")[score].rank(pct=True)
    result = pd.Series(np.nan, index=rows.index, dtype=float)
    state = {}
    for date, group in rows.groupby("date", sort=True):
        fresh = {}
        for i, ticker in zip(group.index, group.ticker):
            value = values.loc[i]
            if not np.isfinite(value):
                continue
            previous = state.get(ticker, value)
            updated = value if abs(value - previous) > threshold else previous
            result.loc[i] = updated
            fresh[ticker] = updated
        state = fresh
    return result


def paired_score_metrics(rows, score="score", baseline="residual_momentum", horizon=20):
    """Both ICs, and score correlation, use exactly the same daily securities."""
    calendar = pd.DatetimeIndex(sorted(rows.date.unique()))
    paired = rows[["date", "ticker", score, baseline, "forward_return"]].dropna()
    model_ic = daily_rank_ic(paired, score).reindex(calendar)
    baseline_ic = daily_rank_ic(paired, baseline).reindex(calendar)
    corr = daily_rank_ic(paired, score, return_col=baseline).reindex(calendar)
    autocorr = adjacent_rank_correlation(rows, score, calendar=calendar)
    baseline_autocorr = adjacent_rank_correlation(rows, baseline, calendar=calendar)
    stats = hac_summary(model_ic, horizon)
    baseline_stats = hac_summary(baseline_ic, horizon)
    delta_stats = hac_summary(model_ic - baseline_ic, horizon)
    sigma = model_ic.std(ddof=1)
    result = {**stats, "icir_daily": float(model_ic.mean() / sigma) if sigma > 0 else np.nan,
        "icir_naive_annualized": float(model_ic.mean() / sigma * np.sqrt(252)) if sigma > 0 else np.nan,
        "resid_mom_ic": baseline_stats["mean_rank_ic"], "resid_mom_hac_t": baseline_stats["hac_t"],
        "paired_ic_difference": float((model_ic - baseline_ic).mean()), "paired_ic_difference_hac_t": delta_stats["hac_t"],
        "mean_score_resid_mom_correlation": float(corr.mean()),
        "mean_adjacent_rank_correlation": float(autocorr.mean()),
        "p05_adjacent_rank_correlation": float(autocorr.quantile(.05)),
        "resid_mom_mean_adjacent_rank_correlation": float(baseline_autocorr.mean()),
        "mean_paired_securities": float(paired.groupby("date").size().mean())}
    series = pd.DataFrame({"model_ic": model_ic, "resid_mom_ic": baseline_ic,
        "score_resid_mom_correlation": corr, "adjacent_rank_correlation": autocorr,
        "resid_mom_adjacent_rank_correlation": baseline_autocorr})
    return result, series


def coefficient_stability(coefficients, zero_tolerance=1e-5):
    """Sign transitions describe fitted paths; they are not a feature deletion rule."""
    rows = []
    for feature, group in coefficients.sort_values("fold").groupby("feature", sort=False):
        values = group.coefficient.to_numpy(dtype=float)
        signs = np.sign(values[np.abs(values) > zero_tolerance])
        rows.append({"feature": feature, "folds": len(values),
            "material_folds": len(signs), "sign_transitions": int((np.diff(signs) != 0).sum()),
            "positive_folds": int((signs > 0).sum()), "negative_folds": int((signs < 0).sum()),
            "mean_coefficient": float(values.mean()), "median_abs_coefficient": float(np.median(np.abs(values))),
            "coefficient_std": float(values.std(ddof=1)) if len(values)>1 else np.nan})
    return pd.DataFrame(rows)
