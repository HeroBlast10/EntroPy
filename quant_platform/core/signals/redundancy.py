"""Redundancy diagnostics and complementary factor selection.

The goal is not to keep every statistically significant factor.  This module
keeps a smaller set of factors that are individually usable and mutually
different across three lenses:

1. Effective signal correlation.
2. Factor long-short return correlation.
3. Exposure-vector similarity.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd

from quant_platform.core.signals.cross_sectional.evaluation import long_short_returns
from quant_platform.core.signals.effective import build_effective_signal


@dataclass(frozen=True)
class RedundancyConfig:
    """Thresholds for production-style complementary factor selection."""

    max_signal_corr: float = 0.70
    max_return_corr: float = 0.70
    max_exposure_similarity: float = 0.80
    min_factors: int = 3
    max_factors: int = 5
    min_incremental_sharpe: float = 0.0
    score_metric: str = "selection_score"


def build_effective_factor_matrix(
    factor_df: pd.DataFrame,
    factor_cols: Iterable[str],
    *,
    direction_map: Optional[Dict[str, int]] = None,
    neutralize_by: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """Return a MultiIndex matrix of effective factor exposures.

    Rows are ``(date, ticker)`` and columns are factors.  Each column has gone
    through the shared direction/winsorize/neutralize/zscore/rank pipeline.
    """
    direction_map = direction_map or {}
    factor_cols = [c for c in factor_cols if c in factor_df.columns]
    if not factor_cols:
        return pd.DataFrame()

    base_index = factor_df[["date", "ticker"]].copy()
    base_index["date"] = pd.to_datetime(base_index["date"])
    base_index = pd.MultiIndex.from_frame(base_index)

    matrix = pd.DataFrame(index=base_index)
    for col in factor_cols:
        eff = build_effective_signal(
            factor_df,
            col,
            direction=direction_map.get(col, 1),
            neutralize_by=neutralize_by,
            rank=True,
        )
        eff_index = pd.MultiIndex.from_frame(eff[["date", "ticker"]])
        matrix[col] = pd.Series(eff[col].values, index=eff_index).reindex(base_index).values

    return matrix


def factor_signal_correlation(
    factor_df: pd.DataFrame,
    factor_cols: Iterable[str],
    *,
    direction_map: Optional[Dict[str, int]] = None,
    neutralize_by: Optional[Iterable[str]] = None,
    method: str = "spearman",
) -> pd.DataFrame:
    """Correlation of effective factor exposures across all date-name rows."""
    matrix = build_effective_factor_matrix(
        factor_df,
        factor_cols,
        direction_map=direction_map,
        neutralize_by=neutralize_by,
    )
    if matrix.empty:
        return pd.DataFrame()
    return matrix.corr(method=method)


def factor_exposure_similarity(
    factor_df: pd.DataFrame,
    factor_cols: Iterable[str],
    *,
    direction_map: Optional[Dict[str, int]] = None,
    neutralize_by: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """Cosine similarity of effective exposure vectors."""
    matrix = build_effective_factor_matrix(
        factor_df,
        factor_cols,
        direction_map=direction_map,
        neutralize_by=neutralize_by,
    )
    if matrix.empty:
        return pd.DataFrame()

    cols = list(matrix.columns)
    values = matrix.fillna(0.0).to_numpy(dtype=float)
    norms = np.linalg.norm(values, axis=0)
    denom = np.outer(norms, norms)
    sim = np.divide(values.T @ values, denom, out=np.zeros((len(cols), len(cols))), where=denom > 0)
    return pd.DataFrame(sim, index=cols, columns=cols)


def factor_long_short_return_panel(
    factor_df: pd.DataFrame,
    factor_cols: Iterable[str],
    *,
    return_col: str = "fwd_ret_1d",
    direction_map: Optional[Dict[str, int]] = None,
    neutralize_by: Optional[Iterable[str]] = None,
    n_quantiles: int = 5,
) -> pd.DataFrame:
    """Daily long-short return panel, one column per effective factor."""
    if return_col not in factor_df.columns:
        return pd.DataFrame(columns=list(factor_cols))

    direction_map = direction_map or {}
    series = {}
    for col in factor_cols:
        if col not in factor_df.columns:
            continue
        eff = build_effective_signal(
            factor_df,
            col,
            direction=direction_map.get(col, 1),
            neutralize_by=neutralize_by,
            rank=True,
        )
        series[col] = long_short_returns(eff, col, return_col=return_col, n_quantiles=n_quantiles)

    if not series:
        return pd.DataFrame()
    return pd.concat(series, axis=1).sort_index()


def factor_return_correlation(
    factor_df: pd.DataFrame,
    factor_cols: Iterable[str],
    *,
    return_col: str = "fwd_ret_1d",
    direction_map: Optional[Dict[str, int]] = None,
    neutralize_by: Optional[Iterable[str]] = None,
    n_quantiles: int = 5,
) -> pd.DataFrame:
    """Correlation of factor long-short return streams."""
    panel = factor_long_short_return_panel(
        factor_df,
        factor_cols,
        return_col=return_col,
        direction_map=direction_map,
        neutralize_by=neutralize_by,
        n_quantiles=n_quantiles,
    )
    if panel.empty:
        return pd.DataFrame()
    return panel.corr()


def build_redundancy_report(
    factor_df: pd.DataFrame,
    factor_cols: Iterable[str],
    *,
    direction_map: Optional[Dict[str, int]] = None,
    neutralize_by: Optional[Iterable[str]] = None,
    return_col: str = "fwd_ret_1d",
    config: Optional[RedundancyConfig] = None,
) -> Dict[str, pd.DataFrame]:
    """Build correlation/similarity matrices and simple redundancy clusters."""
    cfg = config or RedundancyConfig()
    factor_cols = [c for c in factor_cols if c in factor_df.columns]
    signal_corr = factor_signal_correlation(
        factor_df,
        factor_cols,
        direction_map=direction_map,
        neutralize_by=neutralize_by,
    )
    return_panel = factor_long_short_return_panel(
        factor_df,
        factor_cols,
        return_col=return_col,
        direction_map=direction_map,
        neutralize_by=neutralize_by,
    )
    return_corr = return_panel.corr() if not return_panel.empty else pd.DataFrame()
    exposure_sim = factor_exposure_similarity(
        factor_df,
        factor_cols,
        direction_map=direction_map,
        neutralize_by=neutralize_by,
    )
    clusters = _redundancy_clusters(signal_corr, return_corr, exposure_sim, cfg)

    return {
        "signal_correlation": signal_corr,
        "factor_return_correlation": return_corr,
        "exposure_similarity": exposure_sim,
        "factor_return_panel": return_panel,
        "clusters": clusters,
    }


def select_complementary_factors(
    score_table: pd.DataFrame,
    redundancy_report: Dict[str, pd.DataFrame],
    *,
    config: Optional[RedundancyConfig] = None,
) -> pd.DataFrame:
    """Greedy production filter for 3-5 complementary factors.

    Candidates are ranked by deployability/selection score.  A factor is added
    only if it is sufficiently different from the selected set and, when
    return streams are available, has positive residual long-short Sharpe after
    regressing on the already selected factors.
    """
    cfg = config or RedundancyConfig()
    if score_table is None or score_table.empty:
        return pd.DataFrame(columns=["factor", "selected", "selection_reason"])

    candidates = score_table.copy()
    if "factor" not in candidates.columns:
        candidates["factor"] = candidates.index

    score_col = cfg.score_metric
    for fallback in (score_col, "deployability_score", "cost_adj_ls_sharpe", "ls_sharpe", "ric_mean_ic"):
        if fallback in candidates.columns:
            score_col = fallback
            break

    candidates["_score"] = pd.to_numeric(candidates.get(score_col, np.nan), errors="coerce")
    candidates = candidates.sort_values("_score", ascending=False, na_position="last")

    selected: list[str] = []
    rows = []
    for _, row in candidates.iterrows():
        factor = str(row["factor"])
        ok, reason, diagnostics = _can_add_factor(factor, selected, redundancy_report, cfg)
        if ok and len(selected) < cfg.max_factors:
            selected.append(factor)
            status = True
        else:
            status = False
        rows.append({
            "factor": factor,
            "selected": status,
            "selection_reason": "accepted" if status else reason,
            "selection_score": row["_score"],
            **diagnostics,
        })

    result = pd.DataFrame(rows).set_index("factor", drop=False)

    if len(selected) < cfg.min_factors:
        # If thresholds are too tight for a tiny universe, fill to min_factors
        # but prefer the least redundant rejected candidates and make the
        # reason explicit for downstream review.
        remaining = result[~result["selected"]].copy()
        redundancy_cols = [
            "max_abs_signal_corr_to_selected",
            "max_abs_return_corr_to_selected",
            "max_abs_exposure_similarity_to_selected",
        ]
        remaining["_redundancy_max"] = (
            remaining[redundancy_cols]
            .apply(pd.to_numeric, errors="coerce")
            .fillna(0.0)
            .max(axis=1)
        )
        remaining = remaining.sort_values(
            ["_redundancy_max", "selection_score"],
            ascending=[True, False],
        )
        for factor in remaining.index[: max(0, cfg.min_factors - len(selected))]:
            result.loc[factor, "selected"] = True
            result.loc[factor, "selection_reason"] = "accepted_to_meet_min_count_review_redundancy"
            selected.append(factor)

    result["selected_rank"] = np.nan
    for rank, factor in enumerate(selected[: cfg.max_factors], start=1):
        if factor in result.index:
            result.loc[factor, "selected_rank"] = rank

    return result


def _can_add_factor(
    factor: str,
    selected: list[str],
    report: Dict[str, pd.DataFrame],
    cfg: RedundancyConfig,
) -> tuple[bool, str, Dict[str, float]]:
    if not selected:
        return True, "first_factor", {
            "max_abs_signal_corr_to_selected": 0.0,
            "max_abs_return_corr_to_selected": 0.0,
            "max_abs_exposure_similarity_to_selected": 0.0,
            "incremental_sharpe": np.nan,
        }

    max_sig = _max_abs_lookup(report.get("signal_correlation"), factor, selected)
    max_ret = _max_abs_lookup(report.get("factor_return_correlation"), factor, selected)
    max_exp = _max_abs_lookup(report.get("exposure_similarity"), factor, selected)
    inc_sharpe = _incremental_sharpe(report.get("factor_return_panel"), factor, selected)

    diagnostics = {
        "max_abs_signal_corr_to_selected": max_sig,
        "max_abs_return_corr_to_selected": max_ret,
        "max_abs_exposure_similarity_to_selected": max_exp,
        "incremental_sharpe": inc_sharpe,
    }

    if pd.notna(max_sig) and max_sig > cfg.max_signal_corr:
        return False, "signal_corr_too_high", diagnostics
    if pd.notna(max_ret) and max_ret > cfg.max_return_corr:
        return False, "factor_return_corr_too_high", diagnostics
    if pd.notna(max_exp) and max_exp > cfg.max_exposure_similarity:
        return False, "exposure_similarity_too_high", diagnostics
    if pd.notna(inc_sharpe) and inc_sharpe < cfg.min_incremental_sharpe:
        return False, "incremental_alpha_too_weak", diagnostics

    return True, "accepted", diagnostics


def _redundancy_clusters(
    signal_corr: pd.DataFrame,
    return_corr: pd.DataFrame,
    exposure_sim: pd.DataFrame,
    cfg: RedundancyConfig,
) -> pd.DataFrame:
    factors = sorted(set(signal_corr.index) | set(return_corr.index) | set(exposure_sim.index))
    if not factors:
        return pd.DataFrame(columns=["factor", "cluster_id", "cluster_size"])

    parent = {f: f for f in factors}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        pa, pb = find(a), find(b)
        if pa != pb:
            parent[pb] = pa

    for matrix, threshold in (
        (signal_corr, cfg.max_signal_corr),
        (return_corr, cfg.max_return_corr),
        (exposure_sim, cfg.max_exposure_similarity),
    ):
        if matrix is None or matrix.empty:
            continue
        common = [f for f in factors if f in matrix.index and f in matrix.columns]
        for i, a in enumerate(common):
            for b in common[i + 1:]:
                val = matrix.loc[a, b]
                if pd.notna(val) and abs(float(val)) > threshold:
                    union(a, b)

    cluster_key = {root: idx + 1 for idx, root in enumerate(sorted({find(f) for f in factors}))}
    rows = []
    for factor in factors:
        cluster_id = cluster_key[find(factor)]
        rows.append({"factor": factor, "cluster_id": cluster_id})
    clusters = pd.DataFrame(rows)
    sizes = clusters.groupby("cluster_id")["factor"].transform("size")
    clusters["cluster_size"] = sizes
    return clusters.sort_values(["cluster_id", "factor"]).reset_index(drop=True)


def _max_abs_lookup(matrix: Optional[pd.DataFrame], factor: str, selected: list[str]) -> float:
    if matrix is None or matrix.empty or factor not in matrix.index:
        return np.nan
    vals = []
    for other in selected:
        if other in matrix.columns:
            vals.append(matrix.loc[factor, other])
    if not vals:
        return np.nan
    arr = np.asarray(vals, dtype=float)
    if np.isnan(arr).all():
        return np.nan
    return float(np.nanmax(np.abs(arr)))


def _incremental_sharpe(
    return_panel: Optional[pd.DataFrame],
    factor: str,
    selected: list[str],
    annualisation: int = 252,
) -> float:
    if return_panel is None or return_panel.empty or factor not in return_panel.columns:
        return np.nan
    cols = [c for c in selected if c in return_panel.columns]
    if not cols:
        return np.nan
    data = return_panel[[factor] + cols].dropna()
    if len(data) < max(20, len(cols) + 5):
        return np.nan

    y = data[factor].to_numpy(dtype=float)
    x = data[cols].to_numpy(dtype=float)
    x = np.column_stack([np.ones(len(x)), x])
    beta = np.linalg.lstsq(x, y, rcond=None)[0]
    residual = y - x @ beta
    std = residual.std(ddof=1)
    if std <= 0:
        return np.nan
    return float(residual.mean() / std * np.sqrt(annualisation))
