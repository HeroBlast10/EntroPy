"""Redundancy diagnostics and complementary factor selection.

The goal is not to keep every statistically significant factor.  This module
keeps a smaller set of factors that are individually usable and mutually
different across three lenses:

1. Effective signal correlation.
2. Factor long-short return correlation.
3. Exposure-vector similarity.

Two selection strategies are provided:

- :func:`select_complementary_factors`  — greedy, threshold-driven (legacy).
- :func:`cluster_based_factor_selection` — hierarchical clustering on
  ``1 - |correlation|`` distance, picks one representative per cluster.
  Avoids the brittle behaviour of hard correlation thresholds when a small
  factor universe contains many "almost-similar" signals.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

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


# ===================================================================
# Hierarchical-clustering selection
# ===================================================================

def cluster_based_factor_selection(
    score_table: pd.DataFrame,
    redundancy_report: Dict[str, pd.DataFrame],
    *,
    n_clusters: Optional[int] = None,
    max_clusters: int = 5,
    min_clusters: int = 3,
    distance_metric: str = "signal_correlation",
    linkage_method: str = "average",
    score_metric: str = "selection_score",
    distance_threshold: Optional[float] = None,
) -> pd.DataFrame:
    """Select complementary factors via agglomerative hierarchical clustering.

    Distance is ``1 - |corr|`` on one of the redundancy matrices; factors that
    behave similarly across stocks fall into the same cluster.  Within each
    cluster we pick the factor with the highest *score_metric* value.

    Why this beats greedy threshold selection
    -----------------------------------------
    * No hard 0.70 cliff — two factors with corr 0.71 are not auto-rejected
      if they live in different clusters by the global structure.
    * Number of selected factors is controlled by tree cutting (``n_clusters``
      or ``distance_threshold``), giving a deterministic, audit-friendly
      output.  The greedy method's order-dependence is gone.
    * Production-style: AQR / Two Sigma / Rebellion style "factor zoos" are
      typically pruned this way before composite construction.

    Parameters
    ----------
    score_table : DataFrame indexed by (or with) ``factor`` column, must
        contain *score_metric* (or fall back to ``deployability_score`` /
        ``cost_adj_ls_sharpe`` / ``ls_sharpe`` / ``ric_mean_ic``).
    redundancy_report : output of :func:`build_redundancy_report`.
    n_clusters : exact number of clusters to keep.  Overrides
        *distance_threshold* when set.  When ``None``, picks
        ``min(max_clusters, n_factors)`` and clamps to ``min_clusters``.
    max_clusters / min_clusters : safety bounds on cluster count.
    distance_metric : which redundancy matrix to use as distance source.
        One of ``"signal_correlation"`` (default), ``"factor_return_correlation"``,
        or ``"exposure_similarity"``.
    linkage_method : linkage method for ``scipy.cluster.hierarchy.linkage``.
        ``"average"`` is the most stable; ``"ward"`` is also defensible.
    distance_threshold : if set, cuts the tree at this height instead of
        using *n_clusters* (e.g. 0.30 means "merge factors whose
        |corr| > 0.70").

    Returns
    -------
    DataFrame with one row per candidate factor and columns:

    - ``factor``
    - ``cluster_id``
    - ``cluster_size``
    - ``selection_score`` (the underlying metric)
    - ``selected`` (bool, True iff this factor is the cluster representative)
    - ``selected_rank`` (1..K for kept factors, NaN otherwise)
    - ``cluster_members`` (semicolon-separated peers within the cluster)
    """
    candidates = score_table.copy() if score_table is not None else pd.DataFrame()
    if candidates.empty:
        return pd.DataFrame(columns=["factor", "selected", "cluster_id"])
    if "factor" not in candidates.columns:
        candidates["factor"] = candidates.index

    # Resolve scoring column with fallbacks
    score_col = score_metric
    for fallback in (score_metric, "deployability_score", "cost_adj_ls_sharpe", "ls_sharpe", "ric_mean_ic"):
        if fallback in candidates.columns:
            score_col = fallback
            break
    candidates["_score"] = pd.to_numeric(candidates.get(score_col, np.nan), errors="coerce")

    matrix = redundancy_report.get(distance_metric)
    if matrix is None or matrix.empty:
        # Fallback: assign every candidate to its own cluster
        candidates["cluster_id"] = np.arange(len(candidates)) + 1
        candidates["cluster_size"] = 1
        candidates["selected"] = True
        candidates["selected_rank"] = np.arange(len(candidates)) + 1
        candidates["cluster_members"] = candidates["factor"]
        candidates["selection_score"] = candidates["_score"]
        return candidates.drop(columns=["_score"]).reset_index(drop=True)

    # Restrict to factors present in both score table and similarity matrix
    factors = [f for f in candidates["factor"] if f in matrix.index and f in matrix.columns]
    if not factors:
        candidates["cluster_id"] = np.nan
        candidates["selected"] = False
        candidates["selected_rank"] = np.nan
        return candidates.drop(columns=["_score"]).reset_index(drop=True)

    # Single factor: trivially keep
    if len(factors) == 1:
        only = factors[0]
        out = candidates[candidates["factor"] == only].copy()
        out["cluster_id"] = 1
        out["cluster_size"] = 1
        out["selected"] = True
        out["selected_rank"] = 1
        out["cluster_members"] = only
        out["selection_score"] = out["_score"]
        return out.drop(columns=["_score"]).reset_index(drop=True)

    sub = matrix.loc[factors, factors].astype(float).fillna(0.0)
    # Distance on |corr| / |similarity|, symmetrise to be safe
    dist_mat = (1.0 - sub.abs()).clip(lower=0.0).to_numpy()
    dist_mat = 0.5 * (dist_mat + dist_mat.T)
    np.fill_diagonal(dist_mat, 0.0)

    cluster_ids = _hierarchical_clusters(
        dist_mat,
        n_factors=len(factors),
        n_clusters=n_clusters,
        max_clusters=max_clusters,
        min_clusters=min_clusters,
        linkage_method=linkage_method,
        distance_threshold=distance_threshold,
    )
    cluster_lookup = pd.Series(cluster_ids, index=factors, name="cluster_id")

    rows: List[Dict] = []
    selected_in_order: List[str] = []

    score_lookup = candidates.set_index("factor")["_score"].reindex(factors).fillna(-np.inf)
    cluster_to_members: Dict[int, List[str]] = {}
    for f, cid in cluster_lookup.items():
        cluster_to_members.setdefault(int(cid), []).append(f)

    # Pick representative per cluster, ordered by best score across clusters
    cluster_best = []
    for cid, members in cluster_to_members.items():
        best = max(members, key=lambda m: score_lookup.loc[m])
        cluster_best.append((cid, best, float(score_lookup.loc[best])))
    cluster_best.sort(key=lambda triple: triple[2], reverse=True)

    selected_set = set(triple[1] for triple in cluster_best)
    rank_lookup = {triple[1]: rank for rank, triple in enumerate(cluster_best, start=1)}

    for factor in candidates["factor"]:
        cid = int(cluster_lookup.loc[factor]) if factor in cluster_lookup.index else -1
        members = cluster_to_members.get(cid, []) if cid > 0 else [factor]
        rows.append({
            "factor": factor,
            "cluster_id": cid,
            "cluster_size": len(members),
            "cluster_members": ";".join(sorted(members)),
            "selection_score": float(score_lookup.loc[factor]) if factor in score_lookup.index else np.nan,
            "selected": factor in selected_set,
            "selected_rank": rank_lookup.get(factor, np.nan),
        })

    result = pd.DataFrame(rows)
    return result.sort_values(["selected", "selected_rank", "cluster_id"], ascending=[False, True, True]).reset_index(drop=True)


def _hierarchical_clusters(
    dist_mat: np.ndarray,
    *,
    n_factors: int,
    n_clusters: Optional[int],
    max_clusters: int,
    min_clusters: int,
    linkage_method: str,
    distance_threshold: Optional[float],
) -> np.ndarray:
    """Run agglomerative clustering and return cluster id (1-indexed) per factor.

    Tries ``scipy.cluster.hierarchy``; falls back to a simple union-find at
    *distance_threshold* if scipy is unavailable.
    """
    try:
        from scipy.cluster.hierarchy import fcluster, linkage
        from scipy.spatial.distance import squareform

        condensed = squareform(dist_mat, checks=False)
        Z = linkage(condensed, method=linkage_method)
        if distance_threshold is not None:
            return fcluster(Z, t=float(distance_threshold), criterion="distance").astype(int)

        target = n_clusters if n_clusters is not None else min(max(min_clusters, n_factors // 3 or 1), max_clusters, n_factors)
        target = int(max(1, min(target, n_factors)))
        return fcluster(Z, t=target, criterion="maxclust").astype(int)
    except Exception:
        # Fallback: union-find at threshold
        threshold = distance_threshold if distance_threshold is not None else 0.30
        parent = list(range(n_factors))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for i in range(n_factors):
            for j in range(i + 1, n_factors):
                if dist_mat[i, j] <= threshold:
                    a, b = find(i), find(j)
                    if a != b:
                        parent[b] = a

        roots = sorted({find(i) for i in range(n_factors)})
        root_to_id = {root: idx + 1 for idx, root in enumerate(roots)}
        return np.array([root_to_id[find(i)] for i in range(n_factors)], dtype=int)
