"""Unified factor catalog for cross-sectional and typed signal evaluations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from quant_platform.core.utils.io import resolve_data_path


def build_factor_catalog(
    registry,
    comparison: Optional[pd.DataFrame] = None,
    typed_results: Optional[Dict[str, Dict]] = None,
) -> pd.DataFrame:
    """Build one table covering CS, time-series, regime, and relative-value factors."""
    rows = []
    registry_df = registry.list_factors()
    comparison = comparison.copy() if comparison is not None else pd.DataFrame()
    typed_results = typed_results or {}

    for _, meta_row in registry_df.iterrows():
        name = meta_row["name"]
        row = meta_row.to_dict()
        row["factor"] = name
        row["eligible_for_portfolio"] = row["signal_type"] in {
            "cross_sectional",
            "time_series",
            "relative_value",
        }

        if not comparison.empty and name in comparison.index:
            for col, value in comparison.loc[name].items():
                row[col] = value
            row["evaluation_source"] = "cross_sectional"
            row["selection_score"] = _first_numeric(row, ["deployability_score", "cost_adj_ls_sharpe", "ls_sharpe", "ric_mean_ic"])

        if name in typed_results:
            metrics = typed_results[name]
            for col, value in metrics.items():
                row[col] = value
            row["evaluation_source"] = row.get("signal_type", "typed")
            row["selection_score"] = _typed_selection_score(row)

        rows.append(row)

    catalog = pd.DataFrame(rows)
    if catalog.empty:
        return catalog
    catalog = catalog.set_index("factor", drop=False)
    return catalog.sort_values(["eligible_for_portfolio", "selection_score"], ascending=[False, False])


def save_factor_catalog(
    registry,
    comparison: Optional[pd.DataFrame] = None,
    typed_results: Optional[Dict[str, Dict]] = None,
    output_path: Optional[Path | str] = None,
) -> Path:
    """Persist the unified factor catalog to CSV."""
    catalog = build_factor_catalog(registry, comparison, typed_results)
    if output_path is None:
        output_path = resolve_data_path("factors", "factor_catalog.csv")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    catalog.to_csv(output_path, index=True)
    return output_path


def load_factor_catalog(
    *,
    registry=None,
    catalog_path: Optional[Path | str] = None,
    comparison_path: Optional[Path | str] = None,
    typed_path: Optional[Path | str] = None,
) -> pd.DataFrame:
    """Load factor_catalog.csv, rebuilding from legacy artifacts when needed."""
    catalog_path = Path(catalog_path) if catalog_path is not None else resolve_data_path("factors", "factor_catalog.csv")
    if catalog_path.exists():
        catalog = pd.read_csv(catalog_path, index_col=0)
        if "factor" not in catalog.columns:
            catalog["factor"] = catalog.index
        if "eligible_for_portfolio" in catalog.columns:
            catalog["eligible_for_portfolio"] = catalog["eligible_for_portfolio"].map(_to_bool)
        return catalog

    if registry is None:
        from quant_platform.core.signals.registry import FactorRegistry

        registry = FactorRegistry()
        registry.discover()

    comparison_path = Path(comparison_path) if comparison_path is not None else resolve_data_path("factors", "factor_comparison.csv")
    typed_path = Path(typed_path) if typed_path is not None else resolve_data_path("factors", "typed_factor_evaluation.json")

    comparison = pd.read_csv(comparison_path, index_col=0) if comparison_path.exists() else pd.DataFrame()
    typed_results = json.loads(typed_path.read_text(encoding="utf-8")) if typed_path.exists() else {}
    return build_factor_catalog(registry, comparison, typed_results)


def select_best_factor_from_catalog(
    catalog: pd.DataFrame,
    *,
    metric: str = "selection_score",
    portfolio_only: bool = True,
) -> Optional[str]:
    """Return the best factor in the unified catalog."""
    if catalog is None or catalog.empty:
        return None
    candidates = catalog.copy()
    if portfolio_only and "eligible_for_portfolio" in candidates.columns:
        candidates = candidates[candidates["eligible_for_portfolio"].astype(bool)]
    if candidates.empty:
        return None

    if metric not in candidates.columns:
        for fallback in ("deployability_score", "selection_score", "cost_adj_ls_sharpe", "ls_sharpe", "ric_mean_ic"):
            if fallback in candidates.columns:
                metric = fallback
                break
        else:
            return str(candidates.index[0])

    ranked = pd.to_numeric(candidates[metric], errors="coerce").dropna().sort_values(ascending=False)
    if ranked.empty:
        return None
    return str(ranked.index[0])


def _first_numeric(row: Dict, keys: list[str]) -> float:
    for key in keys:
        value = row.get(key)
        try:
            if pd.notna(value):
                return float(value)
        except Exception:
            continue
    return np.nan


def _typed_selection_score(row: Dict) -> float:
    signal_type = row.get("signal_type")
    if signal_type == "time_series":
        return _first_numeric(row, ["directional_sharpe", "hit_rate"])
    if signal_type == "relative_value":
        return _first_numeric(row, ["spread_sharpe", "mean_reversion_quality"])
    if signal_type == "regime":
        return _first_numeric(row, ["sharpe_improvement", "dd_improvement"])
    return _first_numeric(row, ["selection_score"])


def _to_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}
