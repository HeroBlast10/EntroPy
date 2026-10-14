"""Cross-sectional factor ranking alpha model."""
from __future__ import annotations
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from loguru import logger

from quant_platform.core.signals.effective import build_effective_signal


class CrossSectionalRanker:
    """Rank stocks cross-sectionally by a composite factor score.
    
    Combines multiple factors into a single signal via z-score averaging,
    then ranks stocks on each date.
    """
    
    def __init__(
        self,
        factor_names: List[str],
        weights: Optional[List[float]] = None,
        directions: Optional[Dict[str, int]] = None,
    ):
        if not factor_names:
            raise ValueError("factor_names must not be empty")
        if weights is not None and len(weights) != len(factor_names):
            raise ValueError("weights must have the same length as factor_names")
        self.factor_names = factor_names
        self.weights = (
            list(weights)
            if weights is not None
            else [1.0 / len(factor_names)] * len(factor_names)
        )
        self._weight_map = dict(zip(self.factor_names, self.weights))
        self.directions = directions or self._infer_directions(factor_names)
    
    def score(self, factor_df: pd.DataFrame) -> pd.DataFrame:
        """Compute composite z-score for each (date, ticker)."""
        df = factor_df.copy()
        # Build one effective input per factor, then average aligned scores.
        for fname in self.factor_names:
            if fname not in df.columns:
                continue
            eff_col = f"_eff_{fname}"
            eff = build_effective_signal(
                df[["date", "ticker", fname]].copy(),
                fname,
                output_col=eff_col,
                direction=self.directions.get(fname, 1),
            )
            eff_idx = eff.set_index(["date", "ticker"])[eff_col]
            df[eff_col] = eff_idx.reindex(df.set_index(["date", "ticker"]).index).values
        
        # Weighted average of z-scores
        available = [f for f in self.factor_names if f"_eff_{f}" in df.columns]
        eff_cols = [f"_eff_{f}" for f in available]
        if not eff_cols:
            df["alpha_cs"] = 0.0
            return df[["date", "ticker", "alpha_cs"]]
        
        z_matrix = df[eff_cols].values
        w = np.asarray([self._weight_map[name] for name in available], dtype=float)
        if not np.isfinite(w).all() or np.isclose(np.abs(w).sum(), 0.0):
            raise ValueError("available factor weights must be finite and non-zero")
        w = w / np.abs(w).sum()
        valid = np.isfinite(z_matrix)
        numerator = np.nansum(z_matrix * w, axis=1)
        denominator = (valid * np.abs(w)).sum(axis=1)
        df["alpha_cs"] = np.divide(
            numerator,
            denominator,
            out=np.full(len(df), np.nan, dtype=float),
            where=denominator > 0,
        )
        
        return df[["date", "ticker", "alpha_cs"]]
    
    def rank(self, factor_df: pd.DataFrame) -> pd.DataFrame:
        """Return percentile rank [0, 1] of composite score."""
        scored = self.score(factor_df)
        scored["rank_cs"] = scored.groupby("date")["alpha_cs"].rank(pct=True)
        return scored

    @staticmethod
    def _infer_directions(factor_names: List[str]) -> Dict[str, int]:
        """Best-effort lookup of factor directions from the registry."""
        try:
            from quant_platform.core.signals.registry import FactorRegistry

            reg = FactorRegistry()
            reg.discover()
            return {
                name: reg.get(name).meta.direction
                for name in factor_names
                if name in reg
            }
        except Exception:
            return {}
