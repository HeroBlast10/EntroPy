"""Cross-sectional factor ranking alpha model."""
from __future__ import annotations
from typing import List, Optional
import numpy as np
import pandas as pd
from loguru import logger


class CrossSectionalRanker:
    """Rank stocks cross-sectionally by a composite factor score.
    
    Combines multiple factors into a single signal via z-score averaging,
    then ranks stocks on each date.
    """
    
    def __init__(self, factor_names: List[str], weights: Optional[List[float]] = None):
        self.factor_names = factor_names
        self.weights = weights or [1.0 / len(factor_names)] * len(factor_names)
    
    def score(self, factor_df: pd.DataFrame) -> pd.DataFrame:
        """Compute composite z-score for each (date, ticker)."""
        df = factor_df.copy()
        # Z-score each factor cross-sectionally
        for fname in self.factor_names:
            if fname not in df.columns:
                continue
            grp = df.groupby("date")[fname]
            mu = grp.transform("mean")
            std = grp.transform("std").replace(0, np.nan)
            df[f"_z_{fname}"] = (df[fname] - mu) / std
        
        # Weighted average of z-scores
        z_cols = [f"_z_{f}" for f in self.factor_names if f"_z_{f}" in df.columns]
        if not z_cols:
            df["alpha_cs"] = 0.0
            return df[["date", "ticker", "alpha_cs"]]
        
        z_matrix = df[z_cols].values
        w = np.array(self.weights[:len(z_cols)])
        w = w / w.sum()
        df["alpha_cs"] = np.nanmean(z_matrix * w, axis=1)
        
        return df[["date", "ticker", "alpha_cs"]]
    
    def rank(self, factor_df: pd.DataFrame) -> pd.DataFrame:
        """Return percentile rank [0, 1] of composite score."""
        scored = self.score(factor_df)
        scored["rank_cs"] = scored.groupby("date")["alpha_cs"].rank(pct=True)
        return scored
