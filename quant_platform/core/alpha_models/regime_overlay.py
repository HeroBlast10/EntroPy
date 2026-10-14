"""Regime overlay: modulate portfolio weights based on HMM state."""
from __future__ import annotations
from typing import Optional
import numpy as np
import pandas as pd
from loguru import logger


class RegimeOverlay:
    """Scale portfolio weights based on HMM turbulence probability.
    
    In turbulent regimes, reduce exposure. In laminar regimes, maintain
    full exposure. This acts as a risk-management overlay, not a direct alpha.
    
    Parameters
    ----------
    turbulence_col : column name for P(turbulent)
    reduction_factor : multiply weights by this in turbulent state
    threshold : P(turb) above this triggers reduction
    """
    
    def __init__(
        self,
        turbulence_col: str = "HMM_TURBULENCE_PROB",
        reduction_factor: float = 0.5,
        threshold: float = 0.6,
    ):
        self.turbulence_col = turbulence_col
        self.reduction = reduction_factor
        self.threshold = threshold
    
    def apply(self, weights_df: pd.DataFrame, factor_df: pd.DataFrame) -> pd.DataFrame:
        """Scale weights based on regime state.
        
        Parameters
        ----------
        weights_df : DataFrame with columns [date, ticker, weight]
        factor_df : DataFrame with turbulence probability column
        """
        if self.turbulence_col not in factor_df.columns:
            logger.warning("Turbulence column {} not found, skipping overlay", self.turbulence_col)
            return weights_df
        
        turb = factor_df.groupby("date")[self.turbulence_col].mean()
        
        result = weights_df.copy()
        scalar = result["date"].map(
            pd.Series(
                np.where(turb > self.threshold, self.reduction, 1.0),
                index=turb.index,
            )
        ).fillna(1.0)
        result["weight"] = result["weight"] * scalar.to_numpy()

        # Do not renormalise here: doing so would exactly cancel the intended
        # long-only exposure reduction and can explode a near-zero-net book.
        
        return result
    
    def compute_regime_scalar(self, factor_df: pd.DataFrame) -> pd.Series:
        """Return a per-date scalar in [reduction, 1.0] for weight modulation."""
        if self.turbulence_col not in factor_df.columns:
            return pd.Series(dtype=float)
        
        turb = factor_df.groupby("date")[self.turbulence_col].mean()
        scalar = pd.Series(1.0, index=turb.index)
        scalar[turb > self.threshold] = self.reduction
        scalar.name = "regime_scalar"
        return scalar
