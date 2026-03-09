"""Ensemble alpha model: combine cross-sectional ranker + time-series forecaster + regime overlay."""
from __future__ import annotations
from typing import Optional
import numpy as np
import pandas as pd
from loguru import logger

from quant_platform.core.alpha_models.cross_sectional_ranker import CrossSectionalRanker
from quant_platform.core.alpha_models.ts_forecaster import TSForecaster
from quant_platform.core.alpha_models.regime_overlay import RegimeOverlay


class EnsembleAlpha:
    """Weighted combination of cross-sectional and time-series alpha,
    modulated by regime overlay.
    
    final_alpha = w_cs * alpha_cs + w_ts * alpha_ts
    final_alpha *= regime_scalar  (optional)
    
    Parameters
    ----------
    cs_ranker : CrossSectionalRanker instance
    ts_forecaster : TSForecaster instance
    regime_overlay : RegimeOverlay instance (optional)
    w_cs, w_ts : blending weights (must sum to 1)
    """
    
    def __init__(
        self,
        cs_ranker: Optional[CrossSectionalRanker] = None,
        ts_forecaster: Optional[TSForecaster] = None,
        regime_overlay: Optional[RegimeOverlay] = None,
        w_cs: float = 0.6,
        w_ts: float = 0.4,
    ):
        self.cs_ranker = cs_ranker
        self.ts_forecaster = ts_forecaster
        self.regime_overlay = regime_overlay
        self.w_cs = w_cs
        self.w_ts = w_ts
    
    def score(self, factor_df: pd.DataFrame) -> pd.DataFrame:
        """Compute ensemble alpha score."""
        df = factor_df[["date", "ticker"]].drop_duplicates().copy()
        
        # Cross-sectional component
        if self.cs_ranker is not None:
            cs_scored = self.cs_ranker.score(factor_df)
            df = df.merge(cs_scored[["date", "ticker", "alpha_cs"]], on=["date", "ticker"], how="left")
        else:
            df["alpha_cs"] = 0.0
        
        # Time-series component
        if self.ts_forecaster is not None:
            ts_scored = self.ts_forecaster.score(factor_df)
            df = df.merge(ts_scored[["date", "ticker", "alpha_ts"]], on=["date", "ticker"], how="left")
        else:
            df["alpha_ts"] = 0.0
        
        df["alpha_cs"] = df["alpha_cs"].fillna(0.0)
        df["alpha_ts"] = df["alpha_ts"].fillna(0.0)
        
        # Blend
        df["alpha_ensemble"] = self.w_cs * df["alpha_cs"] + self.w_ts * df["alpha_ts"]
        
        # Regime modulation
        if self.regime_overlay is not None:
            scalar = self.regime_overlay.compute_regime_scalar(factor_df)
            df = df.merge(scalar.rename("_regime_scalar").reset_index(), on="date", how="left")
            df["_regime_scalar"] = df["_regime_scalar"].fillna(1.0)
            df["alpha_ensemble"] *= df["_regime_scalar"]
            df.drop(columns=["_regime_scalar"], inplace=True)
        
        return df[["date", "ticker", "alpha_cs", "alpha_ts", "alpha_ensemble"]]
