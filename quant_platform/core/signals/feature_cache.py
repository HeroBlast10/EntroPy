"""Shared feature cache for efficient factor computation.

Precomputes commonly-used price-based features once and reuses them across
multiple factors, eliminating redundant calculations.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd
from loguru import logger


class PriceFeatureCache:
    """Cache of precomputed price features shared across factors.
    
    This class computes and caches intermediate features that many factors
    need (returns, volatilities, volume metrics, etc.), avoiding redundant
    computation when multiple factors use the same base features.
    
    Example
    -------
    >>> cache = PriceFeatureCache(prices)
    >>> ret_1m = cache.get("ret_21d")  # Computed once
    >>> vol_20d = cache.get("vol_20d")  # Reuses ret_1d internally
    """
    
    def __init__(self, prices: pd.DataFrame):
        """Initialize cache with price data.
        
        Parameters
        ----------
        prices : DataFrame with columns [date, ticker, open, high, low, close, 
                 volume, adj_close, adj_factor, ...]
        """
        self.prices = prices.copy()
        self._cache: Dict[str, pd.Series] = {}
        self._grouped = None  # Lazy groupby cache
        
        # Ensure required columns
        if "adj_close" not in prices.columns:
            raise ValueError("prices must have 'adj_close' column")
        if "volume" not in prices.columns:
            raise ValueError("prices must have 'volume' column")
    
    def _ensure_grouped(self):
        """Lazy initialization of grouped DataFrame."""
        if self._grouped is None:
            self._grouped = self.prices.groupby("ticker", group_keys=False)
    
    def get(self, feature_name: str) -> pd.Series:
        """Get a feature, computing it if not cached.
        
        Parameters
        ----------
        feature_name : name of the feature (e.g., "ret_1d", "vol_20d")
        
        Returns
        -------
        Series aligned with self.prices index
        """
        if feature_name in self._cache:
            return self._cache[feature_name]
        
        # Compute and cache
        feature = self._compute_feature(feature_name)
        self._cache[feature_name] = feature
        return feature
    
    def _compute_feature(self, name: str) -> pd.Series:
        """Compute a single feature."""
        self._ensure_grouped()
        
        # --- Returns ---
        if name == "ret_1d":
            return self._grouped["adj_close"].pct_change()
        
        if name.startswith("ret_") and name.endswith("d"):
            periods = int(name.split("_")[1][:-1])
            return self._grouped["adj_close"].pct_change(periods)
        
        # --- Absolute returns ---
        if name == "abs_ret":
            return self.get("ret_1d").abs()
        
        # --- Dollar volume ---
        if name == "dollar_vol":
            return self.prices["adj_close"] * self.prices["volume"]
        
        # --- Market return (equal-weighted) ---
        if name == "market_ret":
            ret_1d = self.get("ret_1d")
            return ret_1d.groupby(self.prices["date"]).mean()
        
        # --- Volatility ---
        if name.startswith("vol_") and name.endswith("d"):
            window = int(name.split("_")[1][:-1])
            ret = self.get("ret_1d")
            return self._grouped.apply(
                lambda g: ret.loc[g.index].rolling(window, min_periods=max(1, window // 2)).std(),
                include_groups=False
            )
        
        # --- Volume metrics ---
        if name.startswith("volume_mean_"):
            window = int(name.split("_")[-1])
            return self._grouped["volume"].transform(
                lambda x: x.rolling(window, min_periods=max(1, window // 2)).mean()
            )
        
        if name.startswith("volume_std_"):
            window = int(name.split("_")[-1])
            return self._grouped["volume"].transform(
                lambda x: x.rolling(window, min_periods=max(1, window // 2)).std()
            )
        
        if name.startswith("volume_median_"):
            window = int(name.split("_")[-1])
            return self._grouped["volume"].transform(
                lambda x: x.rolling(window, min_periods=max(1, window // 2)).median()
            )
        
        # --- High/Low extremes ---
        if name == "high_2d_max":
            return self._grouped["high"].transform(lambda x: x.rolling(2, min_periods=1).max())
        
        if name == "low_2d_min":
            return self._grouped["low"].transform(lambda x: x.rolling(2, min_periods=1).min())
        
        # --- Kalman filter outputs (expensive, compute once) ---
        if name.startswith("kalman_"):
            return self._compute_kalman_features().get(name)
        
        raise ValueError(f"Unknown feature: {name}")
    
    def _compute_kalman_features(self) -> Dict[str, pd.Series]:
        """Compute all Kalman filter features at once (they share state)."""
        from quant_platform.core.signals.time_series.kalman_state_space import _run_kalman_on_panel
        
        # Check if already cached
        if "kalman_filtered" in self._cache:
            return {
                "kalman_filtered": self._cache["kalman_filtered"],
                "kalman_velocity": self._cache["kalman_velocity"],
                "kalman_gain": self._cache["kalman_gain"],
            }
        
        # Use panel-level function for efficiency
        # Returns: (df, filtered, velocity, kg, raw_prices)
        df_sorted, filtered, velocity, kg, raw_prices = _run_kalman_on_panel(
            self.prices,
            Q=1e-5,
            R=1e-3,
        )
        
        # Align results back to original prices index
        # df_sorted is sorted by [ticker, date], need to map back
        aligned_filtered = pd.Series(filtered, index=df_sorted.index).reindex(self.prices.index)
        aligned_velocity = pd.Series(velocity, index=df_sorted.index).reindex(self.prices.index)
        aligned_gain = pd.Series(kg, index=df_sorted.index).reindex(self.prices.index)
        
        # Cache all three outputs
        self._cache["kalman_filtered"] = aligned_filtered
        self._cache["kalman_velocity"] = aligned_velocity
        self._cache["kalman_gain"] = aligned_gain
        
        return {
            "kalman_filtered": self._cache["kalman_filtered"],
            "kalman_velocity": self._cache["kalman_velocity"],
            "kalman_gain": self._cache["kalman_gain"],
        }
    
    def clear(self):
        """Clear the cache."""
        self._cache.clear()
        logger.debug("Feature cache cleared")
    
    def stats(self) -> Dict[str, int]:
        """Return cache statistics."""
        return {
            "cached_features": len(self._cache),
            "cache_size_mb": sum(s.memory_usage(deep=True) for s in self._cache.values()) / 1024**2,
        }


def get_feature_dependencies(factor_class) -> set[str]:
    """Infer which cached features a factor needs (heuristic).
    
    This is a simple heuristic based on factor category and lookback.
    More sophisticated dependency tracking could be added later.
    """
    meta = factor_class.meta
    deps = set()
    
    # All cross-sectional factors need basic returns
    if meta.signal_type == "cross_sectional":
        deps.add("ret_1d")
    
    # Category-specific deps
    if meta.category == "momentum":
        if meta.lookback <= 21:
            deps.add("ret_21d")
        elif meta.lookback <= 63:
            deps.add("ret_63d")
        elif meta.lookback <= 126:
            deps.add("ret_126d")
        else:
            deps.add("ret_252d")
    
    if meta.category == "volatility":
        deps.update({"ret_1d", "vol_20d", "vol_60d"})
    
    if meta.category == "liquidity":
        deps.update({"ret_1d", "dollar_vol", "volume_mean_20", "volume_std_60"})
    
    if "kalman" in meta.name.lower():
        deps.update({"kalman_filtered", "kalman_velocity", "kalman_gain"})
    
    return deps
