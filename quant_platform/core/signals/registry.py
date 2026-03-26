"""Factor registry: auto-discovery, batch computation, and persistence.

Usage::

    from quant_platform.core.signals.registry import FactorRegistry

    reg = FactorRegistry()
    reg.discover()                      # auto-register all built-in factors
    reg.list_factors()                  # show available factors
    results = reg.compute_all(prices)   # compute everything in one call
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Type

import pandas as pd
from loguru import logger

from quant_platform.core.signals.base import FactorBase
from quant_platform.core.signals.feature_cache import PriceFeatureCache
from quant_platform.core.utils.io import load_config, resolve_data_path, save_parquet


class FactorRegistry:
    """Central registry for all alpha factors."""

    def __init__(self) -> None:
        self._registry: Dict[str, Type[FactorBase]] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, cls: Type[FactorBase]) -> None:
        name = cls.meta.name
        if name in self._registry:
            logger.warning("Factor {} already registered — overwriting", name)
        self._registry[name] = cls
        logger.debug("Registered factor: {} [{}]", name, cls.meta.category)

    def discover(self) -> None:
        """Auto-register all built-in factor classes across 4 signal types."""
        # Cross-sectional factors
        from quant_platform.core.signals.cross_sectional.momentum import ALL_MOMENTUM_FACTORS
        from quant_platform.core.signals.cross_sectional.volatility import ALL_VOLATILITY_FACTORS
        from quant_platform.core.signals.cross_sectional.liquidity import ALL_LIQUIDITY_FACTORS
        from quant_platform.core.signals.cross_sectional.value_quality import ALL_VALUE_QUALITY_FACTORS

        # Time-series features
        from quant_platform.core.signals.time_series.kalman_state_space import ALL_KALMAN_FACTORS
        from quant_platform.core.signals.time_series.entropy_hurst import ALL_ENTROPY_HURST_FACTORS
        from quant_platform.core.signals.time_series.higher_moments import ALL_HIGHER_MOMENT_FACTORS

        # Regime overlays
        from quant_platform.core.signals.regime.hmm_regime import ALL_REGIME_FACTORS

        # Relative-value
        from quant_platform.core.signals.relative_value.ou_pairs import ALL_OU_FACTORS

        all_factors = (
            ALL_MOMENTUM_FACTORS
            + ALL_VOLATILITY_FACTORS
            + ALL_LIQUIDITY_FACTORS
            + ALL_VALUE_QUALITY_FACTORS
            + ALL_KALMAN_FACTORS
            + ALL_ENTROPY_HURST_FACTORS
            + ALL_HIGHER_MOMENT_FACTORS
            + ALL_REGIME_FACTORS
            + ALL_OU_FACTORS
        )

        for cls in all_factors:
            self.register(cls)

        logger.info("Discovered {} factors across all categories", len(self._registry))

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def list_factors(
        self,
        category: Optional[str] = None,
        signal_type: Optional[str] = None,
    ) -> pd.DataFrame:
        """Return a summary table of all registered factors.

        Parameters
        ----------
        category : filter by category (e.g. ``"momentum"``).
        signal_type : filter by signal type (``"cross_sectional"`` |
            ``"time_series"`` | ``"regime"`` | ``"relative_value"``).
        """
        rows = []
        for name, cls in sorted(self._registry.items()):
            m = cls.meta
            if category and m.category != category:
                continue
            if signal_type and m.signal_type != signal_type:
                continue
            rows.append({
                "name": m.name,
                "category": m.category,
                "signal_type": m.signal_type,
                "description": m.description,
                "lookback": m.lookback,
                "lag": m.lag,
                "direction": m.direction,
            })
        return pd.DataFrame(rows)

    def get(self, name: str) -> Type[FactorBase]:
        if name not in self._registry:
            raise KeyError(f"Factor not found: {name!r}. "
                           f"Available: {sorted(self._registry.keys())}")
        return self._registry[name]

    def __len__(self) -> int:
        return len(self._registry)

    def __contains__(self, name: str) -> bool:
        return name in self._registry

    # ------------------------------------------------------------------
    # Batch computation
    # ------------------------------------------------------------------

    def compute_all(
        self,
        prices: pd.DataFrame,
        fundamentals: Optional[pd.DataFrame] = None,
        factor_names: Optional[List[str]] = None,
        factor_params: Optional[Dict[str, Dict]] = None,
        use_cache: bool = True,
        incremental: bool = False,
        lookback_buffer: int = 300,
        **kwargs,
    ) -> pd.DataFrame:
        """Compute all (or selected) factors and return a wide DataFrame.

        Parameters
        ----------
        prices : price DataFrame.
        fundamentals : optional fundamentals DataFrame.
        factor_names : subset of factors to compute; ``None`` = all.
        factor_params : per-factor constructor overrides, e.g.::

            {
                "MOM_1M":    {"period": 42, "lag": 2},
                "MOM_12_1M": {"period": 189},
            }

            Keys that match :class:`~quant_platform.core.signals.base.FactorMeta`
            fields override the frozen meta for that instance; other keys are
            forwarded to ``_compute`` via ``self._extra_params``.
        use_cache : if True, precompute shared features (10-50x faster).
        incremental : if True, only compute recent data (not yet implemented).
        lookback_buffer : days to recompute in incremental mode.
        **kwargs : forwarded to each factor's :meth:`compute` pipeline.

        Returns a DataFrame with columns ``[date, ticker, F1, F2, …]``.
        """
        names = factor_names or sorted(self._registry.keys())
        
        # Initialize feature cache if enabled
        cache = PriceFeatureCache(prices) if use_cache else None
        if cache:
            logger.info("Feature cache initialized for {} rows", len(prices))
        
        # Create base table from prices
        base = prices[["date", "ticker"]].copy()
        factor_series: Dict[str, pd.Series] = {}
        
        for name in names:
            cls = self.get(name)
            instance_kwargs = (factor_params or {}).get(name, {})
            try:
                instance = cls(**instance_kwargs)
                # Pass cache to compute method via kwargs
                if cache:
                    kwargs["_feature_cache"] = cache
                
                factor_df = instance.compute(prices, fundamentals, **kwargs)
                
                # Extract the factor column (should be named same as factor)
                factor_col = [c for c in factor_df.columns if c not in ["date", "ticker"]]
                if not factor_col:
                    logger.warning("Factor {} returned no value columns", name)
                    continue
                
                # Merge factor values into base
                factor_name = factor_col[0]
                merged_temp = base.merge(
                    factor_df[["date", "ticker", factor_name]],
                    on=["date", "ticker"],
                    how="left"
                )
                factor_series[factor_name] = merged_temp[factor_name]
                
            except Exception as exc:
                logger.error("Factor {} failed: {}", name, exc)
                continue
        
        if not factor_series:
            raise RuntimeError("All factor computations failed.")
        
        # Assemble final DataFrame
        result = base.copy()
        for col_name, series in factor_series.items():
            result[col_name] = series.values
        
        result.sort_values(["date", "ticker"], inplace=True)
        result.reset_index(drop=True, inplace=True)
        
        if cache:
            stats = cache.stats()
            logger.info("Feature cache: {} features cached, {:.1f} MB",
                        stats["cached_features"], stats["cache_size_mb"])
        
        logger.info("Computed {} factors → {} rows × {} cols",
                     len(factor_series), len(result), len(result.columns) - 2)
        return result

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_factors(
        self,
        factor_df: pd.DataFrame,
        output_path: Optional[Path | str] = None,
    ) -> Path:
        """Save the wide factor DataFrame to Parquet."""
        if output_path is None:
            output_path = resolve_data_path("factors", "factors.parquet")
        return save_parquet(factor_df, output_path)

    def load_factors(
        self,
        path: Optional[Path | str] = None,
    ) -> pd.DataFrame:
        """Load a previously saved factor DataFrame."""
        from quant_platform.core.utils.io import load_parquet
        if path is None:
            path = resolve_data_path("factors", "factors.parquet")
        return load_parquet(path)
