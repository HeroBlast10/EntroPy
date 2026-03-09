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
        **kwargs,
    ) -> pd.DataFrame:
        """Compute all (or selected) factors and return a wide DataFrame.

        Returns a DataFrame with columns ``[date, ticker, F1, F2, …]``.
        """
        names = factor_names or sorted(self._registry.keys())
        results: List[pd.DataFrame] = []

        for name in names:
            cls = self.get(name)
            try:
                factor_df = cls().compute(prices, fundamentals, **kwargs)
                results.append(factor_df)
            except Exception as exc:
                logger.error("Factor {} failed: {}", name, exc)
                continue

        if not results:
            raise RuntimeError("All factor computations failed.")

        # Merge on (date, ticker)
        merged = results[0]
        for extra in results[1:]:
            merged = merged.merge(extra, on=["date", "ticker"], how="outer")

        merged.sort_values(["date", "ticker"], inplace=True)
        merged.reset_index(drop=True, inplace=True)

        logger.info("Computed {} factors → {} rows × {} cols",
                     len(results), len(merged), len(merged.columns) - 2)
        return merged

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
