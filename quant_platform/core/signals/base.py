"""Abstract base class for all factors.

Every factor inherits from :class:`FactorBase` and implements:
- ``_compute(prices, fundamentals) → Series`` — the raw signal.
- Class-level metadata: ``name``, ``category``, ``lag``, ``lookback``.

The base class provides the full pipeline::

    raw signal → lag → winsorize → zscore → neutralize → final

so each concrete factor only has to define *what* to compute, not *how*
to post-process.
"""

from __future__ import annotations

import abc
import dataclasses
import inspect
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger


# ===================================================================
# Metadata descriptor
# ===================================================================

@dataclass(frozen=True)
class FactorMeta:
    """Immutable metadata attached to every factor.

    signal_type classifies the factor into one of four categories:
    - ``"cross_sectional"``: traditional CS factors (momentum, vol, liquidity)
    - ``"time_series"``: per-asset latent-state features (Kalman, entropy, Hurst)
    - ``"regime"``: overlay signals that modulate portfolio exposure (HMM)
    - ``"relative_value"``: pair/spread strategies (OU process)
    """

    name: str                       # unique identifier, e.g. "MOM_12_1M"
    category: str                   # "momentum" | "volatility" | "liquidity" | etc.
    signal_type: str = "cross_sectional"  # "cross_sectional" | "time_series" | "regime" | "relative_value"
    description: str = ""           # one-liner for docs / tearsheet titles
    lookback: int = 252             # trading days of history required
    lag: int = 1                    # trading days to skip (prevent look-ahead)
    direction: int = 1             # +1 = higher is better, −1 = lower is better
    references: List[str] = field(default_factory=list)  # academic paper refs


# ===================================================================
# Base class
# ===================================================================

class FactorBase(abc.ABC):
    """Abstract base class for all alpha factors.

    Subclasses must set the class attribute ``meta`` (:class:`FactorMeta`)
    and implement :meth:`_compute`.
    """

    meta: FactorMeta  # must be set by subclass

    def __init__(self, **param_overrides: Any) -> None:
        """Initialise factor, optionally overriding meta fields or extra params.

        Parameters
        ----------
        **param_overrides :
            Any :class:`FactorMeta` field (``lookback``, ``lag``, ``name``, …)
            overrides the frozen meta dataclass for this instance only.
            Any *other* key is stored in ``self._extra_params`` and is
            available to concrete ``_compute`` implementations.

        Examples
        --------
        >>> Mom1M(period=42)          # extra param read in _compute
        >>> Mom1M(lag=2)              # overrides meta.lag (applied by base)
        >>> Mom1M(lookback=43, period=42)
        """
        meta_field_names = {f.name for f in dataclasses.fields(self.__class__.meta)}
        meta_kwargs = {k: v for k, v in param_overrides.items() if k in meta_field_names}
        self._extra_params: Dict[str, Any] = {
            k: v for k, v in param_overrides.items() if k not in meta_field_names
        }
        if meta_kwargs:
            self._meta: FactorMeta = dataclasses.replace(self.__class__.meta, **meta_kwargs)
        else:
            self._meta = self.__class__.meta

    # ------------------------------------------------------------------
    # Abstract
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def _compute(
        self,
        prices: pd.DataFrame,
        fundamentals: Optional[pd.DataFrame] = None,
        **kwargs,
    ) -> pd.Series:
        """Return raw factor values indexed like *prices* ``(date, ticker)``.

        Must return a ``pd.Series`` with the same index as *prices*
        (or a subset).  The series name should be ``self.meta.name``.
        """
        ...

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------

    def compute(
        self,
        prices: pd.DataFrame,
        fundamentals: Optional[pd.DataFrame] = None,
        *,
        winsorize_limits: Tuple[float, float] = (0.01, 0.99),
        zscore: bool = True,
        neutralize_by: Optional[List[str]] = None,
        fillna_method: str = "drop",
        **kwargs,
    ) -> pd.DataFrame:
        """Run the full factor pipeline: compute → lag → clean → transform.

        Parameters
        ----------
        prices : DataFrame with columns from the ``prices`` schema.
        fundamentals : optional fundamentals DataFrame.
        winsorize_limits : lower/upper quantile clipping bounds.
        zscore : if True, cross-sectional z-score each date.
        neutralize_by : columns in *prices* to neutralize against
            (e.g. ``["sector"]`` for industry neutralization, or
            ``["log_mcap"]`` for size, or both).
        fillna_method : ``"drop"`` | ``"zero"`` | ``"median"``.

        Returns
        -------
        DataFrame with columns ``[date, ticker, <factor_name>]``.
        """
        from quant_platform.core.signals.transforms import (
            apply_lag,
            cross_sectional_zscore,
            handle_missing,
            neutralize,
            winsorize,
        )

        name = self._meta.name
        logger.debug("Computing factor: {}", name)

        # 1. Raw signal
        raw = self._call_compute(prices, fundamentals, **kwargs)
        if isinstance(raw, pd.Series):
            raw.name = name

        # Build a working DataFrame
        if "date" in prices.columns and "ticker" in prices.columns:
            df = prices[["date", "ticker"]].copy()
            df[name] = raw.values if len(raw) == len(df) else raw.reindex(df.index).values
        else:
            df = raw.reset_index()
            df.columns = list(df.columns[:-1]) + [name]

        # 2. Lag
        if self._meta.lag > 0:
            df = apply_lag(df, name, lag=self._meta.lag)

        # 3. Missing values (first pass — before transforms)
        df = handle_missing(df, name, method=fillna_method)

        if df.empty:
            logger.warning("Factor {} produced an empty DataFrame after missing-value handling", name)
            return df

        # 4. Winsorize
        df = winsorize(df, name, limits=winsorize_limits)

        # 5. Z-score
        if zscore:
            df = cross_sectional_zscore(df, name)

        # 6. Neutralize
        if neutralize_by:
            df = neutralize(df, name, group_cols=neutralize_by)

        logger.info("Factor {}: {} rows, date range {} – {}",
                     name, len(df), df["date"].min(), df["date"].max())
        return df[["date", "ticker", name]]

    def _call_compute(
        self,
        prices: pd.DataFrame,
        fundamentals: Optional[pd.DataFrame] = None,
        **kwargs,
    ) -> pd.Series:
        """Dispatch to ``_compute`` while only forwarding supported kwargs."""
        signature = inspect.signature(self._compute)
        if any(
            param.kind == inspect.Parameter.VAR_KEYWORD
            for param in signature.parameters.values()
        ):
            return self._compute(prices, fundamentals, **kwargs)

        supported_kwargs = {
            key: value for key, value in kwargs.items()
            if key in signature.parameters
        }
        return self._compute(prices, fundamentals, **supported_kwargs)

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        m = self._meta
        extra = f" params={self._extra_params}" if self._extra_params else ""
        return f"<Factor {m.name} [{m.category}] lookback={m.lookback} lag={m.lag}{extra}>"
