"""Effective signal construction shared by research and portfolio code.

The raw factor value is not the tradable signal.  This module converts raw
factor columns into a common "higher is better" representation:

direction -> winsorize -> neutralize -> z-score -> rank
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional, Tuple

import pandas as pd

from quant_platform.core.signals.orientation import apply_direction
from quant_platform.core.signals.transforms import (
    cross_sectional_rank,
    cross_sectional_zscore,
    handle_missing,
    neutralize,
    winsorize,
)


@dataclass(frozen=True)
class EffectiveSignalConfig:
    """Configuration for transforming raw factor values into effective signals."""

    direction: int = 1
    winsorize_limits: Optional[Tuple[float, float]] = (0.01, 0.99)
    neutralize_by: Iterable[str] = field(default_factory=tuple)
    zscore: bool = True
    rank: bool = True
    fillna_method: str = "drop"


def build_effective_signal(
    df: pd.DataFrame,
    signal_col: str,
    *,
    output_col: Optional[str] = None,
    config: Optional[EffectiveSignalConfig] = None,
    direction: Optional[int] = None,
    winsorize_limits: Optional[Tuple[float, float]] = None,
    neutralize_by: Optional[Iterable[str]] = None,
    zscore: Optional[bool] = None,
    rank: Optional[bool] = None,
    fillna_method: Optional[str] = None,
) -> pd.DataFrame:
    """Return *df* with an effective signal column.

    Parameters explicitly passed to this function override ``config``.
    The input must include ``date`` and the raw ``signal_col``.  Extra columns
    needed for neutralization are preserved.
    """
    cfg = config or EffectiveSignalConfig()
    out_col = output_col or signal_col
    result = df.copy()

    eff_direction = cfg.direction if direction is None else direction
    eff_winsor = cfg.winsorize_limits if winsorize_limits is None else winsorize_limits
    eff_neutralize = tuple(cfg.neutralize_by if neutralize_by is None else neutralize_by)
    eff_zscore = cfg.zscore if zscore is None else zscore
    eff_rank = cfg.rank if rank is None else rank
    eff_fill = cfg.fillna_method if fillna_method is None else fillna_method

    result[out_col] = apply_direction(result[signal_col], eff_direction)
    result = handle_missing(result, out_col, method=eff_fill)

    if eff_winsor is not None:
        lo, hi = eff_winsor
        if lo > 0.0 or hi < 1.0:
            result = winsorize(result, out_col, limits=eff_winsor)

    if eff_neutralize:
        result = neutralize(result, out_col, group_cols=list(eff_neutralize))

    if eff_zscore:
        result = cross_sectional_zscore(result, out_col)

    if eff_rank:
        result = cross_sectional_rank(result, out_col)

    return result


def build_effective_factor_frame(
    factor_df: pd.DataFrame,
    direction_map: dict[str, int],
    *,
    factor_cols: Optional[Iterable[str]] = None,
    neutralize_by: Optional[Iterable[str]] = None,
    suffix: str = "",
) -> pd.DataFrame:
    """Apply :func:`build_effective_signal` to many factor columns."""
    result = factor_df.copy()
    cols = list(factor_cols) if factor_cols is not None else [
        c for c in result.columns if c not in ("date", "ticker")
    ]

    for col in cols:
        if col not in result.columns:
            continue
        out_col = f"{col}{suffix}" if suffix else col
        eff = build_effective_signal(
            result,
            col,
            output_col=out_col,
            direction=direction_map.get(col, 1),
            neutralize_by=neutralize_by,
        )
        result[out_col] = eff.set_index(["date", "ticker"])[out_col].reindex(
            result.set_index(["date", "ticker"]).index
        ).values

    return result
