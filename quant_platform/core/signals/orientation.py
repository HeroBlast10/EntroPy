"""Helpers for applying factor direction consistently across the stack.

`FactorMeta.direction` encodes whether higher raw values are good (`+1`)
or bad (`-1`).  Downstream portfolio construction and most evaluation
pipelines expect "higher = better", so we expose small helpers that flip
negative-direction signals into a common oriented representation.
"""

from __future__ import annotations

from typing import Iterable, Optional

import pandas as pd


def apply_direction(values: pd.Series, direction: int = 1) -> pd.Series:
    """Return a copy of *values* oriented so that higher is better."""
    direction = -1 if int(direction) < 0 else 1
    return values * direction


def orient_signal_frame(
    df: pd.DataFrame,
    signal_col: str,
    direction: int = 1,
    *,
    output_col: Optional[str] = None,
) -> pd.DataFrame:
    """Add or overwrite an oriented signal column on *df*."""
    out_col = output_col or signal_col
    result = df.copy()
    result[out_col] = apply_direction(result[signal_col], direction)
    return result


def orient_factor_columns(
    df: pd.DataFrame,
    direction_map: dict[str, int],
    *,
    columns: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """Orient multiple factor columns using a ``{factor: direction}`` map."""
    result = df.copy()
    target_cols = list(columns) if columns is not None else [
        c for c in result.columns if c in direction_map
    ]
    for col in target_cols:
        result[col] = apply_direction(result[col], direction_map.get(col, 1))
    return result
