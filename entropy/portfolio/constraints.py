"""Portfolio constraint enforcement.

Constraints are applied as post-processing after raw weight generation.
The pipeline is:

    raw_weights → clip_stock → clip_sector → clip_turnover → re-normalise

Each step respects the portfolio mode (long-only vs long-short).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

from entropy.portfolio.construction import PortfolioConfig, PortfolioMode


# ===================================================================
# Individual constraint functions
# ===================================================================

def clip_stock_weight(
    weights: pd.Series,
    max_weight: float,
    min_weight: float = 0.0,
    mode: PortfolioMode = PortfolioMode.LONG_ONLY,
) -> pd.Series:
    """Clip individual stock weights to [min_weight, max_weight].

    For long-short, the clip is applied symmetrically:
    short positions are clipped to [−max_weight, −min_weight].
    """
    w = weights.copy()

    if mode == PortfolioMode.LONG_ONLY:
        w = w.clip(lower=min_weight, upper=max_weight)
    else:
        longs = w[w > 0].clip(lower=min_weight, upper=max_weight)
        shorts = w[w < 0].clip(lower=-max_weight, upper=-min_weight if min_weight > 0 else 0)
        w = pd.concat([longs, shorts])

    return w


def clip_sector_weight(
    weights: pd.Series,
    max_sector_weight: float,
    sector_map: Optional[pd.DataFrame] = None,
) -> pd.Series:
    """Cap aggregate weight per sector.

    If a sector exceeds *max_sector_weight*, all stocks in that sector
    are scaled down proportionally to fit within the cap.

    Parameters
    ----------
    weights : indexed by ticker.
    max_sector_weight : e.g. 0.30 for 30 %.
    sector_map : ``DataFrame`` with columns ``[ticker, sector]``.
    """
    if sector_map is None or sector_map.empty:
        return weights

    w = weights.copy()
    sec = sector_map.set_index("ticker")["sector"]

    # Map tickers to sectors (tickers not in map are left unconstrained)
    mapped = w.to_frame("weight").join(sec, how="left")
    mapped["sector"] = mapped["sector"].fillna("_unknown_")

    for sector, grp in mapped.groupby("sector"):
        if sector == "_unknown_":
            continue
        sector_total = grp["weight"].abs().sum()
        if sector_total > max_sector_weight:
            scale = max_sector_weight / sector_total
            w.loc[grp.index] = w.loc[grp.index] * scale
            logger.debug("Sector {} capped: {:.2%} → {:.2%}", sector, sector_total, max_sector_weight)

    return w


def clip_turnover(
    weights: pd.Series,
    prev_weights: Optional[pd.Series],
    max_turnover: float,
) -> pd.Series:
    """Limit single-period turnover.

    Turnover = sum(|w_new − w_old|) / 2.
    If turnover exceeds *max_turnover*, blend new weights toward old
    weights until the constraint is satisfied.

    Parameters
    ----------
    weights : new target weights.
    prev_weights : weights from the previous period.
    max_turnover : e.g. 0.30 for 30 % one-way.
    """
    if prev_weights is None or prev_weights.empty:
        return weights

    # Align indices
    all_tickers = weights.index.union(prev_weights.index)
    w_new = weights.reindex(all_tickers, fill_value=0.0)
    w_old = prev_weights.reindex(all_tickers, fill_value=0.0)

    turnover = (w_new - w_old).abs().sum() / 2.0

    if turnover <= max_turnover:
        return weights

    # Blend: w_final = α * w_new + (1 − α) * w_old, choose α so turnover = max
    # turnover(α) = α * turnover(w_new, w_old)  (linear in α)
    alpha = max_turnover / turnover
    w_blended = alpha * w_new + (1 - alpha) * w_old

    # Drop near-zero
    w_blended = w_blended[w_blended.abs() > 1e-10]

    actual_to = (w_blended - w_old.reindex(w_blended.index, fill_value=0.0)).abs().sum() / 2.0
    logger.debug("Turnover capped: {:.2%} → {:.2%} (α={:.3f})", turnover, actual_to, alpha)

    return w_blended


# ===================================================================
# Combined constraint pipeline
# ===================================================================

def apply_constraints(
    weights: pd.Series,
    config: PortfolioConfig,
    sector_map: Optional[pd.DataFrame] = None,
    prev_weights: Optional[pd.Series] = None,
) -> pd.Series:
    """Apply all constraints in sequence, then re-normalise.

    Order: stock clip → sector clip → turnover clip → normalise.
    """
    from entropy.portfolio.construction import PortfolioConstructor

    w = weights.copy()

    # 1. Stock-level clip
    w = clip_stock_weight(w, config.max_stock_weight, config.min_stock_weight, config.mode)

    # 2. Sector-level clip
    if sector_map is not None:
        w = clip_sector_weight(w, config.max_sector_weight, sector_map)

    # 3. Turnover clip
    if config.max_turnover is not None and prev_weights is not None:
        w = clip_turnover(w, prev_weights, config.max_turnover)

    # 4. Re-normalise
    w = PortfolioConstructor.normalise_weights(w, config.mode)

    # 5. Drop negligible positions
    w = w[w.abs() > 1e-10]

    return w
