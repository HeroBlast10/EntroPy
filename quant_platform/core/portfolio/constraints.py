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

from quant_platform.core.portfolio.construction import PortfolioConfig, PortfolioMode

# A few tickers (typically names since removed from the index) may be missing
# from the GICS source. They are grouped into one bucket so the sector cap
# still applies; above this share the map is treated as broken and the run
# fails loudly rather than silently concentrating risk.
UNKNOWN_SECTOR = "Unknown"
MAX_UNKNOWN_SECTOR_SHARE = 0.20


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
    return _project_weights(
        weights,
        config=config,
        sector_map=sector_map,
        prev_weights=prev_weights,
    )


def _project_weights(
    weights: pd.Series,
    *,
    config: PortfolioConfig,
    sector_map: Optional[pd.DataFrame],
    prev_weights: Optional[pd.Series],
) -> pd.Series:
    """Project raw weights onto all configured constraints at once."""
    from scipy.optimize import minimize

    target = pd.to_numeric(weights, errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    ).dropna()
    if target.index.has_duplicates:
        target = target.groupby(level=0).sum()
    if target.empty:
        return pd.Series(dtype=float)

    if config.max_turnover is not None and prev_weights is not None:
        index = target.index.union(prev_weights.index)
    else:
        index = target.index
    target = target.reindex(index, fill_value=0.0).astype(float)
    prev = (
        pd.to_numeric(prev_weights, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .reindex(index, fill_value=0.0)
        if prev_weights is not None
        else pd.Series(0.0, index=index)
    )

    n = len(index)
    max_w = float(config.max_stock_weight)
    min_w = max(float(config.min_stock_weight), 0.0)
    if max_w <= 0:
        raise ValueError("max_stock_weight must be positive")

    constraints = []
    if config.mode == PortfolioMode.LONG_ONLY:
        if n * max_w < 1.0 - 1e-10:
            raise ValueError(
                f"Infeasible long-only stock cap: {n} names * {max_w:.4f} < 1"
            )
        lower = np.where(target.to_numpy() > 0, min_w, 0.0)
        bounds = list(zip(lower, np.full(n, max_w)))
        constraints.append({"type": "eq", "fun": lambda x: np.sum(x) - 1.0})
        x0 = _bounded_sleeve_initial(target.clip(lower=0.0).to_numpy(), max_w)
    else:
        sign = np.sign(target.to_numpy())
        prev_sign = np.sign(prev.to_numpy())
        sign = np.where(sign == 0, prev_sign, sign)
        long_mask = sign > 0
        short_mask = sign < 0
        if long_mask.sum() * max_w < 1.0 - 1e-10:
            raise ValueError("Infeasible long sleeve under max_stock_weight")
        if short_mask.sum() * max_w < 1.0 - 1e-10:
            raise ValueError("Infeasible short sleeve under max_stock_weight")

        bounds = []
        for i in range(n):
            selected = abs(target.iloc[i]) > 1e-12
            floor = min_w if selected else 0.0
            if long_mask[i]:
                bounds.append((floor, max_w))
            elif short_mask[i]:
                bounds.append((-max_w, -floor))
            else:
                bounds.append((0.0, 0.0))
        constraints.extend([
            {
                "type": "eq",
                "fun": lambda x, mask=long_mask: np.sum(x[mask]) - 1.0,
            },
            {
                "type": "eq",
                "fun": lambda x, mask=short_mask: np.sum(x[mask]) + 1.0,
            },
        ])
        x0 = np.zeros(n, dtype=float)
        x0[long_mask] = _bounded_sleeve_initial(
            np.maximum(target.to_numpy()[long_mask], 0.0), max_w
        )
        x0[short_mask] = -_bounded_sleeve_initial(
            np.maximum(-target.to_numpy()[short_mask], 0.0), max_w
        )

    if config.max_sector_weight is not None and config.max_sector_weight > 0:
        if sector_map is None or sector_map.empty:
            raise ValueError(
                "max_sector_weight is enabled but no sector_map was supplied"
            )
        duplicate_sectors = (
            sector_map[["ticker", "sector"]]
            .dropna(subset=["ticker"])
            .groupby("ticker")["sector"]
            .nunique(dropna=False)
        )
        if (duplicate_sectors > 1).any():
            raise ValueError("sector_map contains conflicting sectors for a ticker")
        sector_lookup = (
            sector_map[["ticker", "sector"]]
            .drop_duplicates("ticker")
            .set_index("ticker")["sector"]
        )
        sectors = pd.Series(index=index, data=index.map(sector_lookup))
        missing_sector = sectors[sectors.isna()].index.tolist()
        if missing_sector:
            preview = ", ".join(map(str, missing_sector[:5]))
            if getattr(config, "missing_sector_policy", "fail") != "unknown_bucket":
                # Default: fail closed. An incomplete map is an error, never a
                # silently concentrated portfolio.
                raise ValueError(
                    "Missing sector mapping while max_sector_weight is enabled: "
                    f"{preview}"
                )
            unknown_share = len(missing_sector) / max(len(sectors), 1)
            if unknown_share > MAX_UNKNOWN_SECTOR_SHARE:
                raise ValueError(
                    "Too many tickers lack a sector mapping while "
                    f"max_sector_weight is enabled: {len(missing_sector)} of "
                    f"{len(sectors)} ({unknown_share:.1%}), e.g. {preview}"
                )
            sectors = sectors.fillna(UNKNOWN_SECTOR)
            logger.warning(
                "{} ticker(s) have no GICS sector; grouped as '{}' under the "
                "{:.0%} sector cap: {}",
                len(missing_sector), UNKNOWN_SECTOR,
                config.max_sector_weight, preview,
            )
        for sector in sectors.unique():
            mask = sectors.eq(sector).to_numpy()
            constraints.append({
                "type": "ineq",
                "fun": lambda x, mask=mask, cap=float(config.max_sector_weight): (
                    cap - np.sum(np.abs(x[mask]))
                ),
            })

    if config.max_turnover is not None and prev_weights is not None:
        prev_values = prev.to_numpy(dtype=float)
        constraints.append({
            "type": "ineq",
            "fun": lambda x, p=prev_values, cap=float(config.max_turnover): (
                cap - np.sum(np.abs(x - p)) / 2.0
            ),
        })

    target_values = target.to_numpy(dtype=float)
    result = minimize(
        lambda x: 0.5 * np.sum((x - target_values) ** 2),
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-12},
    )
    if not result.success:
        raise ValueError(f"Portfolio constraints are infeasible: {result.message}")

    projected = pd.Series(result.x, index=index)
    projected = projected[projected.abs() > 1e-10]
    _verify_projected_weights(
        projected,
        config=config,
        sector_map=sector_map,
        prev_weights=prev_weights,
    )
    return projected


def _bounded_sleeve_initial(values: np.ndarray, cap: float) -> np.ndarray:
    """Build a non-negative, sum-one starting point under an upper bound."""
    values = np.asarray(values, dtype=float)
    n = len(values)
    if n == 0 or n * cap < 1.0 - 1e-10:
        raise ValueError("No feasible bounded sleeve")
    values = np.maximum(values, 0.0)
    if values.sum() <= 0:
        values = np.ones(n, dtype=float)

    out = np.zeros(n, dtype=float)
    remaining = 1.0
    active = np.ones(n, dtype=bool)
    scores = values.copy()
    while remaining > 1e-12 and active.any():
        active_idx = np.flatnonzero(active)
        active_scores = scores[active]
        proposal = (
            np.full(active.sum(), remaining / active.sum())
            if active_scores.sum() <= 0
            else remaining * active_scores / active_scores.sum()
        )
        room = cap - out[active_idx]
        hit = proposal >= room - 1e-12
        if not hit.any():
            out[active_idx] += proposal
            remaining = 0.0
            break
        hit_idx = active_idx[hit]
        allocated = room[hit]
        out[hit_idx] += allocated
        remaining -= allocated.sum()
        active[hit_idx] = False
    if remaining > 1e-8:
        raise ValueError("Unable to construct a feasible bounded sleeve")
    return out


def _verify_projected_weights(
    weights: pd.Series,
    *,
    config: PortfolioConfig,
    sector_map: Optional[pd.DataFrame],
    prev_weights: Optional[pd.Series],
    tolerance: float = 1e-6,
) -> None:
    if (weights.abs() > config.max_stock_weight + tolerance).any():
        raise ValueError("Projected portfolio violates max_stock_weight")
    if config.mode == PortfolioMode.LONG_ONLY:
        if abs(weights.sum() - 1.0) > tolerance or (weights < -tolerance).any():
            raise ValueError("Projected long-only portfolio is not fully invested")
    else:
        if abs(weights[weights > 0].sum() - 1.0) > tolerance:
            raise ValueError("Projected long sleeve does not sum to one")
        if abs(weights[weights < 0].sum() + 1.0) > tolerance:
            raise ValueError("Projected short sleeve does not sum to minus one")

    if config.max_sector_weight is not None and config.max_sector_weight > 0:
        if sector_map is None or sector_map.empty:
            raise ValueError(
                "max_sector_weight is enabled but no sector_map was supplied"
            )
        sector = (
            sector_map[["ticker", "sector"]]
            .drop_duplicates("ticker")
            .set_index("ticker")["sector"]
        )
        grouped = weights.to_frame("weight").join(sector)
        # Mirrors the projection above: an incomplete map either already failed
        # there, or was bucketed as Unknown — re-raising here would contradict
        # the projection instead of catching a real violation.
        if getattr(config, "missing_sector_policy", "fail") == "unknown_bucket":
            grouped["sector"] = grouped["sector"].fillna(UNKNOWN_SECTOR)
        elif grouped["sector"].isna().any():
            raise ValueError(
                "Projected portfolio contains tickers without a sector mapping"
            )
        gross = grouped.groupby("sector")["weight"].apply(lambda x: x.abs().sum())
        if (gross > config.max_sector_weight + tolerance).any():
            raise ValueError("Projected portfolio violates max_sector_weight")

    if config.max_turnover is not None and prev_weights is not None:
        idx = weights.index.union(prev_weights.index)
        turnover = (
            weights.reindex(idx, fill_value=0.0)
            - prev_weights.reindex(idx, fill_value=0.0)
        ).abs().sum() / 2.0
        if turnover > config.max_turnover + tolerance:
            raise ValueError("Projected portfolio violates max_turnover")
