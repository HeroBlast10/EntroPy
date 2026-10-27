"""Causal score smoothing and buffered, issuer-capped selection.

The thresholds are fixed research assumptions, never fitted to replay PnL.
Overlapping books use one independent selection state per sleeve. Execution
and mark-to-market of the sleeves belong to the holdings simulator.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from quant_platform.core.alpha_models.fair_comparison import issuer_capped_weights


@dataclass(frozen=True)
class TurnoverPolicy:
    entry_fraction: float = 0.20
    exit_fraction: float = 0.20
    ema_halflife: float | None = None
    tranches: int = 1
    rebalance_sessions: int = 1
    tolerance: float = 0.0

    def __post_init__(self):
        if not 0 < self.entry_fraction <= self.exit_fraction < 0.5:
            raise ValueError("Require 0 < entry_fraction <= exit_fraction < 0.5")
        if self.ema_halflife is not None and (
            not np.isfinite(self.ema_halflife) or self.ema_halflife <= 0
        ):
            raise ValueError("ema_halflife must be finite and positive")
        for name in ("tranches", "rebalance_sessions"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.tranches > 1 and self.rebalance_sessions != 1:
            raise ValueError("Overlapping tranches require a daily decision schedule")
        if not np.isfinite(self.tolerance) or not 0 <= self.tolerance < 1:
            raise ValueError("tolerance must be in [0, 1)")


def smooth_scores(rows: pd.DataFrame, score: str, halflife: float | None) -> pd.Series:
    """Trading-session EMA; missing scores stay missing and reset the state.

    Re-entry after a gap in the prediction universe also starts a new state.
    No labels or future observations are read, including at fold boundaries.
    """
    if rows.duplicated(["date", "ticker"]).any():
        raise ValueError("Duplicate score keys")
    values = pd.to_numeric(rows[score], errors="coerce").replace([np.inf, -np.inf], np.nan)
    if halflife is None:
        return values
    if not np.isfinite(halflife) or halflife <= 0:
        raise ValueError("halflife must be finite and positive")
    calendar = pd.DatetimeIndex(sorted(pd.to_datetime(rows.date).unique()))
    work = rows[["date", "ticker"]].copy()
    work["value"] = values
    work["session"] = calendar.get_indexer(pd.to_datetime(work.date))
    work = work.sort_values(["ticker", "session"])
    result = pd.Series(np.nan, index=rows.index, dtype=float)
    for _, group in work.groupby("ticker", sort=False):
        reset = group.session.diff().ne(1) | group.value.shift().isna() | group.value.isna()
        segments = reset.cumsum()
        result.loc[group.index] = group.groupby(segments, sort=False).value.transform(
            lambda s: s.ewm(halflife=halflife, adjust=False).mean()
        ).where(group.value.notna())
    return result


def buffered_weights(day, score, previous, policy, execution):
    """Enter the outer quintile; retain incumbents until outside the exit band.

    Average ranks preserve the historical treatment of tied predictions.
    Missing scores and universe exits never retain an incumbent.
    """
    if policy.entry_fraction == policy.exit_fraction == 0.2:
        return issuer_capped_weights(
            day, score, min_names=execution["min_names"],
            stock_cap=execution["max_stock_weight"],
            issuer_cap=execution["max_issuer_gross_weight"],
        ) * execution.get("gross_target", 1.0)
    if day[score].notna().sum() < execution["min_names"]:
        return pd.Series(0.0, index=day.index)
    rank = day[score].rank(pct=True)
    prior = day.ticker.map(previous).fillna(0.0)
    long = rank.ge(1 - policy.entry_fraction) | (prior.gt(0) & rank.ge(1 - policy.exit_fraction))
    short = rank.le(policy.entry_fraction) | (prior.lt(0) & rank.le(policy.exit_fraction))
    weights = 0.5 * (long.astype(float) / max(int(long.sum()), 1)
                     - short.astype(float) / max(int(short.sum()), 1))
    weights *= execution.get("gross_target", 1.0)
    weights = weights.clip(-execution["max_stock_weight"], execution["max_stock_weight"])
    issuer = day.permco.fillna(-day.permno)
    gross = weights.abs().groupby(issuer).transform("sum")
    weights *= np.minimum(1, execution["max_issuer_gross_weight"] / gross.replace(0, np.nan)).fillna(1)
    pos, neg = weights.clip(lower=0).sum(), -weights.clip(upper=0).sum()
    sleeve = min(pos, neg)
    weights.loc[weights.gt(0)] *= sleeve / pos if pos else 0
    weights.loc[weights.lt(0)] *= sleeve / neg if neg else 0
    return weights


def transition_sources(previous, current, old_rank, rank, *, boundary_fraction=0.30):
    """An operational boundary proxy, not a causal claim about noise.

    A side change is 'boundary' only when both observed ranks remain inside
    that side's outer 30% band. All other side changes are signal/universe
    changes. Weight changes with the same sign are resizing.
    """
    source = pd.Series("drift_resize", index=current.index)
    side_change = np.sign(previous).ne(np.sign(current))
    source.loc[side_change] = "signal"
    long_transition = previous.ge(0) & current.ge(0) & side_change
    short_transition = previous.le(0) & current.le(0) & side_change
    boundary = ((long_transition & old_rank.ge(1 - boundary_fraction) & rank.ge(1 - boundary_fraction))
                | (short_transition & old_rank.le(boundary_fraction) & rank.le(boundary_fraction)))
    source.loc[boundary] = "boundary"
    return source


def build_turnover_targets(rows, score, execution, policy):
    """Return daily full-universe targets plus auditable score/rank metadata.

    For tranches these are the selected sleeve's *unscaled* targets. The
    simulator allocates 1/H of current NAV and leaves other sleeves to drift.
    """
    work = rows.copy()
    work["date"] = pd.to_datetime(work.date)
    work["selection_score"] = smooth_scores(work, score, policy.ema_halflife)
    work["selection_rank"] = work.groupby("date").selection_score.rank(pct=True)
    states = [pd.Series(dtype=float) for _ in range(policy.tranches)]
    ranks = [pd.Series(dtype=float) for _ in range(policy.tranches)]
    frames = []
    last_weights = pd.Series(dtype=float)
    for number, (date, day) in enumerate(work.groupby("date", sort=True)):
        active = number % policy.tranches
        rebalance = number % policy.rebalance_sessions == 0
        if rebalance:
            weights = buffered_weights(day, "selection_score", states[active], policy, execution)
            current = pd.Series(weights.to_numpy(), index=day.ticker)
            # Include old names so universe/missing-score exits are explicit.
            idx = current.index.union(states[active].index)
            prior = states[active].reindex(idx, fill_value=0.0)
            current = current.reindex(idx, fill_value=0.0)
            new_rank = pd.Series(day.selection_rank.to_numpy(), index=day.ticker).reindex(idx)
            sources = transition_sources(prior, current, ranks[active].reindex(idx), new_rank)
            frame = pd.DataFrame({"date": date, "ticker": idx, "weight": current.to_numpy(),
                                  "transition_source": sources.to_numpy()})
            states[active], ranks[active] = current, new_rank
            if number == 0:
                states = [current.copy() for _ in states]
                ranks = [new_rank.copy() for _ in ranks]
            last_weights = current
        else:
            frame = pd.DataFrame({"date": date, "ticker": last_weights.index,
                                  "weight": last_weights.to_numpy(), "transition_source": "drift_resize"})
        frame["is_rebalance"] = rebalance
        # Only today's prediction universe can keep an inactive tranche name.
        frame["eligible"] = frame.ticker.isin(day.ticker)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True), work
