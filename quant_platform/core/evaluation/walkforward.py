"""Simple rolling out-of-sample (OOS) check.

Implements a basic rolling train/test scheme for sanity checking:

    |--- train_months ---|--- test_months ---|
                         |--- train_months ---|--- test_months ---|
                                              |--- train_months ---|--- ...

For each fold:
1. Compute IC on the **training** window to validate signal has predictive power.
2. Build a fixed top-quintile long-only portfolio on the **test** window.
3. Record OOS performance metrics.

LIMITATIONS (not a full walk-forward framework):
- Does NOT perform factor selection within each fold
- Does NOT tune parameters (e.g., quantile thresholds, lookback windows)
- Does NOT freeze/retrain models per fold
- Portfolio construction is fixed (top quintile, equal weight)

This is a simplified OOS sanity check to verify the signal has some predictive
power out-of-sample. For production use, extend to include:
- Cross-validated factor selection
- Hyperparameter optimization on validation set
- Model retraining per fold
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from quant_platform.core.signals.effective import build_effective_signal


@dataclass
class WalkForwardConfig:
    """Walk-forward parameters."""
    train_months: int = 36          # training window length
    test_months: int = 12           # out-of-sample window length
    step_months: int = 12           # how far to roll between folds
    min_train_obs: int = 100        # minimum trading days in training set
    factor_select_top_k: Optional[int] = None  # if set, pick top K by IC


# ===================================================================
# Fold generation
# ===================================================================

def generate_folds(
    all_dates: pd.DatetimeIndex,
    config: WalkForwardConfig,
) -> List[Tuple[pd.DatetimeIndex, pd.DatetimeIndex]]:
    """Generate (train_dates, test_dates) folds.

    Returns
    -------
    List of (train_idx, test_idx) tuples, each a DatetimeIndex.
    """
    start = all_dates.min()
    end = all_dates.max()
    folds = []

    cursor = start + pd.DateOffset(months=config.train_months)
    while cursor + pd.DateOffset(months=config.test_months) <= end + pd.Timedelta(days=1):
        train_start = cursor - pd.DateOffset(months=config.train_months)
        train_end = cursor - pd.Timedelta(days=1)
        test_start = cursor
        test_end = cursor + pd.DateOffset(months=config.test_months) - pd.Timedelta(days=1)

        train_idx = all_dates[(all_dates >= train_start) & (all_dates <= train_end)]
        test_idx = all_dates[(all_dates >= test_start) & (all_dates <= test_end)]

        if len(train_idx) >= config.min_train_obs and len(test_idx) > 0:
            folds.append((train_idx, test_idx))

        cursor += pd.DateOffset(months=config.step_months)

    logger.info("Walk-forward: {} folds (train={}m, test={}m, step={}m)",
                len(folds), config.train_months, config.test_months, config.step_months)
    return folds


# ===================================================================
# Single-fold evaluation
# ===================================================================

def _evaluate_fold(
    fold_idx: int,
    train_dates: pd.DatetimeIndex,
    test_dates: pd.DatetimeIndex,
    factor_df: pd.DataFrame,
    prices: pd.DataFrame,
    signal_col: str,
    config: WalkForwardConfig,
) -> Dict[str, object]:
    """Run a single walk-forward fold.

    1. On training period: compute IC to validate signal.
    2. On test period: build simple quantile portfolio, compute OOS return.
    """
    from quant_platform.core.signals.cross_sectional.evaluation import compute_rank_ic_series, ic_summary

    # --- Prepare data ---
    prices = prices.copy()
    prices["date"] = pd.to_datetime(prices["date"])

    # Forward returns for IC calc
    prices_sorted = prices.sort_values(["ticker", "date"])
    prices_sorted["fwd_ret_1d"] = prices_sorted.groupby("ticker")["adj_close"].transform(
        lambda s: s.pct_change().shift(-1)
    )
    directions = _infer_factor_directions([c for c in factor_df.columns if c not in ("date", "ticker")])

    # --- Training period ---
    train_factor = factor_df[factor_df["date"].isin(train_dates)]
    train_merged = train_factor.merge(
        prices_sorted[["date", "ticker", "fwd_ret_1d"]],
        on=["date", "ticker"], how="inner",
    )

    selected_factors = [signal_col]
    train_scores: Dict[str, float] = {}
    if config.factor_select_top_k is not None:
        candidates = [c for c in factor_df.columns if c not in ("date", "ticker")]
        selected_factors, train_scores = _select_top_factors(
            train_merged,
            candidates,
            directions,
            top_k=config.factor_select_top_k,
        )

    train_signal_col = signal_col if len(selected_factors) == 1 else "_wf_signal"
    train_eval = _build_effective_fold_signal(
        train_merged,
        selected_factors,
        directions,
        output_col=train_signal_col,
    )

    # IC during training
    if train_signal_col in train_eval.columns and len(train_eval) > 0:
        train_ic = compute_rank_ic_series(train_eval, train_signal_col, "fwd_ret_1d")
        train_ic_stats = ic_summary(train_ic)
    else:
        train_ic_stats = {"mean_ic": np.nan, "icir": np.nan}

    # --- Test period: simple top-quintile long-only ---
    test_factor = factor_df[factor_df["date"].isin(test_dates)]
    test_prices = prices_sorted[prices_sorted["date"].isin(test_dates)]

    if test_factor.empty or test_prices.empty:
        return {
            "fold": fold_idx,
            "train_start": train_dates[0],
            "train_end": train_dates[-1],
            "test_start": test_dates[0],
            "test_end": test_dates[-1],
            "train_ic": train_ic_stats.get("mean_ic", np.nan),
            "train_icir": train_ic_stats.get("icir", np.nan),
            "oos_return": np.nan,
            "oos_sharpe": np.nan,
            "oos_max_dd": np.nan,
            "selected_factors": ",".join(selected_factors),
        }

    # Build daily equal-weight top-quintile portfolio
    test_factor = _build_effective_fold_signal(
        test_factor,
        selected_factors,
        directions,
        output_col=train_signal_col,
    )
    oos_returns = []
    for dt in sorted(test_dates):
        sig = test_factor.loc[test_factor["date"] == dt, ["ticker", train_signal_col]].dropna()
        if len(sig) < 10:
            continue
        # Top quintile
        threshold = sig[train_signal_col].quantile(0.8)
        longs = sig[sig[train_signal_col] >= threshold]["ticker"]
        # Equal weight return
        day_ret = test_prices.loc[
            (test_prices["date"] == dt) & (test_prices["ticker"].isin(longs)),
            "fwd_ret_1d"
        ]
        if len(day_ret) > 0:
            oos_returns.append({"date": dt, "ret": day_ret.mean()})

    if not oos_returns:
        return {
            "fold": fold_idx,
            "train_start": train_dates[0], "train_end": train_dates[-1],
            "test_start": test_dates[0], "test_end": test_dates[-1],
            "train_ic": train_ic_stats.get("mean_ic", np.nan),
            "train_icir": train_ic_stats.get("icir", np.nan),
            "oos_return": np.nan, "oos_sharpe": np.nan, "oos_max_dd": np.nan,
            "selected_factors": ",".join(selected_factors),
        }

    oos_df = pd.DataFrame(oos_returns).set_index("date")["ret"]
    ann_ret = (1 + oos_df).prod() ** (252 / max(len(oos_df), 1)) - 1
    ann_vol = oos_df.std() * np.sqrt(252) if oos_df.std() > 0 else np.nan
    sharpe = ann_ret / ann_vol if ann_vol and ann_vol > 0 else np.nan
    cum = (1 + oos_df).cumprod()
    max_dd = (cum / cum.cummax() - 1).min()

    return {
        "fold": fold_idx,
        "train_start": train_dates[0],
        "train_end": train_dates[-1],
        "test_start": test_dates[0],
        "test_end": test_dates[-1],
        "train_ic": train_ic_stats.get("mean_ic", np.nan),
        "train_icir": train_ic_stats.get("icir", np.nan),
        "oos_return": round(ann_ret, 6),
        "oos_sharpe": round(sharpe, 4) if np.isfinite(sharpe) else np.nan,
        "oos_max_dd": round(max_dd, 6),
        "selected_factors": ",".join(selected_factors),
        "train_selected_mean_ic": float(np.nanmean(list(train_scores.values()))) if train_scores else train_ic_stats.get("mean_ic", np.nan),
    }


def _select_top_factors(
    train_merged: pd.DataFrame,
    candidates: List[str],
    directions: Dict[str, int],
    top_k: int,
) -> Tuple[List[str], Dict[str, float]]:
    """Select factors by training-window effective RankIC."""
    from quant_platform.core.signals.cross_sectional.evaluation import compute_rank_ic_series

    scores: Dict[str, float] = {}
    for col in candidates:
        if col not in train_merged.columns:
            continue
        eff = build_effective_signal(
            train_merged[["date", "ticker", col, "fwd_ret_1d"]].copy(),
            col,
            direction=directions.get(col, 1),
        )
        if eff.empty:
            continue
        ric = compute_rank_ic_series(eff, col, "fwd_ret_1d")
        score = ric.mean()
        if pd.notna(score):
            scores[col] = float(score)

    if not scores:
        return candidates[:1], {}

    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    selected = [name for name, _ in ranked[: max(1, top_k)]]
    return selected, {name: scores[name] for name in selected}


def _build_effective_fold_signal(
    df: pd.DataFrame,
    factor_cols: List[str],
    directions: Dict[str, int],
    output_col: str,
) -> pd.DataFrame:
    """Build one effective factor or an equal-weight selected-factor composite."""
    if not factor_cols:
        result = df.copy()
        result[output_col] = np.nan
        return result

    result = df.copy()
    base_index = result.set_index(["date", "ticker"]).index
    effective_cols = []
    for col in factor_cols:
        if col not in result.columns:
            continue
        eff_col = f"_eff_{col}"
        eff = build_effective_signal(
            result[["date", "ticker", col]].copy(),
            col,
            output_col=eff_col,
            direction=directions.get(col, 1),
        )
        result[eff_col] = eff.set_index(["date", "ticker"])[eff_col].reindex(base_index).values
        effective_cols.append(eff_col)

    if len(effective_cols) == 1:
        result[output_col] = result[effective_cols[0]]
    elif effective_cols:
        result[output_col] = result[effective_cols].mean(axis=1)
    else:
        result[output_col] = np.nan
    return result


def _infer_factor_directions(factor_cols: List[str]) -> Dict[str, int]:
    try:
        from quant_platform.core.signals.registry import FactorRegistry

        reg = FactorRegistry()
        reg.discover()
        return {name: reg.get(name).meta.direction for name in factor_cols if name in reg}
    except Exception:
        return {}


# ===================================================================
# Full walk-forward runner
# ===================================================================

def run_walk_forward(
    factor_df: pd.DataFrame,
    prices: pd.DataFrame,
    signal_col: str,
    config: Optional[WalkForwardConfig] = None,
) -> pd.DataFrame:
    """Execute walk-forward validation across all folds.

    Parameters
    ----------
    factor_df : ``[date, ticker, <factor_cols>]``
    prices : ``[date, ticker, adj_close, close, volume]``
    signal_col : which factor to evaluate
    config : walk-forward parameters

    Returns
    -------
    DataFrame with one row per fold:
        fold, train_start, train_end, test_start, test_end,
        train_ic, train_icir, oos_return, oos_sharpe, oos_max_dd
    """
    if config is None:
        config = WalkForwardConfig()

    factor_df = factor_df.copy()
    factor_df["date"] = pd.to_datetime(factor_df["date"])
    all_dates = pd.DatetimeIndex(sorted(factor_df["date"].unique()))

    folds = generate_folds(all_dates, config)

    results = []
    for i, (train_idx, test_idx) in enumerate(folds):
        logger.debug("Fold {}/{}: train {} – {}, test {} – {}",
                      i + 1, len(folds),
                      train_idx[0].date(), train_idx[-1].date(),
                      test_idx[0].date(), test_idx[-1].date())
        fold_result = _evaluate_fold(
            fold_idx=i + 1,
            train_dates=train_idx,
            test_dates=test_idx,
            factor_df=factor_df,
            prices=prices,
            signal_col=signal_col,
            config=config,
        )
        results.append(fold_result)

    result_df = pd.DataFrame(results)
    if result_df.empty:
        logger.warning("Walk-forward produced no valid folds")
        return result_df

    # Summary
    mean_oos = result_df["oos_sharpe"].mean()
    std_oos = result_df["oos_sharpe"].std()
    logger.info(
        "Walk-forward complete: {} folds, OOS Sharpe = {:.3f} ± {:.3f}",
        len(result_df), mean_oos, std_oos,
    )

    return result_df
