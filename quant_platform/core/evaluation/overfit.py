"""Overfit detection: CSCV and Deflated Sharpe Ratio.

These methods help assess whether backtest performance is likely to
hold out-of-sample or is an artifact of data mining.

References
----------
- Bailey & Lopez de Prado (2014) "The Deflated Sharpe Ratio"
- Bailey, Borwein, Lopez de Prado, Zhu (2017) "The Probability of Backtest
  Overfitting", Journal of Computational Finance.
"""
from __future__ import annotations

import math
from itertools import combinations
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger


def deflated_sharpe_ratio(
    sharpe: float,
    n_trials: int,
    n_obs: int,
    skew: float = 0.0,
    kurt: float = 3.0,
) -> float:
    """Compute the Deflated Sharpe Ratio (DSR).

    Adjusts the observed Sharpe ratio for multiple testing, non-normality,
    and sample length.

    Parameters
    ----------
    sharpe : observed Sharpe ratio
    n_trials : number of strategy variants tested
    n_obs : number of return observations
    skew : skewness of returns
    kurt : kurtosis of returns (normal = 3)
    """
    from scipy import stats

    # Expected maximum Sharpe under null (all trials are noise)
    e_max_sr = stats.norm.ppf(1 - 1.0 / n_trials) if n_trials > 1 else 0.0

    # Standard error of Sharpe estimator
    se = np.sqrt((1 + 0.5 * sharpe**2 - skew * sharpe + ((kurt - 3) / 4) * sharpe**2) / n_obs)

    if se <= 0:
        return 0.0

    # Probability that observed Sharpe exceeds the expected max under null
    dsr = float(stats.norm.cdf((sharpe - e_max_sr) / se))
    return dsr


def _annualised_sharpe(returns: np.ndarray, ann: int = 252) -> float:
    if returns.size < 2:
        return 0.0
    mu = float(np.mean(returns))
    sd = float(np.std(returns, ddof=1))
    if sd <= 0:
        return 0.0
    return mu / sd * math.sqrt(ann)


def probability_of_backtest_overfitting(
    returns_matrix: pd.DataFrame,
    n_splits: int = 10,
    *,
    annualisation: int = 252,
    max_combinations: int = 20000,
    return_logits: bool = True,
) -> Dict[str, object]:
    """Combinatorially Symmetric Cross-Validation (CSCV) — full implementation.

    Bailey, Borwein, Lopez de Prado & Zhu (2017).  The CSCV procedure
    enumerates *all* C(S, S/2) ways to split S equal-length time blocks into
    in-sample (IS) and out-of-sample (OOS) halves.  For each split:

    1. Find the variant with the highest IS Sharpe.
    2. Compute the OOS rank ``r* in {1..N}`` of that variant.
    3. Compute the relative rank ``omega = r*/(N+1)`` and the logit
       ``log( omega / (1-omega) )``.

    The Probability of Backtest Overfitting (PBO) is the fraction of splits
    where the IS-best variant lands in the **bottom half** of the OOS Sharpe
    distribution (i.e. ``omega <= 0.5``).  PBO close to 0.5 = pure noise;
    PBO < 0.1 = the selection process generalises well.

    Parameters
    ----------
    returns_matrix : DataFrame of strategy / parameter-variant daily returns.
        Each column is one variant; index should be time.
    n_splits : number of equal-length time blocks (must be even, default 10).
    annualisation : Sharpe annualisation factor.
    max_combinations : safety cap.  ``C(16, 8) = 12870`` is the typical upper
        bound used in literature.
    return_logits : if True, also return the per-split logit distribution
      (useful for plotting the histogram in your tearsheet).

    Returns
    -------
    Dict with:
        - ``pbo``: float in [0, 1]
        - ``n_combinations``: number of splits actually evaluated
        - ``n_variants``: number of strategy columns
        - ``mean_logit``: mean of the OOS logit distribution
        - ``stochastic_dominance``: fraction of splits where IS-best beat
          OOS median (1 - PBO when ranks have no ties)
        - ``logits`` (when return_logits): list[float]
    """
    if returns_matrix is None or returns_matrix.empty:
        return {"pbo": np.nan, "n_combinations": 0, "n_variants": 0, "note": "empty input"}

    df = returns_matrix.dropna(how="all", axis=1).dropna(how="all", axis=0)
    n_obs, n_var = df.shape
    if n_var < 2:
        return {"pbo": np.nan, "n_combinations": 0, "n_variants": int(n_var), "note": "need >= 2 variants"}

    if n_splits % 2:
        n_splits -= 1  # must be even
    if n_splits < 2:
        return {"pbo": np.nan, "n_combinations": 0, "n_variants": int(n_var), "note": "n_splits too small"}

    block_size = n_obs // n_splits
    if block_size < 20:
        # Bailey et al. recommend each block be large enough to reliably
        # estimate Sharpe; <20 days is too small.
        logger.warning("CSCV: block size {} < 20, results may be unreliable", block_size)
        if block_size < 5:
            return {"pbo": np.nan, "n_combinations": 0, "n_variants": int(n_var),
                    "note": f"insufficient data: block_size={block_size}"}

    blocks: List[np.ndarray] = []
    for i in range(n_splits):
        start = i * block_size
        end = (i + 1) * block_size if i < n_splits - 1 else n_obs
        blocks.append(df.iloc[start:end].to_numpy())

    half = n_splits // 2
    total_combos = math.comb(n_splits, half)
    if total_combos > max_combinations:
        logger.warning("CSCV: {} > max {} combinations, sampling enabled", total_combos, max_combinations)
        rng = np.random.default_rng(0)
        all_combos = list(combinations(range(n_splits), half))
        chosen = rng.choice(len(all_combos), size=max_combinations, replace=False)
        combo_iter = (all_combos[i] for i in chosen)
        n_used = max_combinations
    else:
        combo_iter = combinations(range(n_splits), half)
        n_used = total_combos

    logits: List[float] = []
    is_better_than_median = 0
    for is_idx in combo_iter:
        oos_idx = tuple(i for i in range(n_splits) if i not in is_idx)
        is_data = np.concatenate([blocks[i] for i in is_idx], axis=0)
        oos_data = np.concatenate([blocks[i] for i in oos_idx], axis=0)

        is_sharpes = np.array([_annualised_sharpe(is_data[:, j], annualisation) for j in range(n_var)])
        oos_sharpes = np.array([_annualised_sharpe(oos_data[:, j], annualisation) for j in range(n_var)])

        best = int(np.argmax(is_sharpes))
        # OOS relative rank of the IS-best variant: r* in {1..N}, ascending order
        oos_rank = float((oos_sharpes < oos_sharpes[best]).sum() + 0.5 * (oos_sharpes == oos_sharpes[best]).sum())
        omega = oos_rank / (n_var + 1.0)  # in (0, 1)
        omega = min(max(omega, 1e-6), 1.0 - 1e-6)
        logits.append(math.log(omega / (1.0 - omega)))

        if oos_sharpes[best] > np.median(oos_sharpes):
            is_better_than_median += 1

    # PBO = fraction of logits <= 0  (i.e. omega <= 0.5)
    logits_arr = np.asarray(logits, dtype=float)
    pbo = float((logits_arr <= 0.0).mean()) if logits_arr.size else np.nan

    out: Dict[str, object] = {
        "pbo": pbo,
        "n_combinations": int(n_used),
        "n_combinations_total": int(total_combos),
        "n_variants": int(n_var),
        "n_splits": int(n_splits),
        "block_size": int(block_size),
        "mean_logit": float(np.mean(logits_arr)) if logits_arr.size else np.nan,
        "median_logit": float(np.median(logits_arr)) if logits_arr.size else np.nan,
        "stochastic_dominance": float(is_better_than_median / n_used) if n_used else np.nan,
    }
    if return_logits:
        out["logits"] = logits
    return out
