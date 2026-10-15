"""Production Readiness Score (PRS) — 7-dimensional factor quality scorecard.

Each candidate factor is scored 0-100 across seven dimensions and receives
a single headline PRS number that tells a portfolio manager or allocator:

    "Is this factor ready to deploy in a real portfolio?"

Dimensions and weights
----------------------
| # | Dimension              | Max | Key metrics                                     |
|---|------------------------|-----|-------------------------------------------------|
| 1 | Statistical significance| 20  | NW HAC IC t-stat, FDR q-value, DSR             |
| 2 | Temporal stability     | 20  | subperiod consistency, OOS IC, regime stability |
| 3 | Economic significance  | 15  | cost-adj LS Sharpe, break-even cost             |
| 4 | Capacity               | 15  | $-capacity at 10% ADV, participation rate       |
| 5 | Crowding               | 10  | crowding score (low = good)                     |
| 6 | Cross-sectional width  | 10  | monotonicity, long-only compatible              |
| 7 | Implementation cost    | 10  | turnover, alpha half-life vs rebalance frequency|

Score interpretation
--------------------
- 80-100: Strong production candidate — proceed to portfolio construction.
- 60-79: Conditional — requires further due diligence (cost / capacity concern).
- 40-59: Watchlist — promising signal, needs more data or parameter refining.
- <40:   Reject — insufficient evidence for live deployment.

Usage
-----
>>> from quant_platform.core.evaluation.production_readiness import score_factor_readiness
>>> prs = score_factor_readiness(advanced_metrics, ic_stats, rank_ic_stats)
>>> prs["total"]   # headline score
>>> prs["verdict"] # "ACCEPT" | "CONDITIONAL" | "WATCHLIST" | "REJECT"
>>> prs["breakdown"]  # per-dimension details
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


# ===================================================================
# Scoring helpers
# ===================================================================

def _clamp(x: float, lo: float, hi: float) -> float:
    if not np.isfinite(x):
        return 0.0
    return float(max(lo, min(hi, x)))


def _sigmoid(x: float, midpoint: float = 0.0, slope: float = 5.0) -> float:
    """Smooth transition from 0 to 1, centred at *midpoint*."""
    try:
        return 1.0 / (1.0 + math.exp(-slope * (x - midpoint)))
    except (OverflowError, ValueError):
        return 0.0 if x < midpoint else 1.0


def _safe(d: Dict[str, Any], key: str, default: float = np.nan) -> float:
    v = d.get(key, default)
    try:
        f = float(v)
        return f if np.isfinite(f) else default
    except (TypeError, ValueError):
        return default


# ===================================================================
# Per-dimension scoring functions (return 0-100 raw score + narrative)
# ===================================================================

def _score_significance(
    ic_stats: Dict,
    rank_ic_stats: Dict,
    fdr_q: float = np.nan,
    deflated_sharpe: float = np.nan,
) -> Dict:
    """Dimension 1: Statistical significance (max 100 before weighting).

    Sub-components:
    - NW-adjusted IC t-stat (50 pts): t > 2.5 gets full score.
    - FDR q-value (30 pts): passes at q <= 0.10.
    - Deflated Sharpe (20 pts): DSR > 0.80 for full marks.
    """
    nw_t = _safe(rank_ic_stats, "nw_t_stat")
    t_iid = _safe(rank_ic_stats, "t_stat")

    t_use = nw_t if np.isfinite(nw_t) else t_iid
    t_pts = _clamp(_sigmoid(t_use, midpoint=2.0, slope=1.0) * 50.0, 0, 50)

    if np.isfinite(fdr_q):
        fdr_pts = 30.0 * max(0.0, 1.0 - fdr_q / 0.10)
    else:
        # Fall back to p-value if FDR not available
        pval = _safe(rank_ic_stats, "nw_p_value", _safe(rank_ic_stats, "p_value", 0.5))
        fdr_pts = 30.0 * max(0.0, 1.0 - pval / 0.10)

    if np.isfinite(deflated_sharpe):
        dsr_pts = _clamp(deflated_sharpe * 20.0, 0, 20)
    else:
        hit = _safe(rank_ic_stats, "hit_rate", 0.5)
        dsr_pts = _clamp((hit - 0.5) / 0.10 * 20.0, 0, 20)

    total = t_pts + fdr_pts + dsr_pts
    return {
        "score": _clamp(total, 0, 100),
        "t_pts": t_pts,
        "fdr_pts": fdr_pts,
        "dsr_pts": dsr_pts,
        "nw_t_stat": nw_t,
        "fdr_q": fdr_q,
        "deflated_sharpe": deflated_sharpe,
    }


def _score_stability(adv: Dict) -> Dict:
    """Dimension 2: Temporal & regime stability (max 100).

    Sub-components:
    - Subperiod IC sign consistency (40 pts)
    - Rolling OOS IC sign consistency (30 pts)
    - Market regime sign consistency (30 pts)
    """
    sub = _safe(adv, "subperiod_sign_consistency", 0.5)
    oos = _safe(adv, "oos_rank_ic_sign_consistency", _safe(adv, "oos_rank_ic_mean", 0.0))
    if not (0.0 <= oos <= 1.0):
        oos = 0.5 if oos > 0 else 0.0
    reg = _safe(adv, "regime_sign_consistency", 0.5)

    sub_pts = _clamp((sub - 0.5) / 0.5 * 40.0, 0, 40)
    oos_pts = _clamp((oos - 0.5) / 0.5 * 30.0, 0, 30)
    reg_pts = _clamp((reg - 0.5) / 0.5 * 30.0, 0, 30)

    return {
        "score": sub_pts + oos_pts + reg_pts,
        "sub_pts": sub_pts,
        "oos_pts": oos_pts,
        "reg_pts": reg_pts,
        "subperiod_consistency": sub,
        "oos_consistency": oos,
        "regime_consistency": reg,
    }


def _score_economic(adv: Dict) -> Dict:
    """Dimension 3: Economic significance (max 100).

    Sub-components:
    - Cost-adj LS Sharpe 1d (60 pts): Sharpe 1.5 → full score.
    - Break-even cost (40 pts): BE > 20 bps → full score.
    """
    sharpe = _safe(adv, "cost_adj_ls_sharpe", np.nan)
    bec = _safe(adv, "break_even_cost_bps", np.nan)

    sh_pts = _clamp(_sigmoid(sharpe, midpoint=0.75, slope=2.0) * 60.0, 0, 60) if np.isfinite(sharpe) else 0.0
    be_pts = _clamp(_sigmoid(bec, midpoint=10.0, slope=0.2) * 40.0, 0, 40) if np.isfinite(bec) else 0.0

    return {
        "score": sh_pts + be_pts,
        "sharpe_pts": sh_pts,
        "bec_pts": be_pts,
        "cost_adj_sharpe": sharpe,
        "break_even_cost_bps": bec,
    }


def _score_capacity(adv: Dict) -> Dict:
    """Dimension 4: Capacity (max 100).

    Sub-components:
    - Dollar capacity at 10% ADV (70 pts): $100M → 70 pts, scale log.
    - Median selected ADV (30 pts): median stock ADV > $5M → 30 pts.
    """
    cap = _safe(adv, "capacity_10pct_adv", np.nan)
    adv_med = _safe(adv, "median_selected_adv", np.nan)

    if np.isfinite(cap) and cap > 0:
        cap_pts = _clamp(math.log10(cap / 1e6) / 2.0 * 70.0, 0, 70)
    else:
        cap_pts = 0.0

    if np.isfinite(adv_med) and adv_med > 0:
        adv_pts = _clamp(_sigmoid(math.log10(max(adv_med, 1)), midpoint=6.7, slope=2.0) * 30.0, 0, 30)
    else:
        adv_pts = 0.0

    return {
        "score": cap_pts + adv_pts,
        "capacity_pts": cap_pts,
        "adv_pts": adv_pts,
        "capacity_10pct_adv": cap,
        "median_adv": adv_med,
    }


def _score_crowding(adv: Dict, crowding_score: float = np.nan) -> Dict:
    """Dimension 5: Factor crowding risk (max 100, lower crowding = better).

    If an explicit crowding proxy is supplied, use it directly.
    Otherwise estimate from IC autocorrelation: high positive autocorrelation
    implies the factor tends to persist — which is a mild crowding proxy.
    """
    if np.isfinite(crowding_score):
        score = _clamp((1.0 - crowding_score) * 100.0, 0, 100)
    else:
        hit = _safe(adv, "hit_rate", 0.5)
        skew = _safe(adv, "skew", 0.0)
        ic_mean = _safe(adv, "mean_ic", 0.0)
        # Very high IC mean + low skew could signal crowding; keep neutral at 60
        score = 60.0 if not np.isfinite(ic_mean) else _clamp(70.0 - abs(skew) * 10.0, 30, 90)

    return {
        "score": score,
        "crowding_proxy": crowding_score,
    }


def _score_width(adv: Dict) -> Dict:
    """Dimension 6: Cross-sectional width / monotonicity (max 100).

    Sub-components:
    - Quantile monotonicity 1d (50 pts)
    - Long-only compatibility (50 pts)
    """
    mono = _safe(adv, "monotonicity_1d", np.nan)
    lo_compat = adv.get("long_only_compatible", False)
    lo_share = _safe(adv, "long_share_of_ls", 0.5)

    mono_pts = _clamp(_sigmoid(mono, midpoint=0.5, slope=10.0) * 50.0, 0, 50) if np.isfinite(mono) else 25.0
    if lo_compat:
        lo_pts = 50.0
    elif np.isfinite(lo_share):
        lo_pts = _clamp(lo_share * 50.0, 0, 50)
    else:
        lo_pts = 25.0

    return {
        "score": mono_pts + lo_pts,
        "mono_pts": mono_pts,
        "lo_pts": lo_pts,
        "monotonicity": mono,
        "long_only_compatible": lo_compat,
        "long_share_of_ls": lo_share,
    }


def _score_implementation(adv: Dict) -> Dict:
    """Dimension 7: Implementation cost efficiency (max 100).

    Sub-components:
    - Mean turnover (60 pts): turnover < 0.2 → 60 pts; penalises high churn.
    - Alpha half-life vs rebalance cadence (40 pts): half-life ≥ 5 days.
    """
    turnover = _safe(adv, "mean_turnover", np.nan)
    half_life = _safe(adv, "alpha_half_life_days", np.nan)

    if np.isfinite(turnover):
        to_pts = _clamp((1.0 - turnover / 0.5) * 60.0, 0, 60)
    else:
        to_pts = 30.0

    if np.isfinite(half_life) and half_life > 0 and not math.isinf(half_life):
        hl_pts = _clamp(_sigmoid(half_life, midpoint=5.0, slope=0.4) * 40.0, 0, 40)
    else:
        hl_pts = 20.0

    return {
        "score": to_pts + hl_pts,
        "turnover_pts": to_pts,
        "halflife_pts": hl_pts,
        "mean_turnover": turnover,
        "alpha_half_life_days": half_life,
    }


# ===================================================================
# Dimension weights (sum to 1.0)
# ===================================================================

DIMENSION_WEIGHTS = {
    "significance": 0.20,
    "stability": 0.20,
    "economic": 0.15,
    "capacity": 0.15,
    "crowding": 0.10,
    "width": 0.10,
    "implementation": 0.10,
}


# ===================================================================
# Main scoring function
# ===================================================================

def score_factor_readiness(
    advanced_metrics: Dict[str, Any],
    ic_stats: Optional[Dict[str, Any]] = None,
    rank_ic_stats: Optional[Dict[str, Any]] = None,
    *,
    fdr_q: float = np.nan,
    deflated_sharpe: float = np.nan,
    crowding_score: float = np.nan,
    factor_name: str = "factor",
) -> Dict[str, Any]:
    """Compute the Production Readiness Score (PRS) for one factor.

    Parameters
    ----------
    advanced_metrics : output of
        ``cross_sectional.evaluation.advanced_factor_metrics()``.
    ic_stats : output of ``ic_summary()`` for the Pearson IC series.
    rank_ic_stats : output of ``ic_summary()`` for the RankIC series.
        If not provided, falls back to keys in *advanced_metrics*.
    fdr_q : Benjamini-Hochberg FDR q-value from factor-selection pipeline.
    deflated_sharpe : DSR from ``overfit.deflated_sharpe_ratio()``.
    crowding_score : normalised 0-1 crowding proxy (0 = no crowding).
    factor_name : label for reporting.

    Returns
    -------
    Dict with:
        - ``factor``: factor name
        - ``total``: headline PRS 0-100
        - ``verdict``: "ACCEPT" | "CONDITIONAL" | "WATCHLIST" | "REJECT"
        - ``gate_failures``: list of human-readable hard-gate failure reasons
        - ``breakdown``: per-dimension sub-scores and narrative
        - ``weights``: dimension weights used
    """
    adv = advanced_metrics or {}
    ric_stats = rank_ic_stats or {}
    if not ric_stats and "ric_mean_ic" in adv:
        ric_stats = {
            "mean_ic": adv.get("ric_mean_ic", np.nan),
            "t_stat": adv.get("ric_t_stat", np.nan),
            "nw_t_stat": adv.get("ric_nw_tstat_1d", np.nan),
            "nw_p_value": adv.get("ric_nw_p_value_1d", np.nan),
            "p_value": adv.get("ric_p_value", np.nan),
            "hit_rate": adv.get("hit_rate", np.nan),
            "icir": adv.get("ric_icir", np.nan),
            "skew": adv.get("skew", np.nan),
        }

    dims = {
        "significance": _score_significance(ic_stats or {}, ric_stats, fdr_q, deflated_sharpe),
        "stability": _score_stability(adv),
        "economic": _score_economic(adv),
        "capacity": _score_capacity(adv),
        "crowding": _score_crowding(adv, crowding_score),
        "width": _score_width(adv),
        "implementation": _score_implementation(adv),
    }

    weighted = sum(dims[d]["score"] * DIMENSION_WEIGHTS[d] for d in dims)
    total = round(_clamp(weighted, 0.0, 100.0), 1)

    # Hard gates — these directly cause a REJECT regardless of total score
    gate_failures = _check_hard_gates(adv, ric_stats)

    if gate_failures:
        verdict = "REJECT"
    elif total >= 80:
        verdict = "ACCEPT"
    elif total >= 60:
        verdict = "CONDITIONAL"
    elif total >= 40:
        verdict = "WATCHLIST"
    else:
        verdict = "REJECT"

    # Narrative bar
    bar = _score_bar(total)

    breakdown_rows = []
    for dim_name, dim_result in dims.items():
        breakdown_rows.append({
            "dimension": dim_name,
            "raw_score": round(dim_result["score"], 1),
            "weight": DIMENSION_WEIGHTS[dim_name],
            "weighted_score": round(dim_result["score"] * DIMENSION_WEIGHTS[dim_name], 2),
        })

    return {
        "factor": factor_name,
        "total": total,
        "verdict": verdict,
        "verdict_bar": bar,
        "gate_failures": gate_failures,
        "breakdown": dims,
        "breakdown_table": pd.DataFrame(breakdown_rows),
        "weights": DIMENSION_WEIGHTS,
    }


def _check_hard_gates(adv: Dict, ric_stats: Dict) -> list[str]:
    """Hard production gates that force REJECT regardless of total score."""
    failures = []
    ic_mean = _safe(adv, "ric_mean_ic", _safe(ric_stats, "mean_ic", np.nan))
    oos_ic = _safe(adv, "oos_rank_ic_mean", np.nan)
    sharpe_adj = _safe(adv, "cost_adj_ls_sharpe", np.nan)
    sub_cons = _safe(adv, "subperiod_sign_consistency", np.nan)

    if np.isfinite(ic_mean) and ic_mean < 0:
        failures.append(f"IC sign incorrect: mean RankIC = {ic_mean:.4f} (must be > 0 after direction flip)")
    if np.isfinite(oos_ic) and oos_ic < 0:
        failures.append(f"OOS IC negative: {oos_ic:.4f} — no out-of-sample predictability")
    if np.isfinite(sharpe_adj) and sharpe_adj < -0.1:
        failures.append(f"Cost-adjusted Sharpe deeply negative: {sharpe_adj:.2f}")
    if np.isfinite(sub_cons) and sub_cons < 0.50:
        failures.append(f"Subperiod IC sign consistency = {sub_cons:.0%} < 50% — direction unreliable")
    return failures


def _score_bar(score: float, width: int = 20) -> str:
    """ASCII progress bar for terminal display."""
    filled = int(round(score / 100.0 * width))
    bar = "█" * filled + "░" * (width - filled)
    return f"[{bar}] {score:.1f}/100"


# ===================================================================
# Batch scoring
# ===================================================================

def score_factor_catalog(
    tearsheets: dict,
    comparison: Optional[pd.DataFrame] = None,
    *,
    fdr_q_series: Optional[pd.Series] = None,
    crowding_scores: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """Score all factors in a tearsheet dict and return a ranked DataFrame.

    Parameters
    ----------
    tearsheets : ``{factor_name: tearsheet_dict}`` as returned by
        ``factor_tearsheet()``.  Each value must contain
        ``"advanced_metrics"``, ``"rank_ic_stats"``, ``"ic_stats"``.
    comparison : optional pre-computed comparison table (carries FDR q-values).
    fdr_q_series : per-factor FDR q-values indexed by factor name.
    crowding_scores : per-factor crowding proxy (0-1).

    Returns
    -------
    DataFrame sorted by PRS descending with columns:
        factor, total, verdict, gate_failures, significance, stability,
        economic, capacity, crowding, width, implementation.
    """
    rows = []
    for fname, ts in tearsheets.items():
        adv = ts.get("advanced_metrics", {})
        ric_stats = ts.get("rank_ic_stats", {})
        ic_stats = ts.get("ic_stats", {})

        fdr = np.nan
        if fdr_q_series is not None and fname in fdr_q_series.index:
            fdr = float(fdr_q_series.loc[fname])
        elif comparison is not None and "fdr_q_value" in comparison.columns and fname in comparison.index:
            fdr = float(comparison.loc[fname, "fdr_q_value"])

        cs = np.nan
        if crowding_scores is not None and fname in crowding_scores.index:
            cs = float(crowding_scores.loc[fname])

        prs = score_factor_readiness(
            adv, ic_stats, ric_stats,
            fdr_q=fdr,
            crowding_score=cs,
            factor_name=fname,
        )
        row = {
            "factor": fname,
            "total": prs["total"],
            "verdict": prs["verdict"],
            "gate_failures": "; ".join(prs["gate_failures"]) if prs["gate_failures"] else "",
        }
        for dim in DIMENSION_WEIGHTS:
            row[dim] = round(prs["breakdown"][dim]["score"], 1)
        rows.append(row)

    df = pd.DataFrame(rows).sort_values("total", ascending=False).reset_index(drop=True)
    return df


def format_prs_report(prs: dict) -> str:
    """Return a human-readable PRS report string."""
    factor = prs.get("factor", "factor")
    total = prs.get("total", 0)
    verdict = prs.get("verdict", "?")
    bar = prs.get("verdict_bar", "")
    gates = prs.get("gate_failures", [])
    table = prs.get("breakdown_table", pd.DataFrame())

    lines = [
        f"═══════════════════════════════════════════",
        f"  Production Readiness Score: {factor}",
        f"═══════════════════════════════════════════",
        f"  Overall: {bar}",
        f"  Verdict: {verdict}",
        "",
    ]
    if gates:
        lines.append("  ⚠ Hard-gate failures:")
        for g in gates:
            lines.append(f"    • {g}")
        lines.append("")

    if not table.empty:
        lines.append("  Dimension breakdown:")
        lines.append(f"  {'Dimension':<18} {'Raw':>6} {'Wt':>6} {'Weighted':>9}")
        lines.append(f"  {'-'*43}")
        for _, r in table.iterrows():
            lines.append(
                f"  {r['dimension']:<18} {r['raw_score']:>6.1f} "
                f"{r['weight']:>6.0%} {r['weighted_score']:>9.2f}"
            )
        lines.append(f"  {'-'*43}")
        lines.append(f"  {'TOTAL':<18} {'':>6} {'':>6} {total:>9.1f}")
    lines.append("═══════════════════════════════════════════")
    return "\n".join(lines)
