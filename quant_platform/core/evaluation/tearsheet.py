"""Factor Discovery Tearsheet — one self-contained HTML per factor.

Renders a production-grade single-factor research tearsheet that includes:

  ┌──────────────────────────────────────────────────────────────────┐
  │ Section 1 – Header: factor name, category, PRS verdict bar      │
  │ Section 2 – Headline statistics table (NW HAC t-stat, IC, ICIR) │
  │ Section 3 – IC time series + 60d rolling IC                     │
  │ Section 4 – IC decay curve + alpha half-life annotation         │
  │ Section 5 – Quintile spread bar chart (monotonicity)            │
  │ Section 6 – Long / short leg Sharpe attribution                 │
  │ Section 7 – Cross-sectional stability heatmap (mcap / amount)   │
  │ Section 8 – Parameter robustness (if available)                 │
  │ Section 9 – Production Readiness Score breakdown                │
  │ Section 10 – Verdict + actionable notes                         │
  └──────────────────────────────────────────────────────────────────┘

All charts are inline base64 PNGs → single self-contained HTML file,
no external dependencies required.
"""
from __future__ import annotations

import base64
import datetime as dt
import io
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from loguru import logger


# ===================================================================
# Rendering helpers
# ===================================================================

def _fig_b64(fig: plt.Figure, dpi: int = 130) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def _img(b64: str, width: str = "100%") -> str:
    return f'<img src="data:image/png;base64,{b64}" style="width:{width};max-width:900px;">'


def _tbl(df: pd.DataFrame, fmt: str = ".4f") -> str:
    if df is None or df.empty:
        return "<em>(no data)</em>"
    def _cell(v):
        if isinstance(v, float):
            if math.isnan(v):
                return "—"
            return f"{v:{fmt}}"
        if isinstance(v, bool):
            return "✓" if v else "✗"
        return str(v)
    rows = []
    for idx, row in df.iterrows():
        cells = "".join(f"<td>{_cell(v)}</td>" for v in row)
        rows.append(f"<tr><th>{idx}</th>{cells}</tr>")
    hdrs = "".join(f"<th>{c}</th>" for c in df.columns)
    return f"<table class='tbl'><thead><tr><th></th>{hdrs}</tr></thead><tbody>{''.join(rows)}</tbody></table>"


def _stat_tbl(stats: Dict[str, Any]) -> str:
    """Render a metrics dict as a two-column key/value HTML table."""
    def _fmt(v):
        if isinstance(v, float):
            if math.isnan(v):
                return "—"
            if abs(v) < 0.001:
                return f"{v:.2e}"
            return f"{v:.4f}"
        if isinstance(v, bool):
            return "✓" if v else "✗"
        return str(v)

    rows = "".join(
        f"<tr><th>{k}</th><td>{_fmt(v)}</td></tr>"
        for k, v in stats.items()
        if not isinstance(v, (dict, pd.DataFrame, list))
    )
    return f"<table class='tbl'>{rows}</table>"


_CSS = """
<style>
body { font-family: 'Segoe UI', Helvetica, sans-serif;
       max-width: 1100px; margin: 0 auto; padding: 24px;
       background: #f8f9fa; color: #2d2d2d; }
h1 { border-bottom: 3px solid #1a3a5c; padding-bottom: 8px; color: #1a3a5c; font-size: 1.6em; }
h2 { color: #1a3a5c; margin-top: 36px; border-bottom: 1px solid #ccd; padding-bottom: 4px; font-size: 1.15em; }
h3 { color: #444; font-size: 1em; }
.header-meta { display: flex; gap: 20px; align-items: center; margin: 12px 0 20px; }
.badge { padding: 4px 12px; border-radius: 4px; font-weight: 700; font-size: 0.9em; }
.badge-ACCEPT      { background: #2ecc71; color: white; }
.badge-CONDITIONAL { background: #f39c12; color: white; }
.badge-WATCHLIST   { background: #95a5a6; color: white; }
.badge-REJECT      { background: #e74c3c; color: white; }
.prs-bar { font-family: monospace; font-size: 1.1em; }
.grid2 { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
.card { background: white; border-radius: 8px; padding: 16px;
        box-shadow: 0 1px 4px rgba(0,0,0,.08); margin-bottom: 20px; }
.tbl { border-collapse: collapse; width: auto; font-size: 12.5px; }
.tbl th { background: #1a3a5c; color: white; padding: 5px 10px; text-align: left; }
.tbl td { padding: 4px 10px; border-bottom: 1px solid #eee; }
.tbl tr:hover td { background: #f5f7fa; }
.gate-warn { background: #fdf3e3; border-left: 4px solid #e67e22;
             padding: 8px 14px; border-radius: 4px; margin: 8px 0; font-size: 13px; }
.gate-fail { background: #fde8e8; border-left: 4px solid #e74c3c;
             padding: 8px 14px; border-radius: 4px; margin: 8px 0; font-size: 13px; }
.verdict-notes { background: #eaf7ee; border-left: 4px solid #27ae60;
                  padding: 8px 14px; border-radius: 4px; font-size: 13px; }
.footer { text-align: center; font-size: 11px; color: #aaa; margin-top: 40px; }
</style>
"""


# ===================================================================
# Plot builders
# ===================================================================

def _plot_ic_timeseries(ts: pd.DataFrame, factor_name: str) -> str:
    """IC + rolling IC time series."""
    fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
    fig.suptitle(f"{factor_name} — IC Time Series", fontsize=12, fontweight="bold")

    ax1, ax2 = axes
    for col, label, color in [
        ("ic_series", "Pearson IC", "#2980b9"),
        ("rank_ic_series", "Spearman RankIC", "#e74c3c"),
    ]:
        if col in ts.columns:
            s = ts[col].dropna()
            ax1.plot(s.index, s.values, alpha=0.4, color=color, linewidth=0.8)
            rolled = s.rolling(63, min_periods=30).mean()
            ax1.plot(rolled.index, rolled.values, color=color, linewidth=1.6, label=f"63d rolling {label}")
    ax1.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax1.set_ylabel("IC")
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3)

    # Cumulative IC
    if "rank_ic_series" in ts.columns:
        cum = ts["rank_ic_series"].dropna().cumsum()
        ax2.fill_between(cum.index, cum.values, alpha=0.35, color="#8e44ad")
        ax2.plot(cum.index, cum.values, color="#8e44ad", linewidth=1.4, label="Cumulative RankIC")
        ax2.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax2.set_ylabel("Cumulative IC")
        ax2.legend(fontsize=9)
        ax2.grid(alpha=0.3)

    fig.tight_layout()
    return _fig_b64(fig)


def _plot_ic_decay(decay_df: pd.DataFrame, half_life: Optional[float], factor_name: str) -> str:
    fig, ax = plt.subplots(figsize=(8, 4))
    if decay_df is not None and not decay_df.empty and "horizon" in decay_df.columns:
        ax.bar(decay_df["horizon"], decay_df["mean_ic"], color="#3498db", alpha=0.7, label="Mean IC")
        ax.axhline(0, color="black", linewidth=0.8)
        if half_life is not None and np.isfinite(half_life) and half_life < 50:
            ax.axvline(half_life, color="#e74c3c", linestyle="--", linewidth=1.5,
                       label=f"Half-life ≈ {half_life:.1f}d")
        ax.set_xlabel("Holding Horizon (days)")
        ax.set_ylabel("Mean RankIC")
        ax.legend(fontsize=9)
    ax.set_title(f"{factor_name} — IC Decay", fontsize=11, fontweight="bold")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    return _fig_b64(fig)


def _plot_quintile_spread(qr: pd.DataFrame, factor_name: str) -> str:
    if qr is None or qr.empty:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "No quantile data", ha="center", va="center")
        return _fig_b64(fig)

    avg = qr.groupby("quantile")["mean_ret"].mean() * 252
    fig, ax = plt.subplots(figsize=(7, 4))
    colors = ["#e74c3c", "#e67e22", "#95a5a6", "#27ae60", "#1abc9c"]
    bars = ax.bar(avg.index, avg.values * 100, color=colors[:len(avg)])
    ax.set_xlabel("Quintile (1=bottom, 5=top)")
    ax.set_ylabel("Annualised Return (%)")
    ax.set_title(f"{factor_name} — Quintile Spread", fontsize=11, fontweight="bold")
    ax.axhline(0, color="black", linewidth=0.8)
    for bar, val in zip(bars, avg.values * 100):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=9)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    return _fig_b64(fig)


def _plot_leg_attribution(adv: Dict) -> str:
    long_sh = adv.get("long_leg_sharpe", np.nan)
    short_sh = adv.get("short_leg_sharpe", np.nan)
    ls_sh = adv.get("cost_adj_ls_sharpe", np.nan)
    labels = ["Long Leg", "Short Leg", "Net LS"]
    values = [
        float(long_sh) if np.isfinite(long_sh) else 0.0,
        float(short_sh) if np.isfinite(short_sh) else 0.0,
        float(ls_sh) if np.isfinite(ls_sh) else 0.0,
    ]
    colors = ["#2ecc71", "#e74c3c", "#3498db"]
    fig, ax = plt.subplots(figsize=(6, 3.5))
    bars = ax.bar(labels, values, color=colors)
    ax.axhline(0, color="black", linewidth=0.8)
    for bar, v in zip(bars, values):
        ypos = v + 0.05 if v >= 0 else v - 0.12
        ax.text(bar.get_x() + bar.get_width() / 2, ypos,
                f"{v:.2f}", ha="center", fontsize=10)
    ax.set_ylabel("Annualised Sharpe Ratio")
    ax.set_title("Long / Short Leg Attribution", fontsize=11, fontweight="bold")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    return _fig_b64(fig)


def _plot_stability_heatmap(stability_df: pd.DataFrame, factor_name: str) -> str:
    if stability_df is None or stability_df.empty:
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.text(0.5, 0.5, "No cross-section stability data (needs market_cap / amount columns)",
                ha="center", va="center")
        ax.axis("off")
        return _fig_b64(fig)

    n_segs = stability_df["segment"].nunique()
    n_buckets = stability_df["bucket"].max() + 1 if not stability_df.empty else 4
    fig, axes = plt.subplots(1, n_segs, figsize=(5 * n_segs, 3.5))
    if n_segs == 1:
        axes = [axes]
    for ax, (seg, grp) in zip(axes, stability_df.groupby("segment")):
        mat = grp.set_index("bucket")["rank_ic_mean"].sort_index()
        bar_colors = ["#e74c3c" if v < 0 else "#2ecc71" for v in mat.values]
        ax.bar(mat.index, mat.values, color=bar_colors, alpha=0.8)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title(f"Stability by {seg} quartile", fontsize=10)
        ax.set_xlabel("Bucket (low → high)")
        ax.set_ylabel("RankIC")
        ax.grid(alpha=0.3, axis="y")
    fig.suptitle(f"{factor_name} — Cross-Section Stability", fontsize=11, fontweight="bold")
    fig.tight_layout()
    return _fig_b64(fig)


def _plot_prs_radar(breakdown: Dict, weights: Dict) -> str:
    dims = list(weights.keys())
    scores = [breakdown[d]["score"] for d in dims]
    labels = [d.replace("_", " ").title() for d in dims]
    n = len(dims)
    angles = [i / n * 2 * math.pi for i in range(n)] + [0]
    scores_plot = scores + [scores[0]]

    fig, ax = plt.subplots(figsize=(5, 5), subplot_kw={"polar": True})
    ax.plot(angles, scores_plot, "o-", linewidth=2, color="#1a3a5c")
    ax.fill(angles, scores_plot, alpha=0.25, color="#1a3a5c")
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, size=10)
    ax.set_ylim(0, 100)
    ax.set_title("PRS Radar", size=11, fontweight="bold", pad=15)
    ax.grid(True)
    fig.tight_layout()
    return _fig_b64(fig)


# ===================================================================
# Main tearsheet builder
# ===================================================================

def generate_factor_tearsheet(
    tearsheet: Dict[str, Any],
    factor_name: str,
    *,
    prs_result: Optional[Dict] = None,
    ic_decay_df: Optional[pd.DataFrame] = None,
    output_path: Optional[str | Path] = None,
    extra_meta: Optional[Dict] = None,
) -> str:
    """Generate a self-contained HTML Factor Discovery Tearsheet.

    Parameters
    ----------
    tearsheet : output of ``cross_sectional.evaluation.factor_tearsheet()``.
    factor_name : factor identifier (used in titles and filenames).
    prs_result : output of ``production_readiness.score_factor_readiness()``.
    ic_decay_df : output of ``ic_decay()`` (optional; enhances decay chart).
    output_path : where to save the HTML; returns content as string if None.
    extra_meta : additional key-value metadata to display in the header.

    Returns
    -------
    HTML string.
    """
    adv = tearsheet.get("advanced_metrics", {})
    ric_stats = tearsheet.get("rank_ic_stats", {})
    ic_stats_d = tearsheet.get("ic_stats", {})
    stability = tearsheet.get("cross_sectional_stability", pd.DataFrame())
    qr = tearsheet.get("quantile_returns", pd.DataFrame())
    ls_ret = tearsheet.get("long_short", pd.Series(dtype=float))
    ic_series = tearsheet.get("ic_series", pd.Series(dtype=float))
    ric_series = tearsheet.get("rank_ic_series", pd.Series(dtype=float))

    # Build IC timeseries DataFrame for combined plot
    ts_df = pd.DataFrame({"ic_series": ic_series, "rank_ic_series": ric_series}).dropna(how="all")

    # Half-life
    half_life = adv.get("alpha_half_life_days", None)
    if half_life is not None and (not np.isfinite(half_life) or half_life == np.inf):
        half_life = None

    # PRS
    if prs_result is None:
        from quant_platform.core.evaluation.production_readiness import score_factor_readiness
        prs_result = score_factor_readiness(adv, ic_stats_d, ric_stats, factor_name=factor_name)

    verdict = prs_result.get("verdict", "?")
    total = prs_result.get("total", 0)
    bar = prs_result.get("verdict_bar", "")
    gates = prs_result.get("gate_failures", [])
    breakdown = prs_result.get("breakdown", {})
    weights = prs_result.get("weights", {})

    # ── Build charts ──
    logger.debug("Tearsheet {}: building charts …", factor_name)
    b_ic = _plot_ic_timeseries(ts_df, factor_name)
    b_decay = _plot_ic_decay(ic_decay_df, half_life, factor_name)
    b_quintile = _plot_quintile_spread(qr, factor_name)
    b_leg = _plot_leg_attribution(adv)
    b_stability = _plot_stability_heatmap(stability, factor_name)
    b_radar = _plot_prs_radar(breakdown, weights)

    # ── Headline stats table ──
    key_stats = {
        "Mean RankIC": ric_stats.get("mean_ic", np.nan),
        "ICIR (Annualised)": ric_stats.get("icir", np.nan),
        "i.i.d. t-stat": ric_stats.get("t_stat", np.nan),
        "NW HAC t-stat ★": ric_stats.get("nw_t_stat", np.nan),
        "NW p-value": ric_stats.get("nw_p_value", np.nan),
        "Hit Rate": ric_stats.get("hit_rate", np.nan),
        "Cost-adj LS Sharpe": adv.get("cost_adj_ls_sharpe", np.nan),
        "Break-even Cost (bps)": adv.get("break_even_cost_bps", np.nan),
        "Mean Turnover": adv.get("mean_turnover", np.nan),
        "Alpha Half-life (days)": adv.get("alpha_half_life_days", np.nan),
        "Subperiod Consistency": adv.get("subperiod_sign_consistency", np.nan),
        "OOS RankIC Mean": adv.get("oos_rank_ic_mean", np.nan),
        "Long Leg Sharpe": adv.get("long_leg_sharpe", np.nan),
        "Short Leg Sharpe": adv.get("short_leg_sharpe", np.nan),
        "Long-only Compatible": adv.get("long_only_compatible", False),
        "Capacity @10% ADV ($M)": (adv.get("capacity_10pct_adv", None) or np.nan) / 1e6
            if adv.get("capacity_10pct_adv") else np.nan,
    }

    def _fmt_stat(v):
        if isinstance(v, bool):
            return "✓" if v else "✗"
        if isinstance(v, float) and not np.isfinite(v):
            return "—"
        if isinstance(v, float):
            return f"{v:.4f}"
        return str(v)

    stat_rows = "".join(
        f"<tr><th>{k}</th><td>{_fmt_stat(v)}</td></tr>"
        for k, v in key_stats.items()
    )
    stats_html = f"<table class='tbl'>{stat_rows}</table>"

    # ── PRS breakdown table ──
    bt = prs_result.get("breakdown_table", pd.DataFrame())
    if not bt.empty:
        prs_rows = "".join(
            f"<tr><td>{r['dimension']}</td><td>{r['raw_score']:.1f}</td>"
            f"<td>{r['weight']:.0%}</td><td>{r['weighted_score']:.2f}</td></tr>"
            for _, r in bt.iterrows()
        )
        prs_tbl = (
            "<table class='tbl'>"
            "<thead><tr><th>Dimension</th><th>Raw (0-100)</th><th>Weight</th><th>Weighted</th></tr></thead>"
            f"<tbody>{prs_rows}"
            f"<tr><th colspan='3'>TOTAL</th><td><strong>{total:.1f}</strong></td></tr>"
            "</tbody></table>"
        )
    else:
        prs_tbl = "<em>(no PRS data)</em>"

    # ── Gate / verdict notes ──
    gate_html = ""
    if gates:
        for g in gates:
            gate_html += f"<div class='gate-fail'>⛔ {g}</div>"
    else:
        gate_html = "<div class='verdict-notes'>✓ All hard production gates passed.</div>"

    # ── Actionable advice ──
    advice = _build_advice(adv, prs_result, half_life)

    # ── Extra meta ──
    meta_html = ""
    if extra_meta:
        meta_rows = "".join(f"<li><strong>{k}:</strong> {v}</li>" for k, v in extra_meta.items())
        meta_html = f"<ul style='font-size:13px;'>{meta_rows}</ul>"

    ts_generated = dt.datetime.now().strftime("%Y-%m-%d %H:%M")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Factor Tearsheet: {factor_name}</title>
{_CSS}
</head>
<body>

<h1>Factor Discovery Tearsheet</h1>
<div class="header-meta">
  <div><strong style="font-size:1.4em;">{factor_name}</strong></div>
  <div class="badge badge-{verdict}">{verdict}</div>
  <div class="prs-bar">{bar}</div>
</div>
{meta_html}
<p style="font-size:12px;color:#888;">Generated: {ts_generated} — EntroPy Research Framework</p>

<!-- ── 1. Headline Statistics ── -->
<h2>§1 Headline Statistics</h2>
<div class="card">
  <p style="font-size:12px;color:#666;">★ NW HAC t-stat corrects for IC autocorrelation from overlapping forward returns.
     Use it instead of the i.i.d. t-stat when holding periods > 1 day.</p>
  {stats_html}
</div>

<!-- ── 2. IC Time Series ── -->
<h2>§2 IC Time Series</h2>
<div class="card">{_img(b_ic)}</div>

<!-- ── 3. IC Decay & Alpha Half-life ── -->
<h2>§3 IC Decay — Alpha Half-life</h2>
<div class="card">
  {'<p>Half-life: <strong>' + f'{half_life:.1f}' + ' trading days</strong> → recommended rebalance frequency ≤ ' + f'{max(1, int(half_life/2))}' + 'd.</p>' if half_life else ''}
  {_img(b_decay)}
</div>

<!-- ── 4. Quintile Spread ── -->
<h2>§4 Quintile Return Spread</h2>
<div class="card">{_img(b_quintile)}</div>

<!-- ── 5. Long / Short Leg Attribution ── -->
<h2>§5 Long / Short Leg Attribution</h2>
<div class="card">
  {'<p>⚠ Alpha mainly from short leg — may be difficult to deploy long-only.</p>' if not adv.get("long_only_compatible", True) else ''}
  {_img(b_leg)}
</div>

<!-- ── 6. Cross-Section Stability ── -->
<h2>§6 Cross-Section Stability (Size / Liquidity Buckets)</h2>
<div class="card">
  <p style="font-size:12px;color:#666;">Consistent RankIC across all buckets indicates broad-market applicability,
  not just small-cap or illiquid names.</p>
  {_img(b_stability)}
</div>

<!-- ── 7. Production Readiness Score ── -->
<h2>§7 Production Readiness Score (PRS)</h2>
<div class="card">
  <div class="grid2">
    <div>{_img(b_radar, width="100%")}</div>
    <div>{prs_tbl}</div>
  </div>
</div>

<!-- ── 8. Gate Assessment ── -->
<h2>§8 Hard Gate Assessment</h2>
<div class="card">{gate_html}</div>

<!-- ── 9. Actionable Notes ── -->
<h2>§9 Actionable Research Notes</h2>
<div class="card">
  <ul style="font-size:13px;line-height:1.7;">
    {''.join(f'<li>{a}</li>' for a in advice)}
  </ul>
</div>

<div class="footer">EntroPy Factor Research — results are hypothetical and for research purposes only.</div>
</body>
</html>"""

    if output_path is not None:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(html, encoding="utf-8")
        logger.info("Tearsheet saved → {} ({:.0f} KB)", p, p.stat().st_size / 1024)

    return html


def _build_advice(adv: Dict, prs: Dict, half_life: Optional[float]) -> List[str]:
    advice = []
    hl = adv.get("alpha_half_life_days", None)
    if hl and np.isfinite(hl) and hl < np.inf:
        rec_freq = max(1, int(hl / 2))
        advice.append(
            f"<strong>Rebalance frequency:</strong> IC half-life ≈ {hl:.1f}d → "
            f"rebalance every {rec_freq}d to capture ~50% of available alpha before decay."
        )
    lo = adv.get("long_only_compatible", False)
    ls_share = adv.get("long_share_of_ls", np.nan)
    if not lo and np.isfinite(ls_share):
        advice.append(
            f"<strong>Long-only deployment risk:</strong> Only {ls_share:.0%} of LS alpha comes from the long leg. "
            "Consider using this factor as an exclusion screen rather than a score."
        )
    bec = adv.get("break_even_cost_bps", np.nan)
    if np.isfinite(bec) and bec < 10:
        advice.append(
            f"<strong>Cost sensitivity:</strong> Break-even cost = {bec:.1f} bps. "
            "Very thin margin — favour liquid stocks, use limit orders, and widen rebalance bands."
        )
    cap = adv.get("capacity_10pct_adv", np.nan)
    if np.isfinite(cap):
        cap_m = cap / 1e6
        if cap_m < 10:
            advice.append(f"<strong>Capacity constraint:</strong> Estimated capacity = ${cap_m:.1f}M — suitable for small fund only.")
        elif cap_m < 100:
            advice.append(f"<strong>Capacity:</strong> ${cap_m:.0f}M — mid-size fund range. Monitor slippage at scale.")
        else:
            advice.append(f"<strong>Capacity:</strong> ${cap_m:.0f}M — large capacity; prioritise cost minimisation.")
    gates = prs.get("gate_failures", [])
    if not gates:
        advice.append("<strong>All hard gates passed.</strong> Suitable for inclusion in factor selection pipeline.")
    verdict = prs.get("verdict", "REJECT")
    if verdict == "ACCEPT":
        advice.append("✅ PRS ≥ 80: Proceed to multi-factor combiner. Use orthogonal incremental combiner to test marginal contribution.")
    elif verdict == "CONDITIONAL":
        advice.append("⚠ PRS 60-79: Conditional pass. Strengthen weaker dimensions (see radar) before live deployment.")
    elif verdict == "WATCHLIST":
        advice.append("🔍 PRS 40-59: Monitor. Accumulate more history or refine parameter set. Do not include in production portfolio yet.")
    else:
        advice.append("❌ PRS < 40 or hard gate failure. Retire from current research pipeline. Document learnings.")
    return advice


# ===================================================================
# Batch tearsheet generator
# ===================================================================

def generate_all_tearsheets(
    tearsheets: Dict[str, Dict],
    output_dir: str | Path,
    *,
    comparisons: Optional[pd.DataFrame] = None,
    ic_decay_map: Optional[Dict[str, pd.DataFrame]] = None,
) -> Dict[str, Path]:
    """Generate HTML tearsheets for all factors in *tearsheets*.

    Returns a ``{factor_name: html_path}`` mapping.
    """
    from quant_platform.core.evaluation.production_readiness import score_factor_catalog
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    prs_catalog = score_factor_catalog(tearsheets, comparisons)
    prs_lookup = {row["factor"]: row for _, row in prs_catalog.iterrows()}

    paths = {}
    for fname, ts in tearsheets.items():
        try:
            adv = ts.get("advanced_metrics", {})
            prs_row = prs_lookup.get(fname, {})
            prs_result = None
            if prs_row:
                from quant_platform.core.evaluation.production_readiness import score_factor_readiness
                prs_result = score_factor_readiness(
                    adv, ts.get("ic_stats", {}), ts.get("rank_ic_stats", {}),
                    factor_name=fname,
                )
            p = out / f"tearsheet_{fname}.html"
            generate_factor_tearsheet(
                ts, fname,
                prs_result=prs_result,
                ic_decay_df=ic_decay_map.get(fname) if ic_decay_map else None,
                output_path=p,
            )
            paths[fname] = p
        except Exception as exc:
            logger.error("Tearsheet failed for {}: {}", fname, exc)
    logger.info("Generated {} factor tearsheets → {}", len(paths), out)
    return paths
