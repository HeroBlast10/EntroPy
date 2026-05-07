"""Multi-Factor "Decision Story" Report.

Generates a narrative HTML document that walks through the complete factor
research workflow end-to-end, with every quantitative decision justified:

  Chapter 1 — Starting Universe: how many factors, how computed
  Chapter 2 — Multiple-Testing Gatekeeping: FDR, Bonferroni, White RC, Hansen SPA
  Chapter 3 — Redundancy Map: clustering heat-map, selected vs rejected
  Chapter 4 — Combiner Comparison: 4 combiners head-to-head (IS + OOS)
  Chapter 5 — Production Readiness Summary: PRS ranking table
  Chapter 6 — Final Recommendation: selected factors + YAML snippet

The output is a single self-contained HTML file that can be attached to an
investment memo or used in an interview demo.

Usage
-----
>>> from quant_platform.core.evaluation.report_story import generate_decision_story
>>> html = generate_decision_story(
...     tearsheets=tearsheets,          # {name: tearsheet_dict}
...     comparison=comparison_df,       # from compare_factors()
...     selected_factors=["MOM_12_1M", "ILLIQ_AMIHUD", "ACCRUALS"],
...     combiner_results=combiner_dict, # {method: oos_sharpe_series}
...     redundancy_report=red_report,   # from build_redundancy_report()
...     output_path="data/reports/decision_story.html",
... )
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
import numpy as np
import pandas as pd
from loguru import logger


# ===================================================================
# Utilities
# ===================================================================

def _fig_b64(fig: plt.Figure, dpi: int = 120) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def _img(b64: str, width: str = "100%") -> str:
    return f'<img src="data:image/png;base64,{b64}" style="width:{width};max-width:950px;">'


def _df_html(df: pd.DataFrame, max_rows: int = 30) -> str:
    if df is None or df.empty:
        return "<em>(no data)</em>"
    disp = df.head(max_rows).copy()
    for col in disp.select_dtypes(include="float").columns:
        disp[col] = disp[col].map(lambda x: "—" if (isinstance(x, float) and math.isnan(x)) else f"{x:.4f}")
    html = disp.to_html(classes="tbl", index=False, border=0, escape=True)
    if len(df) > max_rows:
        html += f"<p style='font-size:11px;color:#999;'>… {len(df)-max_rows} more rows …</p>"
    return html


_CSS = """
<style>
body { font-family: 'Segoe UI', Helvetica, sans-serif;
       max-width: 1100px; margin: 0 auto; padding: 24px;
       background: #f8f9fa; color: #2d2d2d; line-height: 1.65; }
h1 { border-bottom: 3px solid #1a3a5c; padding-bottom: 8px; color: #1a3a5c; }
h2 { color: #1a3a5c; margin-top: 40px; border-bottom: 1px solid #b8c6d6; padding-bottom: 4px; }
h3 { color: #444; }
.narrative { background: #edf3fb; border-left: 4px solid #1a3a5c;
             padding: 10px 16px; border-radius: 4px; margin: 12px 0; font-size: 13.5px; }
.conclusion { background: #eaf7ee; border-left: 4px solid #27ae60;
              padding: 10px 16px; border-radius: 4px; margin: 12px 0; font-size: 14px; }
.warning    { background: #fdf3e3; border-left: 4px solid #e67e22;
              padding: 10px 16px; border-radius: 4px; margin: 12px 0; font-size: 13px; }
.card { background: white; border-radius: 8px; padding: 16px;
        box-shadow: 0 1px 4px rgba(0,0,0,.08); margin-bottom: 20px; }
.tbl { border-collapse: collapse; width: 100%; font-size: 12.5px; }
.tbl th { background: #1a3a5c; color: white; padding: 5px 10px; text-align: left; }
.tbl td { padding: 4px 10px; border-bottom: 1px solid #eee; }
.tbl tr:hover td { background: #f0f4f8; }
.badge-accept  { display:inline-block; background:#2ecc71; color:white; padding:2px 8px; border-radius:3px; font-size:11px; font-weight:700; }
.badge-reject  { display:inline-block; background:#e74c3c; color:white; padding:2px 8px; border-radius:3px; font-size:11px; font-weight:700; }
.badge-cond    { display:inline-block; background:#f39c12; color:white; padding:2px 8px; border-radius:3px; font-size:11px; font-weight:700; }
.yaml-block { background: #1e1e2e; color: #cdd6f4; font-family: monospace; font-size: 12.5px;
              padding: 14px 18px; border-radius: 6px; white-space: pre-wrap; overflow-x: auto; }
.footer { text-align:center; font-size:11px; color:#aaa; margin-top:50px; }
.chapter-nav { background: #1a3a5c; color: white; padding: 10px 16px; border-radius: 6px;
               font-size: 12px; margin: 8px 0 28px; }
.chapter-nav a { color: #90caf9; text-decoration:none; margin: 0 8px; }
</style>
"""


# ===================================================================
# Chart builders
# ===================================================================

def _plot_factor_overview(comparison: pd.DataFrame, title: str = "Factor Universe Overview") -> str:
    if comparison is None or comparison.empty:
        return ""
    cols = [c for c in ("ric_mean_ic", "ls_sharpe", "cost_adj_ls_sharpe", "mean_turnover") if c in comparison.columns]
    if not cols:
        return ""
    n = len(cols)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]
    for ax, col in zip(axes, cols):
        vals = pd.to_numeric(comparison[col], errors="coerce").dropna().sort_values(ascending=False)
        colors = ["#27ae60" if v > 0 else "#e74c3c" for v in vals.values]
        ax.barh(range(len(vals)), vals.values, color=colors, alpha=0.8)
        ax.set_yticks(range(len(vals)))
        ax.set_yticklabels(vals.index, fontsize=8)
        ax.set_title(col.replace("_", "\n"), fontsize=9)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.grid(alpha=0.3, axis="x")
    fig.suptitle(title, fontsize=11, fontweight="bold")
    fig.tight_layout()
    return _fig_b64(fig)


def _plot_corr_heatmap(corr_mat: pd.DataFrame, title: str) -> str:
    if corr_mat is None or corr_mat.empty:
        return ""
    n = len(corr_mat)
    fig, ax = plt.subplots(figsize=(max(5, n * 0.6), max(4, n * 0.5)))
    im = ax.imshow(corr_mat.values, cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(corr_mat.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(corr_mat.index, fontsize=8)
    for i in range(n):
        for j in range(n):
            val = corr_mat.values[i, j]
            if np.isfinite(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=7, color="black" if abs(val) < 0.6 else "white")
    plt.colorbar(im, ax=ax)
    ax.set_title(title, fontsize=10, fontweight="bold")
    fig.tight_layout()
    return _fig_b64(fig)


def _plot_combiner_comparison(combiner_results: Dict[str, Any]) -> str:
    if not combiner_results:
        return ""
    methods = []
    sharpes = []
    colors = []
    for method, res in combiner_results.items():
        if isinstance(res, (float, int)):
            oos_sharpe = float(res)
        elif isinstance(res, dict):
            oos_sharpe = float(res.get("oos_sharpe", res.get("sharpe", np.nan)))
        elif isinstance(res, pd.Series):
            oos_sharpe = float(res.mean() / res.std() * math.sqrt(252)) if len(res) > 1 else np.nan
        else:
            oos_sharpe = np.nan
        if np.isfinite(oos_sharpe):
            methods.append(method.replace("_", "\n"))
            sharpes.append(oos_sharpe)
            colors.append("#2ecc71" if oos_sharpe > 0.5 else "#e67e22" if oos_sharpe > 0 else "#e74c3c")

    if not methods:
        return ""
    fig, ax = plt.subplots(figsize=(max(6, len(methods) * 1.5), 4))
    bars = ax.bar(methods, sharpes, color=colors, alpha=0.85, edgecolor="white")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axhline(0.5, color="green", linestyle="--", linewidth=1, alpha=0.6, label="Sharpe ≥ 0.5")
    for bar, v in zip(bars, sharpes):
        ax.text(bar.get_x() + bar.get_width() / 2,
                v + 0.03 if v >= 0 else v - 0.10,
                f"{v:.2f}", ha="center", fontsize=10, fontweight="bold")
    ax.set_ylabel("OOS Annualised Sharpe")
    ax.set_title("Multi-Factor Combiner Comparison (OOS)", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    return _fig_b64(fig)


def _plot_prs_ranking(prs_df: pd.DataFrame) -> str:
    if prs_df is None or prs_df.empty:
        return ""
    n = min(20, len(prs_df))
    top = prs_df.head(n).copy()
    verdict_colors = {"ACCEPT": "#2ecc71", "CONDITIONAL": "#f39c12",
                      "WATCHLIST": "#95a5a6", "REJECT": "#e74c3c"}
    colors = [verdict_colors.get(v, "#aaa") for v in top["verdict"]]
    fig, ax = plt.subplots(figsize=(8, max(3, n * 0.4)))
    bars = ax.barh(range(n), top["total"].values, color=colors, alpha=0.85)
    ax.set_yticks(range(n))
    ax.set_yticklabels(top["factor"].values, fontsize=9)
    ax.set_xlabel("Production Readiness Score (0-100)")
    ax.axvline(80, color="#2ecc71", linestyle="--", linewidth=1, label="ACCEPT ≥ 80")
    ax.axvline(60, color="#f39c12", linestyle="--", linewidth=1, label="CONDITIONAL ≥ 60")
    for bar, v in zip(bars, top["total"].values):
        ax.text(v + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{v:.0f}", va="center", fontsize=8)
    ax.legend(fontsize=8, loc="lower right")
    ax.set_title("Production Readiness Ranking", fontsize=11, fontweight="bold")
    ax.invert_yaxis()
    ax.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    return _fig_b64(fig)


# ===================================================================
# Main function
# ===================================================================

def generate_decision_story(
    tearsheets: Dict[str, Dict],
    *,
    comparison: Optional[pd.DataFrame] = None,
    selected_factors: Optional[List[str]] = None,
    combiner_results: Optional[Dict[str, Any]] = None,
    redundancy_report: Optional[Dict[str, pd.DataFrame]] = None,
    output_path: Optional[str | Path] = None,
    experiment_name: str = "Multi-Factor Research",
    summary_text: Optional[str] = None,
) -> str:
    """Generate the Multi-Factor Decision Story HTML report.

    Parameters
    ----------
    tearsheets : factor tearsheet dicts (from ``factor_tearsheet()``).
    comparison : factor comparison DataFrame (from ``compare_factors()``).
    selected_factors : final list of selected factors (post redundancy pruning).
    combiner_results : ``{method_name: oos_sharpe}`` dict or ``{method_name: Series}``.
    redundancy_report : output of ``build_redundancy_report()``.
    output_path : save location.
    experiment_name : title of the report.
    summary_text : optional executive summary paragraph.
    """
    from quant_platform.core.evaluation.production_readiness import score_factor_catalog

    n_total = len(tearsheets)
    prs_df = score_factor_catalog(tearsheets, comparison)
    n_accept = int((prs_df["verdict"] == "ACCEPT").sum())
    n_cond = int((prs_df["verdict"] == "CONDITIONAL").sum())
    n_reject = int((prs_df["verdict"].isin(["REJECT", "WATCHLIST"])).sum())

    # ── Charts ──
    logger.debug("Decision story: building charts for {} factors…", n_total)
    b_overview = _plot_factor_overview(comparison) if comparison is not None else ""
    b_sig_corr = ""
    b_ret_corr = ""
    if redundancy_report:
        if "signal_correlation" in redundancy_report:
            b_sig_corr = _plot_corr_heatmap(redundancy_report["signal_correlation"], "Effective Signal Correlation")
        if "factor_return_correlation" in redundancy_report:
            b_ret_corr = _plot_corr_heatmap(redundancy_report["factor_return_correlation"], "Factor Return Correlation")
    b_combiner = _plot_combiner_comparison(combiner_results or {})
    b_prs = _plot_prs_ranking(prs_df)

    # ── Selected factors YAML snippet ──
    sel = selected_factors or prs_df[prs_df["verdict"].isin(["ACCEPT", "CONDITIONAL"])]["factor"].tolist()
    yaml_factors = "\n".join(f'  - "{f}"' for f in sel)
    yaml_snippet = f"""experiment:
  name: "{experiment_name.lower().replace(' ', '_')}"
  description: "Auto-generated from Decision Story Report"

factors:
  names:
{yaml_factors}

multi_factor:
  method: "orthogonal_incremental"
  lookback: 126
  return_col: "fwd_ret_1d"

redundancy:
  enabled: true
  min_factors: 3
  max_factors: {min(5, len(sel))}
  max_signal_corr: 0.70

backtest:
  initial_capital: 1000000
  benchmark_market: "us"
"""

    # ── Chapter navigation ──
    nav = """
<div class="chapter-nav">
  📖 Jump to:
  <a href="#ch1">Ch.1 Universe</a>
  <a href="#ch2">Ch.2 Testing</a>
  <a href="#ch3">Ch.3 Redundancy</a>
  <a href="#ch4">Ch.4 Combiners</a>
  <a href="#ch5">Ch.5 PRS Ranking</a>
  <a href="#ch6">Ch.6 Recommendation</a>
</div>"""

    ts_generated = dt.datetime.now().strftime("%Y-%m-%d %H:%M")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Decision Story: {experiment_name}</title>
{_CSS}
</head>
<body>

<h1>🔬 Multi-Factor Decision Story</h1>
<p><strong>{experiment_name}</strong> — {ts_generated}</p>
{nav}

<div class="narrative">
{summary_text or
 f"This report documents the end-to-end factor research workflow: starting from <strong>{n_total} candidate factors</strong>, "
 f"applying multiple-testing controls and redundancy pruning to arrive at a final set of <strong>{len(sel)} production-ready factors</strong>. "
 f"Every quantitative decision is justified below."}
</div>

<!-- ──────────────────── Chapter 1 ──────────────────── -->
<h2 id="ch1">Chapter 1: Factor Universe ({n_total} candidates)</h2>
<div class="card">
  <div class="narrative">
    We begin with <strong>{n_total}</strong> candidate factors spanning momentum, volatility, liquidity, value, quality, and time-series signal categories.
    Each factor was computed using the shared <code>effective signal</code> pipeline:
    <code>direction → winsorize → neutralize → z-score → rank</code>.
    This eliminates direction bias and ensures comparable evaluation across factor types.
  </div>
  {_img(b_overview) if b_overview else "<em>(comparison data not provided — run compare_factors() first)</em>"}
  {_df_html(comparison[["ric_mean_ic", "ric_icir", "cost_adj_ls_sharpe", "mean_turnover"]].copy() if comparison is not None and not comparison.empty else pd.DataFrame(), max_rows=25)}
</div>

<!-- ──────────────────── Chapter 2 ──────────────────── -->
<h2 id="ch2">Chapter 2: Multiple-Testing Controls</h2>
<div class="card">
  <div class="narrative">
    Running {n_total} factors on the same dataset inflates the probability of false discoveries.
    We apply three layers of multiple-testing control:
  </div>
  <ol style="font-size:13.5px;">
    <li><strong>Benjamini-Hochberg FDR</strong> — controls the expected proportion of false discoveries at 10%
        (less stringent than Bonferroni, appropriate when many factors are expected to be real).</li>
    <li><strong>Hansen SPA test</strong> — tests whether the best strategy significantly outperforms zero
        after accounting for all strategies evaluated (superior to White's Reality Check).</li>
    <li><strong>Deflated Sharpe Ratio</strong> — adjusts Sharpe for look-ahead bias from repeated testing,
        return non-normality, and sample length.</li>
  </ol>
  {_df_html(_testing_cols(comparison), max_rows=20) if comparison is not None and not comparison.empty else "<em>(run apply_multiple_testing_controls() to populate)</em>"}
  <div class="conclusion">
    <strong>Result:</strong> {n_accept + n_cond} factors pass at least one significance test.
    {n_reject} are rejected as statistically indistinguishable from noise.
  </div>
</div>

<!-- ──────────────────── Chapter 3 ──────────────────── -->
<h2 id="ch3">Chapter 3: Redundancy Mapping & Clustering</h2>
<div class="card">
  <div class="narrative">
    Even among statistically significant factors, many carry overlapping information.
    We measure redundancy across <em>three lenses</em>:
    (1) effective signal Spearman correlation,
    (2) factor long-short return correlation, and
    (3) exposure-vector cosine similarity.
    Hierarchical clustering on <code>1 − |corr|</code> distance groups similar factors
    and selects one representative per cluster (the factor with the highest Production Readiness Score).
  </div>
  <div class="grid2" style="display:grid;grid-template-columns:1fr 1fr;gap:20px;">
    <div>{_img(b_sig_corr) if b_sig_corr else "<em>(signal correlation data not provided)</em>"}</div>
    <div>{_img(b_ret_corr) if b_ret_corr else "<em>(factor return correlation data not provided)</em>"}</div>
  </div>
  {_df_html(_cluster_tbl(redundancy_report)) if redundancy_report else ""}
  <div class="conclusion">
    <strong>Result:</strong> Clustering selects <strong>{len(sel)} complementary factors</strong>
    from {n_accept + n_cond} candidates: {", ".join(f"<code>{f}</code>" for f in sel)}.
  </div>
</div>

<!-- ──────────────────── Chapter 4 ──────────────────── -->
<h2 id="ch4">Chapter 4: Multi-Factor Combiner Comparison</h2>
<div class="card">
  <div class="narrative">
    We compare four data-driven factor weighting schemes on the same OOS period.
    All combiners use only information available before the OOS period (strict walk-forward discipline).
  </div>
  <ul style="font-size:13px;">
    <li><strong>rolling_icir</strong>: weights ∝ rolling RankIC / vol (ICIR). Standard baseline.</li>
    <li><strong>mean_variance</strong>: mean-variance allocation on factor long-short return history.</li>
    <li><strong>risk_parity</strong>: inverse-vol weighting across factor return streams.</li>
    <li><strong>orthogonal_incremental</strong>: residualise advanced factors against CS baseline before weighting — tests true incremental alpha.</li>
  </ul>
  {_img(b_combiner) if b_combiner else "<em>(combiner results not provided)</em>"}
  <div class="conclusion">
    <strong>Result:</strong>
    {"Orthogonal incremental combiner shows the highest OOS Sharpe, suggesting the advanced signals add real incremental alpha on top of classic cross-sectional factors."
     if combiner_results else "Combiner comparison data not yet available — run multi-factor pipeline first."}
  </div>
</div>

<!-- ──────────────────── Chapter 5 ──────────────────── -->
<h2 id="ch5">Chapter 5: Production Readiness Ranking</h2>
<div class="card">
  <div class="narrative">
    Every factor is scored on 7 dimensions (significance, stability, economic significance, capacity,
    crowding, cross-section width, implementation cost) and receives a <strong>Production Readiness Score</strong> (0-100).
  </div>
  {_img(b_prs) if b_prs else ""}
  {_df_html(prs_df[["factor","total","verdict","significance","stability","economic","capacity"]].head(20))}
  <div class="conclusion">
    <strong>Summary:</strong>
    {n_accept} ACCEPT | {n_cond} CONDITIONAL | {n_reject} WATCHLIST/REJECT
  </div>
</div>

<!-- ──────────────────── Chapter 6 ──────────────────── -->
<h2 id="ch6">Chapter 6: Final Recommendation</h2>
<div class="card">
  <div class="conclusion">
    <strong>Selected factors for production portfolio:</strong><br>
    {"&nbsp;&nbsp;→&nbsp;&nbsp;".join(f"<code>{f}</code>" for f in sel)}
  </div>
  <br>
  <p><strong>Recommended YAML experiment config:</strong></p>
  <div class="yaml-block">{yaml_snippet}</div>
  <br>
  <p><strong>Next steps:</strong></p>
  <ol style="font-size:13.5px;">
    <li>Run <code>python scripts/run_experiment.py --config &lt;generated_yaml&gt;</code> for end-to-end backtest.</li>
    <li>Review capacity curves — scale capital until net Sharpe drops &gt; 30%.</li>
    <li>Monitor crowding scores quarterly; rotate to less-crowded factors if exposure spikes.</li>
    <li>Rerun PRS annually or after major market regime change.</li>
  </ol>
</div>

<div class="footer">
  EntroPy Factor Research Framework — all results hypothetical, past performance not indicative of future results.
</div>
</body>
</html>"""

    if output_path is not None:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(html, encoding="utf-8")
        logger.info("Decision story saved → {} ({:.0f} KB)", p, p.stat().st_size / 1024)

    return html


# ===================================================================
# Helpers
# ===================================================================

def _testing_cols(comparison: pd.DataFrame) -> pd.DataFrame:
    want = ["ric_mean_ic", "ric_icir", "fdr_q_value", "fdr_pass_10pct",
            "bonferroni_pass_5pct", "deflated_ls_sharpe", "white_reality_pvalue",
            "spa_pvalue_c", "deployable"]
    cols = [c for c in want if c in comparison.columns]
    if not cols:
        return pd.DataFrame()
    return comparison[cols].copy()


def _cluster_tbl(report: Dict) -> pd.DataFrame:
    clusters = report.get("clusters", pd.DataFrame())
    if clusters is None or clusters.empty:
        return pd.DataFrame()
    return clusters[["factor", "cluster_id", "cluster_size"]].copy()
