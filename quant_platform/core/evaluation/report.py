"""Auto-generate an HTML research report from backtest artefacts.

Produces a single self-contained HTML file with:
- Executive summary (headline metrics table)
- NAV curve & drawdown
- Monthly return heatmap
- Rolling Sharpe
- IC / RankIC time series
- Quantile return bar chart
- Factor correlation heatmap
- Sector exposure
- Walk-forward OOS Sharpe
- Ablation comparison
- Turnover analysis
- Cost attribution

All charts are embedded as base64 PNGs (no external dependencies).
"""

from __future__ import annotations

import base64
import io
import datetime as dt
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger

from quant_platform.core.utils.io import load_config, load_parquet, resolve_data_path


# ===================================================================
# Helper: figure → base64 img tag
# ===================================================================

def _fig_to_b64(fig: plt.Figure, dpi: int = 120) -> str:
    """Render a matplotlib figure to a base64-encoded <img> tag."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("utf-8")
    return f'<img src="data:image/png;base64,{b64}" style="max-width:100%;">'


def _df_to_html(df: pd.DataFrame, precision: int = 4) -> str:
    """Render a DataFrame as a styled HTML table."""
    return df.to_html(
        classes="table",
        float_format=lambda x: f"{x:.{precision}f}" if isinstance(x, float) else str(x),
        border=0,
        index=True,
    )


# ===================================================================
# HTML template
# ===================================================================

_CSS = """
<style>
    body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
           max-width: 1200px; margin: 0 auto; padding: 20px;
           background: #fafafa; color: #333; }
    h1 { border-bottom: 3px solid #2c3e50; padding-bottom: 10px; color: #2c3e50; }
    h2 { color: #2c3e50; margin-top: 40px; border-bottom: 1px solid #ddd; padding-bottom: 5px; }
    h3 { color: #555; }
    .table { border-collapse: collapse; width: auto; margin: 10px 0; font-size: 13px; }
    .table th { background: #2c3e50; color: white; padding: 8px 12px; text-align: left; }
    .table td { padding: 6px 12px; border-bottom: 1px solid #eee; }
    .table tr:hover td { background: #f5f5f5; }
    .summary-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
    .metric-card { background: white; border-radius: 8px; padding: 15px;
                   box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
    .metric-card h3 { margin-top: 0; color: #2c3e50; }
    .section { margin-bottom: 30px; }
    .chart { background: white; border-radius: 8px; padding: 10px;
             box-shadow: 0 1px 3px rgba(0,0,0,0.1); margin: 15px 0; text-align: center; }
    .footer { margin-top: 50px; padding: 20px 0; border-top: 1px solid #ddd;
              font-size: 12px; color: #999; }
    .positive { color: #27ae60; } .negative { color: #c0392b; }
</style>
"""


def _metric_html(label: str, value, fmt: str = ".2%") -> str:
    """Format a single metric as styled HTML."""
    if isinstance(value, float) and np.isfinite(value):
        text = f"{value:{fmt}}"
        cls = "positive" if value > 0 else "negative" if value < 0 else ""
    else:
        text = str(value)
        cls = ""
    return f"<tr><td><strong>{label}</strong></td><td class='{cls}'>{text}</td></tr>"


# ===================================================================
# Report generator
# ===================================================================

def generate_report(
    output_path: Optional[Path | str] = None,
    signal_col: Optional[str] = None,
    run_walkforward: bool = True,
    run_ablation: bool = True,
    walkforward_kwargs: Optional[Dict] = None,
) -> Path:
    """Generate a full HTML research report from saved artefacts.

    Expects that ``build_dataset``, ``build_factors``, ``build_portfolio``,
    and ``run_backtest`` have already been executed.

    Parameters
    ----------
    output_path : destination HTML file.
    signal_col : factor to highlight in IC analysis.
    run_walkforward : whether to run walk-forward validation.
    run_ablation : whether to run cost ablation.
    walkforward_kwargs : override WalkForwardConfig parameters.

    Returns
    -------
    Path to the generated HTML report.
    """
    cfg = load_config()
    sections: List[str] = []

    # --- Load artefacts ---
    prices = _load_safe("prices", cfg["paths"]["prices_dir"], "prices.parquet")
    factors = _load_safe("factors", "factors", "factors.parquet")
    backtest_dir = resolve_data_path("backtest")

    daily_pnl = None
    if (backtest_dir / "daily_pnl.parquet").exists():
        daily_pnl = load_parquet(backtest_dir / "daily_pnl.parquet")
        daily_pnl["date"] = pd.to_datetime(daily_pnl["date"])
        daily_pnl = daily_pnl.set_index("date")

    trades = None
    if (backtest_dir / "trades.parquet").exists():
        trades = load_parquet(backtest_dir / "trades.parquet")

    perf_path = backtest_dir / "performance_summary.csv"
    perf = None
    if perf_path.exists():
        perf = pd.read_csv(perf_path).iloc[0].to_dict()

    weights_dir = resolve_data_path("portfolio")
    daily_weights = None
    if weights_dir.exists():
        wfiles = sorted(weights_dir.glob("weights_*.parquet"))
        if wfiles:
            daily_weights = load_parquet(wfiles[-1])
            daily_weights["date"] = pd.to_datetime(daily_weights["date"])

    # Detect signal col
    if signal_col is None and factors is not None:
        for c in factors.columns:
            if c not in ("date", "ticker"):
                signal_col = c
                break

    # ============================
    # Section 1: Executive Summary
    # ============================
    sections.append("<h2>1. Executive Summary</h2>")
    if perf:
        sections.append('<div class="summary-grid">')
        sections.append('<div class="metric-card"><h3>Gross Performance</h3><table>')
        sections.append(_metric_html("Ann. Return", perf.get("gross_ann_return")))
        sections.append(_metric_html("Ann. Volatility", perf.get("gross_ann_vol")))
        sections.append(_metric_html("Sharpe", perf.get("gross_sharpe"), ".2f"))
        sections.append(_metric_html("Sortino", perf.get("gross_sortino"), ".2f"))
        sections.append(_metric_html("Calmar", perf.get("gross_calmar"), ".2f"))
        sections.append(_metric_html("Max Drawdown", perf.get("gross_max_drawdown")))
        sections.append("</table></div>")

        sections.append('<div class="metric-card"><h3>Net Performance</h3><table>')
        sections.append(_metric_html("Ann. Return", perf.get("net_ann_return")))
        sections.append(_metric_html("Ann. Volatility", perf.get("net_ann_vol")))
        sections.append(_metric_html("Sharpe", perf.get("net_sharpe"), ".2f"))
        sections.append(_metric_html("Sortino", perf.get("net_sortino"), ".2f"))
        sections.append(_metric_html("Calmar", perf.get("net_calmar"), ".2f"))
        sections.append(_metric_html("Max Drawdown", perf.get("net_max_drawdown")))
        sections.append(_metric_html("Trading Cost (bps)", perf.get("total_trading_cost_bps"), ".1f"))
        sections.append("</table></div></div>")
    else:
        sections.append("<p><em>No backtest results found.</em></p>")

    # ============================
    # Section 2: NAV & Drawdown
    # ============================
    if daily_pnl is not None:
        from quant_platform.core.evaluation.plots import plot_nav, plot_drawdown
        sections.append("<h2>2. NAV & Drawdown</h2>")
        sections.append('<div class="chart">')
        sections.append(_fig_to_b64(plot_nav(daily_pnl)))
        sections.append("</div>")
        sections.append('<div class="chart">')
        sections.append(_fig_to_b64(plot_drawdown(daily_pnl)))
        sections.append("</div>")

    # ============================
    # Section 3: Monthly Returns
    # ============================
    if daily_pnl is not None:
        from quant_platform.core.evaluation.analytics import monthly_return_table
        from quant_platform.core.evaluation.plots import plot_monthly_heatmap
        sections.append("<h2>3. Monthly Return Heatmap</h2>")
        mt = monthly_return_table(daily_pnl["net_ret"])
        sections.append('<div class="chart">')
        sections.append(_fig_to_b64(plot_monthly_heatmap(mt)))
        sections.append("</div>")

    # ============================
    # Section 4: Rolling Sharpe
    # ============================
    if daily_pnl is not None:
        from quant_platform.core.evaluation.plots import plot_rolling_sharpe
        sections.append("<h2>4. Rolling 1-Year Sharpe Ratio</h2>")
        sections.append('<div class="chart">')
        sections.append(_fig_to_b64(plot_rolling_sharpe(daily_pnl["net_ret"])))
        sections.append("</div>")

    # ============================
    # Section 5: Turnover
    # ============================
    if daily_weights is not None:
        from quant_platform.core.evaluation.analytics import rolling_turnover
        sections.append("<h2>5. Turnover Analysis</h2>")
        to = rolling_turnover(daily_weights)
        if len(to) > 0:
            from quant_platform.core.evaluation.plots import _apply_style
            _apply_style()
            fig, ax = plt.subplots(figsize=(14, 3.5))
            ax.plot(to.index, to, linewidth=1.0)
            ax.axhline(to.mean(), color="orange", linestyle=":", label=f"Mean={to.mean():.2%}")
            ax.set_ylabel("Rolling Turnover")
            ax.set_title("21-Day Rolling Average One-Way Turnover")
            import matplotlib.ticker as mtick
            ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
            ax.legend()
            fig.tight_layout()
            sections.append('<div class="chart">')
            sections.append(_fig_to_b64(fig))
            sections.append("</div>")

    # ============================
    # Section 6: IC / RankIC
    # ============================
    if factors is not None and prices is not None and signal_col:
        sections.append(f"<h2>6. IC Analysis — {signal_col}</h2>")
        try:
            from quant_platform.core.signals.cross_sectional.evaluation import (
                add_forward_returns, compute_rank_ic_series, ic_summary,
                quantile_returns as qr_func,
            )
            from quant_platform.core.evaluation.plots import plot_ic_series, plot_quantile_returns

            prices_copy = prices.copy()
            prices_copy["date"] = pd.to_datetime(prices_copy["date"])
            prices_fwd = add_forward_returns(prices_copy, periods=[1])
            merged = factors[["date", "ticker", signal_col]].merge(
                prices_fwd[["date", "ticker", "fwd_ret_1d"]], on=["date", "ticker"], how="inner",
            )

            ric = compute_rank_ic_series(merged, signal_col, "fwd_ret_1d")
            ric_stats = ic_summary(ric)

            sections.append('<div class="metric-card"><h3>RankIC Summary</h3><table>')
            sections.append(_metric_html("Mean RankIC", ric_stats.get("mean_ic"), ".4f"))
            sections.append(_metric_html("ICIR (ann.)", ric_stats.get("icir"), ".2f"))
            sections.append(_metric_html("t-stat", ric_stats.get("t_stat"), ".2f"))
            sections.append(_metric_html("Hit Rate", ric_stats.get("hit_rate")))
            sections.append("</table></div>")

            sections.append('<div class="chart">')
            sections.append(_fig_to_b64(plot_ic_series(ric, title=f"Daily RankIC — {signal_col}")))
            sections.append("</div>")

            # Quantile returns
            qr = qr_func(merged, signal_col, "fwd_ret_1d")
            sections.append('<div class="chart">')
            sections.append(_fig_to_b64(plot_quantile_returns(qr, title=f"Quantile Returns — {signal_col}")))
            sections.append("</div>")
        except Exception as exc:
            sections.append(f"<p><em>IC analysis failed: {exc}</em></p>")
            logger.warning("IC analysis failed: {}", exc)

    # ============================
    # Section 7: Factor Correlation
    # ============================
    if factors is not None:
        sections.append("<h2>7. Factor Correlation Matrix</h2>")
        try:
            from quant_platform.core.evaluation.analytics import factor_correlation
            from quant_platform.core.evaluation.plots import plot_correlation_heatmap
            corr = factor_correlation(factors)
            if not corr.empty:
                sections.append('<div class="chart">')
                sections.append(_fig_to_b64(plot_correlation_heatmap(corr)))
                sections.append("</div>")
        except Exception as exc:
            sections.append(f"<p><em>Correlation analysis failed: {exc}</em></p>")

    # ============================
    # Section 8: Cost Attribution
    # ============================
    if trades is not None and not trades.empty:
        sections.append("<h2>8. Cost Attribution</h2>")
        try:
            from quant_platform.core.execution.backtest.pnl import cost_attribution
            attr = cost_attribution(trades)
            sections.append(_df_to_html(attr.set_index("component"), precision=2))
        except Exception as exc:
            sections.append(f"<p><em>Cost attribution failed: {exc}</em></p>")

    # ============================
    # Section 9: Walk-Forward
    # ============================
    if run_walkforward and factors is not None and prices is not None and signal_col:
        sections.append("<h2>9. Walk-Forward Validation</h2>")
        try:
            from quant_platform.core.evaluation.walkforward import WalkForwardConfig, run_walk_forward
            from quant_platform.core.evaluation.plots import plot_walkforward_sharpe

            wf_cfg = WalkForwardConfig(**(walkforward_kwargs or {}))
            wf = run_walk_forward(factors, prices, signal_col, wf_cfg)

            sections.append(_df_to_html(wf, precision=4))
            sections.append('<div class="chart">')
            sections.append(_fig_to_b64(plot_walkforward_sharpe(wf)))
            sections.append("</div>")

            mean_oos = wf["oos_sharpe"].mean()
            std_oos = wf["oos_sharpe"].std()
            sections.append(f"<p><strong>OOS Sharpe: {mean_oos:.3f} ± {std_oos:.3f}</strong></p>")
        except Exception as exc:
            sections.append(f"<p><em>Walk-forward failed: {exc}</em></p>")
            logger.warning("Walk-forward failed: {}", exc)

    # ============================
    # Section 10: Ablation
    # ============================
    if run_ablation and daily_weights is not None and trades is not None and prices is not None:
        sections.append("<h2>10. Ablation Study (Cost Sensitivity)</h2>")
        try:
            from quant_platform.core.evaluation.ablation import run_cost_ablation
            from quant_platform.core.evaluation.plots import plot_ablation

            abl = run_cost_ablation(daily_weights, prices, trades)

            display_cols = ["scenario", "net_ann_return", "net_sharpe", "net_max_drawdown",
                            "total_trading_cost_bps"]
            available = [c for c in display_cols if c in abl.columns]
            sections.append(_df_to_html(abl[available].set_index("scenario"), precision=4))

            if "net_sharpe" in abl.columns:
                sections.append('<div class="chart">')
                sections.append(_fig_to_b64(plot_ablation(abl, "net_sharpe",
                                                          title="Ablation: Net Sharpe by Cost Scenario")))
                sections.append("</div>")
        except Exception as exc:
            sections.append(f"<p><em>Ablation failed: {exc}</em></p>")
            logger.warning("Ablation failed: {}", exc)

    # ============================
    # Assemble HTML
    # ============================
    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>EntroPy Research Report</title>
    {_CSS}
</head>
<body>
    <h1>EntroPy — Multi-Factor Backtest Research Report</h1>
    <p>Generated: {now} | Signal: <strong>{signal_col or 'N/A'}</strong> |
       Period: {perf.get('start_date', 'N/A')} – {perf.get('end_date', 'N/A') if perf else 'N/A'}</p>
    {''.join(sections)}
    <div class="footer">
        <p>EntroPy Research Framework — auto-generated report.
           All results are hypothetical and do not guarantee future performance.</p>
    </div>
</body>
</html>"""

    # Save
    if output_path is None:
        output_path = resolve_data_path("reports", "research_report.html")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")

    logger.info("Report generated → {} ({:.0f} KB)", output_path, output_path.stat().st_size / 1024)
    return output_path


# ===================================================================
# Helpers
# ===================================================================

def _load_safe(name: str, *path_parts: str) -> Optional[pd.DataFrame]:
    """Load a Parquet file, returning None if it doesn't exist."""
    try:
        p = resolve_data_path(*path_parts)
        if p.exists():
            df = load_parquet(p)
            df["date"] = pd.to_datetime(df["date"]) if "date" in df.columns else df
            return df
    except Exception as exc:
        logger.warning("Could not load {}: {}", name, exc)
    return None
