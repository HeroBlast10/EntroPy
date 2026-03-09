"""
Signal Lab Dashboard — Quant Research Dashboard

Layout:
    Sidebar (Control Panel)  │  Main Panel (Oscilloscope)
    ─────────────────────────┼─────────────────────────────
    Market / Ticker / Params  │  Row 1: K-Line + Signal Overlay
    Physics Parameters        │  Row 2: Phase Portrait │ HMM Regime
    Strategy Config           │  Row 3: Signal Factors
    Cost Model                │  Row 4: Performance Attribution

Run:  streamlit run quant_platform/apps/dashboard/app.py
      (from project root)
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on path
ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(ROOT))

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# quant_platform imports
from quant_platform.core.data.adapters.cn_a_share import (
    AShareConfig,
    AShareDataLoader,
    AShareDataCleaner,
)
from quant_platform.core.data.adapters.us_equity import USEquityAdapter
from quant_platform.core.signals.time_series.kalman_state_space import _run_kalman_on_panel
from quant_platform.core.signals.time_series.entropy_hurst import (
    _rolling_spectral_entropy,
    _rolling_hurst,
)
from quant_platform.core.signals.time_series.higher_moments import (
    _rolling_skewness,
    _rolling_kurtosis,
    _rolling_acf_decay,
)
from quant_platform.core.signals.relative_value.ou_pairs import OUProcess
from quant_platform.core.signals.regime.hmm_regime import RegimeDetector
from quant_platform.core.signals.diagnostics.phase_space import PhaseSpaceAnalyzer
from quant_platform.core.alpha_models.ts_forecaster import MomentumZScoreSignal
from quant_platform.core.execution.cost_models.cn_a_share import AShareCostModel
from quant_platform.core.execution.cost_models.us_equity import CostModel as USCostModel

# ─────────────────────────────────────────────────────────
# Performance metrics (minimal subset from TradeX analytics)
# ─────────────────────────────────────────────────────────
TRADING_DAYS = 252


def _annualized_return(daily_returns: np.ndarray) -> float:
    total = np.prod(1.0 + np.nan_to_num(daily_returns)) - 1.0
    n = len(daily_returns)
    return float((1.0 + total) ** (TRADING_DAYS / max(n, 1)) - 1.0)


def _annualized_volatility(daily_returns: np.ndarray) -> float:
    return float(np.nanstd(daily_returns) * np.sqrt(TRADING_DAYS))


def _sharpe_ratio(daily_returns: np.ndarray, rf_daily: float = 0.025 / TRADING_DAYS) -> float:
    excess = np.nan_to_num(daily_returns) - rf_daily
    std = np.nanstd(excess)
    if std < 1e-12:
        return 0.0
    return float(np.nanmean(excess) / std * np.sqrt(TRADING_DAYS))


def _max_drawdown(equity: np.ndarray) -> float:
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / np.where(peak > 0, peak, 1.0)
    return float(np.nanmin(dd))


# ─────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Signal Lab Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────
# Sidebar — Control Panel
# ─────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("# 📊 Signal Lab Control Panel")
    st.markdown("---")

    # ── Market selector ──
    st.markdown("### 🌍 Market")
    market = st.radio(
        "Market",
        options=["CN", "US"],
        index=0,
        horizontal=True,
    )
    st.markdown("---")

    # ── Instrument ──
    st.markdown("### 🔬 Instrument")
    if market == "CN":
        ticker = st.text_input(
            "Ticker (ts_code)",
            value="600519.SH",
            help="Tushare format: 600519.SH",
        )
        cfg = AShareConfig()
        available_parquets = sorted([p.stem for p in cfg.parquet_path.glob("*.parquet")])
        if available_parquets:
            st.caption(
                f"Available: {', '.join(available_parquets[:10])}"
                f"{'...' if len(available_parquets) > 10 else ''}"
            )
    else:
        ticker = st.text_input(
            "Ticker",
            value="AAPL",
            help="US equity ticker symbol",
        )
        st.caption("Loads from quant_platform prices.parquet (run data pipeline first).")
    st.markdown("---")

    # ── Kalman Filter (Denoiser) ──
    st.markdown("### 🔭 Kalman Filter (Denoiser)")
    kf_process_noise = st.slider(
        "Process Noise  $q$  (system uncertainty)",
        min_value=-5.0, max_value=0.0, value=-3.0, step=0.5,
        format="1e%.0f",
        help="10^x: larger → trust observations more, noisier output",
    )
    kf_measurement_noise = st.slider(
        "Measurement Noise  $R$  (market noise)",
        min_value=-4.0, max_value=0.0, value=-2.0, step=0.5,
        format="1e%.0f",
        help="10^x: larger → trust model more, smoother output",
    )
    st.markdown("---")

    # ── Signal Processing ──
    st.markdown("### 📡 Signal Processing")
    lookback_tau = st.slider(
        "Lookback Period  $τ$  (trading days)",
        min_value=10, max_value=252, value=60, step=5,
        help="Rolling window for higher-order statistics",
    )
    st.markdown("---")

    # ── OU Process (Mean Reversion) ──
    st.markdown("### ⚖ OU Process (Potential Well)")
    ou_entry_z = st.slider(
        "Entry Threshold  $|z|$  (σ units)",
        min_value=1.0, max_value=4.0, value=2.0, step=0.25,
        help="Only enter when particle energy exceeds this",
    )
    ou_theta_min = st.slider(
        "Min Restoring Force  $θ_{min}$",
        min_value=1.0, max_value=20.0, value=3.0, step=1.0,
        help="Minimum mean-reversion speed to trade",
    )
    st.markdown("---")

    # ── Momentum Z-Score ──
    st.markdown("### 🚀 Momentum Z-Score")
    zscore_threshold = st.slider(
        "Overbought / Oversold  $|z|$  threshold",
        min_value=1.0, max_value=4.0, value=2.0, step=0.25,
    )
    st.markdown("---")

    # ── Regime Detection ──
    st.markdown("### 🌊 Regime Detection (HMM)")
    hmm_turbulence_pct = st.slider(
        "Turbulence Threshold  $P(turb)$",
        min_value=0.3, max_value=0.8, value=0.5, step=0.05,
    )
    st.markdown("---")

    run_btn = st.button("▶  Run Analysis", type="primary", use_container_width=True)


# ─────────────────────────────────────────────────────────
# Data loading & computation (cached)
# ─────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading data...")
def load_data_cn(ticker_code: str):
    cfg = AShareConfig()
    loader = AShareDataLoader(config=cfg)
    cleaner = AShareDataCleaner(config=cfg)
    try:
        df = loader.load_parquet(ticker_code)
    except FileNotFoundError:
        return None
    return cleaner.clean(df)


@st.cache_data(show_spinner="Loading US data...")
def load_data_us(ticker_code: str):
    adapter = USEquityAdapter()
    try:
        df = adapter.load_prices()
        if df.empty or "ticker" not in df.columns:
            return None
        sub = df[df["ticker"] == ticker_code].copy()
        if sub.empty:
            return None
        sub = sub.sort_values("date").reset_index(drop=True)
        sub["ret"] = sub["close"].pct_change()
        return sub
    except Exception:
        return None


def load_data(ticker_code: str, market: str):
    if market == "CN":
        return load_data_cn(ticker_code)
    return load_data_us(ticker_code)


def _kalman_filter_df(df: pd.DataFrame, Q: float, R: float, price_col: str = "close"):
    """Apply Kalman filter to single-ticker df. Returns df with kf_price, kf_velocity."""
    df = df.copy()
    if "adj_close" not in df.columns:
        df["adj_close"] = df[price_col]
    df["ticker"] = df["ts_code"].iloc[0] if "ts_code" in df.columns else "single"
    panel = df[["date", "ticker", "adj_close"]].copy()
    _df, filtered, velocity, _kg, _raw = _run_kalman_on_panel(panel, Q=Q, R=R)
    df["kf_price"] = filtered
    df["kf_velocity"] = velocity
    return df


def _ou_generate_position(
    ou_df: pd.DataFrame,
    entry_zscore: float,
    exit_zscore: float,
    theta_min: float,
    max_half_life_days: float = 120.0,
) -> pd.Series:
    """Derive ou_position from OU rolling fit (matching TradeX logic)."""
    z = ou_df["ou_zscore"].values
    theta = ou_df["ou_theta"].values
    hl = ou_df["ou_half_life"].values * 252.0  # convert to days
    n = len(z)
    position = np.zeros(n)
    for i in range(1, n):
        if np.isnan(z[i]) or np.isnan(theta[i]):
            position[i] = position[i - 1]
            continue
        strong_well = theta[i] > theta_min and hl[i] < max_half_life_days
        if strong_well:
            if z[i] < -entry_zscore:
                position[i] = 1.0
            elif z[i] > entry_zscore:
                position[i] = -1.0
            elif abs(z[i]) < exit_zscore:
                position[i] = 0.0
            else:
                position[i] = position[i - 1]
        else:
            position[i] = position[i - 1]
    return pd.Series(position, index=ou_df.index)


@st.cache_data(show_spinner="Computing signal factors...")
def compute_factors(
    _df_json: str,
    q_exp: float, r_exp: float,
    tau: int,
    ou_entry: float, ou_theta: float,
    zs_thresh: float,
    hmm_thresh: float,
):
    df = pd.read_json(_df_json)
    df["date"] = pd.to_datetime(df["date"])

    # Kalman
    Q = 10.0 ** q_exp
    R = 10.0 ** r_exp
    df = _kalman_filter_df(df, Q=Q, R=R)

    # Signal factors (entropy_hurst + higher_moments kernels)
    rets = df["ret"].fillna(0.0).values.astype(np.float64)
    df[f"skew_{tau}d"] = _rolling_skewness(rets, tau)
    df[f"kurt_{tau}d"] = _rolling_kurtosis(rets, tau)
    df[f"hurst_{tau}d"] = _rolling_hurst(rets, tau)
    df[f"spectral_entropy_{tau}d"] = _rolling_spectral_entropy(rets, tau)
    df[f"acf_decay_{tau}d"] = _rolling_acf_decay(rets, tau, max_lag=10)

    # OU Process (on log-price)
    ou = OUProcess(window=tau, dt=1.0 / 252.0)
    log_price = np.log(df["close"].clip(lower=1.0).values.astype(np.float64))
    ou_df = ou.generate_signals(pd.Series(log_price, index=df.index))
    for c in ou_df.columns:
        df[c] = ou_df[c].values
    df["ou_position"] = _ou_generate_position(
        ou_df, entry_zscore=ou_entry, exit_zscore=0.5,
        theta_min=ou_theta,
    ).values

    # Momentum Z-Score
    vel_col = "kf_velocity" if "kf_velocity" in df.columns else "KF_VELOCITY"
    mz = MomentumZScoreSignal(
        zscore_window=tau,
        overbought=zs_thresh,
        oversold=-zs_thresh,
    )
    df = df.assign(ticker=df.get("ts_code", "single"))
    df = mz.compute(df, velocity_col=vel_col)

    # Regime (HMM)
    rd = RegimeDetector(n_states=2, max_iter=200, tol=1e-4)
    ret_valid = df["ret"].dropna()
    gamma = rd.fit_predict(ret_valid.values)
    prob_turb = gamma[:, 1]
    regime = (prob_turb > hmm_thresh).astype(np.float64)
    regime_df = pd.DataFrame(
        {"prob_turbulent": prob_turb, "regime": regime},
        index=ret_valid.index,
    )
    df = df.merge(
        regime_df[["prob_turbulent", "regime"]],
        left_index=True, right_index=True, how="left",
    )
    df["regime"] = df["regime"].ffill().fillna(0)
    df["prob_turbulent"] = df["prob_turbulent"].ffill().fillna(0.5)

    # Phase Space
    psa = PhaseSpaceAnalyzer(price_col="close")
    df = psa.compute_metrics(df)

    # HMM params for display
    hmm_params = {}
    if hasattr(rd, "sigma") and rd.sigma is not None:
        hmm_params["sigma_laminar"] = float(rd.sigma[0])
        hmm_params["sigma_turbulent"] = float(rd.sigma[1])
        if rd.A is not None:
            hmm_params["expected_laminar_duration"] = 1.0 / max(rd.A[0, 1], 1e-10)
            hmm_params["expected_turbulent_duration"] = 1.0 / max(rd.A[1, 0], 1e-10)

    return df, hmm_params


# ─────────────────────────────────────────────────────────
# Main panel
# ─────────────────────────────────────────────────────────
st.markdown(
    "<h1 style='text-align:center;'>📊 Signal Lab Dashboard</h1>"
    "<p style='text-align:center; color:gray;'>Signal Processing · Kalman Filter · OU Process · Regime Detection · Phase Space</p>",
    unsafe_allow_html=True,
)

# Load data
df_raw = load_data(ticker, market)

if df_raw is None:
    st.error(
        f"No data found for **{ticker}** ({market} market). "
        f"For CN: run `python scripts/fetch_data.py --code {ticker}`. "
        f"For US: run the data pipeline first."
    )
    st.stop()

if not run_btn and "df_result" not in st.session_state:
    st.info("👈 Configure parameters in the sidebar and click **▶ Run Analysis**.")
    st.stop()

# Compute (or re-use session state)
if run_btn:
    df_result, hmm_params = compute_factors(
        df_raw.to_json(), kf_process_noise, kf_measurement_noise,
        lookback_tau, ou_entry_z, ou_theta_min, zscore_threshold, hmm_turbulence_pct,
    )
    st.session_state["df_result"] = df_result
    st.session_state["hmm_params"] = hmm_params

df = st.session_state["df_result"]
hmm_params = st.session_state.get("hmm_params", {})
dates = pd.to_datetime(df["date"])

# Cost model (market-dependent)
if market == "CN":
    cost_model = AShareCostModel()
    one_way_cost = (
        cost_model.commission_rate
        + cost_model.stamp_tax_rate * 0.5
        + cost_model.slippage_bps / 10000.0
    )
else:
    cost_model = USCostModel()
    one_way_cost = cost_model.slippage_bps / 10000.0
    if cost_model.commission_pct > 0:
        one_way_cost += cost_model.commission_pct
    else:
        one_way_cost += 0.0001  # ~1 bp commission equivalent

# ═════════════════════════════════════════════════════════
# ROW 1: K-Line + Signal Overlay
# ═════════════════════════════════════════════════════════
st.markdown("## 📈 Row 1 — Price Action & Signal Overlay")

fig1 = make_subplots(
    rows=3, cols=1, shared_xaxes=True,
    row_heights=[0.5, 0.25, 0.25],
    vertical_spacing=0.03,
    subplot_titles=("Price + Kalman Filter + OU Signals", "Kalman Velocity (Trend)", "Velocity Z-Score"),
)

fig1.add_trace(
    go.Scatter(
        x=dates, y=df["close"], name="Market Price",
        line=dict(color="rgba(150,150,150,0.5)", width=1),
    ), row=1, col=1,
)
fig1.add_trace(
    go.Scatter(
        x=dates, y=df["kf_price"], name="Kalman Price",
        line=dict(color="#1f77b4", width=2),
    ), row=1, col=1,
)

if "ou_mu" in df.columns:
    log_close_vals = np.log(df["close"].clip(lower=1.0).values)
    ou_mu_clipped = np.clip(
        df["ou_mu"].values,
        np.nanmin(log_close_vals) - 0.5,
        np.nanmax(log_close_vals) + 0.5,
    )
    ou_mu_price = np.exp(ou_mu_clipped)
    fig1.add_trace(
        go.Scatter(
            x=dates, y=ou_mu_price, name="OU Equilibrium (μ)",
            line=dict(color="#ff7f0e", width=1.5, dash="dash"),
        ), row=1, col=1,
    )

if "ou_position" in df.columns:
    long_mask = df["ou_position"] == 1.0
    short_mask = df["ou_position"] == -1.0
    if long_mask.any():
        fig1.add_trace(
            go.Scatter(
                x=dates[long_mask], y=df["close"].values[long_mask],
                mode="markers", name="OU Long",
                marker=dict(color="green", symbol="triangle-up", size=8),
            ), row=1, col=1,
        )
    if short_mask.any():
        fig1.add_trace(
            go.Scatter(
                x=dates[short_mask], y=df["close"].values[short_mask],
                mode="markers", name="OU Short",
                marker=dict(color="red", symbol="triangle-down", size=8),
            ), row=1, col=1,
        )

if "vel_zscore" in df.columns:
    z = df["vel_zscore"].values
    ob = z > zscore_threshold
    os_ = z < -zscore_threshold
    if ob.any():
        fig1.add_trace(
            go.Scatter(
                x=dates[ob], y=df["close"].values[ob],
                mode="markers", name=f"Overbought (z>{zscore_threshold})",
                marker=dict(color="red", symbol="x", size=6, opacity=0.7),
            ), row=1, col=1,
        )
    if os_.any():
        fig1.add_trace(
            go.Scatter(
                x=dates[os_], y=df["close"].values[os_],
                mode="markers", name=f"Oversold (z<-{zscore_threshold})",
                marker=dict(color="lime", symbol="x", size=6, opacity=0.7),
            ), row=1, col=1,
        )

if "regime" in df.columns:
    regime = df["regime"].values
    turb_start = None
    for i in range(len(regime)):
        if regime[i] == 1 and turb_start is None:
            turb_start = dates.iloc[i]
        elif regime[i] != 1 and turb_start is not None:
            fig1.add_vrect(
                x0=turb_start, x1=dates.iloc[i],
                fillcolor="red", opacity=0.07, line_width=0, row=1, col=1,
            )
            turb_start = None
    if turb_start is not None:
        fig1.add_vrect(
            x0=turb_start, x1=dates.iloc[-1],
            fillcolor="red", opacity=0.07, line_width=0, row=1, col=1,
        )

fig1.add_trace(
    go.Scatter(
        x=dates, y=df["kf_velocity"], name="Velocity",
        line=dict(color="#9467bd", width=1),
        fill="tozeroy", fillcolor="rgba(148,103,189,0.15)",
    ), row=2, col=1,
)
fig1.add_hline(y=0, line_dash="dot", line_color="gray", row=2, col=1)

if "vel_zscore" in df.columns:
    z_vals = df["vel_zscore"].values
    bar_colors = np.where(
        z_vals > zscore_threshold, "red",
        np.where(z_vals < -zscore_threshold, "green", "steelblue"),
    )
    fig1.add_trace(
        go.Bar(
            x=dates, y=z_vals, name="Vel Z-Score",
            marker_color=bar_colors.tolist(), opacity=0.7,
        ), row=3, col=1,
    )
    fig1.add_hline(y=zscore_threshold, line_dash="dash", line_color="red", row=3, col=1)
    fig1.add_hline(y=-zscore_threshold, line_dash="dash", line_color="green", row=3, col=1)

fig1.update_layout(
    height=750, template="plotly_white",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font_size=10),
    margin=dict(l=60, r=20, t=40, b=30),
    xaxis3_title="Date",
)
st.plotly_chart(fig1, use_container_width=True)

# ═════════════════════════════════════════════════════════
# ROW 2: Physics Diagnostics
# ═════════════════════════════════════════════════════════
st.markdown("## 🔬 Row 2 — Physics Diagnostics")

col_left, col_right = st.columns(2)

with col_left:
    st.markdown("### Phase Space Portrait")
    st.caption("Position (filtered price) vs Momentum (velocity) — the physicist's view of market dynamics")
    last_n = st.slider("Last N days", 60, 500, 252, step=10, key="phase_n")
    df_phase = df.tail(last_n)
    if "ps_x" in df_phase.columns and "ps_y" in df_phase.columns:
        x_ps = df_phase["ps_x"].values
        y_ps = df_phase["ps_y"].values
        n_pts = len(x_ps)
        t_color = np.linspace(0, 1, n_pts)
        fig_phase = go.Figure()
        fig_phase.add_trace(go.Scatter(
            x=x_ps, y=y_ps, mode="lines+markers",
            marker=dict(
                color=t_color, colorscale="RdBu_r", size=3, showscale=True,
                colorbar=dict(title="Time", tickvals=[0, 1], ticktext=["Old", "New"]),
            ),
            line=dict(color="rgba(100,100,100,0.3)", width=1),
            name="Trajectory",
        ))
        fig_phase.add_trace(go.Scatter(
            x=[x_ps[0]], y=[y_ps[0]], mode="markers",
            marker=dict(color="blue", size=12, symbol="circle"), name="Start",
        ))
        fig_phase.add_trace(go.Scatter(
            x=[x_ps[-1]], y=[y_ps[-1]], mode="markers",
            marker=dict(color="red", size=14, symbol="star"), name="End",
        ))
        for r in [1, 2, 3]:
            theta_c = np.linspace(0, 2 * np.pi, 100)
            fig_phase.add_trace(go.Scatter(
                x=r * np.cos(theta_c), y=r * np.sin(theta_c),
                mode="lines", line=dict(color="gray", dash="dot", width=0.5),
                showlegend=False,
            ))
        fig_phase.update_layout(
            height=500, template="plotly_white",
            xaxis_title="Position (normalized)",
            yaxis_title="Momentum (normalized)",
            xaxis=dict(scaleanchor="y", scaleratio=1),
            margin=dict(l=40, r=20, t=30, b=40),
        )
        st.plotly_chart(fig_phase, use_container_width=True)

with col_right:
    st.markdown("### Regime Detection (HMM)")
    st.caption("Laminar (low vol, Gaussian) vs Turbulent (high vol, telegraph noise)")
    if hmm_params:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("σ Laminar", f"{hmm_params.get('sigma_laminar', 0) * 100:.2f}%")
        c2.metric("σ Turbulent", f"{hmm_params.get('sigma_turbulent', 0) * 100:.2f}%")
        c3.metric("E[Laminar]", f"{hmm_params.get('expected_laminar_duration', 0):.1f}d")
        c4.metric("E[Turbulent]", f"{hmm_params.get('expected_turbulent_duration', 0):.1f}d")
    if "prob_turbulent" in df.columns:
        fig_hmm = make_subplots(
            rows=2, cols=1, shared_xaxes=True,
            row_heights=[0.6, 0.4], vertical_spacing=0.05,
            subplot_titles=("P(Turbulent)", "Daily Returns by Regime"),
        )
        pt = df["prob_turbulent"].values
        fig_hmm.add_trace(go.Scatter(
            x=dates, y=pt, name="P(Turbulent)",
            fill="tozeroy", fillcolor="rgba(255,100,100,0.2)",
            line=dict(color="crimson", width=1),
        ), row=1, col=1)
        fig_hmm.add_hline(y=hmm_turbulence_pct, line_dash="dash", line_color="black", row=1, col=1)
        ret_vals = df["ret"].values
        regime_vals = df["regime"].values
        colors_ret = np.where(regime_vals == 1, "crimson", "seagreen")
        fig_hmm.add_trace(go.Bar(
            x=dates, y=ret_vals, name="Returns",
            marker_color=colors_ret.tolist(), opacity=0.6,
        ), row=2, col=1)
        fig_hmm.update_layout(
            height=500, template="plotly_white",
            showlegend=False,
            margin=dict(l=40, r=20, t=30, b=40),
        )
        st.plotly_chart(fig_hmm, use_container_width=True)

# ═════════════════════════════════════════════════════════
# ROW 2.5: Signal Processing Factors
# ═════════════════════════════════════════════════════════
st.markdown("## 📡 Row 2.5 — Signal Processing Factors")

tau_str = f"{lookback_tau}d"
factor_cols = {
    f"skew_{tau_str}": ("Rolling Skewness γ₁", "#e377c2"),
    f"kurt_{tau_str}": ("Excess Kurtosis κ", "#d62728"),
    f"hurst_{tau_str}": ("Hurst Exponent H", "#2ca02c"),
    f"spectral_entropy_{tau_str}": ("Spectral Entropy", "#ff7f0e"),
    f"acf_decay_{tau_str}": ("ACF Decay β", "#17becf"),
}
available_factors = {k: v for k, v in factor_cols.items() if k in df.columns}

if available_factors:
    fig_sp = make_subplots(
        rows=len(available_factors), cols=1, shared_xaxes=True,
        vertical_spacing=0.04,
        subplot_titles=[v[0] for v in available_factors.values()],
    )
    for i, (col, (name, color)) in enumerate(available_factors.items(), 1):
        fig_sp.add_trace(go.Scatter(
            x=dates, y=df[col], name=name,
            line=dict(color=color, width=1),
        ), row=i, col=1)
        if "hurst" in col:
            fig_sp.add_hline(y=0.5, line_dash="dash", line_color="gray", row=i, col=1, annotation_text="H=0.5 (random walk)")
        if "spectral" in col:
            fig_sp.add_hline(y=1.0, line_dash="dash", line_color="gray", row=i, col=1, annotation_text="Max entropy")
    fig_sp.update_layout(
        height=150 * len(available_factors) + 50,
        template="plotly_white", showlegend=False,
        margin=dict(l=60, r=20, t=30, b=30),
    )
    st.plotly_chart(fig_sp, use_container_width=True)

# ═════════════════════════════════════════════════════════
# ROW 3: Performance Attribution
# ═════════════════════════════════════════════════════════
st.markdown("## 📊 Row 3 — Performance Attribution")

if "ou_position" in df.columns and "ret" in df.columns:
    pos = df["ou_position"].values
    ret_vals = df["ret"].values
    strat_ret = pos[:-1] * ret_vals[1:]
    strat_ret = np.insert(strat_ret, 0, 0.0)
    delta_pos = np.abs(np.diff(pos, prepend=0))
    daily_cost = delta_pos * one_way_cost
    net_ret = strat_ret - daily_cost
    eq_gross = np.cumprod(1.0 + strat_ret)
    eq_net = np.cumprod(1.0 + net_ret)
    eq_bh = np.cumprod(1.0 + np.nan_to_num(ret_vals))
    ann_ret_net = _annualized_return(net_ret)
    ann_vol = _annualized_volatility(net_ret)
    sharpe = _sharpe_ratio(net_ret)
    mdd = _max_drawdown(eq_net)
    win_rate = float((net_ret > 0).sum() / max((net_ret != 0).sum(), 1))
    total_cost = float(daily_cost.sum())
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Ann. Return (net)", f"{ann_ret_net * 100:.2f}%")
    m2.metric("Ann. Volatility", f"{ann_vol * 100:.2f}%")
    m3.metric("Sharpe Ratio", f"{sharpe:.3f}")
    m4.metric("Max Drawdown", f"{mdd * 100:.2f}%")
    m5.metric("Win Rate", f"{win_rate * 100:.1f}%")
    m6.metric("Total Cost Drag", f"{total_cost * 100:.2f}%")
    fig3 = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.65, 0.35], vertical_spacing=0.05,
        subplot_titles=("Cumulative Returns", "Drawdown"),
    )
    fig3.add_trace(go.Scatter(x=dates, y=eq_net, name="Strategy (net)", line=dict(color="#1f77b4", width=2)), row=1, col=1)
    fig3.add_trace(go.Scatter(x=dates, y=eq_gross, name="Strategy (gross)", line=dict(color="#1f77b4", width=1, dash="dot"), opacity=0.5), row=1, col=1)
    fig3.add_trace(go.Scatter(x=dates, y=eq_bh, name="Buy & Hold", line=dict(color="#ff7f0e", width=1.5)), row=1, col=1)
    peak = np.maximum.accumulate(eq_net)
    dd = (eq_net - peak) / peak
    fig3.add_trace(go.Scatter(
        x=dates, y=dd, name="Drawdown",
        fill="tozeroy", fillcolor="rgba(214,39,40,0.3)",
        line=dict(color="crimson", width=1),
    ), row=2, col=1)
    fig3.update_layout(
        height=500, template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=60, r=20, t=30, b=30),
        yaxis_title="Cumulative Return",
        yaxis2_title="Drawdown",
    )
    st.plotly_chart(fig3, use_container_width=True)
    with st.expander("💰 Cost Model Details"):
        if market == "CN":
            st.code(cost_model.summary())
            st.markdown(f"""
            | Component | Rate |
            |-----------|------|
            | Commission (bilateral) | {cost_model.commission_rate * 10000:.1f} bps |
            | Stamp Tax (sell-only) | {cost_model.stamp_tax_rate * 10000:.1f} bps |
            | Slippage ({cost_model.slippage_model}) | {cost_model.slippage_bps:.1f} bps |
            | **Total one-way** | **~{one_way_cost * 10000:.1f} bps** |
            """)
        else:
            st.markdown(f"US cost model: one-way ~{one_way_cost * 10000:.1f} bps (slippage + commission).")

# ── Footer ──
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:gray; font-size:12px;'>"
    "Signal Lab Dashboard v0.1 — Multi-Factor Research Framework<br>"
    "Kalman Filter · OU Process · HMM Regime Detection · Phase Space Reconstruction"
    "</p>",
    unsafe_allow_html=True,
)
