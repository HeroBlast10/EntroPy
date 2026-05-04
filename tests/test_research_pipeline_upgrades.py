"""Regression tests for production-grade factor research upgrades."""

from __future__ import annotations

import numpy as np
import pandas as pd


def test_effective_signal_flips_negative_direction_and_ranks():
    from quant_platform.core.signals.effective import build_effective_signal

    df = pd.DataFrame({
        "date": [pd.Timestamp("2024-01-02")] * 3,
        "ticker": ["A", "B", "C"],
        "raw": [3.0, 2.0, 1.0],
    })

    eff = build_effective_signal(df, "raw", direction=-1)
    ranked = eff.set_index("ticker")["raw"]

    assert ranked["C"] == 1.0
    assert ranked["A"] < ranked["B"] < ranked["C"]


def test_factor_tearsheet_adds_multi_horizon_and_testing_metrics():
    from quant_platform.core.signals.cross_sectional.evaluation import compare_factors, factor_tearsheet
    from quant_platform.core.signals.factor_selection import (
        apply_deployability_filters,
        apply_multiple_testing_controls,
    )

    dates = pd.bdate_range("2024-01-02", periods=30)
    tickers = [f"S{i:02d}" for i in range(12)]
    rows = []
    for d in dates:
        for i, ticker in enumerate(tickers):
            raw = float(i)
            rows.append({
                "date": d,
                "ticker": ticker,
                "factor": raw,
                "fwd_ret_1d": raw / 1000.0,
                "fwd_ret_5d": raw / 800.0,
                "fwd_ret_10d": raw / 700.0,
                "fwd_ret_20d": raw / 600.0,
                "adj_close": 100.0,
                "volume": 1_000_000 + i * 10_000,
            })
    df = pd.DataFrame(rows)

    ts = factor_tearsheet(df, "factor", forward_periods=[1, 5, 10, 20])
    comparison = compare_factors({"factor": ts})
    comparison = apply_multiple_testing_controls(comparison, {"factor": ts})
    comparison = apply_deployability_filters(comparison)

    row = comparison.loc["factor"]
    assert row["ric_mean_1d"] > 0
    assert row["ric_mean_20d"] > 0
    assert row["monotonicity_1d"] > 0
    assert "fdr_q_value" in comparison.columns
    assert "deployability_score" in comparison.columns


def test_walkforward_factor_select_top_k_uses_training_ic():
    from quant_platform.core.evaluation.walkforward import WalkForwardConfig, run_walk_forward

    dates = pd.bdate_range("2023-01-02", periods=140)
    tickers = [f"S{i:02d}" for i in range(12)]
    rng = np.random.default_rng(123)

    ret = {
        ticker: rng.normal(0, 0.01, len(dates))
        for ticker in tickers
    }
    price_rows = []
    factor_rows = []
    for ticker in tickers:
        prices = 100.0 * np.cumprod(1.0 + ret[ticker])
        for i, date in enumerate(dates):
            next_ret = ret[ticker][i + 1] if i + 1 < len(dates) else np.nan
            price_rows.append({
                "date": date,
                "ticker": ticker,
                "adj_close": prices[i],
                "close": prices[i],
            })
            factor_rows.append({
                "date": date,
                "ticker": ticker,
                "GOOD": next_ret,
                "BAD": -next_ret + rng.normal(0, 0.001),
            })

    wf = run_walk_forward(
        pd.DataFrame(factor_rows),
        pd.DataFrame(price_rows),
        "GOOD",
        WalkForwardConfig(
            train_months=3,
            test_months=1,
            step_months=1,
            min_train_obs=40,
            factor_select_top_k=1,
        ),
    )

    assert not wf.empty
    assert wf["selected_factors"].str.contains("GOOD").any()


def test_simulate_execution_sizes_trades_with_dynamic_nav():
    from quant_platform.core.execution.backtest.vectorized_daily import simulate_execution
    from quant_platform.core.execution.cost_models.us_equity import CostModel

    dates = pd.to_datetime(["2024-01-02", "2024-01-03"])
    prices = pd.DataFrame({
        "date": [dates[0], dates[0], dates[1], dates[1]],
        "ticker": ["A", "B", "A", "B"],
        "close": [100.0, 100.0, 200.0, 100.0],
        "adj_close": [100.0, 100.0, 200.0, 100.0],
        "volume": [1_000_000] * 4,
    })
    weights = pd.DataFrame({
        "date": [dates[0], dates[1]],
        "ticker": ["A", "B"],
        "weight": [1.0, 1.0],
    })
    zero_cost = CostModel(
        commission_per_share=0,
        commission_pct=0,
        commission_min=0,
        slippage_bps=0,
        impact_coeff=0,
        sec_fee_rate=0,
        finra_taf_per_share=0,
    )

    trades = simulate_execution(weights, prices, zero_cost, initial_capital=1_000.0)
    second_day = trades[trades["date"] == dates[1]]

    assert not second_day.empty
    assert second_day["portfolio_value_before_trade"].iloc[0] == 2_000.0
    assert second_day["notional_trade"].max() == 2_000.0


def test_borrow_cost_uses_dynamic_nav():
    from quant_platform.core.execution.backtest.pnl import compute_daily_returns
    from quant_platform.core.execution.cost_models.us_equity import CostModel

    dates = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
    prices = pd.DataFrame({
        "date": dates,
        "ticker": ["A", "A", "A"],
        "adj_close": [100.0, 100.0, 100.0],
        "close": [100.0, 100.0, 100.0],
    })
    weights = pd.DataFrame({
        "date": dates,
        "ticker": ["A", "A", "A"],
        "weight": [-1.0, -1.0, -1.0],
    })
    model = CostModel(borrow_rate_annual=0.252)

    pnl = compute_daily_returns(weights, prices, trades=None, cost_model=model, initial_capital=1_000.0)

    assert pnl["borrow_cost_dollar"].iloc[0] == 1.0
    assert pnl["borrow_cost_dollar"].iloc[1] < pnl["borrow_cost_dollar"].iloc[0]


def test_redundancy_selector_drops_duplicate_factor():
    from quant_platform.core.signals.redundancy import (
        RedundancyConfig,
        build_redundancy_report,
        select_complementary_factors,
    )

    dates = pd.bdate_range("2024-01-02", periods=30)
    tickers = [f"S{i:02d}" for i in range(20)]
    rows = []
    for d in dates:
        for i, ticker in enumerate(tickers):
            base = float(i)
            rows.append({
                "date": d,
                "ticker": ticker,
                "A": base,
                "A_COPY": base + 0.001,
                "B": float((i * 7) % 20),
                "C": float((i * 11) % 20),
                "fwd_ret_1d": base / 1000.0,
            })
    df = pd.DataFrame(rows)
    factors = ["A", "A_COPY", "B", "C"]
    report = build_redundancy_report(df, factors)
    scores = pd.DataFrame({
        "factor": factors,
        "selection_score": [1.0, 0.9, 0.8, 0.7],
    }).set_index("factor", drop=False)
    selected = select_complementary_factors(
        scores,
        report,
        config=RedundancyConfig(min_factors=2, max_factors=3, max_signal_corr=0.8),
    )

    chosen = selected[selected["selected"]]["factor"].tolist()
    assert "A" in chosen
    assert "A_COPY" not in chosen


def test_multi_factor_combiner_outputs_regime_controls():
    from quant_platform.core.alpha_models.multi_factor import (
        MultiFactorCombiner,
        MultiFactorConfig,
        RegimePolicy,
    )

    dates = pd.bdate_range("2024-01-02", periods=60)
    tickers = [f"S{i:02d}" for i in range(12)]
    rows = []
    for d_idx, d in enumerate(dates):
        for i, ticker in enumerate(tickers):
            rows.append({
                "date": d,
                "ticker": ticker,
                "F1": float(i),
                "F2": float(11 - i),
                "fwd_ret_1d": i / 1000.0,
                "HMM_TURBULENCE_PROB": 0.9 if d_idx >= 40 else 0.1,
            })
    df = pd.DataFrame(rows)
    combiner = MultiFactorCombiner(
        MultiFactorConfig(
            method="rolling_icir",
            lookback=20,
            min_periods=10,
            regime_policy=RegimePolicy(regime_col="HMM_TURBULENCE_PROB"),
        )
    )

    out = combiner.fit_transform(df, ["F1", "F2"])

    assert "alpha_multi" in out.columns
    assert not combiner.factor_weights_.empty
    assert not combiner.regime_controls_.empty
    assert combiner.regime_controls_["net_exposure"].min() < 1.0


def test_capacity_analysis_generates_capital_curve():
    from quant_platform.core.evaluation.capacity import capacity_analysis
    from quant_platform.core.execution.cost_models.us_equity import CostModel

    dates = pd.to_datetime(["2024-01-02", "2024-01-03"])
    trades = pd.DataFrame({
        "date": dates,
        "ticker": ["A", "A"],
        "side": ["buy", "sell"],
        "shares": [1_000.0, 1_000.0],
        "price": [100.0, 101.0],
        "adv_shares": [100_000.0, 100_000.0],
        "daily_vol": [0.02, 0.02],
        "notional_trade": [100_000.0, 101_000.0],
        "notional": [100_000.0, 101_000.0],
        "total_cost": [50.0, 50.0],
        "portfolio_value_before_trade": [1_000_000.0, 1_000_000.0],
    })
    pnl = pd.DataFrame({"gross_ret": [0.001, -0.001]}, index=dates)

    result = capacity_analysis(trades, pnl, cost_model=CostModel(commission_min=0), capital_grid=[1_000_000, 5_000_000])

    assert not result["summary"].empty
    assert not result["capital_curve"].empty
    assert result["summary"]["max_participation_rate"].iloc[0] == 0.01


def test_experiment_config_collects_factor_and_regime_names():
    from quant_platform.core.experiments.runner import (
        collect_factor_names,
        collect_regime_col,
        load_experiment_config,
    )

    cfg = load_experiment_config("quant_platform/experiments/us_signal_lab.yaml")
    factors = collect_factor_names(cfg)

    assert "MOM_12_1M" in factors
    assert "KF_VELOCITY" in factors
    assert collect_regime_col(cfg) == "HMM_TURBULENCE_PROB"
