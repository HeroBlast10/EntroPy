"""Regression coverage for optimisation, structural risk, inference and fills."""
from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from quant_platform.core.portfolio.construction import PortfolioConfig
from quant_platform.core.portfolio.optimize import OptimizedPortfolio, PortfolioOptimizationError
from quant_platform.core.portfolio.risk_model import FactorRiskModel
from quant_platform.core.signals.redundancy import _incremental_alpha_stats, _incremental_sharpe, _can_add_factor, RedundancyConfig
from quant_platform.core.execution.backtest.vectorized_daily import simulate_execution, generate_trades
from quant_platform.core.execution.cost_models.us_equity import CostModel
from quant_platform.core.evaluation.capacity import capacity_analysis
from quant_platform.core.evaluation.production_readiness import score_factor_readiness, score_factor_catalog
from quant_platform.core.evaluation.overfit import deflated_sharpe_ratio


def optimizer(**kwargs):
    return OptimizedPortfolio(PortfolioConfig(max_stock_weight=1.0, max_sector_weight=None), use_factor_risk=False, **kwargs)


def zero_cost(**kwargs):
    return CostModel(commission_per_share=0, commission_min=0, slippage_bps=0,
                     impact_coeff=0, sec_fee_rate=0, finra_taf_per_share=0,
                     borrow_rate_annual=0, **kwargs)


def test_alpha_is_daily_return_with_configured_full_spread():
    opt = optimizer(expected_alpha_annual=0.12)
    alpha = opt._daily_expected_returns(np.array([10, 20, 30, 40, 50]))
    assert alpha.mean() == pytest.approx(0, abs=1e-15)
    assert np.ptp(alpha) == pytest.approx(0.12 / 252)


def test_rank_prior_is_invariant_to_score_scale_and_ties():
    opt = optimizer()
    scores = np.array([1, 1, 2, 5, 9])
    assert np.allclose(opt._daily_expected_returns(scores), opt._daily_expected_returns(scores * 100 + 77))
    assert opt._daily_expected_returns(scores)[0] == opt._daily_expected_returns(scores)[1]


@pytest.mark.parametrize("parameter,value", [("risk_aversion", -1), ("risk_aversion", np.nan), ("expected_alpha_annual", 0), ("expected_alpha_annual", np.inf)])
def test_optimizer_rejects_invalid_return_prior(parameter, value):
    with pytest.raises(ValueError):
        optimizer(**{parameter: value})


def test_lambda_reduces_concentration_for_equal_diagonal_covariance():
    cov = np.eye(6) * 0.0004
    maxima = []
    for lam in (1, 3, 10, 30, 100):
        opt = optimizer(risk_aversion=lam)
        w = opt._solve_qp(opt._daily_expected_returns(np.arange(6)), cov, np.arange(6), None)
        maxima.append(w.max())
    assert np.all(np.diff(maxima) <= 1e-6)
    assert maxima[-1] < maxima[0] - 0.05


def test_lambda_reduces_variance_for_general_covariance():
    rng = np.random.default_rng(2)
    a = rng.normal(size=(6, 6))
    cov = (a @ a.T + np.eye(6)) * 0.0001
    variances = []
    for lam in (1, 10, 100):
        opt = optimizer(risk_aversion=lam)
        w = opt._solve_qp(opt._daily_expected_returns(np.arange(6)), cov, np.arange(6), None).to_numpy()
        variances.append(w @ cov @ w)
    assert np.all(np.diff(variances) <= 1e-9)
    assert variances[-1] < variances[0]


def test_covariance_failure_raises_instead_of_fallback():
    date = pd.Timestamp("2024-01-31")
    signal = pd.DataFrame({"date": date, "ticker": list("ABCDEF"), "signal": np.arange(6)})
    with pytest.raises(PortfolioOptimizationError, match="covariance history"):
        optimizer()._generate_weights(signal, signal[["date", "ticker"]], date)


def test_solver_failure_has_typed_exception(monkeypatch):
    import scipy.optimize
    monkeypatch.setattr(scipy.optimize, "minimize", lambda *a, **k: SimpleNamespace(success=False, message="infeasible"))
    with pytest.raises(PortfolioOptimizationError, match="infeasible"):
        optimizer()._solve_qp(np.zeros(6), np.eye(6), np.arange(6), None)


@pytest.mark.parametrize("fallback", [None, "quantile"])
def test_only_explicit_pipeline_fallback_is_saved(tmp_path, monkeypatch, fallback):
    from quant_platform.core.portfolio import pipeline
    date = pd.Timestamp("2024-01-31")
    signal = pd.DataFrame({"date": date, "ticker": list("ABCDEF"), "signal": np.arange(6)})
    prices = signal[["date", "ticker"]].assign(adj_close=100)
    uni = signal[["date", "ticker"]].assign(pass_all_filters=True)
    monkeypatch.setattr(pipeline, "resolve_data_path", lambda *parts: tmp_path.joinpath(*parts))
    monkeypatch.setattr(pipeline, "rebalance_dates", lambda **kwargs: pd.DatetimeIndex([date]))
    monkeypatch.setattr(pipeline.OptimizedPortfolio, "build", lambda *a, **k: (_ for _ in ()).throw(PortfolioOptimizationError("risk unavailable")))
    monkeypatch.setattr(pipeline.QuantilePortfolio, "build", lambda *a, **k: signal[["date", "ticker"]].assign(weight=1/6))
    kwargs = dict(method="optimize", signal_col="signal", config=PortfolioConfig(max_stock_weight=1, max_sector_weight=None),
                  factors_override=signal, prices_override=prices, universe_override=uni, use_factor_risk=False,
                  output_path=tmp_path / "weights.parquet", fallback_method=fallback)
    if fallback is None:
        with pytest.raises(PortfolioOptimizationError):
            pipeline.run_portfolio_pipeline(**kwargs)
        assert not (tmp_path / "weights.metadata.json").exists()
    else:
        pipeline.run_portfolio_pipeline(**kwargs)
        meta = json.loads((tmp_path / "weights.metadata.json").read_text())
        assert meta["fallback"] is True
        assert meta["effective_method"] == "quantile"
        assert meta["fallback_reason"] == "risk unavailable"
        assert json.loads((tmp_path / "portfolio" / "metadata.json").read_text())["fallback"] is True


@pytest.mark.parametrize("in_memory", [False, True])
def test_backtest_metadata_carries_only_matching_portfolio_fallback(tmp_path, monkeypatch, in_memory):
    from quant_platform.core.execution.backtest import pipeline
    dates, prices, weights = execution_inputs(2)
    weights_path = tmp_path / "weights.parquet"
    weights.to_parquet(weights_path, index=False)
    sidecar = {"fallback": True, "fallback_reason": "risk unavailable", "method": "optimize", "effective_method": "quantile"}
    weights_path.with_suffix(".metadata.json").write_text(json.dumps(sidecar))
    monkeypatch.setattr(pipeline, "resolve_data_path", lambda *parts: tmp_path.joinpath(*parts))
    pipeline.run_trading_pipeline(weights_path=weights_path, weights_override=weights if in_memory else None,
                                  prices_override=prices, cost_model=zero_cost(max_participation_rate=None),
                                  benchmark_market=None, output_dir=tmp_path / "backtest", capacity_capitals=[1e6])
    meta = json.loads((tmp_path / "backtest" / "backtest_metadata.json").read_text())
    assert meta["fallback"] is (not in_memory)
    if not in_memory:
        assert meta["fallback_reason"] == "risk unavailable"
        assert meta["effective_portfolio_method"] == "quantile"


def risk_inputs():
    rng = np.random.default_rng(101)
    dates = pd.bdate_range("2023-01-02", periods=120)
    returns = pd.DataFrame(rng.normal(0, .01, (120, 6)), index=dates, columns=list("ABCDEF"))
    caps = pd.DataFrame(np.tile([1, 2, 3, 20, 30, 40], (120, 1)), index=dates, columns=returns.columns)
    sectors = pd.DataFrame({"ticker": list("ABCDEF"), "sector": ["X", "Y", "Z", "X", "Y", "Z"]})
    return returns, caps, sectors


def test_size_factor_is_lagged_small_minus_big():
    ret, caps, _ = risk_inputs()
    caps.iloc[50] = caps.iloc[50, ::-1].to_numpy()
    factors = FactorRiskModel.build_structural_factors(ret, caps)
    expected = ret.iloc[50, :3].mean() - ret.iloc[50, 3:].mean()
    assert factors.loc[ret.index[50], "size"] == pytest.approx(expected)
    assert factors.loc[ret.index[51], "size"] == pytest.approx(ret.iloc[51, 3:].mean() - ret.iloc[51, :3].mean())


def test_future_caps_do_not_change_past_size_factor():
    ret, caps, _ = risk_inputs()
    before = FactorRiskModel.build_structural_factors(ret, caps)
    caps.iloc[60:] = caps.iloc[60:, ::-1].to_numpy()
    after = FactorRiskModel.build_structural_factors(ret, caps)
    pd.testing.assert_frame_equal(before.iloc[:61], after.iloc[:61])


def test_sector_map_adds_independent_sector_dimensions_and_euler_checks():
    ret, caps, sectors = risk_inputs()
    factors = FactorRiskModel.build_structural_factors(ret, caps)
    model = FactorRiskModel().fit(ret, factors, sectors)
    assert set(model.factor_names_) == {"market", "size", "sector:Y", "sector:Z"}
    d = model.decompose_risk(pd.Series(1/6, index=ret.columns))
    assert sum(d["factor_contributions"].values()) == pytest.approx(d["factor_variance"])
    assert sum(d["specific_contributions"].values()) == pytest.approx(d["specific_variance"])
    assert sum(d["volatility_contributions"].values()) == pytest.approx(d["total_risk"])
    assert abs(d["euler_variance_error"]) < 1e-10
    assert np.linalg.eigvalsh(model.cov_matrix_).min() >= -1e-12


def test_risk_model_rejects_candidate_alpha_as_risk_factor():
    ret, _, _ = risk_inputs()
    with pytest.raises(ValueError, match="alpha_spread"):
        FactorRiskModel().fit(ret, pd.DataFrame({"alpha_spread": ret.mean(axis=1)}))


def test_optimizer_structural_covariance_does_not_depend_on_candidate_scores():
    ret, caps, sectors = risk_inputs()
    hist = caps.stack().rename("market_cap").reset_index()
    hist.columns = ["date", "ticker", "market_cap"]
    hist["signal"] = np.arange(len(hist))
    opt = OptimizedPortfolio(PortfolioConfig(max_stock_weight=1, max_sector_weight=None))
    opt.sector_map = sectors
    cov = opt._estimate_factor_risk_covariance(hist, ret, ret.columns.to_numpy(), "signal")
    hist["signal"] *= -100
    other = opt._estimate_factor_risk_covariance(hist, ret, ret.columns.to_numpy(), "signal")
    assert np.allclose(cov, other)
    assert "size" in opt.risk_factor_names_
    assert "alpha_spread" not in opt.risk_factor_names_
    assert np.linalg.eigvalsh(cov).min() >= -1e-12


def test_optimizer_missing_structural_inputs_does_not_switch_to_sample_risk():
    ret, _, _ = risk_inputs()
    opt = OptimizedPortfolio()
    with pytest.raises(PortfolioOptimizationError, match="historical market_cap and sector_map"):
        opt._estimate_factor_risk_covariance(pd.DataFrame(), ret, ret.columns.to_numpy(), "signal")


def test_sector_map_missing_name_is_explicit():
    ret, caps, sectors = risk_inputs()
    with pytest.raises(ValueError, match="incomplete"):
        FactorRiskModel().fit(ret, FactorRiskModel.build_structural_factors(ret, caps), sectors.iloc[:-1])


def test_zero_portfolio_has_complete_risk_decomposition():
    ret, caps, _ = risk_inputs()
    model = FactorRiskModel().fit(ret, FactorRiskModel.build_structural_factors(ret, caps))
    assert model.predict_risk(pd.Series(dtype=float)) == 0
    assert model.decompose_risk(pd.Series(dtype=float))["euler_volatility_error"] == 0


def alpha_panel(alpha=.002):
    rng = np.random.default_rng(23)
    x = rng.normal(.001, .01, 400)
    residual = rng.normal(0, .003, 400)
    for i in range(1, len(residual)):
        residual[i] += .7 * residual[i - 1]
    return pd.DataFrame({"baseline": x, "candidate": alpha + 1.3*x + residual})


def test_incremental_alpha_is_intercept_not_zero_residual_mean():
    panel = alpha_panel()
    result = _incremental_alpha_stats(panel, "candidate", ["baseline"])
    assert result["incremental_alpha"] > .001
    assert result["incremental_alpha_tstat"] > 2
    assert _incremental_sharpe(panel, "candidate", ["baseline"]) > 1


def test_hac_intercept_se_matches_independent_sandwich_reference():
    panel = alpha_panel()
    # Weight each observation's response with its intercept influence, then
    # compute a Bartlett long-run covariance independently of the meat matrix.
    x = np.column_stack([np.ones(len(panel)), panel["baseline"]])
    y = panel["candidate"].to_numpy()
    beta = np.linalg.lstsq(x, y, rcond=None)[0]
    influence = (np.linalg.inv(x.T @ x) @ x.T)[0] * (y - x @ beta)
    lag = 6
    variance = influence @ influence
    for j in range(1, lag + 1):
        variance += 2 * (1-j/(lag+1)) * (influence[j:] @ influence[:-j])
    expected_se = np.sqrt(variance * len(y) / (len(y) - x.shape[1]))
    result = _incremental_alpha_stats(panel, "candidate", ["baseline"], hac_lag=lag)
    assert result["incremental_alpha_hac_se"] == pytest.approx(expected_se)
    assert result["incremental_alpha_tstat"] == pytest.approx(beta[0] / expected_se)


def test_hac_intercept_se_accounts_for_serially_correlated_errors():
    panel = alpha_panel()
    iid = _incremental_alpha_stats(panel, "candidate", ["baseline"], hac_lag=0)
    hac = _incremental_alpha_stats(panel, "candidate", ["baseline"], hac_lag=6)
    assert hac["incremental_alpha_hac_se"] > iid["incremental_alpha_hac_se"]
    assert hac["incremental_alpha_tstat"] < iid["incremental_alpha_tstat"]


def test_incremental_alpha_tstat_gate_is_used_by_selection():
    panel = alpha_panel(alpha=-.003)
    ok, reason, diag = _can_add_factor("candidate", ["baseline"], {"factor_return_panel": panel},
                                     RedundancyConfig(min_incremental_sharpe=-100, min_incremental_alpha_tstat=0))
    assert not ok
    assert reason == "incremental_alpha_tstat_below_threshold"
    assert diag["incremental_alpha_tstat"] < 0


def test_collinear_regression_does_not_report_false_alpha_significance():
    panel = alpha_panel()
    panel["copy"] = panel["baseline"]
    assert np.isnan(_incremental_alpha_stats(panel, "candidate", ["baseline", "copy"])["incremental_alpha_tstat"])


def execution_inputs(n=14):
    dates = pd.bdate_range("2024-01-02", periods=n)
    prices = pd.DataFrame({"date": dates, "ticker": "A", "close": 100.0, "adj_close": 100.0,
                           "volume": 10_000.0, "is_tradable": True})
    weights = pd.DataFrame({"date": dates, "ticker": "A", "weight": 1., "is_rebalance": True})
    return dates, prices, weights


def test_adv_requires_history_and_excludes_untradable_volume():
    dates, px, w = execution_inputs()
    px.loc[0, ["volume", "is_tradable"]] = [1e12, False]
    trades = simulate_execution(w, px, zero_cost(), initial_capital=1_000_000)
    assert trades.loc[trades["date"] <= dates[10], "shares"].sum() == 0
    last = trades.loc[trades["date"] == dates[11]].iloc[0]
    assert last["adv_shares"] == 10_000
    assert last["shares"] == 1_000


def test_min_adv_threshold_rejects_tiny_liquidity():
    _, px, w = execution_inputs()
    px["volume"] = 500
    trades = simulate_execution(w, px, zero_cost(min_adv_shares=1_000))
    assert trades["shares"].sum() == 0
    assert set(trades["fill_status"]) == {"rejected_no_liquidity"}


def test_experiment_accepts_and_applies_minimum_adv_configuration(tmp_path):
    from quant_platform.core.experiments.runner import ExperimentRunner
    runner = ExperimentRunner("quant_platform/experiments/us_baseline.yaml", output_dir=tmp_path)
    runner.config["costs"]["min_adv_shares"] = 12_345.
    runner._validate_config_contract()
    assert runner._cost_model().min_adv_shares == 12_345.


def test_untradable_day_rejects_order_even_with_price_and_adv():
    dates, px, w = execution_inputs()
    w["weight"] = 0.
    w.loc[10:, "weight"] = 1.
    px.loc[10, "is_tradable"] = False
    trades = simulate_execution(w, px, zero_cost())
    reject = trades.loc[trades["date"] == dates[10]].iloc[0]
    assert reject["fill_status"] == "rejected_not_tradable"
    assert reject["shares"] == 0
    assert trades.loc[trades["date"] == dates[11], "shares"].sum() > 0


def test_requested_and_executed_participation_are_distinct():
    orders = generate_trades(pd.Series({"A": 1.}), None, pd.Series({"A": 100.}),
                             pd.Series({"A": 1_000.}), pd.Series({"A": .02}),
                             capital=1_000_000, max_participation_rate=.1)
    assert orders.loc[0, "requested_participation"] == 10
    assert orders.loc[0, "executed_participation"] == .1


def test_vectorized_returns_do_not_pad_price_gaps():
    dates, px, w = execution_inputs(4)
    px["adj_close"] = [100, np.nan, 120, 132]
    w["is_rebalance"] = [True, False, False, False]
    _, ledger = simulate_execution(w, px, zero_cost(max_participation_rate=None), return_daily_ledger=True)
    assert ledger.loc[dates[2], "gross_ret"] == 0
    assert ledger.loc[dates[3], "gross_ret"] == pytest.approx(.1)


def test_event_sim_uses_price_and_signal_tradability_without_padding():
    from quant_platform.core.execution.backtest.event_sim import VectorizedBacktest, BacktestConfig
    dates, px, _ = execution_inputs(4)
    px["adj_close"] = [100, np.nan, 120, 132]
    px.loc[1, "is_tradable"] = False
    panel = px[["date", "ticker"]].assign(signal=1., is_tradable=True)
    panel.loc[2, "is_tradable"] = False
    matrices = VectorizedBacktest(BacktestConfig(max_weight=1))._build_matrices(panel, px, "ticker", None)
    assert np.isnan(matrices[1][2, 0])
    assert not matrices[3][1, 0]
    assert not matrices[3][2, 0]
    assert matrices[1][3, 0] == pytest.approx(.1)


def test_event_sim_preserves_dates_with_all_missing_prices():
    from quant_platform.core.execution.backtest.event_sim import VectorizedBacktest, BacktestConfig
    dates, px, _ = execution_inputs(4)
    px["close"] = px["adj_close"] = [100, np.nan, 120, 132]
    panel = px[["date", "ticker"]].assign(signal=1.)
    matrices = VectorizedBacktest(BacktestConfig(max_weight=1))._build_matrices(panel, px, "ticker", None)
    assert list(matrices[6]) == list(dates)
    assert np.isnan(matrices[1][2, 0])
    assert not matrices[3][1, 0]


def capacity_trades():
    dates = pd.bdate_range("2024-01-02", periods=3)
    return pd.DataFrame({"date": [dates[0]]*20 + [dates[1], dates[2]], "ticker": [f"T{i}" for i in range(22)],
                         "side": "buy", "shares": [1.]*20 + [10., 20.], "requested_shares": [1.]*20 + [100., 200.],
                         "price": 100., "adv_shares": 1_000., "daily_vol": .02, "notional_trade": [100.]*20 + [10_000., 20_000.],
                         "notional": [100.]*20 + [1_000., 2_000.], "portfolio_value_before_trade": 1_000_000., "total_cost": 0.})


def test_capacity_p95_uses_same_daily_max_aggregation():
    tr = capacity_trades()
    summary = capacity_analysis(tr, capital_grid=())["summary"].iloc[0]
    expected = pd.Series([.001, .1, .2]).quantile(.95)
    assert summary["p95_requested_participation"] == pytest.approx(expected)
    assert summary["p95_adv_notional_pct"] == pytest.approx(expected)
    assert summary["max_participation_rate"] == .02
    assert summary["capacity_at_10pct_adv"] == pytest.approx(1e6 * .1 / expected)


def test_capacity_normalizes_out_nav_growth():
    tr = capacity_trades().iloc[-1:].copy()
    tr["requested_shares"] = 100.
    original = capacity_analysis(tr, capital_grid=())["summary"].iloc[0]["capacity_at_10pct_adv"]
    # Add a later day at double NAV and double requested shares.
    later = tr.copy()
    later["date"] += pd.Timedelta(days=1)
    later["portfolio_value_before_trade"] *= 2
    later["requested_shares"] *= 2
    combined = capacity_analysis(pd.concat([tr, later]), capital_grid=())["summary"].iloc[0]["capacity_at_10pct_adv"]
    assert combined == pytest.approx(original)


def test_zero_share_retry_days_cannot_dilute_capacity_percentile():
    tr = capacity_trades()
    original = capacity_analysis(tr, capital_grid=())["summary"].iloc[0]
    # Hundreds of sub-share retries between real rebalance orders must not
    # push every meaningful order outside the 95th percentile.
    zeros = pd.concat([tr.iloc[[0]].copy()] * 100, ignore_index=True)
    zeros["date"] = pd.bdate_range("2025-01-02", periods=100)
    zeros[["shares", "requested_shares", "notional_trade", "notional"]] = 0.
    result = capacity_analysis(pd.concat([tr, zeros]), capital_grid=())["summary"].iloc[0]
    assert result["capacity_at_10pct_adv"] == pytest.approx(original["capacity_at_10pct_adv"])
    assert result["p95_requested_participation"] == pytest.approx(original["p95_requested_participation"])
    assert result["capacity_at_max_10pct_adv"] <= result["capacity_at_10pct_adv"]


def test_capital_curve_reapplies_participation_limit_and_suppresses_invalid_sharpe():
    tr = capacity_trades().iloc[-1:].copy()
    tr["shares"] = 100.
    tr["requested_shares"] = 100.
    pnl = pd.DataFrame({"gross_ret": [.01, -.01]}, index=pd.bdate_range(tr["date"].iloc[0], periods=2))
    curve = capacity_analysis(tr, pnl, cost_model=zero_cost(), capital_grid=[1e6, 5e6])["capital_curve"]
    assert curve["max_executed_participation"].max() <= .1
    assert curve.loc[1, "fill_ratio"] == .2
    assert curve.loc[0, "return_estimate_valid"]
    assert not curve.loc[1, "return_estimate_valid"]
    assert np.isnan(curve.loc[1, "net_sharpe_est"])


def test_capital_curve_rejects_unknown_and_below_threshold_adv():
    tr = capacity_trades().iloc[-2:].copy()
    tr["adv_shares"] = [np.nan, 10]
    curve = capacity_analysis(tr, cost_model=zero_cost(min_adv_shares=100), capital_grid=[1e6])["capital_curve"]
    assert curve.loc[0, "total_notional"] == 0


def test_missing_crowding_is_excluded_and_weights_are_renormalized():
    prs = score_factor_readiness({})
    assert "crowding" not in prs["breakdown"]
    assert "crowding" not in prs["weights"]
    assert sum(prs["weights"].values()) == pytest.approx(1.)
    assert prs["excluded_dimensions"]


def test_explicit_crowding_changes_score_and_is_reported():
    low = score_factor_readiness({}, crowding_score=0.)
    high = score_factor_readiness({}, crowding_score=1.)
    assert low["total"] - high["total"] == pytest.approx(10.)
    assert not low["excluded_dimensions"]


def test_missing_dsr_cannot_get_points_from_hit_rate():
    prs = score_factor_readiness({}, rank_ic_stats={"hit_rate": 1.})
    assert prs["breakdown"]["significance"]["dsr_pts"] == 0


def test_dsr_uses_per_observation_sharpe_and_trial_dispersion():
    sr, n, trials, variance = .08, 300, 10, .002
    gamma = .5772156649015329
    threshold = np.sqrt(variance) * ((1-gamma)*stats.norm.ppf(1-1/trials) + gamma*stats.norm.ppf(1-1/(trials*np.e)))
    expected = stats.norm.cdf((sr-threshold) / np.sqrt((1+.5*sr**2)/(n-1)))
    actual = deflated_sharpe_ratio(sr, trials, n, sharpe_variance=variance)
    assert actual == pytest.approx(expected)
    assert actual < deflated_sharpe_ratio(sr, 1, n, sharpe_variance=variance)


def test_catalog_connects_dsr_to_score_and_reports_trial_assumption():
    rng = np.random.default_rng(99)
    ts = {"F": {"long_short": pd.Series(rng.normal(.001, .01, 300)), "advanced_metrics": {}}}
    one = score_factor_catalog(ts, n_trials=1).iloc[0]
    many = score_factor_catalog(ts, n_trials=100).iloc[0]
    assert 0 <= many["deflated_sharpe"] < one["deflated_sharpe"] <= 1
    assert many["significance"] < one["significance"]
    assert many["dsr_n_trials"] == 100
    assert "independent" in many["dsr_assumption"]


def test_empty_catalog_has_no_fake_scores():
    assert score_factor_catalog({}).empty


def test_batch_tearsheets_use_same_dsr_and_fdr_as_catalog(tmp_path, monkeypatch):
    from quant_platform.core.evaluation import tearsheet
    rng = np.random.default_rng(11)
    sheets = {"F": {"long_short": pd.Series(rng.normal(.001, .01, 120)), "advanced_metrics": {}}}
    comparison = pd.DataFrame({"fdr_q_value": [.02]}, index=["F"])
    catalog = score_factor_catalog(sheets, comparison).iloc[0]
    seen = {}
    def capture(ts, name, **kwargs):
        seen[name] = kwargs["prs_result"]
        return kwargs["output_path"]
    monkeypatch.setattr(tearsheet, "generate_factor_tearsheet", capture)
    paths = tearsheet.generate_all_tearsheets(sheets, tmp_path, comparisons=comparison)
    assert "F" in paths
    assert seen["F"]["total"] == catalog["total"]
    assert seen["F"]["breakdown"]["significance"]["deflated_sharpe"] == catalog["deflated_sharpe"]
