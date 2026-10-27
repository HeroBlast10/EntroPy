import numpy as np
import pandas as pd
import pytest

from quant_platform.core.portfolio.risk_model import FactorRiskModel
from quant_platform.core.portfolio.risk_neutral import NeutralRiskConfig, solve_factor_neutral, industry_quantile_weights
from quant_platform.core.execution.backtest.vectorized_daily import simulate_execution
from quant_platform.core.execution.cost_models.us_equity import CostModel


def model_inputs():
    rng = np.random.default_rng(28)
    names = pd.Index([f"S{i}" for i in range(80)])
    model = FactorRiskModel()
    model.factor_names_ = ["market", "size"]
    model.exposures_ = pd.DataFrame({"market": rng.uniform(.6, 1.4, 80), "size": rng.normal(0, .2, 80)}, index=names)
    model.specific_risk_ = pd.Series(.20, index=names)
    model.cov_matrix_ = np.diag([.04, .01])
    scores = pd.Series(rng.normal(size=80), index=names)
    industries = pd.Series(np.repeat(["A", "B", "C", "D"], 20), index=names)
    issuers = pd.Series(np.repeat(np.arange(40), 2), index=names)
    beta = model.exposures_.market
    return scores, model, industries, issuers, beta


def test_factor_qp_neutrality_caps_and_volatility_scaling_do_not_force_leverage():
    scores, model, industry, issuer, beta = model_inputs()
    cfg = NeutralRiskConfig(stock_limit=.05, issuer_limit=.10, turnover_cost_bps=0, target_volatility=.02)
    w, diagnostics = solve_factor_neutral(scores, model, industry, issuer, beta, pd.Series(dtype=float), cfg)
    assert w.abs().sum() <= 1 + 2e-5
    assert w.abs().max() <= .05 + 2e-6
    assert w.abs().groupby(issuer).sum().max() <= .1 + 2e-6
    assert w.groupby(industry).sum().abs().max() < 2e-6 and abs(w @ beta) < 2e-6
    assert diagnostics["predicted_volatility"] <= .02 + 1e-9
    assert diagnostics["volatility_scale"] <= 1
    high = NeutralRiskConfig(stock_limit=.05, issuer_limit=.10, turnover_cost_bps=100)
    kept, _ = solve_factor_neutral(scores, model, industry, issuer, beta, w, high)
    np.testing.assert_allclose(kept, w, atol=2e-5)
    state = {}
    warm, _ = solve_factor_neutral(scores, model, industry, issuer, beta, pd.Series(dtype=float), cfg, state)
    again, _ = solve_factor_neutral(scores, model, industry, issuer, beta, pd.Series(dtype=float), cfg, state)
    np.testing.assert_allclose(again, warm, atol=2e-6)


def test_industry_ranking_retains_neutrality_after_shared_issuer_and_stock_caps():
    day = pd.DataFrame({"permno": np.arange(40) + 1, "permco": np.arange(40) + 1,
                        "decision_industry": np.repeat(["A", "B"], 20), "score": np.tile(np.arange(20), 2)})
    day.loc[day.score.ge(17), "permco"] = 999
    w = industry_quantile_weights(day, "score", stock_limit=.04, issuer_limit=.05)
    assert w.groupby(day.decision_industry).sum().abs().max() < 1e-12
    assert w.abs().groupby(day.permco).sum().max() <= .05 + 1e-12
    assert w.abs().max() <= .04
    tied = day.assign(score=1.)
    assert industry_quantile_weights(tied, "score").eq(0).all()


def test_vectorized_risk_fit_preserves_pairwise_missingness_and_intercepts():
    rng = np.random.default_rng(12)
    dates = pd.bdate_range("2020-01-02", periods=150)
    factors = pd.DataFrame(rng.normal(0, .01, (150, 3)), index=dates, columns=["market", "size", "sector"])
    y = pd.DataFrame(factors.to_numpy() @ rng.normal(size=(3, 6)) + .001 + rng.normal(0, .005, (150, 6)), index=dates)
    y.iloc[::3, 1] = np.nan; y.iloc[:30, 2] = np.nan; y.iloc[:140, 3] = np.nan
    factors.iloc[4, 0] = np.nan
    factors = factors.astype("Float64")
    y = y.astype("Float64")
    model = FactorRiskModel(min_observations=60).fit(y, factors)
    assert 3 not in model.exposures_.index
    for name in model.exposures_.index:
        aligned = pd.concat([y[name].rename("stock"), factors], axis=1).dropna()
        x = np.c_[np.ones(len(aligned)), aligned[factors.columns].to_numpy(dtype=float)]
        coefficients = np.linalg.lstsq(x, aligned.stock.to_numpy(dtype=float), rcond=None)[0]
        np.testing.assert_allclose(model.exposures_.loc[name], coefficients[1:], atol=1e-12)
        expected = np.std(aligned.stock - x @ coefficients, ddof=1) * np.sqrt(252)
        assert model.specific_risk_.loc[name] == pytest.approx(expected)
    weights = pd.Series(.1, index=model.exposures_.index, dtype="Float64")
    decomposition = model.decompose_risk(weights)
    assert decomposition["total_risk"] > 0
    assert abs(decomposition["euler_variance_error"]) < 1e-10


def test_dated_sector_factors_do_not_backcast_current_classification_or_future_membership():
    dates = pd.bdate_range("2020-01-02", periods=4)
    returns = pd.DataFrame({"A": [.01, .02, .03, .04], "B": [0, 0, 0, 0], "C": [.1, .1, .1, .1]}, index=dates)
    caps = returns * 0 + [10, 100, 1e9]
    eligible = returns * 0 == 0; eligible["C"] = False
    factors = FactorRiskModel.build_structural_factors(returns, caps, eligibility=eligible)
    assert factors.market.iloc[1] == .01 and factors["size"].iloc[1] == .02
    mapping = pd.DataFrame({"date": dates.repeat(3), "ticker": ["A", "B", "C"] * 4,
                            "sector": ["X", "Y", "Y"] * 4})
    original = FactorRiskModel._sector_factors(returns, factors, mapping)
    changed = mapping.copy(); changed.loc[changed.date.eq(dates[-1]) & changed.ticker.eq("A"), "sector"] = "Y"
    altered = FactorRiskModel._sector_factors(returns, factors, changed)
    pd.testing.assert_frame_equal(original.iloc[:-1], altered.iloc[:-1])


def test_execution_policy_receives_actual_drift_and_records_post_fill_risk():
    dates = pd.bdate_range("2020-01-02", periods=2)
    prices = pd.DataFrame({"date": dates.repeat(2), "ticker": ["A", "B"] * 2,
                           "close": [10., 10., 20., 10.], "adj_close": [10., 10., 20., 10.], "volume": 1e6})
    weights = pd.DataFrame({"date": dates.repeat(2), "ticker": ["A", "B"] * 2, "weight": 0., "is_rebalance": True})

    class Policy:
        observations = []
        def __call__(self, *, date, rows, actual_weights, rebalance):
            self.observations.append(actual_weights.copy())
            return pd.Series({"A": .5, "B": .5}), {"target_test": 1.}
        def diagnose(self, date, actual):
            return {"actual_test": float(actual.sum())}

    policy = Policy()
    model = CostModel(commission_per_share=0, commission_min=0, slippage_bps=0, impact_coeff=0,
                       sec_fee_rate=0, finra_taf_per_share=0, borrow_rate_annual=0, max_participation_rate=None)
    trades, ledger = simulate_execution(weights, prices, model, initial_capital=10000,
                                         target_policy=policy, return_daily_ledger=True)
    assert policy.observations[1]["A"] == pytest.approx(2 / 3)
    # Integer fills can leave an actual exposure different from the target.
    assert ledger.risk_actual_test.iloc[-1] == pytest.approx(ledger.gross_exposure.iloc[-1])
    assert len(trades.attrs["target_history"]) == 4
