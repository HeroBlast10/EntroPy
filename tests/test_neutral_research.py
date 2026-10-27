"""Label geometry, pre-decision information and long-history fold boundaries."""
import numpy as np
import pandas as pd
import pytest

from quant_platform.core.alpha_models.targets import TargetConfig, decision_volatility, neutral_scaled_targets
from quant_platform.core.alpha_models.nonlinear_alpha import WalkForwardConfig, expanding_date_folds, _daily_standardize
from quant_platform.core.alpha_models.fair_comparison import fit_predict
from quant_platform.core.alpha_models.feature_schema import prediction_columns


def test_common_market_subtraction_adds_nothing_to_existing_daily_zscore():
    rng = np.random.default_rng(4)
    dates = pd.Series(np.repeat(pd.bdate_range("2020-01-02", periods=3), 100))
    returns = pd.Series(rng.normal(size=300))
    market = pd.Series(np.repeat([.01, -.2, .15], 100))
    np.testing.assert_allclose(_daily_standardize(returns, dates, 3),
                               _daily_standardize(returns - market, dates, 3), atol=1e-14)


def test_neutral_volatility_targets_keep_industry_and_size_orthogonal_after_robustification():
    rng = np.random.default_rng(11)
    n = 200
    panel = pd.DataFrame({"date": pd.Timestamp("2020-01-02"), "ticker": [f"S{i}" for i in range(n)],
                          "forward_return": rng.standard_t(3, n) * .1})
    context = panel[["date", "ticker"]].assign(decision_industry=np.repeat(["A", "B", "C", "D"], 50),
        decision_log_size=rng.normal(size=n), label_exante_vol=rng.uniform(.003, .04, n))
    result = neutral_scaled_targets(panel, context, horizon=20)
    y = result.label_neutral_target
    assert y.abs().max() <= 3 + 1e-6
    assert y.groupby(context.decision_industry).mean().abs().max() < 1e-7
    assert abs(y @ context.decision_log_size) < 1e-6
    # Forward labels can change training targets, never the set of predictions.
    panel.loc[0, "forward_return"] = np.nan
    missing = neutral_scaled_targets(panel, context, horizon=20)
    assert len(missing) == n and pd.isna(missing.label_neutral_target.iloc[0])
    context.loc[1, "decision_industry"] = pd.NA
    assert pd.isna(neutral_scaled_targets(panel, context, horizon=20).label_neutral_target.iloc[1])


def test_decision_volatility_excludes_current_and_future_returns_and_does_not_bridge_gaps():
    dates = pd.bdate_range("2020-01-02", periods=10)
    prices = pd.DataFrame({"date": dates, "ticker": "A", "total_return": np.arange(10) / 100})
    cfg = TargetConfig(volatility_window=3, volatility_min_periods=3)
    original = decision_volatility(prices, cfg)
    changed = prices.copy(); changed.loc[6:, "total_return"] = 99
    np.testing.assert_allclose(original.label_exante_vol.iloc[:7], decision_volatility(changed, cfg).label_exante_vol.iloc[:7], equal_nan=True)
    assert original.label_exante_vol.iloc[3] == pytest.approx(np.std([0, .01, .02], ddof=1))
    gap = prices.drop(index=4)
    # A second security retains the shared session at the missing A quote.
    gap = pd.concat([gap, prices.assign(ticker="B")], ignore_index=True)
    out = decision_volatility(gap, cfg)
    assert pd.isna(out.loc[out.date.eq(dates[6]) & out.ticker.eq("A"), "label_exante_vol"].iloc[0])


def test_date_anchored_folds_start_after_minimum_purged_history_and_keep_final_stub():
    dates = pd.bdate_range("2010-01-01", periods=2000)
    cfg = WalkForwardConfig(horizon=20, min_train_days=756, validation_days=252, oos_start="2013-01-01")
    folds = list(expanding_date_folds(dates, cfg))
    assert folds[0]["validation_start"] == dates[max(776, dates.searchsorted("2013-01-01"))]
    assert folds[-1]["validation_end"] == dates[-1]
    predicted = pd.DatetimeIndex(np.concatenate([f["validation_dates"].values for f in folds]))
    assert predicted.is_unique and predicted.equals(dates[dates.get_loc(folds[0]["validation_start"]):])
    extended = list(expanding_date_folds(pd.bdate_range("2010-01-01", periods=2200), cfg))
    assert extended[0]["validation_start"] == folds[0]["validation_start"]
    for f in folds:
        assert len(f["train_dates"]) >= 756 and f["train_end"] < f["validation_start"]


def test_locked_sampling_cannot_cross_a_purged_mask_or_enter_predictors():
    dates = pd.bdate_range("2020-01-02", periods=40)
    panel = pd.DataFrame({"date": dates, "ticker": "A", "x": np.arange(40, dtype=float),
                          "target": np.arange(40, dtype=float), "label_reference_target": 0.})
    train = np.arange(40) < 30; predict = ~train
    common = dict(model="ridge", params={"alpha": 10}, seeds=[17], max_rows=30,
                  sampling_features=["x"], sampling_reference_col="label_reference_target")
    pred, importance, n = fit_predict(panel, ["x"], train, predict, training_indices=np.arange(30), **common)
    assert len(pred) == 10 and n == 30
    with pytest.raises(ValueError, match="purged mask"):
        fit_predict(panel, ["x"], train, predict, training_indices=np.arange(1, 31), **common)
    for name in ("label_neutral_target", "risk_beta", "decision_industry", "target_scaled"):
        with pytest.raises(ValueError, match="Identifiers"):
            prediction_columns(pd.DataFrame({name: [1]}), [name])
