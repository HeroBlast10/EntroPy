"""Tests for purged non-linear alpha research."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import quant_platform.core.alpha_models.nonlinear_alpha as nonlinear_alpha
from quant_platform.core.alpha_models.nonlinear_alpha import (
    LightGBMConfig,
    WalkForwardConfig,
    build_learning_panel,
    evaluate_alpha,
    expanding_date_folds,
    run_walk_forward_oof,
    select_usable_features,
)


def _synthetic_panel(n_days: int = 80, n_tickers: int = 20) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(7)
    dates = pd.bdate_range("2022-01-03", periods=n_days)
    tickers = [f"S{i:02d}" for i in range(n_tickers)]
    index = pd.MultiIndex.from_product([dates, tickers], names=["date", "ticker"])
    factors = index.to_frame(index=False)
    factors["predictive"] = rng.normal(size=len(factors))
    factors["noise"] = rng.normal(size=len(factors))
    factors["constant"] = 1.0
    signal_matrix = factors["predictive"].to_numpy().reshape(n_days, n_tickers)
    innovation = rng.normal(scale=0.003, size=(n_days, n_tickers))
    # The signal observed on t predicts the price change from t to t+1.
    innovation[1:] += 0.002 * signal_matrix[:-1]
    prices = index.to_frame(index=False)
    prices["adj_close"] = (100.0 * np.cumprod(1.0 + innovation, axis=0)).reshape(-1)
    prices["close"] = prices["adj_close"]
    return factors, prices


def _all_tradable(factors: pd.DataFrame) -> pd.DataFrame:
    universe = factors[["date", "ticker"]].copy()
    universe["pass_all_filters"] = True
    return universe


def test_expanding_folds_purge_by_complete_dates() -> None:
    dates = pd.bdate_range("2020-01-01", periods=100)
    config = WalkForwardConfig(
        horizon=3,
        min_train_days=30,
        validation_days=10,
        n_splits=3,
    )
    folds = list(expanding_date_folds(dates.repeat(4), config))
    assert len(folds) == 3
    for fold in folds:
        train = fold["train_dates"]
        validation = fold["validation_dates"]
        assert train.max() < validation.min()
        assert dates.get_loc(validation.min()) - dates.get_loc(train.max()) == 4
        assert set(train).isdisjoint(validation)


def test_feature_selection_and_learning_panel() -> None:
    factors, prices = _synthetic_panel()
    selected = select_usable_features(factors, ["predictive", "noise", "constant"])
    assert selected == ["predictive", "noise"]
    panel, features = build_learning_panel(
        factors,
        prices,
        _all_tradable(factors),
        feature_cols=["predictive", "noise"],
        horizon=1,
        min_cross_section=10,
    )
    assert features == ["predictive", "noise"]
    assert panel[features].notna().all().all()
    assert panel[features].max().max() <= 1.0
    assert panel[features].min().min() >= -1.0
    assert panel.groupby("date")["target"].mean().abs().max() < 1e-5


def test_learning_panel_ranks_before_future_label_filter_and_uses_calendar() -> None:
    dates = pd.bdate_range("2024-01-02", periods=3)
    factors = pd.DataFrame(
        [
            {"date": date, "ticker": ticker, "feature": float(index)}
            for date in dates
            for index, ticker in enumerate(["A", "B", "C", "D"], start=1)
        ]
    )
    prices = pd.DataFrame([
        *[
            {"date": dates[0], "ticker": ticker, "adj_close": 100.0}
            for ticker in ["A", "B", "C", "D"]
        ],
        {"date": dates[1], "ticker": "A", "adj_close": 101.0},
        # B is suspended/missing on the next calendar trading date.
        {"date": dates[1], "ticker": "C", "adj_close": 103.0},
        {"date": dates[1], "ticker": "D", "adj_close": 104.0},
        {"date": dates[2], "ticker": "A", "adj_close": 102.0},
        {"date": dates[2], "ticker": "B", "adj_close": 110.0},
        {"date": dates[2], "ticker": "C", "adj_close": 104.0},
        {"date": dates[2], "ticker": "D", "adj_close": 106.0},
    ])
    universe = _all_tradable(factors)
    universe.loc[
        (universe["date"] == dates[0]) & (universe["ticker"] == "D"),
        "pass_all_filters",
    ] = False
    # Eligibility is known from the previous completed trading session. Seed
    # that session so the first signal date has a valid lagged cross-section.
    prior_universe = universe.loc[universe["date"] == dates[0]].copy()
    prior_universe["date"] = pd.Timestamp("2023-12-29")
    universe = pd.concat([prior_universe, universe], ignore_index=True)
    panel, _ = build_learning_panel(
        factors,
        prices,
        universe,
        feature_cols=["feature"],
        min_cross_section=2,
        min_feature_coverage=0.0,
    )
    first = panel.loc[panel["date"] == dates[0]].set_index("ticker")
    assert "B" in first.index
    assert "D" not in first.index
    # D is removed by the point-in-time tradable universe before ranking.
    # B remains both in that rank cross-section and in the inference panel even
    # though its next-calendar-day label is unavailable.
    assert first.loc["A", "feature"] == pytest.approx(-1.0 / 3.0)
    assert first.loc["B", "feature"] == pytest.approx(1.0 / 3.0)
    assert first.loc["C", "feature"] == pytest.approx(1.0)
    assert pd.isna(first.loc["B", "forward_return"])
    assert pd.isna(first.loc["B", "target"])


def test_walk_forward_predicts_complete_validation_universe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dates = pd.bdate_range("2024-01-02", periods=8)
    tickers = ["A", "B", "C", "D", "E", "F"]
    index = pd.MultiIndex.from_product(
        [dates, tickers],
        names=["date", "ticker"],
    )
    factors = index.to_frame(index=False)
    factors["feature"] = np.tile(
        np.arange(1.0, len(tickers) + 1.0),
        len(dates),
    )
    prices = index.to_frame(index=False)
    prices["adj_close"] = (
        100.0
        + np.repeat(np.arange(len(dates), dtype=float), len(tickers))
        + np.tile(np.arange(len(tickers), dtype=float), len(dates))
    )
    # B has a valid signal and current price on the first validation date, but
    # no quote on the next trading date, so its future label is unknowable.
    prices = prices.loc[
        ~(
            prices["date"].eq(dates[-1])
            & prices["ticker"].eq("B")
        )
    ].reset_index(drop=True)
    panel, features = build_learning_panel(
        factors,
        prices,
        _all_tradable(factors),
        feature_cols=["feature"],
        min_cross_section=3,
        min_feature_coverage=0.0,
    )

    def fake_fit_predict(
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_validation: np.ndarray,
        _config: LightGBMConfig,
        _seed: int,
        feature_names: list[str],
    ) -> tuple[np.ndarray, pd.Series]:
        assert np.isfinite(y_train).all()
        return (
            x_validation[:, 0].astype("float32"),
            pd.Series(1.0, index=feature_names, dtype=float),
        )

    monkeypatch.setattr(
        nonlinear_alpha,
        "_fit_predict_lightgbm",
        fake_fit_predict,
    )
    oof, _, metadata = run_walk_forward_oof(
        panel,
        features,
        model="lightgbm",
        walk_forward=WalkForwardConfig(
            horizon=1,
            min_train_days=3,
            validation_days=2,
            n_splits=1,
            seed=11,
        ),
    )

    validation_dates = dates[-2:]
    assert set(oof["date"]) == set(validation_dates)
    assert oof.groupby("date")["ticker"].nunique().eq(len(tickers)).all()
    missing_label_row = oof.loc[
        oof["date"].eq(validation_dates[0])
        & oof["ticker"].eq("B")
    ].iloc[0]
    assert pd.isna(missing_label_row["forward_return"])
    assert pd.isna(missing_label_row["target"])
    assert np.isfinite(missing_label_row["prediction"])
    assert metadata[0]["validation_rows"] == 2 * len(tickers)
    assert metadata[0]["validation_rows_labeled"] == len(tickers) - 1
    metrics, daily_ic = evaluate_alpha(oof, "prediction")
    assert metrics["n_rows"] == len(tickers) - 1
    assert daily_ic.index.tolist() == [validation_dates[0]]


def test_feature_selection_rejects_cross_sectionally_constant_columns() -> None:
    dates = pd.bdate_range("2024-01-02", periods=10)
    index = pd.MultiIndex.from_product(
        [dates, ["A", "B", "C", "D"]],
        names=["date", "ticker"],
    )
    frame = index.to_frame(index=False)
    frame["date_only"] = np.repeat(np.arange(len(dates)), 4)
    frame["cross_sectional"] = np.tile(np.arange(4), len(dates))
    frame["sparse_anomaly"] = 0.0
    frame.loc[frame.index[0], "sparse_anomaly"] = 1.0
    selected, diagnostics = select_usable_features(
        frame,
        feature_cols=["date_only", "cross_sectional", "sparse_anomaly"],
        return_diagnostics=True,
    )
    assert selected == ["cross_sectional"]
    reasons = diagnostics.set_index("feature")["reason"]
    assert reasons["date_only"] == "cross_sectionally_constant"
    assert reasons["sparse_anomaly"] == "dominant_value"


def test_lightgbm_walk_forward_finds_nonlinear_signal() -> None:
    pytest.importorskip("lightgbm")
    factors, prices = _synthetic_panel(n_days=100, n_tickers=25)
    panel, features = build_learning_panel(
        factors,
        prices,
        _all_tradable(factors),
        feature_cols=["predictive", "noise"],
        min_cross_section=10,
    )
    config = WalkForwardConfig(
        horizon=1,
        min_train_days=40,
        validation_days=15,
        n_splits=2,
        seed=11,
    )
    oof, importance, metadata = run_walk_forward_oof(
        panel,
        features,
        model="lightgbm",
        walk_forward=config,
        lightgbm=LightGBMConfig(
            n_estimators=35,
            min_child_samples=20,
            n_jobs=1,
        ),
    )
    metrics, _ = evaluate_alpha(oof, "prediction")
    assert len(metadata) == 2
    assert oof["prediction"].notna().all()
    assert metrics["mean_rank_ic"] > 0.10
    mean_importance = importance.groupby("feature")["importance"].mean()
    assert mean_importance["predictive"] > mean_importance["noise"]


def test_evaluate_alpha_has_expected_sign() -> None:
    factors, prices = _synthetic_panel()
    panel, _ = build_learning_panel(
        factors,
        prices,
        _all_tradable(factors),
        feature_cols=["predictive", "noise"],
        min_cross_section=10,
    )
    metrics, daily = evaluate_alpha(panel, "predictive")
    assert metrics["mean_rank_ic"] > 0
    assert metrics["ls_ann_return"] > 0
    assert len(daily) > 50


def test_evaluate_alpha_labels_and_scales_overlapping_horizon() -> None:
    factors, prices = _synthetic_panel()
    panel, _ = build_learning_panel(
        factors,
        prices,
        _all_tradable(factors),
        feature_cols=["predictive", "noise"],
        min_cross_section=10,
    )
    one_day, _ = evaluate_alpha(panel, "predictive", horizon=1)
    five_day, _ = evaluate_alpha(panel, "predictive", horizon=5)
    assert five_day["overlapping_forward_returns"] is True
    assert five_day["return_horizon_days"] == 5
    assert five_day["ls_ann_return"] == pytest.approx(
        one_day["ls_ann_return"] / 5
    )
    assert five_day["ls_gross_sharpe"] == pytest.approx(
        one_day["ls_gross_sharpe"] / np.sqrt(5)
    )
