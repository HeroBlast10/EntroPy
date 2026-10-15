"""Regression tests for point-in-time data and signal correctness."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_platform.core.data.fundamentals import (
    require_point_in_time_fundamentals,
)
from quant_platform.core.data.prices import (
    _calibrate_incremental_adjustments,
    _compute_adjustment_series,
    _restore_unadjusted_yahoo_history,
)
from quant_platform.core.data.universe import (
    _compute_days_since_ipo,
    _daily_market_cap_from_events,
    _membership_mask,
    lag_universe_eligibility,
    load_membership_intervals,
    require_point_in_time_universe,
)
from quant_platform.core.signals.cross_sectional.evaluation import (
    add_forward_returns,
)
from quant_platform.core.signals.cross_sectional.liquidity import SpreadHL
from quant_platform.core.signals.cross_sectional.liquidity import IlliqAmihud
from quant_platform.core.signals.cross_sectional.momentum import Mom1M
from quant_platform.core.signals.cross_sectional.value_quality import (
    BookToMarket,
    EarningsYield,
    _compute_ttm,
)
from quant_platform.core.signals.cross_sectional.volatility import (
    TailRisk,
    Vol20D,
    _rolling_downside_std,
)
from quant_platform.core.signals.registry import FactorRegistry
from quant_platform.core.signals.feature_cache import PriceFeatureCache
from quant_platform.core.signals.regime.hmm_regime import (
    _baum_welch_step,
    HMMTurbulenceProbability,
    RegimeDetector,
)
from quant_platform.core.signals.time_series.kalman_state_space import (
    _run_kalman_on_panel,
)
from quant_platform.core.signals.transforms import neutralize


def _panel(n_dates: int = 30) -> pd.DataFrame:
    dates = pd.bdate_range("2023-01-02", periods=n_dates)
    rows = []
    for i, date in enumerate(dates):
        for ticker, slope in (("A", 1.0), ("B", 2.0)):
            close = 100.0 + slope * i
            rows.append({
                "date": date,
                "ticker": ticker,
                "open": close,
                "high": close * 1.02,
                "low": close * 0.98,
                "close": close,
                "adj_close": close,
                "volume": 1_000,
                "amount": close * 1_000,
            })
    return pd.DataFrame(rows)


def test_factor_alignment_is_key_based_and_order_invariant():
    prices = _panel()
    factor = Mom1M(period=3, lag=0)
    actual = factor.compute(
        prices,
        zscore=False,
        winsorize_limits=(0.0, 1.0),
    )
    expected = prices.copy()
    expected["expected"] = expected.groupby("ticker")["adj_close"].pct_change(3)
    check = actual.merge(expected[["date", "ticker", "expected"]])
    np.testing.assert_allclose(
        check["MOM_1M"],
        check["expected"],
        equal_nan=True,
    )

    shuffled = prices.sample(frac=1.0, random_state=7)
    shuffled_result = factor.compute(
        shuffled,
        zscore=False,
        winsorize_limits=(0.0, 1.0),
    )
    left = actual.sort_values(["date", "ticker"]).reset_index(drop=True)
    right = shuffled_result.sort_values(["date", "ticker"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(left, right)


def test_kalman_is_causal_and_has_nonzero_velocity():
    prices = _panel(90)
    one = prices[prices["ticker"].eq("A")]
    _, _, velocity_short, _, _ = _run_kalman_on_panel(one.iloc[:70])
    _, _, velocity_full, _, _ = _run_kalman_on_panel(one)
    np.testing.assert_allclose(velocity_short, velocity_full[:70], atol=1e-12)
    assert np.nanmedian(velocity_full[-20:]) > 0


def test_hmm_factor_is_prefix_invariant_and_turbulent_label_is_sigma_based():
    rng = np.random.default_rng(42)
    rets = np.r_[rng.normal(0, 0.005, 90), rng.normal(0, 0.03, 30)]
    close = 100 * np.cumprod(1 + rets)
    prices = pd.DataFrame({
        "date": pd.bdate_range("2020-01-02", periods=len(close)),
        "ticker": "A",
        "adj_close": close,
    })
    factor = HMMTurbulenceProbability(
        lag=0,
        min_history=40,
        train_window=60,
        refit_interval=20,
        max_iter=5,
    )
    short = factor.compute(prices.iloc[:100])
    full = factor.compute(prices)
    common = short.merge(full, on=["date", "ticker"], suffixes=("_short", "_full"))
    np.testing.assert_allclose(
        common["HMM_TURBULENCE_PROB_short"],
        common["HMM_TURBULENCE_PROB_full"],
        atol=1e-12,
    )
    assert full["HMM_TURBULENCE_PROB"].dropna().between(0.0, 1.0).all()

    detector = RegimeDetector()
    detector.sigma = np.array([0.20, 0.01])
    assert detector.turbulent_state == 0
    detector.sigma = detector.sigma[::-1]
    assert detector.turbulent_state == 1


def test_hmm_baum_welch_likelihood_is_monotone():
    rng = np.random.default_rng(20260731)
    obs = np.r_[
        rng.normal(-0.01, 0.006, 180),
        rng.normal(0.02, 0.025, 120),
    ].astype(np.float64)
    pi = np.array([0.5, 0.5], dtype=np.float64)
    transition = np.array([[0.90, 0.10], [0.10, 0.90]], dtype=np.float64)
    mu = np.array([-0.005, 0.01], dtype=np.float64)
    sigma = np.array([0.01, 0.03], dtype=np.float64)
    likelihoods = []
    for _ in range(12):
        pi, transition, mu, sigma, likelihood = _baum_welch_step(
            obs,
            pi,
            transition,
            mu,
            sigma,
        )
        likelihoods.append(likelihood)
    assert np.diff(likelihoods).min() >= -1e-7
    np.testing.assert_allclose(transition.sum(axis=1), 1.0, atol=1e-12)


def test_spread_downside_and_tail_direction_regressions():
    prices = _panel(25)
    prices = prices[prices["ticker"].eq("A")].copy()
    prices["high"] = 102.0
    prices["low"] = 98.0
    spread = SpreadHL()._compute(prices)
    assert spread.dropna().iloc[-1] == pytest.approx(0.04, abs=0.01)

    returns = np.resize(np.array([0.01, -0.02]), 60)
    downside = _rolling_downside_std(returns, 60, 40)[-1]
    assert downside == pytest.approx(np.sqrt(0.5 * 0.02**2))
    assert Vol20D.meta.direction == -1
    assert TailRisk.meta.direction == 1


def test_ttm_accumulates_distinct_reports_over_availability_dates():
    fund = pd.DataFrame({
        "date": pd.to_datetime(
            ["2023-05-15", "2023-08-15", "2023-11-15", "2024-02-15"]
        ),
        "ticker": "A",
        "report_date": pd.to_datetime(
            ["2023-03-31", "2023-06-30", "2023-09-30", "2023-12-31"]
        ),
        "net_income": [1.0, 2.0, 3.0, 4.0],
    })
    ttm = _compute_ttm(fund, "net_income")
    assert ttm.iloc[-1]["net_income_ttm"] == pytest.approx(10.0)


@pytest.mark.parametrize("factor_cls", [EarningsYield, BookToMarket])
def test_daily_value_ratios_reprice_with_raw_close(factor_cls):
    prices = pd.DataFrame({
        "date": pd.bdate_range("2024-01-02", periods=3),
        "ticker": "A",
        "close": [10.0, 20.0, 40.0],
    })
    fund = pd.DataFrame({
        "date": [prices["date"].iloc[0]],
        "ticker": ["A"],
        "report_date": [pd.Timestamp("2023-12-31")],
        "net_income": [100.0],
        "total_equity": [1_000.0],
        "market_cap": [10_000.0],
        "shares_outstanding": [1_000.0],
    })
    values = factor_cls()._compute(prices, fund).dropna().to_numpy()
    assert values[0] / values[1] == pytest.approx(2.0)
    assert values[1] / values[2] == pytest.approx(2.0)


def test_forward_returns_use_adjusted_price_and_never_cross_tickers():
    prices = pd.DataFrame({
        "date": pd.to_datetime(
            ["2024-01-02", "2024-01-03", "2024-01-02", "2024-01-03"]
        ),
        "ticker": ["A", "A", "B", "B"],
        "close": [100.0, 51.0, 50.0, 51.0],
        "adj_close": [100.0, 102.0, 50.0, 51.0],
    })
    result = add_forward_returns(prices, periods=[1])
    first = result.groupby("ticker", sort=False).head(1).set_index("ticker")
    last = result.groupby("ticker", sort=False).tail(1)
    assert first.loc["A", "fwd_ret_1d"] == pytest.approx(0.02)
    assert first.loc["B", "fwd_ret_1d"] == pytest.approx(0.02)
    assert last["fwd_ret_1d"].isna().all()


def test_continuous_neutralization_and_cache_are_consistent():
    cross_section = pd.DataFrame({
        "date": pd.Timestamp("2024-01-02"),
        "ticker": list("ABCDE"),
        "signal": [1.0, 3.0, 5.0, 7.0, 9.0],
        "log_mcap": [0.0, 1.0, 2.0, 3.0, 4.0],
    })
    residual = neutralize(
        cross_section,
        "signal",
        group_cols=["log_mcap"],
    )["signal"]
    assert residual.abs().max() < 1e-12

    prices = _panel(70)
    prices["adj_close"] = prices["close"] * 0.5
    uncached = IlliqAmihud()._compute(prices)
    cached = IlliqAmihud()._compute(
        prices,
        _feature_cache=PriceFeatureCache(prices),
    )
    pd.testing.assert_series_equal(
        uncached.sort_index(),
        cached.sort_index(),
        check_names=False,
    )


def test_pit_adjustment_and_incremental_calibration():
    # yfinance auto_adjust=False still returns pre-split history on the
    # latest split basis.  Restore as-traded units before constructing the
    # causal forward adjustment.
    provider = pd.DataFrame({
        "Open": [49.0, 51.0, 52.0],
        "High": [51.0, 53.0, 54.0],
        "Low": [48.0, 50.0, 51.0],
        "Close": [50.0, 52.0, 53.0],
        "Volume": [2_000, 3_000, 4_000],
        "Stock Splits": [0.0, 2.0, 0.0],
        "Dividends": [0.5, 0.0, 0.0],
    })
    raw = _restore_unadjusted_yahoo_history(provider)
    np.testing.assert_allclose(raw["Close"], [100.0, 52.0, 53.0])
    np.testing.assert_allclose(raw["Volume"], [1_000, 3_000, 4_000])
    np.testing.assert_allclose(raw["Dividends"], [1.0, 0.0, 0.0])
    factor, adjusted = _compute_adjustment_series(
        raw,
        method="split_only",
        point_in_time=True,
    )
    np.testing.assert_allclose(factor, [1.0, 2.0, 2.0])
    np.testing.assert_allclose(adjusted, [100.0, 104.0, 106.0])

    dates = pd.to_datetime(["2024-01-02", "2024-01-03"])
    old = pd.DataFrame({
        "date": dates,
        "ticker": "A",
        "adj_factor": [5.0, 5.0],
        "split_factor": [4.0, 4.0],
        "adj_close": [500.0, 505.0],
    })
    new = pd.DataFrame({
        "date": dates,
        "ticker": "A",
        "adj_factor": [1.0, 2.0],
        "split_factor": [1.0, 2.0],
        "adj_close": [101.0, 204.0],
    })
    calibrated = _calibrate_incremental_adjustments(old, new)
    np.testing.assert_allclose(calibrated["adj_factor"], [5.0, 10.0])
    np.testing.assert_allclose(calibrated["split_factor"], [4.0, 8.0])


def test_membership_and_fundamental_pit_contracts(tmp_path):
    membership_file = tmp_path / "membership.csv"
    membership_file.write_text(
        "ticker,membership_start,membership_end\n"
        "A,2024-01-03,2024-01-04\n",
        encoding="utf-8",
    )
    intervals = load_membership_intervals(membership_file)
    rows = pd.DataFrame({
        "date": pd.date_range("2024-01-02", periods=4),
        "ticker": "A",
    })
    assert _membership_mask(rows, intervals).tolist() == [False, True, True, False]
    assert _compute_days_since_ipo(rows).tolist() == [0, 1, 2, 3]

    eligible = rows.assign(pass_all_filters=[True, False, True, True])
    lagged = lag_universe_eligibility(eligible)
    assert lagged["information_date"].tolist() == rows["date"].iloc[:-1].tolist()
    assert lagged["date"].tolist() == rows["date"].iloc[1:].tolist()
    assert lagged["pass_all_filters"].tolist() == [True, False, True]

    research_universe = rows.assign(
        membership_source="current_snapshot",
        membership_is_point_in_time=False,
    )
    with pytest.raises(ValueError, match="non-PIT universe"):
        require_point_in_time_universe(research_universe)

    research_fund = pd.DataFrame({
        "publish_date": [pd.Timestamp("2024-01-02")],
        "data_source": ["yfinance_quarterly_snapshot"],
        "is_point_in_time": [False],
    })
    with pytest.raises(ValueError, match="non-PIT fundamental"):
        require_point_in_time_fundamentals(research_fund)


def test_production_fundamental_build_rejects_non_pit_fallback(monkeypatch):
    import quant_platform.core.data.fundamentals as module

    monkeypatch.setattr(module, "_setup_simfin", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(module, "_fetch_simfin_statements", lambda *_args, **_kwargs: {})
    with pytest.raises(RuntimeError, match="No PIT SimFin filings"):
        module.build_fundamentals(tickers=["A"], mode="production")


def test_universe_market_cap_reprices_latest_known_shares_daily():
    prices = pd.DataFrame({
        "date": pd.bdate_range("2024-01-08", periods=3),
        "ticker": "A",
        "close": [10.0, 20.0, 40.0],
    })
    with_shares = pd.DataFrame({
        "date": [pd.Timestamp("2024-01-08")],
        "ticker": ["A"],
        "shares_outstanding": [100.0],
        "market_cap": [1_000.0],
    })
    np.testing.assert_allclose(
        _daily_market_cap_from_events(prices, with_shares),
        [1_000.0, 2_000.0, 4_000.0],
    )

    # A weekend event anchors implied shares to the first subsequent close.
    inferred = with_shares.drop(columns="shares_outstanding").copy()
    inferred["date"] = pd.Timestamp("2024-01-06")
    np.testing.assert_allclose(
        _daily_market_cap_from_events(prices, inferred),
        [1_000.0, 2_000.0, 4_000.0],
    )

    # Canonical raw prices jump mechanically at the split, while
    # close*split_factor and normalised shares remain on one basis.
    split_prices = pd.DataFrame({
        "date": pd.bdate_range("2024-02-01", periods=3),
        "ticker": "A",
        "close": [100.0, 50.0, 55.0],
        "split_factor": [1.0, 2.0, 2.0],
    })
    split_event = pd.DataFrame({
        "date": [split_prices["date"].iloc[0]],
        "report_date": [split_prices["date"].iloc[0]],
        "ticker": ["A"],
        "shares_outstanding": [10.0],
        "market_cap": [np.nan],
    })
    np.testing.assert_allclose(
        _daily_market_cap_from_events(split_prices, split_event),
        [1_000.0, 1_000.0, 1_100.0],
    )


def test_registry_flags_dominant_value_and_can_fail_closed():
    values = np.r_[np.zeros(96), np.arange(1.0, 5.0)]
    factor_df = pd.DataFrame({
        "date": np.repeat(pd.date_range("2024-01-02", periods=20), 5),
        "ticker": np.tile(list("ABCDE"), 20),
        "X": values,
    })
    report = FactorRegistry()._factor_quality(
        factor_df,
        "X",
        expected_rows=len(factor_df),
    )
    assert report["accepted"]
    assert report["severity"] == "warning"
    assert report["dominant_value_ratio"] == pytest.approx(0.96)

    strict_report = FactorRegistry(
        reject_quality_warnings=True
    )._factor_quality(
        factor_df,
        "X",
        expected_rows=len(factor_df),
    )
    assert not strict_report["accepted"]
