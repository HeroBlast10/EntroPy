from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _zero_cost_model(**overrides):
    from quant_platform.core.execution.cost_models.us_equity import CostModel

    values = {
        "commission_per_share": 0.0,
        "commission_pct": 0.0,
        "commission_min": 0.0,
        "slippage_bps": 0.0,
        "impact_coeff": 0.0,
        "sec_fee_rate": 0.0,
        "finra_taf_per_share": 0.0,
        "stamp_duty_pct": 0.0,
        "borrow_rate_annual": 0.0,
        "max_participation_rate": None,
    }
    values.update(overrides)
    return CostModel(**values)


def test_execution_and_pnl_share_close_target_timing():
    from quant_platform.core.execution.backtest.pnl import compute_daily_returns
    from quant_platform.core.execution.backtest.vectorized_daily import simulate_execution

    dates = pd.to_datetime(["2024-01-02", "2024-01-03"])
    prices = pd.DataFrame(
        {
            "date": [dates[0], dates[0], dates[1], dates[1]],
            "ticker": ["A", "B", "A", "B"],
            "close": [100.0, 100.0, 110.0, 90.0],
            "adj_close": [100.0, 100.0, 110.0, 90.0],
            "volume": [1_000_000] * 4,
        }
    )
    weights = pd.DataFrame(
        {
            "date": dates,
            "ticker": ["A", "B"],
            "weight": [1.0, 1.0],
            "is_rebalance": [True, True],
        }
    )

    trades, ledger = simulate_execution(
        weights,
        prices,
        _zero_cost_model(),
        initial_capital=1_000.0,
        return_daily_ledger=True,
    )
    pnl = compute_daily_returns(
        weights,
        prices,
        trades,
        _zero_cost_model(),
        initial_capital=1_000.0,
        execution_ledger=ledger,
    )

    assert ledger.loc[dates[0], "gross_ret"] == 0.0
    assert ledger.loc[dates[1], "gross_ret"] == pytest.approx(0.10)
    assert ledger.loc[dates[1], "nav_net"] == pytest.approx(1_100.0)
    pd.testing.assert_series_equal(pnl["net_ret"], ledger["net_ret"])


def test_rebalance_trades_from_drifted_actual_holdings():
    from quant_platform.core.execution.backtest.vectorized_daily import simulate_execution

    dates = pd.to_datetime(["2024-01-02", "2024-01-03"])
    prices = pd.DataFrame(
        {
            "date": [dates[0], dates[0], dates[1], dates[1]],
            "ticker": ["A", "B", "A", "B"],
            "close": [100.0, 100.0, 200.0, 100.0],
            "adj_close": [100.0, 100.0, 200.0, 100.0],
            "volume": [1_000_000] * 4,
        }
    )
    weights = pd.DataFrame(
        {
            "date": [dates[0], dates[0], dates[1], dates[1]],
            "ticker": ["A", "B", "A", "B"],
            "weight": [0.5, 0.5, 0.5, 0.5],
            "is_rebalance": [True, True, True, True],
        }
    )

    trades = simulate_execution(
        weights, prices, _zero_cost_model(), initial_capital=10_000.0
    )
    second = trades.loc[trades["date"].eq(dates[1])]
    assert set(second["ticker"]) == {"A", "B"}
    assert second.set_index("ticker").loc["A", "side"] == "sell"
    assert second.set_index("ticker").loc["B", "side"] == "buy"


def test_non_rebalance_retries_only_ticker_residual_without_retargeting_drift():
    from quant_platform.core.execution.backtest.vectorized_daily import simulate_execution

    history_date = pd.Timestamp("2024-01-01")
    dates = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
    all_dates = [history_date, *dates]
    prices = pd.DataFrame(
        {
            "date": [date for date in all_dates for _ in range(2)],
            "ticker": ["A", "B"] * len(all_dates),
            "close": [10.0, 10.0, 10.0, 10.0, 20.0, 10.0, 20.0, 10.0],
            "adj_close": [
                10.0,
                10.0,
                10.0,
                10.0,
                20.0,
                10.0,
                20.0,
                10.0,
            ],
            # B has no ex-ante ADV on the rebalance or first retry.  Its
            # volume on day two becomes usable only on day three.
            "volume": [
                10_000.0,
                np.nan,
                10_000.0,
                np.nan,
                10_000.0,
                10_000.0,
                10_000.0,
                10_000.0,
            ],
        }
    )
    weights = pd.DataFrame(
        {
            "date": [date for date in dates for _ in range(2)],
            "ticker": ["A", "B"] * len(dates),
            "weight": [0.5, 0.5] * len(dates),
            "is_rebalance": [True, True, False, False, False, False],
        }
    )

    trades, ledger = simulate_execution(
        weights,
        prices,
        _zero_cost_model(max_participation_rate=0.10),
        initial_capital=10_000.0,
        return_daily_ledger=True,
    )

    non_rebalance = trades.loc[trades["date"].isin(dates[1:])]
    assert set(non_rebalance["ticker"]) == {"B"}
    assert trades.loc[
        trades["date"].eq(dates[1]), "fill_status"
    ].tolist() == ["rejected_no_liquidity"]
    final_retry = trades.loc[trades["date"].eq(dates[2])].iloc[0]
    assert final_retry["ticker"] == "B"
    assert final_retry["notional_trade"] == pytest.approx(5_000.0)
    assert final_retry["executed_notional"] == pytest.approx(5_000.0)

    # A doubles between closes but must not be sold back to 50% merely because
    # B still has a pending residual order.
    assert ledger.loc[dates[1], "executed_turnover"] == 0.0
    assert ledger.loc[dates[1], "has_pending_orders"]
    assert not ledger.loc[dates[2], "has_pending_orders"]
    assert ledger.loc[dates[2], "gross_exposure"] == pytest.approx(1.0)


def test_new_rebalance_cancels_stale_ticker_residual():
    from quant_platform.core.execution.backtest.vectorized_daily import simulate_execution

    history_date = pd.Timestamp("2024-01-01")
    dates = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
    all_dates = [history_date, *dates]
    prices = pd.DataFrame(
        {
            "date": [date for date in all_dates for _ in range(2)],
            "ticker": ["A", "B"] * len(all_dates),
            "close": [10.0] * 8,
            "adj_close": [10.0] * 8,
            "volume": [10_000.0, np.nan] * len(all_dates),
        }
    )
    # B is pending after the first target, but the next rebalance replaces
    # that target with 100% A.  The stale B instruction must never retry.
    weights = pd.DataFrame(
        {
            "date": [dates[0], dates[0], dates[1], dates[2]],
            "ticker": ["A", "B", "A", "A"],
            "weight": [0.5, 0.5, 1.0, 1.0],
            "is_rebalance": [True, True, True, False],
        }
    )
    trades, ledger = simulate_execution(
        weights,
        prices,
        _zero_cost_model(max_participation_rate=0.10),
        initial_capital=10_000.0,
        return_daily_ledger=True,
    )

    b_dates = trades.loc[trades["ticker"].eq("B"), "date"].tolist()
    assert b_dates == [dates[0]]
    assert not ledger.loc[dates[1], "has_pending_orders"]
    assert trades.loc[trades["date"].eq(dates[2])].empty


def test_execution_uses_cost_model_adv_lookback():
    from quant_platform.core.execution.backtest.vectorized_daily import simulate_execution

    dates = pd.bdate_range("2024-01-02", periods=3)
    prices = pd.DataFrame(
        {
            "date": dates,
            "ticker": "A",
            "close": 100.0,
            "adj_close": 100.0,
            "volume": [100.0, 200.0, 300.0],
        }
    )
    weights = pd.DataFrame(
        {
            "date": dates,
            "ticker": "A",
            "weight": [0.0, 0.0, 1.0],
            "is_rebalance": [False, False, True],
        }
    )
    trades = simulate_execution(
        weights,
        prices,
        _zero_cost_model(adv_lookback=1),
        initial_capital=10_000.0,
    )
    assert trades.loc[
        trades["date"].eq(dates[2]), "adv_shares"
    ].iloc[0] == 200.0


def test_missing_price_and_participation_do_not_fake_fills():
    from quant_platform.core.execution.backtest.vectorized_daily import generate_trades

    missing = generate_trades(
        pd.Series({"A": 1.0}),
        pd.Series(dtype=float),
        pd.Series({"A": np.nan}),
        pd.Series({"A": 1_000.0}),
        pd.Series({"A": 0.02}),
        capital=100_000.0,
        max_participation_rate=0.10,
    )
    assert missing.loc[0, "shares"] == 0
    assert missing.loc[0, "fill_status"] == "rejected_missing_price"

    partial = generate_trades(
        pd.Series({"A": 1.0}),
        pd.Series(dtype=float),
        pd.Series({"A": 100.0}),
        pd.Series({"A": 1_000.0}),
        pd.Series({"A": 0.02}),
        capital=1_000_000.0,
        max_participation_rate=0.10,
    )
    assert partial.loc[0, "requested_shares"] == 10_000
    assert partial.loc[0, "shares"] == 100
    assert partial.loc[0, "fill_status"] == "partial"
    assert partial.loc[0, "executed_notional"] <= partial.loc[0, "notional_trade"]


def test_constraint_projection_enforces_caps_and_missing_sector_fails_closed():
    from quant_platform.core.portfolio.constraints import apply_constraints
    from quant_platform.core.portfolio.construction import PortfolioConfig

    config = PortfolioConfig(
        max_stock_weight=0.60,
        max_sector_weight=0.50,
    )
    raw = pd.Series({"A": 0.90, "B": 0.10, "C": 0.0, "D": 0.0})
    sectors = pd.DataFrame(
        {
            "ticker": ["A", "B", "C", "D"],
            "sector": ["S1", "S2", "S3", "S4"],
        }
    )
    projected = apply_constraints(raw, config, sector_map=sectors)
    assert projected.sum() == pytest.approx(1.0)
    assert projected.abs().max() <= 0.60 + 1e-8
    assert projected.loc["A"] <= 0.50 + 1e-8

    with pytest.raises(ValueError, match="Missing sector mapping"):
        apply_constraints(raw, config, sector_map=sectors.iloc[:-1])


def test_long_short_validator_rejects_one_sided_portfolio():
    from quant_platform.core.portfolio.rebalance import validate_portfolio_weights

    weights = pd.DataFrame(
        {
            "date": ["2024-01-02", "2024-01-02"],
            "ticker": ["A", "B"],
            "weight": [0.5, 0.5],
        }
    )
    with pytest.raises(ValueError, match="net exposure"):
        validate_portfolio_weights(weights, mode="long_short")


def test_market_cap_fallback_and_short_signal_strength_are_directional():
    from quant_platform.core.portfolio.construction import (
        PortfolioConfig,
        WeightScheme,
    )
    from quant_platform.core.portfolio.quantile import QuantilePortfolio

    date = pd.Timestamp("2024-01-02")
    selected = pd.Series({"A": -3.0, "B": -1.0})
    universe = pd.DataFrame(
        {
            "date": [date, date],
            "ticker": ["A", "B"],
            "market_cap": [np.nan, np.nan],
        }
    )
    signal = pd.DataFrame(
        {"date": [date, date], "ticker": ["A", "B"], "alpha": [-3.0, -1.0]}
    )

    mcap = QuantilePortfolio(
        PortfolioConfig(weight_scheme=WeightScheme.MARKET_CAP)
    )._assign_weights(selected, signal, universe, date)
    assert mcap.to_dict() == {"A": 0.5, "B": 0.5}

    short = QuantilePortfolio(
        PortfolioConfig(weight_scheme=WeightScheme.SIGNAL)
    )._assign_weights(selected, signal, universe, date, side="short")
    assert abs(short["A"]) > abs(short["B"])
    assert short.sum() == pytest.approx(-1.0)


def test_event_backtest_has_aligned_dates_and_enforces_lots():
    from quant_platform.core.execution.backtest.event_sim import (
        BacktestConfig,
        VectorizedBacktest,
    )
    dates = pd.bdate_range("2024-01-02", periods=4)
    prices = pd.DataFrame(
        [
            {"date": date, "ticker": ticker, "close": price}
            for date, price in zip(dates, [10.0, 10.2, 10.4, 10.6])
            for ticker in ["A", "B"]
        ]
    )
    panel = pd.DataFrame(
        [
            {
                "date": date,
                "ticker": ticker,
                "signal": 2.0 if ticker == "A" else 1.0,
            }
            for date in dates
            for ticker in ["A", "B"]
        ]
    )
    result = VectorizedBacktest(
        BacktestConfig(
            rebalance_freq="D",
            top_n=1,
            max_weight=1.0,
            lot_size=100,
            initial_capital=100_000.0,
        )
    ).run(panel, prices)

    assert result.nav.index.equals(dates)
    assert result.turnover.index.equals(dates)
    assert len(result.nav) == len(result.daily_returns) == len(dates)
    assert result.nav.iloc[-1] != 100_000.0


def test_event_backtest_uses_raw_close_for_lot_sizing():
    from quant_platform.core.execution.backtest.event_sim import (
        BacktestConfig,
        VectorizedBacktest,
    )
    from quant_platform.core.execution.cost_models.cn_a_share import AShareCostModel

    dates = pd.bdate_range("2024-01-02", periods=2)
    panel = pd.DataFrame(
        {"date": dates, "ticker": ["A", "A"], "signal": [1.0, 1.0]}
    )
    prices = pd.DataFrame(
        {
            "date": dates,
            "ticker": ["A", "A"],
            "close": [6.0, 6.0],
            "adj_close": [60.0, 60.0],
        }
    )
    zero_cn_cost = AShareCostModel(
        commission_rate=0.0,
        stamp_tax_rate=0.0,
        transfer_fee_rate=0.0,
        slippage_bps=0.0,
        min_commission=0.0,
    )
    result = VectorizedBacktest(
        BacktestConfig(
            rebalance_freq="D",
            top_n=1,
            max_weight=1.0,
            lot_size=100,
            initial_capital=1_000.0,
        ),
        zero_cn_cost,
    ).run(panel, prices)
    # Signal lag opens on day two: raw close permits one 100-share lot,
    # whereas the adjusted price would incorrectly round the order to zero.
    assert result.weights["weight"].max() == pytest.approx(0.6)


def test_event_backtest_eligibility_uses_lagged_signal_not_current_signal():
    from quant_platform.core.execution.backtest.event_sim import (
        BacktestConfig,
        VectorizedBacktest,
    )
    from quant_platform.core.execution.cost_models.cn_a_share import AShareCostModel

    dates = pd.bdate_range("2024-01-02", periods=2)
    panel = pd.DataFrame(
        {
            "date": dates,
            "ticker": ["A", "A"],
            "signal": [1.0, np.nan],
        }
    )
    prices = pd.DataFrame(
        {
            "date": dates,
            "ticker": ["A", "A"],
            "close": [10.0, 10.0],
        }
    )
    result = VectorizedBacktest(
        BacktestConfig(
            rebalance_freq="D",
            signal_lag=1,
            top_n=1,
            max_weight=1.0,
            lot_size=1,
            initial_capital=1_000.0,
        ),
        AShareCostModel(
            commission_rate=0.0,
            stamp_tax_rate=0.0,
            transfer_fee_rate=0.0,
            slippage_bps=0.0,
            min_commission=0.0,
        ),
    ).run(panel, prices)
    assert result.weights.loc[
        result.weights["date"].eq(dates[1]), "weight"
    ].max() == pytest.approx(1.0)


def test_event_non_rebalance_does_not_trade_drift_and_limits_are_directional():
    from quant_platform.core.execution.backtest.event_sim import (
        BacktestConfig,
        VectorizedBacktest,
    )
    from quant_platform.core.execution.cost_models.cn_a_share import AShareCostModel

    model = AShareCostModel(
        commission_rate=0.0,
        stamp_tax_rate=0.0,
        transfer_fee_rate=0.0,
        slippage_bps=0.0,
        min_commission=0.0,
    )
    engine = VectorizedBacktest(
        BacktestConfig(lot_size=1, initial_capital=1_000.0), model
    )
    dates = pd.bdate_range("2024-01-02", periods=3)
    _, _, turnover, _, actual = engine._simulate(
        close_mat=np.array([[10.0, 10.0], [20.0, 10.0], [20.0, 10.0]]),
        ret_mat=np.array([[np.nan, np.nan], [1.0, 0.0], [0.0, 0.0]]),
        weights_mat=np.array([[0.5, 0.5], [0.5, 0.5], [0.0, 0.0]]),
        buyable_mat=np.array(
            [[True, True], [True, True], [False, False]], dtype=bool
        ),
        # Day three is limit-up: selling must still be allowed.
        sellable_mat=np.ones((3, 2), dtype=bool),
        rebalance_mask=np.array([True, False, True]),
        dates=dates,
        tickers=["A", "B"],
    )
    assert turnover.iloc[1] == 0.0
    assert actual[1, 0] > actual[1, 1]
    assert turnover.iloc[2] > 0
    assert np.allclose(actual[2], 0.0)


def test_factor_predict_risk_matches_decomposition():
    from quant_platform.core.portfolio.risk_model import FactorRiskModel

    rng = np.random.default_rng(7)
    dates = pd.bdate_range("2023-01-02", periods=100)
    factor = pd.DataFrame({"market": rng.normal(0, 0.01, len(dates))}, index=dates)
    stocks = pd.DataFrame(
        {
            "A": 1.2 * factor["market"] + rng.normal(0, 0.005, len(dates)),
            "B": 0.8 * factor["market"] + rng.normal(0, 0.005, len(dates)),
        },
        index=dates,
    )
    model = FactorRiskModel().fit(stocks, factor)
    weights = pd.Series({"A": 0.5, "B": 0.5})
    decomposition = model.decompose_risk(weights)
    assert model.predict_risk(weights) == pytest.approx(
        decomposition["factor_variance"] + decomposition["specific_variance"]
    )
    with pytest.raises(ValueError, match="MISSING"):
        model.decompose_risk(pd.Series({"A": 0.5, "MISSING": 0.5}))


def test_factor_risk_drops_joint_nans_and_preserves_regression_intercept():
    from quant_platform.core.portfolio.risk_model import FactorRiskModel

    rng = np.random.default_rng(12)
    dates = pd.bdate_range("2024-01-02", periods=80)
    factor = pd.DataFrame({"market": rng.normal(0, 0.01, len(dates))}, index=dates)
    stock = pd.DataFrame(
        {"A": 0.002 + 1.5 * factor["market"]}, index=dates
    )
    factor.iloc[5, 0] = np.nan
    stock.iloc[8, 0] = np.nan
    model = FactorRiskModel().fit(stock, factor)
    assert np.isfinite(model.cov_matrix_).all()
    assert model.intercepts_["A"] == pytest.approx(0.002, abs=1e-10)
    assert model.specific_risk_["A"] < 1e-10


def test_capacity_curve_includes_borrow_drag_and_dynamic_nav():
    from quant_platform.core.evaluation.capacity import capacity_analysis

    dates = pd.to_datetime(["2024-01-02", "2024-01-03"])
    trades = pd.DataFrame(
        {
            "date": [dates[0]],
            "ticker": ["A"],
            "side": ["sell"],
            "shares": [10.0],
            "requested_shares": [10.0],
            "price": [100.0],
            "adv_shares": [1_000_000.0],
            "daily_vol": [0.02],
            "notional_trade": [1_000.0],
            "notional": [1_000.0],
            "total_cost": [0.0],
            "portfolio_value_before_trade": [1_000.0],
            "delta_weight": [-1.0],
        }
    )
    pnl = pd.DataFrame(
        {
            "gross_ret": [0.0, 0.0],
            "borrow_cost_bps": [10.0, 10.0],
        },
        index=dates,
    )
    curve = capacity_analysis(
        trades,
        pnl,
        cost_model=_zero_cost_model(),
        capital_grid=[1_000.0],
    )["capital_curve"]
    assert curve.loc[0, "total_borrow_cost"] > 0
    assert curve.loc[0, "net_ann_return_est"] < 0


def test_cn_minimum_commission_and_volatility_slippage_scale_with_trade():
    from quant_platform.core.execution.cost_models.cn_a_share import AShareCostModel

    model = AShareCostModel(
        commission_rate=0.0003,
        min_commission=5.0,
        transfer_fee_rate=0.0,
        stamp_tax_rate=0.0,
        slippage_model="volatility",
        slippage_k=0.1,
    )
    small = model.compute_cost_matrix(
        np.array([0.0]),
        np.array([0.001]),
        np.array([10.0]),
        nav=10_000.0,
        volatility=np.array([0.02]),
    )
    assert small[0] >= 5.0 / 10_000.0


def test_weekly_portfolio_carries_last_target_to_signal_end(tmp_path):
    from quant_platform.core.portfolio.construction import (
        PortfolioConfig,
        PortfolioMode,
    )
    from quant_platform.core.portfolio.pipeline import run_portfolio_pipeline

    dates = pd.bdate_range("2024-01-02", "2024-01-17")
    tickers = list("ABCDE")
    signal = pd.DataFrame(
        [
            {"date": date, "ticker": ticker, "signal": float(index)}
            for date in dates
            for index, ticker in enumerate(tickers)
        ]
    )
    # The final scheduled date has no usable cross-section.  The last valid
    # target must still be held through the evaluation end.
    signal.loc[signal["date"].eq(dates.max()), "signal"] = np.nan
    prices = signal[["date", "ticker"]].copy()
    prices["close"] = 100.0
    prices["adj_close"] = 100.0
    prices["volume"] = 1_000_000
    universe = signal[["date", "ticker"]].copy()
    universe["pass_all_filters"] = True

    result = run_portfolio_pipeline(
        method="quantile",
        signal_col="signal",
        config=PortfolioConfig(
            mode=PortfolioMode.LONG_ONLY,
            max_stock_weight=1.0,
            max_sector_weight=None,
            rebalance_freq="W",
        ),
        output_path=tmp_path / "weights.parquet",
        factors_override=signal,
        prices_override=prices,
        universe_override=universe,
    )
    assert result["weights"]["date"].max() < dates.max()
    assert result["daily_weights"]["date"].max() == dates.max()
