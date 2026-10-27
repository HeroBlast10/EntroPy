"""Economic invariants for turnover controls, source allocation and sleeves."""
import numpy as np
import pandas as pd
import pytest

from quant_platform.core.execution.backtest.vectorized_daily import (
    _allocate_tranche_fills, generate_trades, simulate_execution,
)
from quant_platform.core.execution.cost_models.us_equity import CostModel
from quant_platform.core.portfolio.low_turnover import (
    TurnoverPolicy, buffered_weights, build_turnover_targets, smooth_scores,
)
from quant_platform.core.alpha_models.fair_comparison import issuer_capped_weights
from quant_platform.core.evaluation.turnover_costs import (
    break_even_trade_bps, turnover_cost_diagnostics,
)

EXECUTION = {"min_names": 10, "max_stock_weight": .2, "max_issuer_gross_weight": .3}


def day(scores):
    n = len(scores)
    return pd.DataFrame({"ticker": [f"S{i}" for i in range(n)], "permno": np.arange(n) + 1,
                         "permco": np.arange(n) + 1, "score": scores})


def cost_model(**kwargs):
    settings = dict(commission_min=0, commission_per_share=0, slippage_bps=0, impact_coeff=0,
                    sec_fee_rate=0, finra_taf_per_share=0, borrow_rate_annual=0,
                    max_participation_rate=None)
    return CostModel(**{**settings, **kwargs})


def test_disabled_policy_reproduces_original_caps_and_ties():
    d = day([1, 1, 1, 2, 3, 4, 5, 6, 7, 7])
    pd.testing.assert_series_equal(buffered_weights(d, "score", pd.Series(dtype=float),
                                                   TurnoverPolicy(), EXECUTION),
                                   issuer_capped_weights(d, "score", min_names=10, stock_cap=.2, issuer_cap=.3))


def test_buffer_retains_only_incumbents_and_forces_missing_score_exit():
    d = day(np.arange(20, dtype=float))
    # S14 is 75th percentile: retains a long, cannot enter as a new long.
    # S5 is 30th percentile: retains a short, cannot enter as a new short.
    p = TurnoverPolicy(exit_fraction=.3)
    held = buffered_weights(d, "score", pd.Series({"S14": .1, "S5": -.1}), p, EXECUTION)
    fresh = buffered_weights(d, "score", pd.Series(dtype=float), p, EXECUTION)
    assert held.iloc[14] > 0 and fresh.iloc[14] == 0
    assert held.iloc[5] < 0 and fresh.iloc[5] == 0
    d.loc[14, "score"] = np.nan
    assert buffered_weights(d, "score", pd.Series({"S14": .1}), p, EXECUTION).iloc[14] == 0
    assert held.sum() == pytest.approx(0)
    assert held.abs().sum() <= 1 + 1e-12
    # Buffered incumbents must still share the issuer's gross cap.
    d.loc[d.index >= 17, "permco"] = 99
    limited = buffered_weights(d, "score", pd.Series({"S14": .1}), p,
                               {**EXECUTION, "max_issuer_gross_weight": .12})
    assert limited.abs().groupby(d.permco).sum().max() <= .12 + 1e-12
    assert limited.sum() == pytest.approx(0)


def test_ema_is_causal_uses_sessions_and_resets_after_missing_and_universe_gaps():
    dates = pd.bdate_range("2024-01-04", periods=6)
    rows = pd.DataFrame({"date": dates, "ticker": "A", "score": [0., 1., np.nan, 8., 10., 12.]})
    smoothed = smooth_scores(rows, "score", 1)
    np.testing.assert_allclose(smoothed, [0, .5, np.nan, 8, 9, 10.5], equal_nan=True)
    altered = rows.copy(); altered.loc[5, "score"] = 1000
    pd.testing.assert_series_equal(smoothed.iloc[:5], smooth_scores(altered, "score", 1).iloc[:5])
    # A second security retains the shared sessions during A's absence.
    gap = pd.concat([rows.drop(index=[2, 3]), rows.assign(ticker="B")], ignore_index=True)
    reentry = gap.ticker.eq("A") & gap.date.eq(dates[4])
    assert smooth_scores(gap, "score", 1).loc[reentry].iloc[0] == 10


def test_tolerance_cancels_small_pending_gap_but_does_not_block_entries_exits_or_reversals():
    target = pd.Series({"hold": .1, "entry": .1, "exit": 0., "flip": -.1})
    actual = pd.Series({"hold": .09, "entry": 0., "exit": .01, "flip": .1})
    names = target.index
    trades = generate_trades(target, actual, pd.Series(10., index=names), pd.Series(1e6, index=names),
                             pd.Series(.01, index=names), 10000, target_tolerance=.2).set_index("ticker")
    assert trades.loc["hold", "fill_status"] == "suppressed_tolerance"
    assert trades.loc["hold", "shares"] == 0
    assert trades.loc["hold", "notional_trade"] == 0
    assert (trades.loc[["entry", "exit", "flip"], "shares"] > 0).all()


def test_tranche_internal_crossing_has_no_external_fill_and_partial_fills_are_not_invented():
    current = pd.DataFrame({"A": [100., -40.], "B": [0., 0.]})
    desired = pd.DataFrame({"A": [40., 20.], "B": [100., 50.]})
    result = _allocate_tranche_fills(current, desired, pd.Series({"A": 0., "B": 75.}))
    pd.testing.assert_frame_equal(result, pd.DataFrame({"A": [40., 20.], "B": [50., 25.]}))
    np.testing.assert_allclose(result.sum() - current.sum(), [0, 75])


def test_overlapping_simulator_does_not_rebalance_inactive_sleeve_price_drift():
    dates = pd.bdate_range("2024-01-02", periods=3)
    prices = pd.DataFrame({"date": dates.repeat(2), "ticker": ["A", "B"] * 3,
                           "close": [10., 10., 20., 10., 20., 10.],
                           "adj_close": [10., 10., 20., 10., 20., 10.], "volume": 1e6})
    weights = pd.DataFrame({"date": dates.repeat(2), "ticker": ["A", "B"] * 3,
                            "weight": [.5, .5] * 3, "is_rebalance": True})
    trades, ledger = simulate_execution(weights, prices, cost_model(), initial_capital=10000,
                                        tranche_count=2, return_daily_ledger=True)
    second = trades.loc[trades.date.eq(dates[1])].set_index("ticker")
    # First day's 2500 dollars of A per sleeve become 5000 each. Only one
    # sleeve sells 1250 of A to reach 15000/2*.5=3750; floor => 62 shares.
    assert second.loc["A", "shares"] == 62
    assert second.loc["B", "shares"] == 125
    assert ledger.loc[dates[1], "nav_net"] == pytest.approx(15000)
    assert ledger.loc[dates[1], "gross_exposure"] == pytest.approx(1.0006666667)


def test_pending_retry_is_cancelled_inside_band_and_exit_is_not_suppressed():
    dates = pd.bdate_range("2024-01-02", periods=8)
    prices = pd.DataFrame({"date": dates, "ticker": "A", "adj_close": 10., "close": 10., "volume": 900.})
    weights = pd.DataFrame({"date": dates[-3:], "ticker": "A", "weight": [1., 1., 0.],
                            "is_rebalance": [True, False, True]})
    trades, ledger = simulate_execution(weights, prices,
        cost_model(max_participation_rate=.1, adv_lookback=5, min_adv_shares=0), initial_capital=1000,
        target_tolerance=.2, return_daily_ledger=True)
    retry = trades.loc[trades.date.eq(dates[-2])].iloc[0]
    assert retry.fill_status == "suppressed_tolerance" and retry.shares == 0
    assert not ledger.loc[dates[-2], "has_pending_orders"]
    assert trades.loc[trades.date.eq(dates[-1]), "shares"].sum() == 90


def test_sources_reconcile_and_break_even_solves_geometric_wealth_with_borrow():
    dates = pd.bdate_range("2024-01-02", periods=3)
    px = pd.DataFrame({"date": dates, "ticker": "A", "close": [10., 11., 12.],
                       "adj_close": [10., 11., 12.], "volume": 1e6})
    w = pd.DataFrame({"date": dates, "ticker": "A", "weight": .5, "is_rebalance": True,
                      "transition_source": ["signal", "drift_resize", "drift_resize"]})
    trades, ledger = simulate_execution(w, px, cost_model(commission_min=1), initial_capital=10000,
                                        return_daily_ledger=True)
    daily, sources, summary = turnover_cost_diagnostics(trades, ledger, 10000)
    assert summary["daily_one_way_turnover"] == pytest.approx(ledger.executed_turnover.mean() / 2)
    assert sources.total_cost_dollar.sum() == pytest.approx(ledger.total_cost_dollar.sum())
    assert daily.loc[dates[0], "signal_cost"] == 1
    assert daily.loc[dates[1], "drift_resize_cost"] > 0
    gross, borrow, fraction = np.array([.01, -.002]), np.array([.0001, .0002]), np.array([1., .4])
    bps = break_even_trade_bps(gross, borrow, fraction)
    assert np.prod(1 + gross - borrow - bps / 10000 * fraction) == pytest.approx(1, abs=1e-12)
    assert break_even_trade_bps([-.01], [0], [1]) == pytest.approx(-100)
    assert np.isnan(break_even_trade_bps([.01], [0], [0]))


def test_buffer_and_tranches_preserve_prediction_universe_exits_without_future_labels():
    dates = pd.bdate_range("2024-01-02", periods=3)
    rows = pd.concat([day(np.arange(20)).assign(date=date, forward_return=np.nan) for date in dates], ignore_index=True)
    rows = rows.loc[~(rows.date.eq(dates[1]) & rows.ticker.eq("S19"))]
    weights, _ = build_turnover_targets(rows, "score", EXECUTION, TurnoverPolicy(exit_fraction=.3, tranches=2))
    assert not weights.loc[weights.date.eq(dates[1]) & weights.ticker.eq("S19"), "eligible"].any()
    changed = rows.assign(forward_return=10000)
    pd.testing.assert_frame_equal(weights, build_turnover_targets(changed, "score", EXECUTION,
                                                                  TurnoverPolicy(exit_fraction=.3, tranches=2))[0])
    prices = pd.MultiIndex.from_product([dates, [f"S{i}" for i in range(20)]],
                                        names=["date", "ticker"]).to_frame(index=False)
    prices = prices.assign(close=10., adj_close=10., volume=1e6, total_return=0., is_terminal=False)
    trades, _ = simulate_execution(weights, prices, cost_model(), initial_capital=10000,
                                    tranche_count=2, target_tolerance=.2, return_daily_ledger=True)
    exit_order = trades.loc[trades.date.eq(dates[1]) & trades.ticker.eq("S19")]
    assert exit_order.side.eq("sell").all() and exit_order.executed_notional.sum() == pytest.approx(1000)
    prices.loc[prices.date.eq(dates[1]) & prices.ticker.eq("S19"), "is_terminal"] = True
    terminal_trades = simulate_execution(weights, prices, cost_model(), initial_capital=10000,
                                         tranche_count=2, target_tolerance=.2)
    assert terminal_trades.loc[terminal_trades.date.ge(dates[1]) & terminal_trades.ticker.eq("S19")].empty


@pytest.mark.parametrize("kwargs", [{"exit_fraction": .1}, {"ema_halflife": 0}, {"tranches": 0},
                                    {"tolerance": 1}, {"tranches": 20, "rebalance_sessions": 20}])
def test_invalid_policies_fail(kwargs):
    with pytest.raises(ValueError):
        TurnoverPolicy(**kwargs)
