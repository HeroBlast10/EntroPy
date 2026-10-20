"""Adversarial WRDS fixtures: units, vintages, timing and terminal accounting."""
import numpy as np
import pandas as pd
import pytest

from scripts.wrds_ingest import fundamentals as fb
from scripts.wrds_ingest import universe as uv
from quant_platform.core.data.wrds import canonical_prices, membership_panel, research_factors
from quant_platform.core.execution.backtest.vectorized_daily import simulate_execution
from quant_platform.core.execution.backtest.event_sim import BacktestConfig, VectorizedBacktest
from quant_platform.core.execution.cost_models.us_equity import CostModel


def reports():
    return pd.DataFrame({"gvkey": ["1"] * 4,
        "datadate": pd.to_datetime(["2023-12-31", "2024-03-31", "2024-06-30", "2024-09-30"]),
        "rdq": pd.to_datetime(["2024-02-01", "2024-05-02", "2024-08-01", "2024-10-31"]),
        "fyearq": [2024] * 4, "fqtr": [1, 2, 3, 4],
        "saleq": [119575., 90753., 85777., 94930.],
        "oancfy": [100., 250., 380., 500.], "atq": [1000.] * 4,
        "vintage": ["compustat_fundq_as_restated"] * 4})


def test_income_quarters_are_unchanged_and_apple_ttm_is_391035():
    f, _ = fb.to_single_quarter(reports(), ["saleq", "oancfy", "atq"])
    assert f.saleq_q.tolist() == reports().saleq.tolist()
    assert f.oancfy_q.tolist() == [100., 150., 130., 120.]
    t = fb.to_ttm(f, ["saleq", "atq"])
    assert t.saleq_ttm.iloc[-1] == 391035.
    assert "atq_ttm" not in t


def test_ttm_rejects_nonconsecutive_quarters_and_partial_window():
    f = reports()
    f.loc[2, "fqtr"] = 4
    f, _ = fb.to_single_quarter(f, ["saleq"])
    assert fb.to_ttm(f, ["saleq"]).saleq_ttm.isna().all()
    with pytest.raises(ValueError, match="four"):
        fb.to_ttm(f, ["saleq"], min_periods=2)


def test_ambiguous_fiscal_keys_do_not_fan_out_yoy():
    f = reports().iloc[[0, 1]].copy()
    f.loc[f.index[1], "datadate"] = pd.Timestamp("2024-02-01")
    f["fyearq"] = 2023
    f["fqtr"] = 1
    later = f.iloc[[0]].copy()
    later["datadate"] = pd.Timestamp("2024-12-31")
    later["fyearq"] = 2024
    f = pd.concat([f, later], ignore_index=True)
    f["saleq_q"] = [1., 2., 3.]
    out = fb.to_yoy(f, ["saleq"])
    assert len(out) == 3
    assert pd.isna(out.saleq_yoy_pct.iloc[-1])


def test_asof_projection_retains_earliest_snapshot_and_never_uses_future_revision():
    f = reports()
    f["vintage"] = "compustat_snapshot_pit"
    f["pitdate1"] = f["rdq"]
    f["pitdate2"] = "9999-12-31"
    revision = f.iloc[[0]].copy()
    f.loc[0, "pitdate2"] = "2024-11-14"
    revision["pitdate1"] = pd.Timestamp("2024-11-15")
    revision["saleq"] = 219575.
    raw = pd.concat([f, revision], ignore_index=True)
    out = fb.project_asof_events(raw, "2024-12-31")
    before = out[out.available_date.eq("2024-11-01")].iloc[0]
    after = out[out.available_date.eq("2024-11-18")].iloc[0]
    assert before.saleq_ttm == 391035.
    assert after.saleq_ttm == 491035.
    assert after.datadate == before.datadate  # revision must not roll report backward
    assert before.saleq_ttm_is_point_in_time
    for c in [c for c in out if c.endswith("_available_date") and c != "source_available_date"]:
        assert (out[c].isna() | out[c].le(out.available_date)).all()


def test_missing_snapshot_periods_fall_back_per_company_and_keep_nonpit_label():
    f = reports()
    f.loc[3, "vintage"] = "compustat_snapshot_pit"
    f["pitdate1"] = pd.NaT
    f.loc[3, "pitdate1"] = f.loc[3, "rdq"]
    f["pitdate2"] = "9999-12-31"
    out = fb.project_asof_events(f, "2024-12-31")
    r = out.iloc[-1]
    assert r.saleq_ttm == 391035.
    assert not r.saleq_ttm_is_point_in_time
    assert r.saleq_ttm_mixed_vintage
    assert not r.is_point_in_time


def test_expired_snapshot_field_is_cleared_and_not_forward_filled():
    f = reports().iloc[[0]].copy()
    f["vintage"] = "compustat_snapshot_pit"
    f["pitdate1"] = "2024-02-01"
    f["pitdate2"] = "2024-02-02"
    # No active periods remains on expiry. A tombstone is required.
    out = fb.project_asof_events(f, "2024-02-10")
    assert out.available_date.max() == pd.Timestamp("2024-02-05")
    assert pd.isna(out.saleq_q.iloc[-1])


def test_availability_is_next_exchange_session_and_imputation_is_never_pit():
    f = reports().iloc[[0]].copy()
    f["rdq"] = pd.NaT
    f["vintage"] = "compustat_snapshot_pit"
    out, _ = fb.availability_dates(f)
    assert not out.is_point_in_time.iloc[0]
    f["rdq"] = pd.Timestamp("2024-03-28")  # Good Friday is closed
    out, _ = fb.availability_dates(f)
    assert out.available_date.iloc[0] == pd.Timestamp("2024-04-01")


def test_nested_membership_does_not_shorten_outer_interval():
    f = pd.DataFrame({"permno": [1, 1], "membership_start": pd.to_datetime(["2020-01-01", "2021-01-01"]),
         "membership_end": pd.to_datetime(["2025-12-31", "2022-12-31"]), "is_current_member": False})
    assert uv._merge_intervals(f).membership_end.iloc[0] == pd.Timestamp("2025-12-31")


def test_ticker_lookup_preserves_payload():
    targets = pd.DataFrame({"permno": [1], "date": pd.to_datetime(["2020-01-02"]), "weight": [.2]})
    names = pd.DataFrame({"permno": [1], "effective_from": pd.to_datetime(["2010-01-01"]),
                          "effective_to": [pd.NaT], "ticker": ["ABC"]})
    assert uv.attach_ticker(targets, names).weight.iloc[0] == .2


def test_ccm_does_not_lose_older_active_link_behind_newer_expired_link():
    f = pd.DataFrame({"gvkey": ["1"], "available_date": pd.to_datetime(["2020-01-01"])})
    link = pd.DataFrame({"gvkey": ["1", "1"], "permno": [1, 2],
        "link_start": pd.to_datetime(["2000-01-01", "2010-01-01"]),
        "link_end": pd.to_datetime([None, "2015-01-01"])})
    copy = link.copy(deep=True)
    out, _ = fb.attach_permno_at_disclosure(f, link)
    assert out.permno.iloc[0] == 1
    pd.testing.assert_frame_equal(link, copy)


def terminal_fixture(ret=-.5):
    dates = pd.date_range("2024-01-02", periods=4, freq="B")
    weights = pd.DataFrame({"date": dates, "ticker": "PERMNO:1", "weight": .5, "is_rebalance": [True, False, True, True]})
    prices = pd.DataFrame({"date": dates, "ticker": "PERMNO:1", "adj_close": [100., 50., 50., 50.],
        "close": [100., 0., np.nan, np.nan], "total_return": [0., ret, np.nan, np.nan],
        "is_terminal": [False, True, False, False], "is_tradable": [True, False, False, False]})
    model = CostModel(commission_per_share=0, commission_min=0, slippage_bps=0, impact_coeff=0,
                     sec_fee_rate=0, finra_taf_per_share=0, borrow_rate_annual=0, max_participation_rate=None)
    return dates, weights, prices, model


def test_untradable_terminal_return_is_marked_once_settled_and_never_retried():
    _, w, p, model = terminal_fixture()
    trades, ledger = simulate_execution(w, p, model, initial_capital=10000, return_daily_ledger=True)
    assert ledger.nav_net.tolist() == [10000., 7500., 7500., 7500.]
    assert ledger.terminal_settlement_value.iloc[1] == 2500
    assert ledger.gross_exposure.iloc[1:].eq(0).all()
    assert not ledger.has_pending_orders.any()
    assert len(trades) == 1


@pytest.mark.parametrize("policy,nav", [("total_loss", 5000), ("zero_return", 10000)])
def test_missing_terminal_return_is_explicit_sensitivity(policy, nav):
    _, w, p, model = terminal_fixture(np.nan)
    with pytest.raises(ValueError, match="Missing terminal"):
        simulate_execution(w, p, model, initial_capital=10000)
    _, ledger = simulate_execution(w, p, model, initial_capital=10000, missing_terminal_return=policy,
                                   return_daily_ledger=True)
    assert ledger.nav_net.iloc[-1] == nav
    assert ledger.terminal_return_imputations.sum() == 1


def test_direct_total_return_prevents_split_loss():
    _, w, p, model = terminal_fixture(0.)
    p["is_terminal"] = False
    p["close"] = [100., 50., 50., 50.]
    p["is_tradable"] = True
    _, ledger = simulate_execution(w, p, model, initial_capital=10000, return_daily_ledger=True)
    assert ledger.nav_net.eq(10000).all()


def test_event_engine_terminal_accounting_and_default_missing_failure():
    dates, w, p, _ = terminal_fixture()
    engine = VectorizedBacktest(BacktestConfig(top_n=1, max_weight=1, lot_size=1, signal_lag=0))
    close, ret, *_ = engine._build_matrices(w.rename(columns={"weight": "signal"}).drop(columns="is_rebalance"), p, "ticker", None)
    engine._terminal_mat = p.is_terminal.to_numpy()[:, None]
    nav, _, _, _, actual = engine._simulate(close, ret, np.full((4, 1), .5), np.ones((4, 1), bool),
        np.ones((4, 1), bool), np.array([True, False, True, True]), dates, ["PERMNO:1"])
    assert actual[1:].sum() == 0
    assert nav.iloc[1] / nav.iloc[0] == pytest.approx(.75, abs=.001)
    ret[1, 0] = np.nan
    with pytest.raises(ValueError, match="Missing terminal"):
        engine._simulate(close, ret, np.full((4, 1), .5), np.ones((4, 1), bool), np.ones((4, 1), bool),
                         np.ones(4, bool), dates, ["PERMNO:1"])


def test_ciz_units_do_not_scale_volume_and_identity_does_not_use_ticker():
    raw = pd.DataFrame({"permno": [1, 2], "date": pd.to_datetime(["2024-01-02"] * 2),
        "price": [10., 20.], "ret": [0., 0.], "volume": [100., 200.],
        "shares_outstanding": [1000., 2000.], "market_cap": [10000., 40000.], "delisting_flag": ["N", "N"]})
    px = canonical_prices(raw)
    assert px.market_cap.tolist() == [1e7, 4e7]
    assert px.shares_outstanding.tolist() == [1e6, 2e6]
    assert px.volume.tolist() == [100., 200.]
    assert px.ticker.tolist() == ["PERMNO:1", "PERMNO:2"]


def test_research_features_use_daily_market_cap_lag_and_currency_guard():
    dates = pd.bdate_range("2024-02-01", periods=5)
    px = pd.DataFrame({"date": dates, "permno": 1, "ticker": "PERMNO:1",
                       "market_cap": [1e8, 1e8, 2e8, 2e8, 2e8], "adj_close": 100., "total_return": 0.})
    events = pd.DataFrame({"available_date": dates[[1, 3]], "permno": 1,
        "datadate": pd.to_datetime(["2023-12-31"] * 2), "link_end": pd.NaT,
        "is_point_in_time": [False, False], "niq_ttm": [10., 20.],
        "curcdq": ["USD", "CAD"]})
    factors = research_factors(px, events)
    assert factors.EARNINGS_YIELD.iloc[:2].isna().all()
    assert factors.EARNINGS_YIELD.iloc[2] == .1
    assert factors.EARNINGS_YIELD.iloc[3] == .05
    assert pd.isna(factors.EARNINGS_YIELD.iloc[4])  # CAD is not comparable to USD market cap
    assert not factors.fundamentals_is_point_in_time.any()


def test_duplicate_fiscal_labels_are_invalidated_without_losing_income():
    f = reports().iloc[[0]].copy()
    duplicate = f.copy()
    duplicate["fyearq"] = 2023
    duplicate["fqtr"] = 4
    out = fb.project_asof_events(pd.concat([f, duplicate]), "2024-12-31")
    assert len(out) == 1
    assert out.saleq_q.iloc[0] == 119575
    assert pd.isna(out.fqtr.iloc[0])
    assert pd.isna(out.oancfy_q.iloc[0])


def test_pit_membership_gate_accepts_stable_permno_aliases():
    from quant_platform.core.data.universe import require_point_in_time_universe
    dates = pd.to_datetime(["2024-01-02", "2024-01-03"])
    px = pd.DataFrame({"date": dates, "permno": [1, 1], "ticker": "PERMNO:1",
                       "market_cap": 1e8, "is_tradable": True})
    member = pd.DataFrame({"permno": [1], "membership_start": dates[[0]], "membership_end": dates[[0]]})
    out = membership_panel(px, member)
    require_point_in_time_universe(out)
    assert out.date.tolist() == [dates[0]]


def test_piotroski_stock_split_is_not_equity_issuance():
    from quant_platform.core.data.wrds import fundamental_factors
    f = pd.DataFrame({"available_date": pd.to_datetime(["2024-02-02"]), "permno": 1,
        "datadate": pd.to_datetime(["2023-12-31"]), "link_end": pd.NaT, "is_point_in_time": False,
        "curcdq": "USD", "atq_q": 1000., "atq_yoy_diff": 100., "niq_ttm": 100.,
        "niq_ttm_prior_year": 80., "oancfy_ttm": 120., "saleq_ttm": 1000., "cogsq_ttm": 400.,
        "saleq_ttm_prior_year": 800., "cogsq_ttm_prior_year": 400., "dlcq_q": 20., "dlttq_q": 100.,
        "dlcq_yoy_diff": -10., "dlttq_yoy_diff": -70., "cshoq_q": 200., "cshoq_yoy_diff": 100.,
        "ajexq_q": .5, "ajexq_yoy_diff": -.5})
    split = fundamental_factors(f).PIOTROSKI_F8.iloc[0]
    f["cshoq_q"], f["cshoq_yoy_diff"] = 300., 200.
    issuance = fundamental_factors(f).PIOTROSKI_F8.iloc[0]
    assert split == 8
    assert issuance == 7


def test_positive_share_retries_cannot_dilute_rebalance_capacity_percentile():
    from quant_platform.core.evaluation.capacity import capacity_analysis
    main = pd.DataFrame({"date": pd.to_datetime(["2024-01-02"]), "ticker": "A", "side": "buy",
                         "shares": 100., "requested_shares": 100., "adv_shares": 1000., "price": 100.,
                         "notional_trade": 10000., "notional": 10000., "portfolio_value_before_trade": 1e6,
                         "daily_vol": .02, "is_rebalance_order": True})
    retries = pd.concat([main] * 100, ignore_index=True)
    retries["date"] = pd.bdate_range("2024-01-03", periods=100)
    retries[["shares", "requested_shares"]] = 1.
    retries[["notional_trade", "notional"]] = 100.
    retries["is_rebalance_order"] = False
    baseline = capacity_analysis(main, capital_grid=())["summary"].iloc[0]
    result = capacity_analysis(pd.concat([main, retries]), capital_grid=())["summary"].iloc[0]
    assert result.capacity_at_10pct_adv == baseline.capacity_at_10pct_adv == 1e6
    assert result.p95_requested_participation == .1
    assert result.p95_executed_participation == .1
    assert result.p95_sample_days == 1


def test_nonpit_primary_source_can_be_rebuilt_and_is_never_labeled_pit(tmp_path, monkeypatch):
    from scripts.wrds_ingest.client import SourcePlan
    monkeypatch.setattr(fb, "pull_fundamentals", lambda *a, **kw: reports())
    summary = fb.run(plan=SourcePlan({"fundamentals_table": "comp.fundq", "fundamentals_is_point_in_time": False}),
                     db=object(), gvkeys=["1"], output_dir=tmp_path, end="2024-12-31")
    assert not summary["is_point_in_time"]
    assert summary["n_rows_point_in_time"] == 0
    assert summary["n_rows_non_pit"] == summary["n_rows"] == 4
