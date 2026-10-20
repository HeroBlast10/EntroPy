"""Counterfactual regressions for WRDS model, return and issuer contracts."""
from argparse import Namespace

import numpy as np
import pandas as pd
import pytest

from quant_platform.core.alpha_models.nonlinear_alpha import (
    select_usable_features, build_learning_panel, run_walk_forward_oof,
    WalkForwardConfig, LightGBMConfig, MLPConfig,
)
from quant_platform.core.data.returns import (
    forward_total_returns, overnight_total_return, trailing_total_return,
)
from quant_platform.core.data.wrds import company_market_caps, research_factors
from quant_platform.core.data.research_contract import require_wrds_identity, require_financial_vintage
from quant_platform.core.signals.feature_cache import PriceFeatureCache
from quant_platform.core.signals.cross_sectional.momentum import Mom1M, OvernightReturn1M


def price_tape():
    dates = pd.bdate_range("2024-01-02", periods=25)
    idx = pd.MultiIndex.from_product([dates, ["PERMNO:1", "PERMNO:2"]], names=["date", "ticker"])
    p = idx.to_frame(index=False)
    p["close"] = p["adj_close"] = p["open"] = 100.
    p["volume"] = 1000.
    p["total_return"] = 0.
    p["is_terminal"] = False
    p["is_tradable"] = True
    return dates, p


@pytest.mark.parametrize("terminal", [-.75, -1., .25])
def test_day_two_delisting_survives_twenty_session_label(terminal):
    dates, p = price_tape()
    p.loc[p.date.eq(dates[1]) & p.ticker.eq("PERMNO:1"), "total_return"] = .1
    mask = p.date.eq(dates[2]) & p.ticker.eq("PERMNO:1")
    p.loc[mask, ["is_terminal", "is_tradable", "total_return", "close"]] = [True, False, terminal, np.nan]
    p = p[~(p.ticker.eq("PERMNO:1") & p.date.gt(dates[2]))]
    out = forward_total_returns(p, 20).set_index(["date", "ticker"])
    row = out.loc[(dates[0], "PERMNO:1")]
    assert row.forward_return == pytest.approx(1.1 * (1 + terminal) - 1)
    assert row.label_end == dates[20]
    assert row.label_terminal_settled
    assert not row.label_terminal_imputed


def test_normal_gap_is_not_terminal_cash_or_endpoint_label():
    dates, p = price_tape()
    p.loc[p.date.eq(dates[2]) & p.ticker.eq("PERMNO:1"), ["close", "total_return"]] = np.nan
    out = forward_total_returns(p, 20)
    assert pd.isna(out.loc[out.date.eq(dates[0]) & out.ticker.eq("PERMNO:1"), "forward_return"].iloc[0])
    assert not out.label_terminal_settled.any()


def test_nullable_wrds_arrow_returns_and_absent_rows_use_numeric_arrays():
    dates, p = price_tape()
    p["total_return"] = p.total_return.astype("Float64")
    p["is_terminal"] = p.is_terminal.astype("boolean")
    p = p[~(p.date.eq(dates[2]) & p.ticker.eq("PERMNO:1"))]
    result = forward_total_returns(p, 20)
    assert pd.isna(result.loc[result.date.eq(dates[0]) & result.ticker.eq("PERMNO:1"), "forward_return"].iloc[0])


@pytest.mark.parametrize("policy,expected", [("total_loss", -1), ("zero_return", 0)])
def test_unknown_terminal_return_never_silently_imputed(policy, expected):
    dates, p = price_tape()
    mask = p.date.eq(dates[2]) & p.ticker.eq("PERMNO:1")
    p.loc[mask, "is_terminal"] = True
    p.loc[mask, "total_return"] = np.nan
    with pytest.raises(ValueError, match="Missing terminal"):
        forward_total_returns(p, 20)
    out = forward_total_returns(p, 20, missing_terminal_return=policy)
    row = out.loc[out.date.eq(dates[0]) & out.ticker.eq("PERMNO:1")].iloc[0]
    assert row.forward_return == expected
    assert row.label_terminal_imputed


def test_price_gap_never_padded_or_bridged_by_cache():
    dates, p = price_tape()
    p = p.loc[p.ticker.eq("PERMNO:1")].iloc[:3].drop(columns="total_return").copy()
    p["adj_close"] = [100., np.nan, 110.]
    assert PriceFeatureCache(p).get("ret_1d").isna().all()
    assert PriceFeatureCache(p).get("ret_2d").isna().all()


def test_absent_security_row_does_not_shift_shared_calendar():
    dates, p = price_tape()
    p = p.drop(columns="total_return")
    p = p[~(p.date.eq(dates[1]) & p.ticker.eq("PERMNO:1"))]
    r = trailing_total_return(p)
    assert pd.isna(r.loc[p.date.eq(dates[2]) & p.ticker.eq("PERMNO:1")].iloc[0])


def test_direct_total_return_overrides_synthetic_index_and_cache_matches_uncached():
    dates, p = price_tape()
    p["total_return"] = .01
    p["adj_close"] = 100.  # deliberately inconsistent fallback
    cache = PriceFeatureCache(p)
    factor = Mom1M(period=3, lag=0)
    cached = factor._compute(p, _feature_cache=cache).sort_index()
    direct = factor._compute(p).sort_index()
    pd.testing.assert_series_equal(cached, direct)
    assert cached.dropna().iloc[-1] == pytest.approx(1.01 ** 3 - 1)


@pytest.mark.parametrize("cash_distribution", [False, True])
def test_zero_economic_split_and_distribution_have_zero_overnight(cash_distribution):
    dates, p = price_tape()
    p = p.loc[p.ticker.eq("PERMNO:1")].copy()
    p.loc[p.date.ge(dates[2]), ["close", "open"]] = 99. if cash_distribution else 50.
    assert overnight_total_return(p).eq(0).all()
    assert OvernightReturn1M()._compute(p).dropna().eq(0).all()
    p = p.drop(columns="total_return")
    with pytest.raises(ValueError, match="adj_factor"):
        overnight_total_return(p)


def test_overnight_intraday_grosses_reconstruct_daily_return():
    _, p = price_tape()
    p["total_return"] = .06
    p["open"], p["close"] = 102., 106.
    o = overnight_total_return(p)
    np.testing.assert_allclose((1 + o) * p.close / p.open - 1, .06)


@pytest.mark.parametrize("forbidden", ["permno", "permco", "gvkey", "pit_score", "source_version", "fundamentals_is_point_in_time", "target"])
def test_feature_whitelist_rejects_identifiers_and_provenance_even_when_explicit(forbidden):
    frame = pd.DataFrame({"date": pd.Timestamp("2024-01-02"), "MOM_1M": range(4), forbidden: range(4), "new_numeric_metadata": range(4)})
    assert select_usable_features(frame) == ["MOM_1M"]
    with pytest.raises(ValueError, match="Identifiers/provenance"):
        select_usable_features(frame, ["MOM_1M", forbidden])


@pytest.mark.parametrize("kind", ["ridge", "lasso", "elastic_net"])
def test_linear_ml_fit_and_cross_validation_share_explicit_feature_contract(kind):
    from quant_platform.core.alpha_models.ml_alpha import MLAlphaModel
    dates = pd.bdate_range("2024-01-02", periods=12)
    f = pd.MultiIndex.from_product([dates, range(1, 6)], names=["date", "permno"]).to_frame(index=False)
    f["ticker"] = "PERMNO:" + f.permno.astype(str)
    f["MOM_1M"] = np.random.default_rng(10).normal(size=len(f))
    f["source_revision"], f["fundamentals_is_point_in_time"] = f.permno * 100, True
    target = pd.Series(f.MOM_1M.to_numpy() * .1, index=pd.MultiIndex.from_frame(f[["date", "ticker"]]))
    m = MLAlphaModel(model_type=kind, alpha=.001).fit(f, target)
    assert m.feature_names_ == ["MOM_1M"]
    original = m.predict(f)
    changed = f.assign(permno=100-f.permno, source_revision=100000-f.source_revision)
    # The economic sample and display keys stay fixed; identifiers change.
    m2 = MLAlphaModel(model_type=kind, alpha=.001).fit(changed, target)
    np.testing.assert_allclose(original, m2.predict(changed), atol=1e-12)
    bad = MLAlphaModel(feature_cols=["MOM_1M", "permno"])
    with pytest.raises(ValueError, match="Identifiers/provenance"):
        bad.fit(f, target)
    with pytest.raises(ValueError, match="Identifiers/provenance"):
        bad.cross_validate(f, target)


@pytest.mark.parametrize("row_cap", [None, 300])
@pytest.mark.parametrize("model", ["lightgbm", "mlp"])
def test_permno_renumbering_preserves_corresponding_predictions(row_cap, model):
    pytest.importorskip("lightgbm" if model == "lightgbm" else "torch")
    rng = np.random.default_rng(10)
    dates = pd.bdate_range("2020-01-02", periods=65)
    idx = pd.MultiIndex.from_product([dates, range(1, 13)], names=["date", "permno"])
    f = idx.to_frame(index=False)
    f["ticker"] = "PERMNO:" + f.permno.astype(str)
    f["MOM_1M"] = rng.normal(size=len(f))
    p = f[["date", "ticker"]].copy()
    r = rng.normal(0, .01, size=(65, 12))
    r[1:] += f.MOM_1M.to_numpy().reshape(65, 12)[:-1] * .003
    p["adj_close"] = (100 * (1+r).cumprod(axis=0)).reshape(-1)
    u = f[["date", "ticker"]].assign(pass_all_filters=True)
    wf = WalkForwardConfig(min_train_days=30, validation_days=10, n_splits=2, max_train_rows=row_cap)
    cfg = LightGBMConfig(n_estimators=15, min_child_samples=10, bagging_fraction=.7, n_jobs=1)
    def fit(f, p, u):
        panel, features = build_learning_panel(f, p, u, min_cross_section=5)
        return run_walk_forward_oof(panel, features, model=model, walk_forward=wf, lightgbm=cfg,
                                   mlp=MLPConfig(epochs=1, hidden_width=8, batch_size=64, inference_batch_size=128))[0]
    original = fit(f, p, u)
    mapping = {f"PERMNO:{i}": f"PERMNO:{100-i*3}" for i in range(1, 13)}
    f2, p2, u2 = f.copy(), p.copy(), u.copy()
    for frame in (f2, p2, u2):
        frame["ticker"] = frame.ticker.map(mapping)
    f2["permno"] = f2.permno.map({i: 100-i*3 for i in range(1, 13)})
    changed = fit(f2, p2, u2)
    changed["ticker"] = changed.ticker.map({v:k for k,v in mapping.items()})
    compare = original.merge(changed, on=["date", "ticker"], suffixes=("_a", "_b"))
    np.testing.assert_allclose(compare.prediction_a, compare.prediction_b, atol=1e-7, rtol=0)


def issuer_fixture():
    dates = pd.bdate_range("2024-02-01", periods=4)
    idx = pd.MultiIndex.from_product([dates, [1, 2]], names=["date", "permno"])
    c = idx.to_frame(index=False).assign(permco=10, market_cap=50_000., securitytype="EQTY", securitysubtype="COM", sharetype="NS", delisting_flag="N")
    link = pd.DataFrame({"permno": [1], "permco": [10], "gvkey": ["000123"],
                         "link_start": [dates[0]], "link_end": [pd.NaT]})
    e = pd.DataFrame({"available_date": [dates[0]], "datadate": [pd.Timestamp("2023-12-31")],
                      "permno": [1], "gvkey": ["000123"], "link_end": [pd.NaT],
                      "is_point_in_time": [False], "niq_ttm": [10.], "ceqq_q": [30.], "curcdq": ["USD"]})
    return dates, c, link, e


def test_two_share_classes_use_company_value_and_same_issuer_financials():
    _, c, link, e = issuer_fixture()
    caps = company_market_caps(c, link)
    assert caps.company_market_cap.eq(1e8).all()
    p = c[["date", "permno", "permco"]].merge(caps, on=["date", "permco"])
    p["ticker"] = "PERMNO:" + p.permno.astype(str)
    p["market_cap"], p["adj_close"], p["total_return"] = 5e7, 100., 0.
    result = research_factors(p, e)
    assert result.EARNINGS_YIELD.dropna().eq(.1).all()  # never .2 per class
    assert result.BOOK_TO_MARKET.dropna().eq(.3).all()
    assert result.groupby("permno").EARNINGS_YIELD.count().eq(3).all()
    single = c[c.permno.eq(1)].assign(market_cap=100_000.)
    assert company_market_caps(single, link).company_market_cap.equals(caps.company_market_cap)


def test_missing_class_cap_invalidates_denominator_and_conflicting_links_block_attribution():
    _, c, link, _ = issuer_fixture()
    c.loc[c.permno.eq(2), "market_cap"] = np.nan
    assert company_market_caps(c, link).company_market_cap.isna().all()
    conflict = pd.concat([link, link.assign(permno=2, gvkey="000999")])
    result = company_market_caps(c, conflict)
    assert result.issuer_ambiguous.all()
    assert result.gvkey.isna().all()


def test_future_share_class_mapping_never_backfilled_and_expired_ccm_not_used():
    dates, c, link, _ = issuer_fixture()
    link["link_start"], link["link_end"] = dates[1], dates[2]
    cap = company_market_caps(c, link).set_index("date")
    assert pd.isna(cap.loc[dates[0], "gvkey"])
    assert cap.loc[dates[1], "gvkey"] == "000123"
    assert pd.isna(cap.loc[dates[3], "gvkey"])


def test_issuer_change_cannot_carry_previous_company_financial_ratio():
    dates, c, link, e = issuer_fixture()
    caps = company_market_caps(c, link)
    p = c[["date", "permno", "permco"]].merge(caps, on=["date", "permco"])
    p["ticker"] = "PERMNO:" + p.permno.astype(str)
    p["market_cap"], p["adj_close"], p["total_return"] = 5e7, 100., 0.
    p.loc[p.date.ge(dates[2]), ["permco", "gvkey"]] = [20, "999999"]
    result = research_factors(p, e)
    assert result.loc[result.date.eq(dates[1]), "EARNINGS_YIELD"].notna().all()
    assert result.loc[result.date.ge(dates[2]), "EARNINGS_YIELD"].isna().all()


def test_code_freeze_includes_untracked_scoped_sources_but_preserves_other_work(tmp_path):
    import subprocess
    from quant_platform.core.utils.provenance import git_dirty
    def git(*args):
        return subprocess.run(["git", *args], cwd=tmp_path, check=True, capture_output=True)
    git("init")
    git("config", "user.name", "Test")
    git("config", "user.email", "test@example.invalid")
    (tmp_path / "research").mkdir()
    (tmp_path / "research/main.py").write_text("pass")
    (tmp_path / "other.py").write_text("pass")
    git("add", ".")
    git("commit", "-m", "baseline")
    (tmp_path / "other.py").write_text("# concurrent task")
    assert git_dirty(tmp_path, paths=["research"]) is False
    (tmp_path / "research/new.py").write_text("pass")
    assert git_dirty(tmp_path, paths=["research"]) is True


def test_restated_financial_waiver_never_waives_universe_or_primary_key():
    d = pd.Timestamp("2024-01-02")
    u = pd.DataFrame({"date": [d], "permno": [1], "ticker": ["PERMNO:1"],
                      "membership_source": ["CRSP"], "membership_is_point_in_time": [True]})
    fund = pd.DataFrame({"available_date": [d], "is_point_in_time": [False]})
    require_financial_vintage(fund, ["EARNINGS_YIELD"], allow_restated=True)
    require_wrds_identity(u, u)
    with pytest.raises(ValueError, match="Restated"):
        require_financial_vintage(fund, ["EARNINGS_YIELD"])
    for bad in (u.assign(membership_is_point_in_time=False), u.assign(membership_is_point_in_time="False"),
                u.assign(ticker="REUSED"), u.drop(columns="permno")):
        with pytest.raises(ValueError):
            require_wrds_identity(bad)
    from scripts.evaluate_nonlinear_alpha import _validate_point_in_time_contract
    with pytest.raises(ValueError, match="non-PIT"):
        _validate_point_in_time_contract(Namespace(allow_non_pit_research=True), [], u.assign(membership_is_point_in_time=False))
