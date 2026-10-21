"""Time, economic-unit and fair-comparison regressions for the second batch."""
import numpy as np
import pandas as pd
import pytest

from quant_platform.core.data.industry import historical_industries,neutralize_industry_size
from quant_platform.core.signals.registry import FactorRegistry
from quant_platform.core.signals.feature_cache import PriceFeatureCache
from quant_platform.core.signals.cross_sectional.liquidity import (ShareTurnover20D,Turnover20D,Turnover60D,
    TurnoverAccel,AbnormalVolume,VolumeCV,IlliqAmihud,SpreadHL)
from quant_platform.core.signals.time_series.kalman_state_space import _run_kalman_on_panel,KalmanVelocity
from quant_platform.core.signals.time_series.entropy_hurst import _rolling_spectral_entropy,SpectralEntropy60D,HurstExponent60D
from quant_platform.core.signals.time_series.higher_moments import RollingSkew60D
from quant_platform.core.signals.relative_value.ou_pairs import OUZScore,_rolling_ou_params
from quant_platform.core.signals.cross_sectional.momentum import Mom1M,_rolling_max_drawdown
from quant_platform.core.signals.regime.hmm_regime import HMMTurbulenceProbability
from quant_platform.core.alpha_models import fair_comparison as fair
from quant_platform.core.alpha_models.nonlinear_alpha import WalkForwardConfig,expanding_date_folds
from quant_platform.core.evaluation.factor_diagnostics import hac_summary,benjamini_hochberg,grouped_rank_ic,factor_diagnostics,calendar_turnover


def prices(n=400,stocks=3):
    rng=np.random.default_rng(14)
    dates=pd.bdate_range("2020-01-02",periods=n)
    p=pd.MultiIndex.from_product([dates,np.arange(1,stocks+1)],names=["date","permno"]).to_frame(index=False)
    ret=rng.normal(0,.015,(n,stocks))
    p["ticker"]="PERMNO:"+p.permno.astype(str)
    p["adj_close"]=p["close"]=(100*np.cumprod(1+ret,axis=0)).reshape(-1)
    p["open"]=p.close*.999
    p["high"]=p.close*1.01; p["low"]=p.close*.99
    p["total_return"]=ret.reshape(-1)
    p["volume"]=rng.uniform(1000,2000,len(p))
    p["shares_outstanding"]=1e6
    p["is_tradable"]=True; p["risk_eligible"]=True
    p["market_cap"]=p.close*1e6
    return p


def test_historical_industry_uses_effective_intervals_and_never_current_backfill():
    dates=pd.bdate_range("2024-01-02",periods=5)
    p=pd.DataFrame({"date":dates,"permno":1})
    h=pd.DataFrame({"permno":[1,1],"effective_from":dates[[1,4]],"effective_to":dates[[2,4]],"siccd":[3571,6021]})
    out=historical_industries(p,h)
    assert out.industry.tolist()==[pd.NA,"SIC2:35","SIC2:35",pd.NA,"SIC2:60"]
    assert not out.classification_is_point_in_time.any()
    changed=h.copy(); changed.loc[1,"siccd"]=6512
    pd.testing.assert_frame_equal(out.iloc[:4],historical_industries(p,changed).iloc[:4])
    h.loc[1,"effective_from"]=dates[2]
    with pytest.raises(ValueError,match="Overlapping"):
        historical_industries(p,h)


def test_joint_industry_size_projection_preserves_both_orthogonality_conditions():
    rng=np.random.default_rng(8)
    industry=pd.Series(np.repeat(["A","B","C"],20))
    size=pd.Series(rng.normal(size=60)+np.repeat([0,2,5],20))
    y=3*size+np.repeat([4,-1,9],20)+rng.normal(size=60)
    dates=pd.Series(pd.Timestamp("2024-01-02"),index=y.index)
    result=neutralize_industry_size(y,dates,industry,size)
    assert result.groupby(industry).mean().abs().max()<1e-12
    assert abs(np.dot(result,size))<1e-10
    missing=industry.copy(); missing.iloc[0]=pd.NA
    assert pd.isna(neutralize_industry_size(y,dates,missing,size).iloc[0])


def test_true_share_turnover_is_split_invariant_and_keeps_legacy_names():
    p=prices(40,1).assign(volume=10000.,shares_outstanding=1e6)
    baseline=ShareTurnover20D()._compute(p)
    p.loc[p.index>=20,["volume","shares_outstanding"]]*=2
    pd.testing.assert_series_equal(baseline,ShareTurnover20D()._compute(p))
    reg=FactorRegistry(); reg.discover()
    assert {"TURNOVER_20D","SHARE_TURNOVER_20D","SHARE_TURNOVER_60D"}<=set(reg.list_factors().name)
    assert len(reg)==44


def test_two_session_high_low_estimator_uses_split_comparable_prices():
    p=prices(80,1).assign(adj_close=100.,close=100.,high=102.,low=98.)
    expected=SpreadHL()._compute(p)
    p.loc[p.index>=30,["close","high","low"]]/=2
    actual=SpreadHL()._compute(p)
    np.testing.assert_allclose(actual,expected,equal_nan=True,atol=1e-12)


@pytest.mark.parametrize("cls",[Turnover20D,Turnover60D,TurnoverAccel,AbnormalVolume,VolumeCV,IlliqAmihud])
def test_liquidity_cache_and_uncached_paths_are_identical_after_shuffling(cls):
    p=prices(165).sample(frac=1,random_state=3)
    cache=PriceFeatureCache(p)
    expected=cls()._compute(p).sort_index()
    actual=cls()._compute(p,_feature_cache=cache).sort_index()
    np.testing.assert_allclose(actual,expected,equal_nan=True,atol=1e-12)


def test_kalman_recovers_from_initial_and_intermittent_missing_observations():
    p=prices(90,1)
    p.loc[[0,1,40],"adj_close"]=np.nan
    _,filtered,velocity,_,_=_run_kalman_on_panel(p)
    assert np.isnan(filtered[:2]).all() and np.isnan(filtered[40])
    assert np.isfinite(filtered[2:40]).all() and np.isfinite(velocity[41:]).all()
    changed=p.copy(); changed.loc[70:,"adj_close"]*=10
    np.testing.assert_allclose(filtered[:70],_run_kalman_on_panel(changed)[1][:70],equal_nan=True)


def test_spectral_entropy_matches_periodogram_and_rejects_gaps():
    x=np.random.default_rng(3).normal(size=100)
    actual=_rolling_spectral_entropy(x,60)
    psd=np.abs(np.fft.fft(x[-60:])[:30])**2
    probability=psd/psd.sum()
    expected=-np.sum(probability*np.log(probability))/np.log(30)
    assert actual[-1]==pytest.approx(expected,abs=1e-12)
    x[70]=np.nan
    assert np.isnan(_rolling_spectral_entropy(x,60)[70:]).all()


def test_higher_moment_features_use_direct_return_and_missing_window_is_unknown():
    p=prices(100,1)
    p["adj_close"]=100.  # Price fallback has no information.
    signal=RollingSkew60D()._compute(p)
    assert np.isfinite(signal.iloc[-1])
    p.loc[70,"total_return"]=np.nan
    assert RollingSkew60D()._compute(p).iloc[70:].isna().all()
    assert HurstExponent60D()._compute(p).iloc[70:].isna().all()


def test_ou_invalid_window_is_missing_and_high_price_deviation_is_short_alpha():
    x=np.random.default_rng(9).normal(0,.01,100)
    for t in range(1,len(x)):x[t]+=.8*x[t-1]
    x[-1]+=.04
    p=prices(100,1); p["adj_close"]=np.exp(x)*100
    raw=OUZScore()._compute(p)
    assert raw.iloc[-1]>0 and raw.iloc[-1]*OUZScore.meta.direction<0
    x[80]=np.nan
    assert np.isnan(_rolling_ou_params(x,60,1/252)[-1][80:]).all()


def test_drawdown_is_contained_within_one_formation_window():
    p=pd.Series([100.,120.,60.,90.,100.,110.])
    result=_rolling_max_drawdown(p,3)
    assert result.iloc[2]==-.5
    assert result.iloc[-1]==0  # The old 120->60 loss has left the window.


@pytest.mark.parametrize("cls",[KalmanVelocity,SpectralEntropy60D,HurstExponent60D,RollingSkew60D,OUZScore,Mom1M,HMMTurbulenceProbability])
def test_future_perturbation_and_historical_prefix_invariance(cls):
    p=prices(350,2)
    cutoff=p.date.unique()[270]
    future=p.date.gt(cutoff)
    changed=p.copy()
    changed.loc[future,["adj_close","close","open","high","low"]]*=4
    changed.loc[future,"total_return"]+=.3
    original=cls()._compute(p).sort_index()
    altered=cls()._compute(changed).sort_index()
    prefix=cls()._compute(p[~future]).sort_index()
    idx=p.index[~future]
    assert original.loc[idx].notna().sum()>20
    np.testing.assert_allclose(original.loc[idx],altered.loc[idx],equal_nan=True,atol=1e-10)
    np.testing.assert_allclose(original.loc[idx],prefix.reindex(idx),equal_nan=True,atol=1e-10)


def learning_panel():
    p=prices(45,10)
    rng=np.random.default_rng(1)
    panel=p[["date","ticker"]].copy()
    panel["MOM_1M"]=rng.normal(size=len(p))
    panel["BOOK_TO_MARKET"]=rng.normal(size=len(p))
    panel["forward_return"]=panel["target"]=panel.MOM_1M*.01+rng.normal(0,.01,len(p))
    calendar=pd.Series(sorted(p.date.unique()))
    panel["label_end"]=panel.date.map(pd.Series(calendar.shift(-2).to_numpy(),index=calendar))
    panel["label_terminal_imputed"]=False
    return panel


def test_inner_and_outer_label_end_boundaries_are_independent_and_strict():
    p=learning_panel()
    fold=next(expanding_date_folds(p.date,WalkForwardConfig(horizon=2,min_train_days=20,validation_days=5,n_splits=1)))
    train,valid=fair.inner_time_masks(p,fold,6)
    assert p.loc[train,"label_end"].max()<p.loc[valid,"date"].min()
    assert p.loc[valid,"label_end"].max()<fold["validation_start"]
    p.loc[p.index[train][0],"label_end"]=fold["validation_start"]
    assert not fair.inner_time_masks(p,fold,6)[0][p.index[train][0]]


def test_capped_training_samples_are_shared_across_information_ablations():
    p=learning_panel()
    train=p.date.lt(p.date.unique()[30]).to_numpy(); test=~train
    kwargs=dict(model="ridge",params={"alpha":10},seeds=[17],max_rows=50,sampling_features=["MOM_1M"],
                sampling_tiebreak_features=["MOM_1M","BOOK_TO_MARKET"])
    _,imp1,_=fair.fit_predict(p,["MOM_1M"],train,test,**kwargs)
    _,imp2,_=fair.fit_predict(p,["MOM_1M","BOOK_TO_MARKET"],train,test,**kwargs)
    assert imp1.attrs["training_key_sha256"]==imp2.attrs["training_key_sha256"]


def test_tied_shared_sampling_key_is_invariant_to_identifier_relabeling():
    n_days,n_assets=30,10
    p=pd.DataFrame({"date":np.repeat(pd.bdate_range("2020-01-01",periods=n_days),n_assets),
        "ticker":np.tile([f"PERMNO:{i}" for i in range(n_assets)],n_days),
        "security":np.tile(np.arange(n_assets),n_days)})
    day=p.date.factorize()[0]
    p["MOM_1M"]=day/20
    p["BOOK_TO_MARKET"]=np.random.default_rng(1).normal(size=len(p))
    p["target"]=np.sin(day/3)
    changed=p.copy()
    changed["ticker"]=changed.security.map(lambda s:f"PERMNO:{9-s}")
    changed=changed.sort_values(["date","ticker"]).reset_index(drop=True)
    features=["MOM_1M","BOOK_TO_MARKET"]
    kwargs=dict(model="ridge",params={"alpha":10},seeds=[17],max_rows=37,
                sampling_features=["MOM_1M"],sampling_tiebreak_features=features)
    predictions=[]
    for frame in (p,changed):
        train=frame.date.lt(p.date.unique()[20]).to_numpy()
        pred,_,_=fair.fit_predict(frame,features,train,~train,**kwargs)
        predictions.append(frame.loc[~train,["date","security"]].assign(prediction=pred)
            .sort_values(["date","security"]).prediction.to_numpy())
    np.testing.assert_allclose(*predictions,rtol=0,atol=1e-7)
    train=p.date.lt(p.date.unique()[20]).to_numpy()
    with pytest.raises(ValueError,match="tie-break"):
        fair.fit_predict(p,features,train,~train,**{**kwargs,"sampling_tiebreak_features":None})
    with pytest.raises(ValueError,match="Identifiers"):
        fair.fit_predict(p,features,train,~train,**{**kwargs,"sampling_tiebreak_features":[*features,"ticker"]})


def test_outer_labels_cannot_select_parameters_or_change_their_own_predictions(monkeypatch):
    p=learning_panel()
    real_fit=fair.fit_predict
    def fast_fit(*args,model,params,**kwargs):
        return real_fit(*args,model="ridge",params={"alpha":params.get("alpha",10)},**kwargs)
    monkeypatch.setattr(fair,"fit_predict",fast_fit)
    settings={"min_train_days":20,"validation_days":5,"n_splits":1,"inner_days":6,"max_train_rows":100,
        "n_jobs":1,"seeds":[17],"sampling_features":["MOM_1M"],
        "model_configs":{"ridge":[{"alpha":1},{"alpha":10}],"lightgbm":[{"alpha":1},{"alpha":10}]}}
    blocks={"price_only":["MOM_1M"],"price_plus_financial":["MOM_1M","BOOK_TO_MARKET"]}
    original=fair.nested_oof(p,blocks,{},settings,horizon=2)
    changed=p.copy(); mask=changed.date.ge(changed.date.unique()[-5])
    changed.loc[mask,["forward_return","target"]]*=-1000
    altered=fair.nested_oof(changed,blocks,{},settings,horizon=2)
    assert original[2]==altered[2]
    np.testing.assert_allclose(original[0][["ridge","lightgbm"]],altered[0][["ridge","lightgbm"]])


def test_issuer_concentration_and_prediction_selection_ignore_future_labels():
    p=prices(1,200).assign(permco=lambda f:(f.permno//2),score=lambda f:f.permno,forward_return=np.nan)
    w=fair.issuer_capped_weights(p,"score",stock_cap=.015,issuer_cap=.03)
    assert w.abs().max()<=.015 and w.abs().groupby(p.permco).sum().max()<=.03+1e-12
    assert abs(w.sum())<1e-12
    pd.testing.assert_series_equal(w,fair.issuer_capped_weights(p.assign(forward_return=1000),"score",stock_cap=.015,issuer_cap=.03))


def test_hac_and_multiple_testing_keep_missing_dates_and_tied_samples():
    q=benjamini_hochberg([.01,.04,.03,np.nan])
    np.testing.assert_allclose(q[:3],[.03,.04,.04])
    a=pd.Series([.1,np.nan,.2,.1,-.1,.1,.2])
    assert hac_summary(a,5)["hac_lags"]>=4
    n=20
    industry=pd.Series(np.repeat(["A","B"],n//2))
    signal=pd.Series(np.arange(n,dtype=float))
    ic=grouped_rank_ic(signal,signal,pd.Series(pd.Timestamp("2024-01-02"),index=signal.index),industry)
    assert ic.eq(1).all()


def test_turnover_counts_membership_entries_and_exits_without_observed_zero_rows():
    dates=pd.Series(pd.to_datetime(["2024-01-02"]*2+["2024-01-03"]*2))
    assert calendar_turnover(pd.Series([.5,-.5,.5,-.5]),dates,pd.Series([1,2,3,4]))==1


def test_uniform_diagnostics_keep_raw_and_adjusted_financial_samples_paired():
    p=prices(70,30)
    rng=np.random.default_rng(18)
    p["MOM_12_1M"]=rng.normal(size=len(p))
    p["GROSS_PROFITABILITY"]=rng.normal(size=len(p))
    p["decision_industry"]="SIC2:"+(p.permno%3).astype(str)
    p["decision_log_size"]=np.log(p.market_cap).astype("Float64")
    p["decision_financial_or_real_estate"]=False
    p.loc[p.permno.eq(1),"decision_industry"]=pd.NA
    p["fwd_1"]=p["fwd_20"]=rng.normal(size=len(p))*.01
    catalog=pd.DataFrame({"factor":["MOM_12_1M","GROSS_PROFITABILITY"],"direction":[1,1],
                           "lag_sessions":[21,1],"signal_type":"cross_sectional"})
    result=factor_diagnostics(p,catalog,horizons=(1,20))
    summary=result["summary"]
    matched=summary[summary.variant.eq("raw_matched")].set_index("horizon")
    adjusted=summary[summary.variant.eq("industry_size")].set_index("horizon")
    pd.testing.assert_series_equal(matched.labeled_rows,adjusted.labeled_rows)
    assert len(summary)==8


def test_real_execution_tables_persist_separately_from_in_memory_ledger(tmp_path):
    from scripts.compare_wrds_models import execute_book
    p=prices(40,100).assign(permco=lambda f:f.permno,volume=100000.)
    p["is_terminal"]=False
    tape=tmp_path/"prices.parquet";p.to_parquet(tape,index=False)
    dates=p.date.unique()[-10:]
    oof=p[p.date.isin(dates)][["date","ticker"]].copy()
    oof["block"]="price_only"
    oof["ridge"]=np.random.default_rng(9).normal(size=len(oof))
    oof.to_parquet(tmp_path/"oof.parquet",index=False)
    cfg={"initial_capital":1000000,"max_participation_rate":.1,"min_names":20,
         "max_stock_weight":.015,"max_issuer_gross_weight":.03}
    result=execute_book(str(tmp_path),"price_only","ridge",str(tape),cfg,5,"total_loss",1)
    dest=tmp_path/"backtests/price_only/ridge/cost_1x"
    trades=pd.read_parquet(dest/"trades.parquet")
    pnl=pd.read_parquet(dest/"daily_pnl.parquet")
    assert result["executed_trades"]>0 and len(pnl)==10
    assert "daily_ledger" not in trades.attrs
