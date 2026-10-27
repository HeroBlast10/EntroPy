import numpy as np
import pandas as pd
from quant_platform.core.data.research_universe import broad_membership


def sample():
    dates=pd.bdate_range("2019-01-01",periods=12)
    return pd.DataFrame({"date":dates.repeat(4),"permno":[1,2,3,4]*12,"permco":[10,10,30,40]*12,
        "ticker":["A","B","C","D"]*12,"close":[10.,10.,8.,7.]*12,"volume":[1e6,2e6,1e6,1e6]*12,
        "market_cap":[100.,100.,150.,90.]*12,"securitytype":"EQTY","securitysubtype":"COM",
        "sharetype":"NS","usincflg":"Y","primaryexch":"N","is_tradable":True,"is_terminal":False})


def test_company_aggregation_one_class_and_no_current_or_future_price_selection():
    data=sample();kwargs=dict(top_n=2,diagnostic_top_n=3,adv_sessions=3,min_listing_days=0,min_adv=1)
    base=broad_membership(data,**kwargs)
    last=data.date.max();changed=data.copy();changed.loc[changed.date.eq(last),["close","market_cap"]]=1e9
    # The contemporaneous cap is observational metadata, not a selector.
    pd.testing.assert_frame_equal(base.drop(columns="company_market_cap"),
        broad_membership(changed,**kwargs).drop(columns="company_market_cap"))
    day=base.loc[base.date.eq(last)]
    assert day.permno.tolist()==[2,3,4] and day.loc[day.permno.eq(2),"lagged_company_market_cap"].iloc[0]==200
    assert day.loc[day.pass_all_filters,"permno"].tolist()==[2,3]


def test_liquidity_is_lagged_and_rank_cut_is_not_refilled_with_smaller_names():
    data=sample();data.loc[data.permno.isin([1,2]),"close"]=4.
    out=broad_membership(data,top_n=2,diagnostic_top_n=3,adv_sessions=3,min_listing_days=0,min_adv=1)
    last=out.loc[out.date.eq(out.date.max())]
    assert last.loc[last.pass_all_filters,"permno"].tolist()==[3]
    assert last.lagged_cap_rank.max()==3


def test_missing_shared_session_breaks_adv_history():
    data=sample();dates=sorted(data.date.unique());data=data.loc[~(data.permno.eq(3)&data.date.eq(dates[6]))]
    out=broad_membership(data,top_n=3,diagnostic_top_n=3,adv_sessions=3,min_listing_days=0,min_adv=1)
    after=out.loc[out.permno.eq(3)&out.date.eq(dates[8])]
    assert after.lagged_adv_dollar.isna().all()
def test_terminal_classification_change_is_retained_and_validated():
    import pandas as pd
    import pytest
    from quant_platform.core.data.research_universe import append_terminal_observations
    common=pd.DataFrame({"date":pd.to_datetime(["2020-01-02"]),"permno":[1],"ret":[.01],"delisting_flag":["N"],"securitytype":["EQTY"]})
    terminal=pd.DataFrame({"date":pd.to_datetime(["2020-01-03"]),"permno":[1],"ret":[-.4],"delisting_flag":["Y"],"securitytype":["N/A"]})
    result=append_terminal_observations(common,terminal)
    assert len(result)==2 and result.iloc[-1]["ret"]==-.4 and result.iloc[-1].securitytype=="N/A"
    from quant_platform.core.data.returns import forward_total_returns
    labels=forward_total_returns(result.assign(ticker="A",total_return=result.ret,
        close=[10.,np.nan],is_terminal=result.delisting_flag.eq("Y")),1)
    assert np.isclose(labels.iloc[0].forward_return,-.4)
    assert labels.iloc[0].label_terminal_settled
    with pytest.raises(ValueError,match="Conflicting"):
        append_terminal_observations(result,terminal.assign(ret=.2))
