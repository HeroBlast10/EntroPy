import pandas as pd
import numpy as np
import pytest
from quant_platform.core.data.alternative_research import next_session,historical_link,project_event


def test_dates_are_conservatively_delayed_and_future_links_do_not_backcast():
    dates=pd.bdate_range("2020-01-03",periods=5)
    assert next_session(pd.Series([pd.Timestamp("2020-01-03")]),dates).iloc[0]==pd.Timestamp("2020-01-06")
    assert pd.isna(next_session(pd.Series([pd.Timestamp("2019-12-01")]),dates).iloc[0])
    events=pd.DataFrame({"ticker":["X","X"],"event_date":pd.to_datetime(["2020-01-03","2020-01-07"]),"v":[1,2]})
    links=pd.DataFrame({"ticker":["X","X"],"permno":[1,2],"sdate":pd.to_datetime(["2019-01-01","2020-01-07"]),"edate":pd.to_datetime(["2020-01-06",None]),"score":0})
    got=historical_link(events,links,"ticker","event_date")
    assert got.permno.tolist()==[1,2]
    ambiguous=pd.concat([links,pd.DataFrame({"ticker":["X"],"permno":[3],"sdate":[pd.Timestamp("2019-01-01")],"edate":[pd.NaT],"score":[1]})],ignore_index=True)
    assert historical_link(events,ambiguous,"ticker","event_date").empty


def test_null_event_clears_previous_value_and_link_expiration_is_enforced():
    dates=pd.bdate_range("2020-01-01",periods=6);spine=pd.DataFrame({"date":dates,"permno":1})
    events=pd.DataFrame({"permno":[1,1],"available_date":[dates[0],dates[2]],"link_end":[dates[3],dates[3]],"v":[2.,np.nan]})
    result=project_event(spine,events,"v")
    assert result.v.iloc[:2].eq(2).all() and result.v.iloc[2:].isna().all()
    single=project_event(spine,events.iloc[:1],"v")
    assert single.v.iloc[4:].isna().all()


def test_snapshot_price_can_include_known_close_but_earnings_denominator_is_pre_event():
    from scripts.build_wrds_alternative_features import price_at,collapse
    dates=pd.to_datetime(["2020-01-02","2020-01-03"])
    prices=pd.DataFrame({"date":dates,"permno":[1,1],"close":[100.,50.]})
    events=pd.DataFrame({"event_date":[dates[1]],"permno":[1]})
    assert price_at(events,prices,"event_date",snapshot_close=True)[0]==50
    assert price_at(events,prices,"event_date")[0]==100
    ambiguous=pd.DataFrame({"permno":[1,1],"available_date":[dates[1]]*2,"link_end":[pd.NaT,pd.NaT],"v":[1.,2.]})
    assert pd.isna(collapse(ambiguous,["v"]).v.iloc[0])


def test_ibes_basis_is_verified_from_vendor_description_not_n_prefix():
    from quant_platform.core.data.alternative_research import require_unadjusted_original
    require_unadjusted_original("Detail History - Actuals Unadjusted File")
    with pytest.raises(ValueError):require_unadjusted_original("Detail History - Normalized Actuals File")
    with pytest.raises(ValueError):require_unadjusted_original("Normalized Actuals Unadjusted File")
