"""Conservative historical links and publication-aware alternative features."""
from __future__ import annotations
import numpy as np
import pandas as pd


def require_unadjusted_original(description):
    text=str(description).lower()
    if "unadjusted" not in text or "normalized" in text or "restated" in text:
        raise ValueError("IBES role requires explicitly unadjusted, original vendor data")


def next_session(values,calendar):
    dates=pd.DatetimeIndex(calendar)
    input_dates=pd.to_datetime(values)
    pos=dates.searchsorted(input_dates,side="right")
    out=pd.Series(pd.NaT,index=values.index,dtype="datetime64[ns]")
    # Outside the supplied calendar we cannot invent the missing sessions.
    valid=input_dates.notna()&input_dates.ge(dates[0])&(pos<len(dates))
    out.loc[valid]=dates.take(pos[valid]).to_numpy()
    return out


def historical_link(events,links,key,source_date):
    """Only a unique contemporaneous issuer survives; ties are not row-order choices."""
    data=events.copy().reset_index(drop=True);data["_event_id"]=np.arange(len(data))
    mapping=links.copy()
    for col in ("sdate","edate"):mapping[col]=pd.to_datetime(mapping[col])
    joined=data.merge(mapping[[key,"permno","sdate","edate","score"]],on=key,how="inner")
    when=pd.to_datetime(joined[source_date])
    joined=joined.loc[when.ge(joined.sdate)&(joined.edate.isna()|when.le(joined.edate))&joined.score.le(1)].copy()
    unique=joined.groupby("_event_id").permno.transform("nunique").eq(1)
    joined=joined.loc[unique].sort_values(["_event_id","edate"],na_position="last").drop_duplicates("_event_id",keep="last")
    joined["permno"]=joined.permno.astype("int64")
    return joined.drop(columns=["_event_id","sdate","score"]).rename(columns={"edate":"link_end"})


def project_event(spine,events,value,*,max_age_days=45):
    """Missing updates explicitly clear values; expired links cannot be resurrected."""
    data=events[["permno","available_date","link_end",value]].copy()
    data["available_date"]=pd.to_datetime(data.available_date);data["link_end"]=pd.to_datetime(data.link_end)
    data=data.dropna(subset=["available_date"]).sort_values("available_date")
    if data.duplicated(["permno","available_date"]).any():
        raise ValueError(f"Ambiguous source events for {value}")
    base=spine[["date","permno"]].copy();base["_row"]=np.arange(len(base))
    if data.empty:return pd.DataFrame({value:np.nan,f"{value}_available_date":pd.NaT},index=spine.index)
    merged=pd.merge_asof(base.sort_values("date"),data,on=None,left_on="date",right_on="available_date",by="permno",direction="backward")
    valid=(merged.date-merged.available_date).dt.days.le(max_age_days)&(merged.link_end.isna()|merged.date.le(merged.link_end))
    merged[value]=merged[value].where(valid)
    merged=merged.sort_values("_row")
    return pd.DataFrame({value:merged[value].to_numpy(),f"{value}_available_date":merged.available_date.where(valid).to_numpy()},index=spine.index)
