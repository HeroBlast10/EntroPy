"""Causal company-cap breadth rules, independent of present index membership."""
from __future__ import annotations
import numpy as np
import pandas as pd


def append_terminal_observations(common, terminal):
    """Preserve settlement rows even when CIZ reclassifies them as N/A / UNK."""
    if not terminal.delisting_flag.eq("Y").all():
        raise ValueError("Settlement supplement contains a nonterminal row")
    extra=terminal.loc[terminal.permno.isin(common.permno.unique())].copy()
    keys=["date","permno"]
    overlap=common.merge(extra,on=keys,suffixes=("_old","_new"))
    for col in ("ret","delisting_flag"):
        if not overlap.empty and not (overlap[f"{col}_old"].eq(overlap[f"{col}_new"]) |
            (overlap[f"{col}_old"].isna()&overlap[f"{col}_new"].isna())).all():
            raise ValueError("Conflicting terminal observations")
    result=pd.concat([common,extra],ignore_index=True).drop_duplicates(keys,keep="first")
    if result.duplicated(keys).any():raise ValueError("Duplicate terminal keys")
    return result.sort_values(keys).reset_index(drop=True)


def broad_membership(prices, *, top_n=2000, diagnostic_top_n=3000, min_price=5.,
                     min_adv=2_000_000., adv_sessions=20, min_listing_days=60):
    if top_n < 1 or diagnostic_top_n < top_n or adv_sessions < 2:
        raise ValueError("Invalid breadth configuration")
    if prices.duplicated(["date", "permno"]).any():
        raise ValueError("Duplicate CRSP universe keys")
    data=prices.copy().sort_values(["permno","date"]).reset_index(drop=True)
    dates=pd.DatetimeIndex(sorted(data.date.unique()))
    data["_session"]=dates.get_indexer(data.date)
    common=data.securitytype.eq("EQTY")&data.securitysubtype.eq("COM")
    # Aggregate all common classes before choosing the representative class.
    cap=data.market_cap.where(data.market_cap.gt(0)&common)
    unknown=common&cap.isna()&~data.is_terminal.eq(True)
    cap_group=[data.date,data.permco]
    total=cap.groupby(cap_group).transform("sum",min_count=1)
    total=total.mask(unknown.groupby(cap_group).transform("any"))
    data["company_market_cap"]=total
    data["_dollar_volume"]=(data.close*data.volume).where(data.is_tradable)
    group=data.groupby("permno",sort=False)
    rolling=group._dollar_volume.transform(lambda s:s.rolling(adv_sessions,min_periods=adv_sessions).mean())
    contiguous=(data._session-group._session.shift(adv_sessions-1)).eq(adv_sessions-1)
    data["_adv"]=rolling.where(contiguous)
    data["_eligible_type"]=(common&data.sharetype.eq("NS")&data.usincflg.eq("Y")&data.primaryexch.isin(["N","A","Q"]))
    group=data.groupby("permno",sort=False)
    prior=group[["close","company_market_cap","_adv","_eligible_type","_session","permco"]].shift(1)
    data["lagged_price"]=prior.close
    data["lagged_adv_dollar"]=prior._adv
    data["lagged_company_market_cap"]=prior.company_market_cap
    data["days_since_ipo"]=(data.date-group.date.transform("min")).dt.days
    present=(data._session-prior._session).eq(1)
    candidates=data.loc[present&data.permco.eq(prior.permco)&prior._eligible_type.eq(True)&prior.company_market_cap.gt(0)
        &data.days_since_ipo.ge(min_listing_days)].copy()
    # A historical, liquid representative prevents multiple share classes
    # from being counted as independent companies or inflating breadth.
    candidates=candidates.sort_values(["date","permco","lagged_adv_dollar","permno"],ascending=[True,True,False,True])
    candidates=candidates.drop_duplicates(["date","permco"],keep="first")
    candidates=candidates.sort_values(["date","lagged_company_market_cap","permno"],ascending=[True,False,True])
    candidates["lagged_cap_rank"]=candidates.groupby("date").cumcount()+1
    candidates=candidates.loc[candidates.lagged_cap_rank.le(diagnostic_top_n)].copy()
    candidates["liquidity_pass"]=(candidates.lagged_price.ge(min_price)&candidates.lagged_adv_dollar.ge(min_adv))
    candidates["pass_all_filters"]=candidates.lagged_cap_rank.le(top_n)&candidates.liquidity_pass
    candidates["cap_bucket"]=pd.cut(candidates.lagged_cap_rank,[0,500,1000,2000,3000],
        labels=["1-500","501-1000","1001-2000","2001-3000"])
    candidates["membership_is_point_in_time"]=True
    candidates["membership_source"]=f"crsp_ciz_lagged_company_cap_top_{top_n}"
    keep=["date","permno","permco","ticker","company_market_cap","lagged_company_market_cap","lagged_cap_rank",
          "lagged_price","lagged_adv_dollar","days_since_ipo","liquidity_pass","pass_all_filters","cap_bucket",
          "membership_is_point_in_time","membership_source"]
    return candidates[keep].sort_values(["date","permno"]).reset_index(drop=True)
