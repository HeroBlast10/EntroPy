"""Archive-aware IBES and 30-day OptionMetrics proxies, strictly delayed for decisions."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from quant_platform.core.data.alternative_research import next_session,historical_link,project_event
from quant_platform.core.data.returns import forward_total_returns
from quant_platform.core.alpha_models.nonlinear_alpha import daily_rank_ic
from quant_platform.core.evaluation.factor_diagnostics import hac_summary,benjamini_hochberg
from quant_platform.core.utils.provenance import write_run_metadata


def parts(root,name):
    return pd.concat([pd.read_parquet(f) for f in sorted(root.glob(f"{name}_*.parquet"))],ignore_index=True)


def numeric(data,columns):
    for col in columns:data[col]=pd.to_numeric(data[col],errors="coerce").astype(float)
    return data


def collapse(events,values):
    """Conflicting same-date records become missing; never select a convenient value."""
    group=events.groupby(["permno","available_date"],sort=False)
    result=group[values].first().mask(group[values].nunique(dropna=False).gt(1))
    result["link_end"]=group.link_end.max()
    return result.reset_index()


def price_at(events,prices,time_col,*,snapshot_close=False):
    q=events[["permno",time_col]].copy();q["_row"]=np.arange(len(q))
    px=prices[["permno","date","close"]].rename(columns={"date":"price_date"}).sort_values("price_date")
    merged=pd.merge_asof(q.sort_values(time_col),px,left_on=time_col,right_on="price_date",by="permno",direction="backward",allow_exact_matches=snapshot_close)
    result=merged.close.where((merged[time_col]-merged.price_date).dt.days.le(7)).sort_index()
    merged["_price"]=result
    return merged.sort_values("_row")._price.to_numpy()


def action_counts(events,actions,date_col):
    query=events[["permno",date_col]].copy();query["_row"]=np.arange(len(query))
    action=actions.sort_values(["permno","date"]).drop_duplicates(["permno","date"]).copy()
    action["_action_count"]=action.groupby("permno").cumcount()+1
    got=pd.merge_asof(query.sort_values(date_col),action[["permno","date","_action_count"]].sort_values("date"),
        left_on=date_col,right_on="date",by="permno",direction="backward")
    return got.sort_values("_row")._action_count.fillna(0).to_numpy()


def build(raw,output):
    if output.exists():raise ValueError("Choose a fresh alternative feature output")
    if not (raw/"completion.json").exists():raise ValueError("Alternative pull is incomplete")
    repair=raw/"ibes_unadjusted_repair"
    if not (repair/"completion.json").exists():raise ValueError("Explicit original/unadjusted IBES role verification is required")
    output.mkdir(parents=True)
    prices_path=ROOT/"data/wrds/p0/prices.parquet"
    px=pd.read_parquet(prices_path,columns=["date","permno","ticker","close","volume","total_return","is_terminal"])
    px=px.sort_values(["permno","date"]).reset_index(drop=True);calendar=pd.DatetimeIndex(sorted(px.date.unique()))
    ibes_links=pd.read_parquet(raw/"ibes_links.parquet");option_links=pd.read_parquet(raw/"option_links.parquet")
    actions=parts(raw,"corporate_actions");actions["permno"]=actions.permno.astype("int64")
    factor=pd.to_numeric(actions.dlyfacprc,errors="coerce")
    actions=actions.loc[factor.notna()&factor.ne(1)&factor.ne(0)].copy()
    summary=numeric(parts(raw,"eps_summary"),["meanest","stdev","numest","numup","numdown"])
    summary=historical_link(summary,ibes_links,"ticker","statpers")
    summary["available_date"]=next_session(summary.statpers,calendar)
    annual=summary.loc[summary.fpi.eq("1")].copy().sort_values(["permno","statpers"])
    annual=annual.drop_duplicates(["permno","statpers","fpedats"])
    annual["IBES_REVISION_BREADTH"]=(annual.numup-annual.numdown)/annual.numest.where(annual.numest.gt(0))
    annual["IBES_DISPERSION"]=annual.stdev/annual.meanest.abs().clip(lower=.05)
    for col in ("IBES_REVISION_BREADTH","IBES_DISPERSION"):
        annual[col]=annual[col].where(annual.curcode.eq("USD")&annual.numest.ge(2))
    annual["previous_period"]=annual.groupby("permno").fpedats.shift()
    annual["previous_mean"]=annual.groupby("permno").meanest.shift()
    annual["previous_snapshot"]=annual.groupby("permno").statpers.shift()
    annual["_action_end"]=action_counts(annual,actions,"statpers")
    previous=annual.dropna(subset=["previous_snapshot"]).copy()
    previous["_guard_start"]=previous.previous_snapshot-pd.Timedelta(days=35)
    old_actions=pd.Series(action_counts(previous,actions,"_guard_start"),index=previous.index)
    annual["_action_start"]=old_actions
    annual["IBES_EPS_REVISION"]=(annual.meanest-annual.previous_mean)/price_at(annual,px,"statpers",snapshot_close=True)
    valid=annual.fpedats.eq(annual.previous_period)&annual._action_end.eq(annual._action_start)&annual.curcode.eq("USD")
    annual["IBES_EPS_REVISION"]=annual.IBES_EPS_REVISION.where(valid)
    events={col:(collapse(annual,[col]),45) for col in ("IBES_REVISION_BREADTH","IBES_DISPERSION","IBES_EPS_REVISION")}
    target=numeric(parts(repair,"target_summary"),["meanptg","numest"])
    target=historical_link(target,ibes_links,"ticker","statpers");target["available_date"]=next_session(target.statpers,calendar)
    # The target and price are both unadjusted at their snapshot date.
    target["IBES_TARGET_IMPLIED_RETURN"]=(target.meanptg/price_at(target,px,"statpers",snapshot_close=True)-1).where(target.curr.eq("USD")&target.numest.ge(2))
    target["_guard_start"]=target.statpers-pd.Timedelta(days=35)
    target["IBES_TARGET_IMPLIED_RETURN"]=target.IBES_TARGET_IMPLIED_RETURN.where(
        action_counts(target,actions,"_guard_start")==action_counts(target,actions,"statpers"))
    events["IBES_TARGET_IMPLIED_RETURN"]=(collapse(target,["IBES_TARGET_IMPLIED_RETURN"]),45)
    actual=numeric(parts(repair,"eps_actual"),["value"])
    actual=actual.loc[actual.pdicity.eq("QTR")&actual.anndats.notna()].copy()
    actual["source_date"]=actual[["anndats","actdats"]].max(axis=1)
    actual=historical_link(actual,ibes_links,"ticker","source_date")
    actual=actual.sort_values("source_date").drop_duplicates(["permno","pends"],keep="first")
    actual["available_date"]=next_session(actual.source_date,calendar)
    qtr=summary.loc[summary.fpi.eq("6")].rename(columns={"fpedats":"pends","available_date":"forecast_available","statpers":"forecast_snapshot","meanest":"consensus","curcode":"forecast_currency"})
    qtr=qtr.sort_values("forecast_available").drop_duplicates(["permno","pends","forecast_available"],keep=False)
    surprise=pd.merge_asof(actual.sort_values("anndats"),qtr[["permno","pends","forecast_available","forecast_snapshot","consensus","forecast_currency"]].dropna(subset=["forecast_available"]).sort_values("forecast_available"),
        left_on="anndats",right_on="forecast_available",by=["permno","pends"],direction="backward",allow_exact_matches=False)
    surprise["_scaled_surprise"]=(surprise.value-surprise.consensus)/price_at(surprise,px,"anndats")
    noaction=surprise.forecast_snapshot.notna()
    safe=surprise.loc[noaction]
    safe=safe.copy();safe["_guard_start"]=safe.forecast_snapshot-pd.Timedelta(days=35)
    safe_action=pd.Series(action_counts(safe,actions,"_guard_start")==action_counts(safe,actions,"anndats"),index=safe.index)
    good=surprise.curr_act.eq("USD")&surprise.forecast_currency.eq("USD")&safe_action.reindex(surprise.index).fillna(False)
    surprise["_scaled_surprise"]=surprise._scaled_surprise.where(good)
    surprise=surprise.sort_values(["permno","available_date"])
    sigma=surprise.groupby("permno")._scaled_surprise.transform(lambda s:s.shift().rolling(8,min_periods=4).std())
    surprise["IBES_SUE_PRICE_SCALED"]=surprise._scaled_surprise/sigma.where(sigma.gt(1e-8))
    events["IBES_SUE_PRICE_SCALED"]=(collapse(surprise,["IBES_SUE_PRICE_SCALED"]),126)
    surface=numeric(parts(raw,"option_surface"),["put25","put50","call25","call50"])
    surface=historical_link(surface,option_links,"secid","date")
    surface=surface.drop_duplicates(["permno","date"],keep=False)
    for col in ("put25","put50","call25","call50"):surface[col]=surface[col].where(surface[col].gt(0)&surface[col].lt(10))
    volume=numeric(parts(raw,"option_volume"),["option_contract_volume"])
    volume=historical_link(volume,option_links,"secid","date").drop_duplicates(["permno","date"],keep=False)
    returns=px.pivot(index="date",columns="permno",values="total_return").sort_index()
    rv=returns.rolling(21,min_periods=21).std()*np.sqrt(252)
    rv=rv.stack(future_stack=True).rename("rv").reset_index()
    surface=surface.merge(rv,on=["date","permno"],how="left",validate="one_to_one")
    surface["OPTION_PUT25_SKEW"]=surface.put25-surface.call50
    surface["OPTION_CP50_IV_DIFF"]=surface.call50-surface.put50
    surface["OPTION_IV_MINUS_RV"]=(surface.call50+surface.put50)/2-surface.rv
    surface["available_date"]=next_session(surface.date,calendar)
    for col in ("OPTION_PUT25_SKEW","OPTION_CP50_IV_DIFF","OPTION_IV_MINUS_RV"):events[col]=(surface,7)
    volume=volume.merge(px[["date","permno","volume"]],on=["date","permno"],how="left",validate="one_to_one")
    volume["OPTION_CONTRACTS_PER_MILLION_SHARES"]=1e6*volume.option_contract_volume/volume.volume.where(volume.volume.gt(0))
    volume["available_date"]=next_session(volume.date,calendar)
    events["OPTION_CONTRACTS_PER_MILLION_SHARES"]=(volume,7)
    features=px[["date","permno","ticker"]].copy()
    coverage=[]
    for name,(event,age) in events.items():
        event=event.dropna(subset=["available_date"])
        projected=project_event(features,event,name,max_age_days=age)
        features=pd.concat([features,projected],axis=1)
        coverage.append({"feature":name,"finite_rows":int(features[name].notna().sum()),"event_rows":len(event),"max_age_calendar_days":age})
        assert (features[f"{name}_available_date"].dropna()<=features.loc[features[f"{name}_available_date"].notna(),"date"]).all()
        print(f"built {name}: {coverage[-1]['finite_rows']:,} valid rows",flush=True)
    features.to_parquet(output/"features.parquet",index=False);pd.DataFrame(coverage).to_csv(output/"coverage.csv",index=False)
    # Descriptive matched-availability scorecards, not automatic model admission.
    eligible=pd.read_parquet(ROOT/"data/wrds/batch3/neutral_2013_2025/models/cross_sectional/price_only/ridge/oof.parquet",columns=["date","ticker"])
    diagnostics=[]
    for horizon in (20,60,120):
        labels=forward_total_returns(px,horizon,missing_terminal_return="total_loss")
        scored=eligible.merge(features,on=["date","ticker"],how="left").merge(labels[["date","ticker","forward_return"]],on=["date","ticker"],how="left")
        for name in events:
            ic=daily_rank_ic(scored,name);stats=hac_summary(ic,horizon)
            diagnostics.append({"feature":name,"horizon":horizon,"coverage":float(scored[name].notna().mean()),**stats})
    stats=pd.DataFrame(diagnostics);stats["q_value"]=benjamini_hochberg(stats.p_value)
    stats.to_csv(output/"factor_ic.csv",index=False)
    write_run_metadata(output/"run_metadata.json",stage="alternative_features_and_reviewed_history_scorecard",
        inputs={"driver":Path(__file__),"raw_completion":raw/"completion.json","ibes_role_repair":repair/"completion.json","prices":prices_path},
        params={"delay":"first exchange session strictly after snapshot/announcement/activation date",
            "sue":"quarterly unadjusted actual minus pre-announcement monthly consensus; divide by pre-event price, then past 8-event std (min4); corporate-action windows excluded",
            "eps_revision":"same fiscal period, unadjusted EPS; corporate-action windows excluded",
            "vendor_split_lag_guard":"exclude non-unit CIZ price adjustment factors from 35 calendar days before the older snapshot through the comparison endpoint; not a vendor-vintage certificate",
            "option_spread":"constant-maturity delta proxy; not an exact matched-strike Bali-Hovakimian replication",
            "volume_units":"option contracts per million underlying shares; no assumed contract multiplier"},
        extra={"historical_review_not_untouched_holdout":True,"source_point_in_time_proven":False,"native_availability_dates_used":True,"production_admitted":False})


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--raw",type=Path,default=ROOT/"data/wrds/batch4/alternative_raw");p.add_argument("--output",type=Path,default=ROOT/"data/wrds/batch4/alternative_features_20261024")
    a=p.parse_args();build(a.raw.resolve(),a.output.resolve())
