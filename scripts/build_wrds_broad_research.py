"""Company-cap 500/2000 controls and size-stratified IC on a historical CIZ tape."""
from __future__ import annotations
import argparse
import gc
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from quant_platform.core.data.wrds import canonical_prices
from quant_platform.core.data.research_universe import broad_membership,append_terminal_observations
from quant_platform.core.data.returns import forward_total_returns
from quant_platform.core.signals.cross_sectional.volatility import _rolling_idiovol
from quant_platform.core.alpha_models.nonlinear_alpha import daily_rank_ic
from quant_platform.core.evaluation.factor_diagnostics import hac_summary
from quant_platform.core.utils.provenance import write_run_metadata,fingerprint


def build(raw,output):
    if output.exists():raise ValueError("Choose a fresh broad-universe output")
    if not (raw/"completion.json").exists():raise ValueError("The CIZ pull is incomplete")
    output.mkdir(parents=True)
    paths=sorted(raw.glob("common_*.parquet"))
    data=pd.concat([pd.read_parquet(p) for p in paths],ignore_index=True)
    supplement=raw/"terminal_repair/terminal.parquet"
    if not (supplement.parent/"completion.json").exists():
        raise ValueError("CIZ terminal-classification supplement is required")
    data=append_terminal_observations(data,pd.read_parquet(supplement))
    for col in ("delisting_flag","securitytype","securitysubtype","sharetype","usincflg","primaryexch","conditionaltype","tradingstatusflg"):
        data[col]=data[col].astype("category")
    missing=data.delisting_flag.eq("Y")&data.ret.isna();imputed=int(missing.sum());data.loc[missing,"ret"]=-1.
    prices=canonical_prices(data);del data;gc.collect()
    print(f"canonical broad tape: {len(prices):,} rows",flush=True)
    universe=broad_membership(prices)
    universe.to_parquet(output/"universe.parquet",index=False)
    company=prices.market_cap.where(prices.market_cap.gt(0)).groupby([prices.date,prices.permco]).transform("sum",min_count=1)
    unknown=(prices.market_cap.isna()&~prices.is_terminal).groupby([prices.date,prices.permco]).transform("any")
    prices["company_market_cap"]=company.where(~unknown)
    keep=["date","permno","permco","ticker","close","adj_close","open","high","low","volume","shares_outstanding",
          "market_cap","company_market_cap","total_return","is_tradable","is_terminal","siccd"]
    prices=prices[keep].copy();prices.to_parquet(output/"prices.parquet",index=False)
    counts=universe.loc[universe.date.ge("2010-01-01")].groupby("date").agg(
        liquid_top2000=("pass_all_filters","sum"),liquid_top3000=("liquidity_pass","sum"))
    counts.to_csv(output/"daily_breadth.csv")
    print(f"universe median investable names: {counts.liquid_top2000.median():.0f}",flush=True)
    ret=prices.pivot(index="date",columns="ticker",values="total_return").sort_index()
    cap=prices.pivot(index="date",columns="ticker",values="company_market_cap").reindex_like(ret)
    mask=universe.pivot(index="date",columns="ticker",values="pass_all_filters").reindex_like(ret).eq(True)
    observed=prices.pivot(index="date",columns="ticker",values="close").reindex_like(ret).notna()
    market=ret.where(mask).mean(axis=1)
    ranks=cap.shift(1).where(mask).rank(axis=1,pct=True)
    smb=ret.where(mask&ranks.le(.25)).mean(axis=1)-ret.where(mask&ranks.ge(.75)).mean(axis=1)
    proxy=pd.DataFrame({"market":market,"size":smb}).fillna(0.)
    proxy.to_parquet(output/"factor_proxies.parquet")
    del cap,ranks;gc.collect()
    residual=np.full(ret.shape,np.nan,dtype=np.float32);idio=np.full(ret.shape,np.nan,dtype=np.float32)
    with threadpool_limits(limits=1):
        for j,name in enumerate(ret.columns):
            present=np.flatnonzero(observed[name].to_numpy())
            if not len(present):continue
            start,stop=present[0],present[-1]+1
            r=ret[name].iloc[start:stop].to_numpy(dtype=float)
            factors=proxy.iloc[start:stop].to_numpy(dtype=float)
            x=np.c_[np.ones(len(r)),factors];errors=np.full(len(r),np.nan)
            for end in range(504,len(r),21):
                window=r[end-504:end];valid=np.isfinite(window)
                if valid.sum()<252:continue
                beta=np.linalg.lstsq(x[end-504:end][valid],window[valid],rcond=None)[0]
                until=min(end+21,len(r));errors[end:until]=r[end:until]-x[end:until]@beta
            residual[start:stop,j]=pd.Series(errors).rolling(231,min_periods=126).sum().shift(21).to_numpy(dtype=np.float32)
            idio[start:stop,j]=pd.Series(_rolling_idiovol(r,factors[:,0],60)).shift(1).to_numpy(dtype=np.float32)
            if (j+1)%1000==0:print(f"broad residual/idio factors: {j+1}/{len(ret.columns)}",flush=True)
    log=np.log1p(ret.where(ret.gt(-1),0))
    momentum=np.expm1(log.rolling(231,min_periods=231).sum()).mask(ret.eq(-1).rolling(231).sum().gt(0),-1)
    momentum=momentum.where(ret.notna().rolling(231).sum().eq(231)).shift(21)
    keys=pd.MultiIndex.from_frame(universe[["date","ticker"]])
    factors=universe.copy()
    for name,array in (("RESID_MOM_12_1M",residual),("IDIOVOL",idio),("MOM_12_1M",momentum.to_numpy())):
        wide=pd.DataFrame(array,index=ret.index,columns=ret.columns)
        factors[name]=wide.stack(future_stack=True).reindex(keys).to_numpy(dtype=np.float32)
    del residual,idio,log,momentum,ret,observed;gc.collect()
    labels=forward_total_returns(prices,20,missing_terminal_return="total_loss")
    factors=factors.merge(labels[["date","ticker","forward_return"]],on=["date","ticker"],how="left",validate="one_to_one")
    factors.to_parquet(output/"factors.parquet",index=False)
    results=[]
    for bucket,group in factors.loc[factors.date.between("2013-02-04","2025-12-31")&factors.liquidity_pass].groupby("cap_bucket",observed=True):
        for name,direction in (("RESID_MOM_12_1M",1),("MOM_12_1M",1),("IDIOVOL",-1)):
            ic=daily_rank_ic(group,name)*direction
            results.append({"cap_bucket":str(bucket),"factor":name,"mean_securities":float(group.groupby("date").size().mean()),**hac_summary(ic,20)})
    pd.DataFrame(results).to_csv(output/"size_stratified_ic.csv",index=False)
    write_run_metadata(output/"completion.json",stage="broad_company_cap_universe_and_matched_signal_research",
        inputs={"driver":Path(__file__),"raw_completion":raw/"completion.json","terminal_supplement":supplement,
            "terminal_supplement_audit":supplement.parent/"completion.json","universe_implementation":ROOT/"quant_platform/core/data/research_universe.py"},
        params={"top_n":2000,"diagnostic_top_n":3000,"price_min":5,"adv_dollar_min":2000000,"adv_sessions":20,
            "rank_before_liquidity":True,"one_class_per_company":True,"minimum_listing_calendar_days":60,
            "market_proxy":"same filtered broad top2000 proxy for every size bucket",
            "residual_momentum":"past504-day market/quartile-SMB OLS; 21-day beta updates; sum231 residuals/min126; lag21",
            "comparison_scope":"new CRSP cap-ranked controls; not Russell index membership or a date-only change to old S&P features"},
        extra={"complete":True,"imputed_missing_terminal_returns":imputed,"historical_review_not_untouched_holdout":True,"production_admitted":False})


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--raw",type=Path,default=ROOT/"data/wrds/batch4/broad_raw");p.add_argument("--output",type=Path,default=ROOT/"data/wrds/batch4/broad_research_20261024");p.add_argument("--wait-for-pull",action="store_true")
    a=p.parse_args()
    if a.wait_for_pull:
        print("Waiting for the bounded CRSP pull to finish",flush=True)
        while not (a.raw/"completion.json").exists():time.sleep(5)
    build(a.raw.resolve(),a.output.resolve())
