"""Bounded IBES/OptionMetrics pulls linked to the historical S&P study IDs."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import pandas as pd
from scripts.acquire_wrds_extensions import connection,frame
from quant_platform.core.utils.provenance import fingerprint,write_run_metadata


def save_checkpoint(data,path,query_info):
    if path.exists():raise ValueError(f"Refusing to overwrite {path}")
    for col in data:
        if col in {"date","statpers","fpedats","pends","anndats","actdats","sdate","edate"}:
            data[col]=pd.to_datetime(data[col])
    data.to_parquet(path,index=False)
    path.with_suffix(".json").write_text(json.dumps({"rows":len(data),"sha256":fingerprint(path)["sha256"],**query_info},indent=2),encoding="utf-8")


def existing(path):
    if not path.exists():return False
    audit=path.with_suffix(".json")
    if not audit.exists() or json.loads(audit.read_text())["sha256"]!=fingerprint(path)["sha256"]:
        raise ValueError(f"Unverified alternative-data checkpoint {path}")
    return True


def pull(output,start=2008,end=2025):
    output.mkdir(parents=True,exist_ok=True)
    prices=ROOT/"data/wrds/p0/prices.parquet"
    ids=sorted(pd.read_parquet(prices,columns=["permno"]).permno.dropna().astype(int).unique().tolist())
    db=connection()
    try:
        for stem,table in (("ibes_links","wrdsapps_link_crsp_ibes.ibcrsphist"),("option_links","wrdsapps_link_crsp_optionm.opcrsphist")):
            path=output/f"{stem}.parquet"
            if not existing(path):save_checkpoint(frame(db,f"SELECT * FROM {table} WHERE permno=ANY(%s) AND score<=1",(ids,)),path,{"table":table,"max_link_score":1})
        ibes=pd.read_parquet(output/"ibes_links.parquet"); options=pd.read_parquet(output/"option_links.parquet")
        tickers=sorted(ibes.ticker.dropna().unique().tolist()); secids=sorted(options.secid.dropna().astype(int).unique().tolist())
        queries={
            "eps_summary":("ibes.statsumu_epsus", "SELECT ticker,statpers,fpedats,fpi,fiscalp,curcode,numest,numup,numdown,meanest,stdev FROM ibes.statsumu_epsus WHERE statpers>=%s AND statpers<%s AND ticker=ANY(%s) AND fpi IN ('1','6') AND measure='EPS'",tickers),
            "eps_actual":("ibes.nact_epsus", "SELECT ticker,pends,pdicity,anndats,anntims,actdats,acttims,value,curr_act FROM ibes.nact_epsus WHERE GREATEST(anndats,actdats)>=%s AND GREATEST(anndats,actdats)<%s AND ticker=ANY(%s) AND measure='EPS'",tickers),
            "target_summary":("ibes.nptgsumu", "SELECT ticker,statpers,numest,meanptg,stdev,curr FROM ibes.nptgsumu WHERE statpers>=%s AND statpers<%s AND ticker=ANY(%s)",tickers),
            "option_volume":("optionm.opvold", "SELECT secid,date,SUM(volume) AS option_contract_volume FROM optionm.opvold WHERE date>=%s AND date<%s AND secid=ANY(%s) AND cp_flag IN ('C','P') GROUP BY secid,date",secids),
            "corporate_actions":("crsp.dsf_v2", "SELECT permno,dlycaldt AS date,dlyfacprc FROM crsp.dsf_v2 WHERE dlycaldt>=%s AND dlycaldt<%s AND permno=ANY(%s) AND dlyfacprc<>0",ids)}
        for year in range(start,end+1):
            began=time.monotonic()
            for stem,(table,query,keys) in queries.items():
                path=output/f"{stem}_{year}.parquet"
                if existing(path):continue
                data=frame(db,query,(f"{year}-01-01",f"{year+1}-01-01",keys))
                save_checkpoint(data,path,{"table":table,"year":year})
            path=output/f"option_surface_{year}.parquet"
            if not existing(path):
                table=f"optionm.vsurfd{year}"
                query=f"""SELECT secid,date,
                    CASE WHEN count(*) FILTER(WHERE cp_flag='P' AND delta=-25)=1 THEN max(impl_volatility) FILTER(WHERE cp_flag='P' AND delta=-25) END AS put25,
                    CASE WHEN count(*) FILTER(WHERE cp_flag='P' AND delta=-50)=1 THEN max(impl_volatility) FILTER(WHERE cp_flag='P' AND delta=-50) END AS put50,
                    CASE WHEN count(*) FILTER(WHERE cp_flag='C' AND delta=50)=1 THEN max(impl_volatility) FILTER(WHERE cp_flag='C' AND delta=50) END AS call50,
                    CASE WHEN count(*) FILTER(WHERE cp_flag='C' AND delta=25)=1 THEN max(impl_volatility) FILTER(WHERE cp_flag='C' AND delta=25) END AS call25
                    FROM {table} WHERE date>=%s AND date<%s AND secid=ANY(%s) AND days=30 AND delta IN(-25,-50,25,50)
                    GROUP BY secid,date"""
                save_checkpoint(frame(db,query,(f"{year}-01-01",f"{year+1}-01-01",secids)),path,{"table":table,"year":year,"days":30,"duplicate_node_policy":"missing"})
            print(f"IBES/OptionMetrics {year}: complete, {time.monotonic()-began:.1f}s",flush=True)
        write_run_metadata(output/"completion.json",stage="linked_ibes_optionmetrics_raw_pull",
            inputs={"driver":Path(__file__),"study_ids":prices,"ibes_links":output/"ibes_links.parquet","option_links":output/"option_links.parquet"},
            params={"start":start,"end":end,"eps_basis":"unadjusted archived monthly consensus and unadjusted actuals",
                "availability":"max announcement/activation for actuals; next session after monthly snapshot for consensus",
                "option_surface":"30-day constant-maturity delta grid, no individual contract quote claim"},
            extra={"complete":True,"native_timestamps_do_not_certify_archived_vintages":True})
    finally:db.close()


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--output",type=Path,default=ROOT/"data/wrds/batch4/alternative_raw")
    p.add_argument("--start-year",type=int,default=2008);p.add_argument("--end-year",type=int,default=2025)
    a=p.parse_args();pull(a.output.resolve(),a.start_year,a.end_year)
