"""Resolve IBES semantics from vendor catalogue descriptions, not filename guesses."""
from pathlib import Path
import json
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import pandas as pd
from scripts.acquire_wrds_extensions import connection,frame
from scripts.acquire_wrds_alternative import existing,save_checkpoint
from quant_platform.core.data.alternative_research import require_unadjusted_original
from quant_platform.core.utils.provenance import write_run_metadata


def main():
    raw=ROOT/"data/wrds/batch4/alternative_raw";output=raw/"ibes_unadjusted_repair";output.mkdir(exist_ok=True)
    tickers=sorted(pd.read_parquet(raw/"ibes_links.parquet").ticker.unique().tolist())
    c=connection()
    try:
        descriptions=frame(c,"SELECT c.relname,obj_description(c.oid,'pg_class') AS description FROM pg_class c JOIN pg_namespace n ON n.oid=c.relnamespace WHERE n.nspname='ibes' AND c.relname IN ('actu_epsus','ptgsumu')")
        for row in descriptions.itertuples():require_unadjusted_original(row.description)
        if len(descriptions)!=2:raise ValueError("Unable to verify both unadjusted IBES roles")
        for year in range(2008,2026):
            for stem,table,date_expr,columns in (
                ("eps_actual","ibes.actu_epsus","GREATEST(anndats,actdats)","ticker,pends,pdicity,anndats,anntims,actdats,acttims,value,curr_act"),
                ("target_summary","ibes.ptgsumu","statpers","ticker,statpers,numest,meanptg,stdev,curr")):
                path=output/f"{stem}_{year}.parquet"
                if existing(path):continue
                query=f"SELECT {columns} FROM {table} WHERE {date_expr}>=%s AND {date_expr}<%s AND ticker=ANY(%s)"
                data=frame(c,query,(f"{year}-01-01",f"{year+1}-01-01",tickers))
                save_checkpoint(data,path,{"table":table,"year":year,"vendor_description":descriptions.loc[descriptions.relname.eq(table.split('.')[-1]),"description"].iloc[0]})
            print(f"verified unadjusted IBES repair {year}",flush=True)
        write_run_metadata(output/"completion.json",stage="ibes_original_unadjusted_role_repair",
            inputs={"driver":Path(__file__),"old_raw_completion":raw/"completion.json"},
            params={"roles":descriptions.to_dict("records"),"excluded_old_roles":["ibes.nact_epsus","ibes.nptgsumu"]},
            extra={"complete":True,"not_a_signal_sign_change":True})
        (raw/"source_role_correction.json").write_text(json.dumps({"original_eps_actual_and_target_roles_not_admitted":True,
            "reason":"n prefix denotes Normalized, not Unadjusted; original completion's EPS basis description was incorrect",
            "authoritative_replacement":"ibes_unadjusted_repair/completion.json"},indent=2),encoding="utf-8")
    finally:c.close()


if __name__=="__main__":main()
