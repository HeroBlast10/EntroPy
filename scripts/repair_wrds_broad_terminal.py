"""Recover CIZ settlement rows whose terminal classification is no longer COM."""
from pathlib import Path
import sys
import json
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import pandas as pd
from scripts.acquire_wrds_extensions import connection,frame
from quant_platform.core.utils.provenance import write_run_metadata,fingerprint


def main():
    root=ROOT/"data/wrds/batch4/broad_raw/terminal_repair"
    root.mkdir(exist_ok=True)
    if (root/"completion.json").exists():raise ValueError("Terminal repair is already complete")
    columns={"date":"dlycaldt","permno":"permno","permco":"permco","siccd":"siccd",
        "price":"dlyprc","ret":"dlyret","volume":"dlyvol","shares_outstanding":"shrout",
        "market_cap":"dlycap","open":"dlyopen","high":"dlyhigh","low":"dlylow",
        "delisting_flag":"dlydelflg","securitytype":"securitytype","securitysubtype":"securitysubtype",
        "sharetype":"sharetype","usincflg":"usincflg","primaryexch":"primaryexch",
        "conditionaltype":"conditionaltype","tradingstatusflg":"tradingstatusflg","cumfacshr":"dlycumfacshr"}
    query="SELECT "+",".join(f"{value} AS {key}" for key,value in columns.items())+" FROM crsp.dsf_v2 WHERE dlycaldt >= %s AND dlycaldt < %s AND dlydelflg='Y' ORDER BY dlycaldt,permno"
    with connection() as db:
        data=frame(db,query,("2008-01-01","2026-01-01"))
    data["date"]=pd.to_datetime(data.date)
    strings={"date","permno","permco","siccd","delisting_flag","securitytype","securitysubtype","sharetype","usincflg","primaryexch","conditionaltype","tradingstatusflg"}
    for name in columns:
        if name not in strings:data[name]=pd.to_numeric(data[name],errors="coerce")
    for name in ("permno","permco"):data[name]=data[name].astype("int64")
    assert data.delisting_flag.eq("Y").all() and not data.duplicated(["date","permno"]).any()
    path=root/"terminal.parquet";data.to_parquet(path,index=False)
    write_run_metadata(root/"completion.json",stage="ciz_terminal_classification_repair",
        inputs={"driver":Path(__file__),"original_pull":root.parent/"completion.json"},
        params={"rows":len(data),"file":fingerprint(path),"terminal_security_types":data.securitytype.value_counts().to_dict(),
            "source":"crsp.dsf_v2; dlydelflg=Y; NO securitytype/subtype restriction",
            "reason":"CIZ terminal observations can have securitytype=N/A, securitysubtype=UNK; COM-only filtering loses settlements"},
        extra={"complete":True,"original_raw_files_preserved":True})
    (root.parent/"source_role_correction.json").write_text(json.dumps({"required_supplement":str(path),
        "original_common_only_tape_is_incomplete_for_terminal_settlements":True,
        "original_broad_research_not_admitted":True},indent=2),encoding="utf-8")
    print(f"recovered {len(data):,} terminal observations",flush=True)


if __name__=="__main__":main()
