"""Explore all nine declared alternative proxies without selecting a winning sign."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import json
import sys
import warnings
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import pandas as pd
from scripts.replay_wrds_extensions import execute
from quant_platform.core.utils.provenance import write_run_metadata

DIRECTIONS={"IBES_REVISION_BREADTH":1,"IBES_EPS_REVISION":1,"IBES_DISPERSION":-1,
    "IBES_TARGET_IMPLIED_RETURN":1,"IBES_SUE_PRICE_SCALED":1,"OPTION_PUT25_SKEW":-1,
    "OPTION_CP50_IV_DIFF":1,"OPTION_IV_MINUS_RV":1,"OPTION_CONTRACTS_PER_MILLION_SHARES":1}


def one(book,stress,output):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore",category=FutureWarning,message="Downcasting object dtype arrays on .fillna.*")
        return execute(book,stress,output)


def main():
    study=ROOT/"data/wrds/batch4/alternative_features_20261024";root=study/"books";output=study/"replays"
    if output.exists():raise ValueError("Alternative replay directory already exists")
    root.mkdir(exist_ok=True);output.mkdir()
    panel=pd.read_parquet(study/"features.parquet")
    old=pd.read_parquet(ROOT/"data/wrds/batch4/prior_study_20261024/models/price_only/ridge_prior/h20/oof.parquet",columns=["date","ticker"])
    panel=old.merge(panel,on=["date","ticker"],how="left",validate="one_to_one")
    books=[]
    for name,direction in DIRECTIONS.items():
        rows=panel[["date","ticker"]].copy();rows["prediction"]=((2*panel.groupby("date")[name].rank(pct=True)-1)*direction).fillna(0.)
        path=root/f"{name}.parquet";rows.to_parquet(path,index=False)
        books.append({"name":name,"scores":str(path),"prices":str(ROOT/"data/wrds/p0/prices.parquet"),"rebalance":20,"ema":10})
    (root/"books.json").write_text(json.dumps(books,indent=2),encoding="utf-8")
    write_run_metadata(output/"experiment_plan.json",stage="exploratory_alternative_proxy_execution",
        inputs={"driver":Path(__file__),"features":study/"features.parquet","feature_governance":study/"run_metadata.json"},
        params={"directions":DIRECTIONS,"costs":[1,2],"policy":"20-session rebalance; EMA10; entry20/exit30; weight tolerance20",
            "selection":"all nine proxies, no favorable IC/sign selection",
            "volume_direction":"positive attention proxy is exploratory; unsigned volume has no unique return-direction prior"},
        extra={"historical_review_not_untouched_holdout":True,"source_point_in_time_proven":False,"production_admitted":False})
    results=[]
    with ProcessPoolExecutor(max_workers=4) as pool:
        jobs={pool.submit(one,b,s,output):(b["name"],s) for b in books for s in (1,2)}
        for future in as_completed(jobs):
            result=future.result();results.append(result)
            pd.DataFrame([{k:v for k,v in r.items() if k!="policy"} for r in results]).to_csv(output/"performance.csv",index=False)
            print(f"alternative completed {len(results)}/{len(jobs)} {jobs[future]} net={result['net_ann_return']:.3%}",flush=True)
    (output/"completion.json").write_text(json.dumps({"completed":len(results),"expected":18,"failures":[],"smoke":False}),encoding="utf-8")


if __name__=="__main__":main()
