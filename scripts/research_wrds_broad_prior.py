"""Apply the same fixed three-factor prior to the expanded company-cap universe."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import json
from pathlib import Path
import sys
import warnings
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from quant_platform.core.alpha_models.prior_models import PriorRidge,full_date_sample
from quant_platform.core.alpha_models.nonlinear_alpha import _daily_standardize,_daily_zscore,daily_rank_ic
from quant_platform.core.utils.provenance import write_run_metadata,fingerprint
from scripts.research_wrds_priors import inner_masks
from scripts.replay_wrds_extensions import execute


def replay_one(book,stress,output):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore",category=FutureWarning,message="Downcasting object dtype arrays on .fillna.*")
        return execute(book,stress,output)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--study",type=Path,default=ROOT/"data/wrds/batch4/broad_research_20261024")
    a=parser.parse_args();study=a.study.resolve();output=study/"prior_model"
    if output.exists():raise ValueError("Choose a fresh broad prior study")
    output.mkdir()
    reference=json.loads((ROOT/"data/wrds/batch4/prior_study_20261024/plan.json").read_text())
    panel=pd.read_parquet(study/"factors.parquet")
    panel=panel.loc[panel.pass_all_filters&panel.date.ge("2010-01-01")].copy().reset_index(drop=True)
    names=["RESID_MOM_12_1M","MOM_12_1M","IDIOVOL"]
    for name in names:panel[name]=(2*panel.groupby("date")[name].rank(pct=True)-1).fillna(0).astype("float32")
    dates=pd.DatetimeIndex(sorted(panel.date.unique()));end=pd.Series(dates,index=dates).shift(-20)
    panel["label_end_20"]=panel.date.map(end);panel["label_target_20"]=_daily_standardize(panel.forward_return,panel.date,3).astype("float32")
    pri=np.array([1,1,-1])/3;parts=[];audits=[]
    with threadpool_limits(limits=2):
        for fold in reference["folds"]:
            valid_dates=pd.to_datetime(fold["validation_dates"])
            outer,inner,valid,valid_start=inner_masks(panel,dates,valid_dates,20,126)
            itr=panel.loc[inner];inner_ids=itr.index[full_date_sample(itr,stride=5)]
            choices=[]
            for penalty in (.01,1,100):
                est=PriorRidge(penalty).fit(panel.loc[inner_ids,names],panel.loc[inner_ids,"label_target_20"],pri)
                check=panel.loc[valid,["date","forward_return"]].copy();check["score"]=est.predict(panel.loc[valid,names])
                choices.append(float(daily_rank_ic(check,"score").mean()))
            chosen=(.01,1,100)[int(np.nanargmax(choices))]
            otr=panel.loc[outer];outer_ids=otr.index[full_date_sample(otr,stride=5)]
            est=PriorRidge(chosen).fit(panel.loc[outer_ids,names],panel.loc[outer_ids,"label_target_20"],pri)
            predict=panel.date.isin(valid_dates);local=panel.loc[predict,["date","ticker","forward_return"]].copy()
            pred=est.predict(panel.loc[predict,names]);local["prediction"]=_daily_zscore(pd.Series(pred,index=local.index),local.date).astype("float32")
            local["fold"]=fold["fold"];parts.append(local)
            max_outer=panel.loc[outer_ids,"label_end_20"].max();max_inner=panel.loc[inner_ids,"label_end_20"].max()
            assert max_outer<valid_dates[0] and max_inner<valid_start
            audits.append({"fold":fold["fold"],"normalized_penalty":chosen,"inner_candidates":choices,"coefficients":est.coef_.tolist(),
                "validation_start":str(valid_dates[0]),"max_training_label_end":str(max_outer),"inner_validation_start":str(valid_start),"max_inner_label_end":str(max_inner)})
            print(f"broad prior fold {fold['fold']}: {len(outer_ids):,} training rows, penalty={chosen}",flush=True)
    oof=pd.concat(parts,ignore_index=True);path=output/"oof.parquet";oof.to_parquet(path,index=False)
    (output/"fold_audit.json").write_text(json.dumps(audits,indent=2),encoding="utf-8")
    write_run_metadata(output/"training_metadata.json",stage="broad_fixed_three_factor_prior_ridge",
        inputs={"driver":Path(__file__),"factors":study/"factors.parquet","reference_plan":ROOT/"data/wrds/batch4/prior_study_20261024/plan.json"},
        params={"features":names,"prior":[1/3,1/3,-1/3],"penalties":[.01,1,100],"date_stride":5,"outer_raw_rank_ic":float(daily_rank_ic(oof,"prediction").mean()),
                "protocol":"same fixed economic prior as the S&P test; expanded pool is a reviewed-history follow-up, not a new holdout"},
        extra={"historical_review_not_untouched_holdout":True,"production_admitted":False})
    book={"name":"top2000_ridge_prior","scores":str(path),"prices":str(study/"prices.parquet"),"rebalance":20,"ema":10}
    result=[];dest=output/"replays";dest.mkdir()
    with ProcessPoolExecutor(max_workers=2) as pool:
        for future in as_completed([pool.submit(replay_one,book,s,dest) for s in (1,2)]):
            result.append(future.result());pd.DataFrame([{k:v for k,v in r.items() if k!="policy"} for r in result]).to_csv(dest/"performance.csv",index=False)
    (dest/"completion.json").write_text(json.dumps({"completed":len(result),"expected":2,"failures":[],"smoke":False}),encoding="utf-8")


if __name__=="__main__":main()
