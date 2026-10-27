"""Nested, reviewed-history tests of small priors, long horizons and robust trees."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import json
from pathlib import Path
import shutil
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
import yaml
import lightgbm as lgb
from threadpoolctl import threadpool_limits
from quant_platform.core.alpha_models.nonlinear_alpha import WalkForwardConfig,expanding_date_folds,daily_rank_ic,_daily_standardize,_daily_zscore
from quant_platform.core.alpha_models.prior_models import PriorRidge,full_date_sample,prior_coefficients,training_ic_selection
from quant_platform.core.data.research_contract import FUNDAMENTAL_FEATURES
from quant_platform.core.data.returns import forward_total_returns
from quant_platform.core.evaluation.factor_diagnostics import hac_summary
from quant_platform.core.utils.provenance import fingerprint,write_run_metadata


def prepare(output,config):
    if output.exists():raise ValueError("Choose a new, independent model study directory")
    cfg=yaml.safe_load(config.read_text(encoding="utf-8"));settings=cfg["model"]
    prior_run=ROOT/cfg["diagnostics"]["old_run"]
    source=pd.read_parquet(ROOT/"data/wrds/batch2/factors.parquet")
    old=pd.read_parquet(prior_run/"learning_panel.parquet",columns=["date","ticker"])
    # The original inference universe is retained, including unavailable labels.
    panel=old.merge(source,on=["date","ticker"],how="left",validate="one_to_one")
    catalog=pd.read_csv(ROOT/"data/wrds/batch2/factor_catalog.csv")
    candidates=catalog.factor.tolist()
    directions={r.factor:int(r.direction) for r in catalog.itertuples()}
    for name in candidates:
        panel[name]=(2*panel.groupby("date")[name].rank(pct=True)-1).fillna(0).astype("float32")
    prices=pd.read_parquet(ROOT/"data/wrds/p0/prices.parquet",columns=["date","ticker","close","total_return","is_terminal"])
    ic={};output.mkdir(parents=True)
    for horizon in (1,20,60,120):
        labels=forward_total_returns(prices,horizon,missing_terminal_return="total_loss")
        labels=labels[["date","ticker","forward_return","label_end"]]
        local=old.merge(labels,on=["date","ticker"],how="left",validate="one_to_one")
        panel=panel.merge(local.rename(columns={"forward_return":f"label_return_{horizon}","label_end":f"label_end_{horizon}"}),
            on=["date","ticker"],how="left",validate="one_to_one")
        panel[f"label_target_{horizon}"]=_daily_standardize(panel[f"label_return_{horizon}"],panel.date,3).astype("float32")
        cols=candidates if horizon in (1,20) else [f for f in candidates if f in FUNDAMENTAL_FEATURES]
        daily={}
        for feature in cols:
            local_feature=panel[["date",feature,f"label_return_{horizon}"]].rename(columns={f"label_return_{horizon}":"forward_return"})
            daily[feature]=daily_rank_ic(local_feature,feature)
        ic[horizon]=pd.DataFrame(daily)
        ic[horizon].to_parquet(output/f"training_ic_h{horizon}.parquet")
        print(f"prepared horizon {horizon}",flush=True)
    dates=pd.DatetimeIndex(sorted(panel.date.unique()))
    reference=WalkForwardConfig(horizon=120,min_train_days=settings["min_train_days"],validation_days=settings["validation_days"],oos_start=settings["oos_start"])
    folds=[]
    for fold in expanding_date_folds(dates,reference):
        folds.append({"fold":fold["fold"],"validation_dates":[str(x) for x in fold["validation_dates"]]})
    panel.to_parquet(output/"panel.parquet",index=False)
    schemas={"price_only":[f for f in candidates if f not in FUNDAMENTAL_FEATURES],"price_plus_financial":candidates,
             "financial_only":[f for f in candidates if f in FUNDAMENTAL_FEATURES]}
    source_hashes={}
    files=[Path(__file__),ROOT/"quant_platform/core/alpha_models/prior_models.py",ROOT/"quant_platform/core/evaluation/ml_diagnostics.py"]
    for file in files:
        relative=file.relative_to(ROOT);target=output/"source_snapshot"/relative;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(file,target);source_hashes[str(relative)]=fingerprint(file)["sha256"]
    plan={"config":cfg,"candidates":candidates,"directions":directions,"schemas":schemas,"folds":folds,
        "source_sha256":source_hashes,"common_oos_start":folds[0]["validation_dates"][0],
        "max_horizon":120,"sampling":"complete eligible cross sections every fifth training session; overlapping labels remain"}
    (output/"plan.json").write_text(json.dumps(plan,indent=2),encoding="utf-8")
    write_run_metadata(output/"prepare_metadata.json",stage="prior_and_long_horizon_study_preparation",
        inputs={"config":config,"factors":ROOT/"data/wrds/batch2/factors.parquet","old_keys":prior_run/"learning_panel.parquet","prices":ROOT/"data/wrds/p0/prices.parquet"},
        params=plan,extra={"historical_review_not_untouched_holdout":True,"fundamentals_is_point_in_time":False,"production_admitted":False})


def inner_masks(panel,dates,validation,horizon,inner_days):
    first=pd.Timestamp(validation[0]);start=int(dates.searchsorted(first))
    train_dates=dates[:start-horizon]
    valid_start=train_dates[-inner_days]
    matured=panel[f"label_end_{horizon}"].lt(first)&panel[f"label_target_{horizon}"].notna()
    outer=panel.date.isin(train_dates)&matured
    inner=outer&panel.date.lt(valid_start)&panel[f"label_end_{horizon}"].lt(valid_start)
    valid=outer&panel.date.ge(valid_start)
    return outer,inner,valid,valid_start


def fit_book(output,block,model,horizon):
    output=Path(output);plan=json.loads((output/"plan.json").read_text());settings=plan["config"]["model"]
    for file,wanted in plan["source_sha256"].items():
        if fingerprint(ROOT/file)["sha256"]!=wanted:raise ValueError(f"Changed frozen study source: {file}")
    panel=pd.read_parquet(output/"panel.parquet");dates=pd.DatetimeIndex(sorted(panel.date.unique()))
    ics=pd.read_parquet(output/f"training_ic_h{horizon}.parquet");ic1=pd.read_parquet(output/"training_ic_h1.parquet")
    candidates=plan["schemas"][block]
    if block=="financial_only":
        prior={f:plan["directions"][f] for f in candidates}
    else:prior=plan["config"]["prior"][block]
    folder=output/f"models/{block}/{model}/h{horizon}";folder.mkdir(parents=True,exist_ok=True)
    parts=[];trial_rows=[];audits=[]
    for fold in plan["folds"]:
        f=fold["fold"];saved=folder/f"fold_{f}.parquet";auditfile=folder/f"fold_{f}.json"
        if saved.exists() and auditfile.exists():
            audit=json.loads(auditfile.read_text())
            if fingerprint(saved)["sha256"]!=audit["output_sha256"]:raise ValueError("Corrupt prior-model checkpoint")
            parts.append(pd.read_parquet(saved));audits.append(audit);trial_rows+=audit["trials"];continue
        validation=pd.to_datetime(fold["validation_dates"]);outer,inner,valid,valid_start=inner_masks(panel,dates,validation,horizon,settings["inner_days"])
        ycol=f"label_target_{horizon}"
        feature_audits=[]

        def feature_set(mask,cutoff,phase):
            train=panel.loc[mask,candidates]
            duplicate={}
            for a,b in settings["duplicates"].items():
                if a in candidates and b in candidates:
                    corr=float(train[a].corr(train[b]))
                    if np.isfinite(corr) and abs(corr)>=.995:duplicate[a]=b
            chosen,audit=training_ic_selection(ics[candidates],sorted(panel.loc[mask,"date"].unique()),
                cutoff=cutoff,horizon=horizon,prior_features=prior,directions=plan["directions"],
                threshold=settings["aligned_train_ic_t_threshold"],duplicates=duplicate,one_day_ic=ic1,
                short_horizon_candidates=settings["short_horizon_candidates"])
            if model=="lightgbm":
                chosen=[c for c in candidates if c not in duplicate]
                audit["selected"]=audit.feature.isin(chosen)
                audit["reason"]=np.where(audit.selected,"full_dictionary_for_nonlinear_test",audit.reason)
            feature_audits.append(audit.assign(fold=f,phase=phase))
            return chosen

        inner_features=feature_set(inner,valid_start,"inner")
        sample_inner=panel.loc[inner]
        inner_ids=sample_inner.index[full_date_sample(sample_inner,stride=settings["date_stride"])].to_numpy()
        valid_ids=np.flatnonzero(valid)
        assert panel.loc[inner,f"label_end_{horizon}"].max()<valid_start
        choices=[]
        with threadpool_limits(limits=2):
            x=panel.loc[inner_ids,inner_features].to_numpy(dtype=np.float32);y=panel.loc[inner_ids,ycol].to_numpy(dtype=np.float32)
            xv=panel.loc[valid_ids,inner_features].to_numpy(dtype=np.float32)
            if model=="lightgbm":
                params={k:v for k,v in settings["lightgbm"].items() if k not in ("early_stopping_rounds","seed")}
                est=lgb.LGBMRegressor(**params,random_state=settings["lightgbm"]["seed"],n_jobs=2,verbosity=-1,
                    deterministic=True,force_col_wise=True)
                est.fit(x,y,eval_set=[(xv,panel.loc[valid_ids,ycol].to_numpy(dtype=np.float32))],eval_metric="l1",
                    callbacks=[lgb.early_stopping(settings["lightgbm"]["early_stopping_rounds"],verbose=False)])
                pred=est.predict(xv);chosen=int(est.best_iteration_ or params["n_estimators"])
                configs=[{"best_iteration":chosen}]
            else:
                beta=prior_coefficients(inner_features,prior) if model=="ridge_prior" else np.zeros(len(inner_features))
                preds=[];configs=[]
                for penalty in settings["normalized_prior_penalties"]:
                    est=PriorRidge(penalty).fit(x,y,beta);pred=est.predict(xv)
                    values=daily_rank_ic(pd.DataFrame({"date":panel.loc[valid_ids,"date"].to_numpy(),"forward_return":panel.loc[valid_ids,f"label_return_{horizon}"].to_numpy(),"prediction":pred}),"prediction")
                    choices.append(float(values.mean()));preds.append(pred);configs.append({"normalized_penalty":penalty,"raw_return_rank_ic":choices[-1]})
                winner=int(np.nanargmax(choices));pred=preds[winner];chosen=settings["normalized_prior_penalties"][winner]
            inner_ic=daily_rank_ic(pd.DataFrame({"date":panel.loc[valid_ids,"date"].to_numpy(),"forward_return":panel.loc[valid_ids,f"label_return_{horizon}"].to_numpy(),"prediction":pred}),"prediction")
            ic_stats=hac_summary(inner_ic,horizon)
            adjusted_icir=ic_stats["hac_t"]/np.sqrt(ic_stats["n_days"]) if np.isfinite(ic_stats["hac_t"]) else 0.
            features=feature_set(outer,validation[0],"outer")
            sample_outer=panel.loc[outer]
            outer_ids=sample_outer.index[full_date_sample(sample_outer,stride=settings["date_stride"])].to_numpy()
            predict_ids=np.flatnonzero(panel.date.isin(validation))
            x=panel.loc[outer_ids,features].to_numpy(dtype=np.float32);y=panel.loc[outer_ids,ycol].to_numpy(dtype=np.float32)
            xt=panel.loc[predict_ids,features].to_numpy(dtype=np.float32)
            if model=="lightgbm":
                params["n_estimators"]=chosen
                est=lgb.LGBMRegressor(**params,random_state=settings["lightgbm"]["seed"],n_jobs=2,verbosity=-1,deterministic=True,force_col_wise=True).fit(x,y)
            else:
                beta=prior_coefficients(features,prior) if model=="ridge_prior" else np.zeros(len(features))
                est=PriorRidge(chosen).fit(x,y,beta)
            prediction=est.predict(xt)
        local=panel.loc[predict_ids,["date","ticker",f"label_return_{horizon}"]].rename(columns={f"label_return_{horizon}":"forward_return"}).copy()
        local["prediction"]=_daily_zscore(pd.Series(prediction,index=local.index),local.date).astype("float32")
        local["fold"]=f;local["inner_hac_icir"]=adjusted_icir
        local.to_parquet(saved,index=False)
        audit={"fold":f,"features":features,"inner_features":inner_features,"selected":chosen,"trials":configs,
            "outer_rows":len(outer_ids),"inner_rows":len(inner_ids),"validation_start":str(validation[0]),
            "max_training_label_end":str(panel.loc[outer_ids,f"label_end_{horizon}"].max()),
            "inner_validation_start":str(valid_start),"max_inner_training_label_end":str(panel.loc[inner_ids,f"label_end_{horizon}"].max()),
            "inner_hac_icir":adjusted_icir,"inner_ic_mean":ic_stats["mean_rank_ic"],"output_sha256":fingerprint(saved)["sha256"]}
        auditfile.write_text(json.dumps(audit,indent=2),encoding="utf-8")
        pd.concat(feature_audits,ignore_index=True).to_csv(folder/f"feature_selection_{f}.csv",index=False)
        if model!="lightgbm":
            pd.DataFrame({"feature":features,"coefficient":est.coef_}).to_csv(folder/f"coefficients_{f}.csv",index=False)
        parts.append(local);audits.append(audit);trial_rows+=configs
        print(f"{block}/{model}/h{horizon} fold {f}: {len(features)} features, {len(outer_ids):,} rows, selected={chosen}",flush=True)
    oof=pd.concat(parts,ignore_index=True);oof.to_parquet(folder/"oof.parquet",index=False)
    (folder/"completion.json").write_text(json.dumps({"folds":audits,"trials":len(trial_rows)},indent=2),encoding="utf-8")
    return {"block":block,"model":model,"horizon":horizon,"prediction_rows":len(oof),
        "raw_rank_ic":float(daily_rank_ic(oof,"prediction").mean()),"oof_sha256":fingerprint(folder/"oof.parquet")["sha256"]}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--output",type=Path,default=ROOT/"data/wrds/batch4/prior_study_20261024")
    p.add_argument("--config",type=Path,default=ROOT/"config/wrds_research_extensions.yaml");p.add_argument("--phase",choices=["prepare","train","all"],default="all");p.add_argument("--jobs",type=int,default=4)
    a=p.parse_args();output=a.output.resolve()
    if a.phase in ("prepare","all"):prepare(output,a.config.resolve())
    if a.phase in ("train","all"):
        # Eight fixed learned books: six 20d model/info choices plus 60/120d financial priors.
        tasks=[(block,model,20) for block in ("price_only","price_plus_financial") for model in ("ridge_zero","ridge_prior","lightgbm")]
        tasks += [("financial_only","ridge_prior",h) for h in (60,120)]
        results=[]
        with ProcessPoolExecutor(max_workers=a.jobs) as pool:
            futures=[pool.submit(fit_book,output,*task) for task in tasks]
            for future in as_completed(futures):
                results.append(future.result());pd.DataFrame(results).to_csv(output/"model_summary.csv",index=False)
        (output/"training_completion.json").write_text(json.dumps({"cohorts":results,"historical_review_not_untouched_holdout":True},indent=2),encoding="utf-8")


if __name__=="__main__":main()
