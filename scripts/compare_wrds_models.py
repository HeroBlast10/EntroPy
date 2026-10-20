"""Frozen nested OOF experiment; a 20-session task and 5-session robustness."""
from __future__ import annotations
import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor,as_completed
from dataclasses import replace
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

import numpy as np
import pandas as pd
import yaml
from loguru import logger
from quant_platform.core.alpha_models.nonlinear_alpha import (build_learning_panel,select_usable_features,
    filter_tradable_universe,WalkForwardConfig,expanding_date_folds,daily_rank_ic,block_bootstrap_ic_delta)
from quant_platform.core.alpha_models.fair_comparison import nested_oof,issuer_capped_weights
from quant_platform.core.data.research_contract import require_wrds_identity,FUNDAMENTAL_FEATURES
from quant_platform.core.evaluation.factor_diagnostics import hac_summary
from quant_platform.core.execution.backtest.vectorized_daily import simulate_execution
from quant_platform.core.execution.backtest.pnl import compute_daily_returns,performance_summary
from quant_platform.core.execution.cost_models.us_equity import CostModel
from quant_platform.core.portfolio.rebalance import carry_forward_weights
from quant_platform.core.utils.provenance import fingerprint,write_run_metadata,git_dirty,WRDS_CODE_PATHS


def execute_book(folder,block,signal,price_path,execution,horizon,policy,stress):
    logger.remove()
    folder=Path(folder)
    oof=pd.read_parquet(folder/"oof.parquet")
    rows=oof[oof.block.eq(block)].copy()
    p=pd.read_parquet(price_path,columns=["date","ticker","permno","permco","adj_close","close","volume","is_tradable","total_return","is_terminal"])
    calendar=pd.DatetimeIndex(sorted(rows.date.unique()),name="date")
    # The schedule is chosen without label availability and never drops an
    # otherwise predictable security because its future quote is missing.
    targets=[]
    context=p[["date","ticker","permno","permco"]]
    decisions=rows[rows.date.isin(calendar[::horizon])].merge(context,on=["date","ticker"],validate="one_to_one")
    for date,day in decisions.groupby("date"):
        weights=issuer_capped_weights(day,signal,min_names=execution["min_names"],
            stock_cap=execution["max_stock_weight"],issuer_cap=execution["max_issuer_gross_weight"])
        target=day[["date","ticker"]].assign(weight=weights)
        targets.append(target)
    weights=pd.concat(targets,ignore_index=True)
    daily=carry_forward_weights(weights,calendar)
    source_dates=pd.DatetimeIndex(sorted(p.date.unique()))
    first=max(0,int(source_dates.searchsorted(calendar.min()))-63)
    p=p[p.date.between(source_dates[first],calendar.max())]
    model=CostModel(max_participation_rate=execution["max_participation_rate"])
    if stress!=1:
        fields=("commission_per_share","commission_pct","commission_min","slippage_bps","impact_coeff",
                "borrow_rate_annual","sec_fee_rate","finra_taf_per_share")
        model=replace(model,**{c:getattr(model,c)*stress for c in fields},finra_taf_max=model.finra_taf_max*stress)
    trades,ledger=simulate_execution(daily,p,model,initial_capital=execution["initial_capital"],
        missing_terminal_return=policy,return_daily_ledger=True)
    pnl=compute_daily_returns(daily,p,trades,model,initial_capital=execution["initial_capital"],execution_ledger=ledger)
    performance=performance_summary(pnl)
    maximum=float(trades.executed_participation.max()) if len(trades) else 0.
    assert maximum<=execution["max_participation_rate"]+1e-12
    dest=folder/"backtests"/block/signal/f"cost_{stress:g}x"
    dest.mkdir(parents=True,exist_ok=True)
    weights.to_parquet(dest/"rebalance_weights.parquet",index=False)
    trades.to_parquet(dest/"trades.parquet",index=False)
    pnl.to_parquet(dest/"daily_pnl.parquet")
    nonrebalance=~trades.is_rebalance_order.eq(True)
    cost=float(trades.total_cost.sum())
    report={"horizon":horizon,"block":block,"signal":signal,"cost_multiplier":stress,
            "max_executed_participation":maximum,"executed_trades":int(trades.executed_notional.gt(0).sum()),
            "nonrebalance_cost_fraction":float(trades.loc[nonrebalance,"total_cost"].sum()/cost) if cost else 0,
            **performance}
    pd.DataFrame([report]).to_csv(dest/"performance.csv",index=False)
    write_run_metadata(dest/"run_metadata.json",stage="wrds_fair_model_execution",code_paths=WRDS_CODE_PATHS,
        inputs={"oof":folder/"oof.parquet","prices":price_path},params={**execution,"horizon":horizon,
            "terminal_policy":policy,"stress":stress,"holding":"fixed session rebalance, terminal cash settlement",
            "last_period":"marked through tape cutoff; final stub may be shorter"},extra={"performance":report})
    return report


def score_oof(oof,horizon):
    summaries,daily=[],{}
    signals=["equal_weight","ridge","lightgbm","ridge_placebo","MOM_12_1M","GROSS_PROFITABILITY"]
    for block,rows in oof.groupby("block"):
        for signal in signals:
            if signal not in rows: continue
            ic=daily_rank_ic(rows,signal).reindex(pd.DatetimeIndex(sorted(rows.date.unique())))
            daily[(block,signal)]=ic
            summaries.append({"horizon":horizon,"block":block,"signal":signal,**hac_summary(ic,horizon),
                              "prediction_rows":int(rows[signal].notna().sum()),"labeled_rows":int((rows[signal].notna()&rows.forward_return.notna()).sum())})
    paired=[]
    for block in oof.block.unique():
        for challenger,baseline in (("ridge","equal_weight"),("lightgbm","equal_weight"),("lightgbm","ridge")):
            paired.append({"horizon":horizon,"block":block,"challenger":challenger,"baseline":baseline,
                **block_bootstrap_ic_delta(daily[(block,challenger)],daily[(block,baseline)],block_size=max(20,horizon),n_boot=1000)})
    for model in ("ridge","lightgbm"):
        paired.append({"horizon":horizon,"block":"price_plus_financial_vs_price_only","challenger":model,"baseline":model,
            **block_bootstrap_ic_delta(daily[("price_plus_financial",model)],daily[("price_only",model)],block_size=max(20,horizon),n_boot=1000)})
    return pd.DataFrame(summaries),pd.DataFrame(paired),pd.concat({f"{k[0]}/{k[1]}":v for k,v in daily.items()},axis=1)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,default=ROOT/"config/wrds_batch2.yaml")
    parser.add_argument("--allow-restated-fundamentals",action="store_true")
    parser.add_argument("--missing-terminal-return",choices=["total_loss","zero_return"],required=True)
    parser.add_argument("--smoke",action="store_true")
    parser.add_argument("--backtests-only",action="store_true")
    parser.add_argument("--skip-backtests",action="store_true")
    parser.add_argument("--sensitivity-from",type=Path)
    parser.add_argument("--jobs",type=int,default=2)
    args=parser.parse_args()
    logger.remove()
    cfg=yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if git_dirty(paths=WRDS_CODE_PATHS) is not False:
        raise ValueError("Freeze WRDS source before model comparison")
    if not args.allow_restated_fundamentals:
        raise ValueError("Financial ablation requires --allow-restated-fundamentals; membership/keys remain mandatory")
    settings=cfg["research"].copy()
    if args.smoke:
        cfg["date_range"]={"start":"2018-01-01","end":"2020-12-31"}
        settings.update(min_train_days=252,validation_days=42,n_splits=1,inner_days=63,max_train_rows=30000)
    unique_configs=2*sum(map(len,settings["model_configs"].values()))+4
    if unique_configs>settings["max_candidate_configurations"]:
        raise ValueError("Experiment exceeds the predeclared parameter-configuration budget")
    dest=ROOT/cfg["paths"]["output"]/args.missing_terminal_return/("smoke" if args.smoke else "full")
    dest.mkdir(parents=True,exist_ok=True)
    inputs={k:ROOT/cfg["paths"][k] for k in ("factors","catalog","prices","universe")}
    inputs["config"]=args.config
    factors=pd.read_parquet(inputs["factors"])
    universe=pd.read_parquet(inputs["universe"])
    prices=pd.read_parquet(inputs["prices"])
    require_wrds_identity(universe,factors,prices)
    start,end=cfg["date_range"]["start"],cfg["date_range"]["end"]
    if pd.Timestamp(end)>prices.date.max(): raise ValueError("End exceeds CRSP tape")
    factors=factors[factors.date.between(start,end)].copy()
    prices_for_labels=prices[prices.date.between(start,end)].copy()
    catalog=pd.read_csv(inputs["catalog"]).set_index("factor")
    directions=catalog.direction.to_dict()
    policy=args.missing_terminal_return
    choices=None
    schemas=None
    total_trials=[]
    all_performance=[]
    primary=settings["primary_horizon"]
    horizons=[primary,settings["robustness_horizon"]]
    for horizon in horizons:
        folder=dest/f"h{horizon}"; folder.mkdir(exist_ok=True)
        if not args.backtests_only:
            if schemas is None:
                # Lock quality gates before the FIRST inner validation date.
                dates=pd.DatetimeIndex(sorted(factors.date.unique()))
                wf=WalkForwardConfig(horizon=primary,min_train_days=settings["min_train_days"],
                    validation_days=settings["validation_days"],n_splits=settings["n_splits"])
                first=next(expanding_date_folds(dates,wf))
                cutoff=first["train_dates"][-settings["inner_days"]-primary]
                training=filter_tradable_universe(factors,universe)
                selected,diagnostics=select_usable_features(training[training.date.lt(cutoff)],
                    cfg["feature_whitelist"],min_coverage=settings["min_feature_coverage"],return_diagnostics=True)
                schemas={"price_only":[c for c in selected if c not in FUNDAMENTAL_FEATURES],"price_plus_financial":selected}
                settings["sampling_features"]=schemas["price_only"]
                if not schemas["price_only"] or not FUNDAMENTAL_FEATURES.intersection(selected):
                    raise ValueError("Both predeclared information ablations need usable features")
                diagnostics.to_csv(dest/"locked_feature_schema.csv",index=False)
                (dest/"feature_schema.json").write_text(json.dumps({"cutoff":str(cutoff.date()),"blocks":schemas},indent=2))
            panel,_=build_learning_panel(factors,prices_for_labels,universe,feature_cols=schemas["price_plus_financial"],
                horizon=horizon,min_feature_coverage=0,missing_terminal_return=policy)
            frozen=choices if horizon!=primary else None
            if args.sensitivity_from:
                frozen=json.loads((args.sensitivity_from/f"h{primary}"/"chosen_parameters.json").read_text())
            oof,trials,selected,importance,folds=nested_oof(panel,schemas,directions,settings,horizon=horizon,frozen_choices=frozen)
            if horizon==primary: choices=selected
            oof.to_parquet(folder/"oof.parquet",index=False)
            importance.to_csv(folder/"feature_importance.csv",index=False)
            folds.to_csv(folder/"fold_audit.csv",index=False)
            (folder/"chosen_parameters.json").write_text(json.dumps(selected,indent=2))
            (folder/"trial_registry.json").write_text(json.dumps(trials,indent=2))
            total_trials.extend(trials)
            summary,paired,ic=score_oof(oof,horizon)
            summary.to_csv(folder/"ic_summary.csv",index=False)
            paired.to_csv(folder/"paired_ic_delta.csv",index=False)
            ic.to_parquet(folder/"daily_ic.parquet")
            write_run_metadata(folder/"run_metadata.json",stage="wrds_nested_oof",inputs=inputs,code_paths=WRDS_CODE_PATHS,
                params={"settings":settings,"feature_blocks":schemas,"horizon":horizon,"date_range":cfg["date_range"],
                        "terminal_policy":policy,"search_or_frozen":"inner selection" if frozen is None else "frozen primary parameters"},
                extra={"output":fingerprint(folder/"oof.parquet"),"primary_key":"permno","membership_is_point_in_time":True,
                       "fundamentals_is_point_in_time":False,"historical_review_not_untouched_holdout":True})
        else:
            old=json.loads((folder/"run_metadata.json").read_text())
            for record in old["inputs"].values():
                if fingerprint(record["path"])["sha256"]!=record["sha256"]: raise ValueError("Stale OOF inputs")
            if fingerprint(folder/"oof.parquet")["sha256"]!=old["output"]["sha256"]: raise ValueError("Stale OOF predictions")
            oof=pd.read_parquet(folder/"oof.parquet")
        if not args.skip_backtests:
            tasks=[]
            for block in ("price_only","price_plus_financial"):
                for signal in ("equal_weight","ridge","lightgbm","ridge_placebo"):
                    tasks += [(block,signal,1),(block,signal,cfg["execution"]["cost_stress_multiplier"])]
            tasks += [("price_plus_financial",signal,1) for signal in ("MOM_12_1M","GROSS_PROFITABILITY") if signal in oof]
            with ProcessPoolExecutor(max_workers=args.jobs) as pool:
                futures=[pool.submit(execute_book,str(folder),b,s,str(inputs["prices"]),cfg["execution"],horizon,policy,c) for b,s,c in tasks]
                for future in as_completed(futures):
                    result=future.result(); all_performance.append(result)
                    pd.DataFrame(all_performance).to_csv(dest/"execution_comparison.csv",index=False)
                    print(f"executed h={horizon} {result['block']}/{result['signal']} cost={result['cost_multiplier']}x",flush=True)
    if total_trials:
        (dest/"trial_registry.json").write_text(json.dumps(total_trials,indent=2))
    write_run_metadata(dest/"experiment_metadata.json",stage="wrds_batch2_comparison",inputs=inputs,code_paths=WRDS_CODE_PATHS,
        params={"candidate_configuration_budget":unique_configs,"seeds":settings["seeds"],"settings":settings,
                "terminal_policy":policy,"all_trials_file":"trial_registry.json","negative_control":"fixed per-date shuffled ridge; not eligible for selection"},
        extra={"smoke":args.smoke,"production_admitted":False,"fundamentals_is_point_in_time":False,
               "historical_review_not_untouched_holdout":True,"trade_capital":cfg["execution"]["initial_capital"]})


if __name__=="__main__":main()
