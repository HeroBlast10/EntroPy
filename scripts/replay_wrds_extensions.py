"""Matched after-cost books for fixed priors, robust models, and breadth controls."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
from dataclasses import replace,asdict
import json
from pathlib import Path
import shutil
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from loguru import logger
from quant_platform.core.portfolio.low_turnover import TurnoverPolicy,build_turnover_targets
from quant_platform.core.execution.backtest.vectorized_daily import simulate_execution
from quant_platform.core.execution.backtest.pnl import performance_summary
from quant_platform.core.execution.cost_models.us_equity import CostModel
from quant_platform.core.evaluation.turnover_costs import turnover_cost_diagnostics
from quant_platform.core.utils.provenance import fingerprint,write_run_metadata


def positive_icir_forecast(scores,weights):
    """Missing scores reset EMA state when every training weight is nonpositive."""
    scores=np.asarray(scores,dtype=float);weights=np.asarray(weights,dtype=float)
    if scores.ndim!=2 or weights.shape!=scores.shape:raise ValueError("Mismatched ensemble arrays")
    weights=np.where(np.isfinite(weights),np.maximum(weights,0),0)
    if (~np.isfinite(scores)&(weights>0)).any():raise ValueError("Nonfinite active ensemble score")
    denominator=weights.sum(axis=1)
    numerator=(np.where(weights>0,scores,0)*weights).sum(axis=1)
    return np.divide(numerator,denominator,out=np.full(len(scores),np.nan),where=denominator>0)


def prepare_model_books(study):
    plan=json.loads((study/"plan.json").read_text());cohorts=json.loads((study/"training_completion.json").read_text())["cohorts"]
    root=study/"books";root.mkdir(exist_ok=True);books=[]
    panel=pd.read_parquet(study/"panel.parquet")
    dates=pd.to_datetime([v for f in plan["folds"] for v in f["validation_dates"]])
    panel=panel.loc[panel.date.isin(dates)].copy()
    prices=ROOT/"data/wrds/p0/prices.parquet"
    for name,prior in [("residual_momentum",{"RESID_MOM_12_1M":1}),*[(f"equal_weight_{block}",prior) for block,prior in plan["config"]["prior"].items()]]:
        rows=panel[["date","ticker"]].copy();rows["prediction"]=sum(panel[f]*sign for f,sign in prior.items())/len(prior)
        path=root/f"{name}.parquet";rows.to_parquet(path,index=False)
        books.append({"name":name,"scores":str(path),"prices":str(prices),"rebalance":20,"ema":10})
    for row in cohorts:
        name=f"{row['block']}_{row['model']}_h{row['horizon']}"
        path=study/f"models/{row['block']}/{row['model']}/h{row['horizon']}/oof.parquet"
        books.append({"name":name,"scores":str(path),"prices":str(prices),"rebalance":row["horizon"],"ema":10})
    components=[]
    for block,h in (("price_only",20),("financial_only",60),("financial_only",120)):
        components.append(pd.read_parquet(study/f"models/{block}/ridge_prior/h{h}/oof.parquet",columns=["date","ticker","prediction","inner_hac_icir"]))
    ref=components[0][["date","ticker"]]
    for frame in components:assert ref.equals(frame[["date","ticker"]])
    weights=np.column_stack([p.inner_hac_icir.clip(lower=0).to_numpy() for p in components])
    scores=np.column_stack([p.prediction.to_numpy() for p in components])
    denominator=weights.sum(axis=1);forecast=positive_icir_forecast(scores,weights)
    ensemble=ref.copy();ensemble["prediction"]=forecast;ensemble.to_parquet(root/"training_icir_ensemble.parquet",index=False)
    pd.DataFrame(weights,columns=["price20","financial60","financial120"]).assign(date=ref.date.to_numpy(),
        fold_cash=denominator==0).groupby("date").first().to_csv(root/"ensemble_training_weights.csv")
    books.append({"name":"training_icir_ensemble","scores":str(root/"training_icir_ensemble.parquet"),"prices":str(prices),"rebalance":20,"ema":10})
    (root/"books.json").write_text(json.dumps(books,indent=2),encoding="utf-8")
    return books


def prepare_broad_books(study):
    if not (study/"completion.json").exists():raise ValueError("Broad feature build is incomplete")
    root=study/"books";root.mkdir(exist_ok=True)
    factors=pd.read_parquet(study/"factors.parquet");books=[]
    for top in (500,2000):
        rows=factors.loc[factors.date.between("2013-06-27","2025-12-31")&factors.liquidity_pass&factors.lagged_cap_rank.le(top)].copy()
        rank=rows.groupby("date")[["RESID_MOM_12_1M","MOM_12_1M","IDIOVOL"]].rank(pct=True)*2-1
        rank=rank.fillna(0)
        for name,values in (("residual_momentum",rank.RESID_MOM_12_1M),("equal_weight_price_only",(rank.RESID_MOM_12_1M+rank.MOM_12_1M-rank.IDIOVOL)/3)):
            path=root/f"top{top}_{name}.parquet";score=rows[["date","ticker"]].copy();score["prediction"]=values
            score.to_parquet(path,index=False)
            books.append({"name":f"top{top}_{name}","scores":str(path),"prices":str(study/"prices.parquet"),"rebalance":20,"ema":10})
    (root/"books.json").write_text(json.dumps(books,indent=2),encoding="utf-8")
    return books


def execute(book,stress,output,smoke_sessions=None):
    logger.remove()
    rows=pd.read_parquet(book["scores"],columns=["date","ticker","prediction"])
    if smoke_sessions:
        rows=rows.loc[rows.date.isin(sorted(rows.date.unique())[:smoke_sessions])].copy()
    first=rows.date.min()-pd.Timedelta(days=130);last=rows.date.max();names=rows.ticker.unique().tolist()
    p=pd.read_parquet(book["prices"],columns=["date","ticker","permno","permco","close","adj_close","volume","total_return","is_terminal","is_tradable"],
        filters=[("date",">=",first),("date","<=",last),("ticker","in",names)])
    rows=rows.merge(p[["date","ticker","permno","permco"]],on=["date","ticker"],how="left",validate="one_to_one")
    if rows.permno.isna().any():raise ValueError("Prediction key has no price identity")
    execution={"initial_capital":1_000_000.,"gross_target":1.,"max_stock_weight":.015,
        "max_issuer_gross_weight":.03,"max_participation_rate":.1,"min_names":100}
    policy=TurnoverPolicy(exit_fraction=.3,ema_halflife=book["ema"],rebalance_sessions=book["rebalance"],tolerance=.2)
    weights,_=build_turnover_targets(rows,"prediction",execution,policy)
    costs=CostModel(max_participation_rate=.1)
    fields=("commission_per_share","commission_pct","commission_min","slippage_bps","impact_coeff","borrow_rate_annual","sec_fee_rate","finra_taf_per_share")
    if stress!=1:costs=replace(costs,**{f:getattr(costs,f)*stress for f in fields},finra_taf_max=costs.finra_taf_max*stress)
    print(f"replaying {book['name']}/{stress:g}x, {rows.date.nunique()} sessions",flush=True)
    trades,pnl=simulate_execution(weights,p,costs,initial_capital=1_000_000.,missing_terminal_return="total_loss",
        return_daily_ledger=True,target_tolerance=.2)
    daily,sources,stats=turnover_cost_diagnostics(trades,pnl,1_000_000.,commission_min=stress)
    prev=pnl.nav_net.shift().fillna(1_000_000.)
    np.testing.assert_allclose(prev*(1+pnl.gross_ret)-pnl.total_cost_dollar,pnl.nav_net,rtol=1e-11)
    assert trades.shares.eq(np.floor(trades.shares)).all()
    assert trades.loc[trades.shares.eq(0),"total_cost"].eq(0).all()
    assert trades.executed_participation.dropna().le(.1+1e-12).all()
    target_gross=weights.groupby("date").weight.apply(lambda s:s.abs().sum())
    assert target_gross.le(1+1e-10).all()
    fills=trades.loc[trades.shares.ne(0)]
    result={"name":book["name"],"cost_multiplier":stress,"rebalance_sessions":book["rebalance"],
        **performance_summary(pnl),**stats,"p95_executed_participation":float(fills.executed_participation.quantile(.95)),
        "max_executed_participation":float(fills.executed_participation.max()),"policy":asdict(policy)}
    dest=Path(output)/book["name"]/f"cost_{stress:g}x";dest.mkdir(parents=True,exist_ok=True)
    trades.attrs={};pnl.attrs={};trades.to_parquet(dest/"trades.parquet",index=False);pnl.to_parquet(dest/"daily_pnl.parquet")
    weights.to_parquet(dest/"targets.parquet",index=False);daily.to_csv(dest/"daily_turnover_costs.csv");sources.to_csv(dest/"cost_sources.csv",index=False)
    (dest/"summary.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--kind",choices=["models","broad"],required=True)
    p.add_argument("--study",type=Path);p.add_argument("--output",type=Path);p.add_argument("--jobs",type=int,default=4);p.add_argument("--smoke-sessions",type=int)
    a=p.parse_args();study=(a.study or ROOT/("data/wrds/batch4/prior_study_20261024" if a.kind=="models" else "data/wrds/batch4/broad_research_20261024")).resolve()
    books=prepare_model_books(study) if a.kind=="models" else prepare_broad_books(study)
    output=(a.output or study/"replays").resolve()
    if output.exists():raise ValueError("Choose a fresh replay output")
    output.mkdir(parents=True)
    sources={}
    for file in [Path(__file__),*list((ROOT/"quant_platform").rglob("*.py"))]:
        if "paper" in file.parts:continue
        rel=file.relative_to(ROOT);saved=output/"source_snapshot"/rel;saved.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(file,saved)
        sources[str(rel)]=fingerprint(file)["sha256"]
    write_run_metadata(output/"experiment_plan.json",stage="matched_prior_or_breadth_execution",
        inputs={"driver":Path(__file__),"books":study/"books/books.json"},params={"books":books,"sources":sources,
            "cost_multipliers":[1,2],"smoothing_and_buffer":"EMA10, entry20/exit30, target tolerance20% on all books"},
        extra={"historical_review_not_untouched_holdout":True,"smoke":bool(a.smoke_sessions),"production_admitted":False})
    results=[];failures=[]
    with ProcessPoolExecutor(max_workers=a.jobs) as pool:
        future_map={pool.submit(execute,b,stress,output,a.smoke_sessions):(b["name"],stress) for b in books for stress in (1,2)}
        for future in as_completed(future_map):
            try:
                result=future.result();results.append(result)
                pd.DataFrame([{k:v for k,v in r.items() if k!="policy"} for r in results]).to_csv(output/"performance.csv",index=False)
                print(f"completed {len(results)}/{len(future_map)} {future_map[future]} net={result['net_ann_return']:.3%}",flush=True)
            except Exception as error:
                failures.append({"book":future_map[future],"error":repr(error)})
                (output/"failures.json").write_text(json.dumps(failures,indent=2),encoding="utf-8")
                print(f"FAILED {future_map[future]}: {error}",flush=True)
    (output/"completion.json").write_text(json.dumps({"completed":len(results),"expected":len(future_map),"failures":failures,"smoke":bool(a.smoke_sessions)},indent=2),encoding="utf-8")
    if failures:raise RuntimeError("Replay failures are preserved")


if __name__=="__main__":main()
