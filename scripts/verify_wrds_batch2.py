"""Audit full-library/model lineage and report evidence without promotion bias."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from loguru import logger
from quant_platform.core.data.research_contract import require_wrds_identity
from quant_platform.core.alpha_models.feature_schema import is_metadata
from quant_platform.core.alpha_models.nonlinear_alpha import block_bootstrap_ic_delta
from quant_platform.core.signals.registry import FactorRegistry
from quant_platform.core.utils.provenance import fingerprint,write_run_metadata,WRDS_CODE_PATHS
from scripts.build_wrds_factor_library import implementation_hash


def check_lineage(path):
    record=json.loads(path.read_text())
    for source in record.get("inputs",{}).values():
        assert fingerprint(source["path"])["sha256"]==source["sha256"],f"Stale input: {source['path']}"
    if "output" in record:
        assert fingerprint(record["output"]["path"])["sha256"]==record["output"]["sha256"],"Stale output"
    return record


def audit_models(root,policy):
    report={"policy":policy}
    experiments=root/"model_comparison"/policy/"full"
    check_lineage(experiments/"experiment_metadata.json")
    all_books=[]; model_checks=[]
    for horizon in (20,5):
        folder=experiments/f"h{horizon}"
        meta=check_lineage(folder/"run_metadata.json")
        assert meta["code"]["reproducible_from_commit"]
        features=meta["params"]["feature_blocks"]
        assert not any(is_metadata(c) for cols in features.values() for c in cols)
        frame=pd.read_parquet(folder/"oof.parquet")
        assert not frame.duplicated(["block","date","ticker"]).any()
        a=frame[frame.block.eq("price_only")][["date","ticker"]].reset_index(drop=True)
        b=frame[frame.block.eq("price_plus_financial")][["date","ticker"]].reset_index(drop=True)
        pd.testing.assert_frame_equal(a,b)
        assert frame[["ridge","lightgbm","equal_weight","ridge_placebo"]].notna().all().all()
        assert frame.loc[frame.forward_return.notna(),"label_end"].le(pd.Timestamp("2025-12-31")).all()
        folds=pd.read_csv(folder/"fold_audit.csv",parse_dates=["validation_start","max_training_label_end","max_inner_training_label_end","inner_validation_start"])
        assert folds.max_training_label_end.lt(folds.validation_start).all()
        assert folds.max_inner_training_label_end.lt(folds.inner_validation_start).all()
        trials=json.loads((folder/"trial_registry.json").read_text())
        sampled=pd.DataFrame([t for t in trials if "training_key_sha256" in t])
        for _,rows in sampled.groupby(["fold","selected_using"]):
            assert rows.training_key_sha256.nunique()==1,"Information blocks used different capped training rows"
        model_checks.append({"horizon":horizon,"prediction_rows_per_block":len(a),"folds":len(folds),
                             "features":{k:len(v) for k,v in features.items()},"identical_prediction_and_training_keys":True})
        for path in (folder/"backtests").glob("*/*/cost_*/run_metadata.json"):
            book=check_lineage(path)
            assert book["code"]["reproducible_from_commit"]
            trades=pd.read_parquet(path.parent/"trades.parquet")
            pnl=pd.read_parquet(path.parent/"daily_pnl.parquet")
            p=book["performance"]
            assert trades.executed_participation.max()<=.1+1e-12
            assert trades.permno.notna().all()
            costs=trades.groupby("date").total_cost.sum().reindex(pnl.index,fill_value=0)
            np.testing.assert_allclose(costs,pnl.trading_cost_dollar,atol=1e-7)
            previous=pnl.nav_net.shift(1).fillna(book["params"]["initial_capital"])
            np.testing.assert_allclose(previous*(1+pnl.gross_ret)-pnl.total_cost_dollar,pnl.nav_net,rtol=1e-10,atol=1e-7)
            weights=pd.read_parquet(path.parent/"rebalance_weights.parquet")
            assert weights.weight.abs().max()<=book["params"]["max_stock_weight"]+1e-12
            assert weights.groupby("date").weight.sum().abs().max()<1e-10
            all_books.append(p)
        # Paired net-return uncertainty uses full executed daily paths.
        daily={}
        for block in ("price_only","price_plus_financial"):
            for model in ("ridge","lightgbm"):
                daily[(block,model)]=pd.read_parquet(folder/"backtests"/block/model/"cost_1x/daily_pnl.parquet").net_ret
        paired=[]
        for block in ("price_only","price_plus_financial"):
            delta=block_bootstrap_ic_delta(daily[(block,"lightgbm")],daily[(block,"ridge")],block_size=max(20,horizon),n_boot=2000)
            delta["mean_daily_net_return_delta"]=delta.pop("mean_delta_ic")
            paired.append({"horizon":horizon,"block":block,"comparison":"lightgbm_minus_ridge",**delta})
        for model in ("ridge","lightgbm"):
            delta=block_bootstrap_ic_delta(daily[("price_plus_financial",model)],daily[("price_only",model)],block_size=max(20,horizon),n_boot=2000)
            delta["mean_daily_net_return_delta"]=delta.pop("mean_delta_ic")
            paired.append({"horizon":horizon,"block":model,"comparison":"financial_information_increment",**delta})
        pd.DataFrame(paired).to_csv(folder/"paired_net_return_delta.csv",index=False)
    books=pd.DataFrame(all_books)
    assert len(books)==36,"Missing a predeclared base/stress/control backtest"
    primary=pd.read_csv(experiments/"h20/paired_ic_delta.csv")
    ic_gate=primary[(primary.block.eq("price_plus_financial"))&(primary.challenger.eq("lightgbm"))&(primary.baseline.eq("ridge"))].iloc[0]
    lgb=books[books.horizon.eq(20)&books.block.eq("price_plus_financial")&books.signal.eq("lightgbm")]
    report.update(model_checks=model_checks,backtests=len(books),
        max_executed_participation=float(books.max_executed_participation.max()),
        descriptive_lgbm_ic_increment_ci_low=float(ic_gate.ci_low),
        lgbm_positive_at_base_and_double_cost=bool(lgb.net_ann_return.gt(0).all()),
        research_increment_supported=bool(ic_gate.ci_low>0 and lgb.net_ann_return.gt(0).all()),
        production_admitted=False,fundamentals_is_point_in_time=False,
        inference_limit="Paired intervals are descriptive historical evidence after inner selection; no untouched holdout or global DSR claim")
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=ROOT/"data/wrds/batch2")
    parser.add_argument("--require-models",action="store_true")
    args=parser.parse_args(); root=args.root
    logger.remove()
    meta=check_lineage(root/"factor_library_metadata.json")
    assert meta["params"]["implementation_sha256"]==implementation_hash(),"Library implementation changed"
    assert meta["code"]["reproducible_from_commit"]
    factors=pd.read_parquet(root/"factors.parquet")
    p=pd.read_parquet(ROOT/"data/wrds/p0/prices.parquet",columns=["date","ticker","permno"])
    u=pd.read_parquet(ROOT/"data/wrds/p0/universe.parquet")
    require_wrds_identity(u,factors,p)
    catalog=pd.read_csv(root/"factor_catalog.csv")
    registry=FactorRegistry(); registry.discover()
    assert set(catalog.factor)==set(registry.list_factors().name) and len(catalog)==44
    for name in catalog.factor:
        part=json.loads((root/"factor_parts"/f"{name}.json").read_text())
        assert part["implementation_sha256"]==meta["params"]["implementation_sha256"]
        assert part["prepared_sha256"]==meta["params"]["prepared_sha256"]
        assert fingerprint(root/"factor_parts"/f"{name}.parquet")["sha256"]==part["output_sha256"]
        assert factors[name].notna().any()
    prepared=pd.read_parquet(root/"prepared_prices.parquet")
    expected=prepared.sort_values(["permno","date"]).groupby("permno").industry.shift(1).reindex(prepared.index)
    assert ((expected.eq(prepared.decision_industry))|(expected.isna()&prepared.decision_industry.isna())).fillna(False).all()
    eligible=factors[factors.risk_eligible&factors.date.ge("2010-01-01")]
    assert eligible.decision_industry.notna().all()
    sample=prepared[prepared.permno.isin([10104,14542,90319])].copy()
    cutoff=pd.Timestamp("2018-12-31")
    changed=sample.copy(); future=changed.date.gt(cutoff)
    changed.loc[future,"adj_close"]*=10; changed.loc[future,"total_return"]+=.4
    past=sample.date.le(cutoff)
    for name in ("KF_VELOCITY","KF_TREND_STRENGTH","KF_NOISE_RATIO","SPECTRAL_ENTROPY_60D","HURST_60D","HMM_TURBULENCE_PROB","OU_ZSCORE"):
        factor=registry.get(name)()
        original=factor._compute(sample).reindex(sample.index)
        altered=factor._compute(changed).reindex(sample.index)
        prefix=factor._compute(sample[past]).reindex(sample.index[past])
        np.testing.assert_allclose(original[past],altered[past],equal_nan=True,atol=1e-10)
        np.testing.assert_allclose(original[past],prefix,equal_nan=True,atol=1e-10)
    results=[]
    if args.require_models:
        for policy in ("total_loss","zero_return"):results.append(audit_models(root,policy))
    report={"factor_count":44,"original_factor_count":42,"new_true_turnover_factors":2,"rows":len(factors),
            "eligible_rows":len(eligible),"industry_coverage":1.,"actual_crsp_prefix_checks":7,
            "factor_source_hashes_verified":True,"industry_effective_dates_verified":True,
            "classification_is_point_in_time":False,"fundamentals_is_point_in_time":False,
            "company_cap_scope":"CRSP-observable common classes; see P0 contract",
            "models":results,"production_admitted":False}
    write_run_metadata(root/"batch2_acceptance.json",stage="verify_wrds_batch2",code_paths=WRDS_CODE_PATHS,
        inputs={"library":root/"factor_library_metadata.json","factors":root/"factors.parquet"},extra=report)
    print(json.dumps(report,indent=2))


if __name__=="__main__":main()
