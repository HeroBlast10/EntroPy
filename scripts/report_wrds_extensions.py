"""Verify and publish the reviewed-history breadth, prior and alternative study."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from quant_platform.core.utils.provenance import fingerprint
from quant_platform.core.alpha_models.nonlinear_alpha import daily_rank_ic
from quant_platform.core.evaluation.factor_diagnostics import hac_summary
from scripts.replay_wrds_extensions import positive_icir_forecast

BATCH=ROOT/"data/wrds/batch4"
ASSETS=ROOT/"docs/assets/research_extensions_20261024"
STUDY=BATCH/"prior_study_20261024"
BROAD=BATCH/"broad_research_20261024"
ALT=BATCH/"alternative_features_20261024"
DIAG=BATCH/"ml_diagnostics_20261024"


def table(headers,rows):
    return "\n".join(["| "+" | ".join(headers)+" |","| "+" | ".join(["---"]*len(headers))+" |",
        *["| "+" | ".join(map(str,row))+" |" for row in rows]])


def percent(value):return f"{value:.2%}"


def frozen_sources():
    old=ROOT/"data/wrds/batch3/neutral_2013_2025"
    manifests=[(old/"plan.json","source_sha256"),(old/"risk_replays_20d/experiment_plan.json","params.source_sha256")]
    checked=0
    for path,key in manifests:
        doc=json.loads(path.read_text());values=doc
        for part in key.split("."):values=values[part]
        folder=path.parent/"source_snapshot"
        for name,sha in values.items():
            assert fingerprint(ROOT/name)["sha256"]==sha,f"Previous frozen source changed: {name}"
            assert fingerprint(folder/name)["sha256"]==sha
            checked+=1
    return checked


def training_audit():
    plan=json.loads((STUDY/"plan.json").read_text())
    summary=pd.read_csv(STUDY/"model_summary.csv")
    for name,sha in plan["source_sha256"].items():
        assert fingerprint(ROOT/name)["sha256"]==sha
        assert fingerprint(STUDY/"source_snapshot"/name)["sha256"]==sha
    keys=None;rows=[];audit=[]
    for folder in sorted((STUDY/"models").glob("*/*/h*")):
        complete=json.loads((folder/"completion.json").read_text())
        rel=folder.relative_to(STUDY).parts
        expected=summary.loc[summary.block.eq(rel[1])&summary.model.eq(rel[2])&summary.horizon.eq(int(rel[3][1:]))].iloc[0]
        assert fingerprint(folder/"oof.parquet")["sha256"]==expected.oof_sha256
        frame=pd.read_parquet(folder/"oof.parquet",columns=["date","ticker"])
        assert not frame.duplicated().any()
        if keys is None:keys=frame
        else:assert keys.equals(frame),f"Different prediction keys: {folder}"
        assert len(complete["folds"])==13
        for fold in complete["folds"]:
            assert pd.Timestamp(fold["max_training_label_end"])<pd.Timestamp(fold["validation_start"])
            assert pd.Timestamp(fold["max_inner_training_label_end"])<pd.Timestamp(fold["inner_validation_start"])
            assert fingerprint(folder/f"fold_{fold['fold']}.parquet")["sha256"]==fold["output_sha256"]
            audit.append({"cohort":str(folder.relative_to(STUDY)),"fold":fold["fold"],
                "features":len(fold["features"]),"selected":fold["selected"],"outer_rows":fold["outer_rows"]})
        rows.append(str(folder.relative_to(STUDY)))
    assert len(rows)==8 and len(keys)==1_584_035 and keys.date.nunique()==3148
    broad=json.loads((BROAD/"prior_model/fold_audit.json").read_text())
    for row in broad:
        assert pd.Timestamp(row["max_training_label_end"])<pd.Timestamp(row["validation_start"])
        assert pd.Timestamp(row["max_inner_label_end"])<pd.Timestamp(row["inner_validation_start"])
    recovered=pd.read_csv(DIAG/"coefficient_recovery_audit.csv")
    assert len(recovered)==52 and recovered.max_prediction_error.max()==0
    components=[pd.read_parquet(STUDY/f"models/{b}/ridge_prior/h{h}/oof.parquet",columns=["prediction","inner_hac_icir"])
        for b,h in (("price_only",20),("financial_only",60),("financial_only",120))]
    w=np.column_stack([p.inner_hac_icir.to_numpy() for p in components]);s=np.column_stack([p.prediction.to_numpy() for p in components])
    forecast=positive_icir_forecast(s,w)
    saved=pd.read_parquet(STUDY/"books/training_icir_ensemble.parquet",columns=["prediction"]).prediction.to_numpy()
    assert np.array_equal(forecast,saved)
    pd.DataFrame(audit).to_csv(ASSETS/"training_fold_summary.csv",index=False)
    return {"cohorts":len(rows),"folds":13,"prediction_rows_per_cohort":len(keys),"sessions":3148,
        "oos_start":str(keys.date.min().date()),"oos_end":str(keys.date.max().date()),
        "prediction_key_sha256":hashlib.sha256(keys.astype(str).to_csv(index=False).encode()).hexdigest(),
        "nested_label_end_purge_verified":True,"signed_coefficient_refits":52,"max_recovered_prediction_error":0,
        "ensemble_cash_guard_change_max_score_difference":0,"ensemble_all_nonpositive_weight_rows":int((np.maximum(w,0).sum(axis=1)==0).sum())}


def data_governance_audit():
    meta=json.loads((BROAD/"completion.json").read_text())
    for name in ("terminal_supplement","terminal_supplement_audit","universe_implementation","driver"):
        row=meta["inputs"][name]
        assert fingerprint(Path(row["path"]))["sha256"]==row["sha256"]
    meta=json.loads((ALT/"run_metadata.json").read_text())
    assert meta["source_point_in_time_proven"] is False and meta["production_admitted"] is False
    for name in ("driver","ibes_role_repair"):
        row=meta["inputs"][name]
        assert fingerprint(Path(row["path"]))["sha256"]==row["sha256"]
    frame=pd.read_parquet(ALT/"features.parquet")
    assert not frame.duplicated(["date","ticker"]).any()
    checked=0
    for name in [c for c in frame.columns if c.endswith("_available_date")]:
        assert (frame[name].isna()|frame[name].le(frame.date)).all()
        checked+=1
    assert checked==9
    shared=json.loads((BATCH/"shared_crsp_return_audit.json").read_text())
    assert shared["max_return_error"]==0 and shared["terminal_flag_disagreements"]==0
    meta=json.loads((BROAD/"prior_model/training_metadata.json").read_text())
    for name in ("factors","driver","reference_plan"):
        row=meta["inputs"][name]
        assert fingerprint(Path(row["path"]))["sha256"]==row["sha256"]
    return {"alternative_feature_availability_columns_checked":checked,
        "source_point_in_time_proven":False,"terminal_supplement_hash_verified":True,"shared_crsp_return_audit":shared}


def load_replays():
    roots=[("models",STUDY/"replays",24),("breadth",BROAD/"replays",8),
        ("broad_prior",BROAD/"prior_model/replays",2),("alternative",ALT/"replays",18)]
    summaries=[];books={};verification=[]
    for stage,root,expected in roots:
        done=json.loads((root/"completion.json").read_text())
        assert done["completed"]==done["expected"]==expected and not done["failures"] and not done["smoke"]
        frame=pd.read_csv(root/"performance.csv")
        assert len(frame)==expected and not frame.duplicated(["name","cost_multiplier"]).any()
        if (root/"experiment_plan.json").exists():
            meta=json.loads((root/"experiment_plan.json").read_text())
            for name,sha in meta.get("params",{}).get("sources",{}).items():
                assert fingerprint(root/"source_snapshot"/name)["sha256"]==sha
        frame["stage"]=stage;summaries.append(frame)
        for row in frame.itertuples():
            folder=root/row.name/f"cost_{row.cost_multiplier:g}x"
            pnl=pd.read_parquet(folder/"daily_pnl.parquet")
            assert len(pnl)==3148 and pnl.index.is_monotonic_increasing and pnl.index.is_unique
            trades=pd.read_parquet(folder/"trades.parquet",columns=["date","shares","total_cost","executed_participation"])
            assert trades.shares.eq(np.floor(trades.shares)).all()
            assert trades.loc[trades.shares.eq(0),"total_cost"].eq(0).all()
            assert trades.executed_participation.dropna().le(.1+1e-12).all()
            costs=trades.groupby("date").total_cost.sum().reindex(pnl.index,fill_value=0.)
            np.testing.assert_allclose(costs,pnl.trading_cost_dollar,atol=1e-7)
            sources=pd.read_csv(folder/"cost_sources.csv")
            np.testing.assert_allclose(sources.total_cost_dollar.sum(),pnl.total_cost_dollar.sum(),atol=1e-6)
            prev=pnl.nav_net.shift().fillna(1_000_000.)
            np.testing.assert_allclose(prev*(1+pnl.gross_ret)-pnl.total_cost_dollar,pnl.nav_net,rtol=1e-11)
            key=(stage,row.name,row.cost_multiplier);books[key]=pnl
            verification.append({"stage":stage,"name":row.name,"cost_multiplier":row.cost_multiplier,
                "integer_shares_zero_reject_fee_and_cost_nav_reconciliation":True,
                "actual_mean_gross":float(pnl.gross_exposure.mean()),"actual_max_gross":float(pnl.gross_exposure.max()),
                "terminal_settlements":int(pnl.terminal_settlement_value.ne(0).sum())})
            print(f"verified {len(verification)}/52 {stage}/{row.name}/{row.cost_multiplier:g}x",flush=True)
    perf=pd.concat(summaries,ignore_index=True)
    perf.to_csv(ASSETS/"all_performance.csv",index=False)
    pd.DataFrame(verification).to_csv(ASSETS/"ledger_verification.csv",index=False)
    return perf,books


def paired_bootstrap(books):
    pairs=[]
    for block in ("price_only","price_plus_financial"):
        baseline=f"equal_weight_{block}"
        for model in ("ridge_zero","ridge_prior","lightgbm"):
            pairs.append((("models",f"{block}_{model}_h20",1),("models",baseline,1)))
        pairs.append((("models",f"{block}_ridge_prior_h20",1),("models",f"{block}_ridge_zero_h20",1)))
    pairs += [(("models","training_icir_ensemble",1),("models","equal_weight_price_plus_financial",1))]
    for name in ("residual_momentum","equal_weight_price_only"):
        pairs.append((("breadth",f"top2000_{name}",1),("breadth",f"top500_{name}",1)))
    pairs.append((("broad_prior","top2000_ridge_prior",1),("breadth","top2000_equal_weight_price_only",1)))
    for key in books:
        if key[0]=="alternative" and key[2]==1:
            pairs.append((key,("models","equal_weight_price_plus_financial",1)))
    n=3148;rng=np.random.default_rng(2718)
    start=rng.integers(0,n,size=(1500,int(np.ceil(n/60))))
    ids=((start[:,:,None]+np.arange(60))%n).reshape(1500,-1)[:,:n]
    def sh(v,axis):return np.mean(v,axis=axis)/np.std(v,axis=axis,ddof=1)*np.sqrt(252)
    rows=[]
    for a,b in pairs:
        x,y=books[a],books[b];assert x.index.equals(y.index)
        xv,yv=x.net_ret.to_numpy(),y.net_ret.to_numpy()
        differences=sh(xv[ids],1)-sh(yv[ids],1)
        log_diffs=(np.log1p(xv[ids])-np.log1p(yv[ids])).mean(axis=1)*252
        rows.append({"stage":a[0],"candidate":a[1],"baseline":b[1],"sharpe_difference":sh(xv,0)-sh(yv,0),
            "ci_low":np.quantile(differences,.025),"ci_high":np.quantile(differences,.975),
            "annual_log_return_difference":float(np.mean(np.log1p(xv)-np.log1p(yv))*252),
            "annual_log_return_ci_low":np.quantile(log_diffs,.025),"annual_log_return_ci_high":np.quantile(log_diffs,.975)})
    result=pd.DataFrame(rows);result.to_csv(ASSETS/"paired_bootstrap.csv",index=False);return result


def figures():
    for name in ["score_diagnostics.csv","ridge_coefficients_by_fold.csv","ridge_coefficient_stability.csv","coefficient_recovery_audit.csv"]:
        shutil.copy2(DIAG/name,ASSETS/name)
    shutil.copy2(DIAG/"ridge_coefficients.png",ASSETS/"ridge_coefficients_actual.png")
    coeff=pd.read_csv(DIAG/"ridge_coefficients_by_fold.csv")
    matrix=coeff.loc[coeff.label.eq("cross_sectional")&coeff.block.eq("price_only")].pivot(index="feature",columns="fold",values="coefficient")
    relative=matrix.div(matrix.abs().max(axis=1).replace(0,np.nan),axis=0).fillna(0)
    fig,ax=plt.subplots(figsize=(10,12));im=ax.imshow(relative,cmap="RdBu_r",vmin=-1,vmax=1,aspect="auto")
    ax.set_yticks(range(len(matrix)),matrix.index,fontsize=8);ax.set_xticks(range(13),range(1,14));ax.set_xlabel("Outer fold")
    ax.set_title("Ridge signed coefficients: row-normalized to each feature's maximum |beta|\nWithin-row signs and changes only; magnitudes across features are not comparable")
    fig.colorbar(im,ax=ax,shrink=.5,label="beta / max |beta| within feature");fig.tight_layout();fig.savefig(ASSETS/"ridge_sign_stability.png",dpi=170);plt.close(fig)
    size=pd.read_csv(BROAD/"size_stratified_ic.csv");size.to_csv(ASSETS/"size_stratified_ic.csv",index=False)
    fig,ax=plt.subplots(figsize=(10,5));buckets=["1-500","501-1000","1001-2000","2001-3000"]
    for j,(name,label) in enumerate([("RESID_MOM_12_1M","Residual momentum"),("MOM_12_1M","12-1 momentum"),("IDIOVOL","Low idiosyncratic volatility")]):
        local=size.loc[size.factor.eq(name)].set_index("cap_bucket").reindex(buckets)
        se=local.mean_rank_ic.div(local.hac_t).abs()
        ax.errorbar(np.arange(4)+(j-1)*.2,local.mean_rank_ic,yerr=1.96*se,marker="o",ls="none",capsize=3,label=label)
    ax.set_xticks(range(4),buckets);ax.axhline(0,color="grey",lw=.7);ax.set_xlabel("Lagged company market-cap rank, before liquidity filter")
    ax.set_ylabel("Mean signed 20-day rank IC; 95% HAC interval");ax.legend(fontsize=8);fig.tight_layout();fig.savefig(ASSETS/"size_ic.png",dpi=170);plt.close(fig)
    return size


def performance_table(perf,stage):
    x=perf.loc[perf.stage.eq(stage)]
    rows=[]
    for name,g in x.groupby("name",sort=True):
        a=g.loc[g.cost_multiplier.eq(1)].iloc[0];b=g.loc[g.cost_multiplier.eq(2)].iloc[0]
        rows.append([name,percent(a.gross_ann_return),percent(a.net_ann_return),percent(b.net_ann_return),f"{a.net_sharpe:.3f}",percent(a.net_max_drawdown),
            percent(a.daily_one_way_turnover),f"{a.mean_gross_exposure:.3f}"])
    return table(["组合","毛 CAGR 1×","净 CAGR 1×","净 CAGR 2×","净 Sharpe 1×","最大回撤 1×","日均单边换手 1×","实际平均 gross 1×"],rows)


def benchmark_ic():
    ref=pd.read_parquet(STUDY/"models/price_only/ridge_prior/h20/oof.parquet",columns=["date","ticker","forward_return"])
    rows=[]
    for name in ("residual_momentum","equal_weight_price_only","equal_weight_price_plus_financial"):
        scores=pd.read_parquet(STUDY/"books"/f"{name}.parquet")
        local=ref.merge(scores,on=["date","ticker"],how="left",validate="one_to_one")
        assert local.prediction.notna().all()
        rows.append({"name":name,**hac_summary(daily_rank_ic(local,"prediction"),20)})
    frame=pd.DataFrame(rows);frame.to_csv(ASSETS/"benchmark_ic.csv",index=False);return frame


def manifest(inputs):
    snapshot=BATCH/"final_source_snapshot";snapshot.mkdir(exist_ok=True)
    sources=[Path(__file__),ROOT/"config/wrds_research_extensions.yaml"]
    sources += [ROOT/"scripts"/name for name in ["acquire_wrds_extensions.py","acquire_wrds_alternative.py","repair_wrds_ibes_basis.py",
        "repair_wrds_broad_terminal.py","build_wrds_broad_research.py","build_wrds_alternative_features.py","diagnose_wrds_ml.py",
        "research_wrds_priors.py","research_wrds_broad_prior.py","replay_wrds_extensions.py","replay_wrds_alternative.py"]]
    sources += [ROOT/name for name in ["quant_platform/core/data/research_universe.py","quant_platform/core/data/alternative_research.py",
        "quant_platform/core/alpha_models/prior_models.py","quant_platform/core/evaluation/ml_diagnostics.py"]]
    sources += [ROOT/"tests"/name for name in ["test_ml_diagnostics.py","test_prior_models.py",
        "test_alternative_research.py","test_broad_research_universe.py","test_extension_ensemble.py"]]
    hashes={}
    for source in sources:
        relative=source.relative_to(ROOT);dest=snapshot/relative;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,dest)
        hashes[str(relative)]=fingerprint(source)["sha256"]
        assert fingerprint(dest)["sha256"]==hashes[str(relative)]
    (snapshot/"manifest.json").write_text(json.dumps(hashes,indent=2),encoding="utf-8")
    doc={"inputs":{str(p.relative_to(ROOT)):fingerprint(p) for p in inputs},
        "assets":{p.name:fingerprint(p) for p in ASSETS.iterdir() if p.is_file() and p.name!="evidence_manifest.json"},
        "final_source_snapshot":str(snapshot.relative_to(ROOT)),"source_hashes":hashes,
        "historical_review_not_untouched_holdout":True,"production_admitted":False,
        "snapshot_scope":"Final new-stage drivers and helpers; phase snapshots take precedence where recorded. Not a claim of an entire environment snapshot."}
    (ASSETS/"evidence_manifest.json").write_text(json.dumps(doc,indent=2),encoding="utf-8")


def main():
    ASSETS.mkdir(parents=True,exist_ok=True)
    old_count=frozen_sources();training=training_audit();governance=data_governance_audit()
    perf,books=load_replays();paired=paired_bootstrap(books);size=figures()
    diagnostics=pd.read_csv(DIAG/"score_diagnostics.csv");raw=diagnostics.loc[diagnostics.smoothing.eq("raw")]
    diagrows=[]
    for r in raw.itertuples():
        smooth=diagnostics.loc[diagnostics.label.eq(r.label)&diagnostics.block.eq(r.block)&diagnostics.model.eq(r.model)&diagnostics.smoothing.eq("ema5")].iloc[0]
        diagrows.append([r.label,r.block,r.model,f"{r.mean_rank_ic:.5f}",f"{r.icir_daily:.3f}",
            f"{r.paired_ic_difference_hac_t:.2f}",f"{r.mean_score_resid_mom_correlation:.3f}",f"{r.mean_adjacent_rank_correlation:.4f}",f"{smooth.mean_adjacent_rank_correlation:.4f}"])
    recovery=pd.read_csv(DIAG/"ridge_coefficient_stability.csv")
    price_recovery=recovery.loc[recovery.label.eq("cross_sectional")&recovery.block.eq("price_only")]
    flips=int(price_recovery.sign_transitions.gt(0).sum())
    altic=pd.read_csv(ALT/"factor_ic.csv");altic.to_csv(ASSETS/"alternative_ic.csv",index=False)
    altrecords=[[r.feature,percent(r.coverage),f"{r.mean_rank_ic:.5f}",f"{r.hac_t:.2f}",f"{r.q_value:.3f}"] for r in altic.loc[altic.horizon.eq(20)].itertuples()]
    models=pd.read_csv(STUDY/"model_summary.csv");models.to_csv(ASSETS/"model_ic.csv",index=False)
    icrows=[[r.block,r.model,r.horizon,f"{r.raw_rank_ic:.5f}"] for r in models.itertuples()]
    benchmarks=benchmark_ic()
    benchrows=[[r.name,f"{r.mean_rank_ic:.5f}",f"{r.hac_t:.2f}"] for r in benchmarks.itertuples()]
    comparison=paired.loc[paired.stage.ne("alternative")]
    cirows=[[r.candidate,r.baseline,f"{r.sharpe_difference:+.3f}",f"[{r.ci_low:+.3f}, {r.ci_high:+.3f}]"] for r in comparison.itertuples()]
    breadth=pd.read_csv(BROAD/"daily_breadth.csv");b=breadth.loc[breadth.date.between("2013-06-27","2025-12-31"),"liquid_top2000"]
    broadmeta=json.loads((BROAD/"completion.json").read_text());repair=json.loads((BATCH/"broad_raw/terminal_repair/completion.json").read_text())
    costs=perf.loc[perf.stage.isin(["breadth","broad_prior"])&perf.cost_multiplier.eq(1)]
    small=costs.loc[costs.name.eq("top500_equal_weight_price_only")].iloc[0]
    large=costs.loc[costs.name.eq("top2000_equal_weight_price_only")].iloc[0]
    costrows=[[r.name,f"{r.unit_trading_cost_bps:.2f}",f"{r.annual_trading_drag_bps:.1f}",f"{r.annual_borrow_drag_bps:.1f}",
        f"${r.median_fill_dollar:,.0f}",percent(r.minimum_commission_fraction),percent(r.p95_executed_participation)] for r in costs.itertuples()]
    tests=(BATCH/"final_pytest.txt").read_text(errors="replace");testline=next(line for line in reversed(tests.splitlines()) if " passed" in line)
    assert "failed" not in testline.lower() and "error" not in testline.lower(),testline
    audit={**training,**governance,"replays_verified":len(perf),"previous_frozen_source_checks":old_count,"pytest_summary":testline,
        "corrected_broad_terminal_rows":repair["params"]["rows"],"no_new_strategy_admitted":True}
    (ASSETS/"verification.json").write_text(json.dumps(audit,indent=2),encoding="utf-8")
    report=f'''# 长窗口、扩池与模型先验研究：实施与结果

日期：2026-10-24。所有窗口均为**审阅过的历史窗口**，没有新增未触碰的留出集；本文比较属于修订研究协议后的描述性证据。许可数据保存在忽略提交的 `data/wrds/batch4/`，文档只展示聚合统计。

## 1. 判断与已完成的修改

建议有研究必要，收益推断需要收敛到实测结果。已完成三项诊断、CRSP 历史市值扩池、IBES / OptionMetrics 特征、固定因子组合、向先验收缩的 Ridge、训练期剪枝、财务 60/120 日模型，以及一次预先限定参数的较强 LightGBM 测试。共验证 **{len(perf)} 个完整成本账本**，包含全部不利结果。新增模型和数据代理均未替换正式研究基准。

- **P1-A：必要且已完成。** 上一阶段已有 2013-02-04–2025-12-31 的 3248 日、13 折、八模型共同 OOF；其 Ridge Sharpe 标准误从近期窗口约 0.50 降至约 0.24，不能将它解释成收益显著。
- **P1-B：值得实证，不是 IR 自动乘二。** 已扩至历史公司市值前 2000 并做前 500 控制、3000 名以内分层诊断和三因子先验 Ridge。证券数不等于独立赌注数，流动性、行业相关、风险与每笔最低佣金都会影响结果。本文给出修复退市结算后的完整结果。
- **P1-C：值得接入，但尚未通过数据准入。** 已实际下载并治理九个代理特征；原生公告日期并不能证明供应商历史版本未被改写，月度快照的可得日期仍含假设。27 项因子 × 持有期检验没有一项 BH q < 0.05。
- **模型先验：改善了预测，尚未兑现为相对简单基准的净收益优势。** 本次先验 Ridge 的 20 日原始 IC 高于零中心 Ridge，但对应净收益仍未超过同口径固定因子组合；60/120 日财务模型与 ICIR 组合也没有解决问题。

仅凭“37 维、约八万个独立观测”的粗略估算，不能证明维度无法学习；横截面和时间相关都会改变有效样本量。深度 3 的树本身已是非线性模型。每 5 日取一个完整截面减少冗余，**仍保留 20 日标签重叠**，因此本研究仍严格 purge 标签结束日并采用 HAC。

## 2. 先诊断旧模型，避免按 OOS 结果删特征

旧模型诊断复用上一阶段固定分数；原始 IC 按相同股票和有标签的日期配对，3228 个可评估日期，统一 20 日 CRSP 总收益标签。配对 RESID_MOM 的 IC 为 **0.008486**、HAC t = 0.86，本身也不显著。日度 ICIR 定义为 IC 均值 / 日度标准差；资产文件另有朴素 √252 年化值，重叠标签下不能据此推断显著性。

{table(["标签","特征块","模型","原始 IC","日 ICIR","IC差 HAC t","与残差动量相关","原始 lag1 rankρ","EMA5 lag1 rankρ"],diagrows)}

八个原始 ML 分数的 IC 点估计都低于残差动量，但配对差异并未形成统计上确定的劣势。相关性仅约 0.10–0.23，不支持“模型只是在复制动量”。EMA5 后 rank 自相关已经约 0.995 或更高；残差动量 EMA5 为 0.99948。高自相关可能仅反映机械平滑，不能替代持有期收益和成本检验，也没有证据支持把 EMA20 直接设为新默认。

按原参数、原训练样本索引和原 BLAS 线程数恢复了四组 Ridge × 13 折的有符号系数，52 次恢复对保存 OOF 的最大预测误差为 **0**。原横截面价量模型 {flips}/37 个特征至少一次发生幅度超过 1e-5 的跨折符号变化。符号翻转可来自共线性和分布变化，不能直接定性为噪声；本轮训练选择没有使用这些 OOS 翻转统计。

![Ridge 跨折符号与相对幅度](assets/research_extensions_20261024/ridge_sign_stability.png)

上图每行除以该特征最大绝对系数，仅供观察同一特征的符号及变化；特征间幅度不可比较。[实际系数量纲图](assets/research_extensions_20261024/ridge_coefficients_actual.png)与[有符号系数](assets/research_extensions_20261024/ridge_coefficients_by_fold.csv)一并保留。

## 3. 新模型协议与简单基准

新比较为了共同容纳 120 日标签，统一从 **{training['oos_start']}** 至 **{training['oos_end']}**，3148 日、13 折，每个模型 **1,584,035** 个相同预测键。起点晚于上一阶段，特征字典、标准化和训练抽样亦有变化，因此不能把差异归因为仅拉长回放窗口。最后一段没有成熟未来标签时仍保留预测用于回放，只从 IC 中排除。

固定价量先验为 RESID_MOM + MOM_12_1M − IDIOVOL 三因子，财务增强版再加入 GROSS_PROFITABILITY − ACCRUALS，共五因子。每个因子先做当日横截面 rank 映射到 [-1,1]，再等权；缺失置 0。它们是本次固定的经济方向假设，**没有伪称为独立预登记后显著筛出的赢家**。

先验 Ridge 最小化 `mean((y − intercept − Xβ)²) + λ||β − β₀||²`。固定先验因子系数为 ±1/k，其余为 0，截距不惩罚；实现中 sklearn 的惩罚尺度为 nλ。零中心对照使用相同特征、样本、内层选择和 λ ∈ {{0.01,1,100}}。每折只从成熟训练标签计算方向对齐的 HAC t，非先验特征须 t > 1.5；固定先验不按 OOS 显著性移除。

重复项仅在训练期 |相关| ≥ 0.995 时删除；短周期候选在 1 日强、20 日弱时不进入该 20 日模型。价量 Ridge 最终每折约 5–8 个特征，财务增强约 7–11 个。执行入场过滤需要独立成本试验，本次未把被删短因子直接加到执行层。

LightGBM 保留去重后的完整价量 / 财务字典以允许交互，最高 1000 棵、lr 0.03、Huber、31 叶、深度 5、min_leaf 200、L2 20、固定 seed 17。每 5 个训练交易日取整个截面，内层 126 日验证、60 轮 early stopping，外层严格按内层迭代数重训；部分折只选 1 棵，最高价量 775 / 财务 577 棵，不强行在验证退化后训练到 300 棵。[LightGBM 参数文档](https://lightgbm.readthedocs.io/en/latest/Parameters.html)

{table(["特征块","模型","标签日数","原始 20/60/120 日 IC"],icrows)}

相同新协议预测键下，固定基准的 20 日原始 IC 如下。它们与上一节旧模型诊断的时期、缺失处理有区别，跨表差值不能视为单一改动的因果效果。

{table(["固定基准","20 日 IC","HAC t"],benchrows)}

两个先验 Ridge 的原始 IC 也低于各自固定等权基准的点估计；与零中心相比改善，不等于 ML 已创造增量。固定基准自身的 HAC t 同样不足以支持显著性结论。

财务 60/120 日模型独立训练，分别按 60/120 日调仓；跨持有期组合仅以当折内层验证的正 HAC ICIR 加权。全部非正时输出缺失分数以清空 EMA 状态、在调仓时退出持仓，避免零分数在 EMA 后残留旧排序；该分支已单测，本次 3148 日没有触发，不影响已保存回放。长标签不仅是把 20 日模型多持有几天。本次财务长标签 IC 为负，不能接受“短期不显著必然是持有期错配”的解释。

## 4. 同口径成本回放

共同执行参数：$1M，目标 gross 1，单股 1.5%、发行人 3% 上限，EMA10，进入前 / 后 20%、退出 30% 缓冲，目标权重 20% 容忍带，整股成交；20 日模型和固定基准均每 20 交易日调仓。成本包括每股佣金及每单最低 $1、5 bps 单边滑点、参与率冲击、监管费，以及空头年化 50 bps 借券。2× 同时重跑所有交易与借券成本，非简单从 CAGR 减常数。参数用于研究，不能当成已核实的当前券商报价。

{performance_table(perf,'models')}

最大回撤列为负数。目标 gross 上限约束不等于连续实际持仓上限：调仓间漂移、整股和容忍带会改变 gross；表中列出实际平均值，完整文件另有最大 gross、净敞口和波动率。本文未启用上一阶段的风险中性优化器，也未向上放大到 10% 波动率，避免同时改动信号、风险目标与执行协议。对照结论只适用于本协议。

以下按同日净收益做循环 60 日块 bootstrap（1500 次，seed 2718），展示净 Sharpe 差的 95% 区间。它条件于已实现策略路径，未重新执行完整自适应研究过程，未校正整段历史反复查看和众多模型比较；不等于新留出集的显著性认证。

{table(["候选","配对基准","Δ净 Sharpe","95% 块区间"],cirows)}

这次较强 LightGBM 没有超过固定因子组合，足以暂停该配置的继续调参；不能由一次实验推广为“该股票池所有非线性都无增量”。先验收缩优于零中心的点估计，也不足以支持替换等权基准。

## 5. 扩池与市值分层

原始 common 股份历史共 23,709,962 行、12,139 个历史 PERMNO，覆盖 2008–2025；2008/09 用作预热。按历史公司总市值排名，先聚合所有普通股类别，再按上一日 ADV 选择一个代表类别。采用上一交易日价格 ≥ $5、完整连续 20 日历史 ADV ≥ $2M、上市至少 60 个日历日、美国注册普通股以及 NYSE / AMEX / Nasdaq；先排名再过滤流动性，不用较小股票补足 2000 个名额。新协议回放窗口内有效公司数中位数 **{b.median():.0f}**，最少 {b.min():.0f}，最多 {b.max():.0f}。

这里的前 500 是按同一 CRSP 规则构造的控制组，**并非 S&P 500 指数成分**；也未声称复制 Russell 指数。不同市值组共用扩池前 2000 的市场 / 大小代理，残差动量用过去 504 日回归、每 21 日更新系数，过去 231 日残差和再滞后 21 日；这与旧缓存特征的定义和预热不同，只能在本次扩池控制内比较。

![市值分层 IC](assets/research_extensions_20261024/size_ic.png)

图中 IDIOVOL 已按负方向对齐；诊断期为 2013-02-04–2025-12-31，点为日度横截面 IC 均值，线段为未对 12 项比较作多重调整的 HAC 95% 区间。分层结果支持继续研究低特质波动在较小市值组的差异，不能把它直接视作去除风险与真实借券成本后的 alpha。

{performance_table(perf,'breadth')}

扩池三因子先验 Ridge 使用相同固定先验、相同 13 折日期、每五日完整截面和相同 λ 内层选择，只估计三维系数，没有再做第二套 LightGBM 网格。

{performance_table(perf,'broad_prior')}

{table(["扩池组合","每单位交易成本 bps","年交易拖累 bps","年借券拖累 bps","成交金额中位数","最低佣金订单占比","实际参与率 p95"],costrows)}

参与率以成交股数 / 历史 ADV 股数计；最低佣金比例是订单数比例，不是总成本比例。$1M 分散到更多名字会缩小订单，并提高单位佣金成本，因此低参与率本身不能保证扩池提升净 IR。年化成本拖累基于实际每天 NAV 的费用比例，不能与 CAGR 直接线性相减。

具体到三因子等权控制，毛 CAGR 从 {small.gross_ann_return:.2%} 上升到 {large.gross_ann_return:.2%}，净 CAGR 却从 {small.net_ann_return:.2%} 降到 {large.net_ann_return:.2%}；单位交易成本从 {small.unit_trading_cost_bps:.2f} 升至 {large.unit_trading_cost_bps:.2f} bps，成交金额中位数从 ${small.median_fill_dollar:,.0f} 降至 ${large.median_fill_dollar:,.0f}。这支持把佣金方案与账户规模纳入扩池研究，而非把低参与率等同于低总成本；本轮没有按已见收益另调佣金参数。

**结算修复及无效结果披露：** 首版扩池拉取把 `securitytype=EQTY AND securitysubtype=COM` 用于所有行，遗漏了退市时被标成 `N/A / UNK` 的结算观察，造成持仓滞留。现已从同一 CIZ 表补取 {repair['params']['rows']:,} 条 `dlydelflg=Y` 记录，并入原历史普通股 PERMNO；采用原生内嵌退市收益、不再次叠加 dlret。缺失结算收益按完全损失的次数为 {broadmeta['imputed_missing_terminal_returns']}。错误版本整目录保存在 `broad_research_20261024_quarantined_missing_terminal` 并标记不准入，本文仅使用补全后重建的特征、标签、训练和回放。

另将扩池后价格与原 S&P 价格交叉对账：2,903,815 个共享日期 / PERMNO 键中，2,900,853 个共同非空收益完全一致，233 条原 S&P 结算记录也全部匹配，退市标记没有分歧。旧 S&P 研究不受此次扩池查询错误影响。

## 6. IBES / OptionMetrics：实际接入与可得性限制

已核验账户的 SELECT 权限，读取真实表结构并使用带有效起止日的 CRSP–IBES 和 CRSP–OptionMetrics 历史链接；只接受 score ≤ 1 且当时唯一的证券匹配，歧义链接抑制信号。新信息范围先限定为原 S&P 研究的历史 PERMNO 并集，尚未声称已覆盖新增 2000 公司全部名字。[WRDS 历史链接说明](https://wrds-www.wharton.upenn.edu/pages/wrds-research/database-linking-matrix/linking-ibes-with-crsp/)

五个 IBES 代理为修正广度、同财政期 EPS 修正、分歧度、目标价隐含收益、价格缩放 SUE；四个期权代理为 30 日 put25 − call50 skew、call50 − put50 IV、IV − 21 日已实现波动率、期权合约量 / 百万股票成交股数。期权表来自 EOD 常期限 / delta 曲面及日成交量，不假设每份期权合约都对应 100 股。[OptionMetrics 产品口径](https://optionmetrics.com/data-products/)

call/put IV 差是同 delta / 同期限代理，不是严格同执行价的 Bali–Hovakimian 复刻；论文提供机制参考，不能替代本项目验证。[Bali 与 Hovakimian，2009](https://pubsonline.informs.org/doi/abs/10.1287/mnsc.1090.1063?journalCode=mnsc)

{table(["特征","可评估覆盖率","20 日原始 IC","HAC t","全 27 项 BH q"],altrecords)}

IC 保留原始方向；回放前固定的方向为：修正、目标价和 SUE 为正，分歧与 put skew 为负，call−put IV 与 IV−RV 为正；无方向期权量采用正注意力代理，明确属于探索性方向假设。这些方向在查看因子 IC 后、运行回放前写入计划，不能伪称事前注册。九个都回放，没有删除失败者或在收益出现后翻转符号。

{performance_table(perf,'alternative')}

SUE 使用未复权原始季度实际 EPS、公告前已可得的同财政期月度共识，差值除公告前股价，再除过去八次事件差值的标准差（至少四次），它是**月度共识、价格缩放的 SUE 代理**。修正只比较相同财政期，不把 FY1 换年跳跃当成盈利修正。股价、EPS 与目标价口径需要一致，拆股窗口排除 35 日，以减少供应商生效日期差异；这仍不是历史版本认证。[WRDS：IBES 未复权数据说明](https://wrds-www.wharton.upenn.edu/documents/5/A_Note_on_IBES_Unadjusted_Data_pdf.pdf)

保守可得时间统一为快照日、公告日与供应商激活日之后的首个交易日；季度共识严格早于公告，滚动标准差只读此前事件。IBES 月度快照的实际发布时间未逐条核实，这个次日规则仍是研究假设。月度特征最多陈旧 45 日、SUE 126 日、期权 7 日；显式缺失会清空旧信号，链接到期立即失效，不用旧值跨链接续接。原生 available_date、严格因果计算和供应商 PIT vintage 是三个不同问题，`source_point_in_time_proven=false`。

**原始数据角色更正：** 最初把 `nact_epsus` 错认成未复权原始实际盈利；WRDS 原生表说明显示 `n` 表示 Normalized。错误版本已隔离，原反向 SUE 推断撤回。最终实际值只来自 `actu_epsus`，目标价来自 `ptgsumu`，并有描述字段守卫拒绝 Normalized 来源；`nactu_epsus` 同样不能因含 unadjusted 就被误认为原始口径。修复清单、原始文件、失败与隔离产物均保留，本文不采用其收益或 IC。

## 7. 准入结论、验证与复现

当前保留固定因子基准、低换手工具、长 OOS 诊断与新数据研究接口；新增学习模型和替代数据策略均为研究候选。SUE 有正向点估计，但多重检验、历史版本、真实借券与独立未来窗口仍待验证。没有依据把新增数据直接并入正式模型，或承诺净收益达到某个目标。

所有 {len(perf)} 个完整账本均检查整股、零成交零费用、参与率 ≤ 10%、订单费用与日账本及成本来源加总、NAV 费用恒等式；八模型预测键一致，内外训练标签严格早于验证起点。上一阶段冻结来源 {old_count} 项哈希保持一致。完整测试：**{testline}**。验证不会证明风险模型正确、供应商无修订或真实卖空一定可执行。

仍有边界：研究使用总收益美元持仓账本，没有另建逐股现金 / 公司行动的第二套独立对账；固定借券利率没有真实可借数量和个股费率，分红及退出处理依赖 CIZ；实际 gross 可能超出目标，不能据此宣称严格中性。训练期 t 筛选也有选择误差，未来应以已冻结协议的增量窗口检验，而非继续查看同段历史后宣布有效。

复现入口（从项目根目录运行；已完成输出目录禁止覆盖，需指定新目录或调整独立研究输出）：

```powershell
python scripts/diagnose_wrds_ml.py --help
python scripts/acquire_wrds_extensions.py --help
python scripts/repair_wrds_broad_terminal.py
python scripts/build_wrds_broad_research.py --help
python scripts/acquire_wrds_alternative.py --help
python scripts/repair_wrds_ibes_basis.py
python scripts/build_wrds_alternative_features.py --help
python scripts/research_wrds_priors.py --help
python scripts/replay_wrds_extensions.py --help
python scripts/research_wrds_broad_prior.py --help
python scripts/replay_wrds_alternative.py
python scripts/report_wrds_extensions.py
```

[统一研究配置](../config/wrds_research_extensions.yaml)、[全部绩效与成本](assets/research_extensions_20261024/all_performance.csv)、[配对区间](assets/research_extensions_20261024/paired_bootstrap.csv)、[验证记录](assets/research_extensions_20261024/verification.json)、[证据清单](assets/research_extensions_20261024/evidence_manifest.json)。阶段输入与输出哈希、训练快照和最终新增代码快照保存在各阶段元数据中；最终代码快照不冒充每个早期阶段完整环境镜像。
'''
    path=ROOT/"docs/RESEARCH_EXTENSIONS_2026-10-24.md";path.write_text(report,encoding="utf-8")
    inputs=[DIAG/"run_metadata.json",STUDY/"plan.json",STUDY/"training_completion.json",STUDY/"model_summary.csv",
        BROAD/"completion.json",BROAD/"size_stratified_ic.csv",BROAD/"prior_model/training_metadata.json",
        BATCH/"broad_raw/terminal_repair/completion.json",BATCH/"alternative_raw/ibes_unadjusted_repair/completion.json",
        ALT/"run_metadata.json",ALT/"factor_ic.csv",ALT/"features.parquet",BATCH/"final_pytest.txt",BATCH/"shared_crsp_return_audit.json",
        BROAD/"prices.parquet",BROAD/"factors.parquet",ROOT/"data/wrds/p0/prices.parquet",path]
    for root in (STUDY/"replays",BROAD/"replays",BROAD/"prior_model/replays",ALT/"replays"):
        inputs += [root/"completion.json",root/"performance.csv"]
        inputs += list(root.glob("*/cost_*/daily_pnl.parquet"))
    for root in (STUDY/"books",BROAD/"books",ALT/"books"):
        inputs += list(root.glob("*.parquet"))
    inputs += list((STUDY/"models").glob("*/*/h*/oof.parquet"))+[BROAD/"prior_model/oof.parquet"]
    manifest(inputs)
    print(json.dumps(audit,ensure_ascii=False,indent=2));print(path)


if __name__=="__main__":main()
