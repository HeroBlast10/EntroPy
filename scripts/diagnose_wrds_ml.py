"""Diagnose frozen long-history OOF and recover signed Ridge coefficients exactly."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits
from quant_platform.core.alpha_models.nonlinear_alpha import _daily_zscore
from quant_platform.core.evaluation.ml_diagnostics import paired_score_metrics, coefficient_stability, threshold_rank_update
from quant_platform.core.portfolio.low_turnover import smooth_scores
from quant_platform.core.utils.provenance import fingerprint, write_run_metadata


def recover_coefficients(research, output):
    plan = json.loads((research / "plan.json").read_text())
    panel = pd.read_parquet(research / "learning_panel.parquet")
    all_rows, audits = [], []
    for label in ("cross_sectional", "industry_size_vol"):
        target = panel.target if label == "cross_sectional" else panel.label_neutral_target.where(panel.label_common_trainable)
        for block, features in plan["schemas"].items():
            folder = research / f"models/{label}/{block}/ridge"
            for saved in json.loads((research / "folds.json").read_text()):
                fold = saved["fold"]
                record = json.loads((folder / f"fold_{fold}.json").read_text())
                train = np.load(research / f"sampling/fold_{fold}_outer.npy")
                valid = panel.date.isin(pd.to_datetime(saved["validation_dates"]))
                x = panel.iloc[train][features].to_numpy(dtype=np.float32)
                y = target.iloc[train].to_numpy(dtype=np.float32)
                with threadpool_limits(limits=plan["settings"]["n_jobs"]):
                    estimator = Ridge(alpha=float(record["audit"]["params"]["alpha"]), solver="lsqr", tol=1e-6).fit(x,y)
                    predicted = estimator.predict(panel.loc[valid, features].to_numpy(dtype=np.float32))
                recovered = _daily_zscore(pd.Series(predicted, index=panel.index[valid]), panel.loc[valid,"date"]).astype("float32")
                frozen = pd.read_parquet(folder / f"fold_{fold}.parquet")
                assert panel.loc[valid,["date","ticker"]].reset_index(drop=True).equals(frozen[["date","ticker"]])
                error = float(np.max(np.abs(recovered.to_numpy() - frozen.prediction.to_numpy())))
                if error > 2e-5:
                    raise ValueError(f"Signed coefficient recovery changed OOF: {label}/{block}/{fold}: {error}")
                audits.append({"label":label,"block":block,"fold":fold,"max_prediction_error":error,
                    "alpha":record["audit"]["params"]["alpha"],"intercept":float(estimator.intercept_)})
                all_rows += [{"label":label,"block":block,"fold":fold,"feature":f,"coefficient":float(v)}
                             for f,v in zip(features,estimator.coef_)]
            print(f"recovered signed coefficients: {label}/{block}", flush=True)
    result = pd.DataFrame(all_rows)
    result.to_csv(output / "ridge_coefficients_by_fold.csv",index=False)
    pd.DataFrame(audits).to_csv(output / "coefficient_recovery_audit.csv",index=False)
    summaries = []
    for (label,block), group in result.groupby(["label","block"]):
        summaries.append(coefficient_stability(group).assign(label=label,block=block))
    pd.concat(summaries,ignore_index=True).to_csv(output / "ridge_coefficient_stability.csv",index=False)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--research",type=Path,default=ROOT/"data/wrds/batch3/neutral_2013_2025")
    parser.add_argument("--output",type=Path,default=ROOT/"data/wrds/batch4/ml_diagnostics_20261024")
    parser.add_argument("--skip-coefficients",action="store_true")
    parser.add_argument("--reuse-scores",action="store_true",help="Resume a complete 32-row score diagnostic checkpoint")
    args=parser.parse_args(); research=args.research.resolve(); output=args.output.resolve()
    output.mkdir(parents=True,exist_ok=True)
    factors_path=ROOT/"data/wrds/batch2/factors.parquet"
    residual=pd.read_parquet(factors_path,columns=["date","ticker","RESID_MOM_12_1M"]).rename(columns={"RESID_MOM_12_1M":"residual_momentum"})
    cohorts=json.loads((research/"training_completion.json").read_text())["cohorts"]
    rows,dailies=[],[]
    if args.reuse_scores:
        saved=pd.read_csv(output/"score_diagnostics.csv")
        if len(saved)!=len(cohorts)*4 or saved.duplicated(["label","block","model","smoothing"]).any():
            raise ValueError("Incomplete score diagnostic checkpoint")
        if not (output/"daily_score_diagnostics.parquet").exists() or not (output/"residual_momentum_ema5.json").exists():
            raise ValueError("Missing daily or momentum checkpoint")
        rows=saved.to_dict("records");cohorts=[]
    for cohort in cohorts:
        label,block,model=(cohort[k] for k in ("label","block","model"))
        oof=research/f"models/{label}/{block}/{model}/oof.parquet"
        work=pd.read_parquet(oof,columns=["date","ticker","prediction","forward_return"])
        work=work.merge(residual,on=["date","ticker"],how="left",validate="one_to_one")
        for name,half in (("raw",None),("ema5",5),("ema10",10),("ema20",20)):
            work["score"]=smooth_scores(work,"prediction",half)
            stats,series=paired_score_metrics(work)
            rows.append({"label":label,"block":block,"model":model,"smoothing":name,**stats})
            dailies.append(series.assign(label=label,block=block,model=model,smoothing=name).rename_axis("date").reset_index())
        print(f"score diagnostics complete: {label}/{block}/{model}",flush=True)
        pd.DataFrame(rows).to_csv(output/"score_diagnostics.csv",index=False)
    if dailies:
        pd.concat(dailies,ignore_index=True).to_parquet(output/"daily_score_diagnostics.parquet",index=False)
    # Measure momentum stability directly; turnover does not identify rank autocorrelation.
    if not args.reuse_scores:
        work["score"]=smooth_scores(work,"residual_momentum",5)
        stats,series=paired_score_metrics(work)
        (output/"residual_momentum_ema5.json").write_text(json.dumps(stats,indent=2),encoding="utf-8")
    if not args.skip_coefficients:
        coefficients=recover_coefficients(research,output)
        plt.rcParams.update({"font.family":"Microsoft YaHei","axes.unicode_minus":False})
        selected=coefficients.loc[coefficients.label.eq("cross_sectional")&coefficients.block.eq("price_only")]
        matrix=selected.pivot(index="feature",columns="fold",values="coefficient")
        scale=float(matrix.abs().max().max())
        fig,ax=plt.subplots(figsize=(11,12))
        img=ax.imshow(matrix,aspect="auto",cmap="RdBu_r",vmin=-scale,vmax=scale)
        ax.set_yticks(np.arange(len(matrix)),matrix.index,fontsize=8)
        ax.set_xticks(np.arange(len(matrix.columns)),matrix.columns)
        ax.set_xlabel("外层预测折");ax.set_title("Ridge 37个带符号系数：原标签 / 价量组",loc="left")
        fig.colorbar(img,ax=ax,label="实际拟合系数（统一特征尺度）",shrink=.6)
        fig.tight_layout();fig.savefig(output/"ridge_coefficients.png",dpi=160);plt.close(fig)
    write_run_metadata(output/"run_metadata.json",stage="frozen_oof_score_and_signed_coefficient_diagnostics",
        inputs={"driver":Path(__file__),"factors":factors_path,"training":research/"training_completion.json",
                "diagnostic_implementation":ROOT/"quant_platform/core/evaluation/ml_diagnostics.py"},
        params={"horizon":20,"halflives":[None,5,10,20],"coefficient_source":"exact refit at frozen parameters/sample; OOF equality checked",
                "icir_annualization":"sqrt252 is descriptive only; 20-day overlap is corrected by HAC for inference",
                "selection_use":"descriptive reviewed OOS; never reused for feature deletion"},
        extra={"historical_review_not_untouched_holdout":True,"production_admitted":False})


if __name__=="__main__":main()
