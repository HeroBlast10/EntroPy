"""Nested chronological Ridge/LightGBM comparison on identical OOF keys.

Scores are dimensionless ranks, not expected-return estimates. Hyperparameters
are selected using inner dates; outer labels are used only for reporting.
"""
from __future__ import annotations
from dataclasses import replace
import hashlib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from .nonlinear_alpha import (LightGBMConfig,WalkForwardConfig,expanding_date_folds,
    _fit_predict_lightgbm,_daily_zscore,daily_rank_ic,_sample_training_rows)
from .feature_schema import is_metadata


def inner_time_masks(panel, fold, inner_days):
    outer_dates=fold["train_dates"]
    if len(outer_dates)<=inner_days:
        raise ValueError("Insufficient outer training history for nested validation")
    first=outer_dates[-inner_days]
    end=pd.to_datetime(panel.label_end)
    dates=pd.to_datetime(panel.date)
    finite=panel.target.notna()
    train=dates.isin(outer_dates)&dates.lt(first)&end.lt(first)&finite
    valid=dates.isin(outer_dates)&dates.ge(first)&end.lt(fold["validation_start"])&finite
    if not train.any() or not valid.any():
        raise ValueError("Empty purged inner training/validation sample")
    return train.to_numpy(),valid.to_numpy()


def fit_predict(panel, features, train_mask, test_mask, *, model, params, seeds,
                max_rows=200000, n_jobs=4, sampling_features=None, sampling_tiebreak_features=None,
                sampling_reference_col="target", training_indices=None):
    sampling_features=list(features if sampling_features is None else sampling_features)
    tiebreak=list(sampling_features if sampling_tiebreak_features is None else sampling_tiebreak_features)
    if any(is_metadata(c) for c in [*features,*sampling_features,*tiebreak]):
        raise ValueError("Identifiers/provenance cannot be prediction features")
    if not set(features).issubset(set(sampling_features)|set(tiebreak)):
        raise ValueError("Shared sampling needs common tie-break features covering every predictor")
    ids=np.flatnonzero(train_mask)
    # All information blocks use the same secondary keys. After price/target
    # ties, differing financial rows must not fall back to ticker/input order.
    # A remaining exact tie has identical predictors and target in every block.
    sort_keys=list(dict.fromkeys(["date",*sampling_features,sampling_reference_col,*tiebreak]))
    if training_indices is None:
        canonical=panel.iloc[ids][sort_keys].reset_index(drop=True).sort_values(
            sort_keys,kind="mergesort").index
        ids=_sample_training_rows(ids[canonical],max_rows,int(seeds[0]))
    else:
        chosen=np.asarray(training_indices)
        if chosen.dtype.kind not in "iu" or (chosen<0).any() or (chosen>=len(panel)).any():
            raise ValueError("Invalid predeclared training indices")
        if len(np.unique(chosen))!=len(chosen) or not np.asarray(train_mask)[chosen].all():
            raise ValueError("Predeclared training rows violate the purged mask")
        if max_rows is not None and len(chosen)>max_rows:
            raise ValueError("Predeclared training rows exceed the sampling budget")
        ids=chosen
    test_ids=np.flatnonzero(test_mask)
    x=panel.iloc[ids][features].to_numpy(dtype=np.float32)
    y=panel.iloc[ids].target.to_numpy(dtype=np.float32)
    xt=panel.iloc[test_ids][features].to_numpy(dtype=np.float32)
    if len(ids)<20 or not len(test_ids):
        raise ValueError("Too few usable training/test rows")
    if model=="ridge":
        estimator=Ridge(alpha=float(params["alpha"]),solver="lsqr",tol=1e-6)
        prediction=estimator.fit(x,y).predict(xt)
        importance=pd.Series(np.abs(estimator.coef_),index=features)
    elif model=="lightgbm":
        config=replace(LightGBMConfig(n_jobs=n_jobs),**params)
        values,importances=[],[]
        for seed in seeds:
            pred,imp=_fit_predict_lightgbm(x,y,xt,config,int(seed),features)
            values.append(pred); importances.append(imp)
        prediction=np.mean(values,axis=0)
        importance=pd.concat(importances,axis=1).mean(axis=1)
    else:
        raise ValueError("model must be ridge or lightgbm")
    keys=panel.iloc[ids][["date","ticker"]].astype(str).to_csv(index=False).encode()
    importance.attrs["training_key_sha256"]=hashlib.sha256(keys).hexdigest()
    return prediction.astype(np.float32),importance,len(ids)


def nested_oof(panel, feature_blocks, directions, settings, *, horizon, frozen_choices=None):
    wf=WalkForwardConfig(horizon=horizon,min_train_days=settings["min_train_days"],
        validation_days=settings["validation_days"],n_splits=settings["n_splits"],
        oos_start=settings.get("oos_start"))
    trials,choices,parts,importance_rows,fold_rows=[],{},[],[],[]
    label_end=pd.to_datetime(panel.label_end)
    common_tiebreak=list(dict.fromkeys(c for cols in feature_blocks.values() for c in cols))
    for fold in expanding_date_folds(panel.date,wf):
        predict=panel.date.isin(fold["validation_dates"]).to_numpy()
        train=(panel.date.isin(fold["train_dates"])&panel.target.notna()&label_end.lt(fold["validation_start"])).to_numpy()
        inner_train,inner_valid=inner_time_masks(panel,fold,settings["inner_days"])
        fold_rows.append({"horizon":horizon,"fold":fold["fold"],
            "validation_start":str(fold["validation_start"].date()),"validation_end":str(fold["validation_end"].date()),
            "max_training_label_end":str(label_end[train].max().date()),
            "max_inner_training_label_end":str(label_end[inner_train].max().date()),
            "inner_validation_start":str(panel.loc[inner_valid,"date"].min().date()),
            "training_rows_available":int(train.sum()),"prediction_rows":int(predict.sum())})
        base=panel.loc[predict,["date","ticker","forward_return","target","label_end","label_terminal_imputed"]].copy()
        for block, features in feature_blocks.items():
            local=base.copy()
            local["block"],local["horizon"],local["fold"]=block,horizon,fold["fold"]
            oriented=panel.loc[predict,features].mul(pd.Series({c:directions.get(c,1) for c in features}))
            local["equal_weight"]=oriented.mean(axis=1)
            for baseline in ("MOM_12_1M","GROSS_PROFITABILITY"):
                if baseline in panel:
                    local[baseline]=panel.loc[predict,baseline]*directions.get(baseline,1)
            for model in ("ridge","lightgbm"):
                key=f"{fold['fold']}:{block}:{model}"
                if frozen_choices is None:
                    scores=[]
                    for config_id,params in enumerate(settings["model_configs"][model]):
                        pred,imp,n=fit_predict(panel,features,inner_train,inner_valid,model=model,params=params,
                            seeds=settings["seeds"],max_rows=settings["max_train_rows"],n_jobs=settings["n_jobs"],
                            sampling_features=settings.get("sampling_features"),sampling_tiebreak_features=common_tiebreak)
                        validation=panel.loc[inner_valid,["date","forward_return"]].assign(score=pred)
                        score=float(daily_rank_ic(validation,"score").mean())
                        trials.append({"horizon":horizon,"fold":fold["fold"],"block":block,"model":model,
                            "config_id":config_id,"params":params,"seeds":settings["seeds"],"inner_rank_ic":score,
                            "train_rows_used":n,"selected_using":"inner labels only"})
                        trials[-1]["training_key_sha256"]=imp.attrs["training_key_sha256"]
                        scores.append(score if np.isfinite(score) else -np.inf)
                    if max(scores)==-np.inf:
                        raise ValueError("No evaluable inner candidate")
                    winner=int(np.argmax(scores))
                    params=settings["model_configs"][model][winner]
                else:
                    params=frozen_choices[key]
                    trials.append({"horizon":horizon,"fold":fold["fold"],"block":block,"model":model,
                                   "params":params,"seeds":settings["seeds"],"selected_using":"frozen primary-horizon choice"})
                choices[key]=params
                pred,imp,n=fit_predict(panel,features,train,predict,model=model,params=params,
                    seeds=settings["seeds"],max_rows=settings["max_train_rows"],n_jobs=settings["n_jobs"],
                    sampling_features=settings.get("sampling_features"),sampling_tiebreak_features=common_tiebreak)
                trials.append({"horizon":horizon,"fold":fold["fold"],"block":block,"model":model,
                    "params":params,"selected_using":"outer training refit; no outer labels in selection",
                    "train_rows_used":n,"training_key_sha256":imp.attrs["training_key_sha256"],
                    "seeds":settings["seeds"] if model=="lightgbm" else [None]})
                local[model]=_daily_zscore(pd.Series(pred,index=local.index),local.date).astype("float32")
                for feature,value in imp.items():
                    importance_rows.append({"horizon":horizon,"fold":fold["fold"],"block":block,"model":model,"feature":feature,"importance":value})
                print(f"h={horizon} fold={fold['fold']} {block}/{model}: train {n:,}, predict {len(local):,}",flush=True)
            parts.append(local)
    oof=pd.concat(parts,ignore_index=True)
    # A fixed, unselected placebo; it is registered separately from candidates.
    rng=np.random.default_rng(settings["seeds"][0])
    oof["ridge_placebo"]=oof.groupby(["block","date"],sort=True).ridge.transform(
        lambda s:rng.permutation(s.to_numpy()))
    return oof,trials,choices,pd.DataFrame(importance_rows),pd.DataFrame(fold_rows)


def issuer_capped_weights(day, score, *, min_names=100, stock_cap=.015, issuer_cap=.03):
    """Symmetric gross <=1 quintile targets with explicit issuer concentration."""
    if day[score].notna().sum()<min_names:
        return pd.Series(0.,index=day.index)
    rank=day[score].rank(pct=True)
    long,short=rank.ge(.8),rank.le(.2)
    weights=(long.astype(float)/max(int(long.sum()),1)-short.astype(float)/max(int(short.sum()),1))*.5
    weights=weights.clip(-stock_cap,stock_cap)
    issuer=day.permco.fillna(-day.permno)
    gross=weights.abs().groupby(issuer).transform("sum")
    weights*=np.minimum(1,issuer_cap/gross.replace(0,np.nan)).fillna(1)
    pos,neg=weights.clip(lower=0).sum(),-weights.clip(upper=0).sum()
    sleeve=min(pos,neg)
    weights.loc[weights.gt(0)]*=sleeve/pos if pos else 0
    weights.loc[weights.lt(0)]*=sleeve/neg if neg else 0
    return weights
