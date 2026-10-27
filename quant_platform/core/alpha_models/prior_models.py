"""Small, training-only models with an explicit economic coefficient prior."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from quant_platform.core.evaluation.factor_diagnostics import hac_summary


@dataclass
class PriorRidge:
    normalized_penalty: float = 1.

    def fit(self, x, y, prior):
        x=np.asarray(x,dtype=float);y=np.asarray(y,dtype=float);prior=np.asarray(prior,dtype=float)
        if x.ndim!=2 or len(y)!=len(x) or prior.shape!=(x.shape[1],) or not np.isfinite(prior).all():
            raise ValueError("Invalid prior regression arrays")
        if self.normalized_penalty<0 or not np.isfinite(self.normalized_penalty):
            raise ValueError("Normalized penalty must be finite and nonnegative")
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError("Regression requires finite training data")
        # Objective: mean squared error + lambda * ||beta-beta0||^2.
        estimator=Ridge(alpha=len(y)*self.normalized_penalty,solver="lsqr",tol=1e-8)
        estimator.fit(x,y-x@prior)
        self.coef_=prior+estimator.coef_
        self.intercept_=float(estimator.intercept_)
        return self

    def predict(self,x):
        return np.asarray(x)@self.coef_+self.intercept_


def full_date_sample(rows, *, stride=5):
    """Deterministically keep full eligible cross sections, never random rows."""
    if stride<1:raise ValueError("Date stride must be positive")
    dates=pd.DatetimeIndex(sorted(rows.date.unique()))
    return rows.date.isin(dates[::stride]).to_numpy()


def prior_coefficients(features, directions):
    missing=set(directions)-set(features)
    if missing:raise ValueError(f"Prior components absent from the feature schema: {sorted(missing)}")
    return np.array([directions.get(name,0)/len(directions) for name in features],dtype=float)


def training_ic_selection(daily_ic, training_dates, *, cutoff, horizon, prior_features,
                          directions, threshold=1.5, duplicates=None,
                          one_day_ic=None, short_horizon_candidates=()):
    """Caller supplies only dates whose forward labels end before cutoff.

    Selection signs come from registered economic hypotheses. A coefficient
    sign change observed in OOS never enters this function.
    """
    dates=pd.DatetimeIndex(training_dates)
    if len(dates)==0 or dates.max()>=pd.Timestamp(cutoff):
        raise ValueError("Feature selection crossed its training cutoff")
    sample=daily_ic.reindex(dates)
    audit=[];selected=[]
    duplicates={} if duplicates is None else duplicates
    for feature in daily_ic.columns:
        aligned=sample[feature]*directions.get(feature,1)
        stats=hac_summary(aligned,horizon)
        reason="training_ic_below_threshold"
        keep=np.isfinite(stats["hac_t"]) and stats["hac_t"]>threshold
        if keep:reason="training_ic"
        if feature in short_horizon_candidates and one_day_ic is not None:
            short=hac_summary(one_day_ic.reindex(dates)[feature]*directions.get(feature,1),1)
            if short["p_value"]<.05 and not keep:reason="short_horizon_only"
        if feature in duplicates:
            keep=False;reason=f"duplicate_of:{duplicates[feature]}"
        if feature in prior_features:
            keep=True;reason="fixed_economic_prior"
        if keep:selected.append(feature)
        audit.append({"feature":feature,"selected":bool(keep),"reason":reason,
            "aligned_train_ic":stats["mean_rank_ic"],"aligned_train_hac_t":stats["hac_t"],
            "train_days":stats["n_days"],"training_cutoff":str(cutoff)})
    if not selected:raise ValueError("No trained or prior features survived selection")
    return selected,pd.DataFrame(audit)
