"""Effective-dated CRSP SIC history and simultaneous industry/size controls."""
import numpy as np
import pandas as pd


def historical_industries(prices, history):
    history = history.copy().sort_values(["permno", "effective_from"])
    if history.duplicated(["permno", "effective_from"]).any():
        raise ValueError("Ambiguous SIC effective-date history")
    end = history.effective_to.fillna(pd.Timestamp.max.normalize())
    previous_end = end.groupby(history.permno).cummax().groupby(history.permno).shift(1)
    if history.effective_from.le(previous_end).any() or history.effective_from.gt(end).any():
        raise ValueError("Overlapping or reversed SIC intervals")
    joined = pd.merge_asof(prices[["date", "permno"]].sort_values("date"),
                           history.sort_values("effective_from"), by="permno", left_on="date",
                           right_on="effective_from", direction="backward")
    valid = joined.effective_to.isna() | joined.date.le(joined.effective_to)
    sic = pd.to_numeric(joined.siccd, errors="coerce").where(valid)
    sic = sic.where(sic.between(100, 9799))
    joined["siccd"] = sic.astype("Int64")
    joined["industry"] = (sic // 100).astype("Int64").astype("string").radd("SIC2:")
    joined["financial_or_real_estate"] = sic.between(6000, 6799).where(sic.notna())
    joined["classification_source"] = "crsp.stksecurityinfohist.siccd"
    joined["classification_effective_dated"] = True
    joined["classification_is_point_in_time"] = False  # no historical ingestion/publication vintage
    return joined


def neutralize_industry_size(values, dates, industry, log_size, *, min_group=5):
    """Joint OLS residual on industry dummies + log size within each date.

    Within-group demeaning of BOTH y and x before regression is equivalent
    to the joint projection. Sequentially demeaning y then raw-size OLS can
    reintroduce industry exposure. Unknown/small groups stay missing.
    """
    frame = pd.DataFrame({"y": values, "date": dates, "industry": industry, "x": log_size})
    frame = frame.replace([np.inf, -np.inf], np.nan)
    usable = frame[["y", "x", "industry"]].notna().all(axis=1)
    frame.loc[~usable, ["y", "x"]] = np.nan
    groups = frame.groupby(["date", "industry"], observed=True)
    good = groups.y.transform("count").ge(min_group)
    yw = (frame.y - groups.y.transform("mean")).where(good)
    xw = (frame.x - groups.x.transform("mean")).where(good)
    numerator = (yw*xw).groupby(frame.date).transform("sum")
    denominator = xw.pow(2).groupby(frame.date).transform("sum")
    beta = numerator / denominator.where(denominator.gt(1e-12))
    return yw - beta.fillna(0)*xw
