"""Uniform historical diagnostics, with paired availability and declared signs."""
from __future__ import annotations

import math
import numpy as np
import pandas as pd
from quant_platform.core.alpha_models.nonlinear_alpha import daily_rank_ic
from quant_platform.core.data.industry import neutralize_industry_size


def hac_summary(values, horizon=1):
    """Bartlett HAC on the original calendar; missing days are not compressed."""
    a = pd.Series(values, dtype=float).to_numpy()
    finite = np.isfinite(a)
    n = int(finite.sum())
    if n < 3:
        return {"mean_rank_ic": np.nan, "hac_t": np.nan, "p_value": np.nan, "n_days": n}
    mean = float(a[finite].mean())
    x = np.where(finite, a-mean, 0.)
    lag = min(len(a)-1, max(horizon-1, int(4*(n/100)**(2/9))))
    variance = np.dot(x, x) / n
    for k in range(1, lag+1):
        variance += 2*(1-k/(lag+1))*np.dot(x[k:], x[:-k])/n
    se = np.sqrt(max(variance, 0)/n)
    t = mean/se if se > 0 else np.nan
    return {"mean_rank_ic": mean, "hac_t": t, "p_value": math.erfc(abs(t)/np.sqrt(2)) if np.isfinite(t) else np.nan,
            "n_days": n, "hac_lags": lag, "positive_ic_fraction": float((a[finite]>0).mean())}


def benjamini_hochberg(p_values):
    p = np.asarray(p_values, dtype=float)
    out = np.full(len(p), np.nan)
    ids = np.flatnonzero(np.isfinite(p))
    order = ids[np.argsort(p[ids])]
    q = p[order]*len(order)/np.arange(1, len(order)+1)
    out[order] = np.minimum.accumulate(q[::-1])[::-1].clip(0, 1)
    return out


def daily_signal_weights(values, dates, gross=1.):
    """Only contemporary signals determine sleeve members, never labels."""
    ranks = values.groupby(dates).rank(pct=True)
    top, bottom = ranks.ge(.8), ranks.le(.2)
    top_n, bottom_n = top.groupby(dates).transform("sum"), bottom.groupby(dates).transform("sum")
    w = top.astype(float)/top_n.where(top_n.gt(0)) - bottom.astype(float)/bottom_n.where(bottom_n.gt(0))
    return w.fillna(0)*gross/2


def residualize_baseline(values, dates, baseline):
    residual = pd.Series(np.nan, index=values.index)
    for _, idx in dates.groupby(dates).groups.items():
        y, x = values.loc[idx], baseline.loc[idx]
        valid = y.notna() & x.notna().all(axis=1)
        if valid.sum() < max(10, len(x.columns)+3):
            continue
        design = np.column_stack([np.ones(valid.sum()), x.loc[valid].to_numpy()])
        beta = np.linalg.lstsq(design, y.loc[valid].to_numpy(), rcond=None)[0]
        residual.loc[y.index[valid]] = y.loc[valid].to_numpy() - design@beta
    return residual


def grouped_rank_ic(signal, returns, dates, industry):
    frame=pd.DataFrame({"signal":signal,"returns":returns,"date":dates,"industry":industry}).dropna()
    g=frame.groupby(["date","industry"],observed=True)
    x,y=g.signal.rank(),g.returns.rank()
    codes,keys=pd.factorize(pd.MultiIndex.from_frame(frame[["date","industry"]]),sort=True)
    n=np.bincount(codes).astype(float)
    sx,sy=np.bincount(codes,weights=x),np.bincount(codes,weights=y)
    xx,yy=np.bincount(codes,weights=x*x),np.bincount(codes,weights=y*y)
    xy=np.bincount(codes,weights=x*y)
    denom=np.sqrt(np.maximum(n*xx-sx*sx,0)*np.maximum(n*yy-sy*sy,0))
    out=np.divide(n*xy-sx*sy,denom,out=np.full(len(n),np.nan),where=(n>=5)&(denom>0))
    return pd.Series(out,index=keys.set_names(["date","industry"]))


def factor_diagnostics(panel, catalog, horizons=(1,5,20,60)):
    """Return raw plus financial industry/size ablations on matched rows.

    Summary returns are overlapping gross horizon spreads, NOT executable
    strategy returns. Model admission never uses these full-history scores.
    """
    work = panel.sort_values(["date", "permno"]).reset_index(drop=True)
    dates = work.date
    calendar = pd.DatetimeIndex(sorted(dates.unique()))
    summaries, quality, year_rows, group_rows, exposures, ic_parts = [], [], [], [], [], []
    names = catalog.factor.tolist()
    rank_frame = work[names].groupby(dates).rank(pct=True)
    baseline = rank_frame[[c for c in ("MOM_12_1M", "GROSS_PROFITABILITY") if c in rank_frame]]
    size_rank = work.decision_log_size.astype("float64").groupby(dates).rank(pct=True)
    size_groups = pd.Series(np.select([size_rank.le(.3), size_rank.ge(.7)], ["small", "large"], default="middle"),index=work.index)
    market=work.groupby("date").fwd_1.mean().reindex(calendar)
    past_vol=market.rolling(63,min_periods=42).std().shift(2)
    threshold=past_vol.expanding(min_periods=126).median().shift(1)
    high_vol=past_vol.gt(threshold).where(threshold.notna())
    financial = {"EARNINGS_YIELD","BOOK_TO_MARKET","GROSS_PROFITABILITY","ASSET_GROWTH","ACCRUALS","PIOTROSKI_F8"}
    for spec in catalog.itertuples():
        name, sign = spec.factor, spec.direction
        raw = work[name].astype(float).replace([np.inf, -np.inf], np.nan)*sign
        count = raw.groupby(dates).count()
        unique = raw.groupby(dates).nunique()
        daily_changes = raw.groupby(work.permno).diff()
        quality.append({"factor":name, "coverage":float(raw.notna().mean()),
                        "nonconstant_daily_fraction":float(unique[count.ge(20)].gt(1).mean()),
                        "unchanged_observation_fraction":float(daily_changes.dropna().eq(0).mean()),
                        "direction":sign,"lag_sessions":spec.lag_sessions,"signal_type":spec.signal_type,
                        "fundamentals_strict_pit":False if name in financial else None,
                        "relative_volume_proxy":name in {"TURNOVER_20D","TURNOVER_60D","TURNOVER_ACCEL"}})
        variants = {"raw":raw}
        if name in financial:
            # Fixed contemporaneous clipping, shared by both matched branches.
            low, high = raw.groupby(dates).transform(lambda s:s.quantile(.01)), raw.groupby(dates).transform(lambda s:s.quantile(.99))
            clipped = raw.clip(low, high)
            adj = neutralize_industry_size(clipped, dates, work.decision_industry, work.decision_log_size)
            variants.update(raw_matched=clipped.where(adj.notna()), industry_size=adj)
        for variant, signal in variants.items():
            weights = daily_signal_weights(signal, dates)
            daily_pnl=(weights*work.fwd_1).groupby(dates).sum(min_count=1)
            known=(work.fwd_1.notna()|weights.eq(0)).groupby(dates).all()
            beta_sample=pd.concat([daily_pnl.where(known).rename("pnl"),market.rename("market")],axis=1).dropna()
            market_beta=float(beta_sample.pnl.cov(beta_sample.market)/beta_sample.market.var()) if len(beta_sample)>20 and beta_sample.market.var()>0 else np.nan
            weight_changes = weights.groupby(work.permno).diff().abs()
            turnover = float(weight_changes.groupby(dates).sum().mean()/2)
            if variant == "raw":
                by_industry = pd.DataFrame({"date":dates,"industry":work.decision_industry,"weight":weights}).groupby(["date","industry"], observed=True).weight.sum()
                for industry, net in by_industry.groupby(level="industry"):
                    exposures.append({"factor":name,"industry":industry,"mean_net_weight":float(net.mean()),"mean_abs_net_weight":float(net.abs().mean())})
                if name not in baseline:
                    incremental = residualize_baseline(signal.groupby(dates).rank(pct=True), dates, baseline)
                else:
                    incremental = pd.Series(np.nan, index=work.index)
            for horizon in horizons:
                ret = work[f"fwd_{horizon}"]
                evaluation = pd.DataFrame({"date":dates,"signal":signal,"forward_return":ret})
                ic = daily_rank_ic(evaluation,"signal").reindex(calendar)
                stats = hac_summary(ic,horizon)
                long = (weights.where(weights.gt(0))*ret).groupby(dates).sum(min_count=1)*2
                short = (weights.where(weights.lt(0))*ret).groupby(dates).sum(min_count=1)*2
                coverage = (ret.notna() | weights.eq(0)).groupby(dates).all()
                long, short = long.where(coverage), short.where(coverage)
                record={"factor":name,"variant":variant,"horizon":horizon,**stats,
                        "gross_long_horizon_mean":float(long.mean()),"gross_short_horizon_mean":float(short.mean()),
                        "gross_spread_horizon_mean":float(((long+short)/2).mean()),
                        "daily_signal_turnover":turnover,"labeled_rows":int((signal.notna()&ret.notna()).sum())}
                record["beta_to_equal_weight_universe"] = market_beta
                if variant == "raw":
                    residual_ic=daily_rank_ic(evaluation.assign(signal=incremental),"signal").reindex(calendar)
                    record["incremental_rank_ic_vs_momentum_profitability"] = float(residual_ic.mean())
                summaries.append(record)
                ic_parts.append(ic.rename((name,variant,horizon)))
                for year, values in ic.groupby(ic.index.year):
                    year_rows.append({"factor":name,"variant":variant,"horizon":horizon,"year":year,**hac_summary(values,horizon)})
                if horizon==20:
                    for label, mask in (("nonfinancial",work.decision_financial_or_real_estate.notna()&~work.decision_financial_or_real_estate.eq(True)),
                                        ("small",size_groups.eq("small")),("large",size_groups.eq("large")),
                                        ("high_past_vol",dates.map(high_vol).eq(True)),
                                        ("low_past_vol",dates.map(high_vol).eq(False)&dates.map(past_vol).notna())):
                        sub=evaluation.loc[mask]
                        subic=daily_rank_ic(sub,"signal").reindex(calendar)
                        group_rows.append({"factor":name,"variant":variant,"group":label,**hac_summary(subic,horizon)})
                    industry_ic=grouped_rank_ic(signal,ret,dates,work.decision_industry)
                    for industry, values in industry_ic.groupby(level="industry"):
                        series=values.droplevel("industry").reindex(calendar)
                        if series.notna().sum()>=60:
                            group_rows.append({"factor":name,"variant":variant,"group":industry,**hac_summary(series,horizon)})
        print(f"diagnosed {name}",flush=True)
    summary=pd.DataFrame(summaries)
    summary["exploratory_bh_q"] = benjamini_hochberg(summary.p_value)
    # Pooled daily ranks: descriptive redundancy only, never OOF selection.
    correlations=rank_frame.corr()
    return {"summary":summary,"quality":pd.DataFrame(quality),"yearly":pd.DataFrame(year_rows),
            "groups":pd.DataFrame(group_rows),"industry_exposures":pd.DataFrame(exposures),
            "rank_correlation":correlations,"daily_ic":pd.concat(ic_parts,axis=1)}
