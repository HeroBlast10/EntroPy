"""Strictly prior-date risk estimates using historical membership/classification."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from loguru import logger
from threadpoolctl import threadpool_limits

from quant_platform.core.alpha_models.nonlinear_alpha import filter_tradable_universe
from quant_platform.core.portfolio.risk_model import FactorRiskModel
from quant_platform.core.utils.provenance import fingerprint, write_run_metadata


def build_cache(research):
    logger.remove()
    plan = json.loads((research / "plan.json").read_text())
    output = research / "risk_cache"
    if output.exists():
        raise ValueError("Risk cache already exists; do not overwrite evidence")
    config = {"specific_risk_floor": .01, **plan["config"]["risk"]}
    prices_path = ROOT / "data/wrds/p0/prices.parquet"
    factors_path = ROOT / "data/wrds/batch2/factors.parquet"
    universe_path = ROOT / "data/wrds/p0/universe.parquet"
    p = pd.read_parquet(prices_path, columns=["date", "ticker", "total_return", "is_terminal", "company_market_cap"])
    rows = pd.read_parquet(factors_path, columns=["date", "ticker", "permno", "permco", "decision_industry", "decision_log_size"])
    universe = pd.read_parquet(universe_path)
    eligible = filter_tradable_universe(rows, universe)
    mask = eligible.assign(member=True).pivot(index="date", columns="ticker", values="member")
    # Same explicit terminal assumption as the execution/label experiment.
    missing = p.is_terminal.fillna(False) & p.total_return.isna()
    p.loc[missing, "total_return"] = -1.
    returns = p.pivot(index="date", columns="ticker", values="total_return").sort_index()
    caps = p.pivot(index="date", columns="ticker", values="company_market_cap").reindex_like(returns)
    membership = mask.reindex_like(returns).fillna(False).astype(bool)
    factors = FactorRiskModel.build_structural_factors(returns, caps, eligibility=membership)
    classification = rows.pivot(index="date", columns="ticker", values="decision_industry").reindex_like(returns)
    division = classification.apply(lambda col: pd.to_numeric(col.str.removeprefix("SIC2:"), errors="coerce") // 10)
    # Broad SIC divisions supply stable factor histories; hard portfolio
    # neutrality is imposed separately at the finer SIC2 level.
    for group in range(2, 9):
        factors[f"sector:SIC1:{group}"] = returns.where(membership & division.eq(group)).mean(axis=1) - factors.market
    folds = json.loads((research / "folds.json").read_text())
    dates = pd.DatetimeIndex([date for fold in folds for date in fold["validation_dates"]])
    output.mkdir()
    snapshot = output / "source_snapshot"
    snapshot.mkdir()
    for source in (Path(__file__), ROOT / "quant_platform/core/portfolio/risk_model.py"):
        shutil.copy2(source, snapshot / source.name)
    eligible.loc[eligible.date.isin(dates)].to_parquet(output / "eligible_context.parquet", index=False)
    rows.loc[rows.date.isin(dates)].to_parquet(output / "all_context.parquet", index=False)
    exposures, covariance, audit = [], [], []
    first = int(returns.index.searchsorted(dates[0]))
    with threadpool_limits(limits=1):
        for number, date in enumerate(dates):
            stop = int(returns.index.searchsorted(date))
            history = returns.iloc[max(0, stop - config["lookback"]):stop]
            past_factors = factors.reindex(history.index)
            assert history.index.max() < date
            model = FactorRiskModel(halflife=config["halflife"], shrinkage=config["shrinkage"],
                                    min_observations=config["min_observations"])
            model.fit(history, past_factors)
            if model.exposures_.empty:
                raise ValueError(f"No risk estimates on {date}")
            model.specific_risk_ = model.specific_risk_.clip(lower=config["specific_risk_floor"])
            names = model.exposures_.index
            y = history[names].to_numpy(dtype=float)
            x = past_factors.market.to_numpy(dtype=float)[:, None]
            valid = np.isfinite(y) & np.isfinite(x)
            n = valid.sum(axis=0)
            xv, yv = np.where(valid, x, 0), np.where(valid, y, 0)
            sx, sy = xv.sum(axis=0), yv.sum(axis=0)
            denominator = (xv ** 2).sum(axis=0) - sx ** 2 / n
            beta = ((xv * yv).sum(axis=0) - sx * sy / n) / denominator
            if not np.isfinite(beta).all():
                raise ValueError("Non-finite single-market betas")
            frame = model.exposures_.assign(risk_specific_vol=model.specific_risk_,
                                             risk_single_market_beta=beta, date=date).reset_index()
            exposures.append(frame)
            covariance.append(model.cov_matrix_)
            w = pd.Series(1 / len(names), index=names)
            decomp = model.decompose_risk(w)
            audit.append({"date": date, "latest_risk_observation": history.index.max(),
                          "names": len(names), "factor_complete_rows": int(past_factors.notna().all(axis=1).sum()),
                          "euler_error": decomp["euler_variance_error"]})
            if (number + 1) % 252 == 0 or number + 1 == len(dates):
                print(f"risk cache {number+1}/{len(dates)}: {date.date()}", flush=True)
    pd.concat(exposures, ignore_index=True).to_parquet(output / "exposures.parquet", index=False)
    np.savez_compressed(output / "covariance.npz", dates=dates.asi8, covariance=np.asarray(covariance))
    factors.to_parquet(output / "factor_returns.parquet")
    pd.DataFrame(audit).to_csv(output / "risk_audit.csv", index=False)
    (output / "schema.json").write_text(json.dumps({"factor_names": list(factors), "config": config,
        "dates": len(dates), "risk_industry": "historical SIC1 divisions 2..8, reference division 1/other",
        "neutrality_industry": "decision-date historical SIC2",
        "classification_is_point_in_time": False}, indent=2), encoding="utf-8")
    write_run_metadata(output / "run_metadata.json", stage="prior_date_factor_risk_cache",
        inputs={"prices": prices_path, "factors": factors_path, "universe": universe_path,
                "driver": Path(__file__), "risk_model": ROOT / "quant_platform/core/portfolio/risk_model.py",
                "plan": research / "plan.json"}, params=config,
        extra={"classification_is_point_in_time": False, "missing_terminal_return": "total_loss",
               "imputed_terminal_rows": int(missing.sum()), "max_euler_error": float(pd.DataFrame(audit).euler_error.abs().max())})
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("research", type=Path)
    args = parser.parse_args()
    print(build_cache(args.research.resolve()))
