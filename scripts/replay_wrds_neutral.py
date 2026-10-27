"""Execute matched label/model books with observable pre/post-trade risk.

Controls use a matched 20-session calendar: original quintiles, industry
quintiles, factor-risk QP, and the same QP with beta/SIC2 neutrality. Only the
QPs use EMA5. Daily optimization is available via --risk-sessions 1.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, replace
import json
from pathlib import Path
import shutil
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from loguru import logger
from threadpoolctl import threadpool_limits

from quant_platform.core.alpha_models.fair_comparison import issuer_capped_weights
from quant_platform.core.portfolio.low_turnover import smooth_scores
from quant_platform.core.portfolio.risk_model import FactorRiskModel
from quant_platform.core.portfolio.risk_neutral import NeutralRiskConfig, industry_quantile_weights, solve_factor_neutral
from quant_platform.core.execution.backtest.vectorized_daily import simulate_execution
from quant_platform.core.execution.backtest.pnl import performance_summary
from quant_platform.core.execution.cost_models.us_equity import CostModel
from quant_platform.core.evaluation.turnover_costs import turnover_cost_diagnostics
from quant_platform.core.utils.provenance import fingerprint, write_run_metadata

METHODS = ("quantile_20d", "industry_20d", "risk_dollar", "risk_neutral")
STATE = {}


def initialize(research):
    logger.remove()
    research = Path(research)
    cache = research / "risk_cache"
    schema = json.loads((cache / "schema.json").read_text())
    exposures = pd.read_parquet(cache / "exposures.parquet")
    all_context = pd.read_parquet(cache / "all_context.parquet")
    eligible = pd.read_parquet(cache / "eligible_context.parquet")
    matrices = np.load(cache / "covariance.npz")
    STATE.update(research=research, schema=schema,
        exposures={date: group.set_index("ticker") for date, group in exposures.groupby("date", sort=False)},
        context={date: group.set_index("ticker") for date, group in all_context.groupby("date", sort=False)},
        eligible={date: group.set_index("ticker") for date, group in eligible.groupby("date", sort=False)},
        covariance={pd.Timestamp(date): cov for date, cov in zip(matrices["dates"], matrices["covariance"])})
    prices = pd.read_parquet(ROOT / "data/wrds/p0/prices.parquet", columns=["date", "ticker", "permno", "adj_close", "close",
        "volume", "is_tradable", "total_return", "is_terminal"])
    dates = sorted(STATE["exposures"])
    history = pd.DatetimeIndex(sorted(prices.date.unique()))
    first = max(0, int(history.searchsorted(dates[0])) - 63)
    STATE["prices"] = prices.loc[prices.date.between(history[first], dates[-1])].copy()


class PortfolioPolicy:
    def __init__(self, method, risk_config, execution, label):
        self.method, self.config, self.execution, self.label = method, risk_config, execution, label
        self.held_target = pd.Series(dtype=float)
        self.day = 0
        self.model = None
        self.warm_state = {}

    def load_date(self, date):
        frame = STATE["exposures"][date]
        model = FactorRiskModel()
        model.factor_names_ = STATE["schema"]["factor_names"]
        model.exposures_ = frame[model.factor_names_]
        model.specific_risk_ = frame.risk_specific_vol
        model.cov_matrix_ = STATE["covariance"][date]
        self.model, self.beta = model, frame.risk_single_market_beta
        self.context = STATE["context"][date]

    def __call__(self, *, date, rows, actual_weights, rebalance):
        self.load_date(date)
        self.day += 1
        diagnostics = {}
        if rebalance:
            # Common ex-ante coverage across all four construction methods.
            context = STATE["eligible"][date]
            keys = pd.Index(rows.ticker).intersection(context.index).intersection(self.model.exposures_.index)
            context = context.reindex(keys).dropna(subset=["decision_industry", "permno"])
            day = context.reset_index().merge(rows[["ticker", "prediction", "smoothed_prediction"]],
                                             on="ticker", validate="one_to_one")
            if len(day) < self.execution["min_names"]:
                raise ValueError("Insufficient common risk-covered decision universe")
            if self.method == "quantile_20d":
                weight = issuer_capped_weights(day, "prediction", min_names=self.execution["min_names"],
                    stock_cap=self.execution["max_stock_weight"], issuer_cap=self.execution["max_issuer_gross_weight"])
                target = pd.Series(weight.to_numpy(), index=day.ticker)
            elif self.method == "industry_20d":
                weight = industry_quantile_weights(day, "prediction", stock_limit=self.execution["max_stock_weight"],
                                                    issuer_limit=self.execution["max_issuer_gross_weight"])
                target = pd.Series(weight.to_numpy(), index=day.ticker)
            else:
                indexed = day.set_index("ticker")
                issuer = indexed.permco.fillna(-indexed.permno)
                try:
                    target, diagnostics = solve_factor_neutral(indexed.smoothed_prediction, self.model,
                        indexed.decision_industry, issuer, self.beta, actual_weights, self.config, self.warm_state)
                except Exception as error:
                    raise RuntimeError(f"Risk optimization on {date}: {error}") from error
            self.held_target = target.astype(float)
            diagnostics["risk_covered_names"] = len(day)
            diagnostics["target_gross"] = float(target.abs().sum())
            if "predicted_volatility" not in diagnostics:
                diagnostics["predicted_volatility"] = self.model.decompose_risk(target)["total_risk"]
        if self.day % 1000 == 0:
            print(f"{self.label}/{self.method} session {self.day}: {date.date()}", flush=True)
        return self.held_target.copy(), diagnostics

    def diagnose(self, date, actual):
        actual = actual.astype(float)
        modeled = actual.index.intersection(self.model.exposures_.index)
        unknown = float(actual.reindex(actual.index.difference(modeled)).abs().sum())
        sector = self.context.decision_industry.reindex(actual.index)
        unknown_industry = float(actual.loc[sector.isna()].abs().sum())
        sector_exposure = actual.groupby(sector).sum()
        known = actual.reindex(modeled)
        decomp = self.model.decompose_risk(known)
        result = {"actual_beta": float(known @ self.beta.reindex(modeled)) if unknown < 1e-8 else np.nan,
                  "unmodelled_gross": unknown, "unknown_industry_gross": unknown_industry,
                  "actual_industry_abs_net": float(sector_exposure.abs().sum()) if unknown_industry < 1e-8 else np.nan,
                  "actual_industry_max_net": float(sector_exposure.abs().max()) if len(sector_exposure) else 0.,
                  "actual_predicted_volatility": decomp["total_risk"] if unknown < 1e-8 else np.nan}
        for name, value in decomp["portfolio_exposures"].items():
            result[f"factor_exposure:{name}"] = value
        return result


def execute(task, output, execution, risk):
    label, block, model, method, stress = task
    cohort = STATE["research"] / f"models/{label}/{block}/{model}"
    scores = pd.read_parquet(cohort / "oof.parquet", columns=["date", "ticker", "prediction"])
    scores["smoothed_prediction"] = smooth_scores(scores, "prediction", risk["ema_halflife"])
    calendar = pd.DatetimeIndex(sorted(scores.date.unique()))
    sessions = 20 if method.endswith("20d") else risk["rebalance_sessions"]
    rebalance = set(calendar[::sessions])
    scores["weight"] = 0.
    scores["is_rebalance"] = scores.date.isin(rebalance)
    config = NeutralRiskConfig(gross_limit=risk["gross_limit"], stock_limit=execution["max_stock_weight"],
        issuer_limit=execution["max_issuer_gross_weight"], expected_alpha_annual=risk["expected_alpha_annual"],
        risk_aversion=risk["risk_aversion"], turnover_cost_bps=risk["turnover_cost_bps"],
        forecast_horizon=risk["forecast_horizon"], target_volatility=risk["target_volatility"], neutral=method == "risk_neutral")
    policy = PortfolioPolicy(method, config, execution, f"{label}/{block}/{model}/cost_{stress:g}x")
    costs = CostModel(max_participation_rate=execution["max_participation_rate"])
    if stress != 1:
        fields = ("commission_per_share", "commission_pct", "commission_min", "slippage_bps", "impact_coeff",
                  "borrow_rate_annual", "sec_fee_rate", "finra_taf_per_share")
        costs = replace(costs, **{field: getattr(costs, field) * stress for field in fields},
                        finra_taf_max=costs.finra_taf_max * stress)
    with threadpool_limits(limits=1):
        trades, pnl = simulate_execution(scores, STATE["prices"], costs,
            initial_capital=execution["initial_capital"], missing_terminal_return="total_loss",
            return_daily_ledger=True, target_policy=policy)
    targets = trades.attrs["target_history"]
    daily, sources, diagnostics = turnover_cost_diagnostics(trades, pnl, execution["initial_capital"],
                                                           commission_min=stress)
    previous = pnl.nav_net.shift().fillna(execution["initial_capital"])
    np.testing.assert_allclose(previous * (1 + pnl.gross_ret) - pnl.total_cost_dollar, pnl.nav_net, rtol=1e-11)
    assert trades.executed_participation.dropna().le(execution["max_participation_rate"] + 1e-12).all()
    assert trades.shares.eq(np.floor(trades.shares)).all()
    assert trades.loc[trades.shares.eq(0), "total_cost"].eq(0).all()
    result = {"label": label, "block": block, "model": model, "method": method, "cost_multiplier": stress,
        **performance_summary(pnl), **diagnostics,
        "actual_beta_abs_p95": float(pnl.risk_actual_beta.abs().quantile(.95)),
        "actual_beta_abs_max": float(pnl.risk_actual_beta.abs().max()),
        "actual_industry_abs_net_p95": float(pnl.risk_actual_industry_abs_net.quantile(.95)),
        "max_unmodelled_gross": float(pnl.risk_unmodelled_gross.max()),
        "mean_predicted_volatility": float(pnl.risk_actual_predicted_volatility.mean())}
    dest = output / f"{label}/{block}/{model}/{method}/cost_{stress:g}x"
    dest.mkdir(parents=True, exist_ok=True)
    trades.attrs = {}; pnl.attrs = {}
    trades.to_parquet(dest / "trades.parquet", index=False)
    targets.to_parquet(dest / "targets.parquet", index=False)
    pnl.to_parquet(dest / "daily_pnl.parquet")
    daily.to_csv(dest / "daily_turnover_costs.csv")
    sources.to_csv(dest / "cost_sources.csv", index=False)
    (dest / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("research", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--jobs", type=int, default=6)
    parser.add_argument("--risk-sessions", type=int, default=20)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--cost-multipliers", nargs="+", type=float, default=[1., 2.])
    parser.add_argument("--cohort", nargs=3, metavar=("LABEL", "BLOCK", "MODEL"))
    args = parser.parse_args()
    if args.risk_sessions < 1:
        parser.error("risk-sessions must be positive")
    research = args.research.resolve()
    output = (args.output or research / "risk_replays").resolve()
    if output.exists():
        raise ValueError("Choose a fresh replay output directory")
    plan = json.loads((research / "plan.json").read_text())
    training = json.loads((research / "training_completion.json").read_text())
    import yaml
    base = yaml.safe_load((ROOT / plan["config"]["base_config"]).read_text(encoding="utf-8"))
    execution = base["execution"]
    risk = {**plan["config"]["risk"], "rebalance_sessions": args.risk_sessions}
    cohorts = [(c["label"], c["block"], c["model"]) for c in training["cohorts"]]
    if args.cohort:
        cohorts = [tuple(args.cohort)]
        if cohorts[0] not in [(c["label"], c["block"], c["model"]) for c in training["cohorts"]]:
            raise ValueError("Requested cohort was not trained")
    tasks = [(*cohort, method, stress) for cohort in cohorts for method in args.methods for stress in args.cost_multipliers]
    output.mkdir(parents=True)
    source = {}
    for file in [Path(__file__), *list((ROOT / "quant_platform").rglob("*.py"))]:
        if "paper" in file.parts:
            continue
        relative = file.relative_to(ROOT)
        dest = output / "source_snapshot" / relative
        dest.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(file, dest)
        source[str(relative)] = fingerprint(file)["sha256"]
    write_run_metadata(output / "experiment_plan.json", stage="long_history_neutral_portfolios",
        inputs={"training": research / "training_completion.json", "plan": research / "plan.json",
                "risk_cache": research / "risk_cache/run_metadata.json", "base_config": ROOT / plan["config"]["base_config"]},
        params={"tasks": tasks, "risk": risk, "execution": execution, "source_sha256": source,
                "dollar_vs_neutral": "identical calendar/factor-risk/EMA5/cost objectives; neutrality differs",
                "quantile_vs_industry": "same raw scores, 20-session calendar and common ex-ante risk coverage"},
        extra={"historical_review_not_untouched_holdout": True, "smoke": plan["smoke"], "production_admitted": False})
    results, errors = [], []
    with ProcessPoolExecutor(max_workers=args.jobs, initializer=initialize, initargs=(research,)) as pool:
        futures = {pool.submit(execute, task, output, execution, risk): task for task in tasks}
        for future in as_completed(futures):
            try:
                result = future.result()
                results.append(result)
                pd.DataFrame(results).to_csv(output / "performance.csv", index=False)
                print(f"completed {len(results)}/{len(tasks)} {futures[future]} CAGR={result['net_ann_return']:.3%} "
                      f"beta_p95={result['actual_beta_abs_p95']:.4f}", flush=True)
            except Exception as error:
                errors.append({"task": futures[future], "error": repr(error), "traceback": traceback.format_exc()})
                print(f"FAILED {futures[future]}: {error}", flush=True)
                (output / "failures.json").write_text(json.dumps(errors, indent=2), encoding="utf-8")
    (output / "completion.json").write_text(json.dumps({"completed": len(results), "expected": len(tasks),
        "failures": errors, "smoke": plan["smoke"]}, indent=2), encoding="utf-8")
    if errors:
        raise RuntimeError("Some books failed; failures are preserved in failures.json")


if __name__ == "__main__":
    main()
