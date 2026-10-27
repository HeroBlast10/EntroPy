"""Audit saved books and replay fixed turnover controls without retraining.

New outputs and a complete source snapshot live in a separate experiment
directory. Existing OOF, factor and execution evidence is read-only.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, replace
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import yaml
from loguru import logger

from quant_platform.core.evaluation.turnover_costs import annotate_recorded_orders, turnover_cost_diagnostics
from quant_platform.core.execution.backtest.vectorized_daily import simulate_execution
from quant_platform.core.execution.backtest.pnl import performance_summary
from quant_platform.core.execution.cost_models.us_equity import CostModel
from quant_platform.core.portfolio.low_turnover import TurnoverPolicy, build_turnover_targets
from quant_platform.core.utils.provenance import fingerprint, write_run_metadata, WRDS_CODE_PATHS

MODEL_BASE = ROOT / "data/wrds/batch2/model_comparison"
FACTOR_BASE = ROOT / "data/wrds/batch2/curve_review_20261023/fixed_factors_daily"
FACTORS = ("MOM_12_1M", "RESID_MOM_12_1M", "PIOTROSKI_F8", "SHARE_TURNOVER_20D")
VARIANTS = {
    "daily": TurnoverPolicy(),
    "tolerance20": TurnoverPolicy(tolerance=.2),
    "buffer30": TurnoverPolicy(exit_fraction=.3),
    "ema5": TurnoverPolicy(ema_halflife=5),
    "ema10": TurnoverPolicy(ema_halflife=10),
    "tranche20": TurnoverPolicy(tranches=20),
    "combined": TurnoverPolicy(exit_fraction=.3, ema_halflife=5, tolerance=.2),
    "combined_tranche20": TurnoverPolicy(exit_fraction=.3, ema_halflife=5, tolerance=.2, tranches=20),
}


def policy_for(variant, strategy):
    return TurnoverPolicy(rebalance_sessions=1 if strategy.startswith("factor/") else 20) if variant == "baseline" else VARIANTS[variant]


def strategy_source(strategy):
    kind, *parts = strategy.split("/")
    if kind == "factor":
        if len(parts) != 1 or parts[0] not in FACTORS:
            raise ValueError(f"Unknown factor strategy {strategy}")
        return FACTOR_BASE / "oof.parquet", "fixed_signal", parts[0], 1
    if kind == "model" and len(parts) == 2:
        if parts[0] not in ("price_only", "price_plus_financial") or parts[1] not in ("ridge", "lightgbm"):
            raise ValueError(f"Unknown model strategy {strategy}")
        return MODEL_BASE / "total_loss/full/h20/oof.parquet", parts[0], parts[1], 20
    raise ValueError(f"Unknown strategy {strategy}")


def load_scores(strategy):
    path, block, signal, horizon = strategy_source(strategy)
    columns = ["date", "ticker", "block", signal]
    # Forward returns are used only for reporting raw/smoothed IC.
    if not strategy.startswith("factor/"):
        columns.append("forward_return")
    rows = pd.read_parquet(path, columns=columns)
    return rows.loc[rows.block.eq(block)].reset_index(drop=True), signal, path


def persist_diagnostics(dest, trades, pnl, capital, **extra):
    daily, sources, stats = turnover_cost_diagnostics(trades, pnl, capital,
        commission_min=extra.get("cost_multiplier", 1.0))
    daily.to_csv(dest / "daily_turnover_costs.csv")
    sources.to_csv(dest / "cost_sources.csv", index=False)
    stats = {**extra, **performance_summary(pnl), **stats}
    for row in sources.itertuples():
        stats[f"{row.source}_annual_cost_bps"] = row.annual_cost_nav_bps
    (dest / "summary.json").write_text(json.dumps(stats, indent=2, default=str), encoding="utf-8")
    return stats


def audit_book(task, output, execution):
    logger.remove()
    path, score_path, block, signal, policy, horizon, stress = task
    t = pd.read_parquet(path / "trades.parquet")
    p = pd.read_parquet(path / "daily_pnl.parquet")
    w = pd.read_parquet(path / "rebalance_weights.parquet")
    scores = pd.read_parquet(score_path, columns=["date", "ticker", "block", signal])
    scores = scores.loc[scores.block.eq(block)].copy()
    t = annotate_recorded_orders(t, w, scores, signal)
    key = f"{policy}/h{horizon}/{block}/{signal}/cost_{stress:g}x"
    dest = Path(output) / "recorded_audit" / key
    dest.mkdir(parents=True, exist_ok=True)
    return persist_diagnostics(dest, t, p, execution["initial_capital"], strategy=key,
                               terminal_policy=policy, cost_multiplier=stress, kind="recorded")


def replay_book(strategy, variant, stress, output, execution, price_path, max_sessions):
    logger.remove()
    policy = policy_for(variant, strategy)
    rows, signal, score_path = load_scores(strategy)
    if max_sessions:
        calendar = sorted(rows.date.unique())[:max_sessions]
        rows = rows.loc[rows.date.isin(calendar)].copy()
    p = pd.read_parquet(price_path, columns=["date", "ticker", "permno", "permco", "adj_close", "close",
                                            "volume", "is_tradable", "total_return", "is_terminal"])
    context = p[["date", "ticker", "permno", "permco"]]
    rows = rows.merge(context, on=["date", "ticker"], validate="one_to_one")
    weights, diagnostics = build_turnover_targets(rows, signal, execution, policy)
    calendar = pd.DatetimeIndex(sorted(p.date.unique()))
    first = max(0, int(calendar.searchsorted(rows.date.min())) - 63)
    p = p.loc[p.date.between(calendar[first], rows.date.max())]
    model = CostModel(max_participation_rate=execution["max_participation_rate"])
    if stress != 1:
        fields = ("commission_per_share", "commission_pct", "commission_min", "slippage_bps", "impact_coeff",
                  "borrow_rate_annual", "sec_fee_rate", "finra_taf_per_share")
        model = replace(model, **{field: getattr(model, field) * stress for field in fields},
                        finra_taf_max=model.finra_taf_max * stress)
    trades, pnl = simulate_execution(weights, p, model, initial_capital=execution["initial_capital"],
        missing_terminal_return="total_loss", return_daily_ledger=True,
        target_tolerance=policy.tolerance, tranche_count=policy.tranches)
    prev = pnl.nav_net.shift().fillna(execution["initial_capital"])
    np.testing.assert_allclose(prev * (1 + pnl.gross_ret) - pnl.total_cost_dollar, pnl.nav_net, rtol=1e-11)
    assert trades.executed_participation.dropna().le(execution["max_participation_rate"] + 1e-12).all()
    gross = weights.groupby("date").weight.agg(lambda s: s.abs().sum())
    assert gross.le(execution.get("gross_target", 1.) + 1e-12).all()
    assert weights.weight.abs().le(execution["max_stock_weight"] + 1e-12).all()
    # Identical targets with all controls off must reproduce the frozen book.
    baseline_difference = None
    if variant == "baseline" and stress == 1:
        source, block, _, _ = strategy_source(strategy)
        recorded = pd.read_parquet(source.parent / "backtests" / block / signal / "cost_1x/daily_pnl.parquet")
        recorded = recorded.reindex(pnl.index)
        baseline_difference = float((pnl.nav_net - recorded.nav_net).abs().max())
        np.testing.assert_allclose(pnl.nav_net, recorded.nav_net, rtol=1e-10, atol=1e-6)
    dest = Path(output) / "replays" / strategy / variant / f"cost_{stress:g}x"
    dest.mkdir(parents=True, exist_ok=True)
    trades.attrs = {}; pnl.attrs = {}
    trades.to_parquet(dest / "trades.parquet", index=False)
    pnl.to_parquet(dest / "daily_pnl.parquet")
    weights.to_parquet(dest / "targets.parquet", index=False)
    diagnostics[["date", "ticker", "selection_score", "selection_rank"]].to_parquet(dest / "scores.parquet", index=False)
    ic = {}
    if "forward_return" in diagnostics:
        ic["raw_rank_ic"] = float(diagnostics.groupby("date")[[signal, "forward_return"]].corr(method="spearman")
                                     .loc[(slice(None), signal), "forward_return"].mean())
        ic["smoothed_rank_ic"] = float(diagnostics.groupby("date")[["selection_score", "forward_return"]].corr(method="spearman")
                                          .loc[(slice(None), "selection_score"), "forward_return"].mean())
    stats = persist_diagnostics(dest, trades, pnl, execution["initial_capital"],
        strategy=strategy, variant=variant, cost_multiplier=stress, kind="replay",
        baseline_max_nav_difference=baseline_difference, **ic)
    (dest / "policy.json").write_text(json.dumps(asdict(policy), indent=2), encoding="utf-8")
    return stats


def snapshot_sources(output):
    sources = [p for p in (ROOT / "quant_platform").rglob("*.py") if "paper" not in p.parts]
    sources += [Path(__file__), ROOT / "config/wrds_batch2.yaml"]
    records = {}
    for source in sources:
        relative = source.relative_to(ROOT)
        dest = output / "source_snapshot" / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, dest)
        records[str(relative)] = fingerprint(source)["sha256"]
    return records


def run_jobs(function, tasks, jobs, label):
    results = []
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        futures = {pool.submit(function, *task): task for task in tasks}
        for number, future in enumerate(as_completed(futures), 1):
            result = future.result()
            results.append(result)
            print(f"{label} {number}/{len(tasks)}: {result['strategy']} {result.get('variant', '')} "
                  f"cost={result.get('cost_multiplier', 1):g}x CAGR={result['net_ann_return']:.3%} "
                  f"one-way={result['daily_one_way_turnover']:.3%}", flush=True)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "data/wrds/batch2/turnover_optimization_20261023")
    parser.add_argument("--phase", choices=["audit", "replay", "all"], default="all")
    parser.add_argument("--strategies", nargs="+")
    parser.add_argument("--variants", nargs="+", choices=["baseline", *VARIANTS], default=["baseline", *VARIANTS])
    parser.add_argument("--cost-multipliers", nargs="+", type=float, default=[1.0])
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--max-sessions", type=int, help="Explicit integration smoke window; never a full research result")
    args = parser.parse_args()
    if args.jobs < 1 or any(not np.isfinite(s) or s <= 0 for s in args.cost_multipliers):
        parser.error("jobs and cost multipliers must be positive")
    if args.max_sessions is not None and args.max_sessions < 1:
        parser.error("max-sessions must be positive")
    # Never replace a completed experiment or silently mix source versions.
    if args.output.exists():
        raise ValueError("Output already exists; choose a fresh --output directory")
    args.output.mkdir(parents=True)
    cfg = yaml.safe_load((ROOT / "config/wrds_batch2.yaml").read_text(encoding="utf-8"))
    execution = cfg["execution"]
    prices = ROOT / cfg["paths"]["prices"]
    strategies = args.strategies or [*[f"factor/{f}" for f in FACTORS],
        *[f"model/{block}/{model}" for block in ("price_only", "price_plus_financial") for model in ("ridge", "lightgbm")]]
    for strategy in strategies:
        strategy_source(strategy)
    snapshots = snapshot_sources(args.output)
    inputs = {"prices": prices, "config": ROOT / "config/wrds_batch2.yaml",
              "factors_oof": FACTOR_BASE / "oof.parquet"}
    for policy in ("total_loss", "zero_return"):
        for horizon in (20, 5):
            inputs[f"{policy}_h{horizon}"] = MODEL_BASE / policy / f"full/h{horizon}/oof.parquet"
    record = write_run_metadata(args.output / "experiment_plan.json", stage="fixed_turnover_controls",
        code_paths=WRDS_CODE_PATHS, inputs=inputs,
        params={"strategies": strategies, "variants": args.variants, "cost_multipliers": args.cost_multipliers,
                "phase": args.phase, "max_sessions": args.max_sessions, "execution": execution,
                "policies": {name: asdict(policy) for name, policy in VARIANTS.items()},
                "primary_control": "combined (buffer30, EMA5, tolerance20); no best-result selection",
                "tranche_initialization": "all sleeves seeded at first close, subsequent one per session",
                "tranche_unfilled": "cancel at close, revisit sleeve next scheduled decision"},
        extra={"source_sha256": snapshots, "is_new_oos_validation": False, "production_admitted": False,
               "smoothing_warmup": "OOF start; no invented earlier predictions", "break_even": "fixed fill-path geometric root"})
    registry = []
    if args.phase in ("audit", "all"):
        tasks = []
        for policy in ("total_loss", "zero_return"):
            for path in sorted((MODEL_BASE / policy / "full").glob("h*/backtests/*/*/cost_*/daily_pnl.parquet")):
                book = path.parent
                horizon = int(book.parents[3].name[1:])
                stress = float(book.name.removeprefix("cost_").removesuffix("x"))
                task = (book, book.parents[3] / "oof.parquet", book.parents[1].name, book.parent.name, policy, horizon, stress)
                tasks.append((task, args.output, execution))
        for factor in FACTORS:
            tasks.append(((FACTOR_BASE / f"backtests/fixed_signal/{factor}/cost_1x", FACTOR_BASE / "oof.parquet",
                           "fixed_signal", factor, "total_loss", 1, 1.0), args.output, execution))
        registry = run_jobs(audit_book, tasks, args.jobs, "audit")
        pd.DataFrame(registry).to_csv(args.output / "recorded_strategy_summary.csv", index=False)
    if args.phase in ("replay", "all"):
        tasks = [(strategy, variant, stress, args.output, execution, prices, args.max_sessions)
                 for strategy in strategies for variant in args.variants for stress in args.cost_multipliers
                 if not (strategy.startswith("factor/") and variant == "daily")]
        replay = run_jobs(replay_book, tasks, args.jobs, "replay")
        pd.DataFrame(replay).to_csv(args.output / "replay_summary.csv", index=False)
        registry += replay
    (args.output / "completion.json").write_text(json.dumps({"completed_books": len(registry),
        "source_sha256": record["source_sha256"], "is_full_window": args.max_sessions is None,
        "failures": 0, "production_admitted": False}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
