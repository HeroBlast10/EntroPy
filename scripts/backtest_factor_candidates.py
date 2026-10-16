#!/usr/bin/env python
"""Run and preserve comparable candidate-factor backtests."""

from __future__ import annotations

import os
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from quant_platform.core.execution.backtest.pipeline import run_trading_pipeline
from quant_platform.core.execution.cost_models.us_equity import CostModel
from quant_platform.core.data.benchmark import load_benchmark
from quant_platform.core.data.sectors import ensure_sector_map
from quant_platform.core.evaluation.benchmark_analytics import benchmark_analysis
from quant_platform.core.portfolio.construction import PortfolioConfig, PortfolioMode, WeightScheme
from quant_platform.core.portfolio.pipeline import _compute_avg_turnover, run_portfolio_pipeline
from quant_platform.core.utils.io import set_project_root


CANDIDATES = ["MOM_12_1M", "STR_1W", "TURNOVER_ACCEL"]


def refresh_capacity(output_root: Path) -> None:
    """Recompute analytics from preserved fills without reusing stale p95s."""
    from quant_platform.core.evaluation.capacity import capacity_analysis
    from quant_platform.core.utils.provenance import write_run_metadata

    summary_path = output_root / "candidate_summary.csv"
    summary = pd.read_csv(summary_path).set_index("factor")
    summary = summary.loc[:, ~summary.columns.str.startswith("capacity_")]
    capacity_rows = []
    inputs = {"capacity_source": ROOT / "quant_platform/core/evaluation/capacity.py", "runner_source": Path(__file__)}
    for factor in summary.index:
        factor_dir = output_root / factor
        meta = json.loads((factor_dir / "backtest_metadata.json").read_text(encoding="utf-8"))
        model = CostModel(**meta["cost_model"])
        capital = float(meta["initial_capital"])
        trades_path = factor_dir / "trades.parquet"
        pnl_path = factor_dir / "daily_pnl.parquet"
        capacity = capacity_analysis(pd.read_parquet(trades_path), pd.read_parquet(pnl_path),
                                     cost_model=model, capital_grid=[capital, capital*5, capital*10, capital*50])
        row = capacity["summary"].iloc[0].to_dict()
        limit = model.max_participation_rate
        if limit is not None:
            if row["max_executed_participation"] > limit + 1e-12 or capacity["capital_curve"]["max_executed_participation"].max() > limit + 1e-12:
                raise RuntimeError(f"{factor}: executed participation exceeds limit")
        capacity["summary"].to_csv(factor_dir / "capacity_summary.csv", index=False)
        capacity["capital_curve"].to_csv(factor_dir / "capacity_curve.csv", index=False)
        capacity_rows.append({"factor": factor, **{f"capacity_{key}" if not key.startswith("capacity_") else key: value for key, value in row.items()}})
        inputs[f"{factor}_trades"] = trades_path
        inputs[f"{factor}_daily_pnl"] = pnl_path
        inputs[f"{factor}_execution_metadata"] = factor_dir / "backtest_metadata.json"
        print(f"{factor}: p95 capacity=${row['capacity_at_10pct_adv']:,.0f}, "
              f"all-date capacity=${row['capacity_at_max_10pct_adv']:,.0f}, "
              f"max executed participation={row['max_executed_participation']:.4%}")
    summary.join(pd.DataFrame(capacity_rows).set_index("factor")).to_csv(summary_path)
    write_run_metadata(output_root / "capacity_refresh_metadata.json", stage="refresh_candidate_capacity",
                       inputs=inputs, params={"factors": list(summary.index)},
                       extra={"note": "Final capacity analytics recomputed from the preserved full execution reruns."})


def main() -> None:
    set_project_root(ROOT)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data" / "experiments" / "candidate_backtests_20261016")
    parser.add_argument("--refresh-capacity", action="store_true", help="Recompute capacity from preserved trades and daily PnL.")
    args = parser.parse_args()
    output_root = args.output_dir.resolve()
    if args.refresh_capacity:
        refresh_capacity(output_root)
        return
    output_root.mkdir(parents=True, exist_ok=True)
    benchmark = load_benchmark("us").set_index("date")["return"]
    sector_map = ensure_sector_map(allow_fetch=False)
    rows = []

    config = PortfolioConfig(
        mode=PortfolioMode.LONG_SHORT,
        weight_scheme=WeightScheme.EQUAL,
        rebalance_freq="M",
        n_quantiles=5,
        max_stock_weight=0.05,
        max_sector_weight=0.30,
        # The cached GICS map comes from today's S&P 500 constituents, so names
        # since removed from the index have no sector. Bucket them instead of
        # aborting; the choice is recorded in the run metadata.
        missing_sector_policy="unknown_bucket",
    )
    cost_model = CostModel(
        slippage_bps=5.0,
        impact_coeff=0.1,
        impact_exponent=0.5,
        commission_per_share=0.005,
        commission_pct=0.0,
        borrow_rate_annual=0.005,
        sec_fee_rate=8e-6,
    )

    requested = os.environ.get("ENTROPY_CANDIDATES")
    candidates = [value.strip() for value in requested.split(",")] if requested else CANDIDATES

    for factor in candidates:
        print(f"\n=== {factor} ===", flush=True)
        factor_dir = output_root / factor
        factor_dir.mkdir(parents=True, exist_ok=True)
        portfolio = run_portfolio_pipeline(
            method="quantile",
            signal_col=factor,
            config=config,
            sector_map=sector_map,
            output_path=factor_dir / "weights.parquet",
        )
        weights = portfolio["weights"]
        exposure = weights.groupby("date")["weight"].agg(
            net="sum",
            gross=lambda x: x.abs().sum(),
            short_count=lambda x: int((x < 0).sum()),
        )
        if exposure["short_count"].min() <= 0:
            raise RuntimeError(f"{factor}: no short positions after construction")
        if exposure["net"].abs().max() > 1e-6:
            raise RuntimeError(f"{factor}: portfolio is not market neutral")

        result = run_trading_pipeline(
            weights_path=portfolio["output_path"],
            cost_model=cost_model,
            initial_capital=1_000_000.0,
            output_dir=factor_dir,
        )
        perf = result["performance"]
        daily_pnl = result["daily_pnl"].copy()
        if "date" in daily_pnl.columns:
            net_returns = daily_pnl.set_index(pd.to_datetime(daily_pnl["date"]))["net_ret"]
        else:
            net_returns = daily_pnl["net_ret"].copy()
            net_returns.index = pd.to_datetime(net_returns.index)
        benchmark_metrics = benchmark_analysis(
            net_returns, benchmark,
        )
        capacity_summary = result["capacity"]["summary"].iloc[0].to_dict()
        if capacity_summary["max_executed_participation"] > cost_model.max_participation_rate + 1e-12:
            raise RuntimeError(f"{factor}: executed participation exceeds limit")

        rows.append({
            "factor": factor,
            "avg_net_exposure": exposure["net"].mean(),
            "avg_gross_exposure": exposure["gross"].mean(),
            "avg_short_count": exposure["short_count"].mean(),
            "avg_portfolio_turnover": _compute_avg_turnover(weights),
            **perf,
            **{f"capacity_{key}" if not key.startswith("capacity_") else key: value for key, value in capacity_summary.items()},
            **{f"benchmark_{key}": value for key, value in benchmark_metrics.items()},
        })
        print(
            f"{factor}: net Sharpe={perf.get('net_sharpe', float('nan')):.3f}, "
            f"net return={perf.get('net_ann_return', float('nan')):.2%}, "
            f"max DD={perf.get('net_max_drawdown', float('nan')):.2%}",
            flush=True,
        )

    summary = pd.DataFrame(rows).set_index("factor")
    summary_path = output_root / "candidate_summary.csv"
    if requested and summary_path.exists():
        previous = pd.read_csv(summary_path).set_index("factor")
        previous = previous.loc[~previous.index.isin(candidates)]
        summary = pd.concat([previous, summary], axis=0)
    summary = summary.sort_values("net_sharpe", ascending=False)
    summary.to_csv(summary_path)
    print("\n", summary.to_string(), flush=True)

    # Provenance: record the exact code and inputs behind these numbers so a
    # reviewer can tell whether artifacts were rebuilt from the same commit.
    from dataclasses import asdict, is_dataclass

    from quant_platform.core.utils.io import resolve_data_path
    from quant_platform.core.utils.provenance import write_run_metadata

    def _describe(obj: object) -> dict:
        if is_dataclass(obj):
            try:
                return asdict(obj)
            except TypeError:
                pass
        return dict(getattr(obj, "__dict__", {})) or {"repr": repr(obj)}

    write_run_metadata(
        output_root / "run_metadata.json",
        stage="backtest_factor_candidates",
        inputs={
            "factors": resolve_data_path("factors", "factors.parquet"),
            "prices": resolve_data_path("prices", "prices.parquet"),
            "universe": resolve_data_path("universe", "universe.parquet"),
            "sector_map": resolve_data_path("reference", "sector_map.parquet"),
        },
        params={
            "candidates": candidates,
            "portfolio_config": _describe(config),
            "cost_model": _describe(cost_model),
            "initial_capital": 1_000_000.0,
        },
        extra={"summary_path": str(summary_path)},
    )
    print(f"\nWrote run metadata to {output_root / 'run_metadata.json'}", flush=True)


if __name__ == "__main__":
    main()
