#!/usr/bin/env python
"""Run and preserve comparable candidate-factor backtests."""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from quant_platform.core.execution.backtest.pipeline import run_trading_pipeline
from quant_platform.core.execution.cost_models.us_equity import CostModel
from quant_platform.core.data.benchmark import load_benchmark
from quant_platform.core.evaluation.benchmark_analytics import benchmark_analysis
from quant_platform.core.portfolio.construction import PortfolioConfig, PortfolioMode, WeightScheme
from quant_platform.core.portfolio.pipeline import _compute_avg_turnover, run_portfolio_pipeline
from quant_platform.core.utils.io import set_project_root


CANDIDATES = ["MOM_12_1M", "STR_1W", "TURNOVER_ACCEL"]


def main() -> None:
    set_project_root(ROOT)
    output_root = ROOT / "data" / "experiments" / "candidate_backtests_20260721"
    output_root.mkdir(parents=True, exist_ok=True)
    benchmark = load_benchmark("us").set_index("date")["return"]
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
        portfolio = run_portfolio_pipeline(
            method="quantile",
            signal_col=factor,
            config=config,
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
        factor_dir = output_root / factor
        factor_dir.mkdir(parents=True, exist_ok=True)
        for path in (ROOT / "data" / "backtest").iterdir():
            if path.is_file():
                shutil.copy2(path, factor_dir / path.name)

        rows.append({
            "factor": factor,
            "avg_net_exposure": exposure["net"].mean(),
            "avg_gross_exposure": exposure["gross"].mean(),
            "avg_short_count": exposure["short_count"].mean(),
            "avg_portfolio_turnover": _compute_avg_turnover(weights),
            **perf,
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
        },
        params={
            "candidates": candidates,
            "portfolio_config": _describe(config),
            "cost_model": _describe(cost_model),
            "initial_capital": 1_000_000.0,
        },
        extra={"summary_path": str(summary_path)},
    )
    write_run_metadata(
        ROOT / "data" / "backtest" / "run_metadata.json",
        stage="backtest_factor_candidates",
        inputs={"factors": resolve_data_path("factors", "factors.parquet")},
        params={"candidates": candidates},
        extra={"note": "copied from the per-factor run; see the experiment dir"},
    )
    print(f"\nWrote run metadata to {output_root / 'run_metadata.json'}", flush=True)


if __name__ == "__main__":
    main()
