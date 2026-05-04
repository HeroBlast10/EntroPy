"""Trading pipeline: weights + prices → trades → PnL → output.

Orchestrates the full backtest simulation:
1. Load daily weights and prices
2. Simulate execution (generate trades + costs)
3. Compute daily PnL (gross / net)
4. Produce cost attribution & performance summary
5. Save all artefacts to Parquet / CSV
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional

import pandas as pd
from loguru import logger

from quant_platform.core.execution.cost_models.us_equity import CostModel, summarise_costs
from quant_platform.core.execution.backtest.vectorized_daily import simulate_execution
from quant_platform.core.execution.backtest.pnl import compute_daily_returns, cost_attribution, performance_summary
from quant_platform.core.evaluation.capacity import capacity_analysis
from quant_platform.core.utils.io import load_config, load_parquet, resolve_data_path, save_parquet


def run_trading_pipeline(
    weights_path: Optional[Path | str] = None,
    cost_model: Optional[CostModel] = None,
    initial_capital: float = 1_000_000.0,
    output_dir: Optional[Path | str] = None,
    benchmark_market: Optional[str] = "us",
    benchmark_path: Optional[Path | str] = None,
    risk_free_rate: float = 0.0,
    capacity_capitals: Optional[Iterable[float]] = None,
) -> Dict[str, object]:
    """End-to-end trading simulation pipeline.

    Parameters
    ----------
    weights_path : path to daily portfolio weights Parquet.
        If ``None``, uses the most recent file in ``data/portfolio/``.
    cost_model : transaction cost parameters.
    initial_capital : starting portfolio value ($).
    output_dir : directory for all output artefacts.

    Returns
    -------
    Dict with keys:
        ``trades``, ``daily_pnl``, ``cost_attribution``,
        ``performance``, ``output_dir``
    """
    cfg = load_config()
    if cost_model is None:
        cost_model = CostModel()

    # --- Load weights ---
    if weights_path is None:
        portfolio_dir = resolve_data_path("portfolio")
        if not portfolio_dir.exists():
            raise FileNotFoundError(f"Portfolio directory not found: {portfolio_dir}")
        
        # Try to read from metadata.json first (more reliable)
        metadata_path = portfolio_dir / "metadata.json"
        if metadata_path.exists():
            import json
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            weights_filename = metadata.get("weights_filename")
            if weights_filename:
                weights_path = portfolio_dir / weights_filename
                if weights_path.exists():
                    logger.info("Loaded weights from metadata: {}", weights_path)
                else:
                    logger.warning("Metadata references missing file: {}", weights_path)
                    weights_path = None
        
        # Fallback to sorted glob if metadata not available
        if weights_path is None:
            parquets = sorted(portfolio_dir.glob("weights_*.parquet"))
            if not parquets:
                raise FileNotFoundError(f"No weight files found in {portfolio_dir}")
            weights_path = parquets[-1]
            logger.warning("Auto-detected weights via glob (fragile): {}", weights_path)

    weights = load_parquet(weights_path)
    weights["date"] = pd.to_datetime(weights["date"])

    # --- Load prices ---
    prices_path = resolve_data_path(cfg["paths"]["prices_dir"], "prices.parquet")
    prices = load_parquet(prices_path)
    prices["date"] = pd.to_datetime(prices["date"])

    # --- Simulate execution ---
    logger.info("Simulating execution with cost model: slippage={} bps, impact_coeff={}",
                cost_model.slippage_bps, cost_model.impact_coeff)

    trades = simulate_execution(
        daily_weights=weights,
        prices=prices,
        cost_model=cost_model,
        initial_capital=initial_capital,
    )

    # --- Compute PnL ---
    daily_pnl = compute_daily_returns(
        daily_weights=weights,
        prices=prices,
        trades=trades,
        cost_model=cost_model,
        initial_capital=initial_capital,
    )

    # --- Cost attribution ---
    attr = pd.DataFrame()
    if not trades.empty:
        attr = cost_attribution(trades)

    # --- Performance summary ---
    benchmark_returns = _load_benchmark_returns(
        market=benchmark_market,
        path=benchmark_path,
        dates=daily_pnl.index,
    )
    perf = performance_summary(
        daily_pnl,
        benchmark_returns=benchmark_returns,
        risk_free_rate=risk_free_rate,
    )

    # --- Capacity analysis ---
    capacity = capacity_analysis(
        trades,
        daily_pnl,
        cost_model=cost_model,
        capital_grid=capacity_capitals or (
            initial_capital,
            initial_capital * 5,
            initial_capital * 10,
            initial_capital * 50,
        ),
    )

    # --- Save outputs ---
    if output_dir is None:
        output_dir = resolve_data_path("backtest")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    save_parquet(trades, output_dir / "trades.parquet") if not trades.empty else None
    save_parquet(daily_pnl.reset_index(), output_dir / "daily_pnl.parquet")

    if not attr.empty:
        attr.to_csv(output_dir / "cost_attribution.csv", index=False)
    if not capacity["summary"].empty:
        capacity["summary"].to_csv(output_dir / "capacity_summary.csv", index=False)
    if not capacity["capital_curve"].empty:
        capacity["capital_curve"].to_csv(output_dir / "capacity_curve.csv", index=False)

    # Save performance summary
    perf_df = pd.DataFrame([perf])
    perf_df.to_csv(output_dir / "performance_summary.csv", index=False)

    # Save backtest metadata for artifact lineage
    import json
    bt_meta = {
        "weights_path": str(weights_path),
        "cost_model": {
            "slippage_bps": cost_model.slippage_bps,
            "impact_coeff": cost_model.impact_coeff,
        },
        "initial_capital": initial_capital,
        "benchmark_market": benchmark_market,
        "benchmark_path": str(benchmark_path) if benchmark_path else None,
        "risk_free_rate": risk_free_rate,
        "start_date": perf.get("start_date"),
        "end_date": perf.get("end_date"),
    }
    # Carry forward signal_col from portfolio metadata if available
    portfolio_meta_path = resolve_data_path("portfolio", "metadata.json")
    if portfolio_meta_path.exists():
        try:
            pmeta = json.loads(portfolio_meta_path.read_text(encoding="utf-8"))
            bt_meta["signal_col"] = pmeta.get("signal_col")
            bt_meta["portfolio_method"] = pmeta.get("method")
        except Exception:
            pass
    bt_meta_path = output_dir / "backtest_metadata.json"
    bt_meta_path.write_text(json.dumps(bt_meta, indent=2, default=str), encoding="utf-8")
    logger.info("Backtest metadata saved → {}", bt_meta_path)

    # Log headline
    logger.info("=" * 60)
    logger.info("BACKTEST RESULTS")
    logger.info("=" * 60)
    logger.info("Period       : {} – {}", perf.get("start_date"), perf.get("end_date"))
    logger.info("Gross Return : {:.2%} ann  |  Sharpe {:.2f}",
                perf.get("gross_ann_return", 0), perf.get("gross_sharpe", 0))
    logger.info("Net Return   : {:.2%} ann  |  Sharpe {:.2f}",
                perf.get("net_ann_return", 0), perf.get("net_sharpe", 0))
    logger.info("Max Drawdown : {:.2%} (net)", perf.get("net_max_drawdown", 0))
    logger.info("Cost Drag    : {:.1f} bps total trading  |  {:.1f} bps total borrow",
                perf.get("total_trading_cost_bps", 0), perf.get("total_borrow_cost_bps", 0))
    logger.info("=" * 60)

    if not attr.empty:
        logger.info("Cost Attribution:")
        for _, row in attr.iterrows():
            logger.info("  {:15s}  ${:>12,.2f}  ({:>5.1f}%  |  {:.1f} bps)",
                        row["component"], row["total_dollar"],
                        row["pct_of_total"] * 100, row["bps_of_notional"])

    return {
        "trades": trades,
        "daily_pnl": daily_pnl,
        "cost_attribution": attr,
        "performance": perf,
        "capacity": capacity,
        "output_dir": output_dir,
    }


def _load_benchmark_returns(
    *,
    market: Optional[str],
    path: Optional[Path | str],
    dates: pd.DatetimeIndex,
) -> Optional[pd.Series]:
    if market is None and path is None:
        return None
    try:
        from quant_platform.core.data.benchmark import load_benchmark

        bench = load_benchmark(market=market or "us", path=path)
    except Exception as exc:
        logger.warning("Benchmark load failed: {}", exc)
        return None
    if bench.empty or "return" not in bench.columns:
        return None
    bench["date"] = pd.to_datetime(bench["date"])
    returns = bench.set_index("date")["return"].sort_index()
    return returns.reindex(dates)
