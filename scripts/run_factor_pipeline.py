#!/usr/bin/env python
"""End-to-end factor research pipeline: portfolio → backtest → report.

PREREQUISITES:
  1. Run 'python scripts/build_factors.py --evaluate' FIRST to generate:
     - data/factors/factors.parquet (all computed factors)
     - data/factors/factor_comparison.csv (evaluation metrics for all factors)

This script orchestrates the research workflow for one or more factors:
  1. Build portfolio weights for specified factor(s)
  2. Run backtest simulation
  3. Generate HTML research report

Usage examples::

    # IMPORTANT: Run this FIRST to compute all factors
    python scripts/build_factors.py --evaluate

    # Then generate report for single factor
    python scripts/run_factor_pipeline.py --factors REALIZED_JUMP

    # Generate reports for multiple factors
    python scripts/run_factor_pipeline.py --factors MOM_12_1M VOL_20D ILLIQ_AMIHUD

    # Generate reports for ALL factors in factor_comparison.csv
    python scripts/run_factor_pipeline.py --all-factors

    # List available factors from factor_comparison.csv
    python scripts/run_factor_pipeline.py --list

    # Auto-select best factor and generate report
    python scripts/run_factor_pipeline.py --auto-best

    # Quick mode (skip walk-forward and ablation in report)
    python scripts/run_factor_pipeline.py --factors MOM_12_1M --quick

    # Custom portfolio settings
    python scripts/run_factor_pipeline.py --factors MOM_12_1M --mode long_short --freq W

    # Custom output directory for multiple reports
    python scripts/run_factor_pipeline.py --factors MOM_12_1M VOL_20D --output-dir reports/batch_2024
"""

from __future__ import annotations

import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

import click
import pandas as pd
from loguru import logger

from quant_platform.core.utils.io import set_project_root, resolve_data_path, load_parquet

logger.remove()
logger.add(sys.stderr, level="INFO", format=(
    "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
    "<level>{message}</level>"
))


def run_step(step_name: str, func, *args, **kwargs):
    """Execute a pipeline step with logging and error handling."""
    logger.info("=" * 80)
    logger.info(f"STEP: {step_name}")
    logger.info("=" * 80)
    try:
        result = func(*args, **kwargs)
        logger.success(f"✓ {step_name} completed successfully")
        return result
    except Exception as exc:
        logger.error(f"✗ {step_name} failed: {exc}")
        raise


def build_portfolio_step(signal_col: str, mode: str, freq: str, method: str, 
                        max_stock_weight: float, max_sector_weight: float):
    """Step 2: Build portfolio weights."""
    from quant_platform.core.portfolio.construction import PortfolioConfig, PortfolioMode, WeightScheme
    from quant_platform.core.portfolio.pipeline import run_portfolio_pipeline

    config = PortfolioConfig(
        mode=PortfolioMode(mode),
        weight_scheme=WeightScheme("equal"),
        max_stock_weight=max_stock_weight,
        max_sector_weight=max_sector_weight,
        rebalance_freq=freq,
    )

    result = run_portfolio_pipeline(
        method=method,
        signal_col=signal_col,
        config=config,
    )

    if result["output_path"]:
        logger.info(f"Portfolio weights saved → {result['output_path']}")
        return result["output_path"]
    else:
        raise RuntimeError("Portfolio construction produced no output")


def run_backtest_step(weights_path=None, capital=1_000_000.0):
    """Step 3: Run backtest simulation."""
    from quant_platform.core.execution.cost_models.us_equity import CostModel
    from quant_platform.core.execution.backtest.pipeline import run_trading_pipeline

    cost_model = CostModel(
        slippage_bps=5.0,
        impact_coeff=0.1,
        impact_exponent=0.5,
        commission_per_share=0.005,
        commission_pct=0.0,
        borrow_rate_annual=0.005,
        sec_fee_rate=8e-6,
    )

    result = run_trading_pipeline(
        weights_path=weights_path,
        cost_model=cost_model,
        initial_capital=capital,
    )

    perf = result["performance"]
    logger.info(f"Backtest complete → {result['output_dir']}/")
    logger.info(f"  Gross: {perf.get('gross_ann_return', 0):+.2%} ann, "
                f"Sharpe {perf.get('gross_sharpe', 0):.2f}")
    logger.info(f"  Net:   {perf.get('net_ann_return', 0):+.2%} ann, "
                f"Sharpe {perf.get('net_sharpe', 0):.2f}")
    
    return result


def generate_report_step(signal_col: str, output_path, run_walkforward: bool, 
                        run_ablation: bool, wf_kwargs: dict):
    """Step 4: Generate HTML research report."""
    from quant_platform.core.evaluation.report import generate_report

    path = generate_report(
        output_path=output_path,
        signal_col=signal_col,
        run_walkforward=run_walkforward,
        run_ablation=run_ablation,
        walkforward_kwargs=wf_kwargs,
    )

    logger.info(f"Report generated → {path} ({path.stat().st_size / 1024:.0f} KB)")
    return path


@click.command()
@click.option("--factors", "-f", multiple=True, default=None,
              help="Factor names to generate reports for. Can specify multiple.")
@click.option("--all-factors", is_flag=True, default=False,
              help="Generate reports for ALL factors in factor_comparison.csv.")
@click.option("--auto-best", is_flag=True, default=False,
              help="Auto-select best factor from factor_comparison.csv.")
@click.option("--optimize-by", type=str, default="ric_mean_ic",
              help="Metric for ranking when using --auto-best.")
@click.option("--list", "list_factors", is_flag=True, default=False,
              help="List available factors from factor_comparison.csv and exit.")
@click.option("--output-dir", type=str, default=None,
              help="Output directory for reports (default: data/reports).")
@click.option("--quick", is_flag=True, default=False,
              help="Quick mode: skip walk-forward and ablation in report.")
@click.option("--mode", type=click.Choice(["long_only", "long_short"]),
              default="long_only", help="Portfolio mode.")
@click.option("--freq", type=click.Choice(["D", "W", "M"]),
              default="M", help="Rebalance frequency.")
@click.option("--method", type=click.Choice(["quantile", "optimize"]),
              default="quantile", help="Portfolio construction method.")
@click.option("--max-stock-weight", type=float, default=0.05,
              help="Max weight per stock (default 5%%).")
@click.option("--max-sector-weight", type=float, default=0.30,
              help="Max weight per sector (default 30%%).")
@click.option("--capital", type=float, default=1_000_000.0,
              help="Initial capital for backtest ($).")
@click.option("--wf-train", type=int, default=36,
              help="Walk-forward training window (months).")
@click.option("--wf-test", type=int, default=12,
              help="Walk-forward test window (months).")
@click.option("--wf-step", type=int, default=12,
              help="Walk-forward step size (months).")
def main(factors, all_factors, auto_best, optimize_by, list_factors,
         output_dir, quick, mode, freq, method,
         max_stock_weight, max_sector_weight, capital,
         wf_train, wf_test, wf_step):
    """EntroPy — End-to-end factor research pipeline.
    
    Orchestrates: portfolio construction → backtest → report.
    
    PREREQUISITE: Must run 'python scripts/build_factors.py --evaluate' first!
    """
    set_project_root(_project_root)

    from quant_platform.core.evaluation.report import select_best_factor

    cmp_path = resolve_data_path("factors", "factor_comparison.csv")

    # ============================================
    # List mode
    # ============================================
    if list_factors:
        if not cmp_path.exists():
            click.echo(
                f"ERROR: {cmp_path} not found.\n"
                "Run 'python scripts/build_factors.py --evaluate' first to compute and evaluate all factors."
            )
            sys.exit(1)
        
        comparison = pd.read_csv(cmp_path, index_col=0)
        available = [f for f in comparison.index if pd.notna(comparison.loc[f, "ric_mean_ic"])]
        
        click.echo(f"\n{'='*80}")
        click.echo(f"Available Factors ({len(available)} total)")
        click.echo(f"{'='*80}")
        
        display_df = comparison.loc[available, [
            "ric_mean_ic", "ric_icir", "ls_sharpe", "mean_turnover"
        ]].copy()
        display_df = display_df.sort_values("ric_mean_ic", ascending=False)
        
        click.echo("\nTop 10 factors ranked by RankIC Mean:")
        click.echo(display_df.head(10).to_string())
        click.echo(f"\n... and {max(0, len(available) - 10)} more factors.")
        click.echo("\nUsage:")
        click.echo("  python scripts/run_factor_pipeline.py --factors <name1> <name2> ...")
        click.echo("  python scripts/run_factor_pipeline.py --all-factors")
        return

    # ============================================
    # Verify prerequisites
    # ============================================
    factors_path = resolve_data_path("factors", "factors.parquet")
    if not factors_path.exists():
        click.echo(
            f"\nERROR: {factors_path} not found.\n"
            "You must run 'python scripts/build_factors.py' first to compute factors."
        )
        sys.exit(1)
    
    if not cmp_path.exists():
        click.echo(
            f"\nERROR: {cmp_path} not found.\n"
            "You must run 'python scripts/build_factors.py --evaluate' first to evaluate factors."
        )
        sys.exit(1)

    # ============================================
    # Determine target factors
    # ============================================
    target_factors = []
    comparison = pd.read_csv(cmp_path, index_col=0)
    
    if all_factors:
        target_factors = [f for f in comparison.index if pd.notna(comparison.loc[f, "ric_mean_ic"])]
        click.echo(f"\n📊 Pipeline will process ALL {len(target_factors)} factors")
        
    elif factors:
        target_factors = list(factors)
        # Verify requested factors exist in comparison CSV
        available = set(comparison.index)
        missing = [f for f in target_factors if f not in available]
        if missing:
            click.echo(
                f"\nERROR: Factor(s) not found in factor_comparison.csv: {', '.join(missing)}\n"
                f"Available factors: {', '.join(sorted(available)[:10])}..."
            )
            sys.exit(1)
        click.echo(f"\n📊 Pipeline will process {len(target_factors)} factor(s): {', '.join(target_factors)}")
        
    elif auto_best:
        best = select_best_factor(comparison, metric=optimize_by)
        if best:
            target_factors = [best]
            click.echo(f"\n✨ Auto-selected best factor: {best} (by {optimize_by})")
        else:
            click.echo("ERROR: Could not determine best factor from comparison CSV.")
            sys.exit(1)
    else:
        click.echo("\nERROR: Must specify --factors, --all-factors, or --auto-best")
        click.echo("Use --help to see usage options.")
        sys.exit(1)

    # ============================================
    # Pipeline execution for each factor
    # ============================================
    base_dir = resolve_data_path(output_dir if output_dir else "reports")
    base_dir.mkdir(parents=True, exist_ok=True)

    wf_kwargs = {
        "train_months": wf_train,
        "test_months": wf_test,
        "step_months": wf_step,
    }

    results = []
    failed = []

    for i, factor_name in enumerate(target_factors, 1):
        click.echo("\n" + "=" * 80)
        click.echo(f"PROCESSING FACTOR [{i}/{len(target_factors)}]: {factor_name}")
        click.echo("=" * 80)
        
        try:
            # Step 1: Build portfolio
            weights_path = run_step(
                f"Build Portfolio ({factor_name})",
                build_portfolio_step,
                factor_name, mode, freq, method, max_stock_weight, max_sector_weight
            )
            
            # Step 2: Run backtest
            backtest_result = run_step(
                f"Run Backtest ({factor_name})",
                run_backtest_step,
                weights_path, capital
            )
            
            # Step 3: Generate report
            report_path = base_dir / f"report_{factor_name}.html"
            final_path = run_step(
                f"Generate Report ({factor_name})",
                generate_report_step,
                factor_name, report_path,
                not quick, not quick, wf_kwargs
            )
            
            results.append({
                "factor": factor_name,
                "report": final_path,
                "sharpe": backtest_result["performance"].get("net_sharpe", 0),
                "return": backtest_result["performance"].get("net_ann_return", 0),
            })
            
            click.echo(f"\n✓ Pipeline completed for {factor_name}")
            click.echo(f"  Report: {final_path.name}")
            
        except Exception as exc:
            failed.append((factor_name, str(exc)))
            click.echo(f"\n✗ Pipeline FAILED for {factor_name}: {exc}")
            logger.exception(exc)
            continue

    # ============================================
    # Summary
    # ============================================
    click.echo("\n" + "=" * 80)
    click.echo("PIPELINE SUMMARY")
    click.echo("=" * 80)
    
    if results:
        click.echo(f"\n✓ Successful: {len(results)}/{len(target_factors)}")
        click.echo(f"\nReports generated in: {base_dir}")
        
        # Sort by Sharpe and display
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values("sharpe", ascending=False)
        
        click.echo("\nRanking by Net Sharpe:")
        for idx, row in results_df.iterrows():
            click.echo(f"  {row['factor']:20s}  Sharpe: {row['sharpe']:6.2f}  "
                      f"Return: {row['return']:+7.2%}  → {row['report'].name}")
    
    if failed:
        click.echo(f"\n✗ Failed: {len(failed)}/{len(target_factors)}")
        for fname, err in failed:
            click.echo(f"  - {fname}: {err}")
    
    if not results:
        click.echo("\n✗ No reports were generated successfully.")
        sys.exit(1)
    
    click.echo("\n" + "=" * 80)
    click.echo("Pipeline complete!")
    click.echo("=" * 80)


if __name__ == "__main__":
    main()
