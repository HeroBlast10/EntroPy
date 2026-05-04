#!/usr/bin/env python
"""CLI entry point for research report generation.

Usage examples::

    # Full report (walk-forward + ablation)
    python scripts/generate_report.py

    # Quick report (skip slow analyses)
    python scripts/generate_report.py --no-walkforward --no-ablation

    # Specify single factor
    python scripts/generate_report.py --factors MOM_12_1M

    # Generate reports for multiple factors (each gets its own HTML)
    python scripts/generate_report.py --factors MOM_12_1M VOL_20D ILLIQ_AMIHUD

    # Generate reports for ALL factors in factor_comparison.csv
    python scripts/generate_report.py --all-factors

    # List available factors from factor_comparison.csv
    python scripts/generate_report.py --list

    # Auto-select best factor from factor_comparison.csv (ranked by ric_mean_ic)
    python scripts/generate_report.py --auto-best

    # Auto-select best factor by a different metric
    python scripts/generate_report.py --auto-best --optimize-by ric_icir
    python scripts/generate_report.py --auto-best --optimize-by ls_sharpe

    # Custom walk-forward parameters
    python scripts/generate_report.py --wf-train 24 --wf-test 6 --wf-step 6

    # Custom output directory for multiple reports
    python scripts/generate_report.py --factors MOM_12_1M VOL_20D --output-dir reports/batch
"""

from __future__ import annotations

import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

import click
from loguru import logger

from quant_platform.core.utils.io import set_project_root

logger.remove()
logger.add(sys.stderr, level="INFO", format=(
    "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
    "<level>{message}</level>"
))


@click.command()
@click.option("--factors", "-f", multiple=True, default=None,
              help="Factor names to generate reports for. Can specify multiple. "
                   "Each factor gets its own HTML report.")
@click.option("--all-factors", is_flag=True, default=False,
              help="Generate reports for ALL factors in factor_comparison.csv.")
@click.option("--auto-best", is_flag=True, default=False,
              help="Auto-select best factor from data/factors/factor_comparison.csv.")
@click.option("--optimize-by", type=str, default="ric_mean_ic",
              show_default=True,
              help="Metric used to rank factors when --auto-best is set "
                   "(e.g. ric_mean_ic, ric_icir, ls_sharpe).")
@click.option("--list", "list_factors", is_flag=True, default=False,
              help="List all available factors from factor_comparison.csv and exit.")
@click.option("--output", "-o", type=str, default=None,
              help="Output HTML file path (only for single factor report).")
@click.option("--output-dir", type=str, default=None,
              help="Output directory for multiple reports (default: data/reports).")
@click.option("--no-walkforward", is_flag=True, default=False,
              help="Skip walk-forward validation (faster).")
@click.option("--no-ablation", is_flag=True, default=False,
              help="Skip ablation study (faster).")
@click.option("--wf-train", type=int, default=36,
              help="Walk-forward training window (months).")
@click.option("--wf-test", type=int, default=12,
              help="Walk-forward test window (months).")
@click.option("--wf-step", type=int, default=12,
              help="Walk-forward step size (months).")
def main(factors, all_factors, auto_best, optimize_by, list_factors, 
         output, output_dir, no_walkforward, no_ablation,
         wf_train, wf_test, wf_step):
    """EntroPy — Generate HTML research report(s) for one or more factors."""
    set_project_root(_project_root)

    import pandas as pd
    from quant_platform.core.evaluation.report import generate_report
    from quant_platform.core.signals.catalog import load_factor_catalog, select_best_factor_from_catalog
    from quant_platform.core.utils.io import resolve_data_path

    cmp_path = resolve_data_path("factors", "factor_comparison.csv")

    # --- List mode ---
    if list_factors:
        if not cmp_path.exists() and not resolve_data_path("factors", "factor_catalog.csv").exists():
            click.echo(
                f"ERROR: {cmp_path} / factor_catalog.csv not found.\n"
                "Run 'python scripts/build_factors.py --evaluate' first."
            )
            sys.exit(1)
        
        catalog = load_factor_catalog()
        if "eligible_for_portfolio" in catalog.columns:
            catalog = catalog[catalog["eligible_for_portfolio"].astype(bool)]
        available = list(catalog.index)
        
        click.echo(f"\n{'='*80}")
        click.echo(f"Available Factors ({len(available)} total)")
        click.echo(f"{'='*80}")
        
        # Display as table with key metrics
        display_cols = [
            c for c in ["signal_type", "selection_score", "deployability_score",
                        "ric_mean_ic", "cost_adj_ls_sharpe", "ls_sharpe", "mean_turnover"]
            if c in catalog.columns
        ]
        display_df = catalog.loc[available, display_cols].copy()
        sort_col = "selection_score" if "selection_score" in display_df.columns else display_cols[-1]
        display_df = display_df.sort_values(sort_col, ascending=False)
        
        click.echo("\nTop factors ranked by RankIC Mean:")
        click.echo(display_df.head(10).to_string())
        click.echo(f"\n... and {len(available) - 10} more factors.\n")
        click.echo("Use --factors <name1> <name2> to generate reports for specific factors.")
        click.echo("Use --all-factors to generate reports for ALL factors.")
        return

    # --- Determine target factors ---
    target_factors = []
    
    if all_factors:
        # Load all factors from comparison CSV
        if not cmp_path.exists() and not resolve_data_path("factors", "factor_catalog.csv").exists():
            click.echo(
                f"ERROR: {cmp_path} / factor_catalog.csv not found.\n"
                "Run 'python scripts/build_factors.py --evaluate' first."
            )
            sys.exit(1)
        
        catalog = load_factor_catalog()
        if "eligible_for_portfolio" in catalog.columns:
            catalog = catalog[catalog["eligible_for_portfolio"].astype(bool)]
        target_factors = list(catalog.index)
        click.echo(f"\n📊 Generating reports for ALL {len(target_factors)} factors...")
        
    elif factors:
        # User specified factors
        target_factors = list(factors)
        click.echo(f"\n📊 Generating reports for {len(target_factors)} factor(s): {', '.join(target_factors)}")
        
    elif auto_best:
        # Auto-select best factor
        if not cmp_path.exists() and not resolve_data_path("factors", "factor_catalog.csv").exists():
            click.echo(
                f"WARNING: {cmp_path} / factor_catalog.csv not found.\n"
                "Run 'python scripts/build_factors.py --evaluate' first."
            )
            sys.exit(1)
        
        catalog = load_factor_catalog()
        best = select_best_factor_from_catalog(catalog, metric=optimize_by)
        if best:
            target_factors = [best]
            click.echo(f"\n✨ Auto-selected best factor: {best} (ranked by {optimize_by})")
        else:
            click.echo("ERROR: Could not determine best factor from comparison CSV.")
            sys.exit(1)
    else:
        # No factor specified - generate default report (auto-detect from factors.parquet)
        click.echo("\n📊 Generating default report (will auto-detect signal from factors)...")
        target_factors = [None]

    # --- Prepare output paths ---
    wf_kwargs = {
        "train_months": wf_train,
        "test_months": wf_test,
        "step_months": wf_step,
    }

    generated_paths = []
    
    # --- Generate reports ---
    if len(target_factors) == 1 and target_factors[0] is not None:
        # Single factor with custom output path
        output_path = output if output else None
        try:
            path = generate_report(
                output_path=output_path,
                signal_col=target_factors[0],
                run_walkforward=not no_walkforward,
                run_ablation=not no_ablation,
                walkforward_kwargs=wf_kwargs,
            )
            generated_paths.append(path)
            click.echo(f"\n✓ Report generated → {path}")
            click.echo(f"  Open in browser: file:///{path.resolve()}")
        except Exception as exc:
            click.echo(f"✗ ERROR generating report for {target_factors[0]}: {exc}")
            logger.exception(exc)
            sys.exit(1)
            
    elif len(target_factors) > 1 or (len(target_factors) == 1 and target_factors[0] is None):
        # Multiple factors or default mode - generate in batch
        base_dir = resolve_data_path(output_dir if output_dir else "reports")
        base_dir.mkdir(parents=True, exist_ok=True)
        
        failed = []
        for i, factor_name in enumerate(target_factors, 1):
            if factor_name is None:
                # Default report
                out_path = base_dir / "research_report.html"
            else:
                # Named factor report
                out_path = base_dir / f"report_{factor_name}.html"
            
            try:
                click.echo(f"\n[{i}/{len(target_factors)}] Generating report for: {factor_name or 'default'}...")
                path = generate_report(
                    output_path=out_path,
                    signal_col=factor_name,
                    run_walkforward=not no_walkforward,
                    run_ablation=not no_ablation,
                    walkforward_kwargs=wf_kwargs,
                )
                generated_paths.append(path)
                click.echo(f"  ✓ Saved → {path.name}")
            except Exception as exc:
                failed.append((factor_name, str(exc)))
                click.echo(f"  ✗ FAILED: {exc}")
                logger.error(f"Report generation failed for {factor_name}: {exc}")
        
        # Summary
        click.echo(f"\n{'='*80}")
        click.echo(f"Summary: {len(generated_paths)}/{len(target_factors)} reports generated successfully")
        click.echo(f"{'='*80}")
        
        if generated_paths:
            click.echo(f"\n✓ Reports saved to: {base_dir}")
            click.echo(f"\nGenerated files:")
            for p in generated_paths:
                click.echo(f"  - {p.name}")
        
        if failed:
            click.echo(f"\n✗ Failed ({len(failed)}):")
            for fname, err in failed:
                click.echo(f"  - {fname}: {err}")
    else:
        click.echo("\nNo factors specified. Use --help to see usage options.")
        sys.exit(1)


if __name__ == "__main__":
    main()
