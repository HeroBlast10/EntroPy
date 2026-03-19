#!/usr/bin/env python
"""Factor parameter grid-search (auto-tuning).

For each factor listed in the search space, exhaustively tries all
combinations of ``period``/``window`` and ``lag``, evaluates each variant
on Rank IC, ICIR, and long-short Sharpe, and saves a ranked result table.

Usage examples::

    # Default grid, default objective (ric_mean_ic)
    python scripts/tune_factors.py

    # Custom objective
    python scripts/tune_factors.py --objective ric_icir
    python scripts/tune_factors.py --objective ls_sharpe

    # Only tune specific factors
    python scripts/tune_factors.py --factors MOM_1M MOM_3M MOM_6M

    # Custom output path
    python scripts/tune_factors.py --output data/factors/tune_results.csv

    # Print top-N results
    python scripts/tune_factors.py --top 5
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path
from typing import Dict, List, Optional

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

import click
import pandas as pd
from loguru import logger

from quant_platform.core.utils.io import set_project_root

logger.remove()
logger.add(sys.stderr, level="INFO", format=(
    "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
    "<level>{message}</level>"
))

# ===================================================================
# Default search space
#
# Each entry maps  factor_name → {param_name: [candidate_values]}
# Any key that matches a FactorMeta field (lookback, lag, …) overrides
# the meta; other keys (period, window, …) land in _extra_params and
# are read by the concrete _compute() implementations.
# ===================================================================

DEFAULT_SEARCH_SPACE: Dict[str, Dict[str, List]] = {
    "MOM_1M": {
        "period": [10, 21, 42],
        "lag":    [1, 2, 5],
    },
    "MOM_3M": {
        "period": [42, 63, 84],
        "lag":    [1, 5],
    },
    "MOM_6M": {
        "period": [63, 126, 189],
        "lag":    [1, 5, 21],
    },
    "MOM_12_1M": {
        "period": [126, 189, 252],
        "lag":    [5, 21],
    },
    "STR_1W": {
        "period": [3, 5, 10],
        "lag":    [1, 2],
    },
    "STR_1M": {
        "period": [10, 21, 42],
        "lag":    [1, 2, 5],
    },
    "MOM_PATH": {
        "window": [63, 126, 189],
        "lag":    [1, 5],
    },
}


# ===================================================================
# Evaluation helper
# ===================================================================

def _evaluate_variant(
    factor_name: str,
    params: Dict,
    prices: pd.DataFrame,
    fwd_ret_col: str = "fwd_ret_1d",
    n_quantiles: int = 5,
) -> Optional[Dict]:
    """Compute one (factor, params) variant and return summary metrics."""
    from quant_platform.core.signals.registry import FactorRegistry
    from quant_platform.core.signals.cross_sectional.evaluation import (
        compute_rank_ic_series,
        ic_summary,
        long_short_returns,
        quantile_returns,
    )

    reg = FactorRegistry()
    reg.discover()

    if factor_name not in reg:
        logger.warning("Factor {} not found in registry — skipping.", factor_name)
        return None

    cls = reg.get(factor_name)
    try:
        factor_df = cls(**params).compute(prices)
    except Exception as exc:
        logger.error("Factor {} with params {} failed: {}", factor_name, params, exc)
        return None

    col = factor_df.columns[-1]
    merged = factor_df.merge(
        prices[["date", "ticker", fwd_ret_col]],
        on=["date", "ticker"],
        how="inner",
    ).dropna(subset=[col, fwd_ret_col])

    if len(merged) < 50:
        return None

    ric = compute_rank_ic_series(merged, col, fwd_ret_col)
    stats = ic_summary(ric)

    ls = long_short_returns(merged, col, fwd_ret_col, n_quantiles=n_quantiles)
    ls_sharpe = (
        ls.mean() / ls.std() * (252 ** 0.5) if len(ls) > 1 and ls.std() > 0 else float("nan")
    )

    row: Dict = {"factor": factor_name, **params}
    row["ric_mean_ic"] = stats["mean_ic"]
    row["ric_icir"]    = stats["icir"]
    row["ric_t_stat"]  = stats["t_stat"]
    row["ric_hit_rate"] = stats["hit_rate"]
    row["ls_sharpe"]   = ls_sharpe
    row["n_obs"]       = stats["n_obs"]
    return row


# ===================================================================
# CLI
# ===================================================================

@click.command()
@click.option("--factors", "-f", multiple=True, default=None,
              help="Factor names to tune. Omit to tune all in the default grid.")
@click.option("--objective", "-obj", type=str, default="ric_mean_ic", show_default=True,
              help="Metric to rank results by (ric_mean_ic | ric_icir | ls_sharpe).")
@click.option("--output", "-o", type=str, default=None,
              help="CSV path for full results table (default: data/factors/tune_results.csv).")
@click.option("--top", type=int, default=10, show_default=True,
              help="Print top-N results per factor.")
@click.option("--fwd-period", type=int, default=1, show_default=True,
              help="Forward return period in trading days used as prediction target.")
def main(factors, objective, output, top, fwd_period):
    """EntroPy — Grid-search factor parameters and rank by an IC/Sharpe objective."""
    set_project_root(_project_root)

    from quant_platform.core.utils.io import load_config, load_parquet, resolve_data_path
    from quant_platform.core.signals.cross_sectional.evaluation import add_forward_returns

    cfg = load_config()
    prices_path = resolve_data_path(cfg["paths"]["prices_dir"], "prices.parquet")
    if not prices_path.exists():
        click.echo(f"ERROR: prices not found at {prices_path}. Run build_dataset.py first.")
        sys.exit(1)

    prices = load_parquet(prices_path)
    prices["date"] = pd.to_datetime(prices["date"])

    fwd_col = f"fwd_ret_{fwd_period}d"
    prices = add_forward_returns(prices, periods=[fwd_period])

    # Which factors to tune
    search_space = {
        k: v for k, v in DEFAULT_SEARCH_SPACE.items()
        if not factors or k in factors
    }
    if not search_space:
        click.echo("ERROR: no matching factors found in search space.")
        sys.exit(1)

    click.echo(f"\nTuning {len(search_space)} factor(s) | objective: {objective}")
    click.echo("=" * 70)

    all_rows: List[Dict] = []

    for factor_name, param_grid in search_space.items():
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        combos = list(itertools.product(*param_values))

        click.echo(f"\n{factor_name}  ({len(combos)} combinations)")

        for combo in combos:
            params = dict(zip(param_names, combo))
            row = _evaluate_variant(factor_name, params, prices, fwd_col)
            if row is not None:
                all_rows.append(row)
                click.echo(
                    f"  {params}  →  "
                    f"ric_mean_ic={row['ric_mean_ic']:.4f}  "
                    f"ric_icir={row['ric_icir']:.2f}  "
                    f"ls_sharpe={row['ls_sharpe']:.2f}"
                )

    if not all_rows:
        click.echo("\nNo results produced. Check data availability.")
        sys.exit(1)

    results = pd.DataFrame(all_rows)

    # Rank by objective (descending)
    if objective in results.columns:
        results = results.sort_values(objective, ascending=False)
    else:
        click.echo(f"WARNING: objective '{objective}' not found in results; defaulting to ric_mean_ic.")
        results = results.sort_values("ric_mean_ic", ascending=False)

    # Save
    if output is None:
        out_path = resolve_data_path("factors", "tune_results.csv")
    else:
        out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(out_path, index=False)
    click.echo(f"\n✓ Full results saved → {out_path}")

    # Print top-N per factor
    click.echo(f"\n{'=' * 70}")
    click.echo(f"TOP-{top} RESULTS PER FACTOR (by {objective})")
    click.echo("=" * 70)
    for factor_name in results["factor"].unique():
        sub = results[results["factor"] == factor_name].head(top)
        click.echo(f"\n{factor_name}:")
        click.echo(sub.to_string(index=False))

    # Overall best
    best_row = results.iloc[0]
    click.echo(f"\n{'=' * 70}")
    click.echo(f"OVERALL BEST ({objective}): {best_row['factor']}  params={best_row.drop('factor').to_dict()}")


if __name__ == "__main__":
    main()
