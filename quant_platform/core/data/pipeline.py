"""Orchestrator that chains all data-layer steps into a single build.

Execution order
---------------
1. **prices** — download OHLCV → ``prices.parquet``
2. **fundamentals** — download financials → ``fundamentals.parquet``
3. **universe** — apply filters using prices + fundamentals → ``universe.parquet``
4. **manifest** — hash every artefact, record metadata → ``manifest.json``

Each step is idempotent: re-running overwrites the previous output.
Steps can also be run individually via the CLI (``scripts/build_dataset.py``).
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Optional

from loguru import logger

from quant_platform.core.data.fundamentals import build_fundamentals
from quant_platform.core.data.manifest import build_manifest
from quant_platform.core.data.prices import build_prices
from quant_platform.core.data.universe import build_universe
from quant_platform.core.utils.io import load_config, load_parquet, resolve_data_path


def run_pipeline(
    steps: Optional[List[str]] = None,
    tickers: Optional[List[str]] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> Dict[str, Path]:
    """Execute the full (or partial) data build pipeline.

    Parameters
    ----------
    steps : subset of ``["prices", "fundamentals", "universe", "manifest"]``.
        ``None`` means run all.
    tickers : override ticker list.
    start, end : override date range.

    Returns
    -------
    Dict mapping step name → output Path.
    """
    all_steps = ["prices", "fundamentals", "universe", "manifest"]
    if steps is None:
        steps = all_steps
    else:
        unknown = set(steps) - set(all_steps)
        if unknown:
            raise ValueError(f"Unknown steps: {unknown}")

    cfg = load_config()
    outputs: Dict[str, Path] = {}
    t0 = time.perf_counter()

    # --- 1. Prices ---
    if "prices" in steps:
        logger.info("=" * 60)
        logger.info("STEP 1/4 — Prices")
        logger.info("=" * 60)
        outputs["prices"] = build_prices(tickers=tickers, start=start, end=end)

    # --- 2. Fundamentals ---
    if "fundamentals" in steps:
        logger.info("=" * 60)
        logger.info("STEP 2/4 — Fundamentals")
        logger.info("=" * 60)
        outputs["fundamentals"] = build_fundamentals(tickers=tickers)

    # --- 3. Universe ---
    if "universe" in steps:
        logger.info("=" * 60)
        logger.info("STEP 3/4 — Universe")
        logger.info("=" * 60)
        prices_path = outputs.get("prices") or resolve_data_path(
            cfg["paths"]["prices_dir"], "prices.parquet")
        fund_path = outputs.get("fundamentals") or resolve_data_path(
            cfg["paths"]["fundamentals_dir"], "fundamentals.parquet")
        outputs["universe"] = build_universe(
            prices_path=prices_path,
            fundamentals_path=fund_path,
        )

    # --- 4. Manifest ---
    if "manifest" in steps:
        logger.info("=" * 60)
        logger.info("STEP 4/4 — Manifest")
        logger.info("=" * 60)
        parquet_paths = [v for k, v in outputs.items() if k != "manifest"]
        # Also pick up files from previous runs if steps were partial
        for table_name, sub_dir, fname in [
            ("prices", cfg["paths"]["prices_dir"], "prices.parquet"),
            ("fundamentals", cfg["paths"]["fundamentals_dir"], "fundamentals.parquet"),
            ("universe", cfg["paths"]["universe_dir"], "universe.parquet"),
        ]:
            p = resolve_data_path(sub_dir, fname)
            if p.exists() and p not in parquet_paths:
                parquet_paths.append(p)

        # Collect row counts
        row_counts = {}
        for p in parquet_paths:
            if p.exists():
                row_counts[p.stem] = len(load_parquet(p))

        outputs["manifest"] = build_manifest(parquet_paths, row_counts=row_counts)

    elapsed = time.perf_counter() - t0
    logger.info("Pipeline completed in {:.1f}s — outputs: {}", elapsed,
                {k: str(v) for k, v in outputs.items()})
    return outputs
