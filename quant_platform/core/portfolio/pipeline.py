"""Portfolio pipeline: signal → construction → weights → daily positions.

Chains together:
1. Load signal (factor) + universe + prices
2. Generate rebalance schedule
3. Build portfolio weights (quantile or optimised)
4. Carry forward weights to every trading day
5. Persist to Parquet
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from loguru import logger

from quant_platform.core.portfolio.construction import PortfolioConfig, PortfolioConstructor, PortfolioMode
from quant_platform.core.portfolio.quantile import QuantilePortfolio
from quant_platform.core.portfolio.optimize import OptimizedPortfolio
from quant_platform.core.portfolio.rebalance import carry_forward_weights, rebalance_dates, validate_portfolio_weights
from quant_platform.core.utils.io import load_config, load_parquet, resolve_data_path, save_parquet


# ===================================================================
# High-level runner
# ===================================================================

def run_portfolio_pipeline(
    method: str = "quantile",
    signal_col: Optional[str] = None,
    config: Optional[PortfolioConfig] = None,
    output_path: Optional[Path | str] = None,
    sector_map: Optional[pd.DataFrame] = None,
    **constructor_kwargs,
) -> Dict[str, object]:
    """End-to-end portfolio construction pipeline.

    Parameters
    ----------
    method : ``"quantile"`` (baseline) or ``"optimize"`` (advanced).
    signal_col : which factor column to use as the alpha signal.
        If ``None``, uses the first non-date/ticker column in the
        factor file.
    config : :class:`PortfolioConfig`.  Uses defaults if ``None``.
    output_path : where to save the resulting weights Parquet.
    sector_map : ``[ticker, sector]`` DataFrame for sector constraints.
    **constructor_kwargs : extra kwargs passed to the constructor
        (e.g. ``risk_aversion=2.0`` for the optimised method).

    Returns
    -------
    Dict with keys:
        - ``weights`` — DataFrame ``[date, ticker, weight]`` on rebalance dates
        - ``daily_weights`` — same but carried forward to every trading day
        - ``config`` — the PortfolioConfig used
        - ``output_path`` — saved Parquet path
    """
    cfg_yaml = load_config()
    if config is None:
        config = PortfolioConfig()

    # --- Load data ---
    factors_path = resolve_data_path("factors", "factors.parquet")
    prices_path = resolve_data_path(cfg_yaml["paths"]["prices_dir"], "prices.parquet")
    universe_path = resolve_data_path(cfg_yaml["paths"]["universe_dir"], "universe.parquet")

    if not factors_path.exists():
        raise FileNotFoundError(f"Factors not found: {factors_path}. Run build_factors.py first.")
    if not prices_path.exists():
        raise FileNotFoundError(f"Prices not found: {prices_path}. Run build_dataset.py first.")

    factors = load_parquet(factors_path)
    prices = load_parquet(prices_path)
    factors["date"] = pd.to_datetime(factors["date"])
    prices["date"] = pd.to_datetime(prices["date"])

    # Detect signal column
    if signal_col is None:
        for c in factors.columns:
            if c not in ("date", "ticker"):
                signal_col = c
                break
    if signal_col is None:
        raise ValueError("No signal column found in factors")

    logger.info("Using signal column: {}", signal_col)

    # Build signal DataFrame (merge adj_close for optimiser if needed)
    signal = factors[["date", "ticker", signal_col]].copy()
    if method == "optimize":
        px_cols = ["date", "ticker", "adj_close"]
        if all(c in prices.columns for c in px_cols):
            signal = signal.merge(prices[px_cols], on=["date", "ticker"], how="left")

    # Universe
    if universe_path.exists():
        universe = load_parquet(universe_path)
        universe["date"] = pd.to_datetime(universe["date"])
    else:
        # Fallback: all tickers in prices are tradable
        logger.warning("Universe file not found — using all tickers in prices as tradable")
        universe = prices[["date", "ticker"]].drop_duplicates()
        universe["pass_all_filters"] = True

    # --- Rebalance schedule ---
    reb_dates = rebalance_dates(
        freq=config.rebalance_freq,
        start=str(signal["date"].min().date()),
        end=str(signal["date"].max().date()),
    )

    # --- Constructor ---
    if method == "quantile":
        constructor: PortfolioConstructor = QuantilePortfolio(config)
    elif method == "optimize":
        constructor = OptimizedPortfolio(config, **constructor_kwargs)
    else:
        raise ValueError(f"Unknown method: {method!r}. Use 'quantile' or 'optimize'.")

    logger.info("Portfolio method: {} | mode: {} | weight: {} | rebal: {}",
                method, config.mode.value, config.weight_scheme.value, config.rebalance_freq)

    # --- Build weights ---
    weights = constructor.build(signal, universe, reb_dates, sector_map=sector_map)

    if weights.empty:
        logger.error("Portfolio construction produced no weights")
        return {"weights": weights, "daily_weights": weights, "config": config, "output_path": None}

    # --- Carry forward to daily ---
    from quant_platform.core.data.calendar import trading_dates as get_trading_dates
    all_dates = get_trading_dates(
        start=str(weights["date"].min().date()),
        end=str(weights["date"].max().date()),
    )
    daily_weights = carry_forward_weights(weights, all_dates)

    # --- Validate weights before saving ---
    validate_portfolio_weights(daily_weights, mode=config.mode.value)

    # --- Summary stats ---
    n_reb = weights["date"].nunique()
    avg_holdings = weights.groupby("date").size().mean()
    avg_turnover = _compute_avg_turnover(weights)
    logger.info(
        "Portfolio summary: {} rebalances, {:.0f} avg holdings, {:.1%} avg turnover",
        n_reb, avg_holdings, avg_turnover,
    )

    # --- Save ---
    if output_path is None:
        safe_signal = str(signal_col).replace(" ", "_") if signal_col else "unknown"
        suffix = f"{safe_signal}_{method}_{config.mode.value}_{config.rebalance_freq}"
        output_path = resolve_data_path("portfolio", f"weights_{suffix}.parquet")

    save_parquet(daily_weights, output_path)

    # Save metadata (signal_col used) for downstream consistency checks
    import json
    meta_path = resolve_data_path("portfolio", "metadata.json")
    meta = {
        "signal_col": signal_col,
        "method": method,
        "mode": config.mode.value,
        "rebalance_freq": config.rebalance_freq,
        "weights_path": str(output_path),
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    logger.info("Portfolio metadata saved → {}", meta_path)

    return {
        "weights": weights,
        "daily_weights": daily_weights,
        "config": config,
        "output_path": output_path,
        "signal_col": signal_col,
    }


# ===================================================================
# Helpers
# ===================================================================

def _compute_avg_turnover(weights: pd.DataFrame) -> float:
    """Compute average one-way turnover across rebalance dates."""
    dates = sorted(weights["date"].unique())
    if len(dates) < 2:
        return 0.0

    turnovers = []
    prev = None
    for d in dates:
        cur = weights.loc[weights["date"] == d].set_index("ticker")["weight"]
        if prev is not None:
            all_t = cur.index.union(prev.index)
            c = cur.reindex(all_t, fill_value=0.0)
            p = prev.reindex(all_t, fill_value=0.0)
            turnovers.append((c - p).abs().sum() / 2.0)
        prev = cur

    return float(pd.Series(turnovers).mean()) if turnovers else 0.0
