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
from typing import Dict, Iterable, List, Optional

import pandas as pd
from loguru import logger

from quant_platform.core.data.sectors import ensure_sector_map
from quant_platform.core.alpha_models.multi_factor import (
    MultiFactorCombiner,
    MultiFactorConfig,
    RegimePolicy,
    infer_factor_metadata,
)
from quant_platform.core.portfolio.construction import PortfolioConfig, PortfolioConstructor, PortfolioMode
from quant_platform.core.portfolio.quantile import QuantilePortfolio
from quant_platform.core.portfolio.optimize import OptimizedPortfolio
from quant_platform.core.portfolio.rebalance import carry_forward_weights, rebalance_dates, validate_portfolio_weights
from quant_platform.core.signals.cross_sectional.evaluation import add_forward_returns
from quant_platform.core.signals.effective import build_effective_signal
from quant_platform.core.signals.registry import FactorRegistry
from quant_platform.core.utils.io import load_config, load_parquet, resolve_data_path, save_parquet


# ===================================================================
# High-level runner
# ===================================================================

def run_portfolio_pipeline(
    method: str = "quantile",
    signal_col: Optional[str] = None,
    factor_cols: Optional[Iterable[str]] = None,
    config: Optional[PortfolioConfig] = None,
    output_path: Optional[Path | str] = None,
    sector_map: Optional[pd.DataFrame] = None,
    factors_override: Optional[pd.DataFrame] = None,
    prices_override: Optional[pd.DataFrame] = None,
    universe_override: Optional[pd.DataFrame] = None,
    multi_factor_method: Optional[str] = None,
    baseline_factors: Optional[Iterable[str]] = None,
    regime_col: Optional[str] = None,
    multi_factor_lookback: int = 126,
    neutralize_by: Optional[Iterable[str]] = None,
    return_col: str = "fwd_ret_1d",
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

    if factors_override is None and not factors_path.exists():
        raise FileNotFoundError(f"Factors not found: {factors_path}. Run build_factors.py first.")
    if prices_override is None and not prices_path.exists():
        raise FileNotFoundError(f"Prices not found: {prices_path}. Run build_dataset.py first.")

    factors = factors_override.copy() if factors_override is not None else load_parquet(factors_path)
    prices = prices_override.copy() if prices_override is not None else load_parquet(prices_path)
    factors["date"] = pd.to_datetime(factors["date"])
    prices["date"] = pd.to_datetime(prices["date"])

    multi_factor_weights = pd.DataFrame()
    regime_controls = pd.DataFrame()
    requested_factors = list(factor_cols or [])

    if requested_factors:
        missing = [c for c in requested_factors if c not in factors.columns]
        if missing:
            raise ValueError(f"Requested factor columns not found: {missing}")

    if requested_factors and len(requested_factors) > 1:
        factors = _ensure_forward_returns(factors, prices, return_col)
        directions, categories = infer_factor_metadata(requested_factors)
        regime_policy = RegimePolicy(regime_col=regime_col) if regime_col else None
        combiner = MultiFactorCombiner(
            MultiFactorConfig(
                method=multi_factor_method or "rolling_icir",
                lookback=multi_factor_lookback,
                baseline_factors=tuple(baseline_factors or ()),
                regime_policy=regime_policy,
                output_col="alpha_multi",
            ),
            direction_map=directions,
            category_map=categories,
        )
        factors = combiner.fit_transform(
            factors,
            requested_factors,
            return_col=return_col,
            neutralize_by=neutralize_by,
        )
        signal_col = "alpha_multi"
        multi_factor_weights = combiner.factor_weights_
        regime_controls = combiner.regime_controls_
        logger.info(
            "Built multi-factor alpha via {} using {} factors",
            multi_factor_method or "rolling_icir",
            len(requested_factors),
        )

    # Detect signal column
    if signal_col is None:
        for c in factors.columns:
            if c not in ("date", "ticker"):
                signal_col = c
                break
    if signal_col is None:
        raise ValueError("No signal column found in factors")

    signal_direction = _resolve_signal_direction(signal_col)
    logger.info("Using signal column: {} (direction={:+d})", signal_col, signal_direction)

    # Build signal DataFrame (merge price context for optimiser/inverse-vol if needed)
    signal = factors[["date", "ticker", signal_col]].copy()
    signal = build_effective_signal(signal, signal_col, direction=signal_direction)
    needs_price_context = method == "optimize" or config.weight_scheme.value == "inverse_vol"
    if needs_price_context:
        px_cols = ["date", "ticker", "adj_close"]
        if all(c in prices.columns for c in px_cols):
            signal = signal.merge(prices[px_cols], on=["date", "ticker"], how="left")

    # Universe
    if universe_override is not None:
        universe = universe_override.copy()
        universe["date"] = pd.to_datetime(universe["date"])
    elif universe_path.exists():
        universe = load_parquet(universe_path)
        universe["date"] = pd.to_datetime(universe["date"])
    else:
        # Fallback: all tickers in prices are tradable
        logger.warning("Universe file not found — using all tickers in prices as tradable")
        universe = prices[["date", "ticker"]].drop_duplicates()
        universe["pass_all_filters"] = True

    if sector_map is None and config.max_sector_weight is not None and config.max_sector_weight > 0:
        sector_map = ensure_sector_map(tickers=sorted(signal["ticker"].dropna().unique()))
        if sector_map is not None and not sector_map.empty:
            logger.info("Loaded sector map for {} tickers", sector_map["ticker"].nunique())

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

    if not weights.empty and not regime_controls.empty:
        weights = _apply_regime_net_exposure(weights, regime_controls)

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
    validate_portfolio_weights(
        daily_weights,
        mode=config.mode.value,
        allow_cash=_regime_holds_cash(regime_controls),
    )

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

    output_path = Path(output_path)
    save_parquet(daily_weights, output_path)

    factor_weights_path = None
    regime_controls_path = None
    if not multi_factor_weights.empty:
        factor_weights_path = output_path.with_name(f"{output_path.stem}_factor_weights.csv")
        multi_factor_weights.to_csv(factor_weights_path, index=False)
    if not regime_controls.empty:
        regime_controls_path = output_path.with_name(f"{output_path.stem}_regime_controls.csv")
        regime_controls.to_csv(regime_controls_path, index=False)

    # Save metadata (signal_col used) for downstream consistency checks
    import json
    meta_path = resolve_data_path("portfolio", "metadata.json")
    meta = {
        "signal_col": signal_col,
        "signal_direction": signal_direction,
        "factor_cols": requested_factors or [signal_col],
        "multi_factor_method": multi_factor_method,
        "baseline_factors": list(baseline_factors or ()),
        "regime_col": regime_col,
        "factor_weights_path": str(factor_weights_path) if factor_weights_path else None,
        "regime_controls_path": str(regime_controls_path) if regime_controls_path else None,
        "method": method,
        "mode": config.mode.value,
        "rebalance_freq": config.rebalance_freq,
        "weights_filename": output_path.name,  # Just the filename for reliable metadata-first lookup
        "weights_path": str(output_path),  # Keep full path for reference
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    logger.info("Portfolio metadata saved → {}", meta_path)

    return {
        "weights": weights,
        "daily_weights": daily_weights,
        "config": config,
        "output_path": output_path,
        "signal_col": signal_col,
        "signal_direction": signal_direction,
        "factor_weights": multi_factor_weights,
        "regime_controls": regime_controls,
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


def _resolve_signal_direction(signal_col: Optional[str]) -> int:
    """Best-effort lookup of factor direction from registry metadata."""
    if not signal_col:
        return 1
    try:
        registry = FactorRegistry()
        registry.discover()
        if signal_col in registry:
            direction = int(registry.get(signal_col).meta.direction)
            return -1 if direction < 0 else 1
    except Exception as exc:
            logger.debug("Could not resolve factor direction for {}: {}", signal_col, exc)
    return 1


def _ensure_forward_returns(
    factors: pd.DataFrame,
    prices: pd.DataFrame,
    return_col: str,
) -> pd.DataFrame:
    """Merge forward returns into a factor frame when a combiner needs them."""
    if return_col in factors.columns:
        return factors
    if "adj_close" not in prices.columns:
        return factors

    periods = [1]
    if return_col.startswith("fwd_ret_") and return_col.endswith("d"):
        try:
            periods = [int(return_col.removeprefix("fwd_ret_").removesuffix("d"))]
        except ValueError:
            periods = [1]

    fwd = add_forward_returns(prices, periods=periods)
    ret_cols = [f"fwd_ret_{p}d" for p in periods if f"fwd_ret_{p}d" in fwd.columns]
    if not ret_cols:
        return factors
    return factors.merge(fwd[["date", "ticker"] + ret_cols], on=["date", "ticker"], how="left")


def _apply_regime_net_exposure(weights: pd.DataFrame, controls: pd.DataFrame) -> pd.DataFrame:
    """Scale portfolio gross/net exposure by regime controls, allowing cash."""
    if "net_exposure" not in controls.columns:
        return weights
    result = weights.copy()
    ctl = controls.copy()
    ctl["date"] = pd.to_datetime(ctl["date"])
    exposure = ctl.set_index("date")["net_exposure"]
    result["date"] = pd.to_datetime(result["date"])
    result["_net_exposure"] = result["date"].map(exposure).fillna(1.0)
    result["weight"] = result["weight"] * result["_net_exposure"]
    return result.drop(columns=["_net_exposure"])


def _regime_holds_cash(controls: pd.DataFrame) -> bool:
    if controls is None or controls.empty or "net_exposure" not in controls.columns:
        return False
    exposure = pd.to_numeric(controls["net_exposure"], errors="coerce").fillna(1.0)
    return bool((exposure < 0.999999).any())
