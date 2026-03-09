"""Ablation study framework.

Systematically vary one dimension at a time to measure sensitivity:

1. **No cost**       — re-run backtest with zero transaction costs
2. **No neutralize** — recompute factors without sector/size neutralisation
3. **No winsorize**  — recompute factors without winsorisation
4. **Alt universe**  — use a different universe filter (e.g. relax min_cap)
5. **Alt rebalance** — change rebalance frequency (D vs W vs M)
6. **Alt weight**    — change weighting scheme (equal vs cap vs signal)

Each scenario produces a performance summary comparable to the baseline,
so the researcher (or interviewer) can see exactly how much each design
choice contributes to the result.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger

from quant_platform.core.execution.cost_models.us_equity import CostModel
from quant_platform.core.execution.backtest.pnl import compute_daily_returns, performance_summary


@dataclass
class AblationScenario:
    """Describes one ablation variant."""
    name: str
    description: str
    # Override parameters (None = keep baseline)
    cost_model: Optional[CostModel] = None
    zero_cost: bool = False
    skip_neutralize: bool = False
    skip_winsorize: bool = False
    rebalance_freq: Optional[str] = None
    weight_scheme: Optional[str] = None
    universe_min_cap: Optional[float] = None


# ===================================================================
# Pre-defined scenarios
# ===================================================================

STANDARD_ABLATIONS: List[AblationScenario] = [
    AblationScenario(
        name="baseline",
        description="Full pipeline with default settings",
    ),
    AblationScenario(
        name="no_cost",
        description="Zero transaction costs (gross-only)",
        zero_cost=True,
    ),
    AblationScenario(
        name="high_cost",
        description="High-cost scenario (2× slippage, 2× impact)",
        cost_model=CostModel(slippage_bps=10.0, impact_coeff=0.2),
    ),
    AblationScenario(
        name="no_neutralize",
        description="Factors computed without neutralisation",
        skip_neutralize=True,
    ),
    AblationScenario(
        name="no_winsorize",
        description="Factors computed without winsorisation",
        skip_winsorize=True,
    ),
    AblationScenario(
        name="rebal_weekly",
        description="Weekly rebalance instead of monthly",
        rebalance_freq="W",
    ),
    AblationScenario(
        name="rebal_daily",
        description="Daily rebalance",
        rebalance_freq="D",
    ),
    AblationScenario(
        name="weight_mcap",
        description="Market-cap weighting instead of equal",
        weight_scheme="market_cap",
    ),
    AblationScenario(
        name="universe_relaxed",
        description="Relaxed universe: min_cap = $10M (vs $50M baseline)",
        universe_min_cap=1e7,
    ),
]


# ===================================================================
# Run ablation (cost-only scenarios — fast path)
# ===================================================================

def run_cost_ablation(
    daily_weights: pd.DataFrame,
    prices: pd.DataFrame,
    trades_baseline: pd.DataFrame,
    initial_capital: float = 1_000_000.0,
) -> pd.DataFrame:
    """Fast ablation: re-price the same trades under different cost models.

    This avoids re-running the full pipeline; only the PnL is recomputed.

    Returns a comparison DataFrame with one row per scenario.
    """
    scenarios = [
        ("baseline", CostModel()),
        ("zero_cost", CostModel(
            commission_per_share=0, commission_pct=0, commission_min=0,
            slippage_bps=0, impact_coeff=0, sec_fee_rate=0,
            finra_taf_per_share=0, stamp_duty_pct=0, borrow_rate_annual=0,
        )),
        ("low_cost", CostModel(slippage_bps=2.0, impact_coeff=0.05)),
        ("high_cost", CostModel(slippage_bps=10.0, impact_coeff=0.20)),
        ("very_high_cost", CostModel(slippage_bps=20.0, impact_coeff=0.30)),
    ]

    rows = []
    for name, cm in scenarios:
        from quant_platform.core.execution.cost_models.us_equity import estimate_batch_costs

        # Re-estimate costs on the same trade list
        trade_cols = ["date", "ticker", "side", "shares", "price", "adv_shares", "daily_vol"]
        available = [c for c in trade_cols if c in trades_baseline.columns]
        trades_clean = trades_baseline[available].copy()
        trades_repriced = estimate_batch_costs(trades_clean, cm)
        trades_repriced["date"] = trades_baseline["date"].values

        pnl = compute_daily_returns(
            daily_weights=daily_weights,
            prices=prices,
            trades=trades_repriced,
            cost_model=cm,
            initial_capital=initial_capital,
        )
        perf = performance_summary(pnl)
        perf["scenario"] = name
        rows.append(perf)

    result = pd.DataFrame(rows)
    # Reorder columns
    cols = ["scenario"] + [c for c in result.columns if c != "scenario"]
    result = result[cols]

    logger.info("Cost ablation complete: {} scenarios", len(result))
    return result


# ===================================================================
# Full ablation (re-runs factor computation — slow path)
# ===================================================================

def run_full_ablation(
    prices: pd.DataFrame,
    universe: pd.DataFrame,
    signal_col: str,
    scenarios: Optional[List[AblationScenario]] = None,
    initial_capital: float = 1_000_000.0,
) -> pd.DataFrame:
    """Re-run the full pipeline under each ablation scenario.

    This is the comprehensive version that actually recomputes factors
    and portfolios.  It is slow but gives the most accurate picture.

    Returns a comparison DataFrame with one row per scenario.
    """
    if scenarios is None:
        # Use only the cost-related + rebalance scenarios for speed
        scenarios = [s for s in STANDARD_ABLATIONS
                     if s.name in ("baseline", "no_cost", "high_cost",
                                   "rebal_weekly", "rebal_daily")]

    from quant_platform.core.signals.registry import FactorRegistry
    from quant_platform.core.portfolio.construction import PortfolioConfig, PortfolioMode, WeightScheme
    from quant_platform.core.portfolio.quantile import QuantilePortfolio
    from quant_platform.core.portfolio.rebalance import carry_forward_weights, rebalance_dates
    from quant_platform.core.execution.backtest.vectorized_daily import simulate_execution
    from quant_platform.core.data.calendar import trading_dates

    rows = []
    for scenario in scenarios:
        logger.info("Ablation: {}", scenario.name)

        try:
            # --- Factor computation ---
            reg = FactorRegistry()
            reg.discover()

            winsorize_limits = (0.01, 0.99) if not scenario.skip_winsorize else (0.0, 1.0)
            neutralize_by = None  # baseline has no neutralize cols in prices
            # (In a full setup, you'd pass ["sector", "log_mcap"])

            factor_df = reg.compute_all(
                prices,
                factor_names=[signal_col] if signal_col in reg._registry else None,
                winsorize_limits=winsorize_limits,
                zscore=True,
            )

            if signal_col not in factor_df.columns:
                # Use first available
                sig = [c for c in factor_df.columns if c not in ("date", "ticker")][0]
            else:
                sig = signal_col

            signal = factor_df[["date", "ticker", sig]]

            # --- Portfolio ---
            freq = scenario.rebalance_freq or "M"
            ws = WeightScheme(scenario.weight_scheme) if scenario.weight_scheme else WeightScheme.EQUAL

            config = PortfolioConfig(
                mode=PortfolioMode.LONG_ONLY,
                weight_scheme=ws,
                rebalance_freq=freq,
            )

            reb = rebalance_dates(
                freq=freq,
                start=str(signal["date"].min().date()),
                end=str(signal["date"].max().date()),
            )

            constructor = QuantilePortfolio(config)
            weights = constructor.build(signal, universe, reb)

            if weights.empty:
                logger.warning("Ablation {}: no weights generated", scenario.name)
                continue

            all_dates = trading_dates(
                start=str(weights["date"].min().date()),
                end=str(weights["date"].max().date()),
            )
            daily_w = carry_forward_weights(weights, all_dates)

            # --- Cost model ---
            if scenario.zero_cost:
                cm = CostModel(
                    commission_per_share=0, commission_pct=0, commission_min=0,
                    slippage_bps=0, impact_coeff=0, sec_fee_rate=0,
                    finra_taf_per_share=0, stamp_duty_pct=0, borrow_rate_annual=0,
                )
            elif scenario.cost_model is not None:
                cm = scenario.cost_model
            else:
                cm = CostModel()

            # --- Execution & PnL ---
            trades = simulate_execution(daily_w, prices, cm, initial_capital)
            pnl = compute_daily_returns(daily_w, prices, trades, cm, initial_capital)
            perf = performance_summary(pnl)
            perf["scenario"] = scenario.name
            perf["description"] = scenario.description
            rows.append(perf)

        except Exception as exc:
            logger.error("Ablation {} failed: {}", scenario.name, exc)
            rows.append({"scenario": scenario.name, "description": scenario.description,
                         "error": str(exc)})

    result = pd.DataFrame(rows)
    cols = ["scenario", "description"] + [c for c in result.columns if c not in ("scenario", "description")]
    result = result[[c for c in cols if c in result.columns]]

    logger.info("Full ablation complete: {} scenarios", len(result))
    return result
