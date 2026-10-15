#!/usr/bin/env python
"""Compare OOF alphas with one portfolio and transaction-cost convention.

The script consumes the out-of-fold predictions produced by
``evaluate_nonlinear_alpha.py`` and evaluates LightGBM, MLP, and MOM_12_1M on
the exact same stock-date sample.  Every input feature is lagged by at least
one trading day; eligibility known after date ``t-1`` is used for a date ``t``
close trade, which begins earning returns from ``t`` to ``t+1``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from quant_platform.core.alpha_models.nonlinear_alpha import evaluate_alpha
from quant_platform.core.data.universe import (
    lag_universe_eligibility,
    require_point_in_time_universe,
)
from quant_platform.core.execution.backtest.pipeline import run_trading_pipeline
from quant_platform.core.execution.cost_models.us_equity import CostModel
from quant_platform.core.portfolio.construction import (
    PortfolioConfig,
    PortfolioMode,
    WeightScheme,
)
from quant_platform.core.portfolio.pipeline import run_portfolio_pipeline


SIGNALS = {
    "LightGBM": "lightgbm_alpha",
    "MLP": "mlp_alpha",
    "MOM_12_1M": "MOM_12_1M",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--oof",
        type=Path,
        default=ROOT
        / "data/experiments/nonlinear_alpha_corrected_20260731/oof_predictions.parquet",
    )
    parser.add_argument(
        "--prices",
        type=Path,
        default=ROOT / "data/prices/prices.parquet",
    )
    parser.add_argument(
        "--universe",
        type=Path,
        default=ROOT / "data/universe/universe.parquet",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "data/experiments/nonlinear_alpha_corrected_20260731/after_costs",
    )
    parser.add_argument(
        "--frequencies",
        nargs="+",
        choices=["D", "W", "M"],
        default=["W"],
    )
    parser.add_argument("--initial-capital", type=float, default=1_000_000.0)
    parser.add_argument("--commission-per-share", type=float, default=0.005)
    parser.add_argument("--commission-min", type=float, default=1.0)
    parser.add_argument("--slippage-bps", type=float, default=5.0)
    parser.add_argument("--impact-coeff", type=float, default=0.1)
    parser.add_argument("--borrow-rate-annual", type=float, default=0.005)
    parser.add_argument("--max-participation-rate", type=float, default=0.10)
    parser.add_argument(
        "--max-abs-net-exposure",
        type=float,
        default=0.02,
        help="Fail if realised net exposure exceeds this tolerance.",
    )
    parser.add_argument(
        "--allow-non-pit-research",
        action="store_true",
        help="Explicitly permit a survivorship-biased research universe.",
    )
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_upstream_metadata(
    args: argparse.Namespace,
    *,
    prices_sha256: str,
    universe_sha256: str,
) -> tuple[dict[str, object], Path, int]:
    metadata_path = args.oof.parent / "run_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Missing upstream run metadata: {metadata_path}"
        )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    evaluation = metadata.get("evaluation", {})
    horizon = int(evaluation.get("return_horizon_days", 0))
    if horizon != 1:
        raise ValueError(
            "Transaction-cost comparison currently requires a 1-day horizon"
        )
    inputs = metadata.get("inputs", {})
    expected_prices = (inputs.get("prices") or {}).get("sha256")
    expected_universe = (inputs.get("universe") or {}).get("sha256")
    if expected_prices != prices_sha256:
        raise ValueError("OOF metadata price hash does not match --prices")
    if expected_universe != universe_sha256:
        raise ValueError("OOF metadata universe hash does not match --universe")

    models = metadata.get("models", {})
    if not models:
        raise ValueError("OOF metadata contains no walk-forward model configs")
    for model_name, config in models.items():
        walk_forward = config.get("walk_forward", {})
        model_horizon = int(walk_forward.get("horizon", 0))
        purge = int(walk_forward.get("effective_purge_days", 0))
        if model_horizon != horizon or purge < horizon:
            raise ValueError(
                f"{model_name} metadata has an invalid horizon/purge contract"
            )

    from quant_platform.core.signals.registry import FactorRegistry

    registry = FactorRegistry()
    registry.discover()
    factor_metadata = registry.list_factors().set_index("name")
    feature_columns = list(metadata.get("feature_columns", []))
    unknown = sorted(set(feature_columns) - set(factor_metadata.index))
    if unknown:
        raise ValueError(f"Unknown feature metadata: {unknown}")
    noncausal = factor_metadata.loc[feature_columns]
    noncausal = noncausal.loc[noncausal["lag"].astype(int) < 1]
    if not noncausal.empty:
        raise ValueError(
            "Same-close execution requires every feature to have lag >= 1: "
            f"{noncausal.index.tolist()}"
        )
    if (
        not args.allow_non_pit_research
        and bool(
            metadata.get("provenance", {}).get(
                "allow_non_pit_research",
                False,
            )
        )
    ):
        raise ValueError("Upstream OOF run explicitly used non-PIT research data")
    return metadata, metadata_path, horizon


def _attach_current_forward_returns(
    signals: pd.DataFrame,
    prices: pd.DataFrame,
    horizon: int,
) -> tuple[pd.DataFrame, pd.Timestamp]:
    price_col = "adj_close" if "adj_close" in prices.columns else "close"
    calendar = pd.DatetimeIndex(
        prices["date"].dropna().unique()
    ).sort_values()
    if len(calendar) <= horizon:
        raise ValueError("Insufficient price dates for forward returns")
    calendar_frame = pd.DataFrame({
        "date": calendar,
        "_label_date": pd.Series(
            calendar,
            index=calendar,
        ).shift(-horizon).to_numpy(),
    })
    current = prices[["date", "ticker", price_col]].rename(
        columns={price_col: "_current_price"}
    )
    future = prices[["date", "ticker", price_col]].rename(
        columns={"date": "_label_date", price_col: "_future_price"}
    )
    labels = (
        signals[["date", "ticker"]]
        .merge(current, on=["date", "ticker"], how="left", validate="one_to_one")
        .merge(calendar_frame, on="date", how="left", validate="many_to_one")
        .merge(
            future,
            on=["_label_date", "ticker"],
            how="left",
            validate="one_to_one",
        )
    )
    labels["forward_return"] = (
        labels["_future_price"] / labels["_current_price"] - 1.0
    )
    refreshed = signals.drop(columns=["forward_return"], errors="ignore").merge(
        labels[["date", "ticker", "forward_return"]],
        on=["date", "ticker"],
        how="left",
        validate="one_to_one",
    )
    max_signal_date = pd.Timestamp(refreshed["date"].max())
    position = int(calendar.searchsorted(max_signal_date, side="left"))
    if position >= len(calendar) or calendar[position] != max_signal_date:
        raise ValueError("Last OOF signal date is absent from current prices")
    end_position = position + horizon
    if end_position >= len(calendar):
        raise ValueError("Prices do not cover the final OOF holding horizon")
    return refreshed, pd.Timestamp(calendar[end_position])


def _load_inputs(
    args: argparse.Namespace,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.Timestamp,
    dict[str, object],
]:
    for path in (args.oof, args.prices, args.universe):
        if not path.exists():
            raise FileNotFoundError(path)

    oof = pd.read_parquet(args.oof)
    required_oof = {"date", "ticker", "forward_return", *SIGNALS.values()}
    if missing := required_oof.difference(oof.columns):
        raise ValueError(f"OOF predictions are missing columns: {sorted(missing)}")
    oof["date"] = pd.to_datetime(oof["date"])
    if oof.duplicated(["date", "ticker"]).any():
        raise ValueError("OOF predictions contain duplicate date/ticker keys")

    prices = pd.read_parquet(args.prices)
    prices["date"] = pd.to_datetime(prices["date"])
    if prices.duplicated(["date", "ticker"]).any():
        raise ValueError("Price data contain duplicate date/ticker keys")
    universe_sha256 = _sha256(args.universe)
    prices_sha256 = _sha256(args.prices)
    upstream, metadata_path, horizon = _validate_upstream_metadata(
        args,
        prices_sha256=prices_sha256,
        universe_sha256=universe_sha256,
    )
    oof = oof.dropna(subset=[*SIGNALS.values()]).copy()
    oof, evaluation_end = _attach_current_forward_returns(
        oof,
        prices,
        horizon,
    )
    signal_dates = pd.DatetimeIndex(oof["date"].unique()).sort_values()
    available_price_dates = pd.DatetimeIndex(
        prices["date"].unique()
    ).sort_values()
    first_location = int(
        available_price_dates.searchsorted(signal_dates.min(), side="left")
    )
    history_start = available_price_dates[max(0, first_location - 63)]
    # Retain pre-OOF history for lagged ADV/volatility estimates.  The
    # portfolio pipeline itself starts on the first valid rebalance date, and
    # the final holding horizon remains available after the last signal.
    prices = prices.loc[
        prices["date"].between(history_start, evaluation_end)
    ].copy()

    universe = pd.read_parquet(args.universe)
    universe["date"] = pd.to_datetime(universe["date"])
    if not args.allow_non_pit_research:
        require_point_in_time_universe(
            universe,
            context="transaction-cost alpha comparison",
        )
    universe = lag_universe_eligibility(universe, periods=1)
    eligible = universe.loc[
        universe["pass_all_filters"].fillna(False).astype(bool)
    ].copy()
    eligible_keys = eligible[["date", "ticker"]].drop_duplicates()
    oof = oof.merge(
        eligible_keys,
        on=["date", "ticker"],
        how="inner",
        validate="one_to_one",
    )
    if oof.empty:
        raise ValueError("No common OOF/universe rows remain")
    universe = eligible.merge(
        oof[["date", "ticker"]],
        on=["date", "ticker"],
        how="inner",
        validate="one_to_one",
    )
    upstream["_validated_metadata_path"] = str(metadata_path.resolve())
    upstream["_validated_metadata_sha256"] = _sha256(metadata_path)
    upstream["_current_prices_sha256"] = prices_sha256
    upstream["_current_universe_sha256"] = universe_sha256
    return oof, prices, universe, evaluation_end, upstream


def main() -> None:
    args = _parse_args()
    if args.initial_capital <= 0:
        raise ValueError("--initial-capital must be positive")
    if not 0 < args.max_participation_rate <= 1:
        raise ValueError("--max-participation-rate must be in (0, 1]")
    if not 0 < args.max_abs_net_exposure < 1:
        raise ValueError("--max-abs-net-exposure must be in (0, 1)")
    if len(set(args.frequencies)) != len(args.frequencies):
        raise ValueError("--frequencies must not contain duplicates")

    args.output.mkdir(parents=True, exist_ok=True)
    oof, prices, universe, evaluation_end, upstream = _load_inputs(args)
    oof.to_parquet(args.output / "eligible_oof_signals.parquet", index=False)
    cost_model = CostModel(
        commission_per_share=args.commission_per_share,
        commission_min=args.commission_min,
        slippage_bps=args.slippage_bps,
        impact_coeff=args.impact_coeff,
        borrow_rate_annual=args.borrow_rate_annual,
        max_participation_rate=args.max_participation_rate,
    )

    rows: list[dict[str, object]] = []
    for frequency in args.frequencies:
        for alpha_name, signal_col in SIGNALS.items():
            print(
                f"Backtesting {alpha_name} at {frequency} frequency ...",
                flush=True,
            )
            signal = oof[["date", "ticker", signal_col]].copy()
            signal = signal.rename(columns={signal_col: "signal"})
            portfolio_dir = args.output / frequency / alpha_name
            portfolio = run_portfolio_pipeline(
                method="quantile",
                signal_col="signal",
                config=PortfolioConfig(
                    mode=PortfolioMode.LONG_SHORT,
                    weight_scheme=WeightScheme.EQUAL,
                    max_stock_weight=0.05,
                    # The bundled sector map is a current classification and
                    # is incomplete historically.  Applying it would add a
                    # second non-PIT input, so this comparison deliberately
                    # uses only stock-level and tradability constraints.
                    max_sector_weight=None,
                    n_quantiles=5,
                    long_quantile=5,
                    short_quantile=1,
                    rebalance_freq=frequency,
                    initial_capital=args.initial_capital,
                ),
                output_path=portfolio_dir / "weights.parquet",
                factors_override=signal,
                prices_override=prices,
                universe_override=universe,
                evaluation_end=evaluation_end,
            )
            if portfolio["daily_weights"].empty:
                raise RuntimeError(
                    f"{alpha_name}/{frequency} produced no portfolio weights"
                )

            trading = run_trading_pipeline(
                weights_override=portfolio["daily_weights"],
                prices_override=prices,
                cost_model=cost_model,
                initial_capital=args.initial_capital,
                output_dir=portfolio_dir / "backtest",
                benchmark_market=None,
            )
            ic_sample = oof.dropna(subset=["forward_return"])
            ic_metrics, _ = evaluate_alpha(ic_sample, signal_col)
            pnl = trading["daily_pnl"]
            perf = trading["performance"]
            trades = trading["trades"]
            max_abs_net_exposure = float(pnl["net_exposure"].abs().max())
            if max_abs_net_exposure > args.max_abs_net_exposure:
                raise RuntimeError(
                    f"{alpha_name}/{frequency} realised net exposure "
                    f"{max_abs_net_exposure:.2%}, above "
                    f"{args.max_abs_net_exposure:.2%}"
                )
            requested_trade_notional = float(
                trades["notional_trade"].sum()
            ) if len(trades) else 0.0
            executed_trade_notional = float(
                trades["executed_notional"].sum()
            ) if len(trades) else 0.0
            rows.append(
                {
                    "alpha": alpha_name,
                    "rebalance_frequency": frequency,
                    "mean_rank_ic": ic_metrics["mean_rank_ic"],
                    "rank_ic_ir_ann": ic_metrics["rank_ic_ir_ann"],
                    "gross_ann_return": perf.get("gross_ann_return"),
                    "gross_sharpe": perf.get("gross_sharpe"),
                    "net_ann_return": perf.get("net_ann_return"),
                    "net_sharpe": perf.get("net_sharpe"),
                    "net_max_drawdown": perf.get("net_max_drawdown"),
                    "mean_one_way_turnover": float(
                        pnl["executed_turnover"].mean() / 2.0
                    ),
                    "mean_gross_traded_notional": float(
                        pnl["executed_turnover"].mean()
                    ),
                    "mean_actual_gross_exposure": float(
                        pnl["gross_exposure"].mean()
                    ),
                    "max_abs_actual_net_exposure": max_abs_net_exposure,
                    "total_trading_cost_bps": perf.get(
                        "total_trading_cost_bps"
                    ),
                    "total_borrow_cost_bps": perf.get(
                        "total_borrow_cost_bps"
                    ),
                    "full_order_fill_rate": (
                        float(trades["fill_status"].eq("filled").mean())
                        if len(trades)
                        else 0.0
                    ),
                    "notional_fill_rate": (
                        executed_trade_notional / requested_trade_notional
                        if requested_trade_notional > 0
                        else 1.0
                    ),
                    "n_oof_rows": len(oof),
                    "n_ic_rows": len(ic_sample),
                    "n_oof_dates": oof["date"].nunique(),
                }
            )

    summary = pd.DataFrame(rows).sort_values(
        ["rebalance_frequency", "net_sharpe"],
        ascending=[True, False],
    )
    summary.to_csv(args.output / "alpha_after_cost_comparison.csv", index=False)
    metadata = {
        "timing": (
            "features and eligibility known by t-1 close -> trade at t close "
            "-> first return earned t to t+1"
        ),
        "comparison_sample": (
            "intersection of all three OOF signals and one-session-lagged "
            "tradable eligibility; forward-return availability is used only "
            "for IC, never for portfolio selection"
        ),
        "portfolio": {
            "construction": "equal-weight top/bottom quintiles",
            "gross_exposure": 2.0,
            "target_net_exposure": 0.0,
            "realised_net_exposure_tolerance": args.max_abs_net_exposure,
            "sector_constraint": None,
            "frequencies": args.frequencies,
            "evaluation_end": str(evaluation_end.date()),
        },
        "cost_model": cost_model.describe(),
        "data_contract": (
            "research_non_pit_explicit"
            if args.allow_non_pit_research
            else "point_in_time_required"
        ),
        "inputs": {
            "oof": {"path": str(args.oof.resolve()), "sha256": _sha256(args.oof)},
            "prices": {
                "path": str(args.prices.resolve()),
                "sha256": _sha256(args.prices),
            },
            "universe": {
                "path": str(args.universe.resolve()),
                "sha256": _sha256(args.universe),
            },
            "upstream_run_metadata": {
                "path": upstream["_validated_metadata_path"],
                "sha256": upstream["_validated_metadata_sha256"],
            },
        },
        "upstream_evaluation": upstream.get("evaluation", {}),
        "upstream_models": upstream.get("models", {}),
    }
    (args.output / "after_cost_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    print(summary.to_string(index=False), flush=True)
    print(f"Saved to {args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
