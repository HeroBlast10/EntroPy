#!/usr/bin/env python
"""Train and compare LightGBM/MLP alphas on EntroPy's local factor panel."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from dataclasses import asdict
from itertools import combinations
from pathlib import Path
from typing import Optional

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from quant_platform.core.alpha_models.multi_factor import infer_factor_metadata
from quant_platform.core.alpha_models.nonlinear_alpha import (
    LightGBMConfig,
    MLPConfig,
    WalkForwardConfig,
    block_bootstrap_ic_delta,
    build_learning_panel,
    evaluate_alpha,
    filter_tradable_universe,
    run_walk_forward_oof,
    select_usable_features,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--factors", type=Path, default=ROOT / "data/factors/factors.parquet")
    parser.add_argument("--prices", type=Path, default=ROOT / "data/prices/prices.parquet")
    parser.add_argument("--universe", type=Path, default=ROOT / "data/universe/universe.parquet")
    parser.add_argument(
        "--fundamentals",
        type=Path,
        default=ROOT / "data/fundamentals/fundamentals.parquet",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data/experiments/nonlinear_alpha",
    )
    parser.add_argument("--models", nargs="+", choices=["lightgbm", "mlp"], default=["lightgbm", "mlp"])
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--min-train-days", type=int, default=756)
    parser.add_argument("--validation-days", type=int, default=252)
    parser.add_argument("--n-splits", type=int, default=4)
    parser.add_argument("--lightgbm-trees", type=int, default=120)
    parser.add_argument(
        "--lightgbm-max-train-rows",
        type=int,
        default=0,
        help="Reproducible row cap for LightGBM; 0 uses all rows.",
    )
    parser.add_argument("--mlp-epochs", type=int, default=3)
    parser.add_argument(
        "--mlp-max-train-rows",
        type=int,
        default=500_000,
        help="Reproducible sample from the expanding history; 0 uses every row.",
    )
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument(
        "--allow-non-pit-research",
        action="store_true",
        help="Explicitly allow research on data without point-in-time provenance.",
    )
    return parser.parse_args()


def _fingerprint(path: Path) -> dict[str, object]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": digest.hexdigest(),
    }


def _git_head() -> Optional[str]:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _package_versions(names: tuple[str, ...]) -> dict[str, Optional[str]]:
    versions: dict[str, Optional[str]] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _validate_point_in_time_contract(
    args: argparse.Namespace,
    feature_cols: list[str],
    universe: pd.DataFrame,
) -> None:
    if args.allow_non_pit_research:
        print("WARNING: explicitly allowing non-PIT research inputs", flush=True)
        return
    from quant_platform.core.data.universe import require_point_in_time_universe

    require_point_in_time_universe(universe, context="non-linear alpha research")
    fundamental_features = {
        "EARNINGS_YIELD", "BOOK_TO_MARKET", "GROSS_PROFITABILITY",
        "ASSET_GROWTH", "ACCRUALS", "PIOTROSKI_F8",
    }
    if fundamental_features.intersection(feature_cols):
        from quant_platform.core.data.fundamentals import (
            require_point_in_time_fundamentals,
        )

        fundamentals = pd.read_parquet(args.fundamentals)
        require_point_in_time_fundamentals(
            fundamentals,
            context="non-linear alpha research",
        )


def main() -> None:
    args = _parse_args()
    if len(set(args.models)) != len(args.models):
        raise ValueError("--models must not contain duplicates")
    if (
        args.horizon <= 0
        or args.min_train_days <= 0
        or args.validation_days <= 0
        or args.n_splits <= 0
        or args.lightgbm_trees <= 0
        or args.mlp_epochs <= 0
        or args.mlp_max_train_rows < 0
        or args.lightgbm_max_train_rows < 0
    ):
        raise ValueError("horizons, folds, trees and epochs must be positive")
    factors = pd.read_parquet(args.factors)
    universe = pd.read_parquet(args.universe)
    import pyarrow.parquet as pq

    price_schema = set(pq.ParquetFile(args.prices).schema.names)
    price_columns = [
        col for col in ("date", "ticker", "adj_close", "close")
        if col in price_schema
    ]
    if not {"date", "ticker"}.issubset(price_columns) or not {
        "adj_close", "close"
    }.intersection(price_columns):
        raise ValueError("Price parquet must contain date, ticker and adj_close or close")
    prices = pd.read_parquet(args.prices, columns=price_columns)
    factor_dates = pd.DatetimeIndex(pd.to_datetime(factors["date"]).unique()).sort_values()
    schema_offset = args.n_splits * args.validation_days + args.horizon
    if len(factor_dates) <= schema_offset:
        raise ValueError("Insufficient dates to lock the feature schema before validation")
    feature_schema_cutoff = factor_dates[-schema_offset]
    tradable_factors = filter_tradable_universe(
        factors,
        universe,
        context="factors",
    )
    feature_cols, feature_diagnostics = select_usable_features(
        tradable_factors.loc[
            pd.to_datetime(tradable_factors["date"]) < feature_schema_cutoff
        ],
        min_coverage=0.80,
        return_diagnostics=True,
    )
    if "MOM_12_1M" in factors.columns and "MOM_12_1M" not in feature_cols:
        raise ValueError("MOM_12_1M is not usable on the locked training schema")
    # Validate before touching the filesystem. The previous ordering wrote the
    # diagnostics first, so any later contract violation overwrote a previous
    # run's artifact with a partial output and left no record that it failed.
    _validate_point_in_time_contract(args, feature_cols, universe)
    args.output.mkdir(parents=True, exist_ok=True)
    feature_diagnostics.to_csv(
        args.output / "feature_selection_diagnostics.csv",
        index=False,
    )
    panel, feature_cols = build_learning_panel(
        factors,
        prices,
        universe,
        feature_cols=feature_cols,
        horizon=args.horizon,
        min_feature_coverage=0.0,
    )
    del factors, prices, universe, tradable_factors
    gc.collect()
    directions, _ = infer_factor_metadata(feature_cols)

    base_wf = dict(
        horizon=args.horizon,
        min_train_days=args.min_train_days,
        validation_days=args.validation_days,
        n_splits=args.n_splits,
        seed=args.seed,
    )
    model_oof: dict[str, pd.DataFrame] = {}
    importance_parts: list[pd.DataFrame] = []
    fold_metadata: list[dict[str, object]] = []
    model_configs: dict[str, object] = {}
    for model in args.models:
        max_rows = (
            (
                None
                if args.lightgbm_max_train_rows == 0
                else args.lightgbm_max_train_rows
            )
            if model == "lightgbm"
            else (
                None
                if args.mlp_max_train_rows == 0
                else args.mlp_max_train_rows
            )
        )
        wf = WalkForwardConfig(**base_wf, max_train_rows=max_rows)
        lgb_config = LightGBMConfig(n_estimators=args.lightgbm_trees)
        mlp_config = MLPConfig(epochs=args.mlp_epochs)
        print(f"Training {model} with {len(feature_cols)} features ...", flush=True)
        oof, importance, metadata = run_walk_forward_oof(
            panel,
            feature_cols,
            model=model,
            walk_forward=wf,
            lightgbm=lgb_config,
            mlp=mlp_config,
        )
        model_oof[model] = oof[["date", "ticker", "prediction"]].copy()
        importance_parts.append(importance)
        fold_metadata.extend(metadata)
        model_configs[model] = {
            "walk_forward": wf.to_dict(),
            "model": asdict(lgb_config if model == "lightgbm" else mlp_config),
        }
        del oof, importance
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    first_model = args.models[0]
    first_prediction = model_oof[first_model]
    validation_keys = first_prediction[["date", "ticker"]]
    common = validation_keys.merge(
        panel[["date", "ticker", "forward_return", "target", *feature_cols]],
        on=["date", "ticker"],
        how="inner",
        validate="one_to_one",
    )
    date_to_fold = {}
    for item in fold_metadata:
        if item["model"] != first_model:
            continue
        fold_dates = pd.date_range(
            item["validation_start"],
            item["validation_end"],
            freq="D",
        )
        date_to_fold.update({date: item["fold"] for date in fold_dates})
    common["fold"] = common["date"].map(date_to_fold).astype("int16")
    for model, oof in model_oof.items():
        common = common.merge(
            oof.rename(
                columns={"prediction": f"{model}_alpha"}
            ),
            on=["date", "ticker"],
            how="inner",
            validate="one_to_one",
        )

    equal_weight_sum = pd.Series(0.0, index=common.index, dtype="float32")
    for feature in feature_cols:
        equal_weight_sum = equal_weight_sum + (
            common[feature].astype("float32") * directions.get(feature, 1)
        )
    common["equal_weight_alpha"] = equal_weight_sum / len(feature_cols)

    signals = [f"{model}_alpha" for model in args.models]
    signals += ["equal_weight_alpha", *feature_cols]
    comparison_rows: list[dict[str, object]] = []
    daily_ic_parts: list[pd.Series] = []
    daily_ic_lookup: dict[str, pd.Series] = {}
    for signal in signals:
        evaluation_signal = signal
        if signal in feature_cols:
            evaluation_signal = f"__oriented_{signal}"
            common[evaluation_signal] = common[signal] * directions.get(signal, 1)
        metrics, ic = evaluate_alpha(
            common,
            evaluation_signal,
            horizon=args.horizon,
        )
        display_name = signal.upper() if signal.endswith("_alpha") else signal
        comparison_rows.append({"alpha": display_name, **metrics})
        daily_ic_lookup[display_name] = ic
        daily_ic_parts.append(ic.rename(display_name))

    comparison = pd.DataFrame(comparison_rows).sort_values(
        ["mean_rank_ic", "ls_gross_sharpe"],
        ascending=False,
    )
    baseline_name = "EQUAL_WEIGHT_ALPHA"
    comparison_pairs: list[tuple[str, str]] = [
        (f"{model}_alpha".upper(), baseline_name)
        for model in args.models
    ]
    if "MOM_12_1M" in daily_ic_lookup:
        comparison_pairs.extend(
            (f"{model}_alpha".upper(), "MOM_12_1M")
            for model in args.models
        )
    comparison_pairs.extend(
        combinations(
            [f"{model}_alpha".upper() for model in args.models],
            2,
        )
    )
    delta_rows = []
    for challenger, baseline in comparison_pairs:
        delta_rows.append({
            "challenger": challenger,
            "baseline": baseline,
            **block_bootstrap_ic_delta(
                daily_ic_lookup[challenger],
                daily_ic_lookup[baseline],
                seed=args.seed,
            ),
        })

    output_oof_cols = [
        "date", "ticker", "forward_return", "target", "fold",
        *[f"{model}_alpha" for model in args.models],
        "equal_weight_alpha",
    ]
    if "MOM_12_1M" in common.columns:
        output_oof_cols.append("MOM_12_1M")
    common[output_oof_cols].to_parquet(args.output / "oof_predictions.parquet", index=False)
    comparison.to_csv(args.output / "alpha_comparison.csv", index=False)
    pd.concat(daily_ic_parts, axis=1).to_csv(args.output / "daily_rank_ic.csv", index_label="date")
    pd.DataFrame(delta_rows).to_csv(args.output / "paired_ic_delta.csv", index=False)
    importance = pd.concat(importance_parts, ignore_index=True)
    importance.to_csv(args.output / "feature_importance_by_fold.csv", index=False)
    (
        importance.groupby(["model", "feature"], as_index=False)["importance"]
        .mean()
        .sort_values(["model", "importance"], ascending=[True, False])
        .to_csv(args.output / "feature_importance_mean.csv", index=False)
    )
    (args.output / "fold_metadata.json").write_text(
        json.dumps(fold_metadata, indent=2),
        encoding="utf-8",
    )
    run_metadata = {
        "inputs": {
            "factors": _fingerprint(args.factors),
            "prices": _fingerprint(args.prices),
            "universe": _fingerprint(args.universe) if args.universe.exists() else None,
            "fundamentals": (
                _fingerprint(args.fundamentals)
                if args.fundamentals.exists()
                else None
            ),
        },
        "feature_columns": feature_cols,
        "feature_schema_cutoff": str(feature_schema_cutoff.date()),
        "feature_directions": directions,
        "models": model_configs,
        "provenance": {
            "git_head": _git_head(),
            "python": sys.version,
            "platform": platform.platform(),
            "package_versions": _package_versions((
                "numpy", "pandas", "pyarrow", "scikit-learn",
                "lightgbm", "torch",
            )),
            "allow_non_pit_research": bool(args.allow_non_pit_research),
        },
        "evaluation": {
            "return_horizon_days": args.horizon,
            "prediction_postprocess": "daily z-score",
            "comparison_sample": "same OOF dates and securities",
            "transaction_costs_included": False,
        },
    }
    (args.output / "run_metadata.json").write_text(
        json.dumps(run_metadata, indent=2),
        encoding="utf-8",
    )
    print("\nTop alpha comparison:\n", comparison.head(12).to_string(index=False), flush=True)
    print(f"\nSaved research artifacts to {args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
