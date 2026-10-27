"""Matched long-history label experiments with purged, nested model selection.

All periods are historical research, not an untouched new holdout. Old OOF
and evidence are read-only. Each model/fold has a durable checkpoint.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import yaml
from threadpoolctl import threadpool_limits

from quant_platform.core.alpha_models.nonlinear_alpha import (
    WalkForwardConfig, expanding_date_folds, build_learning_panel, filter_tradable_universe,
    select_usable_features, daily_rank_ic, _sample_training_rows, _daily_zscore,
)
from quant_platform.core.alpha_models.fair_comparison import inner_time_masks, fit_predict
from quant_platform.core.alpha_models.targets import TargetConfig, decision_volatility, neutral_scaled_targets
from quant_platform.core.data.research_contract import require_wrds_identity, FUNDAMENTAL_FEATURES
from quant_platform.core.utils.provenance import fingerprint, write_run_metadata


def digest_keys(rows):
    return hashlib.sha256(rows[["date", "ticker"]].astype(str).to_csv(index=False).encode()).hexdigest()


def snapshot(output):
    # This phase's actual source dependencies, excluding execution/replay code.
    sources = [*list((ROOT / "quant_platform/core/alpha_models").glob("*.py")),
               *list((ROOT / "quant_platform/core/data").rglob("*.py")),
               *list((ROOT / "quant_platform/core/utils").glob("*.py")), Path(__file__)]
    result = {}
    for source in sources:
        relative = source.relative_to(ROOT)
        dest = output / "source_snapshot" / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, dest)
        result[str(relative)] = fingerprint(source)["sha256"]
    return result


def prepare(config_path, output, smoke):
    if output.exists():
        raise ValueError("Use a new output directory; existing research is never overwritten")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    base_path = ROOT / config["base_config"]
    base = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    settings = {**base["research"], **config["research"]}
    horizon = settings["primary_horizon"]
    if smoke:
        settings.update(validation_days=20, max_train_rows=30000, seeds=[17])
    inputs = {name: ROOT / base["paths"][name] for name in ("factors", "prices", "universe", "catalog")}
    inputs.update(config=config_path, base_config=base_path)
    factors = pd.read_parquet(inputs["factors"])
    prices = pd.read_parquet(inputs["prices"], columns=["date", "ticker", "permno", "total_return", "adj_close", "is_terminal"])
    universe = pd.read_parquet(inputs["universe"])
    require_wrds_identity(universe, factors, prices)
    start, end = base["date_range"]["start"], base["date_range"]["end"]
    factors = factors.loc[factors.date.between(start, end)].copy()
    dates = pd.DatetimeIndex(sorted(factors.date.unique()))
    wf = WalkForwardConfig(horizon=horizon, min_train_days=settings["min_train_days"],
                           validation_days=settings["validation_days"], oos_start=settings["oos_start"])
    folds = list(expanding_date_folds(dates, wf))
    if smoke:
        folds = folds[:1]
    first = folds[0]
    cutoff = first["train_dates"][-settings["inner_days"] - horizon]
    eligible = filter_tradable_universe(factors, universe)
    selected, diagnostics = select_usable_features(eligible.loc[eligible.date.lt(cutoff)], base["feature_whitelist"],
        min_coverage=settings["min_feature_coverage"], return_diagnostics=True)
    schemas = {"price_only": [c for c in selected if c not in FUNDAMENTAL_FEATURES], "price_plus_financial": selected}
    candidate_count = 2 * len(schemas) * sum(len(v) for v in settings["model_configs"].values())
    if candidate_count > settings["max_candidate_configurations"]:
        raise ValueError("Label/model/information configurations exceed the declared budget")
    if not schemas["price_only"] or not FUNDAMENTAL_FEATURES.intersection(selected):
        raise ValueError("Both information groups need pre-validation usable features")
    panel, _ = build_learning_panel(factors, prices.loc[prices.date.between(start, end)], universe,
        feature_cols=selected, horizon=horizon, min_feature_coverage=0, missing_terminal_return="total_loss")
    target_cfg = TargetConfig(**config["target"])
    vol = decision_volatility(prices, target_cfg)
    context = factors[["date", "ticker", "decision_industry", "decision_log_size"]].merge(
        vol, on=["date", "ticker"], how="left", validate="one_to_one")
    neutral = neutral_scaled_targets(panel, context, horizon=horizon, config=target_cfg)
    panel = panel.merge(neutral, on=["date", "ticker"], how="left", validate="one_to_one")
    panel["label_reference_target"] = panel.target
    # Both labels train on exactly the same matured label sample. Inference
    # keys are preserved, including small/unknown industries and terminal stubs.
    panel["label_common_trainable"] = panel.target.notna() & panel.label_neutral_target.notna()
    panel["target"] = panel.target.where(panel.label_common_trainable)
    output.mkdir(parents=True)
    (output / "sampling").mkdir()
    diagnostics.to_csv(output / "locked_feature_schema.csv", index=False)
    panel.to_parquet(output / "learning_panel.parquet", index=False)
    context.to_parquet(output / "decision_context.parquet", index=False)
    audit = []
    common = schemas["price_only"]
    tiebreak = list(dict.fromkeys(c for cols in schemas.values() for c in cols))
    sort_keys = list(dict.fromkeys(["date", *common, "label_reference_target", *tiebreak]))
    end_dates = pd.to_datetime(panel.label_end)
    fold_records = []
    for fold in folds:
        outer = (panel.date.isin(fold["train_dates"]) & panel.target.notna() & end_dates.lt(fold["validation_start"])).to_numpy()
        inner, valid = inner_time_masks(panel, fold, settings["inner_days"])
        item = {k: v for k, v in fold.items() if k not in ("train_dates", "validation_dates")}
        item.update(train_dates=[str(x) for x in fold["train_dates"]],
                    validation_dates=[str(x) for x in fold["validation_dates"]])
        fold_records.append(item)
        for phase, mask in (("inner", inner), ("outer", outer)):
            ids = np.flatnonzero(mask)
            ordered = panel.iloc[ids][sort_keys].reset_index(drop=True).sort_values(sort_keys, kind="mergesort").index
            ids = _sample_training_rows(ids[ordered], settings["max_train_rows"], int(settings["seeds"][0]))
            np.save(output / f"sampling/fold_{fold['fold']}_{phase}.npy", ids)
            audit.append({"fold": fold["fold"], "phase": phase, "rows": len(ids),
                          "training_key_sha256": digest_keys(panel.iloc[ids]),
                          "max_label_end": str(end_dates.iloc[ids].max()),
                          "prediction_start": str(fold["validation_start"])})
        print(f"prepared fold {fold['fold']}: {fold['validation_start'].date()}—{fold['validation_end'].date()}", flush=True)
    (output / "folds.json").write_text(json.dumps(fold_records, indent=2, default=str), encoding="utf-8")
    pd.DataFrame(audit).to_csv(output / "sampling_audit.csv", index=False)
    plan = {"settings": settings, "schemas": schemas, "feature_lock_before": str(cutoff),
            "candidate_configurations": candidate_count,
            "config": config, "smoke": smoke, "label_variants": ["cross_sectional", "industry_size_vol"],
            "inference_rows": int(sum(panel.date.isin(f["validation_dates"]).sum() for f in folds)),
            "target_config": asdict(target_cfg), "source_sha256": snapshot(output),
            "common_trainable_fraction": float(panel.label_common_trainable.mean())}
    (output / "plan.json").write_text(json.dumps(plan, indent=2, default=str), encoding="utf-8")
    write_run_metadata(output / "prepare_metadata.json", stage="neutral_labels_long_history", inputs=inputs,
        params=plan, extra={"historical_review_not_untouched_holdout": True,
                            "fundamentals_is_point_in_time": False, "classification_is_point_in_time": False,
                            "production_admitted": False, "panel_output": fingerprint(output / "learning_panel.parquet")})


def train_cohort(output, label, block, model):
    output = Path(output)
    plan = json.loads((output / "plan.json").read_text())
    settings = plan["settings"]
    for path, wanted in plan["source_sha256"].items():
        if fingerprint(ROOT / path)["sha256"] != wanted:
            raise ValueError(f"Training source changed after preparation: {path}")
    panel = pd.read_parquet(output / "learning_panel.parquet")
    if label == "industry_size_vol":
        panel["target"] = panel.label_neutral_target.where(panel.label_common_trainable)
    folder = output / f"models/{label}/{block}/{model}"
    folder.mkdir(parents=True, exist_ok=True)
    features = plan["schemas"][block]
    tiebreak = list(dict.fromkeys(c for cols in plan["schemas"].values() for c in cols))
    registry, parts, audits = [], [], []
    for saved in json.loads((output / "folds.json").read_text()):
        fold = {**saved, "train_dates": pd.to_datetime(saved["train_dates"]),
                "validation_dates": pd.to_datetime(saved["validation_dates"]),
                "validation_start": pd.Timestamp(saved["validation_start"])}
        prefix = folder / f"fold_{fold['fold']}"
        completed = prefix.with_suffix(".json")
        if completed.exists():
            record = json.loads(completed.read_text())
            if fingerprint(prefix.with_suffix(".parquet"))["sha256"] != record["output_sha256"]:
                raise ValueError("Corrupt model checkpoint")
            parts.append(pd.read_parquet(prefix.with_suffix(".parquet")))
            registry += record["trials"]; audits.append(record["audit"])
            continue
        end = pd.to_datetime(panel.label_end)
        train = (panel.date.isin(fold["train_dates"]) & panel.target.notna() & end.lt(fold["validation_start"])).to_numpy()
        predict = panel.date.isin(fold["validation_dates"]).to_numpy()
        inner, valid = inner_time_masks(panel, fold, settings["inner_days"])
        trials, scores = [], []
        common_args = dict(model=model, seeds=settings["seeds"], max_rows=settings["max_train_rows"],
                           n_jobs=settings["n_jobs"], sampling_features=plan["schemas"]["price_only"],
                           sampling_tiebreak_features=tiebreak, sampling_reference_col="label_reference_target")
        with threadpool_limits(limits=settings["n_jobs"]):
            for candidate, params in enumerate(settings["model_configs"][model]):
                pred, importance, n = fit_predict(panel, features, inner, valid, params=params,
                    training_indices=np.load(output / f"sampling/fold_{fold['fold']}_inner.npy"), **common_args)
                validation = panel.loc[valid, ["date", "forward_return"]].assign(score=pred)
                ic = float(daily_rank_ic(validation, "score").mean())
                if not np.isfinite(ic):
                    raise ValueError("Unevaluable inner selection score")
                scores.append(ic)
                trials.append({"fold": fold["fold"], "phase": "inner", "candidate": candidate,
                               "params": params, "raw_return_rank_ic": ic, "rows": n,
                               "training_key_sha256": importance.attrs["training_key_sha256"]})
            params = settings["model_configs"][model][int(np.argmax(scores))]
            pred, importance, n = fit_predict(panel, features, train, predict, params=params,
                training_indices=np.load(output / f"sampling/fold_{fold['fold']}_outer.npy"), **common_args)
        local = panel.loc[predict, ["date", "ticker", "forward_return", "target", "label_end", "label_terminal_imputed"]].copy()
        local["prediction"] = _daily_zscore(pd.Series(pred, index=local.index), local.date).astype("float32")
        local["fold"] = fold["fold"]
        local.to_parquet(prefix.with_suffix(".parquet"), index=False)
        audit = {"fold": fold["fold"], "validation_start": str(fold["validation_start"]),
                 "validation_end": str(local.date.max()), "max_training_label_end": str(end[train].max()),
                 "max_inner_training_label_end": str(end[inner].max()), "inner_validation_start": str(panel.loc[valid, "date"].min()),
                 "training_key_sha256": importance.attrs["training_key_sha256"],
                 "prediction_key_sha256": digest_keys(local), "predictions": len(local), "params": params}
        trials.append({"fold": fold["fold"], "phase": "outer_refit", "params": params, "rows": n,
                       "training_key_sha256": importance.attrs["training_key_sha256"]})
        completed.write_text(json.dumps({"trials": trials, "audit": audit,
            "output_sha256": fingerprint(prefix.with_suffix(".parquet"))["sha256"]}, indent=2), encoding="utf-8")
        parts.append(local); registry += trials; audits.append(audit)
        print(f"{label}/{block}/{model} fold {fold['fold']}: {len(local):,} predictions", flush=True)
    oof = pd.concat(parts, ignore_index=True)
    if len(oof) != plan["inference_rows"] or oof.duplicated(["date", "ticker"]).any():
        raise ValueError("Incomplete or duplicate inference universe")
    oof.to_parquet(folder / "oof.parquet", index=False)
    pd.DataFrame(audits).to_csv(folder / "fold_audit.csv", index=False)
    (folder / "trial_registry.json").write_text(json.dumps(registry, indent=2), encoding="utf-8")
    return {"label": label, "block": block, "model": model, "prediction_rows": len(oof),
            "prediction_key_sha256": digest_keys(oof), "folds": len(audits), "trials": len(registry),
            "raw_rank_ic": float(daily_rank_ic(oof, "prediction").mean())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config/wrds_neutral_research.yaml")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--phase", choices=["prepare", "train", "all"], default="all")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    output = (args.output or ROOT / cfg["output"]).resolve()
    if args.phase in ("prepare", "all"):
        prepare(args.config, output, args.smoke)
    if args.phase in ("train", "all"):
        plan = json.loads((output / "plan.json").read_text())
        tasks = [(output, label, block, model) for label in plan["label_variants"]
                 for block in plan["schemas"] for model in ("ridge", "lightgbm")]
        results = []
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            for future in as_completed([pool.submit(train_cohort, *task) for task in tasks]):
                results.append(future.result())
                pd.DataFrame(results).to_csv(output / "model_summary.csv", index=False)
        if len({r["prediction_key_sha256"] for r in results}) != 1:
            raise ValueError("Model/label/information groups have different prediction keys")
        (output / "training_completion.json").write_text(json.dumps({"cohorts": results,
            "smoke": plan["smoke"], "historical_review_not_untouched_holdout": True}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
