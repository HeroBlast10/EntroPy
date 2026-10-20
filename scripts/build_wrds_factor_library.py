"""Migrate the complete registry to a calendar-aligned, PERMNO WRDS panel."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from loguru import logger
from quant_platform.core.data.industry import historical_industries
from quant_platform.core.data.wrds import FACTOR_DIRECTIONS
from quant_platform.core.signals.feature_cache import PriceFeatureCache
from quant_platform.core.signals.registry import FactorRegistry
from quant_platform.core.utils.provenance import fingerprint, write_run_metadata, WRDS_CODE_PATHS

_PRICES = _CACHE = _REGISTRY = _FUNDAMENTALS = None


def implementation_hash():
    files = sorted((ROOT / "quant_platform/core/signals").rglob("*.py"))
    files += sorted((ROOT / "quant_platform/core/data").glob("*.py"))
    files += [Path(__file__)]
    digest = hashlib.sha256()
    for path in files:
        digest.update(str(path.relative_to(ROOT)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def calendarize(prices):
    """Insert missing exchange sessions only within each observed lifespan."""
    dates = pd.DatetimeIndex(sorted(prices.date.unique()))
    spans = prices.groupby("permno").date.agg(["min", "max"])
    index = pd.MultiIndex.from_product([dates, spans.index], names=["date", "permno"])
    work = prices.set_index(["date", "permno"]).reindex(index).reset_index()
    start, end = work.permno.map(spans["min"]), work.permno.map(spans["max"])
    work = work[work.date.between(start, end)].copy().reset_index(drop=True)
    work["ticker"] = "PERMNO:" + work.permno.astype(str)
    work["is_tradable"] = work.is_tradable.eq(True).fillna(False)
    work["is_terminal"] = work.is_terminal.eq(True).fillna(False)
    return work


def prepare(prices, universe, history):
    work = calendarize(prices)
    industry = historical_industries(work, history)
    work = work.merge(industry, on=["date", "permno"], how="left", validate="one_to_one")
    keys = pd.MultiIndex.from_frame(universe[["date", "permno"]])
    eligible = pd.Series(universe.pass_all_filters.to_numpy(), index=keys)
    work["_eligible"] = eligible.reindex(pd.MultiIndex.from_frame(work[["date", "permno"]])).to_numpy()
    work = work.sort_values(["permno", "date"]).reset_index(drop=True)
    g = work.groupby("permno", sort=False)
    work["risk_eligible"] = g["_eligible"].shift(1).eq(True).fillna(False)
    work["decision_industry"] = g["industry"].shift(1)
    work["decision_log_size"] = np.log(g["company_market_cap"].shift(1).where(lambda s: s.gt(0)))
    work["decision_financial_or_real_estate"] = g["financial_or_real_estate"].shift(1)
    for c in ("volume", "open", "high", "low", "close"):
        work[c] = work[c].where(work.is_tradable)
    for c in ("adj_close", "total_return", "close", "open", "high", "low", "volume", "shares_outstanding", "market_cap"):
        work[c] = pd.to_numeric(work[c], errors="coerce").astype("float64")
    return work.sort_values(["date", "permno"]).reset_index(drop=True)


def initialize(prepared, p0_factors):
    global _PRICES, _CACHE, _REGISTRY, _FUNDAMENTALS
    logger.remove()
    _PRICES = pd.read_parquet(prepared)
    _CACHE = PriceFeatureCache(_PRICES)
    _REGISTRY = FactorRegistry()
    _REGISTRY.discover()
    _FUNDAMENTALS = pd.read_parquet(p0_factors)


def compute_part(name, output, source_hash, prepared_hash):
    start = time.monotonic()
    p = _PRICES
    meta = _REGISTRY.get(name).meta
    if name in set(FACTOR_DIRECTIONS) - {"MOM_12_1M"}:
        series = p[["date", "permno"]].merge(_FUNDAMENTALS[["date", "permno", name]],
                 on=["date", "permno"], how="left", validate="one_to_one")[name]
        # The version-aware financial adapter already applies one session lag.
    else:
        instance = _REGISTRY.get(name)()
        raw = instance._call_compute(p, None, _feature_cache=_CACHE)
        series = instance._align_raw_signal(raw, p, name).replace([np.inf, -np.inf], np.nan)
        series = series.where(p.is_tradable)
        temp = p[["permno", "date"]].assign(value=series).sort_values(["permno", "date"])
        series = temp.groupby("permno").value.shift(meta.lag).reindex(p.index)
    series = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    path = Path(output) / f"{name}.parquet"
    pd.DataFrame({name: series.astype("float32")}).to_parquet(path, index=False)
    result = {"factor": name, "implementation_sha256": source_hash, "prepared_sha256": prepared_hash,
              "output_sha256": fingerprint(path)["sha256"], "seconds": time.monotonic()-start,
              "finite_rows": int(series.notna().sum()), "lag_sessions": meta.lag,
              "direction": meta.direction, "signal_type": meta.signal_type, "category": meta.category}
    path.with_suffix(".json").write_text(json.dumps(result, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "data/wrds/batch2")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--reuse-parts", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.jobs <= 4:
        raise ValueError("jobs must be 1..4")
    dest = args.output
    dest.mkdir(parents=True, exist_ok=True)
    prior = ROOT / "data/wrds/p0"
    inputs = {"prices": prior / "prices.parquet", "universe": prior / "universe.parquet",
              "fundamentals": prior / "factors.parquet", "industry": dest / "crsp_industry_history.parquet"}
    source_hash = implementation_hash()
    source_fingerprints = {k: fingerprint(v)["sha256"] for k,v in inputs.items()}
    prepared_path = dest / "prepared_prices.parquet"
    prepared_meta = dest / "prepared_inputs.json"
    reuse_prepared = args.reuse_parts and prepared_path.exists() and prepared_meta.exists()
    if reuse_prepared:
        reuse_prepared = json.loads(prepared_meta.read_text()) == {"inputs": source_fingerprints, "implementation": source_hash}
    if not reuse_prepared:
        work = prepare(pd.read_parquet(inputs["prices"]), pd.read_parquet(inputs["universe"]), pd.read_parquet(inputs["industry"]))
        work.to_parquet(prepared_path, index=False)
        prepared_meta.write_text(json.dumps({"inputs": source_fingerprints, "implementation": source_hash}))
        del work
    prepared_hash = fingerprint(prepared_path)["sha256"]
    logger.remove()
    reg = FactorRegistry(); reg.discover()
    catalog = reg.list_factors()
    parts = dest / "factor_parts"; parts.mkdir(exist_ok=True)
    needed, results = [], []
    for name in catalog.name:
        path, meta = parts / f"{name}.parquet", parts / f"{name}.json"
        if args.reuse_parts and path.exists() and meta.exists():
            record = json.loads(meta.read_text())
            if (record["implementation_sha256"] == source_hash and record["prepared_sha256"] == prepared_hash
                    and record["output_sha256"] == fingerprint(path)["sha256"]):
                results.append(record); continue
        needed.append(name)
    with ProcessPoolExecutor(max_workers=args.jobs, initializer=initialize,
                             initargs=(str(prepared_path), str(inputs["fundamentals"]))) as pool:
        futures = {pool.submit(compute_part, n, str(parts), source_hash, prepared_hash): n for n in needed}
        for future in as_completed(futures):
            record = future.result()  # errors are fatal; never silently omit a factor
            results.append(record)
            print(f"{record['factor']}: {record['finite_rows']:,} rows, {record['seconds']:.1f}s", flush=True)
    context = pd.read_parquet(prepared_path, columns=["date", "permno", "ticker", "permco", "gvkey",
                                "risk_eligible", "decision_industry", "decision_log_size", "decision_financial_or_real_estate"])
    for name in catalog.name:
        context[name] = pd.read_parquet(parts / f"{name}.parquet")[name].to_numpy()
    if not all(context[c].notna().any() for c in catalog.name):
        raise ValueError("At least one catalog factor has no usable values; inspect factor_parts before promotion")
    context.to_parquet(dest / "factors.parquet", index=False)
    pd.DataFrame(results).sort_values("factor").to_csv(dest / "factor_catalog.csv", index=False)
    write_run_metadata(dest / "factor_library_metadata.json", stage="wrds_full_factor_library", inputs=inputs,
                       code_paths=WRDS_CODE_PATHS, params={"jobs": args.jobs, "implementation_sha256": source_hash,
                           "prepared_sha256": prepared_hash, "factor_names": catalog.name.tolist()},
                       extra={"factors": len(catalog), "rows": len(context), "fundamentals_is_point_in_time": False,
                              "classification_effective_dated": True, "classification_is_point_in_time": False,
                              "output": fingerprint(dest / "factors.parquet")})
    print(f"Built {len(catalog)} factors on {len(context):,} rows")


if __name__ == "__main__":
    main()
