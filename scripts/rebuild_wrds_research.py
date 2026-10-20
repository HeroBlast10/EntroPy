"""Rebuild isolated WRDS research: python scripts/rebuild_wrds_research.py.

Requires licensed files in data/wrds and the two repaired raw fundamentals
pulls (or --refresh-fundamentals with configured WRDS credentials). Legacy
artifacts are never overwritten. --data-only skips strategy execution.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from quant_platform.core.data.wrds import canonical_prices, membership_panel, research_factors, FACTOR_DIRECTIONS
from quant_platform.core.execution.backtest.pipeline import run_trading_pipeline
from quant_platform.core.execution.cost_models.us_equity import CostModel
from quant_platform.core.portfolio.rebalance import carry_forward_weights
from scripts.wrds_ingest.client import WRDS_DIR, connect, load_source_plan, write_table
from scripts.wrds_ingest.fundamentals import run as rebuild_fundamentals
from scripts.wrds_ingest.universe import pull_ciz_supplement


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def monthly_weights(panel, signal, dates, min_names=100):
    """Monthly symmetric quintile spread, gross target 1, cap <= 0.05.

    Ties at quintile boundaries share their sleeve equally. Fewer than
    min_names produces an explicit cash target rather than a small-stock bet.
    Signals have already been lagged one session by research_factors.
    """
    rebalances = pd.Series(dates, index=dates).groupby(dates.to_period("M")).max()
    rows = []
    for dt in rebalances:
        day = panel[panel["date"].eq(dt) & panel["pass_all_filters"]].dropna(subset=[signal])
        weights = pd.Series(dtype=float)
        if len(day) >= min_names:
            scores = day.set_index("ticker")[signal]
            lower, upper = scores.quantile([0.2, 0.8])
            long, short = scores[scores.ge(upper)], scores[scores.le(lower)]
            if upper > lower and len(long) >= 10 and len(short) >= 10:
                weights = pd.concat([pd.Series(0.5 / len(long), index=long.index),
                                     pd.Series(-0.5 / len(short), index=short.index)])
        if weights.empty:
            weights = pd.Series(0.0, index=panel["ticker"].unique())
        rows.append(pd.DataFrame({"date": dt, "ticker": weights.index, "weight": weights.values}))
    reb = pd.concat(rows, ignore_index=True)
    return carry_forward_weights(reb, pd.DatetimeIndex(dates, name="date"))


def execute_strategy(destination, signal):
    """Independent worker; reload immutable canonical inputs, return summaries."""
    dest = Path(destination)
    px_cols = ["date", "ticker", "permno", "adj_close", "close", "volume", "is_tradable", "total_return", "is_terminal"]
    prices = pd.read_parquet(dest / "prices.parquet", columns=px_cols)
    panel = pd.read_parquet(dest / "factors.parquet")
    universe = pd.read_parquet(dest / "universe.parquet", columns=["date", "permno", "pass_all_filters"])
    panel = panel.merge(universe, on=["date", "permno"], how="left", validate="one_to_one")
    panel["pass_all_filters"] = panel["pass_all_filters"].eq(True)
    panel = panel[panel["date"].ge("2010-01-01")].copy()
    for c, direction in FACTOR_DIRECTIONS.items():
        panel[c] *= direction
    if signal == "QUALITY_COMPOSITE":
        ranks = pd.concat([panel[c].where(panel.pass_all_filters).groupby(panel.date).rank(pct=True)
                           for c in FACTOR_DIRECTIONS if c != "MOM_12_1M"], axis=1)
        panel[signal] = ranks.mean(axis=1).where(ranks.notna().all(axis=1))
    weights = monthly_weights(panel, signal, pd.DatetimeIndex(sorted(panel.date.unique())))
    weights["permno"] = weights.ticker.str.split(":").str[-1].astype("int64")
    root = dest / "candidate_backtests" / signal
    root.mkdir(parents=True, exist_ok=True)
    inputs = {name: dest / (name + ".parquet") for name in ("prices", "factors", "universe")}
    write_table(weights, root / "weights.parquet", stage="wrds_monthly_weights", inputs=inputs,
                extra={"primary_key": "permno", "signal": signal})
    del panel, universe, weights
    results = []
    for policy in ("total_loss", "zero_return"):
        bt = run_trading_pipeline(weights_path=root / "weights.parquet", prices_override=prices,
             cost_model=CostModel(max_participation_rate=0.1), initial_capital=1e6,
             output_dir=root / policy, benchmark_market=None, capacity_capitals=[1e6, 1e7, 1e8],
             missing_terminal_return=policy)
        trades = bt["trades"]
        max_part = float(trades.executed_participation.max()) if len(trades) else 0
        assert max_part <= .1 + 1e-12
        meta_path = root / policy / "backtest_metadata.json"
        meta = json.loads(meta_path.read_text())
        meta.update({"primary_key": "permno", "membership_is_point_in_time": True,
                     "fundamentals_is_point_in_time": signal == "MOM_12_1M",
                     "missing_terminal_return_policy": policy, "gross_target": 1.0,
                     "signal_lag_sessions": 1, "sector_constraints": None,
                     "research_only": True, "fallback": False,
                     "source_inputs": {k: digest(p) for k, p in inputs.items()},
                     "weights_sha256": digest(root / "weights.parquet")})
        meta_path.write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
        results.append({"signal": signal, "terminal_policy": policy, **bt["performance"],
                        "max_participation": max_part,
                        "terminal_return_imputations": meta["terminal_return_imputations"],
                        "capacity_at_10pct_adv": float(bt["capacity"]["summary"].iloc[0]["capacity_at_10pct_adv"])})
        del trades, bt
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh-fundamentals", action="store_true")
    parser.add_argument("--refresh-supplement", action="store_true")
    parser.add_argument("--data-only", action="store_true")
    parser.add_argument("--reuse-derived", action="store_true", help="Use previously rebuilt canonical files")
    parser.add_argument("--jobs", type=int, default=1, help="Independent strategy processes (1-4)")
    parser.add_argument("--end", default="2025-12-31")
    parser.add_argument("--signals", nargs="*")
    args = parser.parse_args()
    if not 1 <= args.jobs <= 4:
        parser.error("--jobs must be 1..4")
    dest = WRDS_DIR / "repaired"
    dest.mkdir(parents=True, exist_ok=True)
    originals = {p.name: digest(p) for p in WRDS_DIR.glob("*.parquet")}
    raw = pd.read_parquet(WRDS_DIR / "crsp_daily.parquet")
    cutoff = min(pd.Timestamp(args.end), raw["date"].max())
    if pd.Timestamp(args.end) > raw["date"].max():
        raise ValueError("Requested end exceeds CRSP coverage; do not splice legacy prices")
    link = pd.read_parquet(WRDS_DIR / "ccm_gvkey_permno_link.parquet")
    supplement_path = dest / "crsp_ciz_supplement.parquet"
    if args.refresh_supplement:
        db = connect()
        try:
            supplement = pull_ciz_supplement(db, load_source_plan(), sorted(raw["permno"].unique()), end=str(cutoff.date()))
            write_table(supplement, supplement_path, stage="wrds_ciz_supplement", params={"sql": supplement.attrs.get("sql")})
        finally:
            db.close()
    supplement = pd.read_parquet(supplement_path) if supplement_path.exists() else None
    if not args.reuse_derived:
        db = connect() if args.refresh_fundamentals else None
        try:
            rebuild_fundamentals(plan=load_source_plan(), db=db, gvkeys=sorted(link["gvkey"].unique()),
                                 start="2009-01-01", end=str(cutoff.date()), link=link,
                                 output_dir=dest, reuse_raw=not args.refresh_fundamentals)
        finally:
            if db is not None:
                db.close()
        prices = canonical_prices(raw[raw["date"].le(cutoff)], supplement)
        member = membership_panel(prices, pd.read_parquet(WRDS_DIR / "sp500_membership_permno.parquet"),
                                  pd.read_parquet(WRDS_DIR / "crsp_security_names.parquet"))
    else:
        prices = pd.read_parquet(dest / "prices.parquet")
        member = pd.read_parquet(dest / "universe.parquet")
        if prices.date.max() != cutoff:
            raise ValueError("Cached prices do not match requested cutoff")
    events = pd.read_parquet(dest / "compustat_quarterly_processed.parquet")
    dependency_violations = sum(int((events[c].notna() & events[c].gt(events["available_date"])).sum())
                                for c in events if c.endswith("_available_date") and c != "source_available_date")
    assert dependency_violations == 0
    factors = pd.read_parquet(dest / "factors.parquet") if args.reuse_derived else research_factors(prices, events)
    inputs = {"prices": WRDS_DIR / "crsp_daily.parquet",
              "fundamentals": dest / "compustat_quarterly_processed.parquet",
              "membership": WRDS_DIR / "sp500_membership_permno.parquet"}
    for frame, name in ([] if args.reuse_derived else ((prices, "prices"), (member, "universe"), (factors, "factors"))):
        write_table(frame, dest / (name + ".parquet"), stage="wrds_canonical_" + name,
                    inputs=inputs, extra={"primary_key": ["date", "permno"],
                        "ticker_semantics": "stable PERMNO alias; exchange ticker is display only",
                        "membership_is_point_in_time": True, "fundamentals_is_point_in_time": False,
                        "delisting_returns": "CIZ embedded, never merged", "cutoff": str(cutoff.date())})
    panel = factors.merge(member[["date", "permno", "pass_all_filters"]], on=["date", "permno"],
                          how="left", validate="one_to_one")
    panel["pass_all_filters"] = panel["pass_all_filters"].fillna(False).astype(bool)
    panel = panel[panel["date"].ge("2010-01-01")].copy()
    eligible = panel["pass_all_filters"]
    quality = [c for c in FACTOR_DIRECTIONS if c != "MOM_12_1M"]
    ranked = []
    coverage = []
    for name, direction in FACTOR_DIRECTIONS.items():
        coverage.append({"factor": name, "coverage": float(panel.loc[eligible, name].notna().mean()),
                         "eligible_rows": int(eligible.sum())})
        panel[name] *= direction
        if name in quality:
            ranked.append(panel[name].where(eligible).groupby(panel["date"]).rank(pct=True))
    ranks = pd.concat(ranked, axis=1)
    panel["QUALITY_COMPOSITE"] = ranks.mean(axis=1).where(ranks.notna().all(axis=1))
    pd.DataFrame(coverage).to_csv(dest / "factor_coverage.csv", index=False)
    report = {"cutoff": str(cutoff.date()), "price_rows": len(prices), "permnos": prices["permno"].nunique(),
              "membership_is_point_in_time": True, "fundamentals_is_point_in_time": False,
              "terminal_rows": int(prices["is_terminal"].sum()),
              "missing_terminal_returns": int((prices["is_terminal"] & prices["total_return"].isna()).sum()),
              "classification_available": supplement is not None,
              "dependency_availability_violations": dependency_violations,
              "processed_fundamental_events": len(events),
              "fundamental_gvkeys": events["gvkey"].nunique(),
              "n_snapshot_current_vintage_events": int(events["vintage"].eq("compustat_snapshot_pit").sum()),
              "n_all_fields_strict_pit_events": int(events["is_point_in_time"].sum()),
              "membership_permnos_without_ccm": int(len(set(prices["permno"]) - set(link["permno"]))),
              "quarter_income_identity_errors": int((events["saleq_q"].notna() & events["saleq_q"].ne(events["saleq"])).sum()),
              "factor_coverage": coverage, "source_sha256": originals,
              "limitations": ["Fundq is restated research data, even after availability alignment.",
                               "No historical sector constraints: no PIT sector classification supplied.",
                               "Total-loss / zero-return terminal policies are sensitivity scenarios, not observed returns.",
                               "Daily cost/borrow parameters are modeled; no TAQ execution calibration."]}
    if not args.data_only:
        results = []
        strategies = args.signals or [*FACTOR_DIRECTIONS, "QUALITY_COMPOSITE"]
        if any(s not in [*FACTOR_DIRECTIONS, "QUALITY_COMPOSITE"] for s in strategies):
            raise ValueError("Unknown strategy")
        # Workers read immutable files rather than serializing multi-million-row frames.
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            futures = [pool.submit(execute_strategy, str(dest), signal) for signal in strategies]
            for future in as_completed(futures):
                results.extend(future.result())
                pd.DataFrame(results).to_csv(dest / "candidate_backtests_summary.csv", index=False)
        report["backtests"] = results
    report["originals_unchanged"] = originals == {p.name: digest(p) for p in WRDS_DIR.glob("*.parquet")}
    assert report["originals_unchanged"]
    (dest / "validation_report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    if not args.data_only:
        from scripts.verify_wrds_research import verify
        report = verify(dest, signals=strategies)
    print(json.dumps({k: v for k, v in report.items() if k not in {"backtests", "source_sha256"}}, indent=2, default=str))


if __name__ == "__main__":
    main()
