"""Validate repaired WRDS artifacts and refresh capacity from original requests.

This changes diagnostics/metadata only, never execution or NAV. Run after
rebuild_wrds_research, or call verify() from that runner.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from quant_platform.core.data.universe import require_point_in_time_universe
from quant_platform.core.evaluation.capacity import capacity_analysis
from quant_platform.core.execution.cost_models.us_equity import CostModel
from scripts.wrds_ingest.client import WRDS_DIR
from quant_platform.core.utils.provenance import fingerprint, write_run_metadata


def verify(destination=None, signals=None):
    dest = Path(destination) if destination else WRDS_DIR / "repaired"
    report_path = dest / "validation_report.json"
    report = json.loads(report_path.read_text()) if report_path.exists() else {}
    sources = report.get("source_sha256") or {
        p.name: fingerprint(p)["sha256"] for p in WRDS_DIR.glob("*.parquet")}
    assert all(fingerprint(WRDS_DIR / name)["sha256"] == h for name, h in sources.items())
    report["originals_unchanged"] = True
    for name in ("prices", "universe", "factors", "compustat_quarterly_processed"):
        metadata_path = dest / (name + "_run_metadata.json")
        metadata = json.loads(metadata_path.read_text())
        for record in metadata.get("inputs", {}).values():
            if record.get("sha256"):
                assert fingerprint(record["path"])["sha256"] == record["sha256"], f"Stale {name} input: {record['path']}"
    px = pd.read_parquet(dest / "prices.parquet", columns=["date", "permno", "ticker", "total_return",
                                                              "market_cap", "shares_outstanding", "volume", "is_terminal"])
    assert not px.duplicated(["date", "permno"]).any()
    assert px.ticker.eq("PERMNO:" + px.permno.astype(str)).all()
    raw = pd.read_parquet(WRDS_DIR / "crsp_daily.parquet")
    check = px.merge(raw[["date", "permno", "ret", "market_cap", "shares_outstanding", "volume"]],
                     on=["date", "permno"], validate="one_to_one", suffixes=("", "_raw"))
    for field, factor in (("market_cap", 1000), ("shares_outstanding", 1000), ("volume", 1)):
        assert np.allclose(check[field], check[field + "_raw"] * factor, equal_nan=True)
    assert np.allclose(check.total_return, check.ret, equal_nan=True)
    del check, raw
    terminal = px[px.is_terminal][["permno", "date", "total_return"]].rename(columns={"date": "terminal_date"})
    assert not terminal.permno.duplicated().any()
    report.update(price_rows=len(px), permnos=px.permno.nunique(), cutoff=str(px.date.max().date()),
                  terminal_rows=len(terminal), missing_terminal_returns=int(terminal.total_return.isna().sum()),
                  price_unit_and_return_identity_checks=True, source_sha256=sources)
    del px
    universe = pd.read_parquet(dest / "universe.parquet")
    assert not universe.duplicated(["date", "permno"]).any()
    require_point_in_time_universe(universe)
    report["membership_is_point_in_time"] = True
    report["fundamentals_is_point_in_time"] = False
    del universe
    events = pd.read_parquet(dest / "compustat_quarterly_processed.parquet")
    violations = sum(int((events[c].notna() & events[c].gt(events.available_date)).sum())
                     for c in events if c.endswith("_available_date") and c != "source_available_date")
    assert violations == 0
    assert events.available_date.le(pd.Timestamp(report["cutoff"])).all()
    assert not events.duplicated(["permno", "available_date"]).any()
    income_errors = int((events.saleq_q.notna() & events.saleq_q.ne(events.saleq)).sum())
    assert income_errors == 0
    report.update(dependency_availability_violations=violations, quarter_income_identity_errors=income_errors,
                  processed_fundamental_events=len(events), fundamental_gvkeys=events.gvkey.nunique(),
                  n_snapshot_current_vintage_events=int(events.vintage.eq("compustat_snapshot_pit").sum()),
                  n_all_fields_strict_pit_events=int(events.is_point_in_time.sum()))
    summary_path = dest / "fundamentals_summary.json"
    summary = json.loads(summary_path.read_text())
    summary["n_rows_non_pit"] = int((~events.is_point_in_time).sum())
    summary["n_rows_restated"] = int(events.vintage.eq("compustat_fundq_as_restated").sum())
    summary.setdefault("raw_availability", summary["availability"].copy())
    summary["availability"].update(population="final linked events through cutoff",
        n_available_date_imputed=int(events.available_date_imputed.sum()),
        n_available_date_clipped=int(events.available_date_clipped_to_period_end.sum()))
    summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    processed_meta = dest / "compustat_quarterly_processed_run_metadata.json"
    pm = json.loads(processed_meta.read_text())
    pm.update(summary)
    processed_meta.write_text(json.dumps(pm, indent=2, default=str), encoding="utf-8")
    del events
    reference_path = dest / "crsp_delisting_reference.parquet"
    if reference_path.exists():
        reference = pd.read_parquet(reference_path)
        reference["deldlydt"] = pd.to_datetime(reference["deldlydt"])
        joined = terminal.merge(reference, left_on=["permno", "terminal_date"], right_on=["permno", "deldlydt"],
                                validate="one_to_one", how="left", indicator=True)
        assert joined["_merge"].eq("both").all()
        assert np.allclose(joined.total_return, joined.delret, atol=1e-6, equal_nan=True)
        report["independent_delisting_reconciliation"] = {"rows": len(joined), "valid_returns": int(joined.delret.notna().sum()),
                                                         "missing_returns": int(joined.delret.isna().sum()), "double_counted": False}
    results = []
    for strategy in sorted((dest / "candidate_backtests").iterdir()):
        if not strategy.is_dir():
            continue
        if signals is not None and strategy.name not in signals:
            continue
        weights = pd.read_parquet(strategy / "weights.parquet", columns=["date", "weight", "is_rebalance"])
        assert weights.weight.abs().max() <= .05 + 1e-12
        rebalance_dates = set(weights.loc[weights.is_rebalance, "date"])
        for policy in ("total_loss", "zero_return"):
            folder = strategy / policy
            if not (folder / "backtest_metadata.json").exists():
                continue
            meta_path = folder / "backtest_metadata.json"
            meta = json.loads(meta_path.read_text())
            for name, expected in meta.get("source_inputs", {}).items():
                assert fingerprint(dest / (name + ".parquet"))["sha256"] == expected, f"Stale {strategy.name}/{policy}: {name}"
            assert meta.get("weights_sha256") == fingerprint(strategy / "weights.parquet")["sha256"], "Stale weights"
            daily = pd.read_parquet(folder / "daily_pnl.parquet")
            if meta.get("capacity_contract") != "original_rebalance_requests_v2":
                trades = pd.read_parquet(folder / "trades.parquet")
                trades["is_rebalance_order"] = trades.date.isin(rebalance_dates)
                # Saved orders retain PERMNO; an exchange ticker must never be
                # used to resolve a terminal event or a capacity input.
                assert trades.permno.notna().all()
                terminal_check = trades.merge(terminal[["permno", "terminal_date"]], on="permno", how="left")
                assert not terminal_check.date.ge(terminal_check.terminal_date).fillna(False).any()
                del terminal_check
                assert trades.executed_participation.max() <= .1 + 1e-12
                trades.attrs = {}
                tmp = folder / "trades.parquet.tmp"
                trades.to_parquet(tmp, index=False, compression="zstd")
                tmp.replace(folder / "trades.parquet")
                cap = capacity_analysis(trades, daily.set_index("date"), cost_model=CostModel(**meta["cost_model"]),
                                        capital_grid=[1e6, 1e7, 1e8])
                cap["summary"].to_csv(folder / "capacity_summary.csv", index=False)
                cap["capital_curve"].to_csv(folder / "capacity_curve.csv", index=False)
                del trades, cap
            cap = pd.read_csv(folder / "capacity_summary.csv").iloc[0]
            curve = pd.read_csv(folder / "capacity_curve.csv")
            assert curve.max_executed_participation.max() <= .1 + 1e-12
            imputed = int(daily.terminal_return_imputations.sum())
            assert imputed == meta["terminal_return_imputations"]
            meta.update(capacity_contract="original_rebalance_requests_v2", fallback_scope="portfolio_construction",
                        fundamentals_source_fallback=strategy.name != "MOM_12_1M")
            meta_path.write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
            write_run_metadata(folder / "validation_metadata.json", stage="wrds_backtest_validation",
                inputs={"weights": strategy / "weights.parquet", "prices": dest / "prices.parquet",
                        "factors": dest / "factors.parquet", "trades": folder / "trades.parquet",
                        "daily_pnl": folder / "daily_pnl.parquet"},
                params={"capacity_contract": "original_rebalance_requests_v2", "terminal_policy": policy})
            perf = pd.read_csv(folder / "performance_summary.csv").iloc[0].to_dict()
            results.append({"signal": strategy.name, "terminal_policy": policy, **perf,
                            "terminal_return_imputations": imputed,
                            "max_participation": cap.max_executed_participation,
                            "capacity_at_10pct_adv": cap.capacity_at_10pct_adv,
                            "capacity_at_max_10pct_adv": cap.capacity_at_max_10pct_adv,
                            "capacity_sample_days": cap.p95_sample_days})
    if results:
        table = pd.DataFrame(results)
        table.to_csv(dest / "candidate_backtests_summary.csv", index=False)
        report["backtests"] = results
        report["n_backtest_scenarios_validated"] = len(results)
        report["capacity_acceptance"] = {
            "max_executed_participation": float(table.max_participation.max()),
            "participation_limit_passed": bool(table.max_participation.le(.1 + 1e-12).all()),
            "capacity_estimate_min": float(table.capacity_at_10pct_adv.min()),
            "capacity_estimate_max": float(table.capacity_at_10pct_adv.max()),
            "original_capacity_target_range": [1e6, 1e8],
            "all_estimates_within_original_target": bool(table.capacity_at_10pct_adv.between(1e6, 1e8).all()),
            "note": "Capacity is a requested-order diagnostic, not a validated deployable AUM or an optimization target."}
    report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k not in {"backtests", "source_sha256"}}, indent=2, default=str))
    return report


if __name__ == "__main__":
    verify()
