"""Reconcile the P0 dataset against raw WRDS rows and independent cash labels."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from quant_platform.core.data.returns import forward_total_returns, overnight_total_return
from quant_platform.core.data.research_contract import require_wrds_identity
from quant_platform.core.data.wrds import FACTOR_DIRECTIONS
from quant_platform.core.utils.provenance import fingerprint, write_run_metadata, WRDS_CODE_PATHS


def main():
    dest = ROOT / "data/wrds/p0"
    prior = dest.parent / "repaired"
    originals = json.loads((prior / "validation_report.json").read_text())["source_sha256"]
    assert all(fingerprint(dest.parent / k)["sha256"] == v for k, v in originals.items())
    for name in ("prices", "factors", "universe", "company_market_caps"):
        meta = json.loads((dest / f"{name}_run_metadata.json").read_text())
        for source in meta["inputs"].values():
            assert fingerprint(source["path"])["sha256"] == source["sha256"]
    p = pd.read_parquet(dest / "prices.parquet")
    f = pd.read_parquet(dest / "factors.parquet")
    u = pd.read_parquet(dest / "universe.parquet")
    require_wrds_identity(u, f, p)
    raw = pd.read_parquet(dest.parent / "crsp_daily.parquet", columns=["date", "permno", "ret"])
    r = p.merge(raw, on=["date", "permno"], validate="one_to_one", how="left", suffixes=("", "_raw"))
    np.testing.assert_allclose(r.total_return, r.ret_raw, equal_nan=True)
    del r, raw
    # Independent small-path loop: each terminal is two sessions into a
    # twenty-session label. No endpoint ratio and no label-builder internals.
    labels = forward_total_returns(p, 20, missing_terminal_return="total_loss").set_index(["date", "ticker"])
    calendar = pd.DatetimeIndex(sorted(p.date.unique()))
    paths = p.set_index(["permno", "date"])
    checks = []
    for row in p[p.is_terminal].itertuples():
        pos = calendar.get_loc(row.date)
        if pos < 2 or pos - 2 + 20 >= len(calendar):
            continue
        start, end = calendar[pos-2], calendar[pos-2+20]
        try:
            base = paths.loc[(row.permno, start)]
        except KeyError:
            continue
        if not base.close > 0:
            continue
        values = []
        for dt in calendar[pos-1:pos+1]:
            try:
                v = paths.loc[(row.permno, dt)].total_return
            except KeyError:
                v = np.nan
            if dt == row.date and pd.isna(v):
                v = -1.  # explicit sensitivity, not an observed payoff
            values.append(v)
        expected = np.prod(1 + np.array(values)) - 1
        label = labels.loc[(start, row.ticker)]
        assert np.isclose(label.forward_return, expected, equal_nan=True)
        assert label.label_end == end
        checks.append({"permno": row.permno, "start": start, "terminal": row.date,
                       "label_end": end, "expected": expected, "actual": label.forward_return,
                       "imputed": bool(label.label_terminal_imputed)})
    assert len(checks) > 200
    pd.DataFrame(checks).to_csv(dest / "terminal_label_reconciliation.csv", index=False)
    overnight = overnight_total_return(p)
    mask = overnight.notna()
    np.testing.assert_allclose(((1+overnight) * p.close / p.open - 1)[mask], p.total_return[mask], atol=1e-12)
    # Every simultaneously observable share class must receive the SAME
    # company valuation ratio. New/listing/issuer-change rows are null.
    ratios = f.groupby(["date", "permco"])[["EARNINGS_YIELD", "BOOK_TO_MARKET"]].nunique()
    assert ratios.le(1).all().all()
    cap = pd.read_parquet(dest / "company_market_caps.parquet")
    classes = pd.read_parquet(dest / "crsp_company_classes.parquet")
    selected = classes[classes.securitytype.eq("EQTY") & classes.securitysubtype.eq("COM")]
    sums = selected.assign(value=selected.market_cap.where(selected.market_cap.gt(0))*1000).groupby(["date", "permco"]).value.sum(min_count=1)
    check = cap.set_index(["date", "permco"]).company_market_cap.dropna()
    np.testing.assert_allclose(check, sums.reindex(check.index))
    quality = f.merge(u[["date", "permno", "pass_all_filters"]], on=["date", "permno"], validate="one_to_one")
    quality = quality[quality.pass_all_filters & quality.date.ge("2010-01-01")]
    coverage = [{"factor": name, "coverage": float(quality[name].notna().mean())} for name in FACTOR_DIRECTIONS]
    pd.DataFrame(coverage).to_csv(dest / "factor_coverage.csv", index=False)
    report = {"original_inputs_unchanged": True, "input_lineage_verified": True,
              "stable_permno_membership_verified": True, "ciz_return_identity_verified": True,
              "terminal_label_paths_reconciled": len(checks),
              "terminal_label_imputed_paths": int(pd.DataFrame(checks).imputed.sum()),
              "overnight_daily_decomposition_rows": int(mask.sum()),
              "company_cap_reconciled_dates": len(check), "share_class_ratios_equal": True,
              "fundamentals_is_point_in_time": False, "cutoff": str(p.date.max().date()),
              "factor_coverage": coverage}
    write_run_metadata(dest / "p0_acceptance.json", stage="verify_wrds_p0",
                       code_paths=WRDS_CODE_PATHS,
                       inputs={name: dest / f"{name}.parquet" for name in ("prices", "factors", "universe", "company_market_caps", "crsp_company_classes")},
                       extra=report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
