"""Build versioned WRDS P0 dataset; preserve previous data and results."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
from quant_platform.core.data.wrds import canonical_prices, membership_panel, research_factors, company_market_caps
from scripts.wrds_ingest.client import WRDS_DIR, connect, load_source_plan, write_table
from scripts.wrds_ingest.company import pull_company_classes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh-company", action="store_true")
    parser.add_argument("--output", type=Path, default=WRDS_DIR / "p0")
    args = parser.parse_args()
    dest, prior = args.output, WRDS_DIR / "repaired"
    dest.mkdir(parents=True, exist_ok=True)
    raw_path = WRDS_DIR / "crsp_daily.parquet"
    raw = pd.read_parquet(raw_path)
    classes_path = dest / "crsp_company_classes.parquet"
    if args.refresh_company:
        db = connect()
        try:
            classes = pull_company_classes(db, load_source_plan(), raw.permno.unique())
        finally:
            db.close()
        write_table(classes, classes_path, stage="wrds_company_classes", params={
            "queries": classes.attrs.get("queries"), "market_cap_unit": "thousand_USD"})
    classes = pd.read_parquet(classes_path)
    supplement_path = prior / "crsp_ciz_supplement.parquet"
    supplement = pd.read_parquet(supplement_path).merge(
        classes[["date", "permno", "permco"]], on=["date", "permno"], how="left", validate="one_to_one")
    prices = canonical_prices(raw, supplement)
    if prices["permco"].isna().any():
        raise ValueError("Missing historical PERMCO for a price row")
    link_path = WRDS_DIR / "ccm_gvkey_permno_link.parquet"
    links = pd.read_parquet(link_path)
    companies = company_market_caps(classes, links)
    prices = prices.merge(companies, on=["date", "permco"], how="left", validate="many_to_one")
    events_path = prior / "compustat_quarterly_processed.parquet"
    events = pd.read_parquet(events_path)
    factors = research_factors(prices, events)
    member_path = WRDS_DIR / "sp500_membership_permno.parquet"
    universe = membership_panel(prices, pd.read_parquet(member_path),
                                pd.read_parquet(WRDS_DIR / "crsp_security_names.parquet"))
    inputs = {"raw_prices": raw_path, "supplement": supplement_path, "classes": classes_path,
              "ccm": link_path, "fundamentals": events_path, "membership": member_path}
    for name, frame in (("prices", prices), ("universe", universe), ("factors", factors),
                        ("company_market_caps", companies)):
        write_table(frame, dest / f"{name}.parquet", stage=f"wrds_p0_{name}", inputs=inputs,
                    extra={"primary_key": ["date", "permno"] if name != "company_market_caps" else ["date", "permco"],
                           "membership_is_point_in_time": True, "fundamentals_is_point_in_time": False,
                           "cutoff": str(prices.date.max().date())})
    report = {"price_rows": len(prices), "securities": prices.permno.nunique(),
              "all_class_securities": classes.permno.nunique(), "class_rows": len(classes),
              "multi_class_company_dates": int(companies.n_share_classes.gt(1).sum()),
              "ambiguous_issuer_dates": int(companies.issuer_ambiguous.sum()),
              "missing_company_market_cap_rows": int(prices.company_market_cap.isna().sum()),
              "cutoff": str(prices.date.max().date()),
              "fundamentals_is_point_in_time": False,
              "inputs": {k: str(v) for k, v in inputs.items()}}
    (dest / "p0_validation.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
