"""Command-line entry point for the WRDS pull.

Order matters and is enforced by the fail-closed source plan:

    python scripts/wrds_pull.py probe          # resolves tables, verifies entitlements
    python scripts/wrds_pull.py universe       # PERMNO-keyed membership + prices
    python scripts/wrds_pull.py fundamentals   # Compustat, linked on the disclosure date
    python scripts/wrds_pull.py all            # probe (if needed) -> universe -> fundamentals

``universe`` must precede ``fundamentals``: the fundamentals step maps GVKEY to
PERMNO through the dated CCM link written by ``universe``, and restricting the
pull to linked GVKEYs is what keeps it to a sane size.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from scripts.wrds_ingest import connect, load_source_plan
from scripts.wrds_ingest import fundamentals as fundamentals_mod
from scripts.wrds_ingest import probe as probe_mod
from scripts.wrds_ingest import universe as universe_mod
from scripts.wrds_ingest.client import (
    PROBE_REPORT_PATH,
    WRDS_DIR,
    SourcePlanMissing,
)


def _plan_or_die() -> "object":
    try:
        return load_source_plan()
    except SourcePlanMissing as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)


def cmd_probe(args: argparse.Namespace) -> int:
    return probe_mod.main()


def cmd_universe(args: argparse.Namespace) -> int:
    plan = _plan_or_die()
    db = connect()
    try:
        summary = universe_mod.run(
            plan=plan, db=db, start=args.start, end=args.end
        )
    finally:
        db.close()
    print()
    print("=" * 78)
    print("UNIVERSE PULL SUMMARY")
    print("=" * 78)
    for key in (
        "n_permnos",
        "n_membership_spells",
        "n_ccm_links",
        "n_permnos_with_gvkey",
        "n_price_rows",
        "contains_delisting_return_rows",
    ):
        if key in summary:
            print(f"  {key:<34} {summary[key]}")
    readiness = summary.get("readiness", {})
    print(f"  {'membership_is_point_in_time':<34} {readiness.get('membership_is_point_in_time')}")
    print(f"  {'ticker_view_safe':<34} {readiness.get('ticker_view_safe')}")
    print(f"  blocking_note")
    print(f"    {readiness.get('blocking_note','')}")
    print("=" * 78)
    return 0


def cmd_fundamentals(args: argparse.Namespace) -> int:
    plan = _plan_or_die()
    link_path = WRDS_DIR / "ccm_gvkey_permno_link.parquet"
    link = None
    gvkeys = None
    if link_path.exists():
        link = pd.read_parquet(link_path)
        gvkeys = sorted(link["gvkey"].astype(str).unique().tolist())
        print(f"  Restricting fundamentals to {len(gvkeys):,} GVKEYs linked to the universe.")
    else:
        raise SystemExit("No CCM links; run universe before fundamentals. Refusing an unrestricted pull.")

    db = connect()
    try:
        summary = fundamentals_mod.run(
            plan=plan, db=db, gvkeys=gvkeys, start=args.start, end=args.end, link=link
        )
    finally:
        db.close()
    print()
    print("=" * 78)
    print("FUNDAMENTALS PULL SUMMARY")
    print("=" * 78)
    print(f"  source_table               {summary['source_table']}")
    print(f"  fallback_table             {summary['fallback_table']}")
    print(f"  rows                       {summary['n_rows']:,}")
    print(f"    point-in-time            {summary['n_rows_point_in_time']:,}")
    print(f"    non-PIT dependencies     {summary['n_rows_non_pit']:,}")
    print(f"    current fundq vintage    {summary['n_rows_restated']:,}")
    print(f"  gvkeys                     {summary['n_gvkeys']:,}")
    for segment in summary["segments"]:
        print(
            f"    segment {segment['vintage']:<28} "
            f"{segment['start']} .. {segment['end']}  {segment['n_rows']:,} rows"
        )
    avail = summary["availability"]
    print(
        f"  rdq missing (imputed)      {avail['n_available_date_imputed']:,} "
        f"(lag {avail['imputation_lag_days']}d)"
    )
    print(f"  aggregation                {summary['aggregation_policy']}")
    if summary["permno_link"]:
        print(f"  rows with valid PERMNO     {summary['permno_link']['n_rows_matched_to_permno']:,}")
    if summary.get("caveat"):
        print(f"  CAVEAT                     {summary['caveat']}")
    print("=" * 78)
    return 0


def cmd_all(args: argparse.Namespace) -> int:
    if args.force_probe or not PROBE_REPORT_PATH.exists():
        rc = cmd_probe(args)
        if rc:
            return rc
    rc = cmd_universe(args)
    if rc:
        return rc
    return cmd_fundamentals(args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="wrds_pull",
        description="Pull CRSP / Compustat data for EntroPy with point-in-time discipline.",
    )
    parser.add_argument(
        "command",
        choices=("probe", "universe", "fundamentals", "all"),
        help="probe first: every other step refuses to run without its report.",
    )
    parser.add_argument("--start", default="2009-01-01", help="history start (YYYY-MM-DD)")
    parser.add_argument("--end", default="2025-12-31", help="history end (YYYY-MM-DD)")
    parser.add_argument(
        "--force-probe",
        action="store_true",
        help="re-resolve tables even if data/wrds/probe_report.json exists",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    handlers = {
        "probe": cmd_probe,
        "universe": cmd_universe,
        "fundamentals": cmd_fundamentals,
        "all": cmd_all,
    }
    print()
    print(f"EntroPy WRDS pull :: {args.command}")
    print(f"  output directory: {WRDS_DIR}")
    print(flush=True)
    return handlers[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
