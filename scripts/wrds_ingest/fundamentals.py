"""Quarterly fundamentals with an honest point-in-time label.

Correction #4, stated precisely: ``fundq`` plus ``rdq`` solves *when* a figure
was disclosed but not *which* figure. Compustat's standard quarterly table
carries the most recently restated values, so a researcher reading it today
sees numbers that were not knowable on the earnings date; nor were all fields
necessarily disclosed that day. The products that actually separate vintages
are Compustat **Snapshot** (As-First-Reported / Most-Recently-Restated /
Point-In-Time) and Unrestated Quarterly. Whether this subscription includes one
is decided by :mod:`scripts.wrds.probe` and recorded in the source plan. Either
way the chosen vintage is written into every artifact's metadata; a possibly
restated series is never silently presented as point-in-time.

The second half of this module is aggregation, and it is where most of the real
damage happens:

* **Compustat quarterly flows are cumulative within the fiscal year.**
  ``saleq`` is year-to-date revenue, not fourth-quarter revenue. Reading it as
  a quarterly flow produces a value that ramps through the fiscal year -- a
  factor built on it loads on fiscal-year position, which is pure noise.
* **TTM must require a full four quarters.** ``rolling(4).sum()`` with the
  default ``min_periods=None`` quietly emits two-quarter sums during warm-up,
  which are roughly half the magnitude of a real TTM and therefore look like a
  legitimate low reading rather than missing data.
* **YoY must compare like with like.** Quarter (t) against quarter (t-4), not
  against the previous row, or fiscal-year effects leak in.

Usage::

    python scripts/wrds_pull.py fundamentals
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from scripts.wrds_ingest.client import WRDS_DIR, SourcePlan, write_table

# Items whose Compustat quarterly value is a fiscal-year-to-date accumulation.
# These MUST be differenced to obtain a single quarter. Getting this list wrong
# is the single most common way to corrupt quarterly fundamentals.
CUMULATIVE_FLOW_ITEMS: frozenset[str] = frozenset(
    {
        "saleq", "revtq", "cogsq", "xsgaq", "xrdq", "oibdpq", "oiadpq",
        "ibq", "niq", "piq", "txtq", "xintq", "dpq", "xoprq",
        "dvtq", "nopiq", "xiq", "miiq",
        "oancfy", "ivncfy", "fincfy", "capxy", "dpcq",
        "txbcofy", "fiaoq", "ibcofy",
    }
)

# Balance-sheet and market items: a stock, not a flow. Never differenced.
STOCK_ITEMS: frozenset[str] = frozenset(
    {
        "atq", "ltq", "ceq", "actq", "lctq", "cheq", "dlcq", "dlttq",
        "invtq", "ppegtq", "ppentq", "rectq", "apq", "txditcq", "aoq",
        "intanq", "gdwlq", "mibq", "pstkq", "cshoq", "prccq", "ajexq",
        "seqq", "wcapq",
    }
)

DEFAULT_ITEMS: tuple[str, ...] = tuple(sorted(CUMULATIVE_FLOW_ITEMS | STOCK_ITEMS))

# Fallback availability lag when rdq is missing. Compustat's own guidance is
# roughly 90 days after period end for quarterly data; the count of imputed rows
# is always reported so this never silently dominates the sample.
DEFAULT_LAG_DAYS = 90
BASE_RATIO_FLOOR = 1e-8


# ---------------------------------------------------------------------------
# Pull
# ---------------------------------------------------------------------------


def _table_columns(db, table: str) -> set[str]:
    schema, name = table.split(".", 1)
    return {
        str(c).lower()
        for c in db.raw_sql(
            "SELECT column_name FROM information_schema.columns "
            f"WHERE table_schema = '{schema}' AND table_name = '{name}'"
        )["column_name"]
    }


def _available_items(db, table: str, items: Sequence[str]) -> list[str]:
    """Keep only items the resolved table actually exposes."""
    existing = _table_columns(db, table)
    return [item for item in items if item.lower() in existing]


def pull_fundamentals(
    db,
    table: str,
    gvkeys: Sequence[str] | None = None,
    start: str = "2008-01-01",
    end: str | None = None,
    items: Sequence[str] | None = None,
    vintage: str = "unknown",
) -> pd.DataFrame:
    """Quarterly rows from one table, tagged with the vintage that produced them.

    Standard population filters are applied: domestic consolidated industrial
    statements in the chosen format. Without them the same filing appears
    several times under different formats and every downstream join fans out.

    ``vintage`` is carried on every row. Downstream that column -- not the
    filename, not a comment -- is what tells a reader whether a number was
    knowable on the date it is being used.
    """
    requested = list(items) if items is not None else list(DEFAULT_ITEMS)
    usable = _available_items(db, table, requested)
    dropped = sorted(set(requested) - set(usable))
    if not usable:
        raise RuntimeError(
            f"{table} exposes none of the requested quarterly items. Inspect "
            "data/wrds/probe_report.json and adjust DEFAULT_ITEMS."
        )
    if dropped:
        print(f"  [{vintage}] {len(dropped)} requested items absent: {dropped[:12]}", flush=True)

    columns = _table_columns(db, table)
    projections = ["gvkey", "datadate", "rdq", "fyearq", "fqtr"]
    # The Snapshot product carries explicit point-in-time validity bounds. They
    # are the whole reason to prefer it, so take them when they exist.
    for optional in ("pitdate1", "pitdate2"):
        if optional in columns:
            projections.append(optional)

    where = [
        "indfmt = 'INDL'",
        "datafmt = 'STD'",
        "popsrc = 'D'",
        "consol = 'C'",
        f"datadate >= DATE '{start}'",
    ]
    if end:
        where.append(f"datadate <= DATE '{end}'")
    if gvkeys:
        quoted = ",".join(f"'{g}'" for g in gvkeys)
        where.append(f"gvkey IN ({quoted})")

    frame = db.raw_sql(
        f"SELECT {', '.join(projections + usable)} FROM {table} "
        f"WHERE {' AND '.join(where)}"
    )
    frame["gvkey"] = frame["gvkey"].astype(str)
    frame["datadate"] = pd.to_datetime(frame["datadate"])
    frame["rdq"] = pd.to_datetime(frame["rdq"])
    for item in usable:
        frame[item] = pd.to_numeric(frame[item], errors="coerce")
    frame["vintage"] = vintage
    return frame.sort_values(["gvkey", "datadate"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Availability
# ---------------------------------------------------------------------------


def availability_dates(
    frame: pd.DataFrame, lag_days: int = DEFAULT_LAG_DAYS
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Attach ``available_date``: the first date the row could have been used.

    ``rdq`` is the actual report date of quarterly earnings, which is the only
    reason any of this is point-in-time-ish at all. Where it is missing we
    impute and count the imputation: an imputed availability is a guess, and
    the share of guesses is a property of the sample worth knowing.
    """
    out = frame.copy()
    implied = out["datadate"] + pd.Timedelta(days=lag_days)

    # Snapshot rows carry pitdate1: the date this vintage actually became the
    # one in force. That is a stronger availability date than rdq, which only
    # says when the period was reported.
    if "pitdate1" in out.columns:
        primary = pd.to_datetime(out["pitdate1"], errors="coerce")
    else:
        primary = pd.Series(pd.NaT, index=out.index)
    rdq = (
        pd.to_datetime(out["rdq"], errors="coerce")
        if "rdq" in out.columns
        else pd.Series(pd.NaT, index=out.index)
    )
    chosen = primary.fillna(rdq)

    imputed = chosen.isna()
    out["available_date"] = chosen.where(~imputed, implied)
    out["available_date_imputed"] = imputed

    # A posted date before period end is a data error, not early disclosure.
    clipped = out["available_date"] < out["datadate"]
    out.loc[clipped, "available_date"] = out.loc[clipped, "datadate"]
    out["available_date_clipped_to_period_end"] = clipped

    stats = {
        "n_available_date_imputed": int(imputed.sum()),
        "n_available_date_clipped": int(clipped.sum()),
        "imputation_lag_days": lag_days,
    }
    return out, stats


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def to_single_quarter(
    frame: pd.DataFrame, items: Iterable[str]
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Difference cumulative flows into true quarterly flows.

    Only consecutive rows inside one fiscal year are differenced. Where the
    prior quarter is missing we emit NaN rather than falling back to the
    year-to-date value: the fallback looks like a real reading and is
    systematically too large late in the fiscal year, which is far worse than a
    visible gap.
    """
    out = frame.sort_values(["gvkey", "fyearq", "fqtr"]).copy()
    # fyearq/fqtr arrive from the driver as object dtype. Comparing an object
    # column to an int yields an object array of bools, which then silently
    # fails as a numpy index -- so coerce before any mask arithmetic.
    out["fyearq"] = pd.to_numeric(out["fyearq"], errors="coerce")
    out["fqtr"] = pd.to_numeric(out["fqtr"], errors="coerce")
    items = [i for i in items if i in out.columns]
    flow = [i for i in items if i in CUMULATIVE_FLOW_ITEMS]
    stock = [i for i in items if i in STOCK_ITEMS]
    stats: dict[str, int] = {"n_rows": int(len(out)), "n_flow_items": len(flow)}

    prev = out.groupby("gvkey")[flow + ["fyearq", "fqtr"]].shift(1)
    same_year = prev["fyearq"] == out["fyearq"]
    is_prior_quarter = prev["fqtr"] == (out["fqtr"] - 1)
    # A missing fyearq/fqtr makes the comparison NA, not False. Left as NA the
    # mask cannot be cast to bool at all, so NA is treated as "not usable":
    # an unidentifiable quarter must never be differenced.
    usable = (same_year & is_prior_quarter).fillna(False).to_numpy(dtype=bool)

    first_quarter = (out["fqtr"] == 1).fillna(False).to_numpy(dtype=bool)
    undifferenced = 0
    for item in flow:
        values = np.full(len(out), np.nan)
        # Q1 has nothing to subtract: year-to-date IS the quarter.
        values[first_quarter] = out[item].to_numpy()[first_quarter]
        later = ~first_quarter & usable
        values[later] = out[item].to_numpy(dtype="float64")[later] - prev[item].to_numpy(
            dtype="float64"
        )[later]
        out[f"{item}_q"] = values
        undifferenced += int(
            (~first_quarter & ~usable & out[item].notna().to_numpy(dtype=bool)).sum()
        )
    for item in stock:
        out[f"{item}_q"] = out[item]

    stats["n_flow_values_left_as_nan"] = undifferenced
    return out, stats


def to_ttm(frame: pd.DataFrame, items: Iterable[str], min_periods: int = 4) -> pd.DataFrame:
    """Trailing four quarters of single-quarter values.

    ``min_periods=4`` is not optional here: without it pandas emits partial sums
    during warm-up, and those partial sums are silently half-scale versions of
    the real quantity.
    """
    out = frame.sort_values(["gvkey", "datadate"]).copy()
    single = [f"{i}_q" for i in items if f"{i}_q" in out.columns]
    window = (
        out.groupby("gvkey")[single]
        .rolling(4, min_periods=min_periods)
        .sum()
        .reset_index(level=0, drop=True)
    ).reindex(out.index)
    for column in single:
        out[column.replace("_q", "_ttm")] = window[column]
    return out


def to_yoy(frame: pd.DataFrame, items: Iterable[str]) -> pd.DataFrame:
    """Same-quarter-prior-year comparison.

    Matching on fiscal *quarter* rather than row offset keeps a missing or
    duplicated period from shifting the comparison by one season.
    """
    single = [f"{i}_q" for i in items if f"{i}_q" in frame.columns]
    prior_cols = [f"_yoy_base_{c}" for c in single]

    base = frame[["gvkey", "fyearq", "fqtr"] + single].copy()
    # Object dtype here would turn `+ 1` into string concatenation.
    base["fyearq"] = pd.to_numeric(base["fyearq"], errors="coerce") + 1
    base["fqtr"] = pd.to_numeric(base["fqtr"], errors="coerce")
    base = base.rename(columns=dict(zip(single, prior_cols)))

    out = frame.merge(base, on=["gvkey", "fyearq", "fqtr"], how="left")
    for column, prior_col in zip(single, prior_cols):
        current = out[column].to_numpy(dtype="float64")
        prior = out[prior_col].to_numpy(dtype="float64")
        out[column.replace("_q", "_yoy_diff")] = current - prior
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(np.abs(prior) > BASE_RATIO_FLOOR, current / prior - 1.0, np.nan)
        out[column.replace("_q", "_yoy_pct")] = ratio
    return out.drop(columns=prior_cols)


# ---------------------------------------------------------------------------
# Point-in-time PERMNO linkage
# ---------------------------------------------------------------------------


def attach_permno_at_disclosure(
    fundamentals: pd.DataFrame, link: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Map GVKEY to PERMNO using the link in force on the availability date.

    Picking today's primary link for a historical filing is look-ahead: it uses
    a mapping decision that Compustat published later. The PERMNO that mattered
    is the one whose ``linkdt <= available_date <= linkenddt``.
    """
    left = fundamentals.sort_values("available_date").reset_index()
    right = link.sort_values("link_start")

    # gvkey is object after a SQL fetch and pandas' nullable string after a
    # parquet round trip; merge_asof refuses to join across the two.
    for frame in (left, right):
        frame["gvkey"] = frame["gvkey"].astype(str)

    out = pd.merge_asof(
        left,
        right[["gvkey", "permno", "link_start", "link_end"]],
        left_on="available_date",
        right_on="link_start",
        by="gvkey",
        direction="backward",
        allow_exact_matches=True,
    )
    # Two distinct NaT meanings must not be conflated: NaN because the link is
    # still open-ended (keep), versus NaN because merge_asof found nothing at
    # all (drop -- otherwise such rows continue with a null PERMNO and quietly
    # join to nothing downstream).
    matched = out["permno"].notna()
    open_ended = out["link_end"].isna()
    within_window = out["available_date"] <= out["link_end"]
    valid = matched & (open_ended | within_window)
    stats = {
        "n_rows": int(len(out)),
        "n_rows_matched_to_permno": int(valid.sum()),
        "n_rows_unmatched_drop": int((~valid).sum()),
    }
    return out[valid].reset_index(drop=True), stats


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run(
    *,
    plan: SourcePlan,
    db,
    gvkeys: Sequence[str] | None = None,
    start: str = "2008-01-01",
    end: str = "2025-12-31",
    link: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Pull fundamentals, preferring the PIT vintage where it actually covers.

    The Snapshot product is entitled here but is a historical sample that ends
    years before the backtest window does. Two wrong ways to handle that, both
    tempting:

    * use Snapshot only, and quietly lose most of the sample;
    * use ``fundq`` only, and quietly label the whole series point-in-time.

    Instead each segment is pulled from the best available source and every row
    keeps its ``vintage`` tag, so a factor built on 2013 data is PIT while one
    built on 2019 data is known to be restated.
    """
    WRDS_DIR.mkdir(parents=True, exist_ok=True)
    is_pit = bool(plan.get("fundamentals_is_point_in_time"))
    pit_table = plan.get("fundamentals_table")
    fallback_table = plan.get("fundamentals_fallback_table")
    coverage = plan.get("fundamentals_pit_coverage") or {}
    pit_max = coverage.get("max_datadate")

    segments: list[pd.DataFrame] = []
    segment_notes: list[dict[str, Any]] = []

    if is_pit and pit_table:
        pit_end = min(end, pit_max) if pit_max else end
        print(f"  Pulling PIT fundamentals from {pit_table} up to {pit_end} ...", flush=True)
        pit_frame = pull_fundamentals(
            db, pit_table, gvkeys=gvkeys, start=start, end=pit_end,
            vintage="compustat_snapshot_pit",
        )
        segments.append(pit_frame)
        segment_notes.append(
            {
                "table": pit_table,
                "vintage": "compustat_snapshot_pit",
                "is_point_in_time": True,
                "start": start,
                "end": pit_end,
                "n_rows": int(len(pit_frame)),
            }
        )
        print(
            f"  [snapshot] {len(pit_frame):,} rows, {pit_frame['gvkey'].nunique():,} GVKEYs",
            flush=True,
        )
    else:
        pit_end = None

    if fallback_table:
        fb_start = (pd.Timestamp(pit_end) + pd.Timedelta(days=1)).date().isoformat() if pit_end else start
        if fb_start <= end:
            print(f"  Pulling restated fundamentals from {fallback_table} from {fb_start} ...", flush=True)
            fb_frame = pull_fundamentals(
                db, fallback_table, gvkeys=gvkeys, start=fb_start, end=end,
                vintage="compustat_fundq_as_restated",
            )
            segments.append(fb_frame)
            segment_notes.append(
                {
                    "table": fallback_table,
                    "vintage": "compustat_fundq_as_restated",
                    "is_point_in_time": False,
                    "start": fb_start,
                    "end": end,
                    "n_rows": int(len(fb_frame)),
                }
            )
            print(
                f"  [fundq] {len(fb_frame):,} rows, {fb_frame['gvkey'].nunique():,} GVKEYs",
                flush=True,
            )

    if not segments:
        raise RuntimeError("No fundamentals source resolved; re-run the probe.")

    raw = pd.concat(segments, ignore_index=True, sort=False)
    raw = raw.sort_values(["gvkey", "datadate"]).reset_index(drop=True)
    print(f"  total {len(raw):,} rows for {raw['gvkey'].nunique():,} GVKEYs", flush=True)

    with_avail, avail_stats = availability_dates(raw)
    with_avail = with_avail.drop_duplicates(
        subset=["gvkey", "datadate", "fyearq", "fqtr"], keep="last"
    ).reset_index(drop=True)

    items = [
        c
        for c in raw.columns
        if c in DEFAULT_ITEMS
        and (c in CUMULATIVE_FLOW_ITEMS or c in STOCK_ITEMS)
    ]

    quarterly, quarter_stats = to_single_quarter(with_avail, items)

    # Differencing a cumulative flow across the vintage boundary subtracts a
    # point-in-time opening balance from a restated closing one. Count it rather
    # than let it pass as an ordinary quarterly figure.
    if "vintage" in quarterly.columns:
        prior_vintage = quarterly.groupby("gvkey")["vintage"].shift(1)
        crossed = (
            prior_vintage.notna()
            & (prior_vintage != quarterly["vintage"])
            & (quarterly["fqtr"] != 1)
        )
        quarter_stats["n_values_differenced_across_vintage_boundary"] = int(crossed.sum())

    ttm = to_ttm(quarterly, items)
    final = to_yoy(ttm, items)
    print(
        f"  differenced {quarter_stats['n_flow_items']} cumulative items; "
        f"{quarter_stats['n_flow_values_left_as_nan']:,} later-quarter values left NaN",
        flush=True,
    )

    link_stats: dict[str, int] = {}
    if link is not None and not link.empty:
        final, link_stats = attach_permno_at_disclosure(final, link)
        print(
            f"  [fundq] {link_stats['n_rows_matched_to_permno']:,}/{link_stats['n_rows']:,} "
            "rows carry a PERMNO valid on the disclosure date",
            flush=True,
        )

    pit_rows = int((final["vintage"] == "compustat_snapshot_pit").sum()) if "vintage" in final else 0
    summary = {
        "source_table": pit_table,
        "fallback_table": fallback_table,
        "is_point_in_time": is_pit,
        "segments": segment_notes,
        "n_rows": int(len(final)),
        "n_rows_point_in_time": pit_rows,
        "n_rows_restated": int(len(final)) - pit_rows,
        "n_gvkeys": int(final["gvkey"].nunique()),
        "availability": avail_stats,
        "differencing": quarter_stats,
        "permno_link": link_stats,
    }
    if pit_rows and pit_rows < len(final):
        summary["caveat"] = (
            f"Mixed vintage. {pit_rows:,} rows are true point-in-time "
            f"({pit_table}, ends {pit_max}); {int(len(final)) - pit_rows:,} rows come "
            "from the restated quarterly table, where rdq fixes the disclosure date "
            "but not the vintage and not every field was public on the announcement "
            "date. The per-row 'vintage' column is authoritative -- never describe "
            "the whole series as point-in-time."
        )
        print(f"  WARNING {summary['caveat']}", flush=True)
    elif not is_pit:
        summary["caveat"] = (
            "This series comes from the restated quarterly table. rdq fixes the "
            "disclosure date but not the vintage. Treat any factor built on it as "
            "NON point-in-time."
        )
        print(f"  WARNING {summary['caveat']}", flush=True)

    if quarter_stats.get("n_values_differenced_across_vintage_boundary"):
        print(
            "  WARNING "
            f"{quarter_stats['n_values_differenced_across_vintage_boundary']:,} quarterly "
            "values were differenced across the vintage boundary (PIT opening balance "
            "subtracted from a restated closing one); treat those as mixed-vintage.",
            flush=True,
        )

    write_table(
        final,
        WRDS_DIR / "compustat_quarterly_processed.parquet",
        stage="wrds_fundamentals",
        params={
            "source_table": pit_table,
            "fallback_table": fallback_table,
            "start": start,
            "end": end,
            "lag_days": DEFAULT_LAG_DAYS,
        },
        extra={
            "is_point_in_time": is_pit,
            "n_rows_point_in_time": pit_rows,
            "n_rows_restated": int(len(final)) - pit_rows,
            "summary": summary,
        },
    )
    (WRDS_DIR / "fundamentals_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    return summary


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit("Use `python scripts/wrds_pull.py fundamentals`.")
