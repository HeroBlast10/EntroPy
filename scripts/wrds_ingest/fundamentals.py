"""Version-aware quarterly fundamentals.

Q income-statement items are already single-quarter flows. Only fiscal YTD
cash-flow items are differenced. Standard fundq is restated research data;
rdq fixes timing, not vintage. Snapshot validity is retained, and every
derived value is computed from versions eligible at its event date.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from quant_platform.core.data.calendar import trading_dates
from scripts.wrds_ingest.client import WRDS_DIR, SourcePlan, write_table

QUARTERLY_FLOW_ITEMS = frozenset({
    "saleq", "revtq", "cogsq", "xsgaq", "xrdq", "oibdpq", "oiadpq",
    "ibq", "niq", "piq", "txtq", "xintq", "dpq", "xoprq", "dvtq",
    "nopiq", "xiq", "miiq", "dpcq", "fiaoq",
})
CUMULATIVE_FLOW_ITEMS = frozenset({
    "oancfy", "ivncfy", "fincfy", "capxy", "txbcofy", "ibcofy",
})
FLOW_ITEMS = QUARTERLY_FLOW_ITEMS | CUMULATIVE_FLOW_ITEMS
STOCK_ITEMS = frozenset({
    "atq", "ltq", "ceqq", "actq", "lctq", "cheq", "dlcq", "dlttq", "invtq",
    "ppegtq", "ppentq", "rectq", "apq", "txditcq", "aoq", "intanq", "gdwlq",
    "mibq", "pstkq", "cshoq", "prccq", "ajexq", "seqq", "wcapq",
})
DEFAULT_ITEMS = tuple(sorted(FLOW_ITEMS | STOCK_ITEMS))
DEFAULT_LAG_DAYS = 90
BASE_RATIO_FLOOR = 1e-8


def _table_columns(db, table: str) -> set[str]:
    if not re.fullmatch(r"[a-z_][a-z_0-9]*\.[a-z_][a-z_0-9]*", table):
        raise ValueError("Invalid source table")
    schema, name = table.split(".")
    return set(db.raw_sql(
        "SELECT column_name FROM information_schema.columns "
        f"WHERE table_schema = '{schema}' AND table_name = '{name}'"
    )["column_name"].str.lower())


def pull_fundamentals(db, table, gvkeys=None, start="2008-01-01", end=None,
                      items=None, vintage="unknown"):
    columns = _table_columns(db, table)
    usable = sorted(set(items or DEFAULT_ITEMS) & columns)
    if not usable:
        raise RuntimeError(f"{table} has no requested fields")
    if not gvkeys or any(not re.fullmatch(r"\d{1,6}", str(g)) for g in gvkeys):
        raise ValueError("A nonempty validated CCM GVKEY list is required")
    start = pd.Timestamp(start).date().isoformat()
    end = pd.Timestamp(end).date().isoformat() if end else "2099-12-31"
    projections = ["gvkey", "datadate", "rdq", "fyearq", "fqtr"]
    projections += [c for c in ("pitdate1", "pitdate2") if c in columns]
    projections += [c for c in ("curcdq",) if c in columns]
    quoted = ",".join(f"'{g}'" for g in gvkeys)
    sql = (f"SELECT {', '.join(projections + usable)} FROM {table} "
           "WHERE indfmt='INDL' AND datafmt='STD' AND popsrc='D' AND consol='C' "
           f"AND datadate BETWEEN DATE '{start}' AND DATE '{end}' "
           f"AND gvkey IN ({quoted})")
    out = db.raw_sql(sql)
    out["gvkey"] = out["gvkey"].astype(str)
    for c in ("datadate", "rdq"):
        out[c] = pd.to_datetime(out[c])
    for c in usable:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    # Keep 9999-12-31 as a string, rather than overflowing pandas timestamps.
    for c in ("pitdate1", "pitdate2"):
        if c in out:
            out[c] = out[c].map(lambda x: str(x)[:10] if pd.notna(x) else None)
    out["vintage"] = vintage
    out.attrs["sql"] = sql
    return out.sort_values(["gvkey", "datadate"]).reset_index(drop=True)


def _next_sessions(values: pd.Series) -> pd.Series:
    dates = pd.to_datetime(values, errors="coerce")
    result = pd.Series(pd.NaT, index=values.index, dtype="datetime64[ns]")
    valid = dates.notna()
    if valid.any():
        sessions = pd.DatetimeIndex(trading_dates(
            dates[valid].min(), dates[valid].max() + pd.Timedelta(days=20), exchange="XNYS"
        )).tz_localize(None)
        result.loc[valid] = sessions.values[np.searchsorted(
            sessions.values, dates[valid].to_numpy(), side="right")]
    return result


def availability_dates(frame, lag_days=DEFAULT_LAG_DAYS):
    out = frame.copy()
    out["datadate"] = pd.to_datetime(out["datadate"])
    primary = pd.to_datetime(out.get("pitdate1", pd.Series(pd.NaT, index=out.index)), errors="coerce")
    rdq = pd.to_datetime(out.get("rdq", pd.Series(pd.NaT, index=out.index)), errors="coerce")
    chosen = primary.fillna(rdq)
    imputed = chosen.isna()
    chosen = chosen.fillna(out["datadate"] + pd.Timedelta(days=lag_days))
    clipped = chosen < out["datadate"]
    out["source_available_date"] = chosen.where(~clipped, out["datadate"])
    out["available_date"] = _next_sessions(out["source_available_date"])
    out["available_date_imputed"] = imputed
    out["available_date_clipped_to_period_end"] = clipped
    end = pd.to_datetime(out.get("pitdate2", pd.Series(pd.NaT, index=out.index)), errors="coerce", format="mixed")
    out["valid_until"] = _next_sessions(end)
    vintage = out.get("vintage", pd.Series("unknown", index=out.index))
    out["is_point_in_time"] = vintage.eq("compustat_snapshot_pit") & primary.notna() & ~imputed & ~clipped
    return out, {"n_available_date_imputed": int(imputed.sum()),
                 "n_available_date_clipped": int(clipped.sum()),
                 "imputation_lag_days": lag_days,
                 "trade_timing": "strictly_next_XNYS_session"}


def _unique_periods(frame):
    if frame.duplicated(["gvkey", "datadate"]).any():
        raise ValueError("Versions must be projected as of a date before aggregation")


def to_single_quarter(frame, items):
    _unique_periods(frame)
    out = frame.sort_values(["gvkey", "datadate"]).copy()
    for c in ("fyearq", "fqtr"):
        out[c] = pd.to_numeric(out[c], errors="coerce")
    prev = out.groupby("gvkey").shift(1)
    gap = (out["datadate"] - prev["datadate"]).dt.days
    usable = (prev["fyearq"].eq(out["fyearq"]) & prev["fqtr"].eq(out["fqtr"] - 1)
              & gap.between(60, 120))
    flow = [i for i in items if i in out and i in CUMULATIVE_FLOW_ITEMS]
    missing = 0
    for i in items:
        if i not in out:
            continue
        if i in CUMULATIVE_FLOW_ITEMS:
            out[f"{i}_q"] = out[i].where(out["fqtr"].eq(1), (out[i] - prev[i]).where(usable))
            missing += int((out[i].notna() & out[f"{i}_q"].isna()).sum())
        elif i in QUARTERLY_FLOW_ITEMS | STOCK_ITEMS:
            out[f"{i}_q"] = out[i]
    return out, {"n_rows": len(out), "n_flow_items": len(flow), "n_flow_values_left_as_nan": missing}


def to_ttm(frame, items, min_periods=4):
    if min_periods != 4:
        raise ValueError("TTM requires exactly four complete quarters")
    _unique_periods(frame)
    out = frame.sort_values(["gvkey", "datadate"]).copy()
    gaps = out.groupby("gvkey")["datadate"].diff().dt.days
    good = gaps.between(60, 120)
    if {"fyearq", "fqtr"} <= set(out):
        ordinal = out["fyearq"] * 4 + out["fqtr"]
        good &= ordinal.groupby(out["gvkey"]).diff().eq(1)
    consecutive = good.groupby(out["gvkey"]).transform(lambda s: s.rolling(3).sum().eq(3))
    for i in items:
        if i in FLOW_ITEMS and f"{i}_q" in out:
            out[f"{i}_ttm"] = out.groupby("gvkey")[f"{i}_q"].transform(
                lambda s: s.rolling(4, min_periods=4).sum()).where(consecutive)
    return out


def to_yoy(frame, items):
    _unique_periods(frame)
    out = frame.copy()
    key = ["gvkey", "fyearq", "fqtr"]
    bases = out[key + ["datadate"]].copy()
    bases["fyearq"] += 1
    bases["_base_row"] = np.arange(len(out))
    bases = bases.rename(columns={"datadate": "_base_date"})
    joined = out[key + ["datadate"]].assign(_row=np.arange(len(out))).merge(bases, on=key)
    joined = joined[(joined["datadate"] - joined["_base_date"]).dt.days.between(300, 430)]
    counts = joined.groupby("_row").size()
    joined = joined[joined["_row"].isin(counts[counts.eq(1)].index)]
    for i in items:
        c = f"{i}_q"
        if c not in out:
            continue
        prior = np.full(len(out), np.nan)
        prior[joined["_row"].to_numpy()] = out[c].to_numpy()[joined["_base_row"].to_numpy()]
        current = out[c].to_numpy(dtype=float)
        out[f"{i}_yoy_diff"] = current - prior
        out[f"{i}_yoy_pct"] = np.divide(current, prior, out=np.full(len(out), np.nan),
                                       where=np.abs(prior) > BASE_RATIO_FLOOR) - 1
    return out


def project_asof_events(raw: pd.DataFrame, end: str) -> pd.DataFrame:
    """Emit latest-period features whenever any eligible version changes.

    Snapshot intervals are interpreted as inclusive source dates; both bounds
    become strictly-next-session timestamps. Missing Snapshot fields can use an
    eligible fundq value, with field-level non-PIT provenance. No dependency
    can have an availability date after the output event.
    """
    source, _ = availability_dates(raw)
    for c in ("fyearq", "fqtr"):
        source[c] = pd.to_numeric(source[c], errors="coerce").astype(float)
    items = [i for i in DEFAULT_ITEMS if i in source]
    # Fiscal-year changes can create multiple rows for the same dated period.
    # Coalesce identical/non-conflicting fields, leave conflicting fields NaN;
    # never arbitrarily pick a fiscal label or multiply observations in joins.
    keys = ["gvkey", "datadate", "vintage", "available_date"]
    repeated = source.duplicated(keys, keep=False)
    if repeated.any():
        coalesced = []
        for _, group in source.loc[repeated].groupby(keys, dropna=False, sort=False):
            row = group.iloc[0].copy()
            fiscal_conflict = False
            for c in ["fyearq", "fqtr", *items]:
                unique = group[c].dropna().unique()
                row[c] = unique[0] if len(unique) == 1 else np.nan
                fiscal_conflict |= c in {"fyearq", "fqtr"} and len(unique) > 1
            if fiscal_conflict:
                row["fyearq"] = row["fqtr"] = np.nan
            coalesced.append(row)
        source = pd.concat([source.loc[~repeated], pd.DataFrame(coalesced)], ignore_index=True)
    result = []
    cutoff = pd.Timestamp(end)
    for gvkey, company in source.groupby("gvkey", sort=False):
        last_output = None
        rows = company.sort_values("available_date").to_dict("records")
        events = sorted({r[c] for r in rows for c in ("available_date", "valid_until")
                         if pd.notna(r[c]) and r[c] <= cutoff})
        periods = sorted({r["datadate"] for r in rows})
        by_period = {p: [r for r in rows if r["datadate"] == p] for p in periods}
        for event in events:
            active = {}
            for period, versions in by_period.items():
                eligible = [r for r in versions if r["available_date"] <= event]
                pit = [r for r in eligible if r["is_point_in_time"] and
                       (pd.isna(r["valid_until"]) or event < r["valid_until"])]
                fallback = [r for r in eligible if r["vintage"] != "compustat_snapshot_pit"]
                if pit or fallback:
                    active[period] = (pit[-1] if pit else fallback[-1], fallback[-1] if fallback else None)
            known_periods = [r["datadate"] for r in rows if r["available_date"] <= event]
            if not active or (known_periods and max(known_periods) > max(active)):
                if last_output is not None:
                    tombstone = {k: (False if k.endswith(("_is_point_in_time", "_mixed_vintage"))
                                      else pd.NaT if k.endswith("_available_date") else np.nan)
                                 for k in last_output}
                    for k in ("gvkey", "datadate", "rdq", "fyearq", "fqtr"):
                        tombstone[k] = last_output[k]
                    tombstone.update(available_date=event, vintage="expired", is_point_in_time=False,
                                     mixed_vintage=False, available_date_imputed=False,
                                     available_date_clipped_to_period_end=False)
                    result.append(tombstone)
                continue
            ordered = sorted(active)
            latest = ordered[-1]
            current = active[latest][0]
            rec = {k: current.get(k) for k in ("gvkey", "datadate", "rdq", "fyearq", "fqtr",
                   "pitdate1", "pitdate2", "source_available_date", "available_date_imputed",
                   "available_date_clipped_to_period_end", "vintage", "curcdq")}
            rec["available_date"] = event
            deps_all = []

            def value(period, item):
                pair = active.get(period)
                if pair is None:
                    return np.nan, []
                chosen, fallback = pair
                if pd.isna(chosen.get(item)) and fallback is not None:
                    chosen = fallback
                v = chosen.get(item, np.nan)
                if pd.isna(v):
                    v = np.nan
                return v, [chosen] if pd.notna(v) else []

            def quarter(period, item):
                v, deps = value(period, item)
                if item not in CUMULATIVE_FLOW_ITEMS or pd.isna(v):
                    return v, deps
                row = active[period][0]
                if row["fqtr"] == 1:
                    return v, deps
                prior = [p for p in active if active[p][0]["fyearq"] == row["fyearq"]
                         and active[p][0]["fqtr"] == row["fqtr"] - 1
                         and 60 <= (period - p).days <= 120]
                if len(prior) != 1:
                    return np.nan, []
                base, prev_deps = value(prior[0], item)
                return v - base, deps + prev_deps if pd.notna(base) else []

            def emit(name, v, deps):
                rec[name] = v
                valid = pd.notna(v) and bool(deps)
                rec[name + "_is_point_in_time"] = valid and all(d["is_point_in_time"] for d in deps)
                rec[name + "_mixed_vintage"] = valid and len({d["vintage"] for d in deps}) > 1
                # A dependency bound is independently auditable, including revisions.
                rec[name + "_available_date"] = max((d["available_date"] for d in deps), default=pd.NaT)
                if valid:
                    deps_all.extend(deps)

            window = ordered[-4:]
            consecutive = len(window) == 4 and all(
                60 <= (b - a).days <= 120 and
                active[b][0]["fyearq"] * 4 + active[b][0]["fqtr"] ==
                active[a][0]["fyearq"] * 4 + active[a][0]["fqtr"] + 1
                for a, b in zip(window, window[1:]))
            prior_year = [p for p in active if active[p][0]["fyearq"] == current["fyearq"] - 1
                          and active[p][0]["fqtr"] == current["fqtr"]
                          and 300 <= (latest - p).days <= 430]
            for item in items:
                v, deps = quarter(latest, item)
                emit(item + "_q", v, deps)
                rec[item] = value(latest, item)[0]
                if item in FLOW_ITEMS:
                    parts = [quarter(p, item) for p in window] if consecutive else []
                    complete = parts and all(pd.notna(x[0]) for x in parts)
                    emit(item + "_ttm", sum(x[0] for x in parts) if complete else np.nan,
                         [d for _, ds in parts for d in ds] if complete else [])
                    if item in {"saleq", "cogsq", "niq", "oancfy"}:
                        old_window = [p for p in ordered if prior_year and p <= prior_year[0]][-4:]
                        old_complete = len(prior_year) == 1 and len(old_window) == 4 and all(
                            60 <= (b - a).days <= 120 and
                            active[b][0]["fyearq"] * 4 + active[b][0]["fqtr"] ==
                            active[a][0]["fyearq"] * 4 + active[a][0]["fqtr"] + 1
                            for a, b in zip(old_window, old_window[1:]))
                        old_parts = [quarter(p, item) for p in old_window] if old_complete else []
                        ok = bool(old_parts) and all(pd.notna(x[0]) for x in old_parts)
                        emit(item + "_ttm_prior_year", sum(x[0] for x in old_parts) if ok else np.nan,
                             [d for _, ds in old_parts for d in ds] if ok else [])
                base, bd = quarter(prior_year[0], item) if len(prior_year) == 1 else (np.nan, [])
                emit(item + "_yoy_diff", v - base, deps + bd)
                emit(item + "_yoy_pct", v / base - 1 if pd.notna(base) and abs(base) > BASE_RATIO_FLOOR else np.nan,
                     deps + bd)
            rec["is_point_in_time"] = bool(deps_all) and all(d["is_point_in_time"] for d in deps_all)
            rec["mixed_vintage"] = len({d["vintage"] for d in deps_all}) > 1
            result.append(rec)
            last_output = rec
    return pd.DataFrame(result).sort_values(["gvkey", "available_date"]).reset_index(drop=True)


def attach_permno_at_disclosure(fundamentals, link):
    left = fundamentals.copy().reset_index(drop=True)
    left["_row"] = np.arange(len(left))
    right = link.copy()
    left["gvkey"] = left["gvkey"].astype(str)
    right["gvkey"] = right["gvkey"].astype(str)
    candidates = left.merge(right[["gvkey", "permno", "link_start", "link_end"]], on="gvkey")
    valid = candidates[(candidates["available_date"] >= candidates["link_start"]) &
                       (candidates["link_end"].isna() | (candidates["available_date"] <= candidates["link_end"]))]
    ambiguous = valid.groupby("_row")["permno"].nunique().gt(1)
    if ambiguous.any():
        raise ValueError(f"Ambiguous active CCM links for {int(ambiguous.sum())} observations")
    valid = valid.sort_values("link_start").drop_duplicates("_row", keep="last").sort_values("_row")
    return valid.drop(columns="_row").reset_index(drop=True), {
        "n_rows": len(left), "n_rows_matched_to_permno": len(valid),
        "n_rows_unmatched_drop": len(left) - len(valid)}


def run(*, plan, db, gvkeys=None, start="2008-01-01", end="2025-12-31", link=None,
        output_dir=None, reuse_raw=False):
    destination = Path(output_dir) if output_dir else WRDS_DIR / "repaired"
    destination.mkdir(parents=True, exist_ok=True)
    segments, notes = [], []
    sources = []
    if plan.get("fundamentals_is_point_in_time"):
        pit_end = min(end, (plan.get("fundamentals_pit_coverage") or {}).get("max_datadate") or end)
        if start <= pit_end:
            sources.append((plan["fundamentals_table"], "compustat_snapshot_pit", pit_end))
    elif plan.get("fundamentals_table"):
        sources.append((plan["fundamentals_table"], "compustat_fundq_as_restated", end))
    if plan.get("fundamentals_fallback_table"):
        if plan["fundamentals_fallback_table"] not in [table for table, _, _ in sources]:
            sources.append((plan["fundamentals_fallback_table"], "compustat_fundq_as_restated", end))
    if not sources:
        raise RuntimeError("No source resolved")
    for table, vintage, hi in sources:
        path = destination / (vintage + "_raw.parquet")
        print(f"Pulling {table}: {start}..{hi}", flush=True)
        if reuse_raw and path.exists():
            frame = pd.read_parquet(path)
            frame = frame[frame["gvkey"].isin(gvkeys) & frame["datadate"].between(start, hi)]
        else:
            if db is None:
                raise RuntimeError(f"Missing raw input {path}; use --refresh-fundamentals")
            frame = pull_fundamentals(db, table, gvkeys, start, hi, vintage=vintage)
            write_table(frame, path, stage="wrds_raw_fundamentals",
                        params={"sql": frame.attrs.get("sql"), "source_table": table},
                        extra={"is_point_in_time": vintage == "compustat_snapshot_pit", "versions_preserved": True})
        segments.append(frame)
        notes.append({"table": table, "vintage": vintage, "n_rows": len(frame), "start": start, "end": hi})
    raw = pd.concat(segments, ignore_index=True)
    # Older raw pulls omitted the currency column. The separate dated pull
    # preserves them intact while making the units of price-scaled ratios explicit.
    currency_path = destination / "fundq_currency.parquet"
    if "curcdq" not in raw and currency_path.exists():
        currency = pd.read_parquet(currency_path).drop_duplicates(["gvkey", "datadate", "curcdq"])
        if currency.duplicated(["gvkey", "datadate"]).any():
            raise ValueError("Ambiguous reporting currency")
        raw = raw.merge(currency, on=["gvkey", "datadate"], how="left", validate="many_to_one")
    _, avail_stats = availability_dates(raw)
    final = project_asof_events(raw, end)
    link_stats = {}
    if link is not None:
        final, link_stats = attach_permno_at_disclosure(final, link)
    pit_rows = int(final["is_point_in_time"].sum())
    summary = {"source_table": plan.get("fundamentals_table"),
               "fallback_table": plan.get("fundamentals_fallback_table"),
               "is_point_in_time": bool(len(final) and pit_rows == len(final)),
               "n_rows": len(final), "n_rows_point_in_time": pit_rows,
               "n_rows_non_pit": len(final) - pit_rows,
               "n_rows_restated": int(final["vintage"].eq("compustat_fundq_as_restated").sum()),
               "n_gvkeys": final["gvkey"].nunique(),
               "n_mixed_vintage_rows": int(final["mixed_vintage"].sum()),
               "n_rows_snapshot_current_vintage": int(final["vintage"].eq("compustat_snapshot_pit").sum()),
               "raw_availability": avail_stats,
               "availability": {**avail_stats,
                    "population": "final linked events through cutoff",
                    "n_available_date_imputed": int(final["available_date_imputed"].sum()),
                    "n_available_date_clipped": int(final["available_date_clipped_to_period_end"].sum())},
               "aggregation_policy": "as-of eligible versions; quarterly income preserved; YTD cashflows differenced; consecutive four-quarter TTM",
               "segments": notes, "permno_link": link_stats, "cutoff": end,
               "caveat": "Restated fundq remains non-PIT even with rdq. Field-level provenance is authoritative."}
    write_table(final, destination / "compustat_quarterly_processed.parquet", stage="wrds_fundamentals",
                params={"start": start, "end": end, "projection": "asof_events"}, extra=summary,
                inputs={**{v: destination / (v + "_raw.parquet") for _, v, _ in sources},
                        **({"currency": currency_path} if currency_path.exists() else {})})
    (destination / "fundamentals_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    return summary
