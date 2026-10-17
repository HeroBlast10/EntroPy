"""Step 0: verify entitlements and resolve physical tables before any pull.

Two corrections motivated this file, and both are checkable rather than
rememberable:

**Correction 1 -- CRSP SIZ vs CIZ, and the delisting return.**
The legacy SIZ layout stopped receiving updates after the December 2024 tape.
CIZ replaced it, renamed most columns (``prc`` -> ``dlyprc``, ``ret`` ->
``dlyret``), made prices strictly positive, replaced ``shrcd``/``exchcd`` with
five separate columns, and moved the **delisting return into the return series
itself**.  In SIZ every researcher hand-merged ``crsp.dsedelist``; doing that
against CIZ double-counts, and the damage concentrates exactly where the
identification lives -- the delisting tail.

So instead of trusting any of that, we *measure* it: the probe finds rows whose
delisting flag is set and checks whether the main return column already carries
a non-zero value on those rows.  That single test decides whether a downstream
merge is required, forbidden, or impossible (``DLRETX`` has no CIZ substitute).

**Correction 4 -- ``fundq`` + ``rdq`` is not strict point-in-time.**
``rdq`` fixes *when* a number was disclosed but not *which* number: plain
``fundq`` carries the most recently restated values, and its fields were not
necessarily all public on the announcement date.  Compustat Snapshot is the
product that actually separates As-First-Reported / Most-Recently-Restated /
Point-In-Time vintages.  Whether this subscription includes it is an
entitlement fact, so we list it rather than assume it.

Usage::

    python scripts/wrds_pull.py probe
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from quant_platform.core.utils.provenance import build_metadata
from scripts.wrds_ingest.client import (
    PROBE_REPORT_PATH,
    WRDS_DIR,
    connect,
    _iso,
    first_existing_column,
    list_columns,
    scalar,
    table_exists,
)

# Candidate physical tables per logical role, most-preferred first.
# CIZ entries always precede SIZ entries: if the migration has happened we must
# bind to CIZ even though the SIZ table still exists (it does -- WRDS keeps the
# frozen tape online for replication).
DAILY_CANDIDATES = [
    ("crsp", "dsf_v2"),                    # WRDS-built CIZ convenience view
    ("crsp", "stkdlysecuritydata"),        # CRSP-native CIZ
    ("crsp", "stkdlysecurityprimarydata"), # CRSP-native CIZ, 12-column subset
    ("crsp", "dsf"),                       # legacy SIZ, frozen 2024-12-31
]
MONTHLY_CANDIDATES = [
    ("crsp", "msf_v2"),
    ("crsp", "stkmthsecuritydata"),
    ("crsp", "msf"),
]
NAMES_CANDIDATES = [
    ("crsp", "stocknames_v2"),
    ("crsp", "stocknames"),
]
SECURITY_INFO_CANDIDATES = [
    ("crsp", "stksecurityinfohist"),
    ("crsp", "stksecurityinfo"),
]
DELISTING_CANDIDATES = [
    ("crsp", "stkdelists"),
    ("crsp", "dsedelist"),
    ("crsp", "msedelist"),
]
MEMBERSHIP_CANDIDATES = [
    ("crsp_a_indexes", "dsp500list_v2"),
    ("crsp_a_indexes", "dsp500list"),
]
CCM_CANDIDATES = [
    ("crsp", "ccmxpf_lnkhist"),
    ("crsp_a_ccm", "ccmxpf_lnkhist"),
]

# Column roles differ between layouts; resolved empirically per resolved table.
COLUMN_ROLES: dict[str, tuple[str, ...]] = {
    "date": ("dlycaldt", "mthcaldt", "date", "caldt"),
    "permno": ("permno",),
    "permco": ("permco",),
    "price": ("dlyprc", "mthprc", "prc"),
    "return": ("dlyret", "mthret", "ret"),
    "return_ex_dividend": ("dlyretx", "mthretx", "retx"),
    "volume": ("dlyvol", "mthvol", "vol"),
    "shares_outstanding": ("dlyshrout", "mthshrout", "shrout"),
    "market_cap": ("dlycap", "mthcap", "dlymktcap"),
    # CIZ exposes real trade OHLC in addition to the legacy quote extremes.
    # Order matters: prefer the true trade columns, fall back to askhi/bidlo.
    "open": ("dlyopen", "mthopen", "dlyopenprc", "mthopenprc", "openprc"),
    "high": ("dlyhigh", "mthhigh", "dlyhighprc", "mthhighprc", "askhi"),
    "low": ("dlylow", "mthlow", "dlylowprc", "mthlowprc", "bidlo"),
    "close": ("dlyclose", "mthclose"),
    "bid": ("dlybid", "mthbid", "bid"),
    "ask": ("dlyask", "mthask", "ask"),
    "price_flag": ("dlyprcflg", "mthprcflg"),
    "return_missing_flag": ("dlyretmissflg", "mthretmissflg"),
    "return_duration_flag": ("dlyretdurflg", "mthretdurflg"),
    "delisting_flag": ("dlydelflg", "mthdelflg"),
    "ticker": ("ticker", "tic"),
    "company_name": ("issuernm", "comnam", "securitynm"),
    "share_code": ("shrcd",),
    "exchange_code": ("exchcd",),
    "security_type": ("securitytype",),
    "security_subtype": ("securitysubtype",),
    "share_type": ("sharetype",),
    "us_incorporated": ("usincflg",),
    "issuer_type": ("issuertype",),
    "primary_exchange": ("primaryexch",),
    "conditional_type": ("conditionaltype",),
    "trading_status": ("tradingstatusflg",),
    "membership_start": ("_startdt", "start_date", "mbrstartdt", "effdt"),
    "membership_end": ("_enddt", "end_date", "mbrenddt", "enddt", "thrudt"),
}

# Libraries whose presence materially changes what this project can do.
LIBRARY_INTEREST = (
    "crsp",
    "crsp_a_indexes",
    "crsp_a_ccm",
    "crsp_q_indexes",
    "comp",
    "comp_na",
    "comp_global",
    "comp_snapshot",
    "funda",
    "ibes",
    "taq",
    "taqm",
    "taq_monthly",
    "wrds_sec",
    "optionm",
    "djones",
    "cboe",
    "audit",
    "execucomp",
    "gpg",
    "tfn",
    "ciq",
)


def _max_date(db, table: str, date_col: str) -> tuple[str | None, float]:
    """Cheapest reliable max-date, walking a ladder of thresholds.

    ``SELECT max(col)`` over the whole daily file forces a full scan of a
    multi-billion-row table. Walking thresholds descending instead only scans
    the tail, and the first non-NULL answer *is* the maximum.
    """
    thresholds = (
        "2026-06-01",
        "2026-01-01",
        "2025-06-01",
        "2025-01-01",
        "2024-06-01",
        "2020-01-01",
        "2010-01-01",
        "1925-01-01",
    )
    for threshold in thresholds:
        value = scalar(
            db,
            f"SELECT max({date_col}) AS d FROM {table} "
            f"WHERE {date_col} >= DATE '{threshold}'",
        )
        if value is not None and not pd.isna(value):
            iso = value.isoformat()[:10] if hasattr(value, "isoformat") else str(value)
            return iso, float(threshold is thresholds[-1])
    return None, 0.0


def _resolve_candidate(
    db,
    candidates: list[tuple[str, str]],
) -> tuple[str | None, list[str] | None, dict[str, str]]:
    """First existing candidate with its role-resolved column map."""
    for schema, table in candidates:
        if not table_exists(db, schema, table):
            continue
        qualified = f"{schema}.{table}"
        columns = list_columns(db, schema, table)
        resolved: dict[str, str] = {}
        for role, options in COLUMN_ROLES.items():
            picked = first_existing_column(columns, options)
            if picked:
                resolved[role] = picked
        return qualified, columns, resolved
    return None, None, {}


def probe() -> dict[str, Any]:
    db = connect()
    report: dict[str, Any] = {
        "probe_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "entitlements": {},
        "candidates_resolved": {},
        "tables": {},
        "source_plan": {},
        "checks": {},
        "warnings": [],
    }

    # ---------------------------------------------------------------- library
    libraries = sorted(db.list_libraries())
    report["entitlements"]["all_libraries"] = libraries
    report["entitlements"]["interesting_present"] = [
        lib for lib in LIBRARY_INTEREST if lib in libraries
    ]
    report["entitlements"]["interesting_absent"] = [
        lib for lib in LIBRARY_INTEREST if lib not in libraries
    ]

    _emit(
        "Entitled libraries relevant to EntroPy",
        ", ".join(report["entitlements"]["interesting_present"]) or "(none)",
    )

    # ------------------------------------------------------------- CRSP tables
    daily, daily_cols, daily_map = _resolve_candidate(db, DAILY_CANDIDATES)
    monthly, monthly_cols, monthly_map = _resolve_candidate(db, MONTHLY_CANDIDATES)
    names, names_cols, names_map = _resolve_candidate(db, NAMES_CANDIDATES)
    security_info, _, _ = _resolve_candidate(db, SECURITY_INFO_CANDIDATES)
    delisting, delist_cols, _ = _resolve_candidate(db, DELISTING_CANDIDATES)
    membership, member_cols, member_map = _resolve_candidate(db, MEMBERSHIP_CANDIDATES)
    ccm, _, ccm_map = _resolve_candidate(db, CCM_CANDIDATES)

    if daily is None:
        raise SystemExit(
            "No CRSP daily stock table is visible to this account. The entire\n"
            "point-in-time universe rebuild depends on it, so stopping here.\n"
            "Check the WRDS subscription page for 'CRSP - US Stock / Indexes'."
        )

    flavour = "ciz" if daily.split(".")[1] != "dsf" else "siz"
    report["source_plan"]["crsp_flavour"] = flavour
    report["source_plan"].update(
        {
            "daily_table": daily,
            "monthly_table": monthly,
            "names_table": names,
            "security_info_table": security_info,
            "delisting_table": delisting,
            "membership_table": membership,
            "ccm_table": ccm,
            "ccm_columns": ccm_map,
            "daily_columns": daily_map,
            "monthly_columns": monthly_map,
            "names_columns": names_map,
            "membership_columns": member_map,
            # Raw column lists are recorded too: when a role fails to resolve,
            # the report itself is what you inspect to fix the resolution.
            "daily_columns_raw": daily_cols,
            "monthly_columns_raw": monthly_cols,
            "names_columns_raw": names_cols,
            "membership_columns_raw": member_cols,
        }
    )

    date_col = daily_map.get("date")
    if date_col:
        max_date, full_scan = _max_date(db, daily, date_col)
        report["tables"]["daily"] = {
            "table": daily,
            "date_column": date_col,
            "max_date": max_date,
            "used_full_scan": bool(full_scan),
        }
        if max_date and max_date < "2025-01-01":
            report["warnings"].append(
                f"{daily} stops at {max_date}. This is the frozen SIZ tape: WRDS "
                "stopped updating Flat File Format 1.0 after the December 2024 "
                "release. Do not use it for anything after 2024."
            )

    if membership is None:
        report["warnings"].append(
            "No CRSP S&P 500 index-constituent table found. Point-in-time "
            "membership cannot be built, so the survivorship caveat stays."
        )

    _emit("CRSP daily table", f"{daily} (flavour={flavour}, max_date={report['tables']['daily'].get('max_date')})")
    _emit("CRSP monthly table", str(monthly))
    _emit("CRSP names table", str(names))
    _emit("Index membership table", str(membership))
    _emit("CCM link table", str(ccm))

    # ---------------------------------------------- delisting return handling
    report["checks"]["delisting"] = _check_delisting(
        db, daily, daily_map, monthly, monthly_map, delisting, delist_cols
    )
    # Record the empirical verdict on the plan itself: downstream must be able
    # to assert it rather than re-derive it, and must fail if absent.
    report["source_plan"]["delisting_return_is_embedded"] = (
        report["checks"]["delisting"]["verdict"] == "embedded"
    )
    report["source_plan"]["delisting_policy"] = report["checks"]["delisting"]["verdict"]

    # ------------------------------------------- high/low availability (spread)
    report["checks"]["price_columns"] = _check_price_columns(daily_map)

    # ------------------------------------------------------- fundamentals PIT
    report["checks"]["fundamentals"] = _probe_fundamentals(db, libraries)
    report["source_plan"].update(report["checks"]["fundamentals"]["source_plan"])

    # ------------------------------------------------------------------ write
    report["code"] = build_metadata(stage="wrds_probe")["code"]
    WRDS_DIR.mkdir(parents=True, exist_ok=True)
    PROBE_REPORT_PATH.write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8"
    )
    _emit("Probe report written", str(PROBE_REPORT_PATH))
    return report


def report_delisting_columns(columns: list[str] | None) -> dict[str, str]:
    """Resolve PERMNO / delisting-date / dlret roles in the delisting table.

    CIZ renamed these too, and a wrong guess here silently probes the wrong
    security or the wrong date -- which would make the whole check vacuous.
    """
    if not columns:
        return {}
    lowered = {c.lower(): c for c in columns}

    def _pick(*candidates: str) -> str | None:
        for candidate in candidates:
            if candidate.lower() in lowered:
                return lowered[candidate.lower()]
        return None

    delisting_date = _pick(
        "dlstdt", "delistingdt", "securitydlstrtstartdt", "delistingdate",
        "dlydelistingdate",
    )
    if delisting_date is None:
        # Last resort: any column whose name mentions both a date and delisting.
        for column in columns:
            lowered_name = column.lower()
            if "dls" in lowered_name and ("dt" in lowered_name or "date" in lowered_name):
                delisting_date = column
                break
    resolved = {
        "permno": _pick("permno"),
        "delisting_date": delisting_date,
        # CIZ renamed it to delret; the legacy SIZ table uses dlret. Both are
        # listed because which one binds decides whether the comparison below
        # is possible at all.
        "return": _pick("delret", "dlret", "dlydelret", "delistingreturn"),
        # The date on which CRSP stamps the delisting return into the daily
        # file. Aligning on it makes the comparison below exact rather than
        # "nearby", which is what turns this from a plausibility argument into
        # a proof.
        "daily_date": _pick("deldlydt", "deldt"),
    }
    return {k: v for k, v in resolved.items() if v}


def _check_delisting(
    db,
    daily: str,
    daily_map: dict[str, str],
    monthly: str | None,
    monthly_map: dict[str, str],
    delisting: str | None,
    delist_cols: list[str] | None,
) -> dict[str, Any]:
    """Decide empirically whether delisting returns are already in the returns.

    Three outcomes matter downstream, and confusing them is silent:

    embedded
        The main return column is non-zero on rows flagged as delisting ->
        using it straight is correct and merging ``dlret`` would double-count.
    must_merge
        No delisting flag exists in the daily file (the SIZ convention) ->
        the delisting return really does live in a separate table.
    unknown
        A flag exists but every flagged return is NULL -> something changed in
        the tape; refuse to guess.
    """
    result: dict[str, Any] = {
        "delisting_table": delisting,
        "delisting_table_columns": delist_cols,
        "flag_column": daily_map.get("delisting_flag"),
    }
    flag = daily_map.get("delisting_flag")
    ret = daily_map.get("return")
    date_col = daily_map.get("date")

    if not flag or not ret:
        result["verdict"] = "must_merge"
        result["reason"] = (
            f"{daily} has no delisting flag column. This is the legacy layout "
            "where the delisting return lives only in the separate delisting "
            "table and must be merged in."
        )
        if daily_map.get("return_ex_dividend") is None:
            result["warnings"] = ["no delisting-return-ex-dividend column present"]
        _emit("Delisting check", f"must_merge from {delisting} ({result['reason']})")
        return result

    # Deliberately NOT `SELECT count(*) ... WHERE flag = 'Y'` over the whole
    # daily file: that scans billions of rows and can hang for tens of minutes.
    # Inspecting a handful of named securities through the PERMNO index is both
    # faster and stronger evidence -- it lets us compare the main-table return on
    # the delisting row against the dlret held in the separate delisting table.
    # If they match, the return was already folded in and merging is provably
    # double-counting rather than merely "probably".
    delist_map = report_delisting_columns(delist_cols)
    result["delisting_column_map"] = delist_map

    evidence: list[dict[str, Any]] = []
    probed = 0
    if delisting and delist_map.get("permno") and delist_map.get("delisting_date"):
        select = [
            f"{delist_map['permno']} AS permno",
            f"{delist_map['delisting_date']} AS dlstdt",
        ]
        if delist_map.get("return"):
            select.append(f"{delist_map['return']} AS dlret")
        if delist_map.get("daily_date"):
            select.append(f"{delist_map['daily_date']} AS deldlydt")
        subject = db.raw_sql(
            f"SELECT {', '.join(select)} FROM {delisting} "
            f"WHERE {delist_map['delisting_date']} >= DATE '2020-01-01' "
            f"  AND {delist_map['permno']} IS NOT NULL "
            + (f" AND {delist_map['return']} IS NOT NULL" if delist_map.get("return") else "")
            + f" ORDER BY {delist_map['delisting_date']} DESC LIMIT 6"
        )
        for row in subject.itertuples(index=False):
            permno = int(row.permno)
            dlstdt = pd.Timestamp(row.dlstdt)
            # Prefer deldlydt: it names the exact daily row that carries the
            # delisting return, so the window can be tight and the match exact.
            anchor = getattr(row, "deldlydt", None)
            anchor = pd.Timestamp(anchor) if pd.notna(anchor) else dlstdt
            lo = (anchor - pd.Timedelta(days=7)).date().isoformat()
            hi = (anchor + pd.Timedelta(days=7)).date().isoformat()
            window = db.raw_sql(
                f"SELECT {date_col} AS d, "
                f"       {daily_map.get('price','NULL')} AS px, "
                f"       {ret} AS r, {flag} AS flg "
                f"FROM {daily} "
                f"WHERE {daily_map['permno']} = {permno} "
                f"  AND {date_col} BETWEEN DATE '{lo}' AND DATE '{hi}' "
                f"ORDER BY {date_col}"
            )
            if window.empty:
                continue
            flagged = window[window["flg"] == "Y"]
            dlret = getattr(row, "dlret", None)
            evidence.append(
                {
                    "permno": permno,
                    "delisting_date": dlstdt.date().isoformat(),
                    "n_flagged_rows_in_window": int(len(flagged)),
                    "flagged_returns": [
                        None if pd.isna(v) else float(v) for v in flagged["r"].tolist()
                    ],
                    "separate_table_dlret": None if dlret is None or pd.isna(dlret) else float(dlret),
                    "last_trade_rows": [
                        {
                            "date": pd.Timestamp(d).date().isoformat(),
                            "price": None if pd.isna(p) else float(p),
                            "ret": None if pd.isna(r) else float(r),
                            "flag": f,
                        }
                        for d, p, r, f in window.tail(4).itertuples(index=False)
                    ],
                }
            )
            probed += 1
            if probed >= 3:
                break

    result["evidence"] = evidence

    def _flagged_value(item: dict[str, Any]) -> float | None:
        values = [v for v in item["flagged_returns"] if v is not None and v != 0.0]
        return values[-1] if values else None

    matches: list[dict[str, Any]] = []
    n_flagged = 0
    n_nonzero = 0
    for item in evidence:
        n_flagged += item["n_flagged_rows_in_window"]
        value = _flagged_value(item)
        if value is not None:
            n_nonzero += 1
            separate = item["separate_table_dlret"]
            if separate is not None and abs(value - separate) < 1e-6:
                matches.append(
                    {
                        "permno": item["permno"],
                        "delisting_date": item["delisting_date"],
                        "value_in_main_return_series": value,
                        "value_in_delisting_table": separate,
                    }
                )

    result["n_flagged_rows_probed"] = n_flagged
    result["n_flagged_return_nonzero"] = n_nonzero
    result["exact_matches_with_separate_dlret"] = matches

    if matches:
        result["verdict"] = "embedded"
        result["reason"] = (
            f"{len(matches)} of {len(evidence)} probed securities carry the delisting "
            f"return inside {ret} on the row where {flag}='Y', and that value equals the "
            f"dlret stored in {delisting} to within 1e-6. The delisting return is already "
            "folded into the return series; merging it again would double-count."
        )
    elif n_nonzero:
        result["verdict"] = "embedded"
        result["reason"] = (
            f"{n_nonzero} probed securities have {flag}='Y' with a non-zero {ret}. The "
            "delisting return is inside the return series. (No exact dlret comparison was "
            "possible; treat the embargo on merging as still binding.)"
        )
    elif n_flagged:
        result["verdict"] = "unknown"
        result["reason"] = (
            f"{n_flagged} flagged rows were found but none carried a non-zero return. "
            "The tape layout may have changed; refusing to guess."
        )
    else:
        result["verdict"] = "no_delistings_probed"
        result["reason"] = (
            "No delisting rows were found in the probed windows. This is inconclusive, "
            "not evidence of safe merging."
        )
        result["warnings"] = [
            "delisting convention could not be established from the delisting table; "
            "verify manually before merging any dlret"
        ]

    # Monthly, probed on a single security rather than aggregated over the file.
    if monthly and monthly_map.get("delisting_flag") and monthly_map.get("return") and evidence:
        permno = evidence[0]["permno"]
        mpermno = monthly_map["permno"]
        m_flag = monthly_map["delisting_flag"]
        m_ret = monthly_map["return"]
        rows = db.raw_sql(
            f"SELECT {m_flag} AS flg, {m_ret} AS r FROM {monthly} "
            f"WHERE {mpermno} = {permno} AND {m_flag} = 'Y'"
        )
        result["monthly_flagged_returns"] = [
            None if pd.isna(v) else float(v) for v in rows["r"].tolist()
        ]

    if daily_map.get("return_ex_dividend") is None:
        result.setdefault("warnings", []).append(
            "no return-ex-dividend column in this layout; DLRETX has no CIZ substitute"
        )

    _emit("Delisting check", f"{result['verdict']}: {result['reason']}")
    return result


def _check_price_columns(daily_map: dict[str, str]) -> dict[str, Any]:
    """Which of open/high/low/bid/ask the daily file actually carries.

    Corwin-Schultz and Abdi-Ranaldi estimate spreads from High/Low/Close at
    zero marginal storage, but CRSP's traditional high/low (``askhi`` /
    ``bidlo``) are *quote* extremes, not trade extremes. Using them biases the
    estimator, so the distinction is recorded rather than hidden.
    """
    present = {k: daily_map.get(k) for k in ("open", "high", "low", "bid", "ask")}
    quote_only = daily_map.get("high") in ("askhi",) or daily_map.get("low") in ("bidlo",)
    verdict = {
        "present": present,
        "high_low_are_quotes": bool(quote_only),
        "note": (
            "high/low resolve to CRSP quote extremes (askhi/bidlo), not trade "
            "extremes; a Corwin-Schultz estimate built on them is biased low."
            if quote_only
            else "high/low appear to be trade prices; Corwin-Schultz is applicable."
            if present.get("high") and present.get("low")
            else "no high/low columns; spread must come from quotes or an external source."
        ),
    }
    _emit("Price columns", verdict["note"])
    return verdict


def _probe_fundamentals(db, libraries: list[str]) -> dict[str, Any]:
    """Is a genuinely point-in-time Compustat product entitled, or only fundq?

    Discovery is done through ``db.list_tables`` rather than an ``ILIKE`` query:
    a literal ``%`` in the SQL text gets mangled by the driver layer, which is
    exactly the sort of failure that looks like "no Snapshot entitled" and would
    have quietly downgraded the pull to the restated series.

    Coverage is recorded alongside entitlement. Snapshot products can be
    historical samples that stop years short of the backtest window; knowing
    *that* is as important as knowing they exist, because it determines whether
    the pull is PIT throughout or only for part of the sample.
    """
    result: dict[str, Any] = {
        "source_plan": {
            "fundamentals_table": None,
            "fundamentals_fallback_table": None,
            "fundamentals_is_point_in_time": False,
            "fundamentals_vintage": None,
        }
    }

    snapshot_schemas = [lib for lib in libraries if "snapshot" in lib.lower()]
    snapshot_tables: list[str] = []
    for schema in snapshot_schemas:
        try:
            tables = db.list_tables(schema)
        except Exception:  # entitlement can allow a schema but not enumeration
            continue
        snapshot_tables.extend(f"{schema}.{t}" for t in tables)
    result["snapshot_schemas"] = snapshot_schemas
    result["snapshot_tables"] = snapshot_tables

    has_plain = table_exists(db, "comp", "fundq")
    result["has_comp_fundq"] = has_plain

    # Quarterly PIT candidates first; annual is a different granularity.
    quarterly_pit = sorted(
        t
        for t in snapshot_tables
        if t.lower().endswith(("_csq_pit", "csq_pit_sorted")) or "_csq_" in t.lower()
    )
    result["candidate_quarterly_tables"] = quarterly_pit[:20]

    chosen = quarterly_pit[0] if quarterly_pit else None
    coverage: dict[str, Any] = {}
    if chosen:
        schema, table = chosen.split(".", 1)
        row = db.raw_sql(
            f"SELECT count(*) AS n, count(DISTINCT gvkey) AS n_gvkey, "
            f"       min(datadate) AS min_datadate, max(datadate) AS max_datadate "
            f"FROM {schema}.{table}"
        ).iloc[0]
        coverage = {
            "n_rows": int(row["n"]),
            "n_gvkeys": int(row["n_gvkey"]),
            "min_datadate": _iso(row["min_datadate"]),
            "max_datadate": _iso(row["max_datadate"]),
        }
        result["source_plan"]["fundamentals_table"] = chosen
        result["source_plan"]["fundamentals_is_point_in_time"] = True
        result["source_plan"]["fundamentals_vintage"] = "compustat_snapshot_pit"
        result["source_plan"]["fundamentals_pit_coverage"] = coverage
        result["source_plan"]["fundamentals_fallback_table"] = (
            "comp.fundq" if has_plain else None
        )

    if chosen and has_plain:
        end = coverage.get("max_datadate")
        gap = (
            f" It stops at {end}, so everything after that date can only come "
            "from the restated fundq and must be labelled non-PIT."
            if end and end < "2026-01-01"
            else ""
        )
        result["verdict"] = (
            f"Compustat Snapshot IS entitled ({chosen}, "
            f"{coverage.get('n_gvkeys'):,} GVKEYs, "
            f"{coverage.get('min_datadate')} to {coverage.get('max_datadate')})."
            + gap
        )
    elif has_plain:
        result["source_plan"]["fundamentals_table"] = "comp.fundq"
        result["source_plan"]["fundamentals_vintage"] = "compustat_fundq_as_restated"
        result["verdict"] = (
            "Only the restated quarterly table is visible. rdq fixes the "
            "disclosure date but NOT the vintage: values may have been restated "
            "afterwards and not every field was public on the announcement date. "
            "Any factor built on it is labelled non-PIT in the metadata."
        )
    elif chosen:
        result["verdict"] = f"Snapshot found ({chosen}) but comp.fundq is not visible."
    else:
        result["verdict"] = "No Compustat quarterly fundamentals visible."

    _emit("Fundamentals PIT", result["verdict"])
    return result


def _emit(label: str, value: str) -> None:
    print(f"  [probe] {label:<28} {value}", flush=True)


def main() -> int:
    report = probe()
    plan = report["source_plan"]
    print()
    print("=" * 78)
    print("WRDS PROBE VERDICT")
    print("=" * 78)
    print(f"  CRSP layout          : {plan['crsp_flavour'].upper()}")
    print(f"  Daily prices         : {plan['daily_table']}")
    print(f"  Max daily date       : {report['tables']['daily'].get('max_date')}")
    print(
        f"  Delisting returns    : {report['checks']['delisting']['verdict']} "
        f"-> {'do NOT merge dlret' if report['checks']['delisting']['verdict'] == 'embedded' else 'merge required / see report'}"
    )
    print(
        f"  Fundamentals         : {plan.get('fundamentals_table')} "
        f"(PIT={plan.get('fundamentals_is_point_in_time')})"
    )
    missing = [
        k
        for k in ("membership_table", "ccm_table", "names_table")
        if not plan.get(k)
    ]
    if missing:
        print(f"  MISSING roles        : {', '.join(missing)}")
    for warning in report["warnings"]:
        print(f"  WARNING              : {warning}")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
