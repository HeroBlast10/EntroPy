"""Point-in-time CRSP universe: membership, identifiers and prices.

Everything here is PERMNO-keyed. That is not a stylistic choice -- it is the
substance of correction #2:

* **Tickers are reused.**  A ticker retired with its company is later issued to
  an unrelated issuer. A ticker-keyed history therefore splices two companies
  into one continuous series, and because the splice makes the series *look*
  healthy it never trips a missing-data check.
* **Tickers change without the company changing.**  Renames, and moving between
  share classes, rewrite the label while PERMNO stays fixed.
* **Delisted securities are the whole point.**  Excluding them is precisely the
  survivorship bias we are trying to remove, so prices are pulled for every
  security that was ever a member, including ones that no longer trade.

GVKEY is reached only through the CRSP/Compustat Merged link restricted to its
effective dates (``linkdt``/``linkenddt``). Choosing today's primary link for a
historical observation is look-ahead: it uses a mapping decision published later.
"""

from __future__ import annotations

import json
import sys
from itertools import groupby
from pathlib import Path
from typing import Any, Iterable, Sequence

import pandas as pd

from scripts.wrds_ingest.client import (
    WRDS_DIR,
    SourcePlan,
    connect,
    write_table,
)

DEFAULT_START = "2009-01-01"   # one year of warm-up before the 2010 universe start
DEFAULT_END = "2025-12-31"


# ---------------------------------------------------------------------------
# Membership (PERMNO intervals)
# ---------------------------------------------------------------------------


def _membership_columns(plan: SourcePlan) -> tuple[str, str, str]:
    plan.require("membership_table")
    start = plan["membership_columns"].get("membership_start")
    end = plan["membership_columns"].get("membership_end")
    permno = plan["membership_columns"].get("permno")
    missing = [
        name
        for name, value in (("permno", permno), ("membership_start", start), ("membership_end", end))
        if not value
    ]
    if missing:
        raise RuntimeError(
            f"Could not resolve {missing} in {plan['membership_table']}. "
            "Column names moved between the SIZ and CIZ index layouts; "
            "inspect data/wrds/probe_report.json under 'tables' and update "
            "COLUMN_ROLES in scripts/wrds/probe.py."
        )
    assert permno and start and end
    return permno, start, end


def pull_membership(
    db,
    plan: SourcePlan,
    start: str = DEFAULT_START,
    end: str = DEFAULT_END,
) -> pd.DataFrame:
    """Historical index membership with inclusive start and end dates.

    Overlapping rows are merged, because CRSP records re-entries as separate
    rows and a naive downstream join would either duplicate a security or drop
    one of its spells.
    """
    table = plan["membership_table"]
    p_col, s_col, e_col = _membership_columns(plan)

    sql = (
        f"SELECT {p_col} AS permno, {s_col} AS membership_start, "
        f"       {e_col} AS membership_end "
        f"FROM {table} "
        f"WHERE {s_col} <= DATE '{end}' "
        f"  AND ({e_col} IS NULL OR {e_col} >= DATE '{start}')"
    )
    frame = db.raw_sql(sql)
    frame["permno"] = frame["permno"].astype("int64")
    frame["membership_start"] = pd.to_datetime(frame["membership_start"])
    frame["membership_end"] = pd.to_datetime(frame["membership_end"])

    # An open-ended interval means "still a member as of the tape date".
    frame["is_current_member"] = frame["membership_end"].isna()
    merged = _merge_intervals(frame.sort_values(["permno", "membership_start"]))
    merged["membership_asof"] = pd.Timestamp(end)
    merged["member_at_asof"] = merged["membership_start"].le(pd.Timestamp(end)) & (
        merged["membership_end"].isna() | merged["membership_end"].ge(pd.Timestamp(end)))
    merged["right_censored"] = merged["membership_end"].eq(pd.Timestamp(end))
    return merged


def _merge_intervals(frame: pd.DataFrame) -> pd.DataFrame:
    """Collapse touching/overlapping (start, end) spells per PERMNO."""
    if frame.empty:
        return frame
    out: list[dict[str, Any]] = []
    for permno, grp in frame.sort_values(["permno", "membership_start"]).groupby("permno", sort=True):
        start = grp["membership_start"].to_numpy()
        end = grp["membership_end"].to_numpy()
        cur_start = start[0]
        cur_end = end[0]
        current = bool(grp["is_current_member"].iloc[0])
        for i in range(1, len(start)):
            # Touching or overlapping if the next start <= current end + 1 day.
            gap_ok = pd.isna(cur_end) or start[i] <= cur_end + pd.Timedelta(days=1)
            if gap_ok:
                cur_end = max(cur_end, end[i]) if not (pd.isna(cur_end) or pd.isna(end[i])) else None
                current = current or bool(grp["is_current_member"].iloc[i])
            else:
                out.append(
                    {
                        "permno": permno,
                        "membership_start": cur_start,
                        "membership_end": cur_end,
                        "is_current_member": current,
                    }
                )
                cur_start, cur_end, current = start[i], end[i], bool(
                    grp["is_current_member"].iloc[i]
                )
        out.append(
            {
                "permno": permno,
                "membership_start": cur_start,
                "membership_end": cur_end,
                "is_current_member": current,
            }
        )
    result = pd.DataFrame(out)
    result["membership_start"] = pd.to_datetime(result["membership_start"])
    result["membership_end"] = pd.to_datetime(result["membership_end"])
    return result.sort_values(["permno", "membership_start"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Identifiers: PERMNO -> ticker over effective dates
# ---------------------------------------------------------------------------


def pull_security_names(
    db,
    plan: SourcePlan,
    permnos: Sequence[int] | None = None,
) -> pd.DataFrame:
    """Full name history for the requested PERMNOs.

    We deliberately pull *every* name row per PERMNO, including ones outside the
    backtest window: the earliest row establishes when the security first
    appeared, which is what the ``days_since_ipo`` filter needs.
    """
    plan.require("names_table")
    table = plan["names_table"]
    cols = plan["names_columns"]
    p_col = cols.get("permno")
    if not p_col:
        raise RuntimeError(f"{table} exposes no PERMNO column; cannot key identifiers.")

    raw = [str(c) for c in plan.get("names_columns_raw") or []]
    lowered = {c.lower(): c for c in raw}

    def _pick_role(role_default: str, *candidates: str) -> str:
        for candidate in candidates:
            if candidate.lower() in lowered:
                return lowered[candidate.lower()]
        return cols.get(candidates[0]) or role_default

    start_col = _pick_role("namedt", "namedt", "securitynamestartdt", "startdt")
    end_col = _pick_role("nameenddt", "nameenddt", "securitynameenddt", "enddt")
    ticker_col = _pick_role("ticker", "ticker", "tic")
    name_col = cols.get("company_name")

    select = [
        f"{p_col} AS permno",
        f"{start_col} AS effective_from",
        f"{end_col} AS effective_to",
        f"{ticker_col} AS ticker",
    ]
    if name_col:
        select.append(f"{name_col} AS company_name")
    if cols.get("share_code"):
        select.append(f"{cols['share_code']} AS shrcd")
    if cols.get("exchange_code"):
        select.append(f"{cols['exchange_code']} AS exchcd")
    if cols.get("primary_exchange"):
        select.append(f"{cols['primary_exchange']} AS primaryexch")

    where = ""
    if permnos is not None and len(permnos):
        where = f"WHERE {p_col} IN ({_int_list(permnos)})"

    frame = db.raw_sql(
        f"SELECT {', '.join(select)} FROM {table} {where} "
        f"ORDER BY {p_col}, {start_col}"
    )
    frame["permno"] = frame["permno"].astype("int64")
    frame["effective_from"] = pd.to_datetime(frame["effective_from"])
    frame["effective_to"] = pd.to_datetime(frame["effective_to"])
    return frame.reset_index(drop=True)


def _int_list(values: Iterable[int]) -> str:
    return ",".join(str(int(v)) for v in values)


def attach_ticker(
    targets: pd.DataFrame,
    names: pd.DataFrame,
    *,
    permno_col: str = "permno",
    date_col: str = "date",
) -> pd.DataFrame:
    """Point-in-time ticker lookup: the ticker in force on each date.

    Implemented as an as-of join restricted by the end date. An inner join on
    ticker at the security level would be wrong twice over -- it would carry a
    future ticker backwards and merge two different securities sharing a
    recycled ticker.
    """
    # Deliberately NOT pd.merge_asof(..., by=permno): merge_asof requires the
    # `on` key to be globally monotonic, which sorting by [permno, date] does
    # not satisfy. That passes a single-security unit test and only blows up on
    # the real universe, which is the worst possible place to find out.
    # A bounded interval filter is exact, order-independent, and cheap here
    # because each PERMNO has only a handful of name rows.
    left = targets.copy()
    left[date_col] = pd.to_datetime(left[date_col])
    left["_row"] = range(len(left))

    candidates = left.merge(names, on=permno_col, how="left")
    starts_ok = candidates["effective_from"] <= candidates[date_col]
    ends_ok = candidates["effective_to"].isna() | (
        candidates[date_col] <= candidates["effective_to"]
    )
    matched = candidates[starts_ok & ends_ok]

    if not matched.empty:
        # Two name rows can overlap after an acquisition; take the most recent.
        matched = (
            matched.sort_values(["_row", "effective_from"])
            .drop_duplicates("_row", keep="last")
        )
        left = left.merge(
            matched[["_row"] + [c for c in names.columns if c != permno_col]],
            on="_row",
            how="left",
        )
    else:
        for column in names.columns:
            if column != permno_col:
                left[column] = pd.NA

    out = left.drop(columns=["_row", "effective_from", "effective_to"], errors="ignore")
    return out[list(targets.columns) + [c for c in out.columns if c not in targets.columns]]


def audit_ticker_ambiguity(names: pd.DataFrame) -> dict[str, Any]:
    """Quantify exactly how unsound a ticker-keyed join would be.

    Two pathologies, both silent if undetected:

    ``reuse``
        One ticker mapped to several PERMNOs. Their histories would be spliced.
    ``rename``
        One PERMNO carrying several tickers. Any pipeline keyed on ticker drops
        or duplicates segments of that security's history.
    """
    by_ticker = names.groupby("ticker", sort=False)["permno"].nunique()
    reused = by_ticker[by_ticker > 1]

    by_permno = names.groupby("permno", sort=False)["ticker"].nunique()
    renamed = by_permno[by_permno > 1]

    # Overlapping reuse -- the dangerous case -- needs interval intersection.
    overlapping: list[dict[str, Any]] = []
    for ticker, grp in names[names["ticker"].isin(reused.index)].groupby("ticker"):
        rows = grp.sort_values("effective_from")
        for i in range(len(rows)):
            for j in range(i + 1, len(rows)):
                a, b = rows.iloc[i], rows.iloc[j]
                if a["permno"] == b["permno"]:
                    continue
                a_end = a["effective_to"] if pd.notna(a["effective_to"]) else pd.Timestamp.max
                if b["effective_from"] <= a_end:
                    overlapping.append(
                        {
                            "ticker": ticker,
                            "permno_a": int(a["permno"]),
                            "permno_b": int(b["permno"]),
                            "overlap_start": str(b["effective_from"].date()),
                        }
                    )
    verdict = {
        "n_permnos": int(names["permno"].nunique()),
        "n_tickers": int(names["ticker"].nunique()),
        "n_tickers_bound_to_multiple_permnos": int(len(reused)),
        "n_permnos_with_multiple_tickers": int(len(renamed)),
        "n_simultaneous_ticker_collisions": int(
            len({(o["ticker"], *sorted((o["permno_a"], o["permno_b"]))) for o in overlapping})
        ),
        "examples_simultaneous_collisions": overlapping[:10],
        "verdict": (
            "A ticker-keyed pipeline is UNSOUND for this universe: "
            if overlapping
            else "No simultaneous ticker reuse found, but ticker->permno is still many-to-one; "
        ),
    }
    verdict["recommendation"] = (
        "keep PERMNO as the primary key and treat ticker purely as a display label."
    )
    return verdict


# ---------------------------------------------------------------------------
# CRSP / Compustat link with effective dates
# ---------------------------------------------------------------------------


def pull_ccm_link(db, plan: SourcePlan, permnos: Sequence[int] | None = None) -> pd.DataFrame:
    """(gvkey, permno) links restricted to their validity windows.

    ``linkprim in ('P','C')`` keeps primary and primary-company links, avoiding
    the multiple secondary links that would fan out a single observation into
    several rows. Rows with an open-ended ``linkenddt`` are still live and are
    given a far-future sentinel rather than dropped.
    """
    plan.require("ccm_table")
    table = plan["ccm_table"]
    cols = dict(plan["ccm_columns"]) if plan.get("ccm_columns") else {}
    gvkey = cols.get("gvkey") or "gvkey"
    permno = cols.get("permno") or "lpermno"
    permco = cols.get("permco") or "lpermco"
    start = cols.get("membership_start") or "linkdt"
    end = cols.get("membership_end") or "linkenddt"

    where = [
        "linktype IN ('LU', 'LC')",
        "linkprim IN ('P', 'C')",
    ]
    if permnos is not None and len(permnos):
        where.append(f"{permno} IN ({_int_list(permnos)})")

    frame = db.raw_sql(
        f"SELECT {gvkey} AS gvkey, {permno} AS permno, {permco} AS permco, "
        f"       linktype, linkprim, {start} AS link_start, {end} AS link_end "
        f"FROM {table} "
        f"WHERE {' AND '.join(where)}"
    )
    frame["permno"] = frame["permno"].astype("int64")
    frame["link_start"] = pd.to_datetime(frame["link_start"])
    frame["link_end"] = pd.to_datetime(frame["link_end"])
    return frame.sort_values(["permno", "link_start"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Daily prices for every historical member, delisted ones included
# ---------------------------------------------------------------------------


def _price_select_plan(plan: SourcePlan) -> tuple[list[str], dict[str, str]]:
    daily_map = dict(plan["daily_columns"])
    plan.require("daily_table")
    # Always present, or nothing else works.
    for required in ("permno", "date", "price", "return"):
        if not daily_map.get(required):
            raise RuntimeError(
                f"daily table {plan['daily_table']} has no resolvable {required!r} column"
            )

    exprs = [
        f"{daily_map['permno']} AS permno",
        f"{daily_map['date']} AS date",
        f"{daily_map['price']} AS price",
        f"{daily_map['return']} AS ret",
    ]
    picked = {"permno": "permno", "date": "date", "price": "price", "return": "ret"}
    aliases = {
        "return_ex_dividend": "retx",
        "volume": "volume",
        "shares_outstanding": "shares_outstanding",
        "market_cap": "market_cap",
        "open": "open",
        "high": "high",
        "low": "low",
        "bid": "bid",
        "ask": "ask",
        "price_flag": "price_flag",
        "return_missing_flag": "return_missing_flag",
        "return_duration_flag": "return_duration_flag",
        "delisting_flag": "delisting_flag",
    }
    for role, alias in aliases.items():
        column = daily_map.get(role)
        if column:
            exprs.append(f"{column} AS {alias}")
            picked[role] = alias
    for column in ("dlyfacprc", "dlycumfacpr", "dlycumfacshr", "securitytype",
                   "securitysubtype", "sharetype", "usincflg", "issuertype",
                   "primaryexch", "conditionaltype", "tradingstatusflg"):
        if column in (plan.get("daily_columns_raw") or []):
            exprs.append(column)
    return exprs, picked


def pull_prices(
    db,
    plan: SourcePlan,
    permnos: Sequence[int],
    start: str = DEFAULT_START,
    end: str = DEFAULT_END,
    *,
    verbose: bool = True,
) -> pd.DataFrame:
    """Daily CRSP rows for every requested PERMNO, year by year.

    Chunking by year keeps each round trip bounded and makes an interrupted
    pull resumable. The PERMNO list comes from membership, so delisted members
    are included by construction -- their final delisting row included, which is
    exactly where naive pipelines lose the observation.
    """
    table = plan["daily_table"]
    exprs, _picked = _price_select_plan(plan)
    projections = ", ".join(exprs)

    frames: list[pd.DataFrame] = []
    years = list(range(pd.Timestamp(start).year, pd.Timestamp(end).year + 1))
    for year in years:
        lo = max(start, f"{year}-01-01")
        hi = min(end, f"{year}-12-31")
        if lo > hi:
            continue
        sql = (
            f"SELECT {projections} FROM {table} "
            f"WHERE {plan['daily_columns']['permno']} IN ({_int_list(permnos)}) "
            f"  AND {plan['daily_columns']['date']} BETWEEN DATE '{lo}' AND DATE '{hi}'"
        )
        chunk = db.raw_sql(sql)
        if chunk.empty:
            continue
        chunk["permno"] = chunk["permno"].astype("int64")
        chunk["date"] = pd.to_datetime(chunk["date"])
        frames.append(chunk)
        if verbose:
            print(f"  [prices] {year}: {len(chunk):>9,} rows", flush=True)

    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    return out.sort_values(["permno", "date"]).reset_index(drop=True)


def pull_ciz_supplement(db, plan, permnos, start=DEFAULT_START, end=DEFAULT_END):
    """Dated classification and adjustment factors omitted by older pulls."""
    if plan.crsp_flavour != "ciz":
        raise ValueError("CIZ supplement requires a CIZ source plan")
    columns = set(plan.get("daily_columns_raw") or [])
    wanted = [c for c in ("dlycumfacpr", "dlycumfacshr", "securitytype", "securitysubtype",
              "sharetype", "usincflg", "issuertype", "primaryexch", "conditionaltype",
              "tradingstatusflg") if c in columns]
    if not {"conditionaltype", "tradingstatusflg"} <= set(wanted):
        raise ValueError("Source plan lacks CIZ trading classification")
    p, d = plan["daily_columns"]["permno"], plan["daily_columns"]["date"]
    lo, hi = pd.Timestamp(start).date().isoformat(), pd.Timestamp(end).date().isoformat()
    sql = (f"SELECT {p} AS permno, {d} AS date, {', '.join(wanted)} FROM {plan['daily_table']} "
           f"WHERE {p} IN ({_int_list(permnos)}) AND {d} BETWEEN DATE '{lo}' AND DATE '{hi}'")
    out = db.raw_sql(sql)
    out["date"] = pd.to_datetime(out["date"])
    out["permno"] = out["permno"].astype("int64")
    out.attrs["sql"] = sql
    return out


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run(
    *,
    plan: SourcePlan,
    db,
    start: str = DEFAULT_START,
    end: str = DEFAULT_END,
) -> dict[str, Any]:
    WRDS_DIR.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {"start": start, "end": end}

    print("  Pulling historical index membership (PERMNO-keyed) ...", flush=True)
    membership = pull_membership(db, plan, start, end)
    write_table(
        membership,
        WRDS_DIR / "sp500_membership_permno.parquet",
        stage="wrds_membership",
        params={"start": start, "end": end, "source_table": plan["membership_table"]},
        extra={"n_permnos": int(membership["permno"].nunique())},
    )
    permnos = sorted(membership["permno"].unique().tolist())
    summary["n_permnos"] = len(permnos)
    summary["n_membership_spells"] = int(len(membership))
    delta = plan["membership_table"]
    print(
        f"  [membership] {len(permnos)} PERMNOs over {len(membership)} spells from {delta}",
        flush=True,
    )

    print("  Pulling security name history (renames + ticker reuse) ...", flush=True)
    names = pull_security_names(db, plan, permnos)
    write_table(
        names,
        WRDS_DIR / "crsp_security_names.parquet",
        stage="wrds_security_names",
        params={"source_table": plan["names_table"], "n_permnos": len(permnos)},
    )
    ambiguity = audit_ticker_ambiguity(names)
    summary["ticker_ambiguity"] = ambiguity
    print(
        f"  [names] tickers bound to >1 PERMNO: {ambiguity['n_tickers_bound_to_multiple_permnos']}, "
        f"simultaneous collisions: {ambiguity['n_simultaneous_ticker_collisions']}",
        flush=True,
    )

    if plan.get("ccm_table"):
        print("  Pulling CRSP/Compustat link with effective dates ...", flush=True)
        link = pull_ccm_link(db, plan, permnos)
        write_table(
            link,
            WRDS_DIR / "ccm_gvkey_permno_link.parquet",
            stage="wrds_ccm_link",
            params={"source_table": plan["ccm_table"]},
        )
        summary["n_ccm_links"] = int(len(link))
        summary["n_permnos_with_gvkey"] = int(link["permno"].nunique())
        print(
            f"  [ccm] {len(link)} dated links covering "
            f"{link['permno'].nunique()} PERMNOs",
            flush=True,
        )

    print("  Pulling daily prices for every historical member (incl. delisted) ...", flush=True)
    prices = pull_prices(db, plan, permnos, start, end)
    has_delisting_row = False
    if "delisting_flag" in prices.columns:
        has_delisting_row = bool((prices["delisting_flag"] == "Y").any())
    write_table(
        prices,
        WRDS_DIR / "crsp_daily.parquet",
        stage="wrds_daily_prices",
        params={
            "source_table": plan["daily_table"],
            "start": start,
            "end": end,
            "n_permnos": len(permnos),
        },
        extra={
            "n_rows": int(len(prices)),
            "contains_delisting_return_rows": has_delisting_row,
            "delisting_return_policy": "already_embedded"
            if plan.get("delisting_return_is_embedded")
            else "merged_required",
        },
    )
    summary["n_price_rows"] = int(len(prices))
    summary["contains_delisting_return_rows"] = has_delisting_row
    print(f"  [prices] {len(prices):,} rows written to {WRDS_DIR / 'crsp_daily.parquet'}")

    # A convenience view in the repository's own membership_file schema, plus the
    # honesty file that says whether consuming it is safe.
    spine = membership[["permno", "membership_start", "membership_end"]].copy()
    lookup = spine.rename(columns={"membership_start": "date"})[["permno", "date"]]
    labelled = attach_ticker(lookup, names)
    ticker_view = (
        spine.merge(
            labelled[["permno", "date", "ticker"]].rename(
                columns={"date": "membership_start"}
            ),
            on=["permno", "membership_start"],
            how="left",
        )
        .loc[lambda d: d["ticker"].notna()]
        .loc[:, ["ticker", "membership_start", "membership_end"]]
        .sort_values(["ticker", "membership_start"])
        .reset_index(drop=True)
    )
    write_table(
        ticker_view,
        WRDS_DIR / "membership_file_ticker_view.parquet",
        stage="wrds_membership_ticker_view",
        params={"note": "derived; primary key remains PERMNO"},
    )

    readiness = {
        "primary_key": "permno",
        "membership_is_point_in_time": True,
        "n_permnos": len(permnos),
        "covers_delisted_members": True,
        "ticker_view_safe": ambiguity["n_simultaneous_ticker_collisions"] == 0,
        "blocking_note": (
            "The ticker-keyed convenience view is UNSAFE for this universe because "
            f"{ambiguity['n_simultaneous_ticker_collisions']} tickers are bound to two "
            "different PERMNOs simultaneously. Migrating the pipeline to PERMNO is "
            "required before any result can be called point-in-time."
            if ambiguity["n_simultaneous_ticker_collisions"]
            else "No simultaneous ticker reuse detected, but ticker->PERMNO remains "
            "many-to-one over time; prefer the PERMNO artifacts."
        ),
    }
    (WRDS_DIR / "pit_membership_readiness.json").write_text(
        json.dumps({"readiness": readiness, "ambiguity": ambiguity}, indent=2, default=str),
        encoding="utf-8",
    )
    summary["readiness"] = readiness
    return summary


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit("Use `python scripts/wrds_pull.py universe`.")
