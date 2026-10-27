"""Read-only, resumable WRDS extension pulls; existing S&P evidence is immutable."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pandas as pd
import psycopg2
from psycopg2 import sql
from scripts.wrds_ingest.client import (resolve_credentials, load_source_plan,
    WRDS_POSTGRES_HOST, WRDS_POSTGRES_PORT, WRDS_POSTGRES_DB)
from quant_platform.core.utils.provenance import fingerprint, write_run_metadata


def connection():
    user, password = resolve_credentials()
    if not user or not password:
        raise RuntimeError("WRDS credentials are unavailable; no interactive prompt is allowed")
    db = psycopg2.connect(host=WRDS_POSTGRES_HOST, port=WRDS_POSTGRES_PORT,
        dbname=WRDS_POSTGRES_DB, user=user, password=password,
        connect_timeout=20, sslmode="require", application_name="EntroPy research extensions")
    db.set_session(readonly=True, autocommit=True)
    with db.cursor() as cursor:
        cursor.execute("SET statement_timeout='240s'")
    return db


def frame(db, query, params=()):
    with db.cursor() as cursor:
        cursor.execute(query, params)
        result = pd.DataFrame.from_records(cursor.fetchall(), columns=[v.name for v in cursor.description])
    return result


def broad_raw(output, start=2008, end=2025):
    output.mkdir(parents=True, exist_ok=True)
    plan = load_source_plan()
    if plan.crsp_flavour != "ciz" or not plan.get("delisting_return_is_embedded"):
        raise ValueError("This extension requires the verified CIZ embedded-return source")
    table = plan["daily_table"]
    schema, name = table.split(".")
    columns = {"date": "dlycaldt", "permno": "permno", "permco": "permco", "siccd": "siccd",
        "price": "dlyprc", "ret": "dlyret", "volume": "dlyvol", "shares_outstanding": "shrout",
        "market_cap": "dlycap", "open": "dlyopen", "high": "dlyhigh", "low": "dlylow",
        "delisting_flag": "dlydelflg", "securitytype": "securitytype", "securitysubtype": "securitysubtype",
        "sharetype": "sharetype", "usincflg": "usincflg", "primaryexch": "primaryexch",
        "conditionaltype": "conditionaltype", "tradingstatusflg": "tradingstatusflg",
        "cumfacshr": "dlycumfacshr"}
    query = sql.SQL("SELECT {} FROM {}.{} WHERE dlycaldt >= %s AND dlycaldt < %s "
        "AND securitytype='EQTY' AND securitysubtype='COM' ORDER BY dlycaldt,permno").format(
        sql.SQL(",").join(sql.SQL("{} AS {}").format(sql.Identifier(v), sql.Identifier(k)) for k,v in columns.items()),
        sql.Identifier(schema), sql.Identifier(name))
    db = connection()
    try:
        known = frame(db, "SELECT column_name FROM information_schema.columns WHERE table_schema=%s AND table_name=%s", (schema,name))
        if not set(columns.values()).issubset(set(known.column_name)):
            raise ValueError("Live CRSP columns disagree with the frozen source plan")
        records = []
        for year in range(start, end + 1):
            path = output / f"common_{year}.parquet"
            audit = output / f"common_{year}.json"
            if path.exists() and audit.exists():
                record = json.loads(audit.read_text())
                if fingerprint(path)["sha256"] != record["sha256"]:
                    raise ValueError(f"Corrupt CRSP year checkpoint: {year}")
                records.append(record)
                continue
            if path.exists():
                raise ValueError(f"Unverified partial checkpoint: {path}")
            began = time.monotonic()
            data = frame(db, query, (f"{year}-01-01", f"{year+1}-01-01"))
            if data.empty or data.duplicated(["date","permno"]).any():
                raise ValueError(f"Empty or duplicate CRSP year: {year}")
            data["date"] = pd.to_datetime(data.date)
            for key in ("permno", "permco"):
                data[key] = data[key].astype("int64")
            for key in columns:
                if key not in {"date","permno","permco","siccd","delisting_flag","securitytype","securitysubtype",
                               "sharetype","usincflg","primaryexch","conditionaltype","tradingstatusflg"}:
                    data[key] = pd.to_numeric(data[key], errors="coerce")
            data.to_parquet(path, index=False)
            record = {"year": year, "rows": len(data), "names": int(data.permno.nunique()),
                "sha256": fingerprint(path)["sha256"], "table": table,
                "min_date": str(data.date.min()), "max_date": str(data.date.max())}
            audit.write_text(json.dumps(record, indent=2), encoding="utf-8")
            records.append(record)
            print(f"CRSP {year}: {len(data):,} rows, {data.permno.nunique():,} names, {time.monotonic()-began:.1f}s", flush=True)
        write_run_metadata(output / "completion.json", stage="broad_ciz_common_equity_pull",
            inputs={"driver": Path(__file__), "source_plan": ROOT / "data/wrds/probe_report.json"},
            params={"start": start, "end": end, "source": table, "partitions": records,
                "selection": "all historical common share classes, including delisted; no present-day constituent list",
                "delisting_return_embedded": True}, extra={"complete": True})
    finally:
        db.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "data/wrds/batch4/broad_raw")
    parser.add_argument("--start-year", type=int, default=2008)
    parser.add_argument("--end-year", type=int, default=2025)
    args = parser.parse_args()
    broad_raw(args.output.resolve(), args.start_year, args.end_year)
