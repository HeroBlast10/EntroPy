"""Historical PERMCO/class market caps, including non-index share classes."""
from __future__ import annotations

import pandas as pd

from .universe import _int_list


def pull_company_classes(db, plan, permnos, start="2009-01-01", end="2025-12-31"):
    if plan.crsp_flavour != "ciz":
        raise ValueError("Historical company adapter requires CIZ")
    table = plan["daily_table"]
    cols = plan["daily_columns"]
    p, d, co = cols["permno"], cols["date"], cols["permco"]
    lo, hi = pd.Timestamp(start).date().isoformat(), pd.Timestamp(end).date().isoformat()
    sql = (f"SELECT DISTINCT {co} AS permco FROM {table} WHERE {p} IN ({_int_list(permnos)}) "
           f"AND {d} BETWEEN DATE '{lo}' AND DATE '{hi}' AND {co} IS NOT NULL")
    companies = db.raw_sql(sql)["permco"].astype(int).tolist()
    queries, frames = [sql], []
    for year in range(pd.Timestamp(start).year, pd.Timestamp(end).year + 1):
        a, b = max(lo, f"{year}-01-01"), min(hi, f"{year}-12-31")
        query = (f"SELECT {p} AS permno, {co} AS permco, {d} AS date, "
                 f"{cols['market_cap']} AS market_cap, securitytype, securitysubtype, "
                 f"sharetype, dlydelflg AS delisting_flag FROM {table} "
                 f"WHERE {co} IN ({_int_list(companies)}) AND {d} BETWEEN DATE '{a}' AND DATE '{b}'")
        frame = db.raw_sql(query)
        frames.append(frame)
        queries.append(query)
        print(f"  company classes {year}: {len(frame):,}", flush=True)
    out = pd.concat(frames, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"])
    out["permno"] = out["permno"].astype("int64")
    out["permco"] = out["permco"].astype("int64")
    if out.duplicated(["date", "permno"]).any():
        raise ValueError("Nonunique company-class tape")
    out.attrs["queries"] = queries
    return out.sort_values(["date", "permno"]).reset_index(drop=True)
