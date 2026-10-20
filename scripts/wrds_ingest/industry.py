"""Historical SIC classifications: effective dates are not vintage dates."""
import pandas as pd
from .universe import _int_list


def pull_industry_history(db, permnos, start="2009-01-01", end="2025-12-31"):
    table = "crsp.stksecurityinfohist"
    # This table/column set is verified against the live WRDS catalog before
    # use. Do not silently substitute current company classifications.
    columns = set(db.raw_sql("select column_name from information_schema.columns where "
                            "table_schema='crsp' and table_name='stksecurityinfohist'").column_name)
    if not {"permno", "secinfostartdt", "secinfoenddt", "siccd", "naics"} <= columns:
        raise ValueError("Historical industry table unavailable or changed")
    sql = (f"select permno,secinfostartdt as effective_from,secinfoenddt as effective_to,siccd,naics "
           f"from {table} where permno in ({_int_list(permnos)}) and secinfostartdt<=DATE '{pd.Timestamp(end).date()}' "
           f"and (secinfoenddt is null or secinfoenddt>=DATE '{pd.Timestamp(start).date()}')")
    out = db.raw_sql(sql)
    for c in ("effective_from", "effective_to"):
        out[c] = pd.to_datetime(out[c])
    out["permno"] = out.permno.astype("int64")
    out.attrs["sql"] = sql
    return out.sort_values(["permno", "effective_from"]).reset_index(drop=True)
