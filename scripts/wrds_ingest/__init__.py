"""WRDS acquisition layer for EntroPy.

Everything under this package exists to enforce four constraints that govern
any WRDS pull into this repository:

1. **Verify the CRSP physical format before writing a single query.**  WRDS
   stopped updating the legacy SIZ layout (Flat File Format 1.0) with the
   December 2024 release; CIZ (Flat File Format 2.0) is the only live source.
   The two are *not* a rename-only migration, and querying a frozen table
   produces data that silently stops in December 2024.  Table names are
   therefore resolved by :mod:`scripts.wrds.probe` and read back from the probe
   report; nothing here hardcodes them.

2. **Never add a delisting return to a return that already contains one.**
   CIZ folds the delisting return into ``DlyRet``/``MthRet`` and marks the row
   with a delisting flag.  ``coalesce(dlyret,0)+coalesce(dlret,0)`` and
   ``(1+dlyret)*(1+dlret)-1`` both double-count.  Which convention applies is
   discovered empirically by the probe, not assumed.

3. **PERMNO is the primary key.**  Tickers are time-varying labels: they are
   reused across securities and CHANGE for a single security after a rename.
   All adjacency here is PERMNO-keyed with effective-date intervals; ticker is
   a derived attribute, never a join key.

4. **Point-in-time means the values actually available on that date.**  For
   prices that means CRSP as-of data; for fundamentals it means either the
   Compustat Snapshot product or an explicitly-labelled, possibly-restated
   ``fundq`` fallback.  Either way the distinction is recorded in the artifact,
   never silently elided.

Licensing note
--------------
CRSP, Compustat and their derivatives are subscription-licensed.  Nothing
produced by this package may be committed to the public repository; all output
lands under ``data/wrds/`` which is git-ignored.
"""

from scripts.wrds_ingest.client import (
    PROBE_REPORT_PATH,
    WRDS_DIR,
    SourcePlan,
    connect,
    load_source_plan,
    require_source_plan,
    resolve_credentials,
    write_table,
)

__all__ = [
    "PROBE_REPORT_PATH",
    "WRDS_DIR",
    "SourcePlan",
    "connect",
    "load_source_plan",
    "require_source_plan",
    "resolve_credentials",
    "write_table",
]
