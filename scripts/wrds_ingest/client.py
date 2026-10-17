"""Connection handling, provenance and fail-closed source resolution.

Why nothing here hardcodes a table name
---------------------------------------
CRSP ships two mutually incompatible physical layouts through WRDS:

=============  ==========================  =====================================
Layout         Lifetime                    Daily prices table
=============  ==========================  =====================================
SIZ (FFF 1.0)  through the Dec-2024 tape   ``crsp.dsf``      (columns ``prc``, ``ret``)
CIZ (FFF 2.0)  sole source from Jan-2025   ``crsp.dsf_v2`` / ``crsp.stkdlysecuritydata``
=============  ==========================  =====================================

Beyond the renames, CIZ changes four things that break naive code:

* The **delisting return is embedded** in the return series (SIZ kept it only
  in ``crsp.dsedelist`` where every researcher hand-merged it).
* **Prices are strictly positive**; the bid/ask-average convention that SIZ
  encoded as a *negative* price is now a flag (``dlyprcflg='BA'``).
* ``shrcd``/``exchcd`` became five separate columns.
* **Monthly returns are compounded from daily returns**, not computed from
  month-end prices, so they differ from SIZ.

A correct SIZ query executed in 2026 therefore returns a series that stops in
December 2024 while looking entirely normal.  The only defence is to resolve
the source from the live catalogue, record the resolution, and refuse to run
when it has not been done.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

# scripts/wrds/client.py -> project root
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from quant_platform.core.utils.provenance import build_metadata, write_run_metadata

WRDS_DIR = ROOT / "data" / "wrds"
PROBE_REPORT_PATH = WRDS_DIR / "probe_report.json"

WRDS_POSTGRES_HOST = "wrds-pgdata.wharton.upenn.edu"
WRDS_POSTGRES_PORT = 9737
WRDS_POSTGRES_DB = "wrds"

USERNAME_ENV = "WRDS_USERNAME"
PASSWORD_ENV = "WRDS_PASSWORD"

# Secrets are read from these paths when the environment does not supply them.
# Both are git-ignored; see scripts/wrds_setup_auth.py.
CREDENTIAL_FILES = (
    ROOT / "config" / "wrds_credentials.json",
    ROOT / ".wrds_credentials.json",
)


class SourcePlanMissing(RuntimeError):
    """Raised when a pull is attempted before :mod:`scripts.wrds.probe` has run."""


class SourcePlanStale(RuntimeError):
    """Raised when the recorded plan disagrees with what the caller requires."""


@dataclass(frozen=True)
class SourcePlan:
    """A verified, recorded mapping from logical role to physical WRDS table.

    Instances are produced only by :func:`scripts.wrds.probe.probe`, persisted
    to ``data/wrds/probe_report.json``, and read back here.  Keeping the
    mapping in a file rather than in code means a reader can see *which*
    physical table produced every downstream artifact, and a reviewer can
    check it against the probe transcript.
    """

    payload: Mapping[str, Any]

    def __getitem__(self, key: str) -> Any:
        try:
            value = self.payload[key]
        except KeyError as exc:
            raise SourcePlanStale(
                f"probe report has no key {key!r}; re-run `python scripts/wrds_pull.py probe`"
            ) from exc
        return value

    def get(self, key: str, default: Any = None) -> Any:
        return self.payload.get(key, default)

    def require(self, *keys: str) -> None:
        """Fail closed when any listed role resolved to nothing."""
        missing = [k for k in keys if self.payload.get(k) in (None, "", [], {})]
        if missing:
            raise SourcePlanStale(
                "probe did not resolve: "
                + ", ".join(missing)
                + ". Downstream tables and columns depend on these, so refusing "
                "to guess. Re-run the probe against the live catalogue."
            )

    # -- Convenience accessors for the roles every pull depends on ----------

    @property
    def crsp_flavour(self) -> str:
        return str(self["crsp_flavour"])

    @property
    def generated_at_utc(self) -> str:
        return str(self["generated_at_utc"])


def load_source_plan(path: Path | str = PROBE_REPORT_PATH) -> SourcePlan:
    """Load the recorded source plan. Raises when the probe has not been run."""
    path = Path(path)
    if not path.exists():
        raise SourcePlanMissing(
            f"{path} does not exist.\n"
            "\n"
            "This pull resolves WRDS table names from the live catalogue because\n"
            "the legacy CRSP SIZ layout stopped updating after the December 2024\n"
            "release, while CIZ renamed most columns and moved the delisting\n"
            "return into the return series. Guessing a table name is the single\n"
            "easiest way to silently pull stale or double-counted data.\n"
            "\n"
            "Fix:    python scripts/wrds_pull.py probe"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    return SourcePlan(payload.get("source_plan", payload))


def require_source_plan(path: Path | str = PROBE_REPORT_PATH) -> SourcePlan:
    """Alias that reads as an assertion at the call site."""
    return load_source_plan(path)


# ---------------------------------------------------------------------------
# Credentials
# ---------------------------------------------------------------------------


def resolve_credentials(
    username: str | None = None,
    password: str | None = None,
) -> tuple[str | None, str | None]:
    """Resolve WRDS credentials without ever logging them.

    Resolution order:

    1. Explicit arguments (never used from the CLI; tests only).
    2. ``WRDS_USERNAME`` / ``WRDS_PASSWORD`` environment variables.
    3. ``config/wrds_credentials.json`` or ``.wrds_credentials.json`` (both
       git-ignored; created by ``scripts/wrds_setup_auth.py``).
    4. PostgreSQL's own ``pgpass`` file (written by ``scripts/wrds_setup_auth.py``).

    Step 4 is not optional. libpq reads pgpass by itself, but
    ``wrds.Connection`` *also* insists on being handed an explicit username and
    prompts interactively when it doesn't get one -- which deadlocks any batch
    pull. So we parse pgpass ourselves purely to feed that constructor.
    """
    user = username or os.environ.get(USERNAME_ENV)
    secret = password or os.environ.get(PASSWORD_ENV)
    if user and secret:
        return user, secret

    for candidate in CREDENTIAL_FILES:
        if candidate.exists():
            try:
                data = json.loads(candidate.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            user = user or data.get("username")
            secret = secret or data.get("password")
            if user and secret:
                return user, secret

    pg_user, pg_secret = read_pgpass()
    return user or pg_user, secret or pg_secret


def _split_pgpass_line(line: str) -> list[str]:
    """Split a pgpass entry, honouring ``\\:`` and ``\\\\`` escapes.

    Naive ``line.split(':')`` breaks on any password containing a colon, which
    is common in generated passwords.
    """
    fields: list[str] = []
    buf: list[str] = []
    i = 0
    while i < len(line):
        char = line[i]
        if char == "\\" and i + 1 < len(line) and line[i + 1] in (":", "\\"):
            buf.append(line[i + 1])
            i += 2
            continue
        if char == ":":
            fields.append("".join(buf))
            buf = []
            i += 1
            continue
        buf.append(char)
        i += 1
    fields.append("".join(buf))
    return fields


def read_pgpass(path: Path | str | None = None) -> tuple[str | None, str | None]:
    """Username/password for the WRDS entry in pgpass, if there is one."""
    target = Path(path) if path is not None else pgpass_path()
    if target is None or not target.exists():
        return None, None
    try:
        content = target.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None, None

    for raw in content.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        fields = _split_pgpass_line(line)
        if len(fields) != 5:
            continue
        host, port, dbname, user, secret = fields
        if host not in ("*", WRDS_POSTGRES_HOST):
            continue
        if port not in ("*", str(WRDS_POSTGRES_PORT)):
            continue
        if dbname not in ("*", WRDS_POSTGRES_DB):
            continue
        return user, secret
    return None, None


def pgpass_path() -> Path | None:
    """Location libpq will look for on this platform, if HOME/APPDATA is set."""
    override = os.environ.get("PGPASSFILE")
    if override:
        return Path(override)
    if os.name == "nt":
        appdata = os.environ.get("APPDATA")
        return Path(appdata) / "postgresql" / "pgpass.conf" if appdata else None
    home = os.environ.get("HOME")
    return Path(home) / ".pgpass" if home else None


def connect(username: str | None = None, password: str | None = None, verbose: bool = False):
    """Open a WRDS connection, with an actionable error when credentials are absent.

    ``wrds.Connection`` prompts interactively when it cannot find a username,
    which deadlocks a batch pull.  We therefore check first and fail with the
    exact remedy.
    """
    try:
        import wrds
    except ImportError as exc:  # pragma: no cover - environment problem
        raise SystemExit(
            "The `wrds` package is not available to this interpreter.\n"
            f"    {sys.executable} -m pip install wrds"
        ) from exc

    user, secret = resolve_credentials(username, password)
    target = pgpass_path()
    have_pgpass = bool(target and target.exists())
    if not user:
        detail = (
            f"pgpass exists at {target} but contains no WRDS entry "
            f"({WRDS_POSTGRES_HOST}:{WRDS_POSTGRES_PORT}:{WRDS_POSTGRES_DB}); "
            "expected one line of hostname:port:database:username:password."
            if have_pgpass
            else f"no pgpass file at {target}"
        )
        raise SystemExit(
            "No WRDS username found. Provide one via any of:\n"
            "  1. python scripts/wrds_setup_auth.py          (writes a pgpass file)\n"
            f"  2. set {USERNAME_ENV} / {PASSWORD_ENV} in the environment\n"
            "  3. create config/wrds_credentials.json (git-ignored):\n"
            '       {"username": "...", "password": "..."}\n'
            f"\n{detail}"
        )

    kwargs: dict[str, Any] = {"wrds_username": user}
    if secret:
        kwargs["wrds_password"] = secret
    kwargs.setdefault("wrds_hostname", WRDS_POSTGRES_HOST)
    kwargs.setdefault("wrds_port", WRDS_POSTGRES_PORT)
    kwargs.setdefault("wrds_dbname", WRDS_POSTGRES_DB)

    return wrds.Connection(autoconnect=True, verbose=verbose, **kwargs)


# ---------------------------------------------------------------------------
# Catalogue introspection
# ---------------------------------------------------------------------------


def table_exists(db, schema: str, table: str) -> bool:
    sql = (
        "SELECT 1 FROM information_schema.tables "
        f"WHERE table_schema = '{schema}' AND table_name = '{table}' LIMIT 1"
    )
    frame = db.raw_sql(sql)
    return bool(len(frame))


def list_columns(db, schema: str, table: str) -> list[str]:
    """Column names in physical order."""
    frame = db.raw_sql(
        "SELECT column_name FROM information_schema.columns "
        f"WHERE table_schema = '{schema}' AND table_name = '{table}' "
        "ORDER BY ordinal_position"
    )
    return [str(v) for v in frame["column_name"].tolist()]


def first_existing_column(columns: Sequence[str], candidates: Sequence[str]) -> str | None:
    """Pick the first candidate present, case-insensitively.

    Physical column names differ between the SIZ and CIZ layouts and, for CIZ,
    between the WRDS-built ``_v2`` views and the CRSP-native tables. Resolving
    by candidate list keeps every column role explicit and reviewable.
    """
    lowered = {c.lower(): c for c in columns}
    for candidate in candidates:
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]
    return None


def scalar(db, sql: str) -> Any:
    frame = db.raw_sql(sql)
    if frame.empty:
        return None
    return frame.iloc[0, 0]


def table_extent(db, table: str, date_col: str) -> dict[str, Any]:
    """Row count and coverage window of a table."""
    row = db.raw_sql(
        f"SELECT count(*) AS n_rows, min({date_col}) AS min_date, max({date_col}) AS max_date "
        f"FROM {table}"
    ).iloc[0]
    return {
        "n_rows": int(row["n_rows"]),
        "min_date": _iso(row["min_date"]),
        "max_date": _iso(row["max_date"]),
    }


def _iso(value: Any) -> str | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, (datetime, date)):
        return value.isoformat()[:10]
    return str(value)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def write_table(
    frame: pd.DataFrame,
    destination: Path | str,
    *,
    stage: str,
    params: Mapping[str, Any] | None = None,
    extra: Mapping[str, Any] | None = None,
    index: bool = False,
) -> Path:
    """Write parquet atomically and record provenance next to it.

    Atomicity matters more than it looks: a pull interrupted mid-write leaves a
    truncated parquet that reads fine but silently drops rows, and there is no
    checksum that would catch it. Writing to ``.tmp`` then replacing means a
    failed pull leaves the previous good artifact untouched.
    """
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_suffix(destination.suffix + ".tmp")
    if index:
        frame.to_parquet(tmp, index=True)
    else:
        frame.to_parquet(tmp, index=False)
    tmp.replace(destination)

    meta_path = destination.with_name(destination.stem + "_run_metadata.json")
    write_run_metadata(
        meta_path,
        stage=stage,
        inputs={"source_plan": PROBE_REPORT_PATH},
        params=dict(params or {}),
        extra={
            "artifact": destination.name,
            "n_rows": int(len(frame)),
            "columns": list(frame.columns),
            **dict(extra or {}),
        },
    )
    return destination


def build_probe_metadata(**kwargs: Any) -> dict[str, Any]:
    return build_metadata(**kwargs)
