"""Run provenance for persisted artifacts.

Every artifact written to ``data/`` should be able to answer three questions:

    1. **Which code produced it?**  -> git commit hash, plus whether the working
       tree was dirty at the time.
    2. **What data went in?**       -> SHA-256 of every input file.
    3. **When, and in what env?**   -> UTC timestamp, python / platform / package
       versions.

The ``git_dirty`` flag matters for reproducibility claims: an artifact produced
from a modified working tree cannot be rebuilt from the recorded commit alone,
so downstream consumers must treat ``git_dirty: true`` as "provenance recorded,
not reproducible".

Typical use::

    from quant_platform.core.utils.provenance import write_run_metadata

    write_run_metadata(
        output_dir / "run_metadata.json",
        stage="backtest_factor_candidates",
        inputs={"factors": factors_path, "prices": prices_path},
        params={"initial_capital": 1_000_000.0},
    )
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

# quant_platform/core/utils/provenance.py -> project root
ROOT = Path(__file__).resolve().parents[3]

# Packages whose versions routinely change numerical results.
DEFAULT_PACKAGES: tuple[str, ...] = (
    "numpy",
    "pandas",
    "pyarrow",
    "scipy",
    "scikit-learn",
    "lightgbm",
    "torch",
    "numba",
)

_CHUNK = 8 * 1024 * 1024

# Explicit dependency scope for the independent WRDS research pipeline.
# Paper-broker workflows can be edited concurrently without changing WRDS
# execution. Untracked Python files INSIDE this scope still block a freeze.
WRDS_CODE_PATHS = (
    "quant_platform", ":(exclude)quant_platform/core/execution/paper",
    "scripts/wrds_ingest", "scripts/wrds_pull.py", "scripts/rebuild_wrds_research.py",
    "scripts/verify_wrds_research.py", "scripts/rebuild_wrds_p0.py", "scripts/verify_wrds_p0.py",
    "scripts/evaluate_nonlinear_alpha.py", "scripts/evaluate_alpha_after_costs.py",
    "scripts/reproduce_all.py", "config/wrds_research.yaml", "config/settings.yaml",
    "scripts/build_wrds_factor_library.py", "scripts/diagnose_wrds_factors.py",
    "scripts/compare_wrds_models.py", "scripts/verify_wrds_batch2.py", "config/wrds_batch2.yaml",
    "pyproject.toml", "requirements.txt",
)


def fingerprint(path: Path | str) -> dict[str, Any]:
    """SHA-256, size and mtime of an input file.

    Missing files are reported rather than raising, so that optional inputs
    (e.g. fundamentals) can still be recorded honestly.
    """
    path = Path(path)
    record: dict[str, Any] = {"path": str(path.resolve())}
    if not path.exists():
        record["missing"] = True
        return record
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK), b""):
            digest.update(chunk)
    stat = path.stat()
    record.update(
        {
            "size_bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "sha256": digest.hexdigest(),
        }
    )
    return record


def now_utc() -> str:
    """ISO-8601 UTC timestamp with second resolution."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def git_head(cwd: Path | str = ROOT) -> Optional[str]:
    """Current commit hash, or None when not in a git repository."""
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(cwd),
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def git_dirty(cwd: Path | str = ROOT, *, paths: Sequence[str] | None = None) -> Optional[bool]:
    """True when the working tree has staged or unstaged modifications.

    Returns None if git is unavailable. Untracked files are ignored: they do
    not affect imported code, unlike modifications to tracked files.
    """
    try:
        command = ["git", "status", "--porcelain", "--untracked-files=all" if paths else "--untracked-files=no"]
        if paths:
            command += ["--", *paths]
        result = subprocess.run(
            command,
            cwd=str(cwd),
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return bool(result.stdout.strip())


def package_versions(names: Iterable[str] = DEFAULT_PACKAGES) -> dict[str, Optional[str]]:
    """Installed version per package name; None when not installed."""
    versions: dict[str, Optional[str]] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def build_metadata(
    *,
    stage: str,
    inputs: Mapping[str, Path | str] | None = None,
    params: Mapping[str, Any] | None = None,
    extra: Mapping[str, Any] | None = None,
    packages: Sequence[str] = DEFAULT_PACKAGES,
    git_dir: Path | str = ROOT,
    code_paths: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Assemble the run metadata dictionary.

    ``extra`` keys are merged into the top level so existing consumers keep
    their expected layout.
    """
    head = git_head(git_dir)
    dirty = git_dirty(git_dir, paths=code_paths)
    record: dict[str, Any] = {
        "stage": stage,
        "generated_at_utc": now_utc(),
        "code": {
            "git_head": head,
            "git_dirty": dirty,
            "scope": list(code_paths) if code_paths else ["tracked working tree"],
            "reproducible_from_commit": bool(head) and dirty is False,
            "python": sys.version.split()[0],
            "platform": platform.platform(),
        },
        "packages": package_versions(packages),
    }
    if inputs:
        record["inputs"] = {name: fingerprint(path) for name, path in inputs.items()}
    if params:
        record["params"] = dict(params)
    if extra:
        record.update(extra)
    return record


def write_run_metadata(
    path: Path | str,
    **kwargs: Any,
) -> dict[str, Any]:
    """Build metadata and write it as pretty JSON. Returns the record."""
    record = build_metadata(**kwargs)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")
    return record
