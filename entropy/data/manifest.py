"""Data versioning via a JSON manifest.

Every time the pipeline produces a new dataset snapshot, we record:

* **build_timestamp** — UTC time the build completed.
* **config_hash** — xxHash of ``settings.yaml`` (detects config drift).
* **file_checksums** — per-file xxHash of every Parquet artefact.
* **row_counts** — quick sanity-check numbers.
* **git_tag** — optional ``data-vYYYYMMDD-HHMMSS`` tag.

The manifest is saved to ``data/manifest.json`` and optionally committed /
tagged in git so any experiment can be traced back to the exact data
version it used.
"""

from __future__ import annotations

import datetime as dt
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

import xxhash
from loguru import logger

from entropy.utils.io import get_project_root, load_config, resolve_data_path


# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------

def _file_hash(path: Path, chunk_size: int = 1 << 20) -> str:
    """Return the xxHash-64 hex digest of a file."""
    h = xxhash.xxh64()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _string_hash(s: str) -> str:
    return xxhash.xxh64(s.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Git helpers
# ---------------------------------------------------------------------------

def _git_tag(tag: str, message: str) -> bool:
    """Create an annotated git tag.  Returns True on success."""
    try:
        subprocess.run(
            ["git", "tag", "-a", tag, "-m", message],
            cwd=str(get_project_root()),
            check=True,
            capture_output=True,
        )
        logger.info("Created git tag: {}", tag)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        logger.warning("Git tag failed: {}", exc)
        return False


def _git_head_sha() -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(get_project_root()),
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Manifest build
# ---------------------------------------------------------------------------

def build_manifest(
    parquet_paths: List[Path],
    row_counts: Optional[Dict[str, int]] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Path:
    """Create ``manifest.json`` summarising the current data build.

    Parameters
    ----------
    parquet_paths : list of Parquet files produced in this build.
    row_counts : ``{"prices": N, "universe": M, ...}`` — if not given,
        the manifest records checksums only.
    extra : any additional metadata to embed (e.g. CLI args).

    Returns
    -------
    Path to the saved manifest file.
    """
    cfg = load_config()
    now = dt.datetime.now(dt.timezone.utc)

    # Config hash
    config_path = get_project_root() / "config" / "settings.yaml"
    config_hash = _file_hash(config_path) if config_path.exists() else "n/a"

    # File checksums
    checksums: Dict[str, str] = {}
    for p in parquet_paths:
        p = Path(p)
        if p.exists():
            checksums[p.name] = _file_hash(p)

    manifest: Dict[str, Any] = {
        "build_timestamp": now.isoformat(),
        "config_hash": config_hash,
        "git_head": _git_head_sha(),
        "file_checksums": checksums,
        "row_counts": row_counts or {},
        "date_range": {
            "start": cfg["date_range"]["start"],
            "end": cfg["date_range"]["end"],
        },
    }
    if extra:
        manifest["extra"] = extra

    # Save
    manifest_path = resolve_data_path(cfg["paths"]["manifest_file"])
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, default=str)
    logger.info("Manifest written → {}", manifest_path)

    # Optional git tag
    versioning = cfg.get("versioning", {})
    if versioning.get("auto_tag", False):
        tag = versioning.get("tag_prefix", "data-v") + now.strftime("%Y%m%d-%H%M%S")
        _git_tag(tag, f"Data build {now.isoformat()}")
        manifest["git_tag"] = tag
        # Re-save with tag embedded
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, default=str)

    return manifest_path


def load_manifest(path: Optional[Path | str] = None) -> Dict[str, Any]:
    """Load an existing manifest."""
    if path is None:
        cfg = load_config()
        path = resolve_data_path(cfg["paths"]["manifest_file"])
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def verify_manifest(path: Optional[Path | str] = None) -> bool:
    """Re-hash every file listed in the manifest and compare.

    Returns ``True`` if all checksums match.
    """
    m = load_manifest(path)
    data_root = resolve_data_path()
    ok = True
    for fname, expected in m.get("file_checksums", {}).items():
        fpath = data_root / fname
        if not fpath.exists():
            # Try searching subdirectories
            matches = list(data_root.rglob(fname))
            if matches:
                fpath = matches[0]
            else:
                logger.error("File missing: {}", fname)
                ok = False
                continue
        actual = _file_hash(fpath)
        if actual != expected:
            logger.error("Checksum mismatch: {} (expected {} got {})", fname, expected, actual)
            ok = False
        else:
            logger.debug("OK: {}", fname)
    if ok:
        logger.info("All checksums verified ✓")
    return ok
