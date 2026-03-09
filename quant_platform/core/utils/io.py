"""I/O helpers: config loading, Parquet read/write, path resolution."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from loguru import logger

# ---------------------------------------------------------------------------
# Project root detection
# ---------------------------------------------------------------------------

_PROJECT_ROOT: Optional[Path] = None


def get_project_root() -> Path:
    """Return the project root (directory containing ``config/``)."""
    global _PROJECT_ROOT
    if _PROJECT_ROOT is not None:
        return _PROJECT_ROOT
    # Walk up from this file until we find config/settings.yaml
    cur = Path(__file__).resolve().parent
    for _ in range(10):
        if (cur / "config" / "settings.yaml").exists():
            _PROJECT_ROOT = cur
            return cur
        cur = cur.parent
    raise FileNotFoundError("Cannot locate project root (config/settings.yaml not found).")


def set_project_root(path: Path | str) -> None:
    """Override the auto-detected project root."""
    global _PROJECT_ROOT
    _PROJECT_ROOT = Path(path).resolve()


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_CONFIG: Optional[Dict[str, Any]] = None


def load_config(path: Optional[Path | str] = None) -> Dict[str, Any]:
    """Load and cache ``config/settings.yaml``."""
    global _CONFIG
    if _CONFIG is not None and path is None:
        return _CONFIG
    if path is None:
        path = get_project_root() / "config" / "settings.yaml"
    else:
        path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        _CONFIG = yaml.safe_load(f)
    logger.info("Config loaded from {}", path)
    return _CONFIG


def resolve_data_path(*parts: str) -> Path:
    """Resolve a path relative to ``data_root`` defined in config."""
    cfg = load_config()
    root = get_project_root() / cfg["paths"]["data_root"]
    return root.joinpath(*parts)


# ---------------------------------------------------------------------------
# Parquet I/O
# ---------------------------------------------------------------------------


def save_parquet(
    df: pd.DataFrame,
    path: Path | str,
    schema: Optional[pa.Schema] = None,
) -> Path:
    """Write *df* to Parquet with optional schema enforcement.

    Creates parent directories as needed.  Returns the resolved path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    table = pa.Table.from_pandas(df, preserve_index=False)
    if schema is not None:
        table = table.cast(schema)

    pq.write_table(table, path, compression="zstd")
    logger.info("Saved {} rows → {}", len(df), path)
    return path


def load_parquet(path: Path | str, columns: Optional[list[str]] = None) -> pd.DataFrame:
    """Read a Parquet file, optionally selecting *columns*."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Parquet file not found: {path}")
    table = pq.read_table(path, columns=columns)
    df = table.to_pandas()
    logger.debug("Loaded {} rows ← {}", len(df), path)
    return df
