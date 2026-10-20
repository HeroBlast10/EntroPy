#!/usr/bin/env python
"""One-command rebuild of every externally reported number.

    python scripts/reproduce_all.py                 # rebuild from pinned inputs
    python scripts/reproduce_all.py --refresh-data  # also re-download raw data
    python scripts/reproduce_all.py --from backtest # skip earlier stages
    python scripts/reproduce_all.py --check-inputs  # are recorded inputs intact?
    python scripts/reproduce_all.py --dry-run       # print the plan only

Why the download step is off by default
---------------------------------------
``build_dataset.py`` pulls prices from yfinance. Today's download differs from
yesterday's: a new session is appended and recent history can be restated. A
run that re-downloads therefore cannot be reproduced byte-for-byte, which would
break the very guarantee this script exists to provide.

So the default path is *computed* reproducibility: reuse the pinned inputs that
are recorded (with their SHA-256) in each artifact's ``run_metadata.json``, and
rebuild every downstream number from them. Pass ``--refresh-data`` only when you
intentionally want to move the inputs forward.

The first four stages retain the legacy Yahoo research workflow. Nonlinear
and aftercosts now require separately rebuilt licensed WRDS inputs, configured
in config/wrds_research.yaml. Their explicit waiver covers restated financials
only; index membership and PERMNO identity are always checked. The explicit
total_loss terminal policy is a sensitivity assumption for missing SIVB/SBNY
payoffs, never an observed return. See docs/wrds_p0.md before interpreting runs.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    # Running as `python scripts/reproduce_all.py` puts scripts/ on sys.path,
    # which hides the package from in-process helpers such as --check-inputs.
    sys.path.insert(0, str(ROOT))

# (name, argv relative to ROOT, touches the network)
STEPS: list[tuple[str, list[str], bool]] = [
    ("dataset", ["scripts/build_dataset.py"], True),
    ("factors", ["scripts/build_factors.py", "--evaluate"], False),
    ("evaluate", ["scripts/evaluate_all_factors_fast.py"], False),
    ("backtest", ["scripts/backtest_factor_candidates.py"], False),
    # Separate WRDS dataset; the financial-vintage waiver cannot bypass keys.
    (
        "nonlinear",
        ["scripts/evaluate_nonlinear_alpha.py", "--config", "config/wrds_research.yaml",
         "--allow-restated-fundamentals", "--missing-terminal-return", "total_loss"],
        False,
    ),
    (
        "aftercosts",
        ["scripts/evaluate_alpha_after_costs.py", "--config", "config/wrds_research.yaml",
         "--allow-restated-fundamentals"],
        False,
    ),
]

NAMES = [name for name, _, _ in STEPS]


def _run(name: str, argv: list[str], python: str) -> tuple[int, float]:
    cmd = [python, *argv]
    print(f"\n{'=' * 68}\n[{name}] $ {' '.join(argv)}\n{'=' * 68}", flush=True)
    started = time.perf_counter()
    completed = subprocess.run(cmd, cwd=str(ROOT))
    elapsed = time.perf_counter() - started
    return completed.returncode, elapsed


def check_inputs() -> int:
    """Compare every recorded input SHA-256 against the file on disk.

    Scans any JSON under ``data/`` that carries an ``inputs`` mapping rather
    than assuming one filename, because the scripts emit several
    (``run_metadata.json``, ``factor_evaluation_run_metadata.json``,
    ``after_cost_metadata.json`` ...) and skipping one would silently exempt
    those artifacts from verification.
    """
    import json

    from quant_platform.core.utils.provenance import fingerprint

    candidates = sorted(
        p
        for p in ROOT.joinpath("data").rglob("*.json")
        # Archived copies are kept for comparison and are expected to drift.
        if "_backup_" not in p.as_posix()
    )
    records = []
    for path in candidates:
        try:
            meta = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError, OSError):
            continue
        if isinstance(meta, dict) and isinstance(meta.get("inputs"), dict):
            records.append((path, meta["inputs"]))
    if not records:
        print("No metadata with recorded inputs found under data/.")
        return 1

    drifted = 0
    total = 0
    for record, inputs in records:
        for role, spec in inputs.items():
            if not spec or spec.get("missing") or not spec.get("sha256"):
                continue
            total += 1
            current = fingerprint(spec["path"])
            if current.get("sha256") != spec["sha256"]:
                drifted += 1
                print(f"  DRIFT  {record.parent.name}/{record.name}: {role} -> {spec['path']}")
    print(f"\n{total - drifted}/{total} recorded inputs unchanged.")
    if drifted:
        print(
            f"{drifted} input(s) differ from what was recorded. Downstream "
            "artifacts must be rebuilt before their numbers can be trusted."
        )
    else:
        print("All pinned inputs match; downstream artifacts can be rebuilt.")
    return 1 if drifted else 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Rebuild all externally reported EntroPy artifacts.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--refresh-data",
        action="store_true",
        help="Also run the network-dependent build_dataset step.",
    )
    parser.add_argument("--only", nargs="+", choices=NAMES, help="Run only these steps.")
    parser.add_argument("--from", dest="from_step", choices=NAMES, help="Start at this step.")
    parser.add_argument("--to", dest="to_step", choices=NAMES, help="Stop after this step.")
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Keep going after a failing step instead of stopping.",
    )
    parser.add_argument(
        "--check-inputs",
        action="store_true",
        help="Verify recorded input hashes still match disk, then exit.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print the plan and exit.")
    args = parser.parse_args()

    if args.check_inputs:
        return check_inputs()

    selected = STEPS
    if args.only:
        keep = set(args.only)
        selected = [step for step in selected if step[0] in keep]
    else:
        if args.from_step:
            selected = selected[NAMES.index(args.from_step):]
        if args.to_step:
            selected = selected[: NAMES.index(args.to_step) + 1]

    python = sys.executable
    print(f"Project root : {ROOT}")
    print(f"Interpreter  : {python}")
    print(f"Steps        : {', '.join(name for name, _, _ in selected) or '(none)'}")
    if any(net for _, _, net in selected) and not args.refresh_data:
        print(
            "\nNote: the 'dataset' step hits the network and is skipped unless "
            "you pass --refresh-data."
        )
        selected = [step for step in selected if not step[2]]

    if args.dry_run:
        for name, argv, _ in selected:
            print(f"  would run [{name}] {' '.join(argv)}")
        return 0

    results: list[tuple[str, int, float]] = []
    for name, argv, _ in selected:
        code, elapsed = _run(name, argv, python)
        results.append((name, code, elapsed))
        status = "OK" if code == 0 else f"FAILED (exit {code})"
        print(f"[{name}] {status} in {elapsed / 60:.1f} min", flush=True)
        if code != 0 and not args.continue_on_error:
            print(f"\nStopping: '{name}' failed.")
            break

    print(f"\n{'=' * 68}\nSummary\n{'=' * 68}")
    for name, code, elapsed in results:
        print(f"  {name:<12} {'PASS' if code == 0 else 'FAIL'}   {elapsed / 60:6.1f} min")
    failed = [name for name, code, _ in results if code != 0]
    if failed:
        print(f"\nFailed step(s): {', '.join(failed)}")
        return 1
    print("\nAll requested artifacts rebuilt.")
    print("Verify the pins with: python scripts/reproduce_all.py --check-inputs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
