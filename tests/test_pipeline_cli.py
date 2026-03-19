#!/usr/bin/env python
"""Quick test to verify the factor pipeline CLI is working correctly."""

import subprocess
import sys
from pathlib import Path

def run_command(cmd, description):
    """Run a command and check if it succeeds."""
    print(f"\n{'='*80}")
    print(f"Testing: {description}")
    print(f"Command: {cmd}")
    print(f"{'='*80}")
    
    result = subprocess.run(
        cmd,
        shell=True,
        capture_output=True,
        text=True,
    )
    
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    
    if result.returncode == 0:
        print(f"✓ {description} - PASSED")
        return True
    else:
        print(f"✗ {description} - FAILED (exit code: {result.returncode})")
        return False


def main():
    """Run pipeline CLI tests."""
    print("=" * 80)
    print("Factor Pipeline CLI Test Suite")
    print("=" * 80)
    
    tests = [
        ("python scripts/run_factor_pipeline.py --help", "Help command"),
    ]
    
    # Only test --list if factor_comparison.csv exists
    comp_csv = Path("data/factors/factor_comparison.csv")
    if comp_csv.exists():
        tests.append((
            "python scripts/run_factor_pipeline.py --list",
            "List factors command"
        ))
    else:
        print(f"\nℹ️  Skipping --list test (factor_comparison.csv not found)")
        print("   Run: python scripts/build_factors.py --evaluate")
    
    results = []
    for cmd, desc in tests:
        results.append(run_command(cmd, desc))
    
    # Summary
    print("\n" + "=" * 80)
    print("Test Summary")
    print("=" * 80)
    passed = sum(results)
    total = len(results)
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("\n✓ All tests passed!")
        return 0
    else:
        print(f"\n✗ {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
