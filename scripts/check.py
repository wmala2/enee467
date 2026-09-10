#!/usr/bin/env python3
"""Lint/format/type-check the whole repository, high-level packages and rover_mujoco alike.

    uv run scripts/check.py          # report problems, change nothing
    uv run scripts/check.py --fix    # apply ruff's formatter and its safe autofixes first

Rules live in the root pyproject.toml's [tool.ruff] section (Google Python Style Guide).
Every check runs even if an earlier one fails, so one pass shows the whole picture.
"""

import subprocess
import sys


def run(name, *cmd):
    """Runs one checker, echoing its command line. Returns True if it passed."""
    print(f"\n=== {name}: {' '.join(cmd)}", flush=True)
    return subprocess.run(cmd, check=False).returncode == 0


def main():
    fix = "--fix" in sys.argv[1:]

    if fix:
        checks = [
            ("format", "ruff", "format", "."),
            ("lint", "ruff", "check", "--fix", "."),
        ]
    else:
        checks = [
            ("format", "ruff", "format", "--check", "."),
            ("lint", "ruff", "check", "--no-fix", "."),
        ]
    checks.append(("types", "ty", "check"))

    failed = [name for name, *cmd in checks if not run(name, *cmd)]

    print()
    if failed:
        print(f"FAILED: {', '.join(failed)}")
        if not fix:
            print("Some lint and formatting problems fix themselves: uv run scripts/check.py --fix")
        return 1
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
