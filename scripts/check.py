"""Automated repository compliance auditor and check script for llms-experiments.

Runs code formatting checks, linter, static type checker, and unit tests using uv.
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run_step(name: str, command: list[str]) -> bool:
    print(f"=== [Step] {name} ===")
    print(f"Executing: {' '.join(command)}")
    env = dict(sys.modules["os"].environ)
    env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    result = subprocess.run(command, cwd=ROOT, env=env, check=False)
    if result.returncode != 0:
        print(f"FAILED: {name} exited with code {result.returncode}\n")
        return False
    print(f"PASSED: {name}\n")
    return True


def main() -> None:
    print(f"Starting compliance check for llms-experiments at {ROOT}...\n")
    steps = [
        ("Ruff Format Check", [sys.executable, "-m", "ruff", "format", "--check", "."]),
        ("Ruff Lint Check", [sys.executable, "-m", "ruff", "check", "."]),
        ("Mypy Type Check", [sys.executable, "-m", "mypy", "src"]),
        ("Pytest Test Suite", [sys.executable, "-m", "pytest"]),
    ]

    failed = []
    for name, cmd in steps:
        if not run_step(name, cmd):
            failed.append(name)

    if failed:
        print(f"Compliance audit FAILED. Failed steps: {', '.join(failed)}")
        sys.exit(1)
    else:
        print("ALL COMPLIANCE CHECKS PASSED SUCCESSFULLY.")
        sys.exit(0)


if __name__ == "__main__":
    main()
