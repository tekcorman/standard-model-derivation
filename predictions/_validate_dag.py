#!/usr/bin/env python3
"""
predictions/_validate_dag.py

Mechanical enforcement of the predictions/ self-contained DAG contract.

Every predictions/*.py must import ONLY from:
  - Other predictions/ files (by name, no path prefix)
  - Python stdlib
  - Approved third-party libraries (numpy, scipy, sympy, mpmath, etc.)

Forbidden imports: proofs/, docs/, research/, core/, memory/, or any path
outside predictions/.  sys.path manipulations that add such paths are also
flagged.

Exit 0 = clean.  Exit 1 = violations found.
"""

import ast
import re
import sys
from pathlib import Path

PREDICTIONS_DIR = Path(__file__).parent
FORBIDDEN_DIRS = {"proofs", "docs", "research", "core", "memory"}

APPROVED_STDLIB = {
    "math", "cmath", "decimal", "fractions", "numbers", "random",
    "statistics", "itertools", "functools", "operator", "collections",
    "abc", "typing", "dataclasses", "enum", "warnings", "sys", "os",
    "pathlib", "json", "csv", "re", "string", "io", "copy",
    "hashlib", "time", "datetime", "struct", "array", "bisect",
    "heapq", "pprint", "textwrap", "unittest", "contextlib",
    "__future__",
}

APPROVED_THIRDPARTY = {
    "numpy", "scipy", "sympy", "mpmath", "matplotlib",
    "pandas", "sklearn",
}


def _is_approved(module_root: str) -> bool:
    return module_root in APPROVED_STDLIB or module_root in APPROVED_THIRDPARTY


def check_file(path: Path) -> list[str]:
    source = path.read_text()
    violations: list[str] = []

    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError as exc:
        return [f"  Line {exc.lineno}: SyntaxError — {exc.msg}"]

    for node in ast.walk(tree):
        # ── import X or import X.Y ───────────────────────────────────────
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root in FORBIDDEN_DIRS:
                    violations.append(
                        f"  Line {node.lineno}: forbidden `import {alias.name}` "
                        f"(module is in {root}/)"
                    )

        # ── from X import Y ──────────────────────────────────────────────
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            root = module.split(".")[0] if module else ""
            if root in FORBIDDEN_DIRS:
                violations.append(
                    f"  Line {node.lineno}: forbidden `from {module} import ...` "
                    f"(module is in {root}/)"
                )

        # ── sys.path.insert / sys.path.append ────────────────────────────
        elif isinstance(node, ast.Call):
            func = node.func
            if (
                isinstance(func, ast.Attribute)
                and func.attr in ("insert", "append")
                and isinstance(func.value, ast.Attribute)
                and func.value.attr == "path"
            ):
                arg_src = ast.unparse(node)
                for forbidden in FORBIDDEN_DIRS:
                    # Match both quoted strings and variable names containing the word
                    if re.search(
                        rf"""['"/]?{re.escape(forbidden)}['"/]""", arg_src
                    ) or (
                        f"'{forbidden}'" in arg_src or f'"{forbidden}"' in arg_src
                    ):
                        violations.append(
                            f"  Line {node.lineno}: sys.path manipulation "
                            f"references `{forbidden}/` — {ast.unparse(node)!r:.80}"
                        )
                        break  # one violation per call is enough

    # Also do a plain-text scan for hard-to-detect variable-mediated path hacks
    for lineno, line in enumerate(source.splitlines(), start=1):
        stripped = line.strip()
        # Skip comments and strings inside source that happen to contain the word
        if stripped.startswith("#"):
            continue
        for forbidden in FORBIDDEN_DIRS:
            # Catch patterns like: _X = "…/proofs" or join(…, "proofs") used later
            if re.search(
                rf"""os\.path\.(join|abspath|dirname).*['"][^'"]*{re.escape(forbidden)}""",
                line,
            ):
                msg = (
                    f"  Line {lineno}: path construction references `{forbidden}/` — "
                    f"{stripped!r:.80}"
                )
                if msg not in violations:
                    violations.append(msg)

    return violations


def main() -> None:
    py_files = sorted(
        p for p in PREDICTIONS_DIR.glob("*.py") if not p.name.startswith("_")
    )

    all_violations: dict[str, list[str]] = {}
    for path in py_files:
        viols = check_file(path)
        if viols:
            all_violations[path.name] = viols

    checked = len(py_files)
    if not all_violations:
        print(f"OK: {checked} file(s) checked — 0 violations.")
        sys.exit(0)

    total = sum(len(v) for v in all_violations.values())
    print(f"DAG VIOLATIONS: {total} violation(s) in {len(all_violations)} file(s)\n")
    for fname in sorted(all_violations):
        print(f"{fname}:")
        for v in all_violations[fname]:
            print(v)
        print()
    print(
        "Fix: inline the dependency into predictions/, or move the helper "
        "module into predictions/ itself."
    )
    sys.exit(1)


if __name__ == "__main__":
    main()
