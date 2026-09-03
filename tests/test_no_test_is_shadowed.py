"""**A test defined twice runs once, and the first one is gone.**

Python's module body is executed top to bottom, so a second `def` with the
same name at the same scope simply rebinds it.  Pytest collects the module's
namespace afterwards and sees one function — the survivor.  The loser leaves
no warning, no skip and no error: it reads as coverage in the file, sits in
review diffs, matches a grep for its own subject, and is not in the suite.

**Found on 2026-09-02**, in `test_molview_mount.py`:
`test_a_picture_is_the_view_and_leaves_through_saveBinary` was defined twice,
48 lines apart, character-identical but for whitespace inside one
`console.log`.  Neither an editor nor a linter run in this repo objected, and
the count of tests "passing" was one short of the count in the file for as
long as it stood.

It is a paste artefact, so the fix is not vigilance -- it is this file.
"""
from __future__ import annotations

import ast
from collections import defaultdict
from pathlib import Path

import pytest

TESTS = Path(__file__).resolve().parent


def _shadowed(tree: ast.AST) -> list[tuple[str, list[int]]]:
    """Every name bound more than once in one scope, with its line numbers.

    Module scope and each class body, because a duplicated method inside a
    `Test...` class disappears exactly the same way and is the harder of the
    two to see.  Nested function scopes are not checked: a helper redefined
    inside one function is local and cannot silently drop a test.
    """
    out: list[tuple[str, list[int]]] = []

    def scan(body, prefix: str) -> None:
        seen: dict[str, list[int]] = defaultdict(list)
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                                 ast.ClassDef)):
                seen[node.name].append(node.lineno)
            if isinstance(node, ast.ClassDef):
                scan(node.body, prefix + node.name + "::")
        for name, lines in seen.items():
            if len(lines) > 1:
                out.append((prefix + name, sorted(lines)))

    scan(tree.body, "")            # type: ignore[attr-defined]
    return out


def test_no_test_is_shadowed_by_a_later_one():
    """No name is bound twice in one scope, anywhere in the suite."""
    findings: list[str] = []
    scanned = 0
    for path in sorted(TESTS.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:                       # pragma: no cover
            continue
        scanned += 1
        for name, lines in _shadowed(tree):
            findings.append(
                f"  {path.relative_to(TESTS)}::{name}  "
                f"defined at lines {', '.join(map(str, lines))}")

    # THE SCAN MUST SEE THE SUITE.  A path filter that matched nothing would
    # make the assertion below vacuously green -- the exact defect class this
    # file exists for, one level up.
    assert scanned >= 200, f"only {scanned} test modules parsed -- blind"

    assert not findings, (
        "these names are defined more than once in one scope, so only the "
        "LAST definition exists.  Every earlier one is dead: it never runs, "
        "and nothing says so.\n\n" + "\n".join(findings)
        + "\n\nDelete the duplicate, or rename it if the two were meant to "
          "be different tests.")


def test_the_scan_catches_a_shadowed_definition():
    """The check itself, mutation-tested inline.

    A guard that reports nothing is indistinguishable from a clean suite, so
    it is shown a file that IS dirty and has to say so.
    """
    dirty = ast.parse(
        "def test_a():\n    pass\n\n\ndef test_a():\n    pass\n"
        "class TestB:\n    def test_c(self): pass\n"
        "    def test_c(self): pass\n")
    found = dict(_shadowed(dirty))
    assert "test_a" in found and found["test_a"] == [1, 5], found
    assert "TestB::test_c" in found, found

    clean = ast.parse("def test_a():\n    pass\n\n\ndef test_b():\n    pass\n")
    assert _shadowed(clean) == []
