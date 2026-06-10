"""Meta-pin: every ``assert <substring> not in body`` (or similar
HTTP-response body variable) must be preceded in the same test
function by an ``assert response.status_code == 2XX`` so a route
that 404s can't silently make the negative-substring assertion
pass trivially against the Flask error page.

Why this test exists
====================

The 2026-06-10 post-B.5 sweep surfaced two such dead pins that
had been silently passing for weeks:

* ``tests/test_web_files.py::test_file_input_not_emitted`` was
  parametrized over ``/spectra`` and ``/modify`` — both retired in
  Phase B.5.  The test asserted ``id="xyz-file" not in body`` etc.
  The retired routes return 404; the Flask error page doesn't
  contain ``id="xyz-file"``; the assertion passed; nothing was
  pinned.  The test was retired wholesale.

* ``tests/spectra/test_blueprint.py::test_inspector_partial_has_mode_viewer``
  asserted ``vendor/3Dmol-min.js not in body`` against ``/spectra``
  (also retired).  Same shape, same silent pass.  Fixed to point at
  ``/spectrum-calculation``; the assertion then **correctly** failed
  because task #296 added 3Dmol there (the inspect-structure card).
  The 404-pass had been hiding genuine drift.

The pattern is dangerous wherever a test takes the shape:

    r = web.get(path)
    body = r.get_data(as_text=True)
    assert "<some-id>" not in body

without a guarding ``assert r.status_code == 200`` between the
fetch and the body check.  This lint test walks every test file's
AST and surfaces any function that combines HTTP requests + a
negative body assertion without a status-code guard.

False-positive control
======================

We only fire on functions that actually make HTTP requests (heuristic:
the function body contains a call like ``<x>.get(...)`` or
``<x>.post(...)``).  Tests over pure-Python helpers don't trip.

If a finding is a true false-positive (rare), add an entry to
``ALLOWLIST`` with a short reason.
"""
from __future__ import annotations

import ast
from pathlib import Path
from typing import Iterable, List, Tuple

TESTS_ROOT = Path(__file__).resolve().parent

# Variable names commonly used to hold an HTTP response body in the
# molbuilder test suite.  Keep tight — too broad and we light up
# every test that defines a local ``text`` or ``content``.
RESPONSE_BODY_NAMES = {
    "body", "html", "markup", "page_html", "page_source", "content",
    "page_body", "rendered",
}

# HTTP-request method names.  Anything called on a client-shaped
# object: ``web.get``, ``web_client.post``, ``client.put``,
# ``page.goto`` is intentionally excluded — that's the Playwright
# path which has its own response-state contract.
HTTP_METHODS = {"get", "post", "put", "delete", "patch"}

# Explicit false-positive allowlist.  Format:
#   ("relative/path.py", "test_function_name"): "reason"
# Use sparingly — usually a missing status check is a real bug.
ALLOWLIST: dict[Tuple[str, str], str] = {
    # (none at time of writing)
}


def _function_makes_http_request(node: ast.AST) -> bool:
    """True if the function body contains a call shaped like
    ``<client>.get(...)`` / ``.post(...)`` / etc. — heuristic for
    'this test drives HTTP'.
    """
    for n in ast.walk(node):
        if not isinstance(n, ast.Call):
            continue
        f = n.func
        if isinstance(f, ast.Attribute) and f.attr in HTTP_METHODS:
            return True
    return False


def _is_status_code_assert(node: ast.AST) -> bool:
    """True if the node is ``assert <expr>.status_code <op> <value>``
    or ``assert <expr>.status_code in (...)`` or similar — anything
    that constrains a status_code field.
    """
    if not isinstance(node, ast.Assert):
        return False
    t = node.test
    # Walk into the comparison; we accept any shape that mentions
    # ``.status_code`` on the LHS — Compare (==, in, <, etc.) or
    # BoolOp (and/or chain), so this is lenient.
    for child in ast.walk(t):
        if isinstance(child, ast.Attribute) and child.attr == "status_code":
            return True
    return False


def _is_negative_body_assert(
    node: ast.AST, var_names: Iterable[str]
) -> bool:
    """True if the node is ``assert <expr> not in <var>`` where
    ``<var>`` matches one of the response-body names.
    """
    if not isinstance(node, ast.Assert):
        return False
    t = node.test
    if not isinstance(t, ast.Compare):
        return False
    if not t.ops or not isinstance(t.ops[0], ast.NotIn):
        return False
    if not t.comparators:
        return False
    rhs = t.comparators[0]
    # ``foo not in body`` — body is an ast.Name.
    # ``foo not in body.lower()`` — body is the Call's receiver.
    if isinstance(rhs, ast.Name):
        return rhs.id in var_names
    if isinstance(rhs, ast.Call):
        # ``foo not in body.lower()`` — receiver is rhs.func.value
        if isinstance(rhs.func, ast.Attribute):
            recv = rhs.func.value
            if isinstance(recv, ast.Name) and recv.id in var_names:
                return True
    return False


def _walk_function_defs(tree: ast.AST):
    """Yield every (sync + async) function def in the tree."""
    for n in ast.walk(tree):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            yield n


def _find_offenders(file_path: Path) -> List[Tuple[str, int, str]]:
    """Return list of (file_rel_name, line, function_qualname) for
    every function in ``file_path`` that fires the negative-body
    pattern without a status-code guard.
    """
    try:
        tree = ast.parse(file_path.read_text(encoding="utf-8"))
    except (SyntaxError, UnicodeDecodeError):
        return []
    rel = str(file_path.relative_to(TESTS_ROOT))
    found: List[Tuple[str, int, str]] = []
    for func in _walk_function_defs(tree):
        if not _function_makes_http_request(func):
            continue
        # Collect all asserts in source order so we can ask
        # "is there a status_code assert anywhere earlier in this
        # function?"  Ordering by lineno is sufficient — Python's
        # AST preserves source order on .lineno.
        asserts = sorted(
            (n for n in ast.walk(func) if isinstance(n, ast.Assert)),
            key=lambda n: (n.lineno, n.col_offset),
        )
        has_status_check = False
        for a in asserts:
            if _is_status_code_assert(a):
                has_status_check = True
                continue
            if _is_negative_body_assert(a, RESPONSE_BODY_NAMES):
                if has_status_check:
                    continue
                found.append((rel, a.lineno, func.name))
    return found


def test_negative_body_asserts_have_preceding_status_check():
    """Every ``assert <X> not in body`` (or similar) in a function
    that makes HTTP requests must be preceded by an
    ``assert response.status_code == 2XX``.

    Without that guard, a route that returns 404 silently makes the
    negative-substring assertion pass against the Flask error page,
    pinning nothing.  Two cases like this lived in the suite for
    weeks (test_file_input_not_emitted, test_inspector_partial_has_mode_viewer)
    before the 2026-06-10 post-B.5 sweep retired them.
    """
    test_files = sorted(TESTS_ROOT.rglob("test_*.py"))
    # Filter out this file itself + non-tests like conftest.
    test_files = [
        f for f in test_files
        if f.is_file() and f.name != Path(__file__).name
    ]

    offenders: List[Tuple[str, int, str]] = []
    for f in test_files:
        offenders.extend(_find_offenders(f))

    # Filter the allowlist.
    real_offenders = [
        (rel, line, func) for (rel, line, func) in offenders
        if (rel, func) not in ALLOWLIST
    ]

    assert not real_offenders, (
        "Found assert-not-in-body patterns missing a preceding "
        "status_code guard.  Without the guard, a route that 404s "
        "makes the assertion pass against the Flask error page, "
        "pinning nothing.\n\n"
        "Add ``assert r.status_code == 200`` (or the expected "
        "code) between the fetch and the body check.\n\n"
        + "\n".join(
            f"  - tests/{rel}:{line}  in {func}"
            for (rel, line, func) in real_offenders
        )
    )
