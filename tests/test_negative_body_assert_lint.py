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
    # Body-validation helper, NOT an HTTP test: every `.get(...)` here is
    # `body.get("key")` (dict access on the response body passed in as an
    # argument), not `client.get("/path")`.  The helper has no response /
    # status_code in scope; its callers (test_envelope_bucket_b_*) assert
    # `r.status_code == 200` BEFORE passing the body in, so the negative
    # `"errors" not in body` assertion is guarded at the call site.  This is
    # exactly the dict.get false positive the heuristic docstring anticipates.
    ("test_checkpoint_routes.py", "_assert_bucket_b"):
        "body-validation helper; .get() is dict.get on the body, not an HTTP "
        "call; callers guard status_code == 200 before calling it",
}


def _function_makes_http_request(node: ast.AST) -> bool:
    """True if the function body contains a call shaped like
    ``<client>.get(...)`` / ``.post(...)`` / etc. — heuristic for
    'this test drives HTTP'.

    Caveat: ``dict.get("key")`` matches this shape too.  In practice
    the combination of ``dict.get`` AND ``assert <X> not in body``
    (where body is from a fixture file, not an HTTP response) is rare
    in this codebase; none observed at lint-introduction time.  If a
    false positive surfaces, add the function to ``ALLOWLIST``.
    """
    for n in ast.walk(node):
        if not isinstance(n, ast.Call):
            continue
        f = n.func
        if isinstance(f, ast.Attribute) and f.attr in HTTP_METHODS:
            return True
    return False


def _is_status_code_assert(node: ast.AST) -> bool:
    """True if the node is an ``assert <X>.status_code <op> <Y>``
    comparison.

    Requires a ``Compare`` expression with ``.status_code`` on one
    side.  Rejects bare truthiness checks like ``assert r.status_code``:
    every non-zero status code is truthy (404 included), so a bare
    truthy assert is no guard at all — it would silence the lint
    without actually preventing the silent-pass that gave rise to
    this whole test in the first place.

    Lenient about WHICH comparison is used: ``== 200``,
    ``in (200, 201)``, ``< 400``, ``!= 302`` all count.  ``!= 500``
    technically wouldn't rule out a 404 silent-pass; we accept that
    edge for now because tightening would block legitimate non-200
    guards (e.g. the ``!= 302`` "no auth redirect" pattern in
    ``test_results_blueprint.py``).  If this becomes a real source
    of bugs, tighten by requiring the comparator to be a 2xx
    constant.
    """
    if not isinstance(node, ast.Assert):
        return False
    t = node.test
    if not isinstance(t, ast.Compare):
        return False
    candidates = [t.left] + list(t.comparators)
    return any(
        isinstance(c, ast.Attribute) and c.attr == "status_code"
        for c in candidates
    )


def _collect_function_local_asserts(func: ast.AST) -> List[ast.Assert]:
    """Return every ``ast.Assert`` declared in ``func``'s own scope.

    Crucially does NOT descend into nested function defs / async
    function defs / lambdas — those introduce their own scope, so an
    ``assert r.status_code == 200`` inside a nested helper does NOT
    guard the outer function's body checks at runtime.  ``ast.walk``
    would conflate them and the lint would silently green over the
    outer's missing guard.

    Comprehensions (ListComp / DictComp / etc.) also create their
    own scope but rarely contain asserts — we descend into them
    anyway for simplicity; the practical risk of an
    ``assert`` inside a comprehension being misclassified is zero.
    """
    out: List[ast.Assert] = []

    def _visit(node: ast.AST, is_root: bool) -> None:
        if not is_root and isinstance(
            node,
            (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda),
        ):
            return
        if isinstance(node, ast.Assert):
            out.append(node)
        for child in ast.iter_child_nodes(node):
            _visit(child, is_root=False)

    _visit(func, is_root=True)
    return out


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


def _find_offenders_in_tree(
    tree: ast.AST, rel: str,
) -> List[Tuple[str, int, str]]:
    """Apply the lint to a parsed AST.

    Split out from :func:`_find_offenders` so the self-test below
    can feed in code strings without round-tripping through a temp
    file.  Returns ``(rel, line, function_name)`` for each function
    that has a negative-body assert in its own scope without a
    preceding status-code guard in the same scope.
    """
    found: List[Tuple[str, int, str]] = []
    for func in _walk_function_defs(tree):
        if not _function_makes_http_request(func):
            continue
        # Function-local asserts only — see _collect_function_local_asserts
        # for why we don't use ast.walk here.  Sort by source
        # position so "is there a status check EARLIER than this
        # body check?" is just an O(n) sweep.
        asserts = sorted(
            _collect_function_local_asserts(func),
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


def _find_offenders(file_path: Path) -> List[Tuple[str, int, str]]:
    """Return list of (file_rel_name, line, function_name) for every
    function in ``file_path`` that fires the negative-body pattern
    without a status-code guard.
    """
    try:
        tree = ast.parse(file_path.read_text(encoding="utf-8"))
    except (SyntaxError, UnicodeDecodeError):
        return []
    rel = str(file_path.relative_to(TESTS_ROOT))
    return _find_offenders_in_tree(tree, rel)


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


# --------------------------------------------------------------------- #
#  Self-tests — pin the lint's own behaviour so a future refactor       #
#  that silently neuters detection (e.g. by widening the status-check   #
#  matcher too far, or breaking scope-locality) surfaces here rather    #
#  than going green by virtue of finding zero offenders.                #
# --------------------------------------------------------------------- #


def _lint(code: str) -> List[Tuple[str, int, str]]:
    """Convenience wrapper for the self-tests.  Returns the offender
    list (filename "<fixture>") for the given code string."""
    return _find_offenders_in_tree(ast.parse(code), "<fixture>")


def test_lint_flags_unguarded_body_check():
    """Canonical failure shape: HTTP GET + negative body assert + no
    status guard.  Lint MUST surface it."""
    code = """\
def test_bad(web):
    body = web.get("/some-route").get_data(as_text=True)
    assert "needle" not in body
"""
    offenders = _lint(code)
    assert len(offenders) == 1, offenders
    assert offenders[0][2] == "test_bad"


def test_lint_passes_when_status_guard_present():
    """The textbook fix: assert status_code == 200 between fetch and
    body check.  Lint MUST NOT flag."""
    code = """\
def test_good(web):
    r = web.get("/some-route")
    assert r.status_code == 200
    body = r.get_data(as_text=True)
    assert "needle" not in body
"""
    assert _lint(code) == []


def test_lint_rejects_bare_truthy_status_check():
    """``assert r.status_code`` is a truthy check — every non-zero
    status is truthy, 404 included.  The lint MUST treat it as no
    guard and still flag the body check.  This pins the
    ``_is_status_code_assert`` tightening (require Compare, not just
    any mention of .status_code)."""
    code = """\
def test_bad_truthy(web):
    r = web.get("/some-route")
    assert r.status_code            # truthy — no real guard
    body = r.get_data(as_text=True)
    assert "needle" not in body
"""
    offenders = _lint(code)
    assert len(offenders) == 1, (
        f"bare truthy ``assert r.status_code`` was accepted as a "
        f"guard; should have been rejected.  Offenders: {offenders}"
    )


def test_lint_does_not_count_nested_function_guard():
    """A nested helper's ``assert r.status_code == 200`` lives in a
    different scope at runtime — it doesn't guard the outer
    function's body check.  The lint MUST treat the outer function
    as unguarded.  This pins the scope-aware traversal
    (``_collect_function_local_asserts`` vs. naive ``ast.walk``)."""
    code = """\
def test_nested(web):
    def _inner_helper(r):
        assert r.status_code == 200   # nested scope; outer is NOT guarded
    body = web.get("/some-route").get_data(as_text=True)
    assert "needle" not in body
"""
    offenders = _lint(code)
    # The outer function MUST be flagged (no status guard in its
    # own scope).  The inner helper makes no HTTP request so isn't
    # processed at all.
    assert any(o[2] == "test_nested" for o in offenders), (
        f"outer test_nested was not flagged despite having no "
        f"status guard in its own scope; nested helper's guard "
        f"was wrongly counted.  Offenders: {offenders}"
    )


def test_lint_accepts_status_in_membership_check():
    """``assert r.status_code in (200, 201)`` is a legitimate guard
    shape (POST endpoints often return 201).  Lint MUST accept it."""
    code = """\
def test_ok_in(web):
    r = web.post("/some-route", json={})
    assert r.status_code in (200, 201)
    body = r.get_data(as_text=True)
    assert "needle" not in body
"""
    assert _lint(code) == []


def test_lint_skips_function_without_http_request():
    """A function that only inspects a fixture file (no HTTP request)
    must not be flagged even if it contains the negative-body
    pattern.  This pins the ``_function_makes_http_request`` gate
    that keeps unit tests of pure-Python helpers out of scope."""
    code = """\
def test_helper_only(tmp_path):
    body = tmp_path.joinpath("fixture.html").read_text()
    assert "needle" not in body
"""
    assert _lint(code) == []
