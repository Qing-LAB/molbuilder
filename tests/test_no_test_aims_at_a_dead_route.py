"""**A request to a route that does not exist answers 404, and a test that
never checks the status passes anyway.**

That is not hypothetical.  `test_in_body_labels_xhr.py` posted to
`/api/build/fdf` and `/api/build/pyscf` for months.  Neither route has
existed in that time — the build blueprint offers `molecule`, `load`,
`schema/<engine>` and `preflight` — so every request 404'd, every body
parsed to `{}`, and the file's single assertion sat inside
`if body.get("ok") is True:` and never ran.  Six green tests, nothing
checked, and a class with a docstring claiming "mirror coverage" and no
body at all.  Retired 2026-09-02; the rule it named is checked at
`struct_from_body` in `test_validation_delivery_contract.py`, which is the
door the property actually lives on.

**The guard is not "assert every response".**  A test may legitimately
expect a 404 — `test_admin_reload.py` exists to prove the reload route is
ABSENT until a supervisor and an admin list are both present, so *"404, not
403"* is its subject.  What cannot be legitimate is aiming at a path that no
configuration of the app registers at all: that is a test pointed at
nothing.

So this compares the literal `/api/...` paths in the suite against the union
of the URL maps of every app shape the suite itself builds.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

pytest.importorskip("flask")

TESTS = Path(__file__).resolve().parent
_VERBS = ("post", "get", "put", "delete", "patch")

#: Paths that are UNMAPPED ON PURPOSE, with the reason.
#:
#: Named here rather than inferred, because the difference between this and
#: the defect above is INTENT, and intent is not in the AST.  Each entry is a
#: test whose subject is what happens to a request that reaches no route --
#: so it asserts a status code and cannot pass by reading an empty body.
_DELIBERATELY_UNMAPPED = {
    # A scanner enumerating paths: the signature check blocks it BEFORE
    # routing, so the assertion is `429`, not anything out of the body.
    "/api/nope",
}


def _known_paths() -> list[re.Pattern]:
    """Every route pattern the app can register, over every shape it takes.

    TWO APPS, because some blueprints are CONDITIONAL: the admin routes
    appear only when the process is supervised, and a guard built on the
    default map alone would call them dead.  A shape the suite builds is a
    shape that exists.
    """
    import os

    from molbuilder.reload_protocol import SUPERVISED_ENV
    from molbuilder.web.app import create_app

    maps = [create_app(config={}).url_map]
    was = os.environ.get(SUPERVISED_ENV)
    os.environ[SUPERVISED_ENV] = "1"
    try:
        maps.append(create_app(
            config={"rate_limit": {"enabled": False},
                    "admin": {"emails": []}}).url_map)
    finally:
        if was is None:
            os.environ.pop(SUPERVISED_ENV, None)
        else:
            os.environ[SUPERVISED_ENV] = was

    pats = set()
    for m in maps:
        for rule in m.iter_rules():
            # `<converter:name>` and `<name>` both stand for one segment.
            pats.add(re.sub(r"<[^>]+>", "[^/]+", rule.rule))
    return [re.compile(p) for p in sorted(pats)]


def _api_literals(tree: ast.AST):
    """Every literal `/api/...` first argument to a request verb."""
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        if not (isinstance(f, ast.Attribute) and f.attr in _VERBS):
            continue
        if not node.args:
            continue
        a = node.args[0]
        if isinstance(a, ast.Constant) and isinstance(a.value, str) \
                and a.value.startswith("/api/"):
            yield a.value.split("?")[0], node.lineno


def test_no_test_aims_at_a_route_the_app_never_registers():
    known = _known_paths()
    assert len(known) > 40, f"only {len(known)} routes found -- blind"

    findings, scanned, checked = [], 0, 0
    for path in sorted(TESTS.rglob("*.py")):
        if path.name == Path(__file__).name:
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:                       # pragma: no cover
            continue
        scanned += 1
        for url, line in _api_literals(tree):
            checked += 1
            if url in _DELIBERATELY_UNMAPPED:
                continue
            if not any(p.fullmatch(url) for p in known):
                findings.append(
                    f"  {path.relative_to(TESTS)}:{line}  {url}")

    # THE SCAN MUST SEE THE SUITE, and must actually find URLs in it --
    # a walker that matched no calls would pass having examined nothing,
    # which is the very defect this file exists to catch.
    assert scanned >= 200, f"only {scanned} test modules parsed -- blind"
    assert checked >= 100, f"only {checked} api URLs examined -- blind"

    assert not findings, (
        "these tests send requests to paths no app shape registers.  The "
        "response is a 404 whatever the test intended, so anything the test "
        "reads out of the body is empty and any assertion guarded by `ok` "
        "never runs.\n\n" + "\n".join(findings)
        + "\n\nEither the route moved -- point the test at where the rule "
          "now lives -- or the feature is gone and the test goes with it.  "
          "If the path is unmapped ON PURPOSE (a scanner probe, a 404 that "
          "is the subject), add it to `_DELIBERATELY_UNMAPPED` with the "
          "reason -- and make sure the test asserts a STATUS, or it can "
          "still pass by reading an empty body.")
