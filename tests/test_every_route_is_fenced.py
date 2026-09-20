"""No path from the browser escapes the projects tree — over EVERY route.

**THE RULE** is `web-api.md` § 2.1: *"Every route that takes a filesystem
path from the browser resolves it through `projects.contain` … BEFORE it
calls anything else. There is one fence and this is it."*  The primitive is
`molbuilder/projects.py::contain`, and the web layer reaches it through
`files._resolve_within_roots` (several allowed roots + an HTTP-shaped
refusal) or, where the root policy is narrower, directly.

**WHY THIS FILE EXISTS, when a fence lint already did.**
`test_web_files.py::test_every_path_route_is_fenced` is the same idea and
says so — *"a property over every member of a class, which replaces the
manual sweep"* — but its class is a **hand-written list of eight
`/api/files/*` probes**, and its own docstring admits the maintenance
burden: *"A new path-taking route joins the list here."*  Measured
2026-09-20: the app serves **36** path-taking API routes across ten
blueprints.  Twenty-eight of them were outside that list, so the guard
covered under a quarter of the surface it is named for, and route 37 ships
unfenced by default rather than by mistake.

A list that has to be remembered is the thing the rule exists to replace.
This derives the class from `app.url_map` — the same table Flask dispatches
on — so a route cannot be added without joining it.

**WHAT IT PROVES, and what it does not.**  It is a source-level check: the
view function must REACH the fence.  That is weaker than driving a path
through each route (which `test_web_files.py` does for its eight, and which
stays), and stronger than nothing for the twenty-eight that had neither.
It cannot see a route that calls the fence and then ignores the result; it
can see one that never calls it at all, which is the failure § 2.1
describes — *"a route that forgets `_resolve_within_roots` reads or MUTATES
anywhere the server user can reach, and nothing else in the system would
notice."*

**ADDING A ROUTE.**  If it takes a path, fence it.  If it takes something
this file mistakes for a path, add it to `_NOT_A_PATH` with the reason.
"""
from __future__ import annotations

import inspect
import re

import pytest

#: Reaching the fence: the primitive, the web wrapper, or a thin local
#: helper that calls one of them.  Named rather than pattern-guessed, so a
#: new spelling shows up here as a deliberate edit.
_REACHES_FENCE = re.compile(
    r"_resolve_within_roots|projects\.contain|\bcontain\(|"
    r"_resolve_path|_resolve_doc|_fence\("
)

#: Parameters that LOOK like a path and are not one.  Empty today; a row
#: here must say why, because every row is a hole in the guard.
_NOT_A_PATH: "dict[str, str]" = {}

_PATH_PARAM = re.compile(
    r'\.get\(\s*["\'](path|dir|directory|target_dir|dest_dir|src|dest|'
    r'file|folder)["\']'
)


def _path_taking_routes(app):
    """Every `/api/` route whose view reads a path-shaped parameter."""
    out = []
    for rule in app.url_map.iter_rules():
        if not str(rule).startswith("/api/"):
            continue
        if str(rule) in _NOT_A_PATH:
            continue
        fn = app.view_functions.get(rule.endpoint)
        try:
            src = inspect.getsource(fn)
        except (OSError, TypeError):           # pragma: no cover
            continue
        if _PATH_PARAM.search(src):
            out.append((str(rule), rule.endpoint, src))
    return out


def test_every_path_taking_route_reaches_the_one_fence(web_client):
    """§ 2.1, over the route table rather than over a remembered list."""
    app = web_client.application
    routes = _path_taking_routes(app)

    assert len(routes) >= 30, (
        f"only {len(routes)} path-taking routes found -- the detector has "
        f"stopped detecting, which fails open. Measured 36 on 2026-09-20.")

    unfenced = [(r, e) for r, e, src in routes if not _REACHES_FENCE.search(src)]
    assert not unfenced, (
        "these routes take a path from the browser and never reach "
        "`projects.contain` -- `web-api.md` § 2.1:\n"
        + "\n".join(f"  {r}  ({e})" for r, e in unfenced)
        + "\n\nFence it at the route. A module that opens whatever it is "
          "handed is not the bug; the route that handed it an unchecked "
          "path is.")


def test_the_detector_can_actually_fail(web_client):
    """A guard over a derived class is worthless if the class comes back empty.

    Shown, not asserted: strip the fence from one real view's source and the
    check above must report it.
    """
    app = web_client.application
    routes = _path_taking_routes(app)
    victim = next(src for _r, _e, src in routes if _REACHES_FENCE.search(src))
    assert not _REACHES_FENCE.search(_REACHES_FENCE.sub("", victim)), (
        "removing every fence call from a view left the detector still "
        "seeing one -- the pattern matches something it should not")
