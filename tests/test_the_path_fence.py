"""A path from the browser is fenced at the route -- asked of EVERY `/api/`
route the app serves (`web-api.md` § 2.1: *"Every route that takes a
filesystem path from the browser resolves it through `projects.contain` …
BEFORE it calls anything else"*).

THE SILENT FAILURE.  A route that forgets the fence reads or changes any
file the server's user can reach -- a logged-in person once POSTed
`{"path": "/etc/shadow"}` and had the parser read it (2026-06-18) -- and
nothing else in the system notices.

THE CASES are `tests/data/path_fence.toml`: one row per route and method,
either what it takes instead of a path, or every place a path enters with
the whole request that reaches the fence.  The table is the app's route map
(:func:`test_the_table_is_every_api_route`), so a route added without a row
fails by name; then each place a path enters is asked with a path OUTSIDE
the projects tree (:func:`test_a_path_outside_the_projects_tree_is_refused`)
-- refused in the fence's words, and a file beside the tree untouched.

MUTATIONS THIS MUST FAIL AGAINST: a route's path handed on unfenced (Prep's
`from`, describe's `dest`, `files/read`); a relative `psml_lib` joined to
the tree unfenced; a tree-relative name joined to the root unfenced.
"""
from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

_TABLE = Path(__file__).parent / "data" / "path_fence.toml"
_ROUTES = tomllib.loads(_TABLE.read_text())["route"]
_PROBES = [(r, p) for r in _ROUTES for p in r.get("path", [])]


def _served(app):
    """Every `/api/` route and method the app dispatches."""
    return {(str(rule), m) for rule in app.url_map.iter_rules()
            if str(rule).startswith("/api/")
            for m in rule.methods - {"HEAD", "OPTIONS"}}


@pytest.fixture
def tree(tmp_path, monkeypatch):
    """The projects tree the app serves -- with a file, a folder and a
    link to the folder outside it -- and, beside it, a folder outside it
    holding a file."""
    from molbuilder.diagnostics import reset_capabilities
    from molbuilder.projects import PROJECTS_ROOT_ENV
    root = tmp_path / "projects"
    inside = root / "P" / "here"
    inside.mkdir(parents=True)
    (inside / "notes.txt").write_text("inside\n")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "loot.txt").write_text("loot\n")
    (inside / "out").symlink_to(outside, target_is_directory=True)
    monkeypatch.setenv(PROJECTS_ROOT_ENV, str(root))
    reset_capabilities()
    yield {"root": root, "inside": inside, "outside": outside}
    reset_capabilities()


def _client():
    from molbuilder.web.app import create_app
    return create_app(config={"rate_limit": {"enabled": False}}).test_client()


def _filled(value, places):
    """``value`` with every ``{name}`` placeholder of ``places`` filled in."""
    if isinstance(value, str):
        for k, v in places.items():
            value = value.replace("{" + k + "}", v)
        return value
    if isinstance(value, dict):
        return {k: _filled(v, places) for k, v in value.items()}
    if isinstance(value, list):
        return [_filled(v, places) for v in value]
    return value


def _send(web, method, rule, request):
    """One request as the row states it -- a query string, a JSON body, or
    a form with its files."""
    import io
    url = request.get("url", rule)
    kw = {}
    if "query" in request:
        kw["query_string"] = request["query"]
    if "json" in request:
        kw["json"] = request["json"]
    if "form" in request:
        kw["data"] = dict(request["form"], **{
            field: (io.BytesIO(f["content"].encode()), f["filename"])
            for field, f in request.get("files", {}).items()})
        kw["content_type"] = "multipart/form-data"
    return web.open(url, method=method, **kw)


def _state(folder: Path):
    """Every file under ``folder``, by its bytes."""
    return {str(p.relative_to(folder)): p.read_bytes()
            for p in sorted(folder.rglob("*")) if p.is_file()}


def test_the_table_is_every_api_route():
    """The table is the app's route map: no route without a row, no row for
    a route the app does not serve -- but one it registers only on a
    machine set up for it (`only_when`)."""
    served = _served(_client().application)
    tabled = {(r["rule"], r["method"]) for r in _ROUTES}
    conditional = {(r["rule"], r["method"]) for r in _ROUTES
                   if r.get("only_when")}
    assert not served - tabled, (
        "routes with no row in tests/data/path_fence.toml -- say what each "
        "takes, or where it takes a path: "
        + ", ".join(f"{m} {r}" for r, m in sorted(served - tabled)))
    assert not tabled - served - conditional, (
        "rows for routes the app does not serve: "
        + ", ".join(f"{m} {r}" for r, m in sorted(tabled - served
                                                   - conditional)))


@pytest.mark.parametrize(
    "route,probe", _PROBES,
    ids=[f"{r['method']} {r['rule']} {p['where']}:{p['name']}"
         for r, p in _PROBES])
def test_a_path_outside_the_projects_tree_is_refused(tree, route, probe):
    """Each place a path enters, given one outside the projects tree: the
    route refuses it in the fence's words, and nothing beside the tree
    changes."""
    import os
    outside, inside = tree["outside"], tree["inside"]
    docs_img = Path(__file__).resolve().parents[1] / "docs" / "img"
    places = {"outside_file": str(outside / "loot.txt"),
              "outside_dir": str(outside),
              "inside_file": str(inside / "notes.txt"),
              "inside_dir": str(inside),
              "outside_by_climbing": "P/" + os.path.relpath(
                  outside, tree["root"] / "P"),
              "link_out": "out",
              "link_out_from_root": str(
                  (inside / "out").relative_to(tree["root"])),
              "outside_from_docs_img": os.path.relpath(
                  outside / "loot.txt", docs_img)}
    before = _state(outside)
    r = _send(_client(), route["method"], route["rule"],
              _filled(probe["request"], places))
    said = r.get_data(as_text=True)
    if "status" in probe:
        assert r.status_code == probe["status"], (r.status_code, said[:300])
    else:
        assert 400 <= r.status_code < 500, (r.status_code, said[:300])
    assert any(w in said for w in probe.get("refusal", ["is outside"])), \
        said[:300]
    assert _state(outside) == before, "a file beside the tree changed"
