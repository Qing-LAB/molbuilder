"""Static-analysis test: every ``return jsonify({"ok": False, ...})``
site in the blueprints carries an explicit HTTP status code in the
documented set per ``docs/protocols/web-api.md`` § 1.6.

Why this exists
===============

The § 1.6 audit (2026-06-17, commit ``0880899`` + ``4003edc``)
codified the four-bucket rule:

    (a) ok:true   → 2xx (typically 200)
    (b) advisory  → 200  + ok:false  (validator hard-fail)
    (c) protocol  → 4xx  + ok:false  (bad input)
    (d) server    → 5xx  + ok:false  (IO / engine fault)

The audit surfaced five misclassifications — for example,
``build.py::api_build_molecule``'s catch-all ``except Exception``
returned HTTP 400 but should have been 500.  The bug was easy to
ship because **Flask's default status is 200 when no tuple is
returned**, so a developer who forgot to add ``, 500`` got HTTP
200 silently.

This test mechanically gates against that drift class.  It does
NOT assert which bucket a given site belongs to (that requires
human judgment about whether the error is advisory vs server
fault); it asserts:

* Every ``ok:false`` return site has an EXPLICIT status — no
  silent Flask default.
* That status is in the documented set.
* The "47 routes" / "59 routes" header in
  ``docs/protocols/web-api.md`` § 2 matches what Flask's URL map
  actually has, so the doc never goes stale.

The companion test
``test_web.py::TestUniformEnvelope`` already pins that every
success path returns ``ok: true``; together the two tests pin both
sides of the envelope contract.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
BLUEPRINT_DIR = REPO_ROOT / "molbuilder" / "web" / "blueprints"
RATE_LIMIT_PY = REPO_ROOT / "molbuilder" / "web" / "rate_limit.py"
SHARED_PY     = REPO_ROOT / "molbuilder" / "web" / "blueprints" / "_shared.py"
# docs migration (docs/MIGRATION.md): web-api.md lives in the frozen
# old_docs/ tree until its reconcile-move to docs/web/ -- repoint then.
WEB_API_MD    = REPO_ROOT / "old_docs" / "protocols" / "web-api.md"


# Allowed HTTP status codes for {ok: false} responses, per § 1.6
# of web-api.md.  200 is allowed for the scientific-advisory class
# (b); the protocol-error class (c) covers 4xx; the server-fault
# class (d) covers 5xx.  Anything outside this set is either
# wrong-bucket or a typo.
ALLOWED_STATUS_CODES = frozenset({
    200,        # advisory class
    400,        # bad request
    401,        # unauthorised (auth gate)
    403,        # forbidden
    404,        # not found (e.g. unknown engine slug)
    409,        # conflict
    413,        # payload too large (file upload over MAX_CONTENT_LENGTH)
    422,        # unprocessable (alt to 400)
    429,        # rate limited
    500,        # internal server error
    501,        # not implemented (TranSIESTA registry path)
    502,        # bad gateway
    503,        # service unavailable
})


# --------------------------------------------------------------------- #
#  AST helpers                                                          #
# --------------------------------------------------------------------- #


def _is_ok_false_jsonify(call: ast.AST) -> bool:
    """True iff ``call`` is ``jsonify({"ok": False, ...})``."""
    if not isinstance(call, ast.Call):
        return False
    # Match both ``jsonify(...)`` and ``flask.jsonify(...)``.
    func = call.func
    if isinstance(func, ast.Name):
        if func.id != "jsonify":
            return False
    elif isinstance(func, ast.Attribute):
        if func.attr != "jsonify":
            return False
    else:
        return False
    if not call.args or not isinstance(call.args[0], ast.Dict):
        return False
    d = call.args[0]
    for k, v in zip(d.keys, d.values):
        if isinstance(k, ast.Constant) and k.value == "ok":
            if isinstance(v, ast.Constant) and v.value is False:
                return True
            return False
    return False


def _extract_status(return_value: ast.AST) -> Tuple[Optional[int], Optional[str]]:
    """For a ``Return``'s value, find the HTTP status code.

    Returns ``(int_status, None)`` for ``return jsonify(...), 400``,
    ``(None, "exc.status")`` for ``return jsonify(...), exc.status``
    (variable; assumed valid since the helper attributes are
    validated upstream), or ``(None, None)`` for
    ``return jsonify(...)`` with no tuple (= Flask default 200,
    which is the bug class this test catches when the body has
    ok:false).
    """
    if isinstance(return_value, ast.Tuple) and len(return_value.elts) == 2:
        status_node = return_value.elts[1]
        if isinstance(status_node, ast.Constant) and isinstance(status_node.value, int):
            return (status_node.value, None)
        if isinstance(status_node, ast.Attribute):
            # ``exc.status`` etc. — passes the test (assumed
            # validated by the exception type that defines it).
            return (None, ast.unparse(status_node))
        if isinstance(status_node, ast.Name):
            return (None, status_node.id)
    return (None, None)


def _iter_ok_false_returns(
        py_path: Path,
) -> Iterator[Tuple[int, Optional[int], Optional[str]]]:
    """Yield ``(line_no, int_status, var_name)`` for every
    ``return jsonify({"ok": False, ...})`` in ``py_path``.

    * ``int_status`` populated when the status is a literal int.
    * ``var_name`` populated when the status is a variable
      (e.g. ``exc.status``) — treated as opaque-but-valid.
    * Both ``None`` means no explicit status — the bug class.
    """
    tree = ast.parse(py_path.read_text())
    for node in ast.walk(tree):
        if not isinstance(node, ast.Return) or node.value is None:
            continue
        # The return value is either a direct jsonify call OR a
        # tuple ``(jsonify(...), status)``.
        rv = node.value
        if isinstance(rv, ast.Tuple) and rv.elts:
            inner = rv.elts[0]
        else:
            inner = rv
        if not _is_ok_false_jsonify(inner):
            continue
        int_status, var_name = _extract_status(rv)
        yield (node.lineno, int_status, var_name)


# --------------------------------------------------------------------- #
#  Tests                                                                #
# --------------------------------------------------------------------- #


# Files that participate in the ok:false contract.  Blueprints
# obviously do; ``rate_limit.py`` also (its admin endpoints can
# return ok:false).  ``_shared.py`` carries the ``err()`` helper
# but parses fine here.
SCANNED_FILES: List[Path] = sorted(BLUEPRINT_DIR.glob("*.py")) + [RATE_LIMIT_PY]


class TestExplicitStatus:
    """Every ``ok:false`` jsonify return must have an explicit
    status code; Flask's default 200 is the documented bug class.
    """

    @pytest.mark.parametrize("py_path", SCANNED_FILES, ids=lambda p: p.name)
    def test_every_ok_false_has_explicit_status(self, py_path):
        violations = []
        for line_no, int_status, var_name in _iter_ok_false_returns(py_path):
            if int_status is None and var_name is None:
                violations.append(
                    f"{py_path.name}:{line_no} — return jsonify({{'ok': False, "
                    f"...}}) has no explicit status code; Flask default 200 "
                    f"is wrong unless this is the scientific-advisory bucket "
                    f"(in which case state ``, 200`` explicitly so the intent "
                    f"is visible).  See web-api.md § 1.6."
                )
        assert not violations, "\n".join(violations)


class TestStatusInAllowedSet:
    """Every explicit status on an ok:false return must be in the
    documented § 1.6 set.  Out-of-set codes are either typos or
    semantically wrong choices.
    """

    @pytest.mark.parametrize("py_path", SCANNED_FILES, ids=lambda p: p.name)
    def test_status_codes_are_in_documented_set(self, py_path):
        violations = []
        for line_no, int_status, _var in _iter_ok_false_returns(py_path):
            if int_status is None:
                continue   # variable status — opaquely accepted
            if int_status not in ALLOWED_STATUS_CODES:
                violations.append(
                    f"{py_path.name}:{line_no} — status {int_status} is not "
                    f"in the web-api.md § 1.6 set "
                    f"{sorted(ALLOWED_STATUS_CODES)}.  Either pick a "
                    f"semantically-correct status from the set, or update "
                    f"§ 1.6 + ALLOWED_STATUS_CODES here to add it."
                )
        assert not violations, "\n".join(violations)


class TestKnownAdvisorySitesAreHTTP200:
    """Regression pins for the four canonical scientific-advisory
    sites — these were the ones the 1b audit fixed (400 → 200).
    Pinning them here so a future refactor that re-introduces 400
    lands as a test failure with a clear pointer to § 1.6 (b).

    The advisory bucket uses ``, 200`` EXPLICITLY at the call site
    (not Flask's default) so the intent is visible to readers and
    the ``TestExplicitStatus`` invariant catches future "I forgot
    the status code" drift uniformly.
    """

    @pytest.mark.parametrize("path,needle", [
        # build.py::api_build_fdf has the ValidationError catch
        # returning the merged issue list at HTTP 200.
        (BLUEPRINT_DIR / "build.py",     '"issues": merged_issues,\n        }), 200'),
        # spectra.py::api_spectra_render preflight failure.
        (BLUEPRINT_DIR / "spectra.py",
         '"errors_only": _issues_to_json(errors_only, cfg=cfg),\n        }), 200'),
        # transport.py::api_transport_render preflight failure.
        (BLUEPRINT_DIR / "transport.py",
         '"errors_only": _issues_to_json(errors_only, cfg=cfg),\n        }), 200'),
    ])
    def test_advisory_site_returns_explicit_200(self, path, needle):
        src = path.read_text()
        assert needle in src, (
            f"{path.name}: expected explicit ``, 200`` after the advisory "
            f"jsonify body.  Snippet searched: {needle!r}.  Per web-api.md "
            f"§ 1.6 (b), validator-hard-fail returns HTTP 200 with "
            f"ok:false EXPLICITLY so the intent is visible at the call "
            f"site."
        )

    def test_build_py_has_two_advisory_200_sites(self):
        # build.py carries TWO advisory sites: api_build_fdf and
        # api_build_pyscf, both catching ValidationError.  Count
        # the literal ``, 200`` occurrences after a jsonify body
        # to make sure neither one regresses to a 400.
        src = (BLUEPRINT_DIR / "build.py").read_text()
        # Count tuples of (jsonify-close, 200).  The strict
        # whitespace pattern ``}), 200`` distinguishes intentional
        # advisory 200s from accidental matches.
        n = src.count("}), 200")
        assert n >= 2, (
            f"build.py expected >= 2 advisory ``}}, 200`` sites "
            f"(api_build_fdf + api_build_pyscf ValidationError catches); "
            f"found {n}.  Both sites are pinned in § 1.6 (b)."
        )


class TestKnownServerFaultSitesAreHTTP500:
    """Regression pins for the two sites the 1b audit moved from
    HTTP 400 / 200 to HTTP 500 (server-fault bucket).
    """

    def test_build_molecule_generic_exception_is_500(self):
        # build.py::api_build_molecule has a catch-all
        # ``except Exception`` that used to return 400; § 1.6 (d)
        # makes it 500.  The exact source line shifts with edits;
        # just assert the pattern fires somewhere with a 500.
        src = (BLUEPRINT_DIR / "build.py").read_text()
        # Find every ``except Exception as exc:`` followed by a
        # ``return jsonify(...), N`` block within ~10 lines.
        # At least one MUST end in ``, 500``.
        pat = re.compile(
            r'except Exception as exc:[\s\S]{0,200}?'
            r'return jsonify\([\s\S]{0,200}?\)\s*,\s*(\d{3})',
            re.MULTILINE,
        )
        found = pat.findall(src)
        assert any(s == "500" for s in found), (
            f"build.py has no `except Exception → return jsonify(...), 500` "
            f"site; § 1.6 (d) classifies an unhandled builder exception as "
            f"server fault.  Found statuses: {found!r}"
        )

    def test_watch_data_parse_error_is_500(self):
        # watch.py::api_data returns 500 on parse error per § 1.6
        # (d).  Pinned via the exact body string.
        src = (BLUEPRINT_DIR / "watch.py").read_text()
        assert re.search(
            r'state,\s*err\s*=\s*_refresh_if_changed\(\)[\s\S]{0,400}?'
            r'return jsonify\({"ok":\s*False,\s*"error":\s*err}\)\s*,\s*500',
            src,
        ), (
            "watch.py::api_data: the parse-error return must be 500 per "
            "web-api.md § 1.6 (d).  Sibling /api/watch/load returns 500 "
            "on the same failure class; aligning the two closes the gap "
            "that motivated § 1.6 itself."
        )


class TestRouteCountDocMatchesReality:
    """The ``## 2. Endpoint index — all NN routes`` header in
    web-api.md must reflect what Flask's URL map actually has —
    otherwise the doc silently goes stale every time a route is
    added or removed.
    """

    def test_route_count_in_section_2_header_matches_app(self):
        pytest.importorskip("flask")
        from molbuilder.web.app import create_app
        app = create_app(config={"rate_limit": {"enabled": False}})
        # ``static`` is Flask's auto-generated ``/static/<path>`` rule;
        # we exclude it because web-api.md only documents API +
        # page routes (not the static-file serving rule).
        actual = sum(
            1 for r in app.url_map.iter_rules() if r.endpoint != "static"
        )
        text = WEB_API_MD.read_text()
        m = re.search(
            r"## 2\.\s+Endpoint index\s+—\s+all\s+(\d+)\s+routes",
            text,
        )
        assert m is not None, (
            "could not locate '## 2. Endpoint index — all NN routes' header "
            "in docs/protocols/web-api.md.  Did the heading rename?  Update "
            "this test's regex to match."
        )
        doc_count = int(m.group(1))
        assert doc_count == actual, (
            f"docs/protocols/web-api.md § 2 says {doc_count} routes; "
            f"Flask's URL map has {actual}.  Update the heading to "
            f"'## 2. Endpoint index — all {actual} routes' and verify "
            f"any new routes are documented in the section bodies."
        )
