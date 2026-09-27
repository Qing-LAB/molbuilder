"""The status a caller RECEIVES: web-api.md § 1, *Status codes*.

An audit on 2026-06-17 (commits ``0880899`` + ``4003edc``)
codified the four-bucket rule, which § 1's *Status codes* now states:

    (a) ok:true   → 2xx (typically 200)
    (b) advisory  → 200  + ok:false  (validator hard-fail)
    (c) protocol  → 4xx  + ok:false  (bad input)
    (d) server    → 5xx  + ok:false  (IO / engine fault)

The audit surfaced five misclassifications -- for example,
``build.py::api_build_molecule``'s catch-all ``except Exception``
returned HTTP 400 but should have been 500.  The bug was easy to
ship because **Flask's default status is 200 when no tuple is
returned**, so a developer who forgot to add ``, 500`` got HTTP
200 silently.

Every test here asks a ROUTE and reads the status it answers.  Until
2026-09-26 two AST lints also read every blueprint's source for an
``ok: False`` return with no explicit status, or one outside the § 1 set.
They were retired with the other source scans (`process/testing.md` § 3a):
an explicit ``, 200`` and Flask's default 200 are the SAME response, so no
request can tell them apart -- that half of the rule is review's
(`process/code-audit.md` § 1c), and each route's own tests assert the
status of the refusals they drive.

The companion test ``test_web.py::TestUniformEnvelope`` pins that every
success path returns ``ok: true``; the ``Endpoint index`` count in
``docs/web/web-api.md`` is checked against Flask's URL map below.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
# The migrated web API contract is the authoritative route-count document.
WEB_API_MD    = REPO_ROOT / "docs" / "web" / "web-api.md"


class TestKnownAdvisorySitesAreHTTP200:
    """Regression pins for the four canonical scientific-advisory
    sites — these were the ones the 1b audit fixed (400 → 200).
    Pinning them here so a future refactor that re-introduces 400
    lands as a test failure with a clear pointer to § 1.6 (b).

    The advisory bucket uses ``, 200`` EXPLICITLY at the call site
    (not Flask's default) so the intent is visible to readers.
    """

    def test_an_uncitable_directory_is_answered_not_refused(
            self, web_client, tmp_path, monkeypatch):
        """The advisory rule, exercised THROUGH THE ROUTE.

        `web-api.md` § 1: a validator hard-fail is answered with HTTP **200**
        and the refusal as the body, not with a 4xx -- the caller asked a fair
        question and the answer is "no, and here is what is missing".

        **This replaced a source-text assertion on 2026-09-17.** The old form
        read a blueprint file and asserted a literal snippet --
        `'"errors_only": _issues_to_json(errors_only, cfg=cfg),\n        }), 200'`
        -- including its indentation. It pinned three sites over its life and
        lost all three to route deletions (build.py's two on 2026-08-17,
        spectra.py's on 2026-08-21, transport.py's render route on
        2026-09-17), each time leaving the rule with one fewer instance and
        the test one edit from asserting nothing at all. A rule about what a
        caller RECEIVES is checked by calling.
        """
        from molbuilder.projects import PROJECTS_ROOT_ENV
        d = tmp_path / "notcitable"
        d.mkdir()
        (d / "readme.txt").write_text("nothing citable here")
        monkeypatch.setenv(PROJECTS_ROOT_ENV, str(tmp_path))

        r = web_client.get("/api/transport/describe_attempt?path=notcitable")
        assert r.status_code == 200, (
            "an uncitable directory is a fair question with a negative answer; "
            "answering it 4xx would make the browser treat a valid reply as a "
            "transport failure")
        body = r.get_json()
        assert body["form"] is None
        assert body["summary"], "the refusal must NAME what is missing"

    # ``test_build_py_has_two_advisory_200_sites`` was deleted 2026-08-17.
    # It counted ``}, 200`` in build.py and required >= 2, naming
    # ``api_build_fdf`` and ``api_build_pyscf`` as the two.  Both routes are
    # gone, so build.py now has ZERO advisory sites and the count it asserted
    # can never be met again -- the test's subject, not its rule, was removed.


class TestKnownServerFaultSitesAreHTTP500:
    """The two sites the 1b audit moved from HTTP 400 / 200 to HTTP 500
    (server-fault bucket), each asked through its route.
    """

    def test_a_builder_that_falls_over_is_answered_500(
            self, web_client, monkeypatch):
        """**A builder that raises is OUR fault, answered 500.**

        `web-api.md` § 1 (d): the request passed its shape checks (a known
        kind, a non-empty input) before the builder ran, so whatever went
        wrong there is on the server.  Answering 400 told the person their
        input was wrong; it was answered 400 until the 1b audit.

        A builder that raises is registered for the route to dispatch to,
        and the route is POSTed.

        MUTATION THIS MUST FAIL AGAINST: the route's ``except Exception``
        answering ``, 400`` (or no status, Flask's 200).
        """
        from molbuilder.web.blueprints import build as build_bp

        def _falls_over(_text):
            raise RuntimeError("the builder fell over")

        monkeypatch.setitem(build_bp._BUILDERS, "falls-over", _falls_over)
        r = web_client.post("/api/build/molecule",
                            json={"kind": "falls-over", "input": "anything"})
        body = r.get_json()
        assert r.status_code == 500, (
            f"a builder that raised was answered {r.status_code}, which "
            f"blames the input for a server fault: {body}")
        assert body["ok"] is False
        assert "the builder fell over" in body["error"], body

    def test_a_trajectory_that_stops_parsing_is_answered_500(
            self, web_client, isolated_projects_root, monkeypatch):
        """**A poll of a file that no longer parses is a server fault, 500.**

        `web-api.md` § 1 (d); the sibling ``/api/watch/load`` answers the same
        failure class 500, and until the 1b audit this poll answered it 200 --
        so anything gating on the status (curl, monitoring) saw a healthy
        reply carrying an error.

        A small SIESTA-like ``.out`` is loaded through ``/api/watch/load``;
        then the file moves on (its mtime advances) and its parser fails,
        and ``/api/watch/data`` is polled.

        MUTATION THIS MUST FAIL AGAINST: the poll's parse-error return
        answering ``, 200`` (or no status).
        """
        from molbuilder.web.blueprints import watch

        # The route keeps ONE loaded file in module state; this test gets its
        # own, and the suite's is back when it ends.
        monkeypatch.setattr(watch, "_state", dict(
            watch._state, path=None, mtime=None, data=None, parser=None,
            uploaded=False, run_dir=None))

        out = isolated_projects_root / "run" / "run.out"
        out.parent.mkdir(parents=True)
        out.write_text(
            "Welcome to SIESTA -- v4.1\n"
            "redata: prelude\n"
            "outcoor: Atomic coordinates (Ang):\n"
            "   1.00000000    2.00000000    3.00000000   1       1  C\n"
            "\n"
            "siesta: E_KS(eV) =          -50.0000\n")
        r = web_client.post("/api/watch/load", json={"path": str(out)})
        assert r.status_code == 200 and r.get_json()["ok"], r.get_json()

        def _no_longer_parses(_path):
            raise ValueError("the file stopped making sense")

        monkeypatch.setattr(watch._state["parser"], "parse",
                            staticmethod(_no_longer_parses))
        st = out.stat()
        os.utime(out, (st.st_atime, st.st_mtime + 10))

        r = web_client.get("/api/watch/data")
        body = r.get_json()
        assert r.status_code == 500, (
            f"a poll whose file no longer parses was answered "
            f"{r.status_code}: {body}")
        assert body["ok"] is False
        assert body["error"].startswith("Parse error"), body


class TestRouteCountDocMatchesReality:
    """The ``## 3. Endpoint index — all NN routes`` header in
    web-api.md must reflect what Flask's URL map actually has —
    otherwise the doc silently goes stale every time a route is
    added or removed.

    2026-08-07: the heading moved from ``## 2.`` to ``## 3.``.  The doc had
    TWO sections numbered 2 — "Security posture" was inserted without
    shifting anything below it — and three disagreeing route counts: the
    heading said 79, the paragraph under it said 78, and the app had 80.
    The regex now anchors on the heading text rather than its number, so a
    future renumber does not break the test that catches the count.
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
            r"##\s+\d+\.\s+Endpoint index\s+—\s+all\s+(\d+)\s+routes",
            text,
        )
        assert m is not None, (
            "could not locate the 'Endpoint index — all NN routes' header "
            "in docs/web/web-api.md.  Did the heading rename?  Update "
            "this test's regex to match."
        )
        doc_count = int(m.group(1))
        assert doc_count == actual, (
            f"docs/web/web-api.md says {doc_count} routes; "
            f"Flask's URL map has {actual}.  Update the heading to "
            f"'## 3. Endpoint index — all {actual} routes' and verify "
            f"any new routes are documented in the section bodies."
        )
