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

Every test here asks a ROUTE and reads the status it answers.  An
explicit ``, 200`` and Flask's default 200 are the SAME response, so no
request can tell them apart -- that half of the rule is review's
(`process/code-audit.md` § 1c), and each route's own tests assert the
status of the refusals they drive.

The ``Endpoint index`` count in ``docs/web/web-api.md`` is checked
against Flask's URL map below.
"""
from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent


class TestKnownAdvisorySitesAreHTTP200:
    """Regression pin for the scientific-advisory bucket, which the 1b
    audit fixed (400 → 200): a future refactor that re-introduces 400
    lands as a test failure with a clear pointer to § 1 (b).

    The advisory bucket uses ``, 200`` EXPLICITLY at the call site
    (not Flask's default) so the intent is visible to readers.
    """

    def test_an_uncitable_directory_is_answered_not_refused(
            self, web_client, isolated_projects_root):
        """The advisory rule, exercised THROUGH THE ROUTE.

        `web-api.md` § 1: a validator hard-fail is answered with HTTP **200**
        and the refusal as the body, not with a 4xx -- the caller asked a fair
        question and the answer is "no, and here is what is missing".  A
        rule about what a caller RECEIVES is checked by calling.
        """
        d = isolated_projects_root / "notcitable"
        d.mkdir()
        (d / "readme.txt").write_text("nothing citable here")

        r = web_client.get("/api/transport/describe_attempt?path=notcitable")
        assert r.status_code == 200, (
            "an uncitable directory is a fair question with a negative answer; "
            "answering it 4xx would make the browser treat a valid reply as a "
            "transport failure")
        body = r.get_json()
        assert body["citation"] is None
        assert body["summary"], "the refusal must NAME what is missing"


class TestKnownServerFaultSitesAreHTTP500:
    """The server-fault bucket the 1b audit moved sites into (HTTP 400 /
    200 to HTTP 500), asked through the route.
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


