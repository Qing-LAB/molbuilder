"""The one reader of a server's answer -- ``lib/server.js`` (`web/ui-contract.md`
§ 12).

Every question a page asks goes through it, so its reading is every page's.
Each row is one kind of answer, named for its failure: a refusal painted as
data, an error page called a network error, a cancelled question said as a
failure, a person signed out never told.

EVERY ANSWER IS REAL (`process/testing.md` § 3b): the app's own routes
(`_node_esm.served`) -- a yes, a refusal in its words and one under a yes
status, its own error page, a page where data was asked, a part of a page, a
file; the same app with its rate limit on, refusing with no words; with
sign-in on, for a person signed out; a port nothing listens on, for no
answer at all.  Each looked, once, like an answer no app route gives; each
was measured, and the app gives it.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from conftest import _NOWHERE
from tests._node_esm import run_node, served

LIB = Path(__file__).resolve().parents[1] / "molbuilder/web/static/lib"
MODULE = LIB / "server.js"
#: The page's notification bar, which a person signed out is told in.
BAR = LIB / "app-notifications.js"


def _sign_in():
    """The app with sign-in on: the suite's CAS entry (`support.pages`)."""
    from support.pages import SIGN_INS
    return {"auth": {"providers": [SIGN_INS[0]]}}


@pytest.fixture(scope="module")
def signed_out_at(isolated_projects_root_module):
    """The app with sign-in on, asked by nobody signed in."""
    from support.live_server import serve
    with serve(_sign_in()) as base:
        yield base


@pytest.fixture(scope="module")
def limited_at(isolated_projects_root_module):
    """The app with its rate limit on for every caller, this test's too: a
    question that looks like a scanner's is refused, with no words."""
    from support.live_server import serve
    with serve({"rate_limit": {"allowlist": []}}) as base:
        yield base


#: THE PAGE: where it is -- the notice's way back; Node has no location.
_PAGE = """
globalThis.location = new URL("http://127.0.0.1/task-setup?x=1");
"""


def _answers(base, asks):
    """Each ``(url, opts)`` of ``asks`` asked through the reader, against the
    app at ``base``, with what each route answered (``__answered``), what
    left (``__calls``) and what the bar showed."""
    snippet = f"""
        const {{ ask }} = await import({json.dumps(MODULE.as_uri())});
        // WHAT THE READER ASKED THE BAR TO SHOW, watched at the bar's door
        // -- each call passed on; the bar keeps its own count of rows.
        const asked = [];
        const show = molbuilder.notify.show;
        molbuilder.notify.show = (o) => {{ asked.push(o); return show(o); }};
        const out = [];
        for (const [url, opts] of {json.dumps(asks)}) {{
            const ctl = new AbortController();
            if (opts.cancel) ctl.abort();
            const a = await ask(url, Object.assign({{ signal: ctl.signal }},
                                                   opts));
            if (a.body instanceof Blob) a.body = {{ size: a.body.size }};
            if (a.headers) a.headers = Object.fromEntries(a.headers);
            out.push(a);
        }}
        console.log(JSON.stringify({{ out,
            answered: globalThis.__answered || [],
            sent: (globalThis.__calls || []).map((c) => ({{
                headers: c.init.headers, body: c.init.body }})),
            shown: molbuilder.notify.list(),
            actions: asked.map((o) => o.action || null) }}));
    """
    return run_node([BAR], snippet, globals_js=_PAGE + served(base))


_UNREADABLE = ("the server's answer could not be read (HTTP {}) — its "
               "log says why")


def test_a_yes_is_its_body_handed_over(app_at, isolated_projects_root_module):
    got = _answers(app_at, [
        (f"/api/files/list?path={isolated_projects_root_module}", {})])
    assert got["out"][0] == {"ok": True, "status": 200,
                             "body": got["answered"][0]}, got


def test_a_refusal_is_said_in_the_servers_own_words(
        app_at, isolated_projects_root_module):
    gone = isolated_projects_root_module / "gone.xyz"
    got = _answers(app_at, [("/api/build/load",
                             {"method": "POST", "body": {"path": str(gone)}})])
    out, said = got["out"][0], got["answered"][0]
    assert out["ok"] is False and out["status"] == 404, out
    assert said["error"] and out["error"] == said["error"], got


def test_an_error_page_is_never_called_a_network_error(app_at):
    """The app's own page for a route it has not -- an HTML page, read as an
    answer that could not be read, its status kept."""
    got = _answers(app_at, [("/api/nothing-here", {})])
    assert got["out"][0] == {"ok": False, "status": 404,
                             "error": _UNREADABLE.format(404)}, got


def test_no_answer_at_all_is_said_plainly():
    """Asked of the suite's port nothing listens on (`conftest._NOWHERE`,
    privileged, so no test process can take it)."""
    got = _answers(_NOWHERE, [("/api/q", {})])
    assert got["out"][0] == {
        "ok": False, "status": 0,
        "error": "could not reach the server — check the connection, "
                 "then try again"}, got


def test_a_refusal_under_a_yes_status_is_still_a_refusal(
        app_at, isolated_projects_root_module):
    """An advisory route answers 200 and ``ok: false`` (`web-api.md` § 1) --
    the checkpoint panel's start, a calculation's name typed with a space:
    the reader asks ``ok``, not the status, and hands on the words.
    MUTATION THIS MUST FAIL AGAINST: a body read without asking whether the
    server said yes."""
    folder = isolated_projects_root_module / "checkpointed"
    folder.mkdir(exist_ok=True)
    got = _answers(app_at, [("/api/checkpoint/init",
                             {"method": "POST",
                              "body": {"path": str(folder),
                                       "calculation": "a b"}})])
    out, said = got["out"][0], got["answered"][0]
    assert out["ok"] is False and out["status"] == 200, out
    assert said["error"] and out["error"] == said["error"], got


def test_a_refusal_with_no_words_says_its_status(limited_at):
    """A rate limit's refusal carries no words; the reader says the status.
    MUTATION: a refusal's empty body handed over as data."""
    out = _answers(limited_at, [("/api/files/list?x=<script>", {})])["out"][0]
    returns = {"ok": False, "status": 429,
               "error": "the server refused (HTTP 429)"}
    assert {k: out.get(k) for k in returns} == returns, out


def test_a_page_where_data_was_asked_is_never_handed_over_as_data(app_at):
    """A page answered with a yes status to a question that asked for data
    -- a gateway's, or a part of a page asked for as JSON -- is an answer
    that could not be read.  MUTATION: a yes's body parsed without the guard
    a refusal's has, the parser's own words handed to the person."""
    got = _answers(app_at, [("/partials/spectra-inspector", {})])
    assert got["out"][0] == {"ok": False, "status": 200,
                             "error": _UNREADABLE.format(200)}, got


def test_a_text_question_is_answered_as_text(app_at):
    """A question asked as text -- a part of a page (`as: "text"`): its yes is
    the text, and an error page is said as one."""
    got = _answers(app_at, [("/partials/spectra-inspector", {"as": "text"}),
                            ("/partials/nothing-here", {"as": "text"})])
    yes, page = got["out"]
    assert yes == {"ok": True, "status": 200, "body": got["answered"][0]}
    assert yes["body"].lstrip().startswith("<"), yes
    assert page == {"ok": False, "status": 404,
                    "error": _UNREADABLE.format(404)}, page


def test_a_cancelled_question_is_said_as_cancelled_not_failed(app_at):
    out = _answers(app_at, [("/api/files/list", {"cancel": True})])["out"][0]
    assert out["ok"] is False and out["aborted"] is True, out


def test_signed_out_is_said_once_for_the_page_with_a_way_back(signed_out_at):
    """A question the gate refuses with ``login_url``, and one it sends to the
    sign-in page, are each *signed out*; the page is told once, its link the
    page itself in a new tab.

    MUTATIONS THIS MUST FAIL AGAINST: a 401 read as a plain refusal; a
    redirect to the sign-in page read as an unreadable answer; the notice
    said at every question."""
    got = _answers(signed_out_at, [
        ("/api/files/list?path=/", {}),
        ("/partials/spectra-inspector", {"as": "text"})])
    assert [a.get("signedOut") for a in got["out"]] == [True, True], got["out"]
    assert [a["error"] for a in got["out"]] == ["you are signed out"] * 2
    assert len(got["actions"]) == 1 and len(got["shown"]) == 1, got
    assert {json.dumps(a, sort_keys=True) for a in got["actions"]} == {
        json.dumps({"label": "Sign in", "newTab": True,
                    "href": "/task-setup?x=1"}, sort_keys=True)}, got


def test_a_question_s_body_is_sent_as_json(app_at):
    got = _answers(app_at, [("/api/build/load",
                             {"method": "POST", "body": {"path": "/a.xyz"}})])
    assert got["sent"] == [{"headers": {"Content-Type": "application/json"},
                            "body": '{"path":"/a.xyz"}'}], got["sent"]


def test_a_file_answer_is_its_bytes_and_headers_and_a_refusal_as_any(
        app_at, isolated_projects_root_module):
    """An export's answer is a file: its bytes and the answer's headers
    beside them (its name, what the server said); refused, it is read as
    any refusal, in the server's own words."""
    kept = isolated_projects_root_module / "kept.txt"
    kept.write_text("xyz")
    got = _answers(app_at, [
        (f"/api/files/download?path={kept}", {"as": "file"}),
        (f"/api/files/download?path={kept.parent / 'gone.txt'}",
         {"as": "file"})])
    file, refused = got["out"]
    assert file["ok"] is True and file["body"] == {"size": 3}, file
    assert "kept.txt" in file["headers"].get("content-disposition", ""), file
    assert refused["ok"] is False and refused["error"], refused
    assert refused["error"] == got["answered"][1]["error"], got
