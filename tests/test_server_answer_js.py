"""The one reader of a server's answer -- ``lib/server.js`` (`web/ui-contract.md`
§ 12).

Every question a page asks goes through it, so its reading is every page's.
Each row is one kind of answer, named for its failure: a refusal painted as
data, an error page called a network error, a cancelled question said as a
failure, a person signed out never told.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

MODULE = (Path(__file__).resolve().parents[1]
          / "molbuilder/web/static/lib/server.js")

#: A server faked at ``fetch``: each row says what it answers -- a status, a
#: body's text, a redirect -- or that it throws.
_DRIVER = r"""
const said = [];
global.window = { location: { pathname: "/task-setup", search: "?x=1" },
                  molbuilder: { notify: { show: (o) => said.push(o) } } };
let answer = null;
global.fetch = async (url, init) => {
    global.__asked = { url, init };
    if (answer.throws) { const e = new Error(answer.throws);
                         e.name = answer.name || "TypeError"; throw e; }
    return { status: answer.status, ok: answer.status >= 200 && answer.status < 300,
             headers: { get: (k) => (answer.headers || {})[k] || null },
             blob: async () => ({ size: answer.text.length }),
             redirected: !!answer.redirectedTo,
             url: answer.redirectedTo || ("https://h" + url),
             text: async () => answer.text };
};
import(process.argv[2]).then(async (m) => {
    const out = [];
    for (const row of JSON.parse(process.argv[3])) {
        answer = row.answer;
        const ctl = new AbortController();
        if (row.cancel) ctl.abort();
        const a = await m.ask("/api/q", Object.assign({ signal: ctl.signal },
                                                      row.opts || {}));
        out.push(a);
    }
    process.stdout.write(JSON.stringify({ out, said,
        asked: global.__asked && { headers: global.__asked.init.headers,
                                   body: global.__asked.init.body } }));
});
"""


def _ask(rows):
    node = shutil.which("node")
    if node is None:
        pytest.skip("node not available")
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        drv = Path(d) / "drive.mjs"
        drv.write_text(_DRIVER)
        proc = subprocess.run(
            [node, str(drv), MODULE.as_uri(), json.dumps(rows)],
            capture_output=True, text=True, timeout=20)
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout)


#: (what the server does, the answer it gives, what the reader returns)
_ROWS = [
    ("a yes, its body handed over",
     {"status": 200, "text": '{"ok": true, "n": 3}'},
     {"ok": True, "status": 200, "body": {"ok": True, "n": 3}}),
    ("a refusal in the server's own words, its status a yes",
     {"status": 200, "text": '{"ok": false, "error": "no such folder"}'},
     {"ok": False, "status": 200, "error": "no such folder"}),
    ("a refusing status in the server's own words",
     {"status": 409, "text": '{"ok": false, "error": "changed on disk"}'},
     {"ok": False, "status": 409, "error": "changed on disk"}),
    ("a refusal with no words (a rate limit)",
     {"status": 429, "text": ""},
     {"ok": False, "status": 429, "error": "the server refused (HTTP 429)"}),
    ("an error page, never called a network error",
     {"status": 500, "text": "<html>Internal Server Error</html>"},
     {"ok": False, "status": 500,
      "error": "the server's answer could not be read (HTTP 500) \u2014 its "
               "log says why"}),
    ("no answer at all, said plainly",
     {"throws": "Failed to fetch"},
     {"ok": False, "status": 0,
      "error": "could not reach the server \u2014 check the connection, "
               "then try again"}),
]


@pytest.mark.parametrize("what,answer,returns", _ROWS,
                         ids=[r[0] for r in _ROWS])
def test_each_answer_is_read_as_what_it_is(what, answer, returns):
    """MUTATIONS THIS MUST FAIL AGAINST: a body read without asking whether
    the server said yes; an error page read as unreachable; a refusal's own
    words dropped for a status."""
    out = _ask([{"answer": answer}])["out"][0]
    assert {k: out.get(k) for k in returns} == returns, what


#: A question asked as text -- a part of a page (`as: "text"`): its yes is
#: the text, and anything else is read as any answer is.
_TEXT_ROWS = [
    ("its yes is the text",
     {"status": 200, "text": "<section>the panel</section>"},
     {"ok": True, "status": 200, "body": "<section>the panel</section>"}),
    ("an error page, said as one",
     {"status": 500, "text": "<html>Internal Server Error</html>"},
     {"ok": False, "status": 500,
      "error": "the server's answer could not be read (HTTP 500) \u2014 its "
               "log says why"}),
    ("the sign-in page, said as signed out",
     {"status": 200, "text": "<html>sign in</html>",
      "redirectedTo": "https://h/login"},
     {"ok": False, "signedOut": True}),
]


@pytest.mark.parametrize("what,answer,returns", _TEXT_ROWS,
                         ids=[r[0] for r in _TEXT_ROWS])
def test_a_text_answer_is_read_as_what_it_is(what, answer, returns):
    """MUTATION THIS MUST FAIL AGAINST: a text question's error page said
    as a plain refusal, the reader's sentence for it lost."""
    out = _ask([{"answer": answer, "opts": {"as": "text"}}])["out"][0]
    assert {k: out.get(k) for k in returns} == returns, what


def test_a_cancelled_question_is_said_as_cancelled_not_failed():
    out = _ask([{"answer": {"throws": "aborted", "name": "AbortError"},
                 "cancel": True}])["out"][0]
    assert out["ok"] is False and out["aborted"] is True, out


def test_signed_out_is_said_once_for_the_page_with_a_way_back():
    """A 401 carrying ``login_url``, and an answer the gate sent to the
    sign-in page, are each *signed out*; the page is told once, its link the
    page itself in a new tab.

    MUTATIONS THIS MUST FAIL AGAINST: a 401 read as a plain refusal; a
    redirect to the sign-in page read as an unreadable answer; the notice
    said at every question."""
    got = _ask([
        {"answer": {"status": 401, "text": json.dumps(
            {"ok": False, "error": "not authenticated",
             "login_url": "/login?next=/api/q"})}},
        {"answer": {"status": 200, "text": "<html>sign in</html>",
                    "redirectedTo": "https://h/login"}},
    ])
    assert [a["signedOut"] for a in got["out"]] == [True, True], got["out"]
    assert [a["error"] for a in got["out"]] == ["you are signed out"] * 2
    assert len(got["said"]) == 1, got["said"]
    assert got["said"][0]["action"] == {
        "label": "Sign in", "newTab": True, "href": "/task-setup?x=1"}


def test_a_question_s_body_is_sent_as_json_and_a_form_as_it_is():
    got = _ask([{"answer": {"status": 200, "text": "{}"},
                 "opts": {"method": "POST", "body": {"a": 1}}}])
    assert got["asked"] == {"headers": {"Content-Type": "application/json"},
                            "body": '{"a":1}'}


def test_a_text_answer_is_handed_over_as_text():
    """A question whose answer is not JSON -- a partial page."""
    out = _ask([{"answer": {"status": 200, "text": "<div>panel</div>"},
                 "opts": {"as": "text"}}])["out"][0]
    assert out == {"ok": True, "status": 200, "body": "<div>panel</div>"}


def test_a_file_answer_is_its_bytes_and_headers_and_a_refusal_as_any():
    """An export's answer is a file: its bytes and the answer's headers
    beside them (its name, what the server said); refused, it is read as
    any refusal, in the server's own words."""
    got = _ask([
        {"answer": {"status": 200, "text": "xyz", "headers": {
            "Content-Disposition": 'attachment; filename="a.zip"'}},
         "opts": {"as": "file"}},
        {"answer": {"status": 404, "text": '{"error": "no such run"}'},
         "opts": {"as": "file"}},
    ])["out"]
    assert got[0]["ok"] is True and got[0]["body"] == {"size": 3}, got[0]
    assert got[1] == {"ok": False, "status": 404,
                      "body": {"error": "no such run"},
                      "error": "no such run"}, got[1]
