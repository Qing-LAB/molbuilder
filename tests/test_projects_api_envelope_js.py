"""``lib/projects/api.js`` -- the requests behind the sidebar's file calls,
and the uniform envelope they answer.

The contract (`web/projects.md` § 5: the file calls each make one server
request and return a uniform ``{ok, ...}`` result -- they never throw):
every async function in api.js returns ``{ok: bool, ...}`` and NEVER
throws.  Network failures and AbortErrors surface as ``{ok: false,
error: "..."}``; a server's answer is handed back as it came.

The module runs under node (`_node_esm.run_node`) with each request answered
by the app's own files routes on a tree of the test's own
(`_node_esm.served`); the browser's own failures -- a network drop, a
cancel -- are raised where the browser raises them.  An answer that is not
JSON is no route's of ours (a proxy's error page), and is not asked here.
"""
from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import quote

import pytest

from tests._node_esm import run_node, served


ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "molbuilder/web/static/lib/projects/api.js"


def _run(snippet: str, app_at: "str | None" = None) -> object:
    """``snippet`` with api.js imported as ``api``.  With ``app_at``, each
    request goes to the app serving there (`_node_esm.served`: each request
    kept as it left in ``__calls``, its options as ``init``, and each
    answer, read, in ``__answered``)."""
    return run_node(
        [], f"const api = await import({json.dumps(MODULE.as_uri())});\n"
            f"{snippet}",
        globals_js=served(app_at) if app_at else "")


@pytest.fixture(scope="module")
def app_at(isolated_projects_root_module):
    """The app, serving: what answers api.js (`support.live_server`)."""
    from support.live_server import serve
    with serve() as base:
        yield base


@pytest.fixture
def tree(isolated_projects_root):
    """This test's projects tree (`isolated_projects_root`): a project
    folder holding a note and a megabyte log."""
    (isolated_projects_root / "P").mkdir()
    (isolated_projects_root / "P" / "notes.txt").write_text("hello\n")
    (isolated_projects_root / "P" / "big.log").write_text("x" * 1_000_000)
    return isolated_projects_root


def _js(path) -> str:
    return json.dumps(str(path))


# ----- A write: the request it makes, the answer handed back ------- #


class TestWrites:

    def test_apiWrite_post_with_overwrite_flag(self, app_at, tree):
        note = tree / "P" / "notes.txt"
        out = _run(f"""
            const r = await api.apiWrite({_js(note)}, "text",
                                         {{overwrite: true}});
            console.log(JSON.stringify({{
                envelope: r,
                answer:   __answered[0],
                url:      __calls[0].url,
                method:   __calls[0].method,
                body:     __calls[0].body,
            }}));
        """, app_at)
        assert out["answer"]["ok"] is True, out["answer"]
        assert out["envelope"] == out["answer"]
        assert out["url"] == "/api/files/write"
        assert out["method"] == "POST"
        assert out["body"] == {"path": str(note), "text": "text",
                               "overwrite": True}
        assert note.read_text() == "text"

    def test_apiRename_posts_with_correct_body(self, app_at, tree):
        """apiRename POSTs {path, new_name} to /api/files/rename."""
        old = tree / "P" / "notes.txt"
        out = _run(f"""
            await api.apiRename({_js(old)}, "new.txt");
            console.log(JSON.stringify({{
                url:    __calls[0].url,
                method: __calls[0].method,
                body:   __calls[0].body,
                answer: __answered[0],
            }}));
        """, app_at)
        assert out["url"] == "/api/files/rename"
        assert out["method"] == "POST"
        assert out["body"] == {"path": str(old), "new_name": "new.txt"}
        assert out["answer"]["ok"] is True
        assert (tree / "P" / "new.txt").read_text() == "hello\n"


# ----- A read asks for the window it was given --------------------- #


@pytest.mark.parametrize("call, path, asked, unasked, window", [
    ("api.apiRead({F}, {{ maxBytes: 16777216 }})", "P/notes.txt",
     ["max_bytes=16777216"], [], {"text": "hello\n"}),
    ("api.apiRead({F})", "P/notes.txt", [], ["max_bytes"],
     {"text": "hello\n"}),
    ("api.apiReadRange({F})", "P/big.log", [], ["offset=", "max_bytes="],
     {"offset": 0, "length": 262144, "file_size": 1_000_000, "eof": False}),
    ("api.apiReadRange({F}, 524288, 262144)", "P/big.log",
     ["offset=524288", "max_bytes=262144"], [],
     {"offset": 524288, "length": 262144, "eof": False}),
    ("api.apiReadRange({F}, -262144, 262144)", "P/big.log",
     ["offset=-262144", "max_bytes=262144"], [],
     {"offset": 1_000_000 - 262144, "length": 262144, "eof": True}),
], ids=["apiRead lifts the budget it is given",
        "apiRead with none leaves the server's",
        "apiReadRange with no window takes the server's from byte 0",
        "apiReadRange asks for its offset and size",
        "apiReadRange's negative offset reads the file's end"])
def test_a_read_asks_for_the_window_it_was_given(app_at, tree, call, path,
                                                 asked, unasked, window):
    """The path URL-encoded, and on the query string exactly what the caller
    gave: the preview inspector's BULK read lifts the budget above the
    server's default (``maxBytes`` -> ``&max_bytes=``); the paginated source
    inspector pages by ``offset`` -- one dropped re-reads the first chunk
    forever -- and reads a log's end by a negative one (#189, 2026-06-02).
    Nothing given, nothing sent: the server's default applies.  The server's
    window comes back as it came."""
    target = tree / path
    out = _run(f"""
        const r = await {call.format(F=_js(target))};
        console.log(JSON.stringify({{ r, url: __calls[0].url,
                                      answer: __answered[0] }}));
    """, app_at)
    assert out["answer"]["ok"] is True, out["answer"]
    assert out["r"] == out["answer"]
    assert quote(str(target), safe="") in out["url"]
    for part in asked:
        assert part in out["url"], (part, out["url"])
    for part in unasked:
        assert part not in out["url"], (part, out["url"])
    assert {k: out["r"][k] for k in window} == window
    if "length" in window:
        assert out["r"]["text"] == "x" * window["length"]


# ----- Network failure: fetch itself rejects ----------------------- #


class TestNetworkFailure:

    def test_apiList_network_drop_returns_envelope(self):
        """``fetch`` throws TypeError on network drop / offline /
        DNS fail; apiList returns the synthetic envelope."""
        out = _run('''
            global.fetch = async () => {
                throw new TypeError("Failed to fetch");
            };
            const r = await api.apiList("/x");
            console.log(JSON.stringify(r));
        ''')
        assert out["ok"] is False
        assert "network error" in out["error"]
        assert "Failed to fetch" in out["error"]


# ----- AbortSignal threading (#174 sidebar gap M1) ---------------- #


class TestAbortSignal:

    def test_apiWrite_aborted_returns_clean_envelope(self, app_at, tree):
        """When the caller's AbortSignal fires, fetch rejects with
        AbortError.  api.js produces a CLEAN envelope (no "network
        error" prose) so callers can distinguish user-cancellation
        from connectivity failure."""
        out = _run(f"""
            const ctl = new AbortController();
            ctl.abort();
            const r = await api.apiWrite({_js(tree / "P" / "notes.txt")},
                                         "text", {{ signal: ctl.signal }});
            console.log(JSON.stringify(r));
        """, app_at)
        assert out == {"ok": False, "error": "aborted", "aborted": True}
        assert (tree / "P" / "notes.txt").read_text() == "hello\n"

    @pytest.mark.parametrize("call", [
        "api.apiWrite({P} + '/notes.txt', 'text', {{overwrite: true, signal}})",
        "api.apiDelete({P} + '/notes.txt', false, {{signal}})",
        "api.apiUpload({P}, new File(['x'], 'f.txt'), {{signal}})",
        "api.apiRead({P} + '/notes.txt', {{signal}})",
        "api.apiReadRange({P} + '/big.log', 0, 262144, {{signal}})",
        "api.apiList({P}, {{signal}})",
        "api.apiRename({P} + '/notes.txt', 'renamed.txt', {{signal}})",
        "api.apiMkdir({P}, 'optimization', {{signal}})",
    ], ids=["apiWrite", "apiDelete", "apiUpload", "apiRead", "apiReadRange",
            "apiList", "apiRename", "apiMkdir"])
    def test_the_signal_reaches_fetch_and_the_server_answers(
            self, app_at, tree, call):
        """The ``signal`` from opts must reach the fetch init -- the SAME
        AbortSignal instance, not a copy -- or the sidebar's Cancel (and
        the lock's three-layer recovery) cancels nothing; and the server's
        answer comes back as it came."""
        out = _run(f"""
            const ctl = new AbortController();
            const signal = ctl.signal;
            const r = await {call.format(P=_js(tree / "P"))};
            console.log(JSON.stringify({{
                forwarded: __calls[0].init.signal === signal,
                r, answer: __answered[0] }}));
        """, app_at)
        assert out["forwarded"] is True
        assert out["answer"]["ok"] is True, out["answer"]
        assert out["r"] == out["answer"]


class TestApiRoots:

    def test_the_roots_come_back_in_the_envelope(self, app_at, tree):
        """`/api/files/roots` answers the roots the picker may browse;
        api.js hands them back in the uniform envelope."""
        out = _run("""
            const r = await api.apiRoots();
            console.log(JSON.stringify({ r, answer: __answered[0] }));
        """, app_at)
        assert out["r"] == {"ok": True, "roots": out["answer"]["roots"]}
        assert str(tree) in [root["path"] for root in out["r"]["roots"]]

    def test_includes_empty_roots_on_failure(self):
        """Failure case carries an empty ``roots`` array so callers
        that destructure ``{roots}`` don't NPE."""
        out = _run('''
            global.fetch = async () => {
                throw new TypeError("Failed to fetch");
            };
            const r = await api.apiRoots();
            console.log(JSON.stringify(r));
        ''')
        assert out["ok"] is False
        assert "network error" in out["error"]
        assert out["roots"] == []


# ----- cache: "no-store" default (#193, 2026-06-02) ---------------- #


class TestNoStoreCache:
    """The central ``_fetchEnvelope`` defaults ``cache: "no-store"`` so
    every projects.* live-data GET reaches the server on a same-URL
    revisit.  Without this default the browser HTTP cache served the
    previous response on every same-URL hit -- the second-half cause of
    the /results stale-dropdown bug (#192).  One caller stands for all of
    them: every function here reaches `fetch` through that one wrapper.
    That a caller's signal survives the default beside it is each row of
    `test_the_signal_reaches_fetch_and_the_server_answers`."""

    def test_the_default_reaches_fetch(self, app_at, tree):
        out = _run(f"""
            await api.apiList({_js(tree / "P")});
            console.log(JSON.stringify(__calls.map(c => c.init.cache)));
        """, app_at)
        assert out == ["no-store"]
