"""`window.molbuilder.projects` -- the sidebar's one door (`web/projects.md`
§ 1): the selection and its channels, and the file calls a tab reaches
through it, run under node (`_node_esm.run_node`), each request answered
by the app's own files routes on a tree of the test's own
(`_node_esm.served`).

What is pinned here is what `state.js` decides on top of `api.js`, whose
requests and envelopes are `test_projects_api_envelope_js.py`'s:

  * the selection: a single click is `onChange`, a double click
    `onCommit`, its file a full path; the same callback registered twice
    is an error (§ 2);
  * each organizing call refreshes the folder it changed, only when the
    change landed, and a refresh that fails does not fail the change (§ 5);
  * every call answers `{ok, ...}` and never throws (§ 5) -- `refresh`,
    `readCurrentFile`, a refused `writeFile`, `upload`;
  * `navigateTo` refuses while the page's busy fence is held
    (`ui-contract.md` § 10, *Programmatic races*).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests._node_esm import run_node, served


ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "molbuilder/web/static/lib/projects/state.js"
PAGE_BUSY = ROOT / "molbuilder/web/static/lib/page-busy.js"

#: What a browser has and node lacks, as a page starts: an empty
#: ``sessionStorage``, a document with nothing on it.
_BROWSER = """
const _store = new Map();
globalThis.sessionStorage = {
    getItem:    k => _store.has(k) ? _store.get(k) : null,
    setItem:    (k, v) => _store.set(k, String(v)),
    removeItem: k => _store.delete(k),
    clear:      () => _store.clear(),
};
globalThis.document = {
    addEventListener: () => {},
    getElementById:   () => null,
};
"""


def _run(snippet: str, app_at: "str | None" = None) -> object:
    """``snippet`` with state.js imported as ``state`` and the page's fence
    as ``pageBusy``.  With ``app_at``, each request goes to the app serving
    there (`_node_esm.served`: each request kept as it left in ``__calls``,
    its options as ``init``, and each answer, read, in ``__answered``)."""
    return run_node(
        [], f"const state = await import({json.dumps(MODULE.as_uri())});\n"
            f"const {{ pageBusy }} = await import("
            f"{json.dumps(PAGE_BUSY.as_uri())});\n"
            f"{snippet}",
        globals_js=_BROWSER + (served(app_at) if app_at else ""))


@pytest.fixture(scope="module")
def app_at(isolated_projects_root_module):
    """The app, serving: what answers the wrappers (`support.live_server`)."""
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


# ----- onCommit + publishCommit (B.5.2) --------------------------- #


class TestOnCommit:
    """`web/projects.md` § 2: a single click is a preview (`onChange`), a
    double click is a commit (`onCommit` only) -- so a subscriber does not
    mistake a casual browse-click for a tab-level action."""

    def test_onCommit_does_NOT_fire_on_subscribe(self):
        """Commits are discrete events, not states.  Firing on
        subscribe would over-react the first time a tab mounts;
        the published payload would be the empty initial state
        and subscribers would think the user committed nothing."""
        out = _run('''
            const calls = [];
            state.projects.onCommit(p => calls.push(p));
            console.log(JSON.stringify({
                callsAfterSubscribe: calls,
            }));
        ''')
        assert out["callsAfterSubscribe"] == []

    def test_publishCommit_also_updates_global_pick_via_setShared(self):
        """`publishCommit` delivers `{dir, file}` -- the file's FULL path --
        to every `onCommit` subscriber, AND moves the pick (`setShared`) when
        it differs, so `onChange` follows too.

        `web/projects.md` § 5 states the payload and is explicit that `path`
        is the full path, "not its name -- subscribers use it as one". A
        publisher sending the basename hands every tab a string it then
        resolves against its own current directory: the double-click opens
        a different file, or none, with no error anywhere.  A commit that
        left the pick behind would leave `getCurrentFile()` naming the file
        before."""
        out = _run('''
            const onChangeCalls = [];
            const onCommitCalls = [];
            state.projects.onChange(p => onChangeCalls.push(p));
            state.projects.onCommit(p => onCommitCalls.push(p));
            // Reset onChange's fire-once-on-subscribe payload.
            onChangeCalls.length = 0;
            state.publishCommit("/p", "/p/x.xyz");
            console.log(JSON.stringify({
                onChange:    onChangeCalls,
                onCommit:    onCommitCalls,
                currentFile: state.projects.getCurrentFile(),
            }));
        ''')
        # Both subscriber sets fire with the same payload.
        assert out["onChange"] == [{"dir": "/p", "file": "/p/x.xyz"}]
        assert out["onCommit"] == [{"dir": "/p", "file": "/p/x.xyz"}]
        # And the global pick is updated.
        assert out["currentFile"] == "/p/x.xyz"

    def test_publishCommit_skips_inner_setShared_when_pick_unchanged(self):
        """BOMB-4 fix (2026-06-07): a real-user dblclick on the
        sidebar fires TWO click events (each calling setShared) plus
        one dblclick event (calling publishCommit).  If publishCommit
        unconditionally calls setShared, ``onChange`` subscribers
        fire THREE times per dblclick — save.js's refreshState ran
        3×, Spectra's schema-reload ran 3× when the onChange
        fallback was active.  Dedup: publishCommit's inner setShared
        is a no-op when the global pick already matches.

        Setup mimics the dblclick flow: setShared the file FIRST
        (the dblclick's first click), then publishCommit.  Without
        the dedup, onChange would fire twice (initial + inner
        setShared); with it, only the first call fires."""
        out = _run('''
            // First click fires setShared.
            state.setShared("/p", "/p/x.xyz");
            // Now wire the subscriber + reset its initial fire.
            const onChangeCalls = [];
            state.projects.onChange(p => onChangeCalls.push(p));
            onChangeCalls.length = 0;
            // Second click fires publishCommit on the SAME pick.
            state.publishCommit("/p", "/p/x.xyz");
            console.log(JSON.stringify(onChangeCalls));
        ''')
        # With the dedup, onChange is silent — the inner setShared
        # was skipped because sessionStorage already had the pick.
        assert out == [], (
            f"onChange should NOT fire when publishCommit's pick "
            f"matches sessionStorage; got {out}"
        )

    def test_setShared_does_NOT_fire_onCommit_subscribers(self):
        """Single-click → setShared → onChange.  publishCommit is
        separate; setShared MUST NOT also fire onCommit, or every
        sidebar click would commit and the candidate-only model
        collapses."""
        out = _run('''
            const onCommitCalls = [];
            state.projects.onCommit(p => onCommitCalls.push(p));
            state.setShared("/p", "/p/y.xyz");
            console.log(JSON.stringify(onCommitCalls));
        ''')
        assert out == []

    def test_unsubscribe_works(self):
        """The function `onCommit` returns actually removes the subscriber.

        A no-op unsubscribe is invisible until a tab is torn down and rebuilt:
        the stale callback keeps firing (its exception swallowed by the
        publisher's per-subscriber isolation, so nothing reports it), and the
        duplicate-registration throw -- `web/projects.md` § 2, the same
        callback twice is an error -- then refuses the re-subscribe. Together
        those turn a leak into a tab that cannot be re-entered.
        """
        out = _run('''
            const calls = [];
            const unsub = state.projects.onCommit(p => calls.push(p));
            unsub();
            state.publishCommit("/p", "/p/x.xyz");
            console.log(JSON.stringify(calls));
        ''')
        assert out == []


# ----- setShared -------------------------------------------------- #


class TestSetSharedOnTheDoor:

    def test_setShared_answers_through_projects(self):
        """`projects.setShared` is what the Results file picker calls
        (`results/file-picker.js`), and every other test here reaches
        `state.setShared` directly.  THE SILENT FAILURE: the name gone from the
        public object, and picking a run's file failing on Results alone."""
        out = _run('''
            const r = state.projects.setShared("/a", "/a/b.xyz");
            console.log(JSON.stringify({ok: r.ok}));
        ''')
        assert out["ok"] is True


# ----- readRange: a network drop -------------------------------- #


def test_readRange_network_drop_returns_uniform_envelope():
    """Same shape as every other projects.* wrapper -- network
    drop must NOT throw."""
    out = _run('''
        global.fetch = async () => {
            throw new TypeError("Failed to fetch");
        };
        const r = await state.projects.readRange("/p/x.log");
        console.log(JSON.stringify(r));
    ''')
    assert out["ok"] is False
    assert "network error" in out["error"]


# ----- createProject / mkdir / upload: refresh on success --------- #


def _refreshed(app_at, tree, call):
    """``call`` through the public surface, the sidebar's refresh recorded
    and its projects root the tree's: what came back, what the server
    answered, and what was refreshed."""
    return _run(f"""
        const refreshArgs = [];
        state.setRefreshHandler(async (dir) => refreshArgs.push(dir));
        state.setProjectsRoot({json.dumps(str(tree))});
        const P = {json.dumps(str(tree / "P"))};
        const r = await {call};
        console.log(JSON.stringify({{
            envelope: r, answer: __answered[0] || null, refreshArgs }}));
    """, app_at)


class TestRefreshOnSuccess:
    """`web/projects.md` § 5: each organizing call "refreshes the sidebar on
    success" -- the wrapper's OWN decision (`api.js` knows nothing about the
    sidebar), and the directory it refreshes is what makes it correct:
    refreshing the wrong one leaves the change invisible while re-listing
    something else; a file that lands and does not appear reads as an
    upload that failed, and a row left for a deleted file is one someone
    clicks.  Each call is answered by the route, and lands on disk."""

    @pytest.mark.parametrize("call, refreshed, landed", [
        ("state.projects.createProject('myjob')", "", "myjob"),
        ("state.projects.mkdir(P, 'optimization')", "P", "P/optimization"),
        ("state.projects.upload(P, new File(['x'], 'f.xyz'))", "P", "P/f.xyz"),
        ("state.projects.rename(P + '/notes.txt', 'new.txt')", "P",
         "P/new.txt"),
        ("state.projects.deleteEntry(P + '/notes.txt', false)", "P", None),
    ], ids=["createProject refreshes the root", "mkdir its parent",
            "upload its target", "rename the old path's parent",
            "delete its parent"])
    def test_a_change_that_landed_refreshes_where_it_landed(
            self, app_at, tree, call, refreshed, landed):
        out = _refreshed(app_at, tree, call)
        assert out["answer"]["ok"] is True, out["answer"]
        assert out["envelope"]["ok"] is True
        assert out["refreshArgs"] == [str(tree / refreshed).rstrip("/")]
        if landed:
            assert (tree / landed).exists()
        else:
            assert not (tree / "P" / "notes.txt").exists()


# ----- Failure path: no refresh ----------------------------------- #


class TestNoRefreshOnFailure:
    """A refused change must NOT trigger the listing refresh -- the folder
    did not change -- and comes back in the server's words."""

    @pytest.mark.parametrize("call", [
        "state.projects.rename(P + '/notes.txt', 'big.log')",
        "state.projects.mkdir(P, 'not-a-topic')",
    ], ids=["a rename onto an existing name (409)",
            "a folder the project's topics do not name"])
    def test_a_refused_change_does_not_refresh(self, app_at, tree, call):
        out = _refreshed(app_at, tree, call)
        assert out["answer"]["ok"] is False, out["answer"]
        assert out["envelope"]["ok"] is False
        assert out["envelope"]["error"] == out["answer"]["error"]
        assert out["refreshArgs"] == []

    def test_upload_network_failure_does_not_refresh(self):
        """A network drop during upload returns the failure envelope AND does not
        refresh.

        Two halves that must hold together: `fetch` throwing must not propagate
        (`web/projects.md` § 5 -- these calls never throw), and the refresh must
        be gated on `r.ok` rather than merely on having returned. A refresh after a
        failed upload re-lists a directory that did not change, which on a slow
        filesystem makes a failure look like a success that has not appeared yet.
        """
        out = _run('''
            const refreshArgs = [];
            state.setRefreshHandler(async (dir) => refreshArgs.push(dir));
            global.fetch = async () => {
                throw new TypeError("Failed to fetch");
            };
            const r = await state.projects.upload("/p", { name: "f" });
            console.log(JSON.stringify({
                envelope:    r,
                refreshArgs: refreshArgs,
            }));
        ''')
        assert out["envelope"]["ok"] is False
        assert out["refreshArgs"] == []


# ----- Refresh failure swallowed ---------------------------------- #


class TestRefreshFailureSwallowed:

    def test_mkdir_success_with_failing_refresh_still_succeeds(self, app_at,
                                                               tree):
        """If the post-success refresh itself throws (e.g. transient
        listing fetch failure), the mutation envelope should still
        report success -- the dir DID land on disk; the user can
        refresh manually."""
        out = _run(f"""
            state.setRefreshHandler(async () => {{
                throw new Error("listing fetch failed");
            }});
            const r = await state.projects.mkdir(
                {json.dumps(str(tree / "P"))}, "optimization");
            console.log(JSON.stringify({{ r, answer: __answered[0] }}));
        """, app_at)
        assert out["answer"]["ok"] is True
        assert out["r"] == out["answer"]
        assert (tree / "P" / "optimization").is_dir()


# ----- navigateTo ------------------------------------------------ #


class TestNavigateTo:
    """`navigateTo(path)` lists a folder and moves there (`web/projects.md`
    § 5): the sidebar's own lister, wired by ``setNavigateToImpl`` at the
    sidebar's init.  Before that, it answers a clean "unavailable"
    envelope rather than throwing, and it refuses while the page's busy
    fence is held."""

    def test_unavailable_before_setNavigateToImpl_wires_it(self):
        """Without setNavigateToImpl being called, navigateTo returns
        the documented "unavailable" envelope -- NOT throws.  Pins
        the fail-safe contract for tabs that race against sidebar
        init."""
        out = _run('''
            // sidebar's init NOT run; setNavigateToImpl never called.
            const r = await state.projects.navigateTo("/d");
            console.log(JSON.stringify(r));
        ''')
        assert out["ok"] is False
        assert "unavailable" in out["error"]
        assert "sidebar not initialised" in out["error"]

    def test_busy_navigateTo_rejects_without_calling_impl(self):
        """`ui-contract.md` § 10, *Programmatic races*: the cover blocks
        clicks, not code, so `navigateTo` refuses while the fence is held
        -- without calling the lister.  The lister (`openDir`) is not
        guarded because it doubles as the refresh handler a held
        operation's own Save calls; the refusal lives on the public
        call."""
        out = _run('''
            let implCalled = false;
            state.setNavigateToImpl(async () => {
                implCalled = true;
                return { ok: true, path: "/", entries: [] };
            });
            pageBusy.claim("Saving FDF...", []);
            const r = await state.projects.navigateTo("/somewhere");
            console.log(JSON.stringify({
                envelope:    r,
                implCalled:  implCalled,
            }));
        ''')
        assert out["envelope"]["ok"] is False
        assert "page is busy" in out["envelope"]["error"]
        assert "Saving FDF" in out["envelope"]["error"]
        assert out["implCalled"] is False


# ----- readCurrentFile ------------------------------------------- #


class TestReadCurrentFileEnvelope:
    """`readCurrentFile` reads the selected file (`web/projects.md` § 5) and
    answers one of three: `null` when nothing is picked, `{ok:true, path,
    text}` for a read, `{ok:false, error}` for a failure (`state.js`).  The
    file is picked as the sidebar picks it (`setShared`), and read by the
    route."""

    def test_no_file_selected_returns_null(self):
        """With nothing selected, `readCurrentFile()` returns `null` -- distinct from an
        error envelope.

        Three terminal states, three shapes: `null` for "no file selected",
        `{ok:true,...}` for a read, `{ok:false,error}` for a failure. Callers branch on
        `null` first, so folding it into an error envelope makes every tab show a read
        error on a page where the user has simply not picked anything yet.
        """
        out = _run('''
            const r = await state.projects.readCurrentFile();
            console.log(JSON.stringify(r));
        ''')
        assert out is None

    def test_success_returns_envelope_with_ok(self, app_at, tree):
        """A successful `readCurrentFile()` returns the read envelope with `ok:true`.

        The middle of the three states above. It also pins that the file read is the
        one in the shared selection slot (`web/projects.md` § 2) rather than one
        passed in -- a reader pointed at the wrong slot returns another file's
        contents with `ok:true`, which no caller checks for.
        """
        note = tree / "P" / "notes.txt"
        out = _run(f"""
            state.setShared({json.dumps(str(tree / "P"))},
                            {json.dumps(str(note))});
            const r = await state.projects.readCurrentFile();
            console.log(JSON.stringify(r));
        """, app_at)
        assert out == {"ok": True, "path": str(note), "text": "hello\n"}

    def test_failure_returns_envelope_not_null(self, app_at, tree):
        """A failed read of the selected file returns `{ok:false, error}` -- NOT `null`.

        The third state, and the one that actually goes wrong: `null` is the caller's
        "nothing is selected" signal, so returning it for a read failure makes a
        missing or unreadable file look like an empty selection, and the tab silently
        does nothing instead of reporting the error.  Here the picked file was
        deleted since it was picked.
        """
        note = tree / "P" / "notes.txt"
        note.unlink()
        out = _run(f"""
            state.setShared({json.dumps(str(tree / "P"))},
                            {json.dumps(str(note))});
            const r = await state.projects.readCurrentFile();
            console.log(JSON.stringify({{ r, answer: __answered[0] }}));
        """, app_at)
        assert out["answer"]["ok"] is False
        assert out["r"] == {"ok": False, "error": out["answer"]["error"]}


# ----- refresh --------------------------------------------------- #


class TestRefreshEnvelope:
    """`refresh()` re-lists the current folder and, as every call on the
    door, answers `{ok:true}` or `{ok:false, error}` and never throws
    (`web/projects.md` § 5).  It answered `undefined` on every path
    before."""

    def test_no_current_dir_returns_envelope_not_undefined(self):
        """`refresh()` with no current directory returns `{ok:false, error}`, not
        `undefined`.

        Commemorates the defect the envelope rule was written for: `refresh` returned
        `undefined` on every path, so a caller writing `if (r.ok)` threw on a property
        of undefined instead of handling the case. Every call on this surface returns
        a uniform result (`web/projects.md` § 5).
        """
        out = _run('''
            const r = await state.projects.refresh();
            console.log(JSON.stringify({
                r: r,
                isUndefined: r === undefined,
            }));
        ''')
        assert out["isUndefined"] is False
        assert out["r"]["ok"] is False
        assert "no current directory" in out["r"]["error"]

    def test_no_refresh_handler_returns_envelope(self):
        """`refresh()` before the sidebar has wired its handler answers a clean failure
        envelope rather than throwing.

        Tabs subscribe and then act, while the sidebar's init is asynchronous. A throw
        here lands in another tab's setup, where nothing catches it and it stops that
        tab loading -- for a condition that resolves itself a moment later. The same
        fail-safe shape `navigateTo` has before `setNavigateToImpl` runs.
        """
        out = _run('''
            state.setShared("/p", "");
            const r = await state.projects.refresh();
            console.log(JSON.stringify(r));
        ''')
        assert out["ok"] is False
        assert "no refresh handler" in out["error"]

    def test_success_returns_ok_envelope(self):
        """A refresh whose handler succeeds returns exactly `{ok:true}`.

        The positive control for the three failure envelopes around it: a `refresh`
        that always reported failure would satisfy every one of them. Asserted as an
        equality, which also pins that no internal detail rides out on the result.
        """
        out = _run('''
            state.setShared("/p", "");
            state.setRefreshHandler(async (dir) => { /* success */ });
            const r = await state.projects.refresh();
            console.log(JSON.stringify(r));
        ''')
        assert out == {"ok": True}

    def test_handler_throws_returns_error_envelope(self):
        """A refresh handler that throws is caught, and the underlying message is kept.

        `web/projects.md` § 5 -- these calls never throw -- but swallowing the
        REASON is the other half of the failure: "refresh failed" alone tells a person
        nothing, so the handler's own message is carried inside it. A transient
        listing failure must not take down the caller that asked for the refresh.
        """
        out = _run('''
            state.setShared("/p", "");
            state.setRefreshHandler(async () => {
                throw new Error("listing failed");
            });
            const r = await state.projects.refresh();
            console.log(JSON.stringify(r));
        ''')
        assert out["ok"] is False
        assert "refresh failed" in out["error"]
        assert "listing failed" in out["error"]


# ----- writeFile: what a refusal carries -------------------------- #


class TestWriteFileEdgeFields:
    """A refused write keeps what the server said beside its words: the
    preview's Save uses the file's on-disk timestamp so a change on disk
    since it opened is caught (`web/projects.md` § 4, the preview pop-up),
    and the 409's `actual_mtime` is how the editor knows that is what
    happened.  `writeFile` once kept only `error`."""

    def test_writeFile_preserves_actual_mtime_on_409(self, app_at, tree):
        """A 409 edit-conflict carries `actual_mtime` out of `writeFile`, not just
        `error`.

        The recorded defect: `writeFile` destructured only `error` and dropped
        `actual_mtime`, so a tab could not tell an edit conflict from any other
        failure programmatically -- the file on disk had changed under the editor and
        the only thing offered was a generic error. That field is what lets the caller
        offer "reload and re-apply" instead.  The editor's copy is of a time
        before the file last changed; the server refuses it, saying when the
        file changed.
        """
        note = tree / "P" / "notes.txt"
        out = _run(f"""
            const r = await state.projects.writeFile(
                {json.dumps(str(note))}, "text",
                {{ overwrite: true, expected_mtime: 1 }});
            console.log(JSON.stringify({{ r, answer: __answered[0] }}));
        """, app_at)
        answer = out["answer"]
        assert answer["ok"] is False and answer["actual_mtime"], answer
        assert out["r"] == {"ok": False, "error": answer["error"],
                            "actual_mtime": answer["actual_mtime"]}
        assert note.read_text() == "hello\n"

    def test_writeFile_preserves_aborted_flag(self, app_at, tree):
        """A Cancel before the write lands: the AbortError envelope carries
        aborted:true, and writeFile must not drop the flag."""
        out = _run(f"""
            const ac = new AbortController();
            ac.abort();
            const r = await state.projects.writeFile(
                {json.dumps(str(tree / "P" / "notes.txt"))}, "text",
                {{ overwrite: true, signal: ac.signal }});
            console.log(JSON.stringify(r));
        """, app_at)
        assert out["ok"] is False
        assert out["aborted"] is True


#: What a cancellation arrives as across the page, and what is not one:
#: ``(the value, a cancel?)``.  A fetch aborted by its signal throws an
#: ``AbortError``; api.js and ``writeFile`` answer ``aborted: true``; a
#: door's answer for a question the person declined is ``cancelled: true``
#: (projects.parser, molview-doors); mol-viewer-embed's ViewerError says
#: ``code: "aborted"``.  ``code: "disposed"`` is a host tearing an embed
#: down, not the person's cancel.
CANCEL_SHAPES = [
    ('Object.assign(new Error("aborted"), { name: "AbortError" })', True),
    ('{ ok: false, aborted: true, error: "x" }', True),
    ('{ ok: false, cancelled: true }', True),
    ('{ code: "aborted", message: "x" }', True),
    ('{ code: "disposed", message: "x" }', False),
    ('new TypeError("Failed to fetch")', False),
    ('{ ok: false, error: "permission denied" }', False),
    ('null', False),
    ('undefined', False),
]


def test_isCancelError_tells_a_cancel_from_an_error():
    """`isCancelError` tells a person's cancel apart from a real error
    (`web/projects.md` § 5), whichever shape the cancel arrives in
    (:data:`CANCEL_SHAPES`).  THE SILENT FAILURE: a shape missed is a
    cancel shown as an error; a shape too many is a failure dismissed as
    the person's choice."""
    out = _run("console.log(JSON.stringify([%s].map("
               "(e) => state.projects.isCancelError(e))));"
               % ", ".join(js for js, _ in CANCEL_SHAPES))
    assert dict(zip((js for js, _ in CANCEL_SHAPES), out)) == dict(
        CANCEL_SHAPES)


# ----- upload adds relPath --------------------------------------- #


class TestUploadEnvelopeShape:
    """`/api/files/upload` answers `{ok, path, size, mtime}`; `upload`
    adds `relPath` from the projects root, so its answer has the shape of
    a `writeFile`'s (`state.js`)."""

    def test_upload_computes_relPath_from_projects_root(self, app_at, tree):
        """`upload` adds `relPath` -- the path relative to the projects root -- which the
        server does not send.

        This is the wrapper DECIDING something the door cannot: `/api/files/upload`
        returns `{ok, path, size, mtime}` only, and `relPath` is computed here from
        the resolved projects root so an upload envelope has the same shape as a write
        envelope. A caller displaying `relPath` otherwise shows the absolute
        filesystem path -- a home directory and username in the UI -- or `undefined`.
        """
        out = _run(f"""
            state.setProjectsRoot({json.dumps(str(tree))});
            const r = await state.projects.upload(
                {json.dumps(str(tree / "P"))}, new File(["xyz"], "data.xyz"));
            console.log(JSON.stringify({{ r, answer: __answered[0] }}));
        """, app_at)
        answer = out["answer"]
        assert answer["ok"] is True and "relPath" not in answer, answer
        assert out["r"] == {**answer, "relPath": "P/data.xyz"}
        assert (tree / "P" / "data.xyz").read_text() == "xyz"


# ----- setShared when sessionStorage throws ----------------------- #


class TestSetSharedStorageFailure:
    """A sessionStorage write may throw (quota, private-mode SecurityError,
    storage denied).  `setShared` publishes its payload even then
    (`web/projects.md` § 5, the `showPreview` row): subscribers update,
    and only the cursor's survival of a reload is lost."""

    def test_setShared_swallows_sessionStorage_error_and_publishes(self):
        """A `sessionStorage` write that throws must not stop the selection being
        published -- and the publish carries the INTENDED payload, not the stale
        stored one.

        Private-browsing `SecurityError` and quota failures are real, and the slots
        are the browser's short-term storage (`web/projects.md` § 2). The subtle
        half is the second assertion: the publish reads its payload explicitly instead
        of re-reading storage, so after a failed write subscribers get `/after` rather
        than the `/before` still on disk. Without it, every tab quietly follows a
        selection the user has already moved away from.
        """
        out = _run('''
            // Make sessionStorage.setItem throw.  Use a wrapper
            // so we can flip it on/off.
            let storageBroken = false;
            const origSet = sessionStorage.setItem.bind(sessionStorage);
            global.sessionStorage.setItem = (k, v) => {
                if (storageBroken) {
                    const e = new Error("QuotaExceededError");
                    e.name = "QuotaExceededError";
                    throw e;
                }
                return origSet(k, v);
            };
            const calls = [];
            state.projects.onChange(p => calls.push({...p}));
            // First setShared works.
            const r1 = state.setShared("/before", "/before/x");
            // Now break storage.
            storageBroken = true;
            const r2 = state.setShared("/after", "/after/y");
            console.log(JSON.stringify({
                r1: r1,
                r2: r2,
                calls: calls,
            }));
        ''')
        assert out["r1"] == {"ok": True}
        assert out["r2"] == {"ok": True}    # MUST NOT throw
        # 3 calls: initial-fire on subscribe + 2 from setShared.
        assert len(out["calls"]) == 3
        # The post-failure publish carries the INTENDED payload
        # (sessionStorage didn't update, so without the explicit
        # payload the publish would carry "/before" + "/before/x").
        assert out["calls"][2] == {"dir": "/after", "file": "/after/y"}


# ----- Subscriber contract: throw on duplicate ------------------- #


class TestSubscribeDedupThrows:
    """`web/projects.md` § 2: "Registering the same callback twice is an
    error, on purpose" -- on each subscribe call, each of which keeps its
    own set.  It catches a forgotten unsubscribe or a double init at the
    call site."""

    @pytest.mark.parametrize("call", ["onChange", "onCommit"])
    def test_the_same_callback_twice_throws_naming_the_call(self, call):
        """Without it the callback fires twice per event and the tab does
        its work twice -- for a save pipeline or a schema reload a real
        double execution, not a cosmetic one."""
        out = _run(f'''
            const cb = (p) => {{}};
            state.projects.{call}(cb);
            let said = null;
            try {{ state.projects.{call}(cb); }}
            catch (e) {{ said = e.message; }}
            console.log(JSON.stringify(said));
        ''')
        assert out is not None, f"{call} took the same callback twice"
        assert call in out and "already registered" in out, out

    def test_different_callbacks_register_independently(self):
        """Two DIFFERENT callbacks (different fn objects) both register
        and both fire.  Only by-reference duplicates throw."""
        out = _run('''
            const calls = [];
            const cb1 = (p) => calls.push("cb1");
            const cb2 = (p) => calls.push("cb2");
            state.projects.onChange(cb1);
            state.projects.onChange(cb2);
            state.setShared("/d", "/d/f");
            console.log(JSON.stringify(calls));
        ''')
        # Initial fires (each subscribe fires once immediately) plus
        # the setShared fire.  Order: cb1 immediate, cb2 immediate,
        # then both fire from publishSelectionChange.
        assert out == ["cb1", "cb2", "cb1", "cb2"]

    def test_unsubscribe_then_resubscribe_succeeds(self):
        """The throw only fires for duplicate REGISTRATIONS.  After
        an unsub() the same callback can be re-registered."""
        out = _run('''
            const cb = (p) => {};
            const unsub = state.projects.onChange(cb);
            unsub();
            let threw = false;
            try { state.projects.onChange(cb); }
            catch (e) { threw = true; }
            console.log(JSON.stringify(threw));
        ''')
        assert out is False


# ----- Publish snapshot semantics (A2) --------------------------- #


class TestPublishSnapshotSemantics:
    """A publish takes its subscribers BEFORE iterating (`state.js`,
    `_publishToSet`): one registered DURING a publish loop fires only on
    later events -- it got the current state on its own subscribe call."""

    def test_new_subscriber_registered_during_publish_loop_does_not_fire_in_progress(self):
        """A subscriber registered from INSIDE a publish loop does not also receive that
        same event.

        Without the snapshot taken before iterating, the new subscriber is visited by
        the in-progress loop as well as by its own fire-once-on-subscribe, so it gets
        the same selection twice and the tab acts on it twice. Iterating a live set
        while callbacks add to it is also where a non-terminating publish lives. The
        expected sequence is spelled out in the body's comment.
        """
        out = _run('''
            const calls = [];
            const lateCb = (p) => calls.push({fn:"late", file:p.file});
            let lateRegistered = false;
            const earlyCb = (p) => {
                calls.push({fn:"early", file:p.file});
                // Register late ONLY while the publish loop is
                // running (when p.file is the post-setShared value).
                // Registering during early's initial-fire (p.file
                // === "") would put late in the subscriber set
                // BEFORE setShared's publish snapshots, which is a
                // different case.
                if (!lateRegistered && p.file === "/d/file.out") {
                    lateRegistered = true;
                    state.projects.onChange(lateCb);
                }
            };
            state.projects.onChange(earlyCb);
            state.setShared("/d", "/d/file.out");
            console.log(JSON.stringify(calls));
        ''')
        # Sequence:
        #   1. earlyCb subscribes -> fires immediately with "".
        #   2. setShared updates sessionStorage + calls publish.
        #      Snapshot taken = [earlyCb] (lateCb not yet registered).
        #   3. earlyCb fires from snapshot with /d/file.out.
        #      Inside its callback, registers lateCb.
        #      lateCb's fire-once-immediately fires with current
        #      state (/d/file.out).
        #   4. Publish loop continues; snapshot has only earlyCb,
        #      so lateCb is NOT visited by the in-progress loop.
        # Total entries: 3.  NOT 4 (no double-fire of lateCb).
        assert out == [
            {"fn": "early", "file": ""},
            {"fn": "early", "file": "/d/file.out"},
            {"fn": "late",  "file": "/d/file.out"},
        ]

    def test_unsubscribed_during_callback_is_skipped(self):
        """A subscriber that unsubscribes itself during the publish
        loop IS skipped on subsequent iterations of the same loop --
        Set.delete on the live set takes effect immediately, and the
        snapshot iteration checks membership before invoking each
        entry."""
        out = _run('''
            const calls = [];
            const cb1 = (p) => calls.push("cb1");
            let unsub2;
            const cb2 = (p) => {
                calls.push("cb2");
                unsub2();   // unsubscribe self
            };
            const cb3 = (p) => calls.push("cb3");
            state.projects.onChange(cb1);
            unsub2 = state.projects.onChange(cb2);
            state.projects.onChange(cb3);
            // Pre-publish: each subscribe fired once.
            calls.length = 0;
            state.setShared("/d", "/d/x");
            console.log(JSON.stringify(calls));
        ''')
        # Publish snapshot is [cb1, cb2, cb3].  cb1 fires, cb2 fires +
        # unsubs, cb3 fires (cb2 already gone but cb3 still in set).
        assert out == ["cb1", "cb2", "cb3"]


# ----- whenProjectsRoot ------------------------------------------ #


class TestWhenProjectsRoot:
    """`whenProjectsRoot()`: the projects root's answer, once the sidebar's
    start has one -- the root, or the words it gives when it cannot resolve
    it -- and the one way to wait for it (`web/projects.md` § 5).  The start
    is driven through the two calls the sidebar makes; what the sidebar
    passes and what a picker draws are not this module's."""

    @pytest.mark.parametrize("start, answer", [
        ('state.setProjectsRoot("/home/u/projects")',
         {"ok": True, "root": "/home/u/projects"}),
        ('state.projectsRootUnavailable("the start\'s words")',
         {"ok": False, "error": "the start's words"}),
    ], ids=["resolved", "not resolved"])
    def test_asked_before_the_start_answers_it_is_answered_when_it_does(
            self, start, answer):
        """Asked first, answered after: nothing before the start says, then
        its answer.

        MUTATION THIS MUST FAIL AGAINST: an answer given early (a folder
        picker opening on an empty root and joining paths onto it), or never
        -- a start that cannot resolve the root settling nothing, so a picker
        asked meanwhile waits for ever."""
        out = _run(f'''
            const asked = state.whenProjectsRoot();
            let early = null;
            asked.then(a => {{ early = a; }});
            await new Promise(r => setTimeout(r, 0));
            const before = early;
            {start};
            console.log(JSON.stringify({{ before, answer: await asked }}));
        ''')
        assert out == {"before": None, "answer": answer}

    def test_asked_after_the_start_has_answered_it_is_answered_at_once(self):
        """A tab that asks after the sidebar's start has resolved the root
        still gets it -- the answer already given, nothing waited for.

        MUTATION THIS MUST FAIL AGAINST: an answer only for whoever asked
        before the start, so a late asker waits for ever."""
        out = _run('''
            state.setProjectsRoot("/p");
            console.log(JSON.stringify(await state.projects.whenProjectsRoot()));
        ''')
        assert out == {"ok": True, "root": "/p"}
