"""Unit tests for the public projects.* mutator surface (sidebar
gap M4, #175).

Promotes ``readFile / createProject / mkdir / deleteEntry / upload /
navigateTo`` onto ``window.molbuilder.projects.*`` so external
callers (the /results tab-level file-picker dropdown at
``lib/results/file-picker.js``, future programmatic file
managers) don't need to import ``./api.js`` directly.

Each wrapper auto-fires a sidebar refresh on success so the tree
stays in sync.  These tests pin:
  * Each method is present + callable on projects.*.
  * Success path delegates to the matching apiX function +
    triggers refreshHandler with the right argument.
  * Failure path passes through the apiX response verbatim AND
    does NOT trigger refresh (the mutation didn't land).
  * navigateTo is a thin shim over setShared (lock guard inherited).
  * Refresh-handler failure is swallowed (a flaky listing fetch
    shouldn't fail an otherwise-successful mutation).
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "molbuilder/web/static/lib/projects/state.js"


def _run_node(snippet: str) -> object:
    node = shutil.which("node")
    if node is None:
        pytest.skip("node not available")
    module_url = MODULE.resolve().as_uri()
    bootstrap = f"""
        const _store = new Map();
        global.sessionStorage = {{
            getItem:    k => _store.has(k) ? _store.get(k) : null,
            setItem:    (k, v) => _store.set(k, String(v)),
            removeItem: k => _store.delete(k),
            clear:      () => _store.clear(),
        }};
        global.window = global;
        global.document = {{
            addEventListener: () => {{}},
            getElementById:   () => null,
        }};
        global.navigator = {{ language: "en-US" }};
        // Default fetch must throw if called -- the wrappers should
        // route through the captured-fetch the tests install per-case.
        global.fetch = () => {{
            throw new Error(
                "test must override global.fetch; api.js called "
                + "the default stub"
            );
        }};
        global.FormData = function () {{ this.append = () => {{}}; }};
        const statePromise = import("{module_url}");
        statePromise.then(async (state) => {{
            const pageBusy = (await import("{(ROOT / 'molbuilder/web/static/lib/page-busy.js').resolve().as_uri()}")).pageBusy;
            {snippet}
        }}).catch(err => {{
            console.log(JSON.stringify({{
                __test_unexpected_throw: true,
                message: err && err.message ? err.message : String(err),
                stack:   err && err.stack ? err.stack : null,
            }}));
        }});
    """
    proc = subprocess.run(
        [node, "--input-type=commonjs", "-e", bootstrap],
        capture_output=True,
        text=True,
        timeout=15,
    )
    if proc.returncode != 0:
        pytest.fail(
            f"node exited {proc.returncode}\n"
            f"stderr:\n{proc.stderr}\n"
            f"stdout:\n{proc.stdout}"
        )
    last_line = proc.stdout.strip().splitlines()[-1]
    out = json.loads(last_line)
    # The throw-marker check only applies when the snippet emits a
    # JSON object (the typical case).  Tests that legitimately emit
    # a scalar / array bypass the check.
    if isinstance(out, dict):
        assert "__test_unexpected_throw" not in out, (
            "module threw: " + str(out)
        )
    return out


# ----- Surface presence ------------------------------------------- #


class TestSurfacePresence:

    def test_all_M4_methods_are_callable(self):
        out = _run_node('''
            const p = state.projects;
            console.log(JSON.stringify({
                readFile:                   typeof p.readFile,
                readRange:                  typeof p.readRange,
                createProject:              typeof p.createProject,
                mkdir:                      typeof p.mkdir,
                deleteEntry:                typeof p.deleteEntry,
                rename:                     typeof p.rename,
                upload:                     typeof p.upload,
                setShared:                  typeof p.setShared,
                navigateTo:                 typeof p.navigateTo,
                onProjectsRootResolved:     typeof p.onProjectsRootResolved,
                onCommit:                   typeof p.onCommit,
                safeSave:                   typeof p.safeSave,
                isCancelError:              typeof p.isCancelError,
            }));
        ''')
        for fn in ("readFile", "readRange", "createProject", "mkdir",
                   "deleteEntry", "rename", "upload", "setShared",
                   "navigateTo", "onProjectsRootResolved",
                   "onCommit",
                   "safeSave", "isCancelError"):
            assert out[fn] == "function", f"missing public method: {fn}"


# ----- onCommit + publishCommit (B.5.2) --------------------------- #


class TestOnCommit:
    """The universal commit-event subscription per docs/web/
    tabs.md.  Sidebar single-click = preview
    (fires onChange).  Double-click = commit (fires onCommit
    only).  This keeps subscribers from mistaking a casual
    browse-click for a tab-level action."""

    def test_onCommit_does_NOT_fire_on_subscribe(self):
        """Commits are discrete events, not states.  Firing on
        subscribe would over-react the first time a tab mounts;
        the published payload would be the empty initial state
        and subscribers would think the user committed nothing."""
        out = _run_node('''
            const calls = [];
            state.projects.onCommit(p => calls.push(p));
            console.log(JSON.stringify({
                callsAfterSubscribe: calls,
            }));
        ''')
        assert out["callsAfterSubscribe"] == []

    def test_publishCommit_fires_subscribers(self):
        out = _run_node('''
            const calls = [];
            state.projects.onCommit(p => calls.push(p));
            state.publishCommit("/p/proj", "/p/proj/water.xyz");
            console.log(JSON.stringify(calls));
        ''')
        assert out == [{"dir": "/p/proj", "file": "/p/proj/water.xyz"}]

    def test_publishCommit_also_updates_global_pick_via_setShared(self):
        """publishCommit's contract: it fires the commit subscribers
        AND updates the global pick (via setShared) so a
        cross-tab handoff via sessionStorage works without
        publishers needing to call both.  Mirrors the real-user
        flow: dblclick's first click sets setShared, the second
        fires publishCommit; programmatic callers shouldn't have
        to know that detail.

        Practical consequence: publishCommit fires BOTH onChange
        (preview/candidate update) AND onCommit (commit signal)."""
        out = _run_node('''
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
        out = _run_node('''
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

    def test_publishCommit_fires_setShared_when_pick_changes(self):
        """The flip side: publishCommit with a NEW pick (different
        from sessionStorage) DOES need to update setShared so
        cross-tab handoff via sessionStorage works."""
        out = _run_node('''
            state.setShared("/p", "/p/old.xyz");
            const onChangeCalls = [];
            state.projects.onChange(p => onChangeCalls.push(p));
            onChangeCalls.length = 0;
            // Different file — must update sessionStorage.
            state.publishCommit("/p", "/p/new.xyz");
            console.log(JSON.stringify({
                onChange:    onChangeCalls,
                currentFile: state.projects.getCurrentFile(),
            }));
        ''')
        assert out["onChange"] == [{"dir": "/p", "file": "/p/new.xyz"}]
        assert out["currentFile"] == "/p/new.xyz"

    def test_setShared_does_NOT_fire_onCommit_subscribers(self):
        """Single-click → setShared → onChange.  publishCommit is
        separate; setShared MUST NOT also fire onCommit, or every
        sidebar click would commit and the candidate-only model
        collapses."""
        out = _run_node('''
            const onCommitCalls = [];
            state.projects.onCommit(p => onCommitCalls.push(p));
            state.setShared("/p", "/p/y.xyz");
            console.log(JSON.stringify(onCommitCalls));
        ''')
        assert out == []

    def test_unsubscribe_works(self):
        out = _run_node('''
            const calls = [];
            const unsub = state.projects.onCommit(p => calls.push(p));
            unsub();
            state.publishCommit("/p", "/p/x.xyz");
            console.log(JSON.stringify(calls));
        ''')
        assert out == []

    def test_duplicate_onCommit_throws(self):
        """Same subscriber-dedup contract every other onChange-style
        API enforces (per design § 5.5 + § 11.5)."""
        out = _run_node('''
            const cb = () => {};
            state.projects.onCommit(cb);
            let threw = false;
            try { state.projects.onCommit(cb); }
            catch (e) { threw = true; }
            console.log(JSON.stringify(threw));
        ''')
        assert out is True


# ----- readFile --------------------------------------------------- #


class TestReadFile:

    def test_delegates_to_apiRead(self):
        out = _run_node('''
            let capturedUrl = null;
            global.fetch = async (url) => {
                capturedUrl = url;
                return {
                    ok: true,
                    status: 200,
                    json: async () => ({
                        ok: true, text: "hello", mtime: 42
                    }),
                };
            };
            const r = await state.projects.readFile("/projects/job/a.xyz");
            console.log(JSON.stringify({
                envelope: r,
                url:      capturedUrl,
            }));
        ''')
        assert out["envelope"] == {"ok": True, "text": "hello", "mtime": 42}
        assert "/api/files/read" in out["url"]
        # Path is URL-encoded -- ``/`` becomes ``%2F``.
        assert "%2Fprojects%2Fjob%2Fa.xyz" in out["url"]

    def test_failure_passes_through(self):
        out = _run_node('''
            global.fetch = async () => ({
                ok: false,
                status: 404,
                json: async () => ({ ok: false, error: "no such file" }),
            });
            const r = await state.projects.readFile("/missing");
            console.log(JSON.stringify(r));
        ''')
        assert out == {"ok": False, "error": "no such file"}


# ----- readRange (#189, 2026-06-02) ------------------------------- #


class TestReadRange:
    """The v2 paginated source inspector goes through
    ``projects.readRange`` instead of raw ``fetch`` on
    ``/api/files/read_range``.  These tests pin the wrapper's
    URL composition, envelope passthrough, and abort plumbing so
    a future refactor can't quietly drop one of them."""

    def test_default_offset_and_max_bytes_omitted_from_url(self):
        """When ``offset`` and ``maxBytes`` are undefined, the wrapper
        emits the bare ``?path=...`` URL and lets the server apply
        its default 256 KB window starting at byte 0."""
        out = _run_node('''
            let capturedUrl = null;
            global.fetch = async (url) => {
                capturedUrl = url;
                return {
                    ok: true, status: 200,
                    json: async () => ({
                        ok: true, path: "/p/big.log", offset: 0,
                        length: 262144, file_size: 1000000,
                        mtime: 42, text: "first chunk", eof: false,
                    }),
                };
            };
            const r = await state.projects.readRange("/p/big.log");
            console.log(JSON.stringify({envelope: r, url: capturedUrl}));
        ''')
        assert out["envelope"]["ok"] is True
        assert out["envelope"]["length"] == 262144
        assert "/api/files/read_range" in out["url"]
        assert "%2Fp%2Fbig.log" in out["url"]
        # No offset / max_bytes params when they weren't passed.
        assert "offset=" not in out["url"]
        assert "max_bytes=" not in out["url"]

    def test_explicit_offset_and_max_bytes_in_url(self):
        out = _run_node('''
            let capturedUrl = null;
            global.fetch = async (url) => {
                capturedUrl = url;
                return {
                    ok: true, status: 200,
                    json: async () => ({
                        ok: true, path: "/p/big.log", offset: 524288,
                        length: 262144, file_size: 1000000,
                        mtime: 42, text: "second chunk", eof: false,
                    }),
                };
            };
            const r = await state.projects.readRange(
                "/p/big.log", 524288, 262144);
            console.log(JSON.stringify({envelope: r, url: capturedUrl}));
        ''')
        assert out["envelope"]["offset"] == 524288
        assert "offset=524288" in out["url"]
        assert "max_bytes=262144" in out["url"]

    def test_negative_offset_for_tail_read(self):
        """``offset = -N`` reads the last N bytes; URL-encoding must
        preserve the minus sign (encodeURIComponent leaves ``-``
        alone but the wrapper still has to pass the raw value)."""
        out = _run_node('''
            let capturedUrl = null;
            global.fetch = async (url) => {
                capturedUrl = url;
                return {
                    ok: true, status: 200,
                    json: async () => ({
                        ok: true, path: "/p/big.log", offset: 737856,
                        length: 262144, file_size: 1000000,
                        mtime: 42, text: "tail chunk", eof: true,
                    }),
                };
            };
            await state.projects.readRange("/p/big.log", -262144, 262144);
            console.log(JSON.stringify({url: capturedUrl}));
        ''')
        assert "offset=-262144" in out["url"]

    def test_server_error_envelope_passes_through(self):
        out = _run_node('''
            global.fetch = async () => ({
                ok: false, status: 400,
                json: async () => ({
                    ok: false,
                    error: "offset past end of file",
                }),
            });
            const r = await state.projects.readRange(
                "/p/small.txt", 999999);
            console.log(JSON.stringify(r));
        ''')
        assert out == {"ok": False, "error": "offset past end of file"}

    def test_network_drop_returns_uniform_envelope(self):
        """Same shape as every other projects.* wrapper -- network
        drop must NOT throw."""
        out = _run_node('''
            global.fetch = async () => {
                throw new TypeError("Failed to fetch");
            };
            const r = await state.projects.readRange("/p/x.log");
            console.log(JSON.stringify(r));
        ''')
        assert out["ok"] is False
        assert "network error" in out["error"]

    def test_abort_signal_threads_into_fetch(self):
        out = _run_node('''
            let capturedSignal = null;
            global.fetch = async (url, init) => {
                capturedSignal = init && init.signal;
                return {
                    ok: true, status: 200,
                    json: async () => ({ok: true, text: "x"}),
                };
            };
            const ac = new AbortController();
            await state.projects.readRange(
                "/p/x.log", 0, 1024, {signal: ac.signal});
            console.log(JSON.stringify({
                signal_present: capturedSignal !== null
                              && capturedSignal !== undefined,
            }));
        ''')
        assert out["signal_present"] is True


# ----- createProject / mkdir / upload: refresh on success --------- #


class TestRefreshOnSuccess:

    def test_createProject_refreshes_root(self):
        """createProject success -> refreshHandler(projects_root)."""
        out = _run_node('''
            const refreshArgs = [];
            state.setRefreshHandler(async (dir) => {
                refreshArgs.push(dir);
            });
            // setProjectsRoot is module-private; we can prime it by
            // calling the public setter (exposed for the sidebar init).
            state.setProjectsRoot("/projects");
            global.fetch = async () => ({
                ok: true,
                status: 200,
                json: async () => ({ ok: true, path: "/projects/myjob" }),
            });
            const r = await state.projects.createProject("myjob");
            console.log(JSON.stringify({
                envelope:     r,
                refreshArgs:  refreshArgs,
            }));
        ''')
        assert out["envelope"]["ok"] is True
        # Refresh was called with the projects root.
        assert out["refreshArgs"] == ["/projects"]

    def test_mkdir_refreshes_parent(self):
        out = _run_node('''
            const refreshArgs = [];
            state.setRefreshHandler(async (dir) => refreshArgs.push(dir));
            global.fetch = async () => ({
                ok: true,
                status: 200,
                json: async () => ({ ok: true, path: "/p/parent/new" }),
            });
            await state.projects.mkdir("/p/parent", "new");
            console.log(JSON.stringify(refreshArgs));
        ''')
        assert out == ["/p/parent"]

    def test_upload_refreshes_target_dir(self):
        out = _run_node('''
            const refreshArgs = [];
            state.setRefreshHandler(async (dir) => refreshArgs.push(dir));
            global.fetch = async () => ({
                ok: true,
                status: 200,
                json: async () => ({ ok: true, path: "/p/dest/f.xyz" }),
            });
            await state.projects.upload("/p/dest", { name: "f.xyz" });
            console.log(JSON.stringify(refreshArgs));
        ''')
        assert out == ["/p/dest"]

    def test_rename_refreshes_parent_dir(self):
        out = _run_node('''
            const refreshArgs = [];
            state.setRefreshHandler(async (dir) => refreshArgs.push(dir));
            global.fetch = async () => ({
                ok: true, status: 200,
                json: async () => ({ ok: true, path: "/p/parent/new.xyz" }),
            });
            await state.projects.rename("/p/parent/old.xyz", "new.xyz");
            console.log(JSON.stringify(refreshArgs));
        ''')
        assert out == ["/p/parent"]

    def test_rename_409_does_not_refresh(self):
        """A 409 destination-conflict must NOT trigger refresh — the
        source path didn't change and the destination was already
        listed in the sidebar."""
        out = _run_node('''
            const refreshArgs = [];
            state.setRefreshHandler(async (dir) => refreshArgs.push(dir));
            global.fetch = async () => ({
                ok: false, status: 409,
                json: async () => ({
                    ok: false, error: "destination already exists",
                }),
            });
            const r = await state.projects.rename("/p/old.xyz", "exists.xyz");
            console.log(JSON.stringify({
                envelope:    r,
                refreshArgs: refreshArgs,
            }));
        ''')
        assert out["envelope"]["ok"] is False
        assert "destination already exists" in out["envelope"]["error"]
        assert out["refreshArgs"] == []

    def test_deleteEntry_refreshes_parent_dir(self):
        out = _run_node('''
            const refreshArgs = [];
            state.setRefreshHandler(async (dir) => refreshArgs.push(dir));
            global.fetch = async () => ({
                ok: true,
                status: 200,
                json: async () => ({ ok: true }),
            });
            await state.projects.deleteEntry("/p/parent/file.xyz", false);
            console.log(JSON.stringify(refreshArgs));
        ''')
        assert out == ["/p/parent"]


# ----- Failure path: no refresh ----------------------------------- #


class TestNoRefreshOnFailure:

    def test_mkdir_failure_does_not_refresh(self):
        """A failed mutation must NOT trigger the listing refresh --
        otherwise a stale 4xx (e.g. 409 conflict) would silently
        force the sidebar to re-fetch the parent for no reason."""
        out = _run_node('''
            const refreshArgs = [];
            state.setRefreshHandler(async (dir) => refreshArgs.push(dir));
            global.fetch = async () => ({
                ok: false,
                status: 409,
                json: async () => ({ ok: false, error: "already exists" }),
            });
            const r = await state.projects.mkdir("/p", "dup");
            console.log(JSON.stringify({
                envelope:    r,
                refreshArgs: refreshArgs,
            }));
        ''')
        assert out["envelope"]["ok"] is False
        assert out["refreshArgs"] == []

    def test_upload_network_failure_does_not_refresh(self):
        out = _run_node('''
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

    def test_mkdir_success_with_failing_refresh_still_succeeds(self):
        """If the post-success refresh itself throws (e.g. transient
        listing fetch failure), the mutation envelope should still
        report success -- the file/dir DID land on disk; the user
        can refresh manually."""
        out = _run_node('''
            state.setRefreshHandler(async () => {
                throw new Error("listing fetch failed");
            });
            global.fetch = async () => ({
                ok: true,
                status: 200,
                json: async () => ({ ok: true, path: "/p/new" }),
            });
            const r = await state.projects.mkdir("/p", "new");
            console.log(JSON.stringify(r));
        ''')
        assert out["ok"] is True
        assert out["path"] == "/p/new"


# ----- navigateTo per design § C7 (openDir-aliased) -------------- #


class TestNavigateTo:
    """navigateTo per design § C7 is the openDir-aliased async lister:
    takes ``(absPath, opts?)``, fetches the directory listing, updates
    the cursor, and returns ``{ok, path, entries}`` (or ``{ok:false,
    error}`` on failure).  Wired by ``setNavigateToImpl`` at sidebar
    init; falls back to a clean "unavailable" envelope when init
    hasn't run yet (so tabs that subscribe + immediately call it
    don't throw)."""

    def test_unavailable_before_setNavigateToImpl_wires_it(self):
        """Without setNavigateToImpl being called, navigateTo returns
        the documented "unavailable" envelope -- NOT throws.  Pins
        the fail-safe contract for tabs that race against sidebar
        init."""
        out = _run_node('''
            // sidebar's init NOT run; setNavigateToImpl never called.
            const r = await state.projects.navigateTo("/d");
            console.log(JSON.stringify(r));
        ''')
        assert out["ok"] is False
        assert "unavailable" in out["error"]
        assert "sidebar not initialised" in out["error"]

    def test_after_wiring_delegates_to_impl(self):
        """When setNavigateToImpl(fn) has been called, navigateTo
        calls fn(absPath, opts) and returns its envelope verbatim."""
        out = _run_node('''
            let capturedArgs = null;
            state.setNavigateToImpl(async (absPath, opts) => {
                capturedArgs = { absPath: absPath, opts: opts };
                return {
                    ok:      true,
                    path:    absPath,
                    entries: [{ name: "a.out", kind: "file" }],
                };
            });
            const r = await state.projects.navigateTo("/p/sub", {
                signal: "fake-signal-token",
            });
            console.log(JSON.stringify({
                envelope:     r,
                argsAbsPath:  capturedArgs.absPath,
                argsOptsSig:  capturedArgs.opts.signal,
            }));
        ''')
        assert out["envelope"] == {
            "ok": True,
            "path": "/p/sub",
            "entries": [{"name": "a.out", "kind": "file"}],
        }
        assert out["argsAbsPath"] == "/p/sub"
        assert out["argsOptsSig"] == "fake-signal-token"

    def test_impl_failure_envelope_returned_verbatim(self):
        """openDir-side failure ({ok:false, error}) flows through
        navigateTo unchanged."""
        out = _run_node('''
            state.setNavigateToImpl(async () => ({
                ok:    false,
                error: "Failed to list directory.",
            }));
            const r = await state.projects.navigateTo("/missing");
            console.log(JSON.stringify(r));
        ''')
        assert out == {"ok": False, "error": "Failed to list directory."}

    def test_busy_navigateTo_rejects_without_calling_impl(self):
        """Per § 8.5, navigateTo MUST check pageBusy.isClaimed() and
        refuse while the fence is held -- without calling the
        underlying impl.  The impl (openDir) is intentionally NOT
        guarded because it doubles as the refreshHandler that runs
        mid-Save-pipeline; the public-surface wrapper enforces § 8.5."""
        out = _run_node('''
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


# ----- readCurrentFile envelope (design § C3) -------------------- #


class TestReadCurrentFileEnvelope:
    """Per design § C3: ReadResult = ReadOk | ReadErr | null.
    null only for the no-file-selected case.  ReadOk is
    {ok:true, path, text}; ReadErr is {ok:false, error}."""

    def test_no_file_selected_returns_null(self):
        out = _run_node('''
            const r = await state.projects.readCurrentFile();
            console.log(JSON.stringify(r));
        ''')
        assert out is None

    def test_success_returns_envelope_with_ok(self):
        out = _run_node('''
            sessionStorage.setItem("molbuilder.current_file", "/p/f.xyz");
            global.fetch = async () => ({
                ok: true, status: 200,
                json: async () => ({
                    ok: true, path: "/p/f.xyz", text: "hello",
                }),
            });
            const r = await state.projects.readCurrentFile();
            console.log(JSON.stringify(r));
        ''')
        assert out == {"ok": True, "path": "/p/f.xyz", "text": "hello"}

    def test_failure_returns_envelope_not_null(self):
        out = _run_node('''
            sessionStorage.setItem("molbuilder.current_file", "/p/f.xyz");
            global.fetch = async () => ({
                ok: false, status: 404,
                json: async () => ({ ok: false, error: "not found" }),
            });
            const r = await state.projects.readCurrentFile();
            console.log(JSON.stringify(r));
        ''')
        assert out["ok"] is False
        assert "not found" in out["error"]


# ----- refresh envelope (design § C6) ---------------------------- #


class TestRefreshEnvelope:
    """Per design § C6: refresh returns {ok:true} | {ok:false, error}.
    Previously returned undefined on every path, violating Principle 6."""

    def test_no_current_dir_returns_envelope_not_undefined(self):
        out = _run_node('''
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
        out = _run_node('''
            sessionStorage.setItem("molbuilder.current_dir", "/p");
            const r = await state.projects.refresh();
            console.log(JSON.stringify(r));
        ''')
        assert out["ok"] is False
        assert "no refresh handler" in out["error"]

    def test_success_returns_ok_envelope(self):
        out = _run_node('''
            sessionStorage.setItem("molbuilder.current_dir", "/p");
            state.setRefreshHandler(async (dir) => { /* success */ });
            const r = await state.projects.refresh();
            console.log(JSON.stringify(r));
        ''')
        assert out == {"ok": True}

    def test_handler_throws_returns_error_envelope(self):
        out = _run_node('''
            sessionStorage.setItem("molbuilder.current_dir", "/p");
            state.setRefreshHandler(async () => {
                throw new Error("listing failed");
            });
            const r = await state.projects.refresh();
            console.log(JSON.stringify(r));
        ''')
        assert out["ok"] is False
        assert "refresh failed" in out["error"]
        assert "listing failed" in out["error"]


# ----- writeFile preserves actual_mtime on 409 (design § C4) ----- #


class TestWriteFileEdgeFields:
    """Per design § C4 WriteErr includes actual_mtime? on a 409
    edit-conflict.  Previously writeFile destructured only `error`
    and dropped actual_mtime, so tabs couldn't distinguish edit-
    conflict programmatically per § 6.2."""

    def test_writeFile_preserves_actual_mtime_on_409(self):
        out = _run_node('''
            global.fetch = async () => ({
                ok: false, status: 409,
                json: async () => ({
                    ok: false, error: "edit conflict",
                    actual_mtime: 1717174420.5,
                }),
            });
            const r = await state.projects.writeFile("/p/f", "text");
            console.log(JSON.stringify(r));
        ''')
        assert out["ok"] is False
        assert out["error"] == "edit conflict"
        assert out["actual_mtime"] == 1717174420.5

    def test_writeFile_preserves_aborted_flag(self):
        """AbortError envelope carries aborted:true; writeFile must
        not drop the flag."""
        out = _run_node('''
            global.fetch = async () => {
                const err = new Error("aborted");
                err.name = "AbortError";
                throw err;
            };
            const r = await state.projects.writeFile("/p/f", "text");
            console.log(JSON.stringify(r));
        ''')
        assert out["ok"] is False
        assert out["aborted"] is True


class TestSafeSave:
    """``safeSave`` wraps ``saveToWorkspace`` so the three terminal
    states are distinct: null (no current dir), {cancelled:true}
    (user cancel), {ok:true|false, ...} (write outcome).  Tests
    here cover the shapes the three real callers (SIESTA + PySCF
    save in viewer.js, Spectra save in spectra/core.js) actually
    consume.  Tests for theoretical caller mistakes were removed
    after the 8-audit-round cleanup: callers are reviewed for
    contract compliance, the helper is not hardened against
    impossible inputs."""

    def test_safeSave_folds_abort_envelope_to_cancelled(self):
        """The only fold safeSave does: ``{ok:false, aborted:true}``
        from saveToWorkspace → ``{ok:false, cancelled:true}`` for
        the call site to branch on.  All three real callers
        check ``r.cancelled`` first."""
        out = _run_node('''
            global.fetch = async () => {
                const err = new Error("aborted");
                err.name = "AbortError";
                throw err;
            };
            global.sessionStorage = {
                _v: {"molbuilder.current_dir": "/projects/proj1"},
                getItem(k) { return this._v[k] || null; },
                setItem(k, v) { this._v[k] = v; },
            };
            state.setProjectsRoot("/projects");
            const ac = new AbortController();
            const r = await state.projects.safeSave("text", "f.xyz",
                { overwrite: true, signal: ac.signal });
            console.log(JSON.stringify(r));
        ''')
        assert out["ok"] is False
        assert out["cancelled"] is True

    def test_safeSave_signal_propagates_to_fetch(self):
        """Pin that ``opts.signal`` reaches fetch end-to-end.  A
        future refactor that drops the signal would deadlock this
        test (the mock fetch only resolves when the signal fires)."""
        out = _run_node('''
            global.fetch = async (url, opts) => {
                return new Promise((resolve, reject) => {
                    if (opts && opts.signal) {
                        if (opts.signal.aborted) {
                            const e = new Error("aborted");
                            e.name = "AbortError";
                            reject(e);
                            return;
                        }
                        opts.signal.addEventListener("abort", () => {
                            const e = new Error("aborted");
                            e.name = "AbortError";
                            reject(e);
                        });
                    }
                    // No signal → no resolution; deadlock by design.
                });
            };
            global.sessionStorage = {
                _v: {"molbuilder.current_dir": "/projects/proj1"},
                getItem(k) { return this._v[k] || null; },
                setItem(k, v) { this._v[k] = v; },
            };
            state.setProjectsRoot("/projects");
            const ac = new AbortController();
            const p = state.projects.safeSave("text", "f.xyz",
                { signal: ac.signal });
            await new Promise(r => setImmediate(r));
            ac.abort();
            const r = await p;
            console.log(JSON.stringify(r));
        ''')
        assert out["ok"] is False
        assert out["cancelled"] is True

    def test_safeSave_returns_null_on_no_current_dir(self):
        """The null backstop — callers gate on this before calling
        in practice; the helper preserves the saveToWorkspace
        contract."""
        out = _run_node('''
            global.sessionStorage = {
                getItem() { return null; },
                setItem() {},
            };
            state.setProjectsRoot("/projects");
            const r = await state.projects.safeSave("text", "f.xyz");
            console.log(JSON.stringify(r));
        ''')
        assert out is None

    def test_safeSave_returns_null_at_projects_root(self):
        """current_dir IS the projects/ root → null (no writeable
        subdir).  Matches saveToWorkspace behaviour."""
        out = _run_node('''
            global.sessionStorage = {
                _v: {"molbuilder.current_dir": "/projects"},
                getItem(k) { return this._v[k] || null; },
                setItem(k, v) { this._v[k] = v; },
            };
            state.setProjectsRoot("/projects");
            const r = await state.projects.safeSave("text", "f.xyz");
            console.log(JSON.stringify(r));
        ''')
        assert out is None

    def test_safeSave_passes_through_real_failure(self):
        """Real 409 stays ``{ok:false, error:"..."}``.  Callers
        check ``r.cancelled`` THEN ``!r.ok`` so the failure path
        gets the real error text."""
        out = _run_node('''
            global.fetch = async () => ({
                ok: false, status: 409,
                json: async () => ({
                    ok: false, error: "file already exists: 'f.xyz'",
                }),
            });
            global.sessionStorage = {
                _v: {"molbuilder.current_dir": "/projects/proj1"},
                getItem(k) { return this._v[k] || null; },
                setItem(k, v) { this._v[k] = v; },
            };
            state.setProjectsRoot("/projects");
            const r = await state.projects.safeSave("text", "f.xyz");
            console.log(JSON.stringify(r));
        ''')
        assert out["ok"] is False
        assert "already exists" in out["error"]
        assert "cancelled" not in out

    def test_safeSave_success_passthrough(self):
        """Success envelope flows through unchanged."""
        out = _run_node('''
            global.fetch = async () => ({
                ok: true, status: 200,
                json: async () => ({
                    ok: true,
                    path: "/projects/proj1/f.xyz",
                    size: 7, mtime: 1717174420.5,
                }),
            });
            global.sessionStorage = {
                _v: {"molbuilder.current_dir": "/projects/proj1"},
                getItem(k) { return this._v[k] || null; },
                setItem(k, v) { this._v[k] = v; },
            };
            state.setProjectsRoot("/projects");
            const r = await state.projects.safeSave("text", "f.xyz");
            console.log(JSON.stringify(r));
        ''')
        assert out["ok"] is True
        assert out["path"] == "/projects/proj1/f.xyz"
        assert "cancelled" not in out


class TestIsCancelError:
    """Predicate accepting every shape the cancellation contract
    arrives in across the codebase:

      - DOMException ``name === "AbortError"`` — thrown by fetch
        when an AbortSignal aborts (the 4 wrapper-install / pseudo-
        install catches in SIESTA/PySCF/Spectra pipelines).
      - ``{aborted: true}`` — the raw envelope shape returned by
        api.js / writeFile / saveToWorkspace.
      - ``{cancelled: true}`` — the post-fold envelope shape
        returned by safeSave.
      - ``{code: "aborted"}`` — the ViewerError shape produced by
        mol-viewer-embed.

    Each shape exists in production code paths; the predicate
    centralises the match so a future change to the cancellation
    contract has ONE place to touch."""

    def test_isCancelError_matches_abort_error_name(self):
        """The shape fetch throws on abort."""
        out = _run_node('''
            const e = new Error("aborted");
            e.name = "AbortError";
            console.log(JSON.stringify({
                hit: state.projects.isCancelError(e),
            }));
        ''')
        assert out["hit"] is True

    def test_isCancelError_matches_aborted_flag(self):
        """The raw api.js / writeFile envelope shape."""
        out = _run_node('''
            console.log(JSON.stringify({
                hit: state.projects.isCancelError(
                    { ok: false, aborted: true, error: "x" }),
            }));
        ''')
        assert out["hit"] is True

    def test_isCancelError_matches_aborted_code(self):
        """The ViewerError shape produced by mol-viewer-embed."""
        out = _run_node('''
            console.log(JSON.stringify({
                hit: state.projects.isCancelError(
                    { code: "aborted", message: "x" }),
            }));
        ''')
        assert out["hit"] is True

    def test_isCancelError_matches_safeSave_cancelled_envelope(self):
        """The post-fold envelope shape returned by safeSave.  A
        maintainer using both safeSave AND isCancelError naturally
        would pass a safeSave result here; pinning that the match
        works avoids the footgun where mixed use leaks cancels as
        errors."""
        out = _run_node('''
            console.log(JSON.stringify({
                hit: state.projects.isCancelError(
                    { ok: false, cancelled: true }),
            }));
        ''')
        assert out["hit"] is True

    def test_isCancelError_does_NOT_match_disposed(self):
        """``code: "disposed"`` is a distinct lifecycle event (host
        tore an embed down), not a user-initiated cancel.  Pinning
        the exclusion so a future widening is a conscious decision,
        not accidental drift."""
        out = _run_node('''
            console.log(JSON.stringify({
                hit: state.projects.isCancelError(
                    { code: "disposed", message: "x" }),
            }));
        ''')
        assert out["hit"] is False

    def test_isCancelError_rejects_null_and_other_errors(self):
        """Null/undefined guard + non-cancel error rejection."""
        out = _run_node('''
            console.log(JSON.stringify({
                forNull:    state.projects.isCancelError(null),
                forUndef:   state.projects.isCancelError(undefined),
                forNetwork: state.projects.isCancelError(
                    new TypeError("Failed to fetch")),
                forGeneric: state.projects.isCancelError(
                    { ok: false, error: "permission denied" }),
            }));
        ''')
        assert out["forNull"] is False
        assert out["forUndef"] is False
        assert out["forNetwork"] is False
        assert out["forGeneric"] is False


# ----- upload adds relPath (design § C6) ------------------------- #


class TestUploadEnvelopeShape:
    """Per design § C6 UploadOk = WriteOk = {ok, path, relPath,
    size, mtime}.  Backend's /api/files/upload returns only
    {ok, path, size, mtime} (no relPath); state.upload computes
    relPath from the projects root."""

    def test_upload_computes_relPath_from_projects_root(self):
        out = _run_node('''
            state.setProjectsRoot("/home/u/projects");
            global.fetch = async () => ({
                ok: true, status: 200,
                json: async () => ({
                    ok: true,
                    path: "/home/u/projects/myjob/data.xyz",
                    size: 1234, mtime: 999,
                }),
            });
            const r = await state.projects.upload(
                "/home/u/projects/myjob", { name: "data.xyz" }
            );
            console.log(JSON.stringify(r));
        ''')
        assert out["ok"] is True
        assert out["path"] == "/home/u/projects/myjob/data.xyz"
        assert out["relPath"] == "myjob/data.xyz"
        assert out["size"] == 1234
        assert out["mtime"] == 999


# ----- setShared sessionStorage failure (design § 11.4) ---------- #


class TestSetSharedStorageFailure:
    """Per design § 11.4: sessionStorage write may throw (quota,
    private-mode SecurityError, storage denied).  setShared MUST NOT
    propagate the throw (violates Principle 6); it MUST publish the
    new state regardless so subscribers update; the cursor just
    won't survive a reload."""

    def test_setShared_swallows_sessionStorage_error_and_publishes(self):
        out = _run_node('''
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


# ----- Subscriber contract: throw on duplicate (A1b) ------------- #


class TestSubscribeDedupThrows:
    """Per design § 5.5 + § 11.5 (2026-05-31): registering the SAME
    callback (by reference) twice on any subscribe API is a
    programming error and must throw an Error.  Catches forgotten-
    unsubscribe + double-init bugs at the call site."""

    def test_onChange_throws_on_duplicate(self):
        out = _run_node('''
            const cb = (p) => {};
            state.projects.onChange(cb);
            let threw = false;
            let msg = "";
            try { state.projects.onChange(cb); }
            catch (e) { threw = true; msg = e.message; }
            console.log(JSON.stringify({ threw: threw, msg: msg }));
        ''')
        assert out["threw"] is True
        assert "onChange" in out["msg"]
        assert "already registered" in out["msg"]

    def test_onProjectsRootResolved_throws_on_duplicate(self):
        out = _run_node('''
            const cb = (p) => {};
            state.projects.onProjectsRootResolved(cb);
            let threw = false;
            try { state.projects.onProjectsRootResolved(cb); }
            catch (e) { threw = true; }
            console.log(JSON.stringify(threw));
        ''')
        assert out is True

    def test_different_callbacks_register_independently(self):
        """Two DIFFERENT callbacks (different fn objects) both register
        and both fire.  Only by-reference duplicates throw."""
        out = _run_node('''
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
        out = _run_node('''
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
    """Per design § 5.5: publish snapshots subscribers BEFORE
    iterating.  Subscribers registered DURING a publish loop fire
    only on subsequent events -- they got the current state via
    fire-once-immediately on their own subscribe call."""

    def test_new_subscriber_registered_during_publish_loop_does_not_fire_in_progress(self):
        out = _run_node('''
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
        out = _run_node('''
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


# ----- onProjectsRootResolved (design § C2) ---------------------- #


class TestOnProjectsRootResolved:
    """Design § C2 requires a one-shot-ish subscriber that fires
    when init resolves the projects root.  Subscribers registered
    BEFORE resolution receive the call when resolution lands;
    subscribers registered AFTER get an immediate fire-once-with-
    resolved-state per the standard contract."""

    def test_subscriber_fires_when_setProjectsRoot_lands(self):
        out = _run_node('''
            const calls = [];
            state.projects.onProjectsRootResolved(p => calls.push(p));
            // Initially no fire (root not resolved).
            const before = calls.slice();
            state.setProjectsRoot("/home/u/projects");
            console.log(JSON.stringify({
                before: before,
                after:  calls,
            }));
        ''')
        assert out["before"] == []
        assert out["after"] == [{"root": "/home/u/projects"}]

    def test_subscriber_registered_after_resolution_fires_immediately(self):
        """Late subscribers (e.g. a tab that loads after the
        sidebar's init completes) MUST still receive the resolved
        root.  Fire-once-immediately per the standard subscribe
        contract in § 6."""
        out = _run_node('''
            state.setProjectsRoot("/p");
            const calls = [];
            state.projects.onProjectsRootResolved(p => calls.push(p));
            console.log(JSON.stringify(calls));
        ''')
        assert out == [{"root": "/p"}]

    def test_unsubscribe_works(self):
        out = _run_node('''
            const calls = [];
            const unsub = state.projects.onProjectsRootResolved(
                p => calls.push(p)
            );
            unsub();
            state.setProjectsRoot("/p");
            console.log(JSON.stringify(calls));
        ''')
        assert out == []

    def test_only_fires_once_per_resolution(self):
        """A second setProjectsRoot call (theoretical; sidebar only
        calls once) does NOT re-fire subscribers.  Otherwise tabs
        would over-react to a no-op setProjectsRoot."""
        out = _run_node('''
            const calls = [];
            state.projects.onProjectsRootResolved(p => calls.push(p));
            state.setProjectsRoot("/p");
            state.setProjectsRoot("/p");
            state.setProjectsRoot("/p-different");
            console.log(JSON.stringify(calls));
        ''')
        # Single fire from the first non-empty resolution.
        assert out == [{"root": "/p"}]
