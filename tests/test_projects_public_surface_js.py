"""Unit tests for the public projects.* mutator surface (sidebar
gap M4, #175).

Promotes ``readFile / createProject / mkdir / deleteEntry / upload /
navigateTo`` onto ``window.molbuilder.projects.*`` so external
callers (the /results result-list dropdown + future programmatic
file managers) don't need to import ``./api.js`` directly.

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
                createProject:              typeof p.createProject,
                mkdir:                      typeof p.mkdir,
                deleteEntry:                typeof p.deleteEntry,
                rename:                     typeof p.rename,
                upload:                     typeof p.upload,
                setShared:                  typeof p.setShared,
                navigateTo:                 typeof p.navigateTo,
                onProjectsRootResolved:     typeof p.onProjectsRootResolved,
            }));
        ''')
        for fn in ("readFile", "createProject", "mkdir",
                   "deleteEntry", "rename", "upload", "setShared",
                   "navigateTo", "onProjectsRootResolved"):
            assert out[fn] == "function", f"missing public method: {fn}"


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

    def test_locked_navigateTo_rejects_without_calling_impl(self):
        """Per § 8.5, navigateTo MUST check isLocked() and refuse
        when the lock is held -- without calling the underlying
        impl.  The impl (openDir) is intentionally NOT lock-guarded
        because it doubles as the refreshHandler that runs mid-Save-
        pipeline; the public-surface wrapper enforces § 8.5 instead."""
        out = _run_node('''
            let implCalled = false;
            state.setNavigateToImpl(async () => {
                implCalled = true;
                return { ok: true, path: "/", entries: [] };
            });
            state.projects.lock("Saving FDF...", []);
            const r = await state.projects.navigateTo("/somewhere");
            console.log(JSON.stringify({
                envelope:    r,
                implCalled:  implCalled,
            }));
        ''')
        assert out["envelope"]["ok"] is False
        assert "sidebar is locked" in out["envelope"]["error"]
        assert "Saving FDF" in out["envelope"]["error"]
        assert out["implCalled"] is False


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
