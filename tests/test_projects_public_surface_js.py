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
                readFile:       typeof p.readFile,
                createProject:  typeof p.createProject,
                mkdir:          typeof p.mkdir,
                deleteEntry:    typeof p.deleteEntry,
                upload:         typeof p.upload,
                setShared:      typeof p.setShared,
                navigateTo:     typeof p.navigateTo,
            }));
        ''')
        for fn in ("readFile", "createProject", "mkdir",
                   "deleteEntry", "upload", "setShared",
                   "navigateTo"):
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


# ----- navigateTo is an alias for setShared ----------------------- #


class TestNavigateTo:

    def test_navigateTo_with_file_writes_both(self):
        out = _run_node('''
            const r = state.projects.navigateTo("/d", "/d/f.out");
            console.log(JSON.stringify({
                envelope: r,
                dir:      sessionStorage.getItem("molbuilder.current_dir"),
                file:     sessionStorage.getItem("molbuilder.current_file"),
            }));
        ''')
        assert out["envelope"] == {"ok": True}
        assert out["dir"] == "/d"
        assert out["file"] == "/d/f.out"

    def test_navigateTo_without_file_clears_file(self):
        """Single-arg form clears file selection but keeps the dir.
        This is the use case for tabs that want to position the
        sidebar without selecting any particular file (e.g. a
        ``cd`` operation)."""
        out = _run_node('''
            // Prime a file selection.
            state.setShared("/old", "/old/sel.out");
            // Now navigateTo without a file.
            state.projects.navigateTo("/new");
            console.log(JSON.stringify({
                dir:  sessionStorage.getItem("molbuilder.current_dir"),
                file: sessionStorage.getItem("molbuilder.current_file"),
            }));
        ''')
        assert out["dir"] == "/new"
        assert out["file"] == ""

    def test_navigateTo_respects_lock(self):
        """navigateTo is a thin shim over setShared; the lock guard
        from #177 carries through automatically."""
        out = _run_node('''
            state.projects.lock("Save in flight", []);
            const r = state.projects.navigateTo("/d", "/d/f.out");
            console.log(JSON.stringify({
                envelope:    r,
                dir_post:    sessionStorage.getItem("molbuilder.current_dir"),
            }));
        ''')
        assert out["envelope"]["ok"] is False
        assert "sidebar is locked" in out["envelope"]["error"]
        assert out["dir_post"] is None
