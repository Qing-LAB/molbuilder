"""Unit tests for the uniform-envelope contract in ``lib/projects/
api.js`` (sidebar gap M3, #173).

The contract (per docs/protocols/projects-sidebar.md Principle 6):

  Every async function in api.js returns ``{ok: bool, ...}`` and
  NEVER throws.  Network failures, DNS errors, non-JSON responses,
  and AbortErrors ALL surface as ``{ok: false, error: "..."}``.

Tests stub ``fetch`` under Node to simulate each failure mode +
the happy path, and verify the returned envelope shape.  No
backend required.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "molbuilder/web/static/lib/projects/api.js"


def _run_node(snippet: str) -> object:
    """Run a JS snippet under Node with api.js pre-loaded via
    dynamic import.  ``snippet`` must end with a line that calls
    ``console.log(JSON.stringify(result))`` so the Python side can
    parse the return value back.

    Each test sets ``global.fetch`` to a stub that returns whatever
    the test needs, then invokes one of the api functions and
    checks the envelope shape.
    """
    node = shutil.which("node")
    if node is None:
        pytest.skip("node not available")
    # The module uses ES-module syntax (export async function ...).
    # We import it via dynamic ``import()`` which works under Node
    # without any tsconfig / package.json so long as we tell Node
    # to treat the import source as a module.  We do that by
    # serving the module via a file:// URL.
    module_url = MODULE.resolve().as_uri()
    bootstrap = f"""
        // Node 16+ exposes fetch globally; we OVERRIDE it per-test
        // so the snippet runs in a controlled environment.  Tests
        // that don't reassign get the Node-default fetch which
        // hits the real network -- not what we want.  Default to
        // "no fetch should be called" so an accidental real-fetch
        // path produces a visible error.
        global.fetch = () => {{
            throw new Error(
                "test must override global.fetch; api.js called the "
                + "default no-op stub"
            );
        }};
        const apiPromise = import("{module_url}");
        // Defer all snippet execution until the dynamic import
        // resolves; that way the snippet can reference the api
        // module under its alias ``api``.
        apiPromise.then(async (api) => {{
            {snippet}
        }}).catch(err => {{
            console.log(JSON.stringify({{
                __test_unexpected_throw: true,
                message: err && err.message ? err.message : String(err),
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
    # Last JSON line is the test result; earlier lines may include
    # diagnostic prints from the module.
    last_line = proc.stdout.strip().splitlines()[-1]
    out = json.loads(last_line)
    assert "__test_unexpected_throw" not in out, (
        "api.js threw instead of returning envelope: " + str(out)
    )
    return out


# ----- Happy path: server returns valid JSON envelope -------------- #


class TestHappyPath:

    def test_apiList_returns_server_body_verbatim(self):
        out = _run_node('''
            global.fetch = async () => ({
                ok: true,
                status: 200,
                json: async () => ({ ok: true, entries: ["a","b"] }),
            });
            const r = await api.apiList("/some/path");
            console.log(JSON.stringify(r));
        ''')
        assert out == {"ok": True, "entries": ["a", "b"]}

    def test_apiRead_returns_server_body_verbatim(self):
        out = _run_node('''
            global.fetch = async () => ({
                ok: true,
                status: 200,
                json: async () => ({ ok: true, text: "hello", mtime: 12345 }),
            });
            const r = await api.apiRead("/x");
            console.log(JSON.stringify(r));
        ''')
        assert out == {"ok": True, "text": "hello", "mtime": 12345}

    def test_apiReadRange_returns_server_body_verbatim(self):
        """The raw wrapper (#189, 2026-06-02) mirrors apiRead's shape
        but takes offset + maxBytes and surfaces the server's range-
        read envelope (file_size + eof in addition to text + mtime)."""
        out = _run_node('''
            global.fetch = async () => ({
                ok: true,
                status: 200,
                json: async () => ({
                    ok: true, path: "/p/big.log", offset: 0,
                    length: 262144, file_size: 1000000,
                    mtime: 42, text: "first chunk", eof: false,
                }),
            });
            const r = await api.apiReadRange("/p/big.log", 0, 262144);
            console.log(JSON.stringify(r));
        ''')
        assert out["ok"] is True
        assert out["length"] == 262144
        assert out["eof"] is False
        assert out["text"] == "first chunk"

    def test_apiWrite_post_with_overwrite_flag(self):
        out = _run_node('''
            let captured = null;
            global.fetch = async (url, init) => {
                captured = { url, init };
                return {
                    ok: true,
                    status: 200,
                    json: async () => ({ ok: true, mtime: 999 }),
                };
            };
            const r = await api.apiWrite("/p", "text", {overwrite: true});
            console.log(JSON.stringify({
                envelope: r,
                url:      captured.url,
                method:   captured.init.method,
                body:     JSON.parse(captured.init.body),
            }));
        ''')
        assert out["envelope"] == {"ok": True, "mtime": 999}
        assert out["url"] == "/api/files/write"
        assert out["method"] == "POST"
        assert out["body"] == {"path": "/p", "text": "text",
                                "overwrite": True}


# ----- Network failure: fetch itself rejects ----------------------- #


class TestNetworkFailure:

    def test_apiList_network_drop_returns_envelope(self):
        """``fetch`` throws TypeError on network drop / offline /
        DNS fail.  Pre-this-commit, apiList re-threw; now it returns
        the synthetic envelope."""
        out = _run_node('''
            global.fetch = async () => {
                throw new TypeError("Failed to fetch");
            };
            const r = await api.apiList("/x");
            console.log(JSON.stringify(r));
        ''')
        assert out["ok"] is False
        assert "network error" in out["error"]
        assert "Failed to fetch" in out["error"]

    def test_apiMkdir_network_drop_returns_envelope(self):
        out = _run_node('''
            global.fetch = async () => {
                throw new Error("offline");
            };
            const r = await api.apiMkdir("/parent", "newdir");
            console.log(JSON.stringify(r));
        ''')
        assert out["ok"] is False
        assert "network error" in out["error"]

    def test_apiUpload_network_drop_returns_envelope(self):
        out = _run_node('''
            // Provide a FormData stub since Node may not have one.
            global.FormData = function () {
                this.append = () => {};
            };
            global.fetch = async () => {
                throw new TypeError("Failed to fetch");
            };
            const r = await api.apiUpload("/x", { name: "f.xyz" });
            console.log(JSON.stringify(r));
        ''')
        assert out["ok"] is False
        assert "network error" in out["error"]

    def test_apiWrite_network_drop_returns_envelope(self):
        """apiWrite had the envelope behaviour pre-this-commit;
        regression-pin so a refactor doesn't accidentally drop it."""
        out = _run_node('''
            global.fetch = async () => {
                throw new TypeError("Failed to fetch");
            };
            const r = await api.apiWrite("/x", "text");
            console.log(JSON.stringify(r));
        ''')
        assert out["ok"] is False
        assert "network error" in out["error"]


# ----- Non-JSON response: server returns 501 / HTML 5xx ----------- #


class TestNonJsonResponse:

    def test_apiDelete_non_json_returns_envelope(self):
        out = _run_node('''
            global.fetch = async () => ({
                ok: false,
                status: 501,
                json: async () => { throw new SyntaxError("Unexpected token"); },
            });
            const r = await api.apiDelete("/x", false);
            console.log(JSON.stringify(r));
        ''')
        assert out["ok"] is False
        assert "non-JSON" in out["error"]
        assert "501" in out["error"]

    def test_apiList_non_json_returns_envelope(self):
        out = _run_node('''
            global.fetch = async () => ({
                ok: true,
                status: 200,
                json: async () => { throw new SyntaxError("Unexpected token <"); },
            });
            const r = await api.apiList("/x");
            console.log(JSON.stringify(r));
        ''')
        assert out["ok"] is False
        assert "non-JSON" in out["error"]


# ----- apiRoots envelope normalisation ----------------------------- #


# ----- AbortSignal threading (#174 sidebar gap M1) ---------------- #


class TestAbortSignal:

    def test_apiWrite_aborted_returns_clean_envelope(self):
        """When the caller's AbortSignal fires, fetch rejects with
        AbortError.  api.js produces a CLEAN envelope (no "network
        error" prose) so callers can distinguish user-cancellation
        from connectivity failure."""
        out = _run_node('''
            global.fetch = async (url, init) => {
                // Simulate the runtime aborting mid-flight.
                const err = new Error("operation aborted");
                err.name = "AbortError";
                throw err;
            };
            const r = await api.apiWrite("/x", "text");
            console.log(JSON.stringify(r));
        ''')
        assert out["ok"] is False
        assert out["error"] == "aborted"
        assert out["aborted"] is True

    def test_apiDelete_aborted_returns_clean_envelope(self):
        out = _run_node('''
            global.fetch = async () => {
                const err = new Error("aborted");
                err.name = "AbortError";
                throw err;
            };
            const r = await api.apiDelete("/x", false);
            console.log(JSON.stringify(r));
        ''')
        assert out["aborted"] is True
        assert out["ok"] is False

    def test_apiUpload_aborted_returns_clean_envelope(self):
        out = _run_node('''
            global.FormData = function () { this.append = () => {}; };
            global.fetch = async () => {
                const err = new Error("aborted");
                err.name = "AbortError";
                throw err;
            };
            const r = await api.apiUpload("/x", { name: "f" });
            console.log(JSON.stringify(r));
        ''')
        assert out["aborted"] is True
        assert out["ok"] is False

    def test_apiWrite_forwards_signal_into_fetch(self):
        """The ``signal`` from opts must reach the fetch init -- pin
        the wire-format contract so a refactor can't accidentally
        drop the thread-through.  The lock's three-layer recovery
        depends on this."""
        out = _run_node('''
            let captured = null;
            global.fetch = async (url, init) => {
                captured = init;
                return {
                    ok: true,
                    status: 200,
                    json: async () => ({ ok: true }),
                };
            };
            const ctl = new AbortController();
            await api.apiWrite("/x", "text", { signal: ctl.signal });
            // Identity check: the SAME AbortSignal instance reaches
            // fetch -- not a copy / wrapper / clone.
            console.log(JSON.stringify({
                wasSignalForwarded: captured.signal === ctl.signal,
            }));
        ''')
        assert out == {"wasSignalForwarded": True}

    def test_apiDelete_forwards_signal_into_fetch(self):
        out = _run_node('''
            let captured = null;
            global.fetch = async (url, init) => {
                captured = init;
                return { ok: true, status: 200, json: async () => ({ok: true}) };
            };
            const ctl = new AbortController();
            await api.apiDelete("/x", false, { signal: ctl.signal });
            console.log(JSON.stringify({
                wasSignalForwarded: captured.signal === ctl.signal,
            }));
        ''')
        assert out == {"wasSignalForwarded": True}

    def test_apiUpload_forwards_signal_into_fetch(self):
        out = _run_node('''
            global.FormData = function () { this.append = () => {}; };
            let captured = null;
            global.fetch = async (url, init) => {
                captured = init;
                return { ok: true, status: 200, json: async () => ({ok: true}) };
            };
            const ctl = new AbortController();
            await api.apiUpload("/x", { name: "f" }, { signal: ctl.signal });
            console.log(JSON.stringify({
                wasSignalForwarded: captured.signal === ctl.signal,
            }));
        ''')
        assert out == {"wasSignalForwarded": True}

    def test_no_signal_is_safe(self):
        """Backwards-compat: callers that don't pass opts.signal
        still work.  fetch receives ``signal: undefined`` which is
        equivalent to no signal."""
        out = _run_node('''
            global.fetch = async (url, init) => {
                return { ok: true, status: 200, json: async () => ({ok: true}) };
            };
            const r = await api.apiWrite("/x", "text");  // no opts
            console.log(JSON.stringify(r));
        ''')
        assert out == {"ok": True}

    def test_apiRead_forwards_signal(self):
        """2026-05-31 (post-design audit): read endpoints also accept
        opts.signal so a tab can cancel a slow read.  Identity-checked
        to pin the wire-format contract."""
        out = _run_node('''
            let captured = null;
            global.fetch = async (url, init) => {
                captured = init;
                return { ok: true, status: 200,
                         json: async () => ({ ok: true, text: "x" }) };
            };
            const ctl = new AbortController();
            await api.apiRead("/p", { signal: ctl.signal });
            console.log(JSON.stringify({
                wasSignalForwarded: captured.signal === ctl.signal,
            }));
        ''')
        assert out == {"wasSignalForwarded": True}

    def test_apiList_forwards_signal(self):
        out = _run_node('''
            let captured = null;
            global.fetch = async (url, init) => {
                captured = init;
                return { ok: true, status: 200,
                         json: async () => ({ ok: true, entries: [] }) };
            };
            const ctl = new AbortController();
            await api.apiList("/p", { signal: ctl.signal });
            console.log(JSON.stringify({
                wasSignalForwarded: captured.signal === ctl.signal,
            }));
        ''')
        assert out == {"wasSignalForwarded": True}

    def test_apiRename_forwards_signal(self):
        """apiRename (added 2026-05-31 alongside the backend rename
        endpoint) honours opts.signal like every other writer."""
        out = _run_node('''
            let captured = null;
            global.fetch = async (url, init) => {
                captured = init;
                return { ok: true, status: 200,
                         json: async () => ({ ok: true, path: "/x/new" }) };
            };
            const ctl = new AbortController();
            await api.apiRename("/x/old", "new", { signal: ctl.signal });
            console.log(JSON.stringify({
                wasSignalForwarded: captured.signal === ctl.signal,
            }));
        ''')
        assert out == {"wasSignalForwarded": True}

    def test_apiRename_posts_with_correct_body(self):
        """apiRename POSTs {path, new_name} to /api/files/rename."""
        out = _run_node('''
            let captured = null;
            global.fetch = async (url, init) => {
                captured = { url, init };
                return { ok: true, status: 200,
                         json: async () => ({ ok: true, path: "/x/new" }) };
            };
            await api.apiRename("/x/old", "new");
            console.log(JSON.stringify({
                url:    captured.url,
                method: captured.init.method,
                body:   JSON.parse(captured.init.body),
            }));
        ''')
        assert out["url"] == "/api/files/rename"
        assert out["method"] == "POST"
        assert out["body"] == {"path": "/x/old", "new_name": "new"}

    def test_apiMkdir_forwards_signal(self):
        out = _run_node('''
            let captured = null;
            global.fetch = async (url, init) => {
                captured = init;
                return { ok: true, status: 200,
                         json: async () => ({ ok: true }) };
            };
            const ctl = new AbortController();
            await api.apiMkdir("/p", "new", { signal: ctl.signal });
            console.log(JSON.stringify({
                wasSignalForwarded: captured.signal === ctl.signal,
            }));
        ''')
        assert out == {"wasSignalForwarded": True}


class TestApiRoots:

    def test_normalises_to_envelope_on_success(self):
        """The /api/files/roots backend responds with ``{roots: [...]}``
        without a top-level ``ok``.  api.js normalises to the uniform
        envelope shape."""
        out = _run_node('''
            global.fetch = async () => ({
                ok: true,
                status: 200,
                json: async () => ({
                    roots: [{ path: "/projects", label: "projects" }],
                }),
            });
            const r = await api.apiRoots();
            console.log(JSON.stringify(r));
        ''')
        assert out["ok"] is True
        assert out["roots"] == [
            {"path": "/projects", "label": "projects"},
        ]

    def test_includes_empty_roots_on_failure(self):
        """Failure case carries an empty ``roots`` array so callers
        that destructure ``{roots}`` don't NPE."""
        out = _run_node('''
            global.fetch = async () => {
                throw new TypeError("Failed to fetch");
            };
            const r = await api.apiRoots();
            console.log(JSON.stringify(r));
        ''')
        assert out["ok"] is False
        assert "network error" in out["error"]
        assert out["roots"] == []
