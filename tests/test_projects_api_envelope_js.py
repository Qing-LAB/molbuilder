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

    def test_apiRead_forwards_maxBytes(self):
        """The preview inspector's BULK read lifts the budget above the
        server default via ``opts.maxBytes`` -> ``&max_bytes=`` on the wire.
        Omitting it must NOT append the param (server default applies)."""
        out = _run_node('''
            let withCap = null, withoutCap = null;
            global.fetch = async (url, init) => {
                if (url.indexOf("max_bytes") >= 0) withCap = url; else withoutCap = url;
                return { ok: true, status: 200,
                         json: async () => ({ ok: true, text: "x" }) };
            };
            await api.apiRead("/p", { maxBytes: 16777216 });
            await api.apiRead("/p");
            console.log(JSON.stringify({
                capped:    withCap && withCap.indexOf("max_bytes=16777216") >= 0,
                uncapped:  withoutCap && withoutCap.indexOf("max_bytes") < 0,
            }));
        ''')
        assert out == {"capped": True, "uncapped": True}

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


# ----- cache: "no-store" default (#193, 2026-06-02) ---------------- #


class TestNoStoreCache:
    """The central ``_fetchEnvelope`` defaults ``cache: "no-store"`` so
    every projects.* live-data GET (apiList / apiRead / apiReadRange /
    apiStat / apiRoots) reaches the server on a same-URL revisit.
    Without this default the browser HTTP cache served the previous
    response on every same-URL hit -- the second-half cause of the
    /results stale-dropdown bug (#192).

    These tests pin the default + the override path so a future
    refactor can't quietly drop either.
    """

    def test_apiList_passes_cache_no_store(self):
        out = _run_node('''
            let capturedInit = null;
            global.fetch = async (_url, init) => {
                capturedInit = init || {};
                return {
                    ok: true, status: 200,
                    json: async () => ({ ok: true, entries: [] }),
                };
            };
            await api.apiList("/p");
            console.log(JSON.stringify({
                cache: capturedInit ? capturedInit.cache : null,
            }));
        ''')
        assert out["cache"] == "no-store"

    def test_apiRead_passes_cache_no_store(self):
        out = _run_node('''
            let cache = null;
            global.fetch = async (_u, init) => {
                cache = init && init.cache;
                return {
                    ok: true, status: 200,
                    json: async () => ({ ok: true, text: "" }),
                };
            };
            await api.apiRead("/p");
            console.log(JSON.stringify({ cache: cache }));
        ''')
        assert out["cache"] == "no-store"

    def test_apiReadRange_passes_cache_no_store(self):
        out = _run_node('''
            let cache = null;
            global.fetch = async (_u, init) => {
                cache = init && init.cache;
                return {
                    ok: true, status: 200,
                    json: async () => ({
                        ok: true, path: "/p", offset: 0, length: 0,
                        file_size: 0, mtime: 0, text: "", eof: true,
                    }),
                };
            };
            await api.apiReadRange("/p", 0, 1024);
            console.log(JSON.stringify({ cache: cache }));
        ''')
        assert out["cache"] == "no-store"

    def test_apiStat_passes_cache_no_store(self):
        out = _run_node('''
            let cache = null;
            global.fetch = async (_u, init) => {
                cache = init && init.cache;
                return {
                    ok: true, status: 200,
                    json: async () => ({ ok: true }),
                };
            };
            await api.apiStat("/p");
            console.log(JSON.stringify({ cache: cache }));
        ''')
        assert out["cache"] == "no-store"

    def test_apiRoots_passes_cache_no_store(self):
        """Roots is quasi-static but goes through the same wrapper.
        Forcing no-store here is harmless + keeps the default uniform
        so a future refactor can't drop the safety net for one caller."""
        out = _run_node('''
            let cache = null;
            global.fetch = async (_u, init) => {
                cache = init && init.cache;
                return {
                    ok: true, status: 200,
                    json: async () => ({ roots: [] }),
                };
            };
            await api.apiRoots();
            console.log(JSON.stringify({ cache: cache }));
        ''')
        assert out["cache"] == "no-store"

    def test_post_endpoints_also_get_no_store(self):
        """``cache: "no-store"`` is a no-op for POST (browsers never
        cache non-GET responses), so applying it uniformly is safe.
        Pin that it's still passed -- so a future caller that switches
        a GET endpoint to POST or adds a new POST through the same
        wrapper inherits the default without surprises."""
        out = _run_node('''
            let cache = null;
            global.fetch = async (_u, init) => {
                cache = init && init.cache;
                return {
                    ok: true, status: 200,
                    json: async () => ({ ok: true }),
                };
            };
            await api.apiMkdir("/parent", "new");
            console.log(JSON.stringify({ cache: cache }));
        ''')
        assert out["cache"] == "no-store"

    def test_signal_is_preserved_alongside_cache_default(self):
        """The default-cache injection must NOT clobber the AbortSignal
        the caller passes -- both have to land in the fetch init.  A
        regression that drops the signal would silently break the
        sidebar's Cancel button (#159) without breaking the cache
        contract above."""
        out = _run_node('''
            let ac;
            let initShape = null;
            global.fetch = async (_u, init) => {
                initShape = {
                    cache:        init && init.cache,
                    signalIsSet:  init && init.signal !== undefined
                                       && init.signal !== null,
                };
                return {
                    ok: true, status: 200,
                    json: async () => ({ ok: true, entries: [] }),
                };
            };
            ac = new AbortController();
            await api.apiList("/p", { signal: ac.signal });
            console.log(JSON.stringify(initShape));
        ''')
        assert out["cache"] == "no-store"
        assert out["signalIsSet"] is True


# ----- Source guard: consumers must not bypass api.js -------------- #


class TestNoRawFetchInPreview:
    """projects-sidebar.md Principle 6: api.js is the SOLE fetch caller
    for the /api/files/* endpoints.  preview.js (edit-save + the two read
    paths) must route every file call through the envelope wrappers so the
    uniform {ok,error,aborted} shape + no-store cache apply everywhere --
    no scattered raw ``fetch("/api/files/...")`` that re-implements error
    handling and silently diverges (the 2026-07 framework-bypass sweep)."""

    def test_preview_has_no_raw_files_fetch(self):
        src = (ROOT / "molbuilder/web/static/lib/projects/preview.js"
               ).read_text(encoding="utf-8")
        # No bare fetch( of an /api/files/ endpoint; all four sites
        # (write / stat / read / read_range) now go through api.js.
        assert 'fetch("/api/files' not in src and "fetch('/api/files" not in src, (
            "preview.js must call api.js wrappers, not raw fetch(\"/api/files/...\")"
        )
        # And it must actually import the wrappers it now depends on.
        assert 'from "./api.js"' in src, (
            "preview.js should import its file IO from api.js"
        )
