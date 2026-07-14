/* Workspace — the PERSISTENCE layer (workspace-contract.md).
 *
 * This module is ONLY session state + concealed file access.  It holds NO in-memory data model
 * and never interprets what it stores: the MolView data model (lib/molview/data-model.js) owns
 * the structure/selection/periodicity/frames and their format, serialises itself, and hands the
 * BYTES here to persist.  The workspace writes them format-blind.
 *
 * Public surface (mounted on ``window.molbuilder.workspace``):
 *   - persist(sessionBytes, snapshotBlob, identity)  -- write the consumer's serialised state now
 *                                                    (session mirror + on-disk indexed state file).
 *   - readState(identity)      -- read the opaque snapshot bytes at {workspace_id, state_index}
 *                                 from disk (a history index popState navigates to), or null.
 *   - pruneStatesAbove(workspace_id, index)  -- tail-delete on-disk state files above ``index``
 *                                 (index === -1 clears the whole timeline).
 *   - workspaceId()            -- the stable id a sourceless workspace's draft is keyed under.
 *   - readPersistedSnapshot()  -- the persisted session snapshot (or null).
 *   - mountRestoreTarget()     -- the source-file a mount-time restore owns (single-authority
 *                                 rule, workspace-contract §4.5), or null.
 *   - STORAGE_KEY              -- the sessionStorage key (shared constant).
 *
 * The DEBOUNCE + suspend/resume live in the data model (it owns "when the data changed"); this
 * layer's persist() is a synchronous write of the bytes it is handed.
 */
(function (root) {
    "use strict";

    function _snapshotIO() {
        return (root.molbuilder && root.molbuilder.workspaceSnapshot) || null;
    }
    function _runtime() {
        return (root.molbuilder && root.molbuilder.runtime) ? root.molbuilder.runtime : null;
    }

    // Shared classic-script constant — see lib/constants.js.  Fallback keeps the module
    // functional in test contexts that don't load constants.js.
    var STORAGE_KEY = ((root.molbuilder || {}).constants || {}).SS_WORKSPACE
        || "molbuilder.workspace.v1";

    // ─── Owner namespace (molview-module.md §18.4) ────────────────── //
    // Each consumer that mounts a molview declares an ``owner`` (mount.js). We fold it into
    // BOTH storage keys so one consumer's session never overwrites another's:
    //   - the sessionStorage mirror key (delegated to snapshot-io's setNamespace), and
    //   - the on-disk ``workspace_id`` -- indirectly: workspaceId() derives from the
    //     now-namespaced mirror, so a fresh namespace generates its own id.  Switching
    //     namespace clears the cached id so it is recomputed against the new mirror.
    // A single active namespace is coherent: each page mounts one active owner at a time.
    function useNamespace(owner) {
        owner = owner || null;
        var io = _snapshotIO();
        if (io && typeof io.setNamespace === "function") io.setNamespace(owner);
        _workspaceId = null;   // recompute against this namespace's mirror
    }

    // ─── Session identity ─────────────────────────────────────────── //
    // A stable id for a SOURCELESS workspace's draft, so repeated updates hit the SAME draft
    // file and a same-tab reload keeps it (not a fresh orphan each time).
    var _workspaceId = null;
    function workspaceId() {
        if (_workspaceId) return _workspaceId;
        var snap = readPersistedSnapshot();      // reuse across same-tab reload
        if (snap && snap.workspace_id) {
            _workspaceId = snap.workspace_id;
            return _workspaceId;
        }
        _workspaceId = "ws-" + Date.now().toString(36) + "-"
                     + Math.random().toString(36).slice(2, 10);
        return _workspaceId;
    }

    // ─── Reads from persistence (restore) ─────────────────────────── //
    /**
     * The parsed session snapshot if the unified key has one, else null.  Callers (the data
     * model's restore + canvas-state init) check this before falling back to legacy mirrors.
     * Delegates to the shared snapshot IO (snapshot-io.js) — the ONE owner of the read.
     */
    function readPersistedSnapshot() {
        var io = _snapshotIO();
        return (io && typeof io.read === "function") ? io.read() : null;
    }

    /**
     * MOUNT-RESTORE OWNERSHIP (workspace-contract.md §4.5).  Returns the source-file path a
     * mount-time snapshot restore will hydrate, or null when the snapshot carries no restorable
     * structure.  Every mount-time writer MUST honor this and defer when it equals the file it
     * was about to load (the two-writer mount race).  Order-independent: reads the SAME persisted
     * snapshot the restore derives from.
     */
    function mountRestoreTarget() {
        var snap = readPersistedSnapshot();
        if (!snap || !snap.state || !snap.state.structure
                || !snap.state.structure.text) return null;
        return (snap.state.source && snap.state.source.file) || null;
    }

    // ─── Writes (persist) — format-blind ──────────────────────────── //
    function _persistToSession(sessionBytes) {
        // The shared snapshot IO owns the sessionStorage write — the SAME module canvas-state
        // reads on reload, so there is one key + one format.
        var io = _snapshotIO();
        if (io && typeof io.write === "function" && sessionBytes) io.write(sessionBytes);
    }

    // The on-disk indexed STATE FILE (workspace-contract §4.7, the push-only state timeline).
    // ``snapshotBlob`` is the consumer's already-serialised OPAQUE session snapshot; ``identity``
    // = {workspace_id, state_index} keys the filename ``<workspace_id>.<state_index>.wc.json``.
    // The server stores it FORMAT-BLIND (never through the structure codec).  Best-effort.
    // Ordered event tracer (diagnostic; no-op unless window.__MV_TRACE).  Mirrors
    // data-model's _trace so client seams + HTTP round-trips share one timeline.
    function _trace(ev, extra) {
        if (!root.__MV_TRACE) return;
        try {
            var t = (root.performance && root.performance.now)
                ? root.performance.now() : 0;
            root.console.log("[MV-TRACE " + t.toFixed(1) + "] " + ev
                + (extra !== undefined ? " " + JSON.stringify(extra) : ""));
        } catch (_) { /* never throws */ }
    }

    // Persist contract: NON-BLOCKING but ERROR-EXPLICIT.  The on-disk state
    // write is fire-and-forget (the hot path never awaits it -- the in-memory
    // model + synchronous session mirror are the source of truth), BUT a failure
    // is NEVER swallowed: it is reported to the console AND emitted as a
    // ``molbuilder:persist-error`` DOM event so a UI layer can warn the user
    // ("state didn't reach disk; retract history / crash recovery may be
    // incomplete").  A failure is either a rejected fetch (network) OR a
    // non-2xx response (server refused, e.g. bad workspace_id / disk).
    var _persistErrorHandlers = [];
    // Subscribe to persist failures (the UI layer registers here to warn the
    // user).  Returns an unsubscribe fn.  Part of the non-blocking/error-explicit
    // contract: the write is fire-and-forget, but every failure reaches here.
    function onPersistError(fn) {
        if (typeof fn !== "function") return function () {};
        _persistErrorHandlers.push(fn);
        return function () {
            var i = _persistErrorHandlers.indexOf(fn);
            if (i >= 0) _persistErrorHandlers.splice(i, 1);
        };
    }
    function _reportPersistError(detail) {
        try { root.console.error("[workspace] persist FAILED (non-blocking)", detail); }
        catch (_) { /* console may be absent */ }
        _persistErrorHandlers.slice().forEach(function (fn) {
            try { fn(detail); } catch (_) { /* one bad handler can't muzzle the rest */ }
        });
        try {   // also a DOM event, for decoupled listeners in a real browser
            if (root.dispatchEvent && typeof root.CustomEvent === "function") {
                root.dispatchEvent(new root.CustomEvent(
                    "molbuilder:persist-error", { detail: detail }));
            }
        } catch (_) { /* event dispatch is best-effort surfacing */ }
    }

    function _persistState(snapshotBlob, identity) {
        if (!root.fetch || !snapshotBlob) return;
        var idx = identity && identity.state_index;
        _trace("http:write-state:issue", { idx: idx });
        root.fetch("/api/workingcopy/write-state", {
            method:  "POST",
            headers: { "Content-Type": "application/json" },
            body:    JSON.stringify(Object.assign({}, identity || {}, { data: snapshotBlob })),
        }).then(function (res) {
            _trace("http:write-state:done", { idx: idx, status: res && res.status });
            if (!res || !res.ok) {
                _reportPersistError({ op: "write-state", state_index: idx,
                                      status: res && res.status });
            }
        }).catch(function (err) {
            _trace("http:write-state:error", { idx: idx });
            _reportPersistError({ op: "write-state", state_index: idx,
                                  error: (err && err.message) || String(err) });
        });
    }

    /**
     * Persist the consumer's serialised state NOW — the single write-in.  The data model owns
     * WHEN (push-only: load anchor + each pushState); this just writes the bytes, format-blind:
     *   sessionBytes -> the sessionStorage session mirror (fast same-tab reload)
     *   snapshotBlob -> the on-disk indexed state file, keyed by ``identity`` {workspace_id,
     *                   state_index} -> ``<workspace_id>.<state_index>.wc.json``
     */
    function persist(sessionBytes, snapshotBlob, identity) {
        if (!root.sessionStorage && !root.fetch) return;
        _persistToSession(sessionBytes);
        _persistState(snapshotBlob, identity);
    }

    /**
     * Read the OPAQUE snapshot bytes on disk at {workspace_id, state_index} (§4.7 read-by-index),
     * what popState calls to fetch a *history* index the session mirror no longer holds.  Resolves
     * the parsed JSON, or null when the file is missing / unreadable.  Format-blind — the data
     * model interprets what comes back.
     */
    function readState(identity) {
        if (!root.fetch || !identity) return Promise.resolve(null);
        return root.fetch("/api/workingcopy/read-state", {
            method:  "POST",
            headers: { "Content-Type": "application/json" },
            body:    JSON.stringify(identity),
        }).then(function (res) {
            if (!res || !res.ok) return null;          // 404 (missing) -> null
            return res.json();
        }).then(function (j) {
            return (j && j.data != null) ? j.data : null;
        }).catch(function () { return null; });
    }

    /**
     * Tail-delete the on-disk state files whose index > ``index`` (§4.7 pruning: a pushState after
     * a popState drops the abandoned tail).  ``index === -1`` clears the whole ``<workspace_id>.*``
     * timeline.  Best-effort; resolves when the server has acted.
     */
    function pruneStatesAbove(workspace_id, index) {
        if (!root.fetch || !workspace_id) return Promise.resolve();
        _trace("http:prune-states:issue", { above: index });
        return root.fetch("/api/workingcopy/prune-states", {
            method:  "POST",
            headers: { "Content-Type": "application/json" },
            body:    JSON.stringify({ workspace_id: workspace_id, above_index: index }),
        }).then(function (res) {
            _trace("http:prune-states:done", { above: index, status: res && res.status });
            if (!res || !res.ok) {
                _reportPersistError({ op: "prune-states", above_index: index,
                                      status: res && res.status });
            }
            return res;   // resolve either way: the anchor write still proceeds
        }).catch(function (err) {
            _trace("http:prune-states:error", { above: index });
            _reportPersistError({ op: "prune-states", above_index: index,
                                  error: (err && err.message) || String(err) });
            // Resolve (undefined) so _anchorTimeline's ordered write still runs;
            // a failed prune leaves a stale tail, not a lost anchor.
        });
    }

    var api = {
        persist:               persist,
        readState:             readState,
        pruneStatesAbove:      pruneStatesAbove,
        useNamespace:          useNamespace,
        workspaceId:           workspaceId,
        readPersistedSnapshot: readPersistedSnapshot,
        mountRestoreTarget:    mountRestoreTarget,
        onPersistError:        onPersistError,   // subscribe to non-blocking write failures
        STORAGE_KEY:           STORAGE_KEY,
    };

    // MERGE into any pre-existing ``workspace`` namespace, not replace it: canvas-state-impl.js
    // (still) mounts its singleton on ``workspace._canvasState`` before this file loads; a plain
    // ``= api`` would clobber that slot.  Preserve every private slot the impls already set.
    root.molbuilder = root.molbuilder || {};
    root.molbuilder.workspace = Object.assign(
        root.molbuilder.workspace || {}, api);
    if (_runtime() && typeof _runtime().register === "function") {
        _runtime().register("workspace", api);
    }

    if (typeof module !== "undefined" && module.exports) {
        module.exports = api;
    }
})(typeof window !== "undefined" ? window : this);
