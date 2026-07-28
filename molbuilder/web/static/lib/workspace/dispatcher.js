/* Workspace — the session-persistence layer.
 *
 * MODULE: workspace persistence  (lib/workspace/; contract: docs/protocols/workspace-contract.md).
 *   Two files make up the module:
 *     - dispatcher.js  (this file; a native ES module)  -> window.molbuilder.workspace : persist/restore
 *         transport, session identity, owner namespace, non-blocking error surface.
 *     - snapshot-io.js              -> window.molbuilder.workspaceSnapshot: the SOLE sessionStorage
 *         read/write owner (namespaced); this file delegates every sessionStorage touch to it.
 *   Server backend: POST /api/state-timeline/{write,read,prune} (blueprints/state_timeline.py) —
 *   the on-disk indexed STATE TIMELINE (workspace-contract §4.7).
 *
 * ROLE: session state + concealed file access ONLY.  Holds NO in-memory data model and never
 *   interprets what it stores.  The MolView data model (lib/molview/data-model.js) owns the
 *   structure/selection/periodicity/frames + their format, serialises itself, and hands the BYTES
 *   here to persist; this layer writes them format-blind.  The suspend/resume + the
 *   "when the data changed" decision (PUSH-ONLY, no debounce) live in the data model —
 *   persist() here is a synchronous write of the bytes it is handed.
 *
 * USED BY (callers of window.molbuilder.workspace):
 *   - lib/molview/data-model.js — the state save/retract TIMELINE: persist(), readState(),
 *     pruneStatesAbove(), workspaceId().  The primary consumer; it owns WHEN to write.
 *   - lib/molview/mount.js — useNamespace(owner) at each mount, so one owner's session never
 *     overwrites another's (per-owner key isolation).
 *   - lib/molview/_canvas-state-impl.js — reload restore; reads the session snapshot via the
 *     shared snapshot-io owner (NOT this dispatcher — see that file's rationale).
 *   - tabs that mount a molview hand window.molbuilder.workspace to molview.mount:
 *     modify/, spectra/viewer.js, transport/core.js, static/viewer.js, molview/demo.js,
 *     lib/inspectors/structure.js.
 *   - a UI notification layer subscribes to onPersistError() to warn on a failed disk write.
 *
 * Public surface (window.molbuilder.workspace):
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
 *   - hasRestorableSnapshot()  -- true when a persisted snapshot exists for the current owner
 *                                 (gate for the §4.5 mount-restore rule).
 *   - useNamespace(owner)      -- switch the active owner namespace (mirror key + workspace_id).
 *   - onPersistError(fn)       -- subscribe to non-blocking disk-write failures.
 *   - STORAGE_KEY              -- the sessionStorage key (shared constant).
 */
"use strict";

import { workspaceSnapshot } from "./snapshot-io.js";

const root = (typeof window !== "undefined") ? window : globalThis;

    // The sessionStorage half is a package-internal sibling -- IMPORTED (its window global
    // stays published by snapshot-io itself for the _canvas-state-impl cross-package read).
    function _snapshotIO() { return workspaceSnapshot; }
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
     *
     * CAUTION: a null return is AMBIGUOUS -- it means EITHER "no restorable snapshot" OR "a
     * restorable snapshot with no source file" (a GENERATED structure: SMILES / RNA / peptide /
     * name build has ``structure.text`` but no ``source.file``).  A mount-time writer that needs
     * to know "does the restore own the canvas AT ALL" (regardless of by-file identity) must use
     * ``hasRestorableSnapshot()`` below, NOT a ``!== mountRestoreTarget()`` file comparison --
     * the latter wrongly treats a generated structure as "no snapshot" and clobbers it.
     */
    function mountRestoreTarget() {
        var snap = readPersistedSnapshot();
        if (!snap || !snap.state || !snap.state.structure
                || !snap.state.structure.text) return null;
        return (snap.state.source && snap.state.source.file) || null;
    }

    /**
     * MOUNT-RESTORE OWNERSHIP (workspace-contract.md §4.5).  True iff the persisted snapshot
     * carries a restorable structure (``structure.text`` present), whether or not it came from a
     * file.  This is the invariant a mount-time writer actually needs: when it is true, the
     * mount-time restore is the SOLE authority for the canvas and no other writer may load
     * anything -- persistency wins over a stale sidebar selection (file load stays explicit).
     * Unlike ``mountRestoreTarget()`` this does not conflate "no snapshot" with "file-less
     * generated structure".
     */
    function hasRestorableSnapshot() {
        var snap = readPersistedSnapshot();
        return !!(snap && snap.state && snap.state.structure
                  && snap.state.structure.text);
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

    // Serialise state writes in ISSUE ORDER.  The writes are fire-and-forget, so two
    // POSTs to the SAME <workspace_id>.<state_index> file (a rapid
    // save(1) -> load(-1) -> save(1)) could otherwise land out of order on a threaded
    // server -- the STALE bytes winning, so a later Retract restores an abandoned
    // state.  Chaining each write on the previous one guarantees the last-issued write
    // is the last-written.  Each link ALWAYS resolves (errors are handled, never
    // re-thrown) so one failed write can't stall the chain.  (Same-class fix as the
    // anchor's prune-before-write ordering.)
    var _stateWriteChain = Promise.resolve();
    function _persistState(snapshotBlob, identity) {
        if (!root.fetch || !snapshotBlob) return;
        var idx = identity && identity.state_index;
        _stateWriteChain = _stateWriteChain.then(function () {
            _trace("http:write-state:issue", { idx: idx });
            return root.fetch("/api/state-timeline/write", {
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
        return root.fetch("/api/state-timeline/read", {
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
        return root.fetch("/api/state-timeline/prune", {
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
        hasRestorableSnapshot: hasRestorableSnapshot,
        onPersistError:        onPersistError,   // subscribe to non-blocking write failures
        STORAGE_KEY:           STORAGE_KEY,
    };

    // MERGE into any pre-existing ``workspace`` namespace, not replace it -- defensive, so a
    // plain ``= api`` can't clobber a slot some other module set first.  (The canvas-state
    // store used to mount ``workspace._canvasState`` here; it now lives on
    // ``molview._canvasState`` -- the 2026-07 carve keeps the workspace persistence-only --
    // so this merge is no longer load-bearing for it, but stays as a general safeguard.)
    root.molbuilder = root.molbuilder || {};
    root.molbuilder.workspace = Object.assign(
        root.molbuilder.workspace || {}, api);
    if (_runtime() && typeof _runtime().register === "function") {
        _runtime().register("workspace", api);
    }

    export { api as workspace };
