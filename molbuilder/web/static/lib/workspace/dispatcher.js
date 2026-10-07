/* Workspace — the session-persistence layer.
 *
 * MODULE: workspace persistence  (lib/workspace/; contract: docs/web/workspace.md).
 *   ONE file: this one. It is the whole module -- the transport to the state
 *   files, the per-tag identity, and the non-blocking error surface.
 *   Server backend: POST /api/workspace-storage/{write,read,prune}
 *   the on-disk indexed STATE TIMELINE (`workspace.md` § 9).
 *
 * ROLE: session state + concealed file access ONLY.  Holds NO in-memory data model and never
 *   interprets what it stores.  The MolView data model (lib/molview/) owns the
 *   structure/selection/periodicity/frames + their format, serialises itself, and hands the BYTES
 *   here to persist; this layer writes them format-blind.  The suspend/resume + the
 *   "when the data changed" decision (PUSH-ONLY, no debounce) live in the data model —
 *   persist() here is a synchronous write of the bytes it is handed.
 *
 * EVERY SAVE AND LOAD NAMES ITS TAG (docs/web/workspace.md § 4). The tag is who the
 * bytes belong to -- "modify", "results:structure", "results:trajectory" -- and it
 * becomes the workspace_id the state files are named after. Two tags are two ids
 * and two sets of files, so one saver cannot reach another's work.
 *
 * It is an ARGUMENT, not something set beforehand: one value set beforehand and
 * used by every later write would let the second of two savers on a page silently
 * take the first one's writes. A tag passed with each call has no such window.
 *
 * USED BY (callers of window.molbuilder.workspace):
 *   - lib/molview/history.js — the state save/retract TIMELINE: persist(), readState(),
 *     pruneStatesAbove(), workspaceId(). The primary consumer; it owns WHEN to write.
 *   - tabs that mount a molview hand window.molbuilder.workspace to molview.mount:
 *     modify/, spectra/viewer.js, transport/core.js, structure-optimization/viewer.js,
 *     molview/demo.js, lib/inspectors/structure.js.
 *   - a UI notification layer subscribes to onPersistError() to warn on a failed disk write.
 *
 * Public surface (window.molbuilder.workspace) -- THE ONLY WAY IN. There is no
 * layer under it to reach past (workspace.md § 4).
 *   - persist(tag, bytes, identity)  -- send the consumer's serialised state to its
 *                                 state file. Does not wait; `true` means sent.
 *   - readState(identity)      -- read the opaque snapshot bytes at {workspace_id, state_index}
 *                                 from disk (a history index popState navigates to), or null.
 *   - pruneStatesAbove(workspace_id, index)  -- tail-delete on-disk state files above ``index``
 *                                 (index === -1 clears the whole timeline).
 *   - workspaceId(tag)         -- the stable id this tag's draft is keyed under.
 *   - onPersistError(fn)       -- subscribe to non-blocking disk-write failures.
 */
"use strict";


const root = (typeof window !== "undefined") ? window : globalThis;

    function _runtime() {
        return (root.molbuilder && root.molbuilder.runtime) ? root.molbuilder.runtime : null;
    }


    // ─── Session identity, per tag (workspace.md § 4, § 9) ─────────── //
    //
    // A stable id for a tag's draft, so repeated updates hit the SAME state files
    // and a same-tab reload keeps them (not a fresh orphan each time).
    /* THE ID IS THE TAG, MADE SAFE FOR A FILE NAME.
     *
     * The state files are `<workspace_id>.<step>.wc.json`, and the id has to be
     * the same every time this tag comes back — otherwise a reopened page looks
     * for its work under a name it has never used and finds nothing. Working it
     * out from the tag makes that true by construction, with nothing remembered
     * anywhere.
     *
     * TWO CONSEQUENCES, both deliberate. Two browser windows open on the same
     * page share one saved workspace, because they are two windows onto the
     * same work rather than two workspaces that happen to look alike. And the
     * file pile is bounded: no closed tab abandons its timeline under an id
     * nothing would ever use again.
     *
     * The server allows letters, digits, `_` and `-` in an id — no dots, so the
     * index stays unambiguous — so anything else in a tag becomes a dash.
     */
    function workspaceId(tag) {
        _requireTag(tag);
        return "ws-" + tag.replace(/[^A-Za-z0-9_-]+/g, "-");
    }

    /* EVERY CALL NAMES ITS TAG (workspace.md § 4), and a missing one is an error
     * rather than a default. A default would be a shared slot wearing the look of
     * an isolated one — the failure the tag exists to prevent, arrived at by
     * omission instead of by collision. */
    function _requireTag(tag) {
        if (!tag || typeof tag !== "string") {
            throw new Error(
                "workspace: every save and load names its tag (workspace.md § 4)");
        }
    }

    // ─── The write (format-blind) ─────────────────────────────────── //
    // The on-disk indexed STATE FILE (`workspace.md` § 9, the state files
    // on the server and their two ordering rules).
    // ``snapshotBlob`` is the consumer's already-serialised OPAQUE session snapshot; ``identity``
    // = {workspace_id, state_index} keys the filename ``<workspace_id>.<state_index>.wc.json``.
    // The server stores it FORMAT-BLIND (never through the structure codec).  Best-effort.
    // Ordered event tracer (diagnostic; no-op unless window.__MV_TRACE).
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
    // model is the source of truth), BUT a failure
    // is NEVER swallowed: it is reported to the console AND emitted as a
    // ``molbuilder:persist-error`` DOM event so a UI layer can warn the user
    // ("state didn't reach disk; retract history / crash recovery may be
    // incomplete").  A failure is either a rejected fetch (network) OR a
    // non-2xx response (server refused, e.g. bad workspace_id / disk).
    var _persistErrorHandlers = [];
    var _persistFailing = false;   // see Recovery, below
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
        _persistFailing = true;
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

    // ---- Recovery -------------------------------------------------------- //
    // A failure is surfaced (above) and a persist-error banner is RAISED.
    // A write that lands announces itself, and the UI layer clears the
    // warning.  Emitted ONLY on a transition out of the failed state, because
    // a per-write event on a healthy session is noise on every keystroke.
    function _reportPersistOk(detail) {
        if (!_persistFailing) return;      // nothing to take back
        _persistFailing = false;
        try {
            if (root.dispatchEvent && typeof root.CustomEvent === "function") {
                root.dispatchEvent(new root.CustomEvent(
                    "molbuilder:persist-ok", { detail: detail }));
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
            return root.fetch("/api/workspace-storage/write", {
                method:  "POST",
                headers: { "Content-Type": "application/json" },
                body:    JSON.stringify(Object.assign({}, identity || {}, { data: snapshotBlob })),
            }).then(function (res) {
                _trace("http:write-state:done", { idx: idx, status: res && res.status });
                if (!res || !res.ok) {
                    _reportPersistError({ op: "write-state", state_index: idx,
                                          status: res && res.status });
                } else {
                    _reportPersistOk({ op: "write-state", state_index: idx });
                }
            }).catch(function (err) {
                _trace("http:write-state:error", { idx: idx });
                _reportPersistError({ op: "write-state", state_index: idx,
                                      error: (err && err.message) || String(err) });
            });
        });
    }

    /**
     * Send the consumer's serialised state to its state file.  The consumer owns
     * WHEN to call this; here it is only bytes, moved without being read:
     *   bytes    -> ``<workspace_id>.<state_index>.wc.json``, keyed by ``identity``
     *               ({workspace_id, state_index}).
     */
    function persist(tag, bytes, identity) {
        _requireTag(tag);
        if (!root.fetch) return false;
        /* ONE PLACE -- the server -- and the call does not wait for it.
         *
         * NOTHING USEFUL CAN COME BACK FROM HERE. The write is sent without
         * waiting, so a slow disk never stalls an edit, and whether it arrived is
         * not known yet. A failure turns up afterwards on `onPersistError`.
         * `true` means "sent", never "saved". */
        _persistState(bytes, identity);
        return true;
    }

    /**
     * Read the OPAQUE snapshot bytes on disk at {workspace_id, state_index} (§4.7 read-by-index),
     * what the history calls to fetch a saved index.  Resolves
     * the parsed JSON, or null when the file is missing / unreadable.  Format-blind — the data
     * model interprets what comes back.
     */
    function readState(identity) {
        if (!root.fetch || !identity) return Promise.resolve(null);
        return root.fetch("/api/workspace-storage/read", {
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
        return root.fetch("/api/workspace-storage/prune", {
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
            // Resolve (undefined) so a caller's ordered write still runs;
            // a failed prune leaves a stale tail, not a lost save.
        });
    }

    var api = {
        persist:               persist,
        readState:             readState,
        pruneStatesAbove:      pruneStatesAbove,
        workspaceId:           workspaceId,
        onPersistError:        onPersistError,   // subscribe to non-blocking write failures
    };

    // MERGE into any pre-existing ``workspace`` namespace, not replace it -- defensive, so a
    // plain ``= api`` can't clobber a slot some other module set first.
    root.molbuilder = root.molbuilder || {};
    root.molbuilder.workspace = Object.assign(
        root.molbuilder.workspace || {}, api);
    if (_runtime() && typeof _runtime().register === "function") {
        _runtime().register("workspace", api);
    }

    export { api as workspace };
