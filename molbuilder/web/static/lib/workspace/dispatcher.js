/* Workspace — the session-persistence layer.
 *
 * MODULE: workspace persistence  (lib/workspace/; contract: docs/web/workspace.md).
 *   ONE file: this one. It is the whole module -- the transport to the state
 *   files, the per-tag identity, and the non-blocking error surface.
 *   Server backend: POST /api/workspace-storage/{write,read,prune}
 *   the on-disk indexed STATE TIMELINE (`workspace.md` § 9).
 *
 * ROLE: session state + concealed file access ONLY.  Holds NO in-memory data model and never
 *   interprets what it stores.  The MolView data model (lib/molview/data-model.js) owns the
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
 * It is an ARGUMENT, not something set beforehand. There used to be
 * ``useNamespace(owner)``, which set one value that every later write then used --
 * so on a page with two savers the second to set it silently took the first one's
 * writes, and neither found out until one read back something it never wrote. A
 * tag passed with each call has no such window.
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
 *   - STORAGE_KEY is gone with the browser copy: there is no browser-side slot.
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
    //
    // REMEMBERED PER TAG, and that is load-bearing rather than tidy. The id is
    // what the state files are named after -- <workspace_id>.<step>.wc.json --
    // so one shared memory would hand the first tag's id to every tag that asked
    // afterwards, and two savers would write over each other's numbered history
    // while their browser copies stayed properly apart. Isolation has to hold for
    // the identity as well as for the content; holding for only one of them is
    // broken in the half that survives a crash, which is the half the timeline
    // exists for.
    //
    // This replaced a single `_workspaceId` variable that `useNamespace(owner)`
    // reset. That worked only because a page had one saver: the setter was the
    // thing keeping the cache honest, so removing it without this would have
    // taken the guarantee away in silence.
    /* THE ID IS THE TAG, MADE SAFE FOR A FILE NAME.
     *
     * The state files are `<workspace_id>.<step>.wc.json`, and the id has to be
     * the same every time this tag comes back — otherwise a reopened page looks
     * for its work under a name it has never used and finds nothing. Working it
     * out from the tag makes that true by construction, with nothing remembered
     * anywhere.
     *
     * It used to be a random string minted per browser tab and kept in
     * sessionStorage so a reload could find it again. That went with the
     * sessionStorage copy: there is one place work is kept now, and it is on the
     * server.
     *
     * TWO CONSEQUENCES, both deliberate. Two browser windows open on the same
     * page now share one saved workspace, because they are two windows onto the
     * same work rather than two workspaces that happen to look alike. And the
     * file pile is bounded: a closed tab used to abandon its whole timeline under
     * an id nothing would ever use again, which is why this directory grew until
     * an operator deleted it by hand.
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

    /* THERE IS NO `hasRestorableSnapshot` OR `mountRestoreTarget` HERE ANY MORE,
     * and their absence is the rule rather than an omission.
     *
     * They answered two questions a tab asks when a page loads: "have I got work
     * saved here worth bringing back?" and "which project file was it from?" This
     * module answered them by opening the saved bytes and looking for a molecule
     * inside — which it is documented not to do, and which it got wrong: a
     * molecule you build from SMILES has no file it came from, so the second one
     * reported "nothing saved" and the tab wiped work that was there.
     *
     * Both are the TAB's questions about the TAB's own bytes. It wrote them, so it
     * knows what is in them: it reads its own state file back and looks. And
     * `mountRestoreTarget` handed out a PROJECT FILE PATH, which belongs to the
     * projects module and has no business being read out of here at all.
     */

    // ─── The write (format-blind) ─────────────────────────────────── //
    // The on-disk indexed STATE FILE (`workspace.md` § 9, the state files
    // on the server and their two ordering rules).
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
    // Nothing used to lower it: there was no success signal at all, so once a
    // write failed the warning stood for the life of the page -- through the
    // backend coming back, through every later write succeeding -- and its
    // "(xN)" only ever grew.  Measured 2026-09-05, after a server outage: the
    // route was answering 200 again while the banner still said saving was
    // broken, and the only way out was to dismiss it by hand.
    //
    // So a write that lands announces itself, and the UI layer clears the
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
        /* ONE PLACE, and the call does not wait for it. There used to be a second
         * copy in the browser's own storage, written on the way past; it was
         * dropped because nothing ever read it back to restore anything — every
         * restore went to the server anyway — so it cost a write per edit and
         * bought nothing.
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
     * what popState calls to fetch a *history* index the session mirror no longer holds.  Resolves
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
            // Resolve (undefined) so _anchorTimeline's ordered write still runs;
            // a failed prune leaves a stale tail, not a lost anchor.
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
