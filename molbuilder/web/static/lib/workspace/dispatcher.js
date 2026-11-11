/* Workspace — the session-persistence layer.
 *
 * MODULE: workspace persistence  (lib/workspace/; contract: docs/web/workspace.md).
 *   ONE file: this one. It is the whole module -- the transport to the state
 *   files, the per-tab and per-tag identity, and the non-blocking error surface.
 *   Server backend: POST /api/workspace-storage/{write,read,prune}
 *   the on-disk indexed STATE TIMELINE (`workspace.md` § 9).
 *
 * ROLE: session state + concealed file access ONLY.  Holds NO in-memory data model and never
 *   interprets what it stores.  The MolView data model (lib/molview/) owns the
 *   structure/selection/periodicity/frames + their format, serialises itself, and hands the BYTES
 *   here to persist; this layer writes them format-blind.  WHEN to save is the
 *   caller's; a save here is sent without waiting.
 *
 * EVERY CALL NAMES ITS TAG, AND ONLY ITS TAG (docs/web/workspace.md § 4). The tag
 * is who the bytes belong to -- "modify", "results:structure", "transport:form" --
 * and the files are named from it HERE, never by a caller: two tags are two sets
 * of files, so one saver cannot reach another's work, whatever it passes.
 *
 * It is an ARGUMENT, not something set beforehand: one value set beforehand and
 * used by every later write would let the second of two savers on a page silently
 * take the first one's writes. A tag passed with each call has no such window.
 *
 * USED BY (callers of window.molbuilder.workspace):
 *   - lib/molview/history.js -- a viewer's points and its draft (MolView owns
 *     WHEN to write); lib/molview/ui-context.js -- a viewer's look, the draft
 *     of `<owner>:ui`.  Every page that mounts MolView hands this object in.
 *   - the pages' own drafts: lib/spectra/core.js (the spectrum knobs),
 *     lib/transport/core.js (the form and the panel), modify/structure/page.js
 *     (the panel), structure-optimization/viewer.js (the panel).
 *   - lib/app-notifications.js, through onPersistError().
 *
 * Public surface (window.molbuilder.workspace) -- THE ONLY WAY IN. There is no
 * layer under it to reach past (workspace.md § 5).
 *   - persist(tag, bytes, step)      -- a numbered point. Does not wait; `true` means sent.
 *   - readState(tag, step)           -- a numbered point back, or null.
 *   - pruneStatesAbove(tag, step)    -- drop the tag's points above ``step``
 *                                       (-1: every point).
 *   - persistDraft(tag, bytes)       -- the tag's draft: one file, rewritten each time.
 *   - readDraft(tag)                 -- the tag's draft back, or null.
 *   - onPersistError(fn)             -- subscribe to the store's failures: a write, a prune, a read.
 */
"use strict";

// THE ONE READER of the server's answers (ui-contract.md § 12).
import { ask } from "../server.js";

const root = (typeof window !== "undefined") ? window : globalThis;

    function _runtime() {
        return (root.molbuilder && root.molbuilder.runtime) ? root.molbuilder.runtime : null;
    }


    // ─── Session identity, per browser tab and tag (workspace.md § 4, § 9) ─ //
    //
    /* EACH BROWSER TAB ITS OWN FILES (the user, 2026-11-11: "same module,
     * one set per tab").  A tab's id is made once and kept IN that tab
     * (`pagePrefs`, a value of the tab's session, runtime.md § 3), so a
     * reload, or another page in the same tab, finds the same files; a new
     * tab makes its own, and a tab the browser duplicates starts with the
     * original's.  Twelve lowercase letters and digits, so `ws-<tab>-<tag>`
     * spells one tab and one tag.  A browser that refuses to keep it makes a
     * new one per load: the work is kept for the page, not across its
     * reload.  A closed tab's files stay in the folder, unread (workspace.md
     * § 9: the server warns once the folder passes 300 files). */
    const TAB_PREF = "workspace_tab";
    let _tab = null;

    function _tabId() {
        if (_tab) return _tab;
        const prefs = root.molbuilder.pagePrefs;
        const kept = prefs.read(TAB_PREF);
        if (kept) return (_tab = kept);
        const letters = "abcdefghijklmnopqrstuvwxyz0123456789";
        const bytes = new Uint8Array(12);
        root.crypto.getRandomValues(bytes);
        _tab = Array.from(bytes, (b) => letters[b % letters.length]).join("");
        prefs.write(TAB_PREF, _tab);
        return _tab;
    }

    /* THE ID IS THE TAB'S AND THE TAG'S, MADE SAFE FOR A FILE NAME.
     *
     * The state files are `<workspace_id>.<step>.wc.json`.  The server allows
     * letters, digits, `_` and `-` in an id — no dots, so the index stays
     * unambiguous.  TWO TAGS ARE NEVER ONE ID (workspace.md § 4): each
     * character of the tag has one spelling and no two share one -- a letter
     * or a digit as it is, a `:` as `-`, a `-` as `_d`, a `_` as `_u`, any
     * other as `_x` and its six hex digits.
     */
    function workspaceId(tag) {
        _requireTag(tag);
        return "ws-" + _tabId() + "-" + Array.from(tag).map((c) => {
            if (/[A-Za-z0-9]/.test(c)) return c;
            if (c === ":") return "-";
            if (c === "-") return "_d";
            if (c === "_") return "_u";
            return "_x" + c.codePointAt(0).toString(16).padStart(6, "0");
        }).join("");
    }

    /* EVERY CALL NAMES ITS TAG (workspace.md § 4), and a missing one is an error
     * rather than a default. A default would be a shared slot wearing the look of
     * an isolated one — the failure the tag exists to prevent, arrived at by
     * omission instead of by collision.  NOR DOES A TAG END IN ``:draft``: a
     * tag's draft is its id with ``-draft``, which ``:draft`` would spell the
     * same, so that tag's points would be another tag's draft. */
    function _requireTag(tag) {
        if (!tag || typeof tag !== "string") {
            throw new Error(
                "workspace: every save and load names its tag");
        }
        if (/:draft$/.test(tag)) {
            throw new Error(
                "workspace: a tag never ends in ':draft' -- that is the name "
                + "of another tag's draft");
        }
    }

    /* WHERE A TAG'S BYTES LIVE -- the one place a file is named
     * (`<workspace_id>.<state_index>.wc.json`, workspace.md § 9): a numbered
     * point under the tag's id, its draft under the id with ``-draft`` at 0. */
    function _point(tag, step) {
        return { workspace_id: workspaceId(tag), state_index: step };
    }
    function _draft(tag) {
        return { workspace_id: workspaceId(tag) + "-draft", state_index: 0 };
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
    // incomplete").  A failure is any answer the one reader returns not ok
    // (ui-contract.md § 12): a refusal, an unreadable answer, no answer.
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
        try { root.console.error("[workspace] store FAILED (non-blocking)", detail); }
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
    // A failed SAVE is surfaced (above) and a persist-error banner is RAISED.
    // A write that lands announces itself, and the UI layer clears the
    // warning.  Emitted ONLY on a transition out of the failed state, because
    // a per-write event on a healthy session is noise on every keystroke.
    // A failed READ is no part of it: a save that lands later does not put
    // back what the read could not.
    function _saveFailed(detail) {
        _persistFailing = true;
        _reportPersistError(detail);
    }
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

    // Serialise state writes in ISSUE ORDER, ONE QUEUE PER WORKSPACE ID
    // (workspace.md § 9.2).  The writes are fire-and-forget, so two POSTs to
    // the SAME <workspace_id>.<state_index> file (a rapid save(1) -> load(-1)
    // -> save(1)) could otherwise land out of order on a threaded server -- the
    // STALE bytes winning, so a later Retract restores an abandoned state; and
    // the delete of a tag's abandoned points must finish before its next point
    // is written.  Both are about ONE id's files, so each id has its queue:
    // a tag's draft (an id of its own, `-draft`) never waits behind its
    // points' writes or their delete -- an edit's draft held there was lost
    // to a reload that came first (2.69.26-2.69.28: the Molbuilder tab showed
    // the file, not the edit).  Each link ALWAYS resolves (errors are handled,
    // never re-thrown) so one failed write can't stall its queue.
    var _queues = new Map();      // workspace_id -> its last queued request
    function _queued(workspace_id, send) {
        var done = (_queues.get(workspace_id) || Promise.resolve()).then(send);
        _queues.set(workspace_id, done);
        return done;
    }
    function _persistState(snapshotBlob, identity) {
        if (!root.fetch || !snapshotBlob) return;
        var idx = identity && identity.state_index;
        _queued(identity.workspace_id, function () {
            _trace("http:write-state:issue", { idx: idx });
            return ask("/api/workspace-storage/write", {
                method:  "POST",
                body:    Object.assign({}, identity || {}, { data: snapshotBlob }),
            }).then(function (res) {
                _trace("http:write-state:done", { idx: idx, status: res.status });
                if (!res.ok) {
                    _saveFailed({ op: "write-state", state_index: idx,
                                  status: res.status, error: res.error });
                } else {
                    _reportPersistOk({ op: "write-state", state_index: idx });
                }
            });
        });
    }

    /* ONE PLACE -- the server -- and the call does not wait for it.
     *
     * NOTHING USEFUL CAN COME BACK FROM HERE. The write is sent without
     * waiting, so a slow disk never stalls an edit, and whether it arrived is
     * not known yet. A failure turns up afterwards on `onPersistError`.
     * `true` means "sent", never "saved". */
    function _send(bytes, identity) {
        if (!root.fetch) return false;
        _persistState(bytes, identity);
        return true;
    }

    /** A numbered point of ``tag``'s: ``bytes`` as its point ``step``. */
    function persist(tag, bytes, step) {
        return _send(bytes, _point(tag, step));
    }

    /** ``tag``'s draft: ``bytes`` as its one file, rewritten each time. */
    function persistDraft(tag, bytes) {
        return _send(bytes, _draft(tag));
    }

    /** ``tag``'s point ``step`` back -- the parsed JSON, or null. */
    function readState(tag, step) {
        return _read(_point(tag, step));
    }

    /** ``tag``'s draft back -- the parsed JSON, or null. */
    function readDraft(tag) {
        return _read(_draft(tag));
    }

    /* Read the OPAQUE bytes at one file.  Resolves the parsed JSON, or null
     * when nothing is saved there or the read failed.  Format-blind -- the
     * caller interprets what comes back. */
    function _read(identity) {
        if (!root.fetch) return Promise.resolve(null);
        return ask("/api/workspace-storage/read", {
            method:  "POST",
            body:    identity,
        }).then(function (res) {
            if (res.ok) {
                return (res.body && res.body.data != null) ? res.body.data : null;
            }
            /* NOTHING SAVED IS AN ANSWER, ``data: null`` (the route's own rule),
             * so an answer not ok is a FAILURE: a file the server could not
             * read, a request refused, a server not reached.  The caller still
             * gets null -- "nothing to put back" is the only safe reading of it
             * (workspace.md § 5) -- and the failure is said, never passed off
             * as nothing saved. */
            _reportPersistError({ op: "read-state",
                                  workspace_id: identity.workspace_id,
                                  state_index: identity.state_index,
                                  status: res.status, error: res.error });
            return null;
        });
    }

    /**
     * Drop ``tag``'s points above ``step`` (a save after a retract drops the
     * abandoned tail); ``step === -1`` drops every point.  Its draft is a file
     * of its own and stays.  Best-effort; resolves when the server has acted.
     *
     * IN THE WRITE QUEUE (workspace.md § 9.2, *clear the tail before
     * re-anchoring*): the delete finishes before any write asked after it is
     * sent, so a slow delete never erases the point written next.
     */
    function pruneStatesAbove(tag, step) {
        const workspace_id = _point(tag, step).workspace_id;
        if (!root.fetch) return Promise.resolve();
        return _queued(workspace_id, function () {
            _trace("http:prune-states:issue", { above: step });
            return ask("/api/workspace-storage/prune", {
                method:  "POST",
                body:    { workspace_id: workspace_id, above_index: step },
            });
        }).then(function (res) {
            _trace("http:prune-states:done", { above: step, status: res.status });
            if (!res.ok) {
                // A failed prune leaves a stale tail, not a lost save.
                _saveFailed({ op: "prune-states", above_index: step,
                              status: res.status, error: res.error });
            }
            return res;   // resolve either way: the anchor write still proceeds
        });
    }

    var api = {
        persist:               persist,
        readState:             readState,
        pruneStatesAbove:      pruneStatesAbove,
        persistDraft:          persistDraft,
        readDraft:             readDraft,
        onPersistError:        onPersistError,   // subscribe to the store's failures
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
