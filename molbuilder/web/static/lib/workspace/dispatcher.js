/* Workspace — the PERSISTENCE layer (workspace-contract.md).
 *
 * This module is ONLY session state + concealed file access.  It holds NO in-memory data model
 * and never interprets what it stores: the MolView data model (lib/molview/data-model.js) owns
 * the structure/selection/periodicity/frames and their format, serialises itself, and hands the
 * BYTES here to persist.  The workspace writes them format-blind.
 *
 * Public surface (mounted on ``window.molbuilder.workspace``):
 *   - persist(sessionBytes, draftBlob, identity)  -- write the consumer's serialised state now
 *                                                    (session mirror + on-disk transient draft).
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

    // The on-disk transient DRAFT (workspace-contract "update = the only automatic disk write").
    // ``draftBlob`` is the consumer's already-serialised working-copy blob; ``identity`` is the
    // key it is filed under (source path, or {workspace_id}).  Best-effort crash-safety.
    function _persistDraft(draftBlob, identity) {
        if (!root.fetch || !draftBlob) return;
        root.fetch("/api/workingcopy/update", {
            method:  "POST",
            headers: { "Content-Type": "application/json" },
            body:    JSON.stringify(Object.assign({}, identity || {}, { data: draftBlob })),
        }).catch(function () { /* draft is best-effort crash-safety */ });
    }

    /**
     * Persist the consumer's serialised state NOW — the single write-in.  The data model has
     * already debounced + serialised; this just writes the bytes, format-blind:
     *   sessionBytes -> the sessionStorage session mirror (fast same-tab reload)
     *   draftBlob    -> the on-disk transient working-copy draft, keyed by ``identity``
     */
    function persist(sessionBytes, draftBlob, identity) {
        if (!root.sessionStorage && !root.fetch) return;
        _persistToSession(sessionBytes);
        _persistDraft(draftBlob, identity);
    }

    var api = {
        persist:               persist,
        workspaceId:           workspaceId,
        readPersistedSnapshot: readPersistedSnapshot,
        mountRestoreTarget:    mountRestoreTarget,
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
