/* Molbuilder-tab page orchestrator.
 *
 * Owns the "load a structure into the canvas" gate: every Sources
 * panel (Load from project, Generate from SMILES, future 3DNA /
 * peptide / name / file generators) calls in here instead of
 * touching ``canvas-state.setStructure`` directly.  The gate's
 * job is the unsaved-modifications check —
 *
 *   if canvas is empty → set immediately
 *   if canvas is clean → set immediately
 *   if canvas is dirty → ask the user via warning-modal, set
 *                         only on "Discard and continue"
 *
 * Public surface (mounted on ``window.molbuilder.structurePage``):
 *
 *   loadIntoCanvas(structure, source)
 *     -> Promise<{ok: bool, cancelled?: bool}>
 *
 *     ``structure``: ``{source_format: "xyz"|"pdb", text: string}``
 *     ``source``:    ``{kind, file?, generator_input?}``
 *       (see canvas-state.js for the source schema)
 *
 *     Returns ``{ok: true}`` when the canvas was updated, or
 *     ``{ok: false, cancelled: true}`` when the user picked Cancel
 *     on the warning modal.  Caller branches on ``cancelled`` to
 *     decide whether to surface "no changes" status text vs. a
 *     real error.
 *
 *   markDirtyAfterModification()
 *     -> void.  Sets ``canvas.dirty = true``.  Modifier panels
 *     (delete / add / orient / electrode / geom) call after a
 *     successful op so the next Load/Generate triggers the
 *     warning.
 *
 *   markSavedTo(path)
 *     -> void.  Clears ``dirty`` + records ``last_save_to``.  Save
 *     handler calls after a successful project write.
 *
 *   getCanvasSnapshot()
 *     -> the full canvas snapshot from canvas-state.  Read-only;
 *     caller-side mutations don't leak through.
 *
 *   onCanvasChange(cb) -> unsubscribe()
 *     Subscribe to canvas state changes.  Direct passthrough to
 *     canvas-state.onChange — callers can use this OR subscribe
 *     to canvas-state directly; the page module re-exposes it so
 *     a future change to the orchestrator's notification model
 *     has one consumer-facing seam.
 *
 * Used by:
 *   - The Sources panels (Load, Generate, ...).
 *   - The Save-to-project handler.
 *   - Existing modify-tab modifier ops (after migration; see B.3).
 *
 * Design ref: docs/tabs/architecture.md § 5.2 (no auto-load on
 * sidebar selection) + § 5.4 (warning-modal contract).
 */
(function (root) {
    "use strict";

    // Phase 10 (workspace-contract.md §3): the orchestrator now
    // binds against the unified ws.* surface, not the legacy
    // structureCanvas store.  Public methods unchanged so existing
    // consumers see no API drift; internals use ws.installStructure
    // / ws.markDirty / ws.markSaved / ws.isEmpty / ws.isDirty /
    // ws.getStructure / ws.getSource / ws.getLastSavedTo / ws.subscribe.
    var _workspace = null;
    var _modal     = null;

    function _bind(workspaceApi, modalApi) {
        if (!workspaceApi
                || typeof workspaceApi.installStructure !== "function") {
            throw new Error(
                "structure-page: ws.* API missing (installStructure)");
        }
        if (!modalApi
                || typeof modalApi.confirmDiscardUnsaved !== "function") {
            throw new Error(
                "structure-page: warning-modal API missing");
        }
        _workspace = workspaceApi;
        _modal     = modalApi;
    }

    /**
     * Replace the canvas with ``structure``, gated on the dirty
     * flag.  If the canvas is dirty, the user must confirm.
     *
     * @returns {Promise<{ok: bool, cancelled?: bool}>}
     */
    function loadIntoCanvas(structure, source) {
        if (!_workspace || !_modal) {
            return Promise.reject(new Error(
                "structure-page: not bound — call _bind() first"));
        }
        // Empty canvas — set directly; no warning.
        if (_workspace.isEmpty()) {
            _workspace.installStructure(structure, source);
            return Promise.resolve({ ok: true });
        }
        // Clean canvas — set directly; no warning.  The user has
        // saved (or just loaded) the current canvas; overwriting it
        // does not lose modifications.
        if (!_workspace.isDirty()) {
            _workspace.installStructure(structure, source);
            return Promise.resolve({ ok: true });
        }
        // Dirty canvas — ask before overwriting.
        return _modal.confirmDiscardUnsaved().then(function (proceed) {
            if (!proceed) {
                return { ok: false, cancelled: true };
            }
            _workspace.installStructure(structure, source);
            return { ok: true };
        });
    }

    function markDirtyAfterModification() {
        if (!_workspace) {
            throw new Error("structure-page: not bound");
        }
        _workspace.markDirty();
    }

    function markSavedTo(path) {
        if (!_workspace) {
            throw new Error("structure-page: not bound");
        }
        _workspace.markSaved(path);
    }

    function getCanvasSnapshot() {
        if (!_workspace) {
            throw new Error("structure-page: not bound");
        }
        return {
            isEmpty:      _workspace.isEmpty(),
            isDirty:      _workspace.isDirty(),
            structure:    _workspace.getStructure(),
            source:       _workspace.getSource(),
            lastSaveTo:   _workspace.getLastSavedTo(),
        };
    }

    function onCanvasChange(cb) {
        if (!_workspace) {
            throw new Error("structure-page: not bound");
        }
        return _workspace.subscribe(cb);
    }

    var api = {
        _bind:                      _bind,
        loadIntoCanvas:             loadIntoCanvas,
        markDirtyAfterModification: markDirtyAfterModification,
        markSavedTo:                markSavedTo,
        getCanvasSnapshot:          getCanvasSnapshot,
        onCanvasChange:             onCanvasChange,
    };

    if (typeof module !== "undefined" && module.exports) {
        module.exports = api;
    } else {
        root.molbuilder = root.molbuilder || {};
        root.molbuilder.structurePage = api;
        // Auto-bind to the production workspace dispatcher + modal
        // when both are present.  Phase 10 (workspace-contract.md
        // §1): the page template loads canvas-state.js +
        // warning-modal.js + dispatcher.js (in that order) BEFORE
        // page.js so all three are mounted at this point.  The
        // dispatcher is what page.js binds against; canvas-state is
        // now the dispatcher's internal implementation.
        if (root.molbuilder.workspace
            && root.molbuilder.warningModal) {
            _bind(root.molbuilder.workspace,
                  root.molbuilder.warningModal);
        }
        if (root.molbuilder.runtime
            && typeof root.molbuilder.runtime.register === "function") {
            root.molbuilder.runtime.register(
                "structure.page", api);
        }
    }
})(typeof window !== "undefined" ? window : globalThis);
