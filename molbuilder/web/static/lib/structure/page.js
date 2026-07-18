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

    // The orchestrator binds against MolView's unified DATA model
    // (``molbuilder.molview.data``) for DATA ops — load / markDirty /
    // markSaved / isEmpty / isDirty / getStructure / getSource /
    // getLastSavedTo / subscribe.  ``molbuilder.workspace`` is the
    // persistence layer only and is no longer bound here.  Public
    // methods are unchanged so existing consumers see no API drift.
    var _workspace = null;
    var _modal     = null;

    function _bind(workspaceApi, modalApi) {
        if (!workspaceApi
                || typeof workspaceApi.installMolecule !== "function") {
            throw new Error(
                "structure-page: molview.data API missing (installMolecule)");
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
        // The ONE atomic whole-model load door (molview-module.md
        // §19.3): it parses the text, replaces the whole model
        // (canvas + atoms + render) and anchors the undo timeline at
        // index 0.  Provenance + sidecar ride ON the door (no
        // side-channel): the generator ``source`` (kind /
        // generator_input) and any sidecar ``periodicity`` /
        // ``annotations`` the caller resolved (that /api/build/load
        // can't re-derive) are forwarded so the model keeps them.
        var filename = (source && source.file) || null;
        function _apply() {
            return _workspace.installMolecule({
                text:        structure.text,
                filename:    filename,
                source:      source || null,
                periodicity: structure.periodicity || null,
                annotations: structure.annotations || null,
                // Sidecar-enriched atoms (a project-file open resolved these via
                // readWorkingCopy) ride IN so the load installs the FINAL per-atom
                // state in ONE write -- the caller must NOT follow up with a second
                // store write (see installMolecule's load contract).  Omitted by
                // generators / raw-text loads.
                atoms:       structure.atoms || null,
            }).then(function () { return { ok: true }; });
        }
        // Empty canvas — load directly; no warning.
        if (_workspace.isEmpty()) {
            return _apply();
        }
        // Clean canvas — load directly; no warning.  The user has
        // saved (or just loaded) the current canvas; overwriting it
        // does not lose modifications.
        if (!_workspace.isDirty()) {
            return _apply();
        }
        // Dirty canvas — ask before overwriting.
        return _modal.confirmDiscardUnsaved().then(function (proceed) {
            if (!proceed) {
                return { ok: false, cancelled: true };
            }
            return _apply();
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
        // Auto-bind to MolView's data model + the warning modal when
        // both are present.  The page template loads the molview
        // data-model (+ its stores) and warning-modal.js BEFORE
        // page.js, so both are mounted at this point.  page.js binds
        // against ``molview.data`` for DATA ops; ``workspace`` is the
        // persistence layer and is not bound here.
        if (root.molbuilder.molview
            && root.molbuilder.molview.data
            && root.molbuilder.warningModal) {
            _bind(root.molbuilder.molview.data,
                  root.molbuilder.warningModal);
        }
        if (root.molbuilder.runtime
            && typeof root.molbuilder.runtime.register === "function") {
            root.molbuilder.runtime.register(
                "structure.page", api);
        }
    }
})(typeof window !== "undefined" ? window : globalThis);
