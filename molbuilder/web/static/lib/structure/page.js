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

    var _canvas = null;
    var _modal  = null;

    function _bind(canvasApi, modalApi) {
        if (!canvasApi || typeof canvasApi.setStructure !== "function") {
            throw new Error(
                "structure-page: canvas-state API missing");
        }
        if (!modalApi || typeof modalApi.confirmDiscardUnsaved !== "function") {
            throw new Error(
                "structure-page: warning-modal API missing");
        }
        _canvas = canvasApi;
        _modal  = modalApi;
    }

    /**
     * Replace the canvas with ``structure``, gated on the dirty
     * flag.  If the canvas is dirty, the user must confirm.
     *
     * @returns {Promise<{ok: bool, cancelled?: bool}>}
     */
    function loadIntoCanvas(structure, source) {
        if (!_canvas || !_modal) {
            return Promise.reject(new Error(
                "structure-page: not bound — call _bind() first"));
        }
        // Empty canvas — set directly; no warning.
        if (_canvas.isEmpty()) {
            _canvas.setStructure(structure, source);
            return Promise.resolve({ ok: true });
        }
        // Clean canvas — set directly; no warning.  The user has
        // saved (or just loaded) the current canvas; overwriting it
        // does not lose modifications.
        if (!_canvas.isDirty()) {
            _canvas.setStructure(structure, source);
            return Promise.resolve({ ok: true });
        }
        // Dirty canvas — ask before overwriting.
        return _modal.confirmDiscardUnsaved().then(function (proceed) {
            if (!proceed) {
                return { ok: false, cancelled: true };
            }
            _canvas.setStructure(structure, source);
            return { ok: true };
        });
    }

    function markDirtyAfterModification() {
        if (!_canvas) {
            throw new Error("structure-page: not bound");
        }
        _canvas.markDirty();
    }

    function markSavedTo(path) {
        if (!_canvas) {
            throw new Error("structure-page: not bound");
        }
        _canvas.markSaved(path);
    }

    function getCanvasSnapshot() {
        if (!_canvas) {
            throw new Error("structure-page: not bound");
        }
        return {
            isEmpty:      _canvas.isEmpty(),
            isDirty:      _canvas.isDirty(),
            structure:    _canvas.getStructure(),
            source:       _canvas.getSource(),
            lastSaveTo:   _canvas.getLastSavedTo(),
        };
    }

    function onCanvasChange(cb) {
        if (!_canvas) {
            throw new Error("structure-page: not bound");
        }
        return _canvas.onChange(cb);
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
        // Auto-bind to the production canvas + modal modules when
        // they're both present.  The page template loads
        // canvas-state.js + warning-modal.js BEFORE page.js so both
        // are mounted at this point; the bind is a single
        // wiring step the template doesn't have to repeat.
        if (root.molbuilder.structureCanvas
            && root.molbuilder.warningModal) {
            _bind(root.molbuilder.structureCanvas,
                  root.molbuilder.warningModal);
        }
        if (root.molbuilder.runtime
            && typeof root.molbuilder.runtime.register === "function") {
            root.molbuilder.runtime.register(
                "structure.page", api);
        }
    }
})(typeof window !== "undefined" ? window : globalThis);
