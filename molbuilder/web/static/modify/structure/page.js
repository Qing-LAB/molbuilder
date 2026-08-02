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
 *     -> void.  A NO-OP, kept only while the modifier panels still
 *     call it: the edit raised the unsaved badge itself, inside the
 *     viewer's gate, when it landed (molview.md § 11.2).
 *
 *   markSavedTo(path)
 *     -> void.  Records where the structure was last saved TO.  The
 *     save handler calls it after a successful project write.  It
 *     does NOT clear anything: whether there is unsaved work is the
 *     viewer's own answer (``uncommitted``), and where a file went is
 *     the page's, because the viewer tracks contents, not files
 *     (molview.md § 6.7).
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
 * Design ref: docs/web/tabs.md (no auto-load on
 * sidebar selection) + § 5.4 (warning-modal contract).
 */
(function (root) {
    "use strict";

    // The orchestrator works against the viewer this page mounted, through the
    // surface molview.md § 9.3 lists: installMolecule, getStructure, uncommitted,
    // subscribe.  Its own public methods are unchanged, so the panels calling it
    // see no difference.
    /* THE VIEWER THIS PAGE MOUNTED, handed over by modify/selection-bootstrap.js
     * once it has one (`useViewer` below). This is a classic script — it loads
     * before that file and cannot import — so being TOLD is the only way it can
     * have a viewer at all.
     *
     * It used to look one up in `window.molbuilder.molview.data`, which MolView
     * has published nothing to since it was rebuilt, so every load, every dirty
     * check and every save gate on this tab was reading `undefined`. */
    var _viewer    = null;
    var _modal     = null;   // TEST override (set by _bind); production looks up (below)

    // molview.data + warningModal are LOOKED UP at call time (molview-module.md §D.0): read the
    // LIVE model through the door, never import/auto-bind it (the molview module is deferred, so it
    // is not published when this classic script loads).  A test injects stubs via _bind.
    function _model()  { return (_viewer && _viewer.ok) ? _viewer.data : null; }
    function _mod() { return _modal || (root.molbuilder && root.molbuilder.warningModal) || null; }

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
        _viewer = { ok: true, data: workspaceApi };
        _modal  = modalApi;
    }

    /**
     * Replace the canvas with ``structure``, gated on the dirty
     * flag.  If the canvas is dirty, the user must confirm.
     *
     * @returns {Promise<{ok: bool, cancelled?: bool}>}
     */
    function loadIntoCanvas(structure, source) {
        if (!_model() || !_mod()) {
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
            return _model().installMolecule({
                text:        structure.text,
                filename:    filename,
                source:      source || null,
                periodicity: structure.periodicity || null,
                annotations: structure.annotations || null,
                // Sidecar-enriched atoms (a project-file open resolved these via
                // the parser door) ride IN so the load installs the FINAL per-atom
                // state in ONE write -- the caller must NOT follow up with a second
                // store write (see installMolecule's load contract).  Omitted by
                // generators / raw-text loads.
                atoms:       structure.atoms || null,
            }).then(function () { return { ok: true }; });
        }
        // Nothing loaded — load directly; no warning. A read answers nothing
        // when there is nothing, which is a different answer from a structure
        // with no atoms (molview.md § 9.3).
        if (_model().getStructure() === null) {
            return _apply();
        }
        // No unsaved work — load directly; no warning.  The user has saved (or
        // just loaded) what is there, so overwriting it loses nothing.
        if (!_model().uncommitted) {
            return _apply();
        }
        // Dirty canvas — ask before overwriting.
        return _mod().confirmDiscardUnsaved().then(function (proceed) {
            if (!proceed) {
                return { ok: false, cancelled: true };
            }
            return _apply();
        });
    }

    /* NEITHER OF THESE MARKS THE VIEWER ANY MORE, and both are kept only so the
     * panels that call them keep working while the tab is rewired.
     *
     * "There is unsaved work here" is the viewer's own answer, raised inside its
     * gate after a change lands and cleared when a state is saved (molview.md
     * § 11.2) — not a flag set from outside. And where a structure was saved TO
     * is a fact about a file operation the page performed, so the page keeps it
     * (§ 6.7); the viewer never knew it. */
    var _lastSavedTo = null;

    function markDirtyAfterModification() {
        // Nothing to do: the edit itself raised the badge.
    }

    function markSavedTo(path) {
        _lastSavedTo = path || null;
    }

    function getCanvasSnapshot() {
        if (!_model()) {
            throw new Error("structure-page: not bound");
        }
        var structure = _model().getStructure();
        return {
            isEmpty:      structure === null,
            isDirty:      !!_model().uncommitted,
            structure:    structure,
            // The page's own note, not the viewer's: the viewer tracks contents,
            // not files (molview.md § 6.7).
            lastSaveTo:   _lastSavedTo,
        };
    }

    function onCanvasChange(cb) {
        if (!_model()) {
            throw new Error("structure-page: not bound");
        }
        return _model().subscribe(cb);
    }

    var api = {
        _bind:                      _bind,
        // The production door: the page hands over the viewer it mounted.
        useViewer:                  function (viewer) { _viewer = viewer || null; },
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
        if (root.molbuilder.runtime
            && typeof root.molbuilder.runtime.register === "function") {
            root.molbuilder.runtime.register(
                "structure.page", api);
        }
    }
})(typeof window !== "undefined" ? window : globalThis);
