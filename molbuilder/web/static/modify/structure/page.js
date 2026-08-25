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
            }).then(function () {
                /* AND THE PAGE RECORDS WHAT IT JUST DID. This is the one gate
                 * every generator and the file upload come through, and it
                 * already knows whether a file is behind the structure: a
                 * SMILES/DNA/RNA/peptide/name build passes no `file`, so the
                 * note becomes null and the loader readout stops claiming a file
                 * that never existed.
                 *
                 * (The sidebar's own load does not come through here -- it goes
                 * via `projects.parser.openMolecule` -- and records the same
                 * note at its own gate.) */
                markLoadedFrom(filename);
                return { ok: true };
            });
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
    var _loadedFrom  = null;

    /* ── The page's own two facts, kept under the page's own tag ────────────
     *
     * workspace.md § 4: a page can have more than one thing worth keeping, and
     * "the Modify tab has a viewer holding a molecule AND its own panel state".
     * The tag it names for that is `modify:panel`. The viewer saves under
     * `modify`; these are two tags, so they are two slots and neither can reach
     * the other.
     *
     * § 6 says the rest: say the tag on every call, decide what goes in the
     * bytes and be able to read them back, and decide when to save. So this
     * writes at the moments the page CHANGES one of these — a load, a generate,
     * a save — and never on a timer.
     *
     * ONE WRITER FOR ONE SLOT. Both facts live here and both are written by this
     * one function, because two writers on a single state file is how one of
     * them silently drops the other's field.
     */
    var PANEL_TAG = "modify:panel";

    function _ws() {
        return (root.molbuilder && root.molbuilder.workspace) || null;
    }

    /* THE NOTE HAS ITS OWN READERS, so it needs its own channel.
     *
     * The viewer's `subscribe` announces a structure change from INSIDE
     * `installMolecule`, before the promise that call returns has resolved -- so
     * a readout listening there re-renders while this note still holds the
     * PREVIOUS load's filename, and nothing tells it to look again afterwards.
     * That is how a molecule generated from SMILES came to sit under
     * "Loaded: water.xyz".
     *
     * This is the page's own state with the page's own readers, so the channel
     * is the page's too -- not something asked of the viewer, which has no
     * business knowing a file was involved (molview.md § 6.7). */
    var _panelListeners = [];

    function _rememberPanel() {
        var ws = _ws();
        if (ws && typeof ws.persist === "function") {
            ws.persist(PANEL_TAG,
                       { v: 1, loadedFrom: _loadedFrom, lastSavedTo: _lastSavedTo },
                       { workspace_id: ws.workspaceId(PANEL_TAG), state_index: 0 });
        }
        _panelListeners.slice().forEach(function (fn) {
            try { fn({ loadedFrom: _loadedFrom, lastSavedTo: _lastSavedTo }); }
            catch (_) { /* one bad reader cannot muzzle the rest */ }
        });
    }

    function onPanelChange(cb) {
        if (typeof cb !== "function") return function () {};
        _panelListeners.push(cb);
        return function () {
            var at = _panelListeners.indexOf(cb);
            if (at >= 0) _panelListeners.splice(at, 1);
        };
    }

    /* Read the page's own note back. Version-stamped like every other state file:
     * these outlive the code that wrote them, and bytes from a layout this build
     * has never seen are not something to guess at. */
    async function restorePanelNote() {
        var ws = _ws();
        if (!ws || typeof ws.readState !== "function") return null;
        var saved;
        try {
            saved = await ws.readState({
                workspace_id: ws.workspaceId(PANEL_TAG), state_index: 0,
            });
        } catch (_) {
            return null;
        }
        if (!saved || saved.v !== 1) return null;
        _loadedFrom  = saved.loadedFrom  || null;
        _lastSavedTo = saved.lastSavedTo || null;
        return { loadedFrom: _loadedFrom, lastSavedTo: _lastSavedTo };
    }

    function markDirtyAfterModification() {
        // Nothing to do: the edit itself raised the badge.
    }

    function markSavedTo(path) {
        _lastSavedTo = path || null;
        _rememberPanel();
    }

    /* Which file is on the canvas -- or null when what is on it came from a
     * generator and has no file behind it at all. The page knows because the
     * page performed the load; it never asks the viewer, which tracks contents
     * and not files (molview.md § 6.7). */
    function markLoadedFrom(path) {
        _loadedFrom = path || null;
        /* A LOAD ALSO RETIRES THE SAVE TARGET.
         *
         * `_lastSavedTo` means "where the thing on the canvas was written",
         * and after a load the thing on the canvas is something else.  It
         * was left standing, so the Save readout went on naming the
         * PREVIOUS structure's file: build ethanol over a restored
         * BDT-Au junction and the panel still said
         * "Target: BDT-Au-junction.xyz" (browser walk, 2026-08-24).
         *
         * The overwrite confirmation inside `_saveDataset` means this
         * misleads rather than silently destroys -- but a confirmation you
         * answer while reading the wrong filename is not much of a guard.
         * Cleared here because this is the one gate every generator, every
         * upload and the sidebar's own load come through. */
        _lastSavedTo = null;
        _rememberPanel();
    }

    function getLoadedFrom() { return _loadedFrom; }

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
            loadedFrom:   _loadedFrom,
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
        markLoadedFrom:             markLoadedFrom,
        getLoadedFrom:              getLoadedFrom,
        restorePanelNote:           restorePanelNote,
        onPanelChange:              onPanelChange,
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
