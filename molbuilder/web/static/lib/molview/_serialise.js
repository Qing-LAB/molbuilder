/* MolView data model — SERIALISATION codec (model -> bytes).  MolView-internal submodule.
 *
 * MODULE: molview (lib/molview/).  Extracted from data-model.js (the god-hub split) as an
 *   INJECTED FACTORY, like _operations.js / _state-timeline-impl.js.  No global shim: data-model.js
 *   is the sole consumer and IMPORTS `createSerialiser` directly (the 4a real import graph).
 *
 * ROLE: the model -> bytes direction, in BOTH shapes the app persists:
 *   - `serialise()`   -> the SESSION snapshot (structure + source + selection + view + state_index)
 *                        the state timeline files as the crash/reload mirror (§19.5).
 *   - `scratchBlob()` -> the project-FILE codec {xyz, sidecar} `exportFile()` emits for a durable
 *                        save (the inverse of openMolecule; §19.4).  Enforces the geometry<->store
 *                        atom-count invariant (review c) -- refuses to serialise a mismatched pair.
 *   It is pure model->bytes: it READS the model through injected accessors and holds NO state.  The
 *   inverse direction (bytes -> model: `_applySnapshot` / install) stays in the hub with the write
 *   path.  This is the serialise HALF of the codec.
 *
 * USED BY: lib/molview/data-model.js ONLY.  It builds ONE instance via createSerialiser({...})
 *   and delegates its (hoisted) `_serialise` / `_scratchBlob` to it -- so the timeline's injected
 *   `serialise` seam and `exportFile()` keep working unchanged.
 */
"use strict";

const root = (typeof window !== "undefined") ? window : globalThis;

// deps: read-only model accessors + a few thunks for state defined later in the hub.
//   { getStructure, getSource, getSelection, getElements, getRegions, getFrozen, isDirty,
//     getCanvas: () -> canvasStore, viewGetState: () -> viewState,
//     getStateIndex: () -> int, getWorkspaceId: () -> string|null }
export function createSerialiser(deps) {
    deps = deps || {};
    const getStructure   = deps.getStructure;
    const getSource      = deps.getSource;
    const getSelection   = deps.getSelection;
    const getElements    = deps.getElements;
    const getRegions     = deps.getRegions;
    const getFrozen      = deps.getFrozen;
    const isDirty        = deps.isDirty;
    const getCanvas      = deps.getCanvas;
    const viewGetState   = deps.viewGetState;
    const getStateIndex  = deps.getStateIndex;
    const getWorkspaceId = deps.getWorkspaceId;

    function _lastSavedTo() {
        var cs = getCanvas();
        return (cs && typeof cs.getLastSavedTo === "function")
            ? cs.getLastSavedTo() : null;
    }

    // The session snapshot (§19.5): the full model state the timeline mirrors + files.  The
    // snapshot always carries the full structure (including structure.atoms); the clean-with-file
    // "refetch from disk on restore" gate is applied AT RESTORE TIME by the consumer (it reads
    // state.dirty + state.source.file), not here -- nulling atoms here broke downstream reads that
    // derive elements/atom_names from atoms[].
    function serialise() {
        return {
            v:            1,
            saved_at:     new Date().toISOString(),
            // Keep the sourceless-workspace draft id stable across a same-tab reload (so we
            // recover the same server draft, not a fresh orphan).
            workspace_id: getWorkspaceId(),
            state: {
                structure:    getStructure(),
                source:       getSource(),
                dirty:        isDirty(),
                last_save_to: _lastSavedTo(),
                selection:    getSelection(),
                view:         viewGetState(),
                // The timeline position, so a reload restores WHERE in the undo history we are
                // (not just the geometry) -- §19.5.
                state_index:  getStateIndex(),
            },
        };
    }

    // The geometry's own atom count from the text, or null if unparseable.  Handles both source
    // formats (review c-PDB: a PDB has no leading count line, so the xyz-only parse skipped it).
    function _geometryAtomCount(text, format) {
        if (format === "pdb") {
            var n = 0, lines = String(text).split("\n");
            for (var i = 0; i < lines.length; i++) {
                if (/^(ATOM|HETATM)/.test(lines[i])) n++;
            }
            return n;
        }
        var first = String(text).split("\n", 1)[0];   // xyz: first line = atom count
        var c = parseInt(first, 10);
        return (isFinite(c) && String(c) === first.trim()) ? c : null;
    }

    // The scratch blob {xyz, sidecar} -- the codec shape exportFile() emits for a project-file
    // save (projects.writeFile) and /api/structure/resolve-cell consumes (StructureCodec.
    // from_scratch).  ONE serialisation for BOTH the durable save AND the transient draft, built
    // through the §1.2.1 accessors (getRegions/getFrozen).  Periodicity persists RAW (null when
    // truly unset) so the file is truthful.
    function scratchBlob() {
        var s = getStructure();
        if (!s || !s.text) return null;
        // INVARIANT (review c): the geometry (text, from the canvas store) and the labels/frozen
        // (indices, from the selection store) live in two stores.  They MUST index the same atom
        // set.  If they desync, refuse to serialise -- a mismatched .xyz/.json pair (regions
        // pointing outside the geometry) must NEVER reach disk.  Surfaces the bug instead of
        // silently corrupting.
        var nGeom = _geometryAtomCount(s.text, s.source_format);
        var nStore = getElements().length;
        if (nGeom !== null && nStore !== nGeom) {
            if (root.console && root.console.error) {
                root.console.error(
                    "workspace: atom-count desync -- geometry has " + nGeom
                    + " atoms but the store holds " + nStore
                    + "; refusing to serialise a mismatched .xyz/.json pair.");
            }
            return null;
        }
        var per = s.periodicity || {};
        return {
            xyz: s.text,
            sidecar: {
                n_atoms_total:   getElements().length || s.n_atoms || 0,
                structure_hash:  "",
                regions:         getRegions(),
                frozen_atoms:    getFrozen(),
                cell:            per.cell || null,
                // §3c: the low corner an explicit cell wraps off-origin atoms from, saved so an
                // electrode junction's box survives a save/reload.
                cell_origin:     per.cell_origin || null,
                axis_kind:       per.axis_kind || null,
                vacuum:          per.vacuum || null,
                // (no kgrid: it's a sampling knob on SiestaConfig, not geometry)
                // F1: re-emit the annotation channels carried opaquely from load, so a Modify Save
                // preserves them instead of clobbering to {}.
                annotations:     s.annotations || {},
                // F2: selection_rules is NOT a working-copy field -- dropped (was empty-{} noise).
            },
        };
    }

    return { serialise: serialise, scratchBlob: scratchBlob };
}
