/* MolView data model — the in-memory DATA MODEL for the structure (molview-module.md).
 *
 * This is MolView's OWN data, mounted on ``molbuilder.molview.data``.  It owns the structure /
 * selection / periodicity / frames (via the store singletons in lib/molview/), every read/write
 * accessor, the server operations, AND the serialisation (the .xyz/.molstruct.json format is
 * MolView's).  It never persists to disk itself: on a data change it serialises and hands the
 * BYTES to the workspace PERSISTENCE layer (lib/workspace/dispatcher.js) via ``ws.persist(...)``.
 * The workspace never reads this data.
 *
 * (Formerly the workspace ``dispatcher.js``; the data half was carved out here 2026-07 so the
 * workspace could shrink to session + file-access persistence, per workspace-contract.md.)
 *
 * Public surface (indicative -- the `api` object at the END of this file is authoritative):
 *
 *   window.molbuilder.molview.data.{
 *     // Reads + subscription
 *     subscribe, getState, getStructure, getSource, getSelection,
 *     getRegions, getFrozen, getElements, getCoordinates, isDirty, isEmpty,
 *     // Periodicity (structure-periodicity.md): raw reads, {value,isDefault}
 *     // display accessors, setters, + the server-resolved commit
 *     getUnitCell, getVacuum, getAxisKind,
 *     getUnitCellInfo, getUnitCellOrigin, getVacuumInfo, getAxisKindInfo,
 *     // Operations (server round-trip + atomic state replacement)
 *     installMolecule, generate, applyOp, discard, undo,
 *     // Session-state timeline (§19.5): one save + one load, index-delta parameterized
 *     save, load, state_index, uncommitted,
 *     // Frames (molview-module.md §14.5) — coords owned by the embed movie (single owner)
 *     reloadFrames, setForces, addFrame, addFrames, setFrame, getFrame, currentFrame, frameCount,
 *     // Serialisation (the format is MolView's): project-file export + draft key + persist bracket
 *     exportFile, draftIdentity, suspendPersist, resumePersist,
 *     // Sub-namespaces
 *     selection.{toggle,set,add,remove,all,invert,clear,
 *                setMode,setFilters,setCombinator,
 *                applyFilter,writeLabel},
 *     view.{applyState,getState}}
 *
 * Architecture (as of 2026-06-08, after migration phases 1-9):
 *
 *   * Reads synthesise from canvas-state + selection store +
 *     3Dmol embed via lazy resolvers.  Defensive copies.
 *   * ``applyOp`` owns the modifier-op fetch + cross-store update pipeline, and is FULLY
 *     SELF-CONTAINED: it builds the op-request STRUCTURE body from THIS module's OWN
 *     accessors (``_structureBody`` -> getStructure/getFrozen/getRegions), POSTs
 *     ``/api/modify/<op>``, and routes the response through ``_applyWorkspacePayload``.
 *     Consumers pass ONLY op-params as ``args`` -- the module no longer reaches out to a
 *     ``modify.currentStateBody`` hook (that inverted the dependency; removed with the
 *     Modify state.* rip-out).
 *   * ``_applyWorkspacePayload`` is THE single cross-store sync point:
 *     canvas-state.replaceContent + top-level-metadata distribution onto atoms + selection
 *     store adoptAtoms + the §19.3.2 selection rule (clear on count change), then a
 *     subscription notify.  It no longer calls any
 *     consumer hook -- consumers read the single source through the unified API and react to
 *     the subscription.  Every entry point (applyOp, loadFromText) routes through it.
 *   * Persistence: **PUSH-ONLY** (NOT on every edit) -- on an anchor / a data change the model
 *     serialises its state and calls ``ws.persist(session, snapshot, identity)`` (workspace-
 *     contract.md §4.4.2/§4.7): the session mirror to ``sessionStorage["molbuilder.workspace.v1"]``
 *     and the indexed draft to the on-disk state timeline. There is **no** auto-write / debounce,
 *     and the old parallel legacy mirrors (``molbuilder.structure_canvas`` / ``modify-state``)
 *     were **removed** in the carve -- this model is the single writer.
 *
 * Underlying stores (module-internal; consumers never touch them directly):
 *
 *   - canvas-state (``window.molbuilder.structureCanvas``)
 *     - owns structure text + source provenance + dirty flag.
 *   - selection store (``window.molbuilder.molview.selection.store``)
 *     - owns the atoms list (element + coords + per-atom metadata) + selection + filters + mode.
 *
 * The Modify tab is now a pure CONSUMER: it holds NO structure data (no per-state mirror,
 * no embed handle) -- it reads the single source through this module's unified API and drives
 * ops via ``applyOp``.  (See modify/viewer.js.)
 *
 * Loaded ONLY on /molbuilder (modify.html includes the script).
 * On task tabs (structure-optimization / spectrum-calculation /
 * transport-calculation / results) the dispatcher script is NOT
 * loaded — ``window.molbuilder.workspace`` is undefined there.
 * No cross-tab consumer relies on it; the cross-tab handoff is
 * the Projects-sidebar pointer in ``sessionStorage[
 * "molbuilder.current_file"]`` (task #294, save-first Send-to-
 * Optimization).  Each receiving tab re-reads bytes from disk on
 * mount — the previous ``sessionStorage["builder-structure"]``
 * payload was retired in task #306 (2026-06-09).
 *
 * Tests: tests/test_workspace_dispatcher_js.py.
 */
"use strict";

// 4a real-import graph: the data model's dependencies are IMPORTED (not resolved via
// window.molbuilder.* at call time), so the edges are legible from these lines.  All three are
// single instances per module load -- canvasState is a module singleton; the store factory +
// timeline factory are each called ONCE by this module (it is their sole creator), so importing
// them keeps the one-instance-per-page invariant the whole UI shares through `data.selection`.
import { createStore, cloneAtom, surfaceSnapshot } from "./_selection-store-impl.js";
import { canvasState } from "./_canvas-state-impl.js";
import { createStateTimeline } from "./_state-timeline-impl.js";
import { createOperations } from "./_operations.js";
import { createSerialiser } from "./_serialise.js";
import { createInstall } from "./_install.js";

const root = (typeof window !== "undefined") ? window : globalThis;

    // ─── Internal references resolved on demand ─────────────────── //

    // Phase 9 (2026-06-13): the canvas-state singleton is no
    // longer mounted as a public global.  The impl file
    // (lib/molview/_canvas-state-impl.js) places it on the
    // private ``molview._canvasState`` slot in the browser.  It lives on the MolView
    // namespace (NOT ``workspace``): the canvas store is part of MolView's concealed
    // data model, and the 2026-07 carve made ``workspace`` persistence-only, so a data
    // store hanging off ``workspace.*`` was the model leaking back into the persistence
    // layer.  Test escape hatch: a harness that ``require()``s the impl and assigns the
    // return value to ``root.molbuilder.structureCanvas`` is honoured too.
    function _canvas() {
        // The canvas is a module singleton, IMPORTED (4a).  Kept as a resolver (not an inline
        // ref) so the split-out sub-modules can be handed it uniformly.
        return canvasState;
    }

    // Phase 9 (2026-06-13): the selection store is no longer a
    // public global — the dispatcher creates the one process-wide
    // instance from the factory at lib/workspace/
    // _selection-store-impl.js and holds it for the lifetime of
    // the page.  The factory must be mounted before this IIFE
    // executes (templates load the impl file just before the
    // dispatcher; test_workspace_dispatcher_js.py mirrors that
    // order in its require chain).
    //
    // Test escape hatch: if a harness pre-mounted an instance on
    // ``root.molbuilder.molview.selection.store`` BEFORE the dispatcher
    // loaded, honour it.  This lets test setup share the same
    // store instance the dispatcher reads (so stubs of e.g.
    // ``adoptAtoms`` affect what the dispatcher sees) without
    // re-exposing the legacy global in production templates.
    let _selectionStore = null;
    function _store() {
        // The ONE store instance this data model owns: created ONCE via the imported factory
        // (4a) and cached.  This module is the store's sole creator; the whole UI shares this
        // instance through `data.selection` (mount passes it to the panel/engine/adapter), so a
        // single create here IS the single source of truth.
        if (_selectionStore) return _selectionStore;
        // TEST-INJECTION SEAM: a node harness (tests/support/seed_selection_store.js) may pre-mount
        // a store at `molview.selection.store` BEFORE this model loads so the test and the model
        // share ONE instance.  PRODUCTION never mounts that global (Phase 9 concealed the raw store
        // behind `data.selection`), so in the browser this is undefined and the model creates its
        // own single instance via the imported factory.  Either way there is exactly one store.
        const seeded = root.molbuilder && root.molbuilder.molview
                    && root.molbuilder.molview.selection && root.molbuilder.molview.selection.store;
        _selectionStore = seeded || createStore();
        return _selectionStore;
    }
    // The coordinate time axis (§14.5) lives in the embed's native 3Dmol movie -- the
    // SINGLE coord owner (task #33).  The frame API below (reloadFrames / addFrame /
    // setFrame / getFrame / …) is a thin coordinator: it validates the same-atoms
    // invariant and forwards to the embed handle; it holds NO coordinate copy of its own.
    // The 3Dmol embed handle is the view-state render target (§20).  The ACTIVE molview
    // registers it here at mount (onReady -> attachViewHandle); §18 "molview owns the whole
    // assembly", one active molview per page.  Pre-migration this was read off the tab-owned
    // global ``molbuilder.modify.handle``; that coupling is retired -- the module holds its own.
    var _viewHandle = null;
    // View state restored from a session snapshot BEFORE the embed exists (the restore runs at
    // DOMContentLoaded; the embed's onReady fires later) is stashed here and applied the moment
    // the handle attaches -- so camera / style / axes / labels survive a tab round-trip (§20 +
    // owner-namespaced persistence §18.4).
    var _pendingView = null;
    function _handle() { return _viewHandle; }
    function attachViewHandle(h) {
        _viewHandle = h || null;
        if (_viewHandle && _pendingView) {
            try { view.applyState(_pendingView); } catch (_) { /* stub embed */ }
            _pendingView = null;
        }
    }
    function detachViewHandle(h) {
        if (!h || h === _viewHandle) _viewHandle = null;
    }

    // ── The render engine (molview-render-streamline.md) ─────────────────────────────── //
    // The data model OWNS the data; the engine is the single render place. mount.js registers
    // the engine here at onReady (attachEngine). The data model builds StructureData (§7.1) from
    // its atoms + periodicity and hands it to the engine; the engine reads view flags from the
    // store and draws. The data model never touches 3Dmol -- every draw goes engine -> embedIo.
    var _engine = null;
    var _engineFrameUnsub = function () {};
    // Register the engine + relay its frame channel into the data model's onFrameChange, so the
    // frame bar / measurement (which subscribe to data.onFrameChange) track engine-driven swaps
    // and playback regardless of when they subscribed (before or after the engine attaches).
    function attachEngine(engine) {
        try { _engineFrameUnsub(); } catch (_) {}
        _engine = engine || null;
        _engineFrameUnsub = (_engine && typeof _engine.onFrameChange === "function")
            ? _engine.onFrameChange(_notifyFrame) : function () {};
        // Backfill on attach: a session restore (or any load) runs at DOMContentLoaded, BEFORE
        // the embed's onReady attaches the engine -- so the _pushToEngine() at load time found
        // no engine and no-op'd, leaving the viewer blank for a restored structure. Push the
        // already-loaded structure now that the engine is here (no-op if nothing is loaded).
        if (_engine) _pushToEngine();
    }
    function detachEngine(engine) {
        if (engine && engine !== _engine) return;
        try { _engineFrameUnsub(); } catch (_) {}
        _engineFrameUnsub = function () {};
        _engine = null;
    }

    // Build the engine's cell (explicit lattice + the origin that wraps the atoms), or null.
    function _cellForEngine() {
        var info = (typeof getUnitCellInfo === "function") ? getUnitCellInfo() : null;
        var lattice = info && info.value;
        if (!lattice) return null;
        var origin = (typeof getUnitCellOrigin === "function") ? getUnitCellOrigin() : null;
        return { lattice: lattice, origin: origin || [0, 0, 0] };
    }
    // Build StructureData (§7.1) for the given coordinate frames, taking atom IDENTITY
    // (elements + annotations) and the cell from the CURRENT store atoms / periodicity.
    function _structureDataFor(frames, forces) {
        var s = _store() && _store().getState();
        var atoms = (s && Array.isArray(s.atoms)) ? s.atoms : [];
        return {
            frames:         frames,
            elements:       atoms.map(function (a) { return a.element; }),
            annotations:    atoms.map(function (a) {
                return { label:  (a.labels && a.labels[0]) || undefined,
                         frozen: !!a.isFrozen };
            }),
            cell:           _cellForEngine(),
            forcesPerFrame: Array.isArray(forces) ? forces : null,
        };
    }
    // Push the CURRENT single structure (one frame, from the store atoms) to the engine.
    // Called after a load/edit adopts atoms -- the engine re-renders from this clean data.
    function _pushToEngine() {
        if (!_engine) return;
        var s = _store() && _store().getState();
        var atoms = (s && Array.isArray(s.atoms)) ? s.atoms : [];
        if (!atoms.length) return;
        var frame = atoms.map(function (a) { return [a.x, a.y, a.z]; });
        _engine.setData(_structureDataFor([frame], null));
        _lastCellSig = JSON.stringify(_cellForEngine());   // geometry baseline
    }
    // The periodicity tier's change detector (§6.2 v3): the canvas onChange
    // (the ONE channel) diffs the engine-facing cell geometry and hands a
    // change to engine.setCell — a box move without an atom reload.  The
    // baseline is (re)set by every full push above.
    var _lastCellSig = null;
    function _syncEngineCell() {
        if (!_engine || typeof _engine.setCell !== "function") return;
        var geom = _cellForEngine();
        var sig = JSON.stringify(geom);
        if (sig === _lastCellSig) return;
        _lastCellSig = sig;
        _engine.setCell(geom);
    }
    function _runtime() {
        return (root.molbuilder && root.molbuilder.runtime)
            ? root.molbuilder.runtime : null;
    }
    function _molbuilderTab() {
        return (root.molbuilder && root.molbuilder.molbuilderTab)
            ? root.molbuilder.molbuilderTab : null;
    }

    function _missing(name) {
        return new Error(
            "molview.data: " + name + " not available on this "
            + "page.  Phase 4 wires the dispatcher on /molbuilder; "
            + "other tabs do not host the canvas.");
    }

    // ─── Reads: synthesise the unified state from legacy stores ─── //

    /** Return the structure slice (or null when nothing's loaded). */
    // Serialise the CURRENT atoms (element + live x/y/z) to plain XYZ.  Used only when
    // a frame is scrubbed (§14.5.4): the canvas text is frame 0, but the SHOWN frame's coords
    // come from the embed's native movie (read on demand via getFrame/_scrubbedFrameCoords),
    // so ``getStructure().text`` / a save must reflect the VISIBLE frame, not frame 0.
    function _atomsToXyz(title) {
        var els = getElements();
        var co  = getCoordinates();
        var lines = [String(els.length), title || ""];
        for (var i = 0; i < els.length; i++) {
            var p = co[i] || [0, 0, 0];
            lines.push(els[i] + " " + Number(p[0]).toFixed(6) + " "
                     + Number(p[1]).toFixed(6) + " " + Number(p[2]).toFixed(6));
        }
        return lines.join("\n") + "\n";
    }
    // The VISIBLE frame's coords when a trajectory is scrubbed (currentFrame > 0),
    // read from the embed movie (the single owner) ON DEMAND -- so getStructure /
    // getCoordinates / export reflect the shown frame without a per-frame store push
    // (§14.5.4).  Returns null for frame 0 / a static structure (fast path unchanged).
    function _scrubbedFrameCoords() {
        var cf = currentFrame();
        if (cf <= 0) return null;
        var fc = getFrame(cf);   // -> embed getFrameCoords(cf)
        return (Array.isArray(fc) && fc.length === _atomsInternal().length) ? fc : null;
    }
    function getStructure() {
        var cs = _canvas();
        var st = _store();
        if (!cs || cs.isEmpty()) return null;
        var canvas = cs.getStructure();   // {source_format, text, periodicity}
        if (!canvas) return null;
        var s = st ? st.getState() : null;
        var per = canvas.periodicity || null;
        // Frame coherence (§14.5.4): a scrubbed frame's coords live in the embed movie
        // (the single owner), not canvas.text (frame 0).  Serialise the visible frame so
        // text, atoms, and a save all agree; frame 0 / no-frames keeps the canvas text
        // (fast path, unchanged).  Only trajectories scrub, and those are xyz.
        var _fc = _scrubbedFrameCoords();
        var _text = _fc ? _atomsToXyz((s && s.title) || "") : canvas.text;
        var _fmt  = _fc ? "xyz" : canvas.source_format;
        // Defensive per-atom copy (workspace-contract.md §1.2.1): the store snapshot shares
        // the atom OBJECTS, so without cloning them a consumer mutating
        // getStructure().atoms[i].x would leak straight into the store.  Reuse the selection
        // module's shared _cloneAtom (same copy the ws.selection.getState() surface uses).
        var _clone = cloneAtom;
        return {
            text:          _text,
            source_format: _fmt,
            title:         (s && s.title) || "",
            n_atoms:       s ? s.atoms.length : 0,
            // Overlay the visible frame's coords onto the (frame-independent) atom identity.
            atoms:         s ? s.atoms.map(function (a, i) {
                               var c = _clone(a);
                               if (_fc) { c.x = _fc[i][0]; c.y = _fc[i][1]; c.z = _fc[i][2]; }
                               return c;
                           }) : [],
            // Periodicity rides with the geometry (structure-periodicity.md):
            // `lattice` = the cell (kept for existing consumers); `periodicity`
            // carries the full cell/axis_kind/vacuum so a save writes the
            // whole structure (workspace-contract.md §4.0).
            lattice:       per ? per.cell : null,
            periodicity:   per,
            annotations:   canvas.annotations || null,   // F1: opaque carry
        };
    }

    // --------------------------------------------------------------- //
    //  §1.2.1 accessor API -- the ONLY way consumers read the model.  //
    //  The internal layout (per-atom atoms today, columnar later) is  //
    //  CONCEALED behind these; each materialises the view the caller  //
    //  needs.  No consumer hand-crafts extraction (no state.xyz.split,//
    //  no atoms[i].labels scan) -- they call these.                   //
    // --------------------------------------------------------------- //
    function _atomsInternal() {
        var st = _store();
        var s = st ? st.getState() : null;
        return (s && s.atoms) || [];
    }
    function _periodicityInternal() {
        var cs = _canvas();
        if (!cs || cs.isEmpty()) return null;
        var canvas = cs.getStructure ? cs.getStructure() : null;
        return (canvas && canvas.periodicity) || null;
    }
    function getElements() {
        return _atomsInternal().map(function (a) { return a.element; });
    }
    function getCoordinates() {
        // Reflect the VISIBLE frame when scrubbed (§14.5.4) -- coords from the movie.
        var fc = _scrubbedFrameCoords();
        if (fc) return fc.map(function (p) { return [p[0], p[1], p[2]]; });
        return _atomsInternal().map(function (a) { return [a.x, a.y, a.z]; });
    }
    // RAW explicit cell -- null when unset.  Used by the render's base wireframe (draw a
    // box only for a REAL cell).  For DISPLAY use getUnitCellInfo().value,
    // which falls back to the server-resolved bbox.  The two INTENTIONALLY differ.
    function getUnitCell() {
        var per = _periodicityInternal();
        return (per && per.cell) || null;
    }
    // Defaults for SAFE-to-default fields only (§1.2.1): vacuum -> 0.
    // (k-grid is not here: it's a reciprocal-space SAMPLING knob on
    // SiestaConfig / TransportConfig, not geometry -- structure-periodicity.md.)
    function getVacuum() {
        var per = _periodicityInternal();
        return (per && per.vacuum) || [0, 0, 0];
    }
    // axis_kind is NOT safe to default: periodic vs isolated vs transport is a
    // scientific choice (guessing periodic would generate a WRONG PBC/transport
    // boundary for a transport cell).  Return the explicit value, or null when unset
    // -- the consumer must resolve an unspecified periodicity, never get a guess.
    // Review finding a3.
    function getAxisKind() {
        var per = _periodicityInternal();
        return (per && per.axis_kind) || null;
    }
    // --- §3b display accessors: { value, isDefault } for the Cell page --------- //
    // ONE contract across all four: `isDefault` = the parameter is in its DEFAULT state
    // (no meaningful user override); `value` is what to DISPLAY either way.  The default
    // STATE differs per parameter because each default differs:
    //   cell      -> no explicit cell   (value = server-resolved bbox; may be null pre-
    //                                     first-resolve, so cell `value` alone can be null)
    //   vacuum    -> [0,0,0]            (no padding)
    //   axis_kind -> every axis isolated (the vacuum box)
    // Value-based defaults (vacuum/axis_kind) can't tell "user set the default
    // value" from "never set" -- intentional: the tag reflects the DISPLAYED value being
    // the default, not its provenance.
    function getUnitCellInfo() {
        var per = _periodicityInternal() || {};
        // ONE resolver: the SERVER (struct.resolve_cell §3a) computes the effective
        // cell and sends it as periodicity.resolved_cell; this accessor only SURFACES
        // it -- no re-implemented bbox math on the client.  Explicit cell wins, else
        // the resolved bbox+vacuum default.  (A periodicity edit re-resolves through
        // the server, so resolved_cell stays consistent with cell/vacuum/axis_kind.)
        var explicit = per.cell || null;
        return { value: explicit || per.resolved_cell || null, isDefault: !explicit };
    }
    // §3a/§3c: the world-space CORNER the unit-cell box is drawn from -- ALWAYS the
    // server's resolved_cell_origin (= struct.resolve_cell_origin()).  For a
    // bbox+vacuum default that is (atom_min - vacuum); for an EXPLICIT cell that
    // WRAPS off-origin atoms (an electrode junction) it is the stored cell_origin, so
    // the box wraps the structure instead of jumping to (0,0,0); for an imported
    // crystal (atoms already in [0,cell)) the server returns null -> the world origin.
    // The render feeds this to the cell wireframe.
    function getUnitCellOrigin() {
        var per = _periodicityInternal() || {};
        return per.resolved_cell_origin || null;
    }
    // §3b/§3c display accessor for the Cell page's Origin editor -- the { value,
    // raw, isDefault } shape the other cell editors use.  `value` = the corner the
    // box is DRAWN from (resolved_cell_origin, server-resolved); `raw` = the stored
    // explicit override (cell_origin, null when unset); `isDefault` = no explicit
    // override (the origin is the auto-resolved bbox/world corner).  The editor
    // shows `value`, tags default off `isDefault`, and commits an explicit `raw`.
    function getUnitCellOriginInfo() {
        var per = _periodicityInternal() || {};
        var raw = per.cell_origin || null;
        return {
            value:     per.resolved_cell_origin || null,
            raw:       raw,
            isDefault: raw == null,
        };
    }
    function getVacuumInfo() {
        var v = getVacuum();
        return { value: v, isDefault: (v[0] === 0 && v[1] === 0 && v[2] === 0) };
    }
    function getAxisKindInfo() {
        var per = _periodicityInternal() || {};
        var ak = per.axis_kind || null;
        var val = ak || ["isolated", "isolated", "isolated"];
        // Default = the isolated vacuum box (§3a): unset, or every axis isolated (the
        // loader sets isolated on a fresh molecule; that's still the default config).
        var isDefault = !ak || val.every(function (k) { return k === "isolated"; });
        return { value: val, isDefault: isDefault };
    }
    // Direct label -> atom-indices lookup.  (Scans the per-atom layout today; becomes
    // a plain ``regions[label]`` read once D4 makes the internals columnar -- the
    // caller never sees the difference, which is the point of the accessor.)
    // An atom "carries" a label if it is in the atom's region labels OR it is the
    // built-in "frozen" flag -- regions and frozen are queried UNIFORMLY ("frozen" is
    // the frozen tag channel, atom-annotations.md).  The ONE label-membership rule.
    function _atomHasLabel(atom, label) {
        if (!atom) return false;
        if (label === "frozen" && atom.isFrozen) return true;
        return ((atom.labels || []).indexOf(label) !== -1);
    }
    // Indices of atoms carrying `label` (a region name OR the built-in "frozen").
    function getAtomsByLabel(label) {
        var atoms = _atomsInternal();
        var out = [];
        for (var i = 0; i < atoms.length; i++) {
            if (_atomHasLabel(atoms[i], label)) out.push(i);
        }
        return out;
    }
    // The unified "which atoms carry label X, and WHERE are they" accessor: rows
    // { index, element, x, y, z } (the getAtoms row shape; coords reflect the VISIBLE
    // frame via getCoordinates) for every atom carrying `label` (a region name OR
    // "frozen").  Consumers no longer combine getAtomsByLabel + getCoordinates by hand.
    function getLabelAtoms(label) {
        var atoms = _atomsInternal();
        var coords = getCoordinates();
        var out = [];
        for (var i = 0; i < atoms.length; i++) {
            if (!_atomHasLabel(atoms[i], label)) continue;
            var c = coords[i] || [0, 0, 0];
            out.push({ index: i, element: atoms[i] && atoms[i].element,
                       x: c[0], y: c[1], z: c[2] });
        }
        return out;
    }
    // All label NAMES present on the structure -- every region key, plus "frozen" when
    // any atom is frozen.  The set to enumerate for a label picker / iterate over.
    function getLabels() {
        var names = Object.keys(getRegions() || {});
        if (getFrozen().length) names.push("frozen");
        return names;
    }
    function getFrozen() {
        var atoms = _atomsInternal();
        var out = [];
        for (var i = 0; i < atoms.length; i++) {
            if (atoms[i] && atoms[i].isFrozen) out.push(i);
        }
        return out;
    }
    // The FULL label -> indices map (all regions) -- the single place per-atom labels
    // are gathered for save/draft serialisation (kills the duplicate scans that used
    // to live in both save.js and _scratchBlob).  A plain ``regions`` read once D4
    // makes the internals columnar.
    function getRegions() {
        var atoms = _atomsInternal();
        var regions = {};
        for (var i = 0; i < atoms.length; i++) {
            var labels = (atoms[i] && atoms[i].labels) || [];
            for (var j = 0; j < labels.length; j++) {
                if (!regions[labels[j]]) regions[labels[j]] = [];
                regions[labels[j]].push(i);
            }
        }
        return regions;
    }

    // --- WRITE accessors (§1.2.1) -- the ONLY way consumers mutate the model. --- //
    // _setPeriodicity is PRIVATE: the door's adoption writes through it.  The
    // legacy public writers (setUnitCell / setCellOrigin / setAxisKind /
    // setVacuum / commitPeriodicity) were REMOVED 2026-07-29 -- they bypassed
    // the frame-contract gate (no validation, no heal, no notices; the
    // hemeC-corruption class) and contaminated the API: there is ONE
    // periodicity write path, commitPeriodicityOp below.
    function _setPeriodicity(patch) {
        var cs = _canvas();
        if (cs && typeof cs.setPeriodicity === "function") cs.setPeriodicity(patch);
    }
    // § 6.2 v3 — the unified periodicity door.  PYTHON OWNS THE CHANGE:
    // one POST per Cell-page edit; the gate applies the regime model
    // (vacuum/axis edits reset to derived; cell respects origin-then-
    // vacuum; origin edits warn) and returns the corrected TRUTH blob +
    // resolved views + notices.  This client adopts — it never computes.
    // Returns Promise<{ok, notices, error?}>.
    //
    // NOTICES (the contract's machine-readable half, § 6.1): each entry is
    // {level: "info"|"warn", message}, and one reporting state the gate
    // MODIFIED also carries kind: "heal" — _install.js keys on that to mark
    // the session dirty, since the disk pair still holds the old value until
    // the user saves.  Surface notices; never parse message text.
    //
    // The blob is adopted VERBATIM: a resolved corner is a view, so
    // sidecar.cell_origin stays null whenever the corner is derived, and
    // resolved_cell_origin (not the truth) is what the box is drawn from.
    function commitPeriodicityOp(op, payload) {
        var blob = _scratchBlob();
        if (!blob || !root.fetch) {
            return Promise.resolve(
                { ok: false, error: "no structure loaded", notices: [] });
        }
        return root.fetch("/api/structure/periodicity", {
            method:  "POST",
            headers: { "Content-Type": "application/json" },
            body:    JSON.stringify({ data: blob, op: op, payload: payload }),
        }).then(function (r) { return r.json(); }).then(function (j) {
            if (!j || !j.ok) {
                return { ok: false, notices: [],
                         error: (j && j.error) || "periodicity edit failed" };
            }
            var sc = (j.blob && j.blob.sidecar) || {};
            _setPeriodicity({
                cell:                 sc.cell || null,
                cell_origin:          sc.cell_origin || null,
                vacuum:               sc.vacuum || [0, 0, 0],
                axis_kind:            sc.axis_kind || null,
                resolved_cell:        j.resolved_cell || null,
                resolved_cell_origin: j.resolved_cell_origin || null,
            });
            try { markDirty(); } catch (_) { /* empty canvas */ }
            return { ok: true, notices: j.notices || [] };
        }).catch(function (e) {
            return { ok: false, notices: [],
                     error: String((e && e.message) || e) };
        });
    }

    // Assign/replace a region label on a set of atom indices (in-memory; persists on
    // Save).  Routes to the selection store's label writer.
    function setLabel(label, indices) {
        var st = _store();
        if (!st || typeof st.writeLabel !== "function") return;
        var p = st.writeLabel(label, indices);
        // MUST mark dirty (like ws.selection.writeLabel) -- else the restore gate
        // (dirty || !source.file) refetches disk atoms on reload and the label is
        // silently lost.  Review finding a1.
        try { markDirty(); } catch (_) { /* empty/absent canvas -> no-op */ }
        return p;
    }
    // NOTE: generic key-value metadata was REMOVED (review finding a2): the accessor
    // wrote to an in-memory map that no persisted form carried -> a silent data-loss
    // sink.  Persisting it needs a real sidecar schema field (`metadata` on Structure
    // + molstruct + the codec); that is a designed follow-up, not a bug-fix bolt-on.
    // NOTE: adding/deleting ATOMS is a geometry mutation -- it goes through the §3
    // write mutator ``ws.applyOp(op, args)`` (the server modify pipeline), NOT a
    // direct accessor, so the geometry stays consistent with bonds/validation.

    function getSource() {
        var cs = _canvas();
        if (!cs) return {kind: "blank", file: null, generator_input: null};
        var src = cs.getSource ? cs.getSource() : null;
        if (!src) return {kind: "blank", file: null, generator_input: null};
        return {
            kind:            src.kind || "blank",
            file:            src.file || null,
            generator_input: src.generator_input || null,
        };
    }

    function getSelection() {
        var st = _store();
        if (!st) return {indices: [], pickOrder: [], mode: "click", filters: [],
                         combinator: "or", isolate: false};
        var s = st.getState();
        return {
            indices:    s.selection.slice(),
            // §19.2: the full selection snapshot -- NOT just indices.  pickOrder is
            // the click ORDER (angle vertex = pickOrder[1]); isolate is the VIEW
            // toggle.  Dropping them here silently reset "Show selected only" and the
            // measurement vertex on every reload / Retract (§19.5 says the restore
            // brings back what you had).
            pickOrder:  (s.pickOrder || s.selection).slice(),
            mode:       s.mode,
            filters:    s.filters.map(function (f) { return Object.assign({}, f); }),
            combinator: s.combinator,
            isolate:    !!s.isolate,
            // View-toggle flags (task #64, render-streamline §7.2): the ENGINE renders
            // from THESE store flags, so they are the view state that must survive a tab
            // round-trip (axes / unit-cell / index-labels / force overlay + scale).
            showAxis:   !!s.showAxis,
            showCell:   !!s.showCell,
            showIndex:  !!s.showIndex,
            showForces: !!s.showForces,
            forceScale: s.forceScale,
        };
    }

    function isDirty() {
        var cs = _canvas();
        return cs ? !!cs.isDirty() : false;
    }
    function isEmpty() {
        var cs = _canvas();
        return cs ? !!cs.isEmpty() : true;
    }

    /**
     * Atoms accessor (workspace-contract.md §2).  Returns a slice of
     * the underlying atom array so callers can iterate without
     * leaking mutations back into state.  Always reflects the
     * current state — never stale relative to ``getStructure().atoms``
     * because both read from the same selection store on each call.
     *
     * Returns ``[]`` (not ``null``) when the workspace is empty so
     * callers can ``.length`` / ``.forEach`` without a null guard.
     */
    function getAtoms() {
        var st = _store();
        if (!st) return [];
        var s = st.getState();
        if (!Array.isArray(s.atoms)) return [];
        // Defensive per-atom copy (workspace-contract §1.2.1): the store snapshot SHARES
        // the atom OBJECTS, so a bare .slice() lets a consumer mutate the store by writing
        // getAtoms()[i].x / .labels.push(...).  Use the same _cloneAtom getStructure() does.
        var _clone = cloneAtom;
        // Reflect the VISIBLE frame when scrubbed (§14.5.4): overlay the current frame's
        // coords (from the movie) onto the frame-independent atom identity.
        var fc = _scrubbedFrameCoords();
        return s.atoms.map(function (a, i) {
            var c = _clone(a);
            if (fc) { c.x = fc[i][0]; c.y = fc[i][1]; c.z = fc[i][2]; }
            return c;
        });
    }

    /**
     * Source-file accessor (workspace-contract.md §2).  Convenience
     * over ``getSource().file`` — used by selection-panel +
     * viewer-adapter + bootstrap which previously read
     * ``selection.store.getState().sourceFile`` directly.
     *
     * Returns ``null`` when the workspace has no file source
     * (kind="blank" or kind="smiles"/"name"/etc.).
     */
    function getSourceFile() {
        return getSource().file;
    }

    /**
     * Last-saved-to accessor (workspace-contract.md §1.2 +
     * structure-save migration target).  Returns the disk path the
     * workspace was last successfully saved to this session, or
     * ``null`` if it hasn't been saved yet.
     *
     * Used by ``structure-save.targetPath()`` and the modify-tab's
     * Send-to-Optimization gate, which previously read
     * ``structureCanvas.getLastSavedTo()`` directly.
     */
    function getLastSavedTo() {
        var cs = _canvas();
        return (cs && typeof cs.getLastSavedTo === "function")
            ? (cs.getLastSavedTo() || null) : null;
    }

    /**
     * Composite snapshot — every read at one entry point.  Defensive
     * copies via the inner getters; mutating the returned object does
     * not leak into the underlying stores.
     */
    function getState() {
        return {
            structure: getStructure(),
            source:    getSource(),
            dirty:     isDirty(),
            selection: getSelection(),
            view:      view.getState(),
            loading:   _isLoading(),
        };
    }

    function _isLoading() {
        var st = _store();
        return st ? !!st.getState().loading : false;
    }

    // ─── Subscriptions: fan in canvas + store + view notifications ─ //

    var _subscribers = [];
    var _wired = false;

    // ─── State timeline (molview-module.md §19.5) ────────────────── //
    // Push-only undo: the state-timeline submodule (`_timeline`, below) owns the `state_index`
    // (position in the tab's operation sequence; 0 = the opened anchor) and the `uncommitted`
    // flag (true iff the in-memory model changed since the last `save` -- what a `load(-1)` would
    // discard).  There is NO automatic write on change: only `installMolecule` (the ONE anchor
    // write), `save`, and `load` touch disk.  save/load are SERIALIZED through the submodule's
    // push/pop chain so the index advances/retreats only after each workspace round-trip resolves.
    // The state timeline (Save-state / Retract) is a MolView-internal SUBMODULE
    // (lib/molview/_state-timeline-impl.js): it owns `state_index`, the `uncommitted` flag, and
    // the ordered save/load chain.  This model injects the serialise / applySnapshot / workspace
    // / trace seams and still owns WHEN the data changed — it calls `_timeline.markUncommitted()`
    // on every data change.  The factory is IMPORTED (4a); this module is its sole caller.  Browser: the factory is a global mounted by the sibling <script>
    // loaded before this file.  Node tests: fall back to require() so bootstraps that require()
    // data-model.js need no extra wiring.  (`_serialise` / `_applySnapshot` / `_ws` / `_trace`
    // are hoisted function declarations, so passing them here — before their source lines — is
    // safe.)
    var _timeline = createStateTimeline({
        serialise:     _serialise,
        applySnapshot: _applySnapshot,
        getWorkspace:  _ws,
        // Route the timeline's save/load/anchor WRITES through the suspend gate (reads/prune still
        // use getWorkspace directly).  This is what makes suspendPersist() gate the timeline.
        persist:       _persist,
        trace:         _trace,
    });
    // Guard set true while an internal apply (installMolecule / load snapshot application) drives
    // the stores, so the resulting canvas/frame signals do NOT mark the model uncommitted.
    var _applying = false;

    function _notify() {
        // Defensive copy: a subscriber that synchronously calls
        // subscribe()/unsubscribe()/notify() must not corrupt the
        // iteration.  Pattern matches the selection store's _notify.
        var snapshot = _subscribers.slice();
        for (var i = 0; i < snapshot.length; i++) {
            try { snapshot[i](getState()); }
            catch (e) {
                // Subscriber errors are non-fatal — they don't wedge
                // the dispatcher.  Log and continue.
                if (root.console && root.console.warn) {
                    root.console.warn(
                        "molview.data: subscriber threw", e);
                }
            }
        }
        // NOTE: _notify() is RENDER-ONLY -- it fires on EVERY change (data OR view) and
        // NEVER persists.  Persistence is push-only (§19.5): nothing reaches disk until the
        // consumer calls save(delta) (or a load anchors the timeline).  So a hundred slider
        // drags -- and a hundred edits -- cost zero writes.
    }

    function _ensureSubscribed() {
        if (_wired) return;
        _wired = true;
        // Canvas-state fires on every DATA change (text / periodicity / dirty toggle) ->
        // render, and (when NOT inside an internal apply) mark the model uncommitted so a
        // consumer knows a save(1) would checkpoint real work.  NO auto-write (§19.5).
        var cs = _canvas();
        if (cs && typeof cs.onChange === "function") {
            cs.onChange(function () {
                _syncEngineCell();     // periodicity tier: box geometry follows the store
                _notify();
                if (!_applying) _timeline.markUncommitted();
            });
        }
        // Selection store fires on atoms / selection / coords / isolate.  Those
        // are VIEW ops -> RE-RENDER ONLY (§18.3); they do not mark the model uncommitted.
        var st = _store();
        if (st && typeof st.subscribe === "function") {
            st.subscribe(_notify);
        }
        // 3Dmol embed view-state changes aren't (today) push-based;
        // consumers wanting view updates poll via ``ws.view.getState()``
        // after a setState call.  Future enhancement: forward
        // viewer.onCameraChange or similar.
    }

    function subscribe(fn) {
        if (typeof fn !== "function") {
            throw new TypeError("molview.data.subscribe(fn): function required");
        }
        _ensureSubscribed();
        _subscribers.push(fn);
        // Match the selection store's contract: fire once on subscribe
        // so the subscriber sees the current state without waiting
        // for the next mutation.
        try { fn(getState()); } catch (_) { /* swallow */ }
        return function unsubscribe() {
            var ix = _subscribers.indexOf(fn);
            if (ix >= 0) _subscribers.splice(ix, 1);
        };
    }

    // ─── Selection sub-namespace: passthrough to selection.store ── //
    //
    // workspace-contract.md §5 — the public selection surface.  The
    // dispatcher OWNS the shape contract: getState() and the value
    // passed to subscribe(fn) callbacks both return the SAME
    // ``_selectionSnapshot``-shaped object, regardless of what the
    // underlying legacy store happens to call its fields.  Without
    // this single rewrite, the panel's render path (gets shape via
    // subscribe → reads ``.selection``) and click handlers (call
    // getState() directly → read ``.indices``) would see different
    // field names; one would always be broken.

    /**
     * Normalise a legacy-store state object to the contract shape.
     * Used by BOTH ``ws.selection.getState()`` and the wrapper
     * around ``ws.selection.subscribe(fn)`` so subscribers + direct
     * readers see identical objects.
     *
     * Delegates to the ONE surface-snapshot shaper in
     * _selection-store-impl.js (loaded first) — was a hand-maintained
     * character-identical twin of that function; now a single source so a
     * new store field can't be mirrored into one copy but not the other.
     * Returns a fresh defensive copy.
     */
    function _selectionSnapshot(st) {
        return surfaceSnapshot(st);
    }

    var selection = {
        toggle:          function (i) {
            var s = _store(); if (!s) throw _missing("selection store");
            return s.toggleAtom(i);
        },
        set:             function (indices) {
            var s = _store(); if (!s) throw _missing("selection store");
            return s.setSelection(indices);
        },
        add:             function (indices) {
            var s = _store(); if (!s) throw _missing("selection store");
            return s.addToSelection(indices);
        },
        remove:          function (indices) {
            var s = _store(); if (!s) throw _missing("selection store");
            return s.removeFromSelection(indices);
        },
        all:             function () {
            var s = _store(); if (!s) throw _missing("selection store");
            return s.selectAll();
        },
        invert:          function () {
            var s = _store(); if (!s) throw _missing("selection store");
            return s.invertSelection();
        },
        clear:           function () {
            var s = _store(); if (!s) throw _missing("selection store");
            return s.clearSelection();
        },
        setMode:         function (mode) {
            var s = _store(); if (!s) throw _missing("selection store");
            return s.setMode(mode);
        },
        setIsolate:      function (on) {
            var s = _store(); if (!s) throw _missing("selection store");
            return s.setIsolate(on);
        },
        // View-toggle flags (task #64): the render engine reads these; the View menu / rail
        // toggles WRITE them here. molview-render-streamline.md §7.2.
        setViewFlag:     function (name, value) {
            var s = _store(); if (!s) throw _missing("selection store");
            return s.setViewFlag(name, value);
        },
        setFilters:      function (filters) {
            var s = _store(); if (!s) throw _missing("selection store");
            return s.setFilters(filters);
        },
        // workspace-contract.md §5 — per-filter mutators used by
        // selection-panel's filter-row UI.  These wrap the
        // selection store's same-named methods; consumers should
        // call them through ws.selection so the legacy store stays
        // private.
        addFilter:       function (filter) {
            var s = _store(); if (!s) throw _missing("selection store");
            return s.addFilter(filter);
        },
        removeFilter:    function (index) {
            var s = _store(); if (!s) throw _missing("selection store");
            return s.removeFilter(index);
        },
        updateFilter:    function (index, patch) {
            var s = _store(); if (!s) throw _missing("selection store");
            return s.updateFilter(index, patch);
        },
        setCombinator:   function (c) {
            var s = _store(); if (!s) throw _missing("selection store");
            return s.setCombinator(c);
        },
        applyFilter:     function () {
            var s = _store(); if (!s) throw _missing("selection store");
            return s.applyFilter();
        },
        writeLabel:      function (target, indices) {
            var s = _store(); if (!s) throw _missing("selection store");
            var p = s.writeLabel(target, indices);
            // §4.0: assigning a label is an in-memory edit -> the workspace is
            // now dirty (unsaved); it reaches disk only on explicit Save.
            try { markDirty(); } catch (_) { /* empty/absent canvas -> no-op */ }
            return p;
        },
        // workspace-contract.md §5.3 — alias for ws.getAtoms() so
        // call sites that read ``selection.store.getState().atoms``
        // have a 1:1 migration target.
        getAtoms:        function () { return getAtoms(); },
        // Per workspace-contract.md §5 — defensive snapshot of the
        // selection slice in the CONTRACT shape (``indices``, NOT
        // legacy ``selection``).  Identical shape to what
        // ``subscribe(fn)`` passes its callback — see
        // ``_selectionSnapshot`` above for the rationale.
        getState:        function () {
            var s = _store();
            return _selectionSnapshot(s ? s.getState() : null);
        },
        // Per §5 — selection-only subscribe.  Wraps the legacy
        // store's subscribe so the callback receives the contract-
        // shaped object (``indices`` not ``selection``, plus every
        // field consumers read).  Without this wrapper, the panel's
        // render path (subscribe-delivered state) and click handlers
        // (getState() direct) would see different field names.
        subscribe:       function (fn) {
            var s = _store(); if (!s) throw _missing("selection store");
            // Forward the store's `changes` (the dirty-bit array) as the 2nd arg so consumers
            // (the panel) can do the minimal update -- a selection click diffs highlight instead
            // of rebuilding all rows. Dropping it here made every consumer see `undefined` and
            // over-render (the ~1.2s freeze).
            return s.subscribe(function (legacyState, changes) {
                fn(_selectionSnapshot(legacyState), changes);
            });
        },
        // Per workspace-contract.md §5 — atomically install path +
        // atoms + selection in one promise.  Used by the modify-tab
        // selection-bootstrap on file commit so the store's atoms
        // come from the just-loaded payload (no second /api/selection/
        // atoms refetch).  Awaitable; resolves once the store has
        // settled.
        adoptSession:    function (opts) {
            var s = _store(); if (!s) throw _missing("selection store");
            if (typeof s.adoptSession !== "function") {
                throw _missing("selection.store.adoptSession");
            }
            return s.adoptSession(opts);
        },
    };

    // ─── Canvas write helpers (workspace-contract.md §3) ─────────── //
    //
    // These wrap the legacy canvas-state methods so consumers can
    // stop reaching into ``window.molbuilder.structureCanvas`` for
    // wholesale-replace writes.  Each delegates to canvas-state in
    // the current implementation; once Phase 10 finishes the
    // delegation becomes the dispatcher's internal write.

    /**
     * Mark the workspace dirty.  Modifier panels call this after a
     * successful op so a subsequent Load / Generate triggers the
     * warning-modal gate.  Idempotent: re-marking dirty is a no-op
     * if already dirty.
     */
    function markDirty() {
        var cs = _canvas();
        if (!cs) throw _missing("canvas store");
        if (typeof cs.markDirty !== "function") {
            throw _missing("canvas.markDirty");
        }
        // markDirty() is the explicit "this is an unsaved DATA edit" signal --
        // label / frozen / metadata writes (which route through the selection
        // store, not the canvas text/periodicity) call it.  Set the TIMELINE
        // dirty flag directly here rather than relying on the canvas onChange:
        // cs.markDirty() is a NO-OP when the canvas dirty bit is ALREADY set, so
        // its onChange never fires and the edit would go untracked -- Retract
        // would then skip it and Save could miss it.  Guarded by _applying so a
        // snapshot restore that re-sets the dirty bit doesn't spuriously mark
        // the model uncommitted (load() sets the committed flag afterward).
        if (!_applying) _timeline.markUncommitted();
        return cs.markDirty();
    }

    /**
     * Record a successful save: clears dirty + sets
     * last_save_to=path.  ``save.js`` calls this after a successful
     * project write.
     */
    function markSaved(path) {
        var cs = _canvas();
        if (!cs) throw _missing("canvas store");
        if (typeof cs.markSaved !== "function") {
            throw _missing("canvas.markSaved");
        }
        var r = cs.markSaved(path);
        // Re-anchor the selection store's sourceFile through the SAME door: a
        // save-as makes ``path`` the new home, so later label edits + the "Loaded"
        // readout target it.  Consumers call markSaved and NOTHING else -- save.js
        // used to reach into ``selection.adoptSession({sourceFile})`` here, a
        // low-level store poke.  noteSavedTo is sync + sourceFile-only (a save does
        // not change the model), so it cannot disturb atoms/selection.
        var st = _store();
        if (st && typeof st.noteSavedTo === "function") st.noteSavedTo(path);
        return r;
    }

    // ─── View sub-namespace: passthrough to the 3Dmol embed ──────── //

    var view = {
        applyState: function (patch) {
            var h = _handle(); if (!h) throw _missing("view embed");
            if (typeof h.applyState !== "function") {
                throw _missing("view embed.applyState");
            }
            return h.applyState(patch);
        },
        getState: function () {
            var h = _handle(); if (!h) return null;
            // Synthesise from the embed's get* methods (per embed
            // protocol § 3.13).  Defensive: each accessor may be
            // absent on a stub embed; just return what's available.
            return {
                camera:  (typeof h.getCamera === "function") ? h.getCamera() : null,
                style:   (typeof h.getStyle  === "function") ? h.getStyle()  : null,
                axes:    (typeof h.getAxes   === "function") ? h.getAxes()   : null,
                labels:  (typeof h.getLabels === "function") ? h.getLabels() : null,
            };
        },
    };

    // §20 view flush.  Persistence is PUSH-ONLY (§19.5) and a pure view change (camera rotate,
    // menu toggle) has NO push trigger -- so without this, view state would only reach the mirror
    // incidentally, on the next structure op / Save state.  This patches ONLY the view slot of the
    // already-persisted session mirror with the current embed view state, then re-mirrors (MIRROR
    // ONLY, no disk).  It is VIEW-ONLY by design: it never touches the mirror's structure/selection,
    // so it cannot change structure-persistence semantics (data-safety).  Called from a
    // pagehide / visibilitychange('hidden') flush the module registers at mount (§18.2), so a
    // view-only change survives a tab round-trip.
    function flushViewState() {
        var ws = _ws();
        if (!ws || typeof ws.persist !== "function"
                || typeof ws.readPersistedSnapshot !== "function") return;
        var m = ws.readPersistedSnapshot();
        if (!m || !m.state) return;               // nothing persisted yet -> nothing to patch
        var v = null;
        try { v = view.getState(); } catch (_) { v = null; }
        if (!v) return;                            // no embed / no view -> leave the mirror as-is
        m.state.view = v;
        // task #64: the store's view-toggle flags (showAxis/showCell/showIndex/showForces)
        // are the view state the ENGINE renders from -- refresh them in the persisted
        // selection slice from the LIVE store so a view-only toggle (setViewFlag, which has
        // no push-persist trigger) survives, exactly like the embed camera/style in
        // m.state.view above.  Only patches the flags; the rest of the selection slice
        // (indices/pickOrder/mode/...) stays as last persisted.
        if (m.state.selection) {
            try {
                var liveSel = getSelection();
                m.state.selection.showAxis   = liveSel.showAxis;
                m.state.selection.showCell   = liveSel.showCell;
                m.state.selection.showIndex  = liveSel.showIndex;
                m.state.selection.showForces = liveSel.showForces;
                m.state.selection.forceScale = liveSel.forceScale;
            } catch (_) { /* best-effort */ }
        }
        var wid = ws.workspaceId();
        var idx = (typeof m.state.state_index === "number") ? m.state.state_index : _timeline.stateIndex;
        // MIRROR ONLY, routed through the suspend gate (held if a multi-step op is in flight).
        try { _persist(m, null, { workspace_id: wid, state_index: idx }); }
        catch (_) { /* best-effort on unload */ }
    }

    // ─── Operations ──────────────────────────────────────────────── //
    //
    // ``applyOp`` owns its pipeline end-to-end.  The other op methods
    // (loadFromFile, generate, save, undo) delegate to the existing
    // legacy modules — those are intentional thin wrappers that exist
    // so consumers can stay on the unified ``ws.*`` API while the
    // underlying implementations migrate at their own pace.

    // ─── The model install / serialise primitives (structure-load-save-contract.md) ─── //
    //
    // The FORMAT-AWARE project-file DOORS -- `projects.parser.openMolecule(path)` /
    // `saveMolecule(path)` -- live in the projects package (lib/projects/parser.js),
    // NOT here.  molview.data owns only the MODEL primitives those doors call:
    //   * installMolecule({text[, sidecar, source, periodicity, annotations]})
    //       -- parse text (+ optional sidecar) via /api/build/load and install it into
    //          the model in ONE write (also the generators' install path);
    //   * exportFile()  -- serialise the settled model -> {xyz, sidecar};
    //   * markSaved(path) -- clear dirty + re-anchor the store sourceFile.
    // molview.data touches NO file endpoint; the projects package moves the bytes.


    // ─── Install / codec write-back (bytes -> model) — extracted to _install.js ─── //
    // The atomic cross-store sync point (applyWorkspacePayload), the install primitives
    // (installMolecule/loadText), and generate live in their own injected-factory submodule.
    // The hub wires it with the stores + the persist bracket + notify/engine-push; the `_applying`
    // flag STAYS here (set via the setApplying seam) since notify/markDirty/frame writers read it.
    var _install = createInstall({
        getStore:       _store,
        getCanvas:      _canvas,
        suspendPersist: suspendPersist,
        resumePersist:  resumePersist,
        notify:         _notify,
        pushToEngine:   _pushToEngine,
        trace:          _trace,
        missing:        _missing,
        setApplying:    function (b) { _applying = b; },
        anchor:         function () { return _timeline.anchor(); },
    });
    // Hoisted delegators so the api entries + the operations applyWorkspacePayload injection +
    // the timeline _applySnapshot keep their existing references unchanged.
    function _applyWorkspacePayload(payload, opts) { return _install.applyWorkspacePayload(payload, opts); }
    function installMolecule(input) { return _install.installMolecule(input); }
    function generate(kind, input, opts) { return _install.generate(kind, input, opts); }

    /**
     * Apply a modifier op (delete / add_atom / orient / translate /
     * rotate / electrode / symmetric_electrodes).  Owns the full
     * pipeline: build the request body from the current workspace
     * state, POST to ``/api/modify/<op>``, and route the response
     * through ``_applyWorkspacePayload`` for atomic store update.
     * Returns the server's workspace payload (or rejects on
     * non-ok HTTP / non-ok envelope).
     *
     * Phase 5 (2026-06-07): self-sufficient — no longer delegates
     * to the modify-tab's ``postOp``.  The reverse is now true:
     * modify-tab's postOp is a thin wrapper around this method.
     */

    // Ordered event tracer (diagnostic; no-op unless window.__MV_TRACE).  Emits
    // "[MV-TRACE <ms>] <event> <json?>" so ONE reproduction shows the exact
    // SEQUENCE + timing across openMolecule/anchor/persist -- observe, don't guess.
    function _trace(ev, extra) {
        if (!root.__MV_TRACE) return;
        try {
            var t = (root.performance && root.performance.now)
                ? root.performance.now() : 0;
            root.console.log("[MV-TRACE " + t.toFixed(1) + "] " + ev
                + (extra !== undefined ? " " + JSON.stringify(extra) : ""));
        } catch (_) { /* tracing never throws */ }
    }


    // Build the op-request STRUCTURE body from molview.data's OWN accessors.  The module
    // owns the data, so it constructs the request itself -- consumers pass ONLY op-PARAMS
    // (element/plane/size/gap/anchor/…) as ``args``.  This replaces the old inverted
    // dependency where applyOp reached OUT to the consumer's modify.currentStateBody (which
    // read a parallel state.* mirror that could go stale, e.g. the electrode None>None bug).
    // Metadata arrays are reconstructed per-atom from getStructure().atoms -- the SAME shape
    // _scratchBlob emits -- so the request always reflects the live single source.
    function _structureBody() {
        var s = getStructure();
        var atoms = (s && Array.isArray(s.atoms)) ? s.atoms : [];
        // A metadata column is sent ONLY when EVERY atom carries it; otherwise []. NEVER a
        // list with nulls -- the server does max(residue_ids) etc. and a null poisons the
        // comparison (TypeError '>' between NoneType, the electrode/add-atom None>None bug).
        // [] means "absent" and the server applies its own default (residue_id 1, atom_name
        // = element, …), matching the pre-rip-out contract (which sent the payload array or []).
        function _col(pick) {
            var out = [];
            for (var i = 0; i < atoms.length; i++) {
                var v = pick(atoms[i]);
                if (v == null) return [];      // incomplete -> let the server default
                out.push(v);
            }
            return out;
        }
        var body = {
            xyz:           (s && s.text) || "",
            title:         (s && s.title) || "",
            atom_names:    _col(function (a) { return a.atomName; }),
            residue_ids:   _col(function (a) { return a.residueId; }),
            residue_names: _col(function (a) { return a.residueName; }),
            chain_ids:     _col(function (a) { return a.chainId; }),
            frozen_atoms:  (typeof getFrozen === "function") ? getFrozen() : [],
            regions:       (typeof getRegions === "function") ? getRegions() : {},
        };
        // COMPLETE body (§19.3.2): the v4 per-atom annotation channels ride opaquely so the
        // op reindexes them WITH the geometry.  Sent only when present.  (The server must read
        // + reindex + return them for them to survive an add/delete -- the modify round-trip
        // completeness step; until then this is a harmless no-op.)
        if (s && s.annotations != null) body.annotations = s.annotations;
        // Periodicity (cell / axis_kind / vacuum) rides through the op so an edit
        // does not silently wipe a cell / transport axis the user set (the server
        // reads it back in struct_from_body, symmetric with structure_to_dict's emission).
        if (s && s.periodicity != null) body.periodicity = s.periodicity;
        return body;
    }

    // ─── Modifier operations (server round-trips) — extracted to _operations.js ─── //
    // The applyOp pipeline (op registry + fetch + count-invariant + atomic apply, §19.3.2) lives
    // in its own injected-factory submodule.  The hub wires it with the accessors + the two codec
    // seams that STAY here (shared with install + the timeline): _structureBody builds the request
    // body, _applyWorkspacePayload is THE atomic response apply.  applyOp stays on the api surface.
    var _operations = createOperations({
        getStructure:          getStructure,
        getCoordinates:        getCoordinates,
        getElements:           getElements,
        structureBody:         _structureBody,
        applyWorkspacePayload: _applyWorkspacePayload,
        getStore:              _store,
    });
    function applyOp(op, args) { return _operations.applyOp(op, args); }

    /**
     * Save the workspace structure to disk.  Delegates to the
     * Sources-card Save panel's ``structureSave.save()`` — that
     * module owns the path-resolution + writeFile pipeline.  The
     * ``opts`` parameter is reserved for future Save as… support
     * (target path + overwrite policy); structureSave.save()
     * today takes no arguments and writes to its resolved
     * targetPath().  This wrapper exists so consumers can stay
     * on the unified ``ws.*`` API; the dispatcher does NOT
     * re-implement the save pipeline.
     */
    // ---- THE exportFile door (molview-module.md §19.4) ------------------------- //
    // `exportFile()` = serialize the WHOLE model to PROJECT-FILE bytes ({xyz, sidecar})
    // -- the inverse of `openMolecule`.  This is the DOCUMENT serialization: the bytes a
    // consumer writes to the user's project file.  Writing bytes to disk is the
    // consumer's job (the sidecar), NOT the data model's (two-saves-never-mix).  It is
    // NOT the session-state timeline save -- that is `save(delta)` (§19.5).
    function exportFile() { return _scratchBlob(); }

    /**
     * Wipe the workspace canvas + selection.  UNCONDITIONAL — the
     * caller is expected to gate on dirty-state + warning modal
     * BEFORE calling this method (the Sources-card Discard button
     * is the canonical caller and goes through
     * ``warningModal.confirmDiscardUnsaved`` first).  Pure local
     * action; no HTTP.
     */
    function discard() {
        var cs = _canvas();
        if (!cs) return Promise.resolve();
        if (typeof cs.clear !== "function") {
            return Promise.reject(_missing("canvas.clear"));
        }
        cs.clear();
        var st = _store();
        if (st && typeof st.clearSelection === "function") {
            st.clearSelection();
        }
        return Promise.resolve();
    }

    /**
     * Undo the last modifier op.  Delegates to the modify-tab's
     * ``applyUndo`` (Phase 4 exposed it on the runtime registry).
     */
    // Undo IS `load(-1)` (§19.5): the model's public `undo` delegates to the state-timeline
    // submodule (`_timeline.undo`, mounted on the api below).  The old in-memory undo stack
    // (and its `modify.applyUndo` delegation) was retired with the state timeline.

    // ─── Frames — the coordinate time axis (workspace-contract.md §1.5) ─────── //
    // The caller (a Results trajectory inspector, a live-job stream) decides the op; the
    // workspace validates the same-atoms invariant + applies it.  setFrame pushes the current
    // frame's coords onto the selection store's atoms (a coords-only swap) so the render (which
    // reads store atoms) shows that frame.  Persistence (extxyz) is a later step; these are the
    // in-memory ops.  Synchronous — they throw on a violation (not a rejected promise).

    function _atomCount() {
        const st = _store();
        const s = st && st.getState();
        return (s && Array.isArray(s.atoms)) ? s.atoms.length : 0;
    }
    // Guard the STRUCTURE-identity (count) on top of the frame-series' internal-consistency
    // check: a frame's atom count must match the loaded structure.  (Element-order identity is
    // enforced upstream at parse time — an in-memory coords frame carries no elements.)
    function _requireMatch(coords, label) {
        const n = _atomCount();
        if (n === 0) throw new Error(label + ": load a structure first (no atoms)");
        if (Array.isArray(coords) && coords.length !== n) {
            throw new Error(label + ": frame atom count " + coords.length
                + " does not match the loaded structure's " + n + " atoms (§1.5)");
        }
    }
    // Same-atoms invariant across a frame SET (§14.5): every frame same atom count.
    // Enforced HERE (was the removed frame-series' job) before the coords reach the
    // movie -- a mismatch is a hard error, never coerced.
    function _requireSameAtoms(frames, label) {
        if (!Array.isArray(frames) || frames.length < 1) {
            throw new Error(label + ": needs at least one frame");
        }
        const n0 = Array.isArray(frames[0]) ? frames[0].length : 0;
        for (let i = 0; i < frames.length; i++) {
            const fr = frames[i];
            if (!Array.isArray(fr) || fr.length !== n0) {
                throw new Error(label + ": frame " + i + " has "
                    + (Array.isArray(fr) ? fr.length : "?") + " atoms, expected "
                    + n0 + " (same-atoms invariant, §14.5)");
            }
        }
        return n0;
    }
    // Frame-change channel -- DELIBERATELY SEPARATE from the selection store's
    // notify.  A native frame swap must update ONLY the frame-controls bar
    // (slider/counter); firing the whole store would re-render the selection panel
    // every frame and steal focus from a filter input the user is typing in during
    // playback.  So the bar subscribes HERE (mount.js), not to the store.
    var _frameListeners = [];
    function onFrameChange(fn) {
        if (typeof fn !== "function") return function () {};
        _frameListeners.push(fn);
        return function () {
            var i = _frameListeners.indexOf(fn);
            if (i >= 0) _frameListeners.splice(i, 1);
        };
    }
    function _notifyFrame() {
        for (var i = 0; i < _frameListeners.length; i++) {
            try { _frameListeners[i](); } catch (_) {}
        }
    }
    function reloadFrames(frames, opts) {
        opts = opts || {};
        // Same-atoms invariant + structure-identity check (§6) -- reject before anything
        // reaches the engine; never coerce.
        _requireSameAtoms(frames, "reloadFrames");
        _requireMatch(frames[0], "reloadFrames");
        // Hand the WHOLE trajectory to the render engine as StructureData (§6.1): it re-renders
        // (structural regen) with all frames + the per-frame forces (from which it builds the
        // force arrows). The engine owns the movie; the data model no longer drives 3Dmol.
        // (Legacy opts.arrowsPerFrame -- pre-built arrows -- is superseded by opts.forces; a
        // consumer still on arrowsPerFrame gets no force overlay until it moves to forces.)
        if (_engine) _engine.setData(_structureDataFor(frames, opts.forces));
        if (!_applying) _timeline.markUncommitted();   // frame DATA changed -> uncommitted (§19.5)
        return frameCount();
    }
    // Force overlays are built by the ENGINE from the raw forces (process.js §2.4) -- the
    // consumer hands per-frame forces, never pre-built arrows. setForces swaps the force DATA
    // and re-bakes the arrow overlay IN PLACE (no movie reload), so a force-filter change
    // (threshold / hide-frozen) is cheap. forcesPerFrame is in ORIGINAL atom order; a zero
    // vector suppresses that atom's arrow; null clears the overlay.
    function setForces(forcesPerFrame) {
        if (_engine) _engine.setForces(forcesPerFrame);
    }
    function addFrame(coords) { return addFrames([coords]); }
    // Append one OR MANY new frames (a live poll can bring several at once). Validates the
    // same-atoms invariant, then hands the batch to the engine (§6.2 append -- extend, no reload).
    function addFrames(list, opts) {
        opts = opts || {};
        const frames = Array.isArray(list) ? list : [];
        if (!frames.length) return frameCount();
        frames.forEach(function (coords) { _requireMatch(coords, "addFrames"); });
        if (_engine) _engine.appendFrames(frames, { forces: opts.forces });
        if (!_applying) _timeline.markUncommitted();   // frame DATA changed -> uncommitted (§19.5)
        return frameCount();
    }
    function setFrame(i) {
        const n = frameCount();
        const idx = Math.floor(Number(i));
        if (!(idx >= 0 && idx < n)) {
            throw new Error("setFrame(" + i + ") out of range [0.." + (n - 1) + "]");
        }
        // Native frame swap (§3): the engine swaps to the pre-parsed movie frame + re-applies
        // the shown frame's overlays. It fires its OWN frame channel (engine.onFrameChange),
        // which the frame bar + measurement subscribe to -- not the selection store.
        if (_engine) _engine.showFrame(idx);
        return currentFrame();
    }
    // Frame reads -- delegate to the render engine (the single coord + frame owner, §7.1/§9).
    // getFrame returns the CLEAN original-indexed coords (all atoms), the accessor measurement
    // reads by the panel's original atom index.
    function getFrame(i) {
        return _engine ? _engine.getFrame(i) : null;
    }
    function currentFrame() {
        return _engine ? _engine.currentFrame() : 0;
    }
    function frameCount() {
        return _engine ? _engine.frameCount() : 0;
    }

    // ─── Persistence (Phase 8) ─────────────────────────────────── //

    /**
     * Persistence (§19.5): the model is PUSH-ONLY -- the state timeline (save/load/anchor) is the
     * ONLY writer, handing serialised BYTES to the workspace dispatcher (lib/workspace/dispatcher.js),
     * which owns the single sessionStorage mirror.  There are NO legacy per-store mirrors and no
     * auto-write (the old ``structure_canvas`` / ``modify-state`` parallel writers were removed).
     * The session-snapshot shape (v/saved_at/workspace_id/state{structure,source,dirty,last_save_to,
     * selection,view,state_index}) is defined by the serialiser codec below; the restore side reads
     * ``state.dirty`` + ``state.source.file`` to decide saved-snapshot vs disk-refetch.
     */
    // The serialisation codec (model -> bytes, BOTH shapes) is extracted to _serialise.js as an
    // injected factory; the hub wires it with read-only accessors + thunks for the state defined
    // above (view / _timeline / _ws).  `_serialise` / `_scratchBlob` stay as hoisted delegators so
    // the timeline's injected `serialise` seam + `exportFile()` keep their existing references.
    var _serialiser = createSerialiser({
        getStructure:   getStructure,
        getSource:      getSource,
        getSelection:   getSelection,
        getElements:    getElements,
        getRegions:     getRegions,
        getFrozen:      getFrozen,
        isDirty:        isDirty,
        getCanvas:      _canvas,
        viewGetState:   function () { return view.getState(); },
        getStateIndex:  function () { return _timeline.stateIndex; },
        getWorkspaceId: function () { return _ws() ? _ws().workspaceId() : null; },
    });

    function _serialise() { return _serialiser.serialise(); }

    // Reload-restore is now `load(0)` (§19.5): it reads the sessionStorage MIRROR
    // (the current committed snapshot + state_index) and applies it to the WHOLE model
    // WITHOUT re-anchoring the timeline.  The former standalone `restoreSnapshot(snap)`
    // door was folded into that delta-parameterized `load`.

    // exportFile()'s project-file blob {xyz, sidecar} -- delegated to the serialiser codec.
    function _scratchBlob() { return _serialiser.scratchBlob(); }

    // Keep the transient DRAFT in sync with memory on every edit -- the contract's
    // "update = the only automatic disk write" (workspace-contract.md).  The draft
    // exists for ANY workspace with data, NO project dir required: a loaded file
    // drafts next to itself; a brand-new molecule drafts to the top-level
    // projects-root, keyed by ``workspace_id``.  Best-effort crash-safety.
    // The identity the transient draft is keyed under: the source file, or a stable
    // workspace_id for a not-yet-saved molecule.  A Save MUST use THIS to drop the
    // right draft -- review finding b1: the save used to send the target path, so it
    // dropped a phantom and the real (source-keyed) draft leaked as a permanent
    // orphan.  Mirrors _persistDraft's key exactly.
    // The AUTOMATIC session/tab draft is opaque bytes keyed by the tab's workspace id
    // and NOTHING else -- no filename (two-saves-never-mix).  A filename belongs ONLY to
    // the explicit "save to a project file" action, which is a separate consumer flow
    // through the project sidebar; it must never leak into automatic crash-safety.  (An
    // earlier version keyed by the source file, which made the auto-persist try to
    // resolve a load-time filename as a real project path -> a spurious 400 for anything
    // loaded from bytes rather than an on-disk project file.)
    // (`draftIdentity` itself lives in the state-timeline submodule; the model re-exposes it on
    // the api below as `_timeline.draftIdentity`.)

    // F4: persistence is SUSPENDED across a multi-step load (_commitFile sets the
    // canvas text, then adopts the store atoms a network-round-trip later).  A
    // persist tick in that window would write a draft pairing the NEW geometry with
    // the PREVIOUS file's regions/frozen (the atom-count invariant only catches a
    // count change, not a same-count swap).  Suspend around the load; resume writes
    // the now-consistent state once.
    // THE atomic gate variable: >0 == persistence SUSPENDED.  A nesting COUNTER (not a bool) so
    // nested suspend brackets compose -- each suspendPersist() must be paired with a resumePersist().
    // JS is single-threaded so ++/-- are atomic; `_persist()` below is the single chokepoint EVERY
    // persist write flows through and CONSULTS this counter, which is what makes the framework real.
    var _persistSuspended = 0;
    // A persist requested while suspended is HELD here (shape { session, snapshot, identity }) and
    // flushed ONCE when the counter returns to 0.  Coalescing keeps the LATEST session/identity but
    // NEVER drops a pending DISK snapshot for a later MIRROR-only request (that would lose a
    // checkpoint).  A bracket is meant to wrap ONE coherent op (a stable state_index); do not wrap
    // several DISTINCT checkpoint saves in one bracket.
    var _pendingPersist = null;

    // suspendPersist / resumePersist (molview.data.*): the framework bracket a consumer wraps a
    // multi-step data operation with so no INTERIM (inconsistent) state is persisted -- "suspend
    // around the op; resume writes the now-consistent state once."  Pair them (try/finally); the
    // counter tolerates nesting.
    // Observers of the suspend state (the UI indicator subscribes here).  Fired ONLY on the
    // 0<->suspended EDGE transitions, not on every nested suspend/resume, so the indicator toggles
    // once per bracket.
    var _persistListeners = [];
    function _notifyPersistState() {
        var on = _persistSuspended > 0;
        for (var i = 0; i < _persistListeners.length; i++) {
            try { _persistListeners[i](on); } catch (_) { /* an observer must not break the gate */ }
        }
    }
    function suspendPersist() {
        _persistSuspended++;
        if (_persistSuspended === 1) _notifyPersistState();       // 0 -> suspended (leading edge)
        else if (_persistSuspended > 16 && root.console && root.console.warn) {
            // Runaway nesting is almost certainly a suspendPersist() with no matching resumePersist()
            // (a bracket missing its try/finally).  Left unpaired it wedges ALL future persistence, so
            // surface it rather than fail silently.
            root.console.warn("[molview] persistence suspended " + _persistSuspended
                + " levels deep -- a suspendPersist() without a matching resumePersist()?");
        }
    }
    function resumePersist() {
        if (_persistSuspended === 0) return;   // unbalanced extra resume -> no-op, no spurious edge
        _persistSuspended--;
        if (_persistSuspended !== 0) return;   // still inside an outer bracket
        // Trailing edge FIRST, THEN flush: a throwing/rejecting write must not skip the notify and
        // leave the "Saving paused" indicator stuck visible.
        _notifyPersistState();                                     // suspended -> 0 (trailing edge)
        if (_pendingPersist) {
            var p = _pendingPersist;
            _pendingPersist = null;
            _writePersist(p.session, p.snapshot, p.identity);      // flush the coalesced state ONCE
        }
    }
    // Observability (molview.data.*): the UI reflects the suspend state via these.
    function isPersistSuspended() { return _persistSuspended > 0; }
    function onPersistStateChange(fn) {
        if (typeof fn !== "function") return function () {};
        _persistListeners.push(fn);
        return function () {
            var i = _persistListeners.indexOf(fn);
            if (i >= 0) _persistListeners.splice(i, 1);
        };
    }

    // THE persist chokepoint (§19.5): the state timeline's save/load/anchor AND flushViewState all
    // route through here, so `suspendPersist()` actually gates them.  Not suspended -> write straight
    // through.  Suspended -> HOLD + coalesce into _pendingPersist, flushed once at the outermost
    // resume.  Returns a RESOLVED promise on defer (the write is best-effort crash-safety, not awaited
    // by the caller).  CONTRACT: the bracket is for suppressing INTERIM / automatic persists during a
    // multi-step op -- do NOT wrap an explicit index-advancing `save()` in a suspend bracket (its
    // index advance would decouple from the deferred write); call save() AFTER resume instead.
    function _persist(session, snapshot, identity) {
        if (_persistSuspended <= 0) return _writePersist(session, snapshot, identity);
        if (!_pendingPersist) {
            _pendingPersist = { session: session, snapshot: snapshot, identity: identity };
        } else {
            // Coalesce onto the existing pending: take the LATEST session/identity, but keep a DISK
            // snapshot if EITHER request carried one -- a later MIRROR-only request (snapshot==null)
            // must NOT drop a pending checkpoint.
            _pendingPersist.session  = session;
            _pendingPersist.identity = identity;
            if (snapshot != null) _pendingPersist.snapshot = snapshot;
        }
        return Promise.resolve();
    }
    function _writePersist(session, snapshot, identity) {
        var ws = _ws();
        if (!ws || typeof ws.persist !== "function") return Promise.resolve();
        return Promise.resolve(ws.persist(session, snapshot, identity));
    }

    // The workspace PERSISTENCE layer (lib/workspace/dispatcher.js) -- data-model hands it
    // serialised BYTES; it never reads our data.
    function _ws() { return (root.molbuilder && root.molbuilder.workspace) || null; }

    // ─── The state timeline: save / load / anchor (§19.5) ─────────── //
    //
    // Apply a session snapshot (a `_serialise()` output) to the WHOLE model atomically,
    // WITHOUT resetting the timeline -- the mechanism `load` and session-restore reuse.
    // Distinct from `openMolecule`, which additionally prunes + re-anchors the timeline (§19.5).  The
    // snapshot is the getState-based session state ({structure, source, dirty, selection,
    // view}); restore structure + selection + view + dirty from it.  `_applying` is held true
    // so the driven canvas/frame signals do not mark the model uncommitted.
    function _applySnapshot(snap) {
        if (typeof snap === "string") {
            try { snap = JSON.parse(snap); } catch (_) { return; }
        }
        var s = snap && snap.state;
        if (!s) return;
        var structure = s.structure || null;
        var src = s.source || {};
        _applying = true;
        try {
            if (structure && structure.text) {
                // Reuse the ONE atomic sync point (_applyWorkspacePayload) -- the same apply
                // a fresh load runs -- but WITHOUT load's prune/re-anchor.  installSource
                // (re)installs the whole structure into the canvas (dirty=false); we restore
                // the snapshot's dirty bit below.
                var _atoms = structure.atoms || [];
                _applyWorkspacePayload({
                    text:          structure.text,
                    // Emit the SAME payload shape a server load returns, so ANY consumer of
                    // the sync point gets the same payload fields on a RESTORE as on a
                    // load.  Derived from the snapshot's atoms[] -- no server round-trip.
                    xyz:           structure.text,
                    source_format: structure.source_format,
                    title:         structure.title || "",
                    n_atoms:       (structure.n_atoms != null)
                                       ? structure.n_atoms : _atoms.length,
                    elements:      _atoms.map(function (a) { return a.element; }),
                    atom_names:    _atoms.map(function (a) {
                        return a.atom_name != null ? a.atom_name : (a.element || "");
                    }),
                    residue_ids:   _atoms.map(function (a) {
                        return a.residue_id != null ? a.residue_id : null;
                    }),
                    residue_names: _atoms.map(function (a) {
                        return a.residue_name != null ? a.residue_name : null;
                    }),
                    chain_ids:     _atoms.map(function (a) {
                        return a.chain_id != null ? a.chain_id : null;
                    }),
                    periodicity:   structure.periodicity || null,
                    annotations:   structure.annotations || null,
                    atoms:         _atoms,
                }, {
                    touchCanvas:    false,
                    resetSelection: true,
                    installSource:  { kind:            src.kind || "file",
                                      file:            src.file || null,
                                      generator_input: src.generator_input || null },
                });
            }
            // Restore the FULL selection snapshot the apply above reset (§19.2/§19.5):
            // mode, filters, combinator, and the VIEW toggle (isolate), then
            // the selection itself.  setSelection LAST and passed the pickOrder (falling
            // back to indices) so it re-derives BOTH the sorted set AND the click order
            // (the angle-measurement vertex) in one call.
            var st = _store();
            var sel = s.selection;
            if (st && sel) {
                if (typeof st.setMode === "function" && sel.mode) st.setMode(sel.mode);
                if (typeof st.setFilters === "function" && Array.isArray(sel.filters))
                    st.setFilters(sel.filters);
                if (typeof st.setCombinator === "function" && sel.combinator)
                    st.setCombinator(sel.combinator);
                if (typeof st.setIsolate === "function" && typeof sel.isolate === "boolean")
                    st.setIsolate(sel.isolate);
                // View-toggle flags (task #64): restore what the ENGINE renders from, so
                // axes / cell / index-labels / forces survive a tab round-trip.
                if (typeof st.setViewFlag === "function") {
                    ["showAxis", "showCell", "showIndex", "showForces"].forEach(function (k) {
                        if (typeof sel[k] === "boolean") st.setViewFlag(k, sel[k]);
                    });
                    if (sel.forceScale !== undefined)
                        st.setViewFlag("forceScale", sel.forceScale);
                }
                if (typeof st.setSelection === "function") {
                    var order = (Array.isArray(sel.pickOrder) && sel.pickOrder.length)
                        ? sel.pickOrder
                        : (Array.isArray(sel.indices) ? sel.indices : []);
                    st.setSelection(order);
                }
            }
            // Restore the view (camera / style / axes / labels).  The restore usually runs
            // BEFORE the embed exists (viewer.js's DOMContentLoaded load(0) fires before mount's
            // onReady), so if no handle is attached yet, stash it -- attachViewHandle applies it
            // the instant the embed arrives, so the view survives a tab round-trip (§20).
            if (s.view) {
                if (_handle()) { try { view.applyState(s.view); } catch (_) { /* stub embed */ } }
                else { _pendingView = s.view; }
            }
            // Restore the dirty bit (installSource set it false).
            if (s.dirty) { try { markDirty(); } catch (_) { /* empty/absent canvas */ } }
        } finally {
            _applying = false;
        }
        _notify();
    }

    // The save-state / retract mechanics (`_enqueue`, `save`, `load`, `undo`, anchor) live in the
    // state-timeline submodule (`_state-timeline-impl.js`), built as `_timeline` above.  The model
    // injects `_serialise` / `_applySnapshot` (kept here — they read/write the model) and exposes
    // the public save/load/undo (mapped to `_timeline.*` on the api below).


    // ─── Mount on window.molbuilder.molview.data ────────────────── //

    var api = {
        subscribe:             subscribe,
        getState:              getState,
        getStructure:          getStructure,
        getSource:             getSource,
        getSourceFile:         getSourceFile,
        getLastSavedTo:        getLastSavedTo,
        getSelection:          getSelection,
        getAtoms:              getAtoms,
        // §1.2.1 accessor API -- the concealed model's ONLY read surface.
        getElements:           getElements,
        getCoordinates:        getCoordinates,
        getUnitCell:           getUnitCell,
        getAxisKind:           getAxisKind,
        getVacuum:             getVacuum,
        // §3b display accessors ({ value, isDefault }) for the Cell page:
        getUnitCellInfo:       getUnitCellInfo,
        getUnitCellOrigin:     getUnitCellOrigin,
        getUnitCellOriginInfo: getUnitCellOriginInfo,   // §3c Origin editor display
        getVacuumInfo:         getVacuumInfo,
        getAxisKindInfo:       getAxisKindInfo,
        getAtomsByLabel:       getAtomsByLabel,   // label -> [index] (region or "frozen")
        getLabelAtoms:         getLabelAtoms,     // label -> [{index, element, x, y, z}]
        getLabels:             getLabels,         // all label names (regions + "frozen")
        getFrozen:             getFrozen,
        getRegions:            getRegions,
        draftIdentity:         _timeline.draftIdentity,  // the draft key a Save must drop (b1)
        suspendPersist:        suspendPersist, // F4: bracket a multi-step data op (holds persists)
        resumePersist:         resumePersist,  //     -> flushes the coalesced final state on the outermost resume
        isPersistSuspended:    isPersistSuspended,   // current gate state (the UI indicator reads this)
        onPersistStateChange:  onPersistStateChange, // subscribe to 0<->suspended edges (UI indicator)
        // §1.2.1 WRITE accessors -- the concealed model's ONLY mutation surface.
        commitPeriodicityOp:   commitPeriodicityOp, // §6.2 v3: THE periodicity door (Python owns)
        setLabel:              setLabel,
        isDirty:               isDirty,
        isEmpty:               isEmpty,
        markDirty:             markDirty,
        markSaved:             markSaved,
        // ---- THE model install / serialise primitives (structure-load-save-contract.md §2) ---- //
        // The FORMAT-AWARE project-file DOORS live in the projects package
        // (projects.parser.openMolecule / saveMolecule).  molview.data owns only these
        // MODEL primitives those doors call -- it touches NO file endpoint (the old
        // openProjectFile/saveProjectFile/readWorkingCopy stack that POSTed
        // /api/workingcopy/* is gone).
        installMolecule:       installMolecule, // parse text (+ sidecar) via /api/build/load -> whole model, atomic + timeline reset (generators + parser.openMolecule)
        exportFile:            exportFile,    // serialize whole model -> {xyz,sidecar} project-file bytes (parser.saveMolecule's serialiser)
        // The state timeline (§19.5): ONE save + ONE load, index-delta parameterized.
        // `state_index` / `uncommitted` are LIVE getters (defined on the mounted object below).
        // save(1) commits a checkpoint; load(-1) retracts to the previous index (undo-only);
        // load(0) reloads from the mirror.
        save:                  _timeline.save,
        load:                  _timeline.load,
        generate:              generate,
        applyOp:               applyOp,
        discard:               discard,
        undo:                  _timeline.undo,
        // Frames — the coordinate time axis; coords owned by the embed movie (§14.5).
        reloadFrames:          reloadFrames,
        setForces:             setForces,
        onFrameChange:         onFrameChange,
        addFrame:              addFrame,
        addFrames:             addFrames,
        setFrame:              setFrame,
        getFrame:              getFrame,
        currentFrame:          currentFrame,
        frameCount:            frameCount,
        selection:             selection,
        view:                  view,
        attachViewHandle:      attachViewHandle,   // §20: the active molview registers its embed
        detachViewHandle:      detachViewHandle,   // handle here at mount (retires modify.handle)
        attachEngine:          attachEngine,       // the render engine (molview-render-streamline.md)
        detachEngine:          detachEngine,
        flushViewState:        flushViewState,     // §20: pagehide flush -- mirror the live view
    };

    // The state-timeline reads are LIVE getters (§19.5) -- a plain value would freeze at 0.
    // Object.assign (used to mount below) copies a getter's VALUE, not the accessor, so the
    // live getters must be (re)defined on the FINAL mounted object; do it via this helper so
    // ``api`` (module.exports / runtime.register) and the mount both carry live reads.
    function _defineTimelineGetters(target) {
        Object.defineProperty(target, "state_index", {
            get: function () { return _timeline.stateIndex; },
            enumerable: true, configurable: true,
        });
        Object.defineProperty(target, "uncommitted", {
            get: function () { return _timeline.uncommitted; },
            enumerable: true, configurable: true,
        });
    }
    _defineTimelineGetters(api);

    // UMD-ish mount: ALWAYS mount on ``root.molbuilder.workspace``
    // (browser + Node test contexts both see the global) AND ALSO
    // expose the API as ``module.exports`` when running under
    // CommonJS so tests can ``require()`` the file without driving
    // a browser shim.  Test bootstraps still need to mount the
    // legacy stores (canvas-state, selection.store) on
    // ``window.molbuilder.*`` before reads return non-null data —
    // see tests/test_workspace_dispatcher_js.py for the canonical
    // setup.
    root.molbuilder = root.molbuilder || {};
    // MolView's in-memory DATA MODEL — mounted on ``molbuilder.molview.data``.  Merge (not
    // replace) so a pre-existing molview namespace survives.  The stores this reads
    // (canvas-state, selection store) mount
    // themselves separately and are looked up lazily via _canvas()/_store().
    root.molbuilder.molview = root.molbuilder.molview || {};
    root.molbuilder.molview.data = Object.assign(
        root.molbuilder.molview.data || {}, api);
    // Re-attach the LIVE timeline getters: Object.assign copied their VALUES (0 / false)
    // as data properties, so redefine them as accessors on the mounted object.
    _defineTimelineGetters(root.molbuilder.molview.data);
    if (_runtime() && typeof _runtime().register === "function") {
        _runtime().register("molview.data", api);
    }

    // Eagerly subscribe to the underlying stores so a data change renders (and flips the
    // `uncommitted` flag) even before any UI consumer subscribes.  Push-only: NO auto-write.
    _ensureSubscribed();

    // ── ESM public surface (the MolView ESM migration) ──
    // The transitional global (``root.molbuilder.molview.data`` above) is the §3.2 shim that
    // classic consumers still read; this named export is what index.js re-exports for module
    // consumers.  Both reference the SAME ``api`` object.
    export const data = api;
