/* Structure-tab canvas state — MolView-internal.  This module no longer mounts a
 * public singleton on ``window.molbuilder.structureCanvas``.  The one process-wide
 * instance is placed on the private ``window.molbuilder.molview._canvasState`` slot;
 * MolView's data model (lib/molview/data-model.js) reads it there.  It lives on the
 * MOLVIEW namespace (NOT ``workspace``): the canvas store is part of MolView's concealed
 * data model, and the 2026-07 carve made ``workspace`` persistence-only, so mounting a
 * data store under ``workspace.*`` was the model leaking into the persistence layer.
 *
 * Single source of truth for "what's loaded in the Structure tab
 * canvas."  Survives browser refresh via sessionStorage; cleared
 * when the tab is closed.  The Structure tab's interactive
 * primitives (load-from-project, generators, modifier panels) read
 * and write this state through the documented surface below.
 *
 * Schema (subject to extension as features land):
 *
 *   {
 *     // Core geometry — the bytes that round-trip to disk on save.
 *     source_format: "xyz" | "pdb",
 *     text:          <the XYZ or PDB body as a string>,
 *
 *     // Provenance — answers "where did this canvas come from?"
 *     source: {
 *       kind: "file" | "smiles" | "dna" | "rna" | "peptide" | "name" |
 *             "load" | "blank",
 *       file: <absolute file path the load came from, or null>,
 *       generator_input: <generator-specific request blob, or null>,
 *     },
 *
 *     // Save tracking.
 *     dirty:        <true if any modifier op has fired since last save>,
 *     last_save_to: <absolute file path of last save-to-project, or null>,
 *   }
 *
 * Storage:
 *   - Live source of truth: this module's private ``_state`` variable.
 *   - Persistence: the workspace dispatcher writes the unified snapshot
 *     under ``sessionStorage["molbuilder.workspace.v1"]``
 *     (workspace-contract.md §4.1 — sole persistence key); this module
 *     reads from it via ``ws.readPersistedSnapshot()`` on init.
 *   - Tab close: sessionStorage is cleared by the browser (default
 *     per-tab semantics).
 *
 * Subscribers:
 *   - ``onChange(cb)`` fires the callback after every state change with
 *     a snapshot of the new state.  Returns an unsubscribe function.
 *   - The viewer adapter, save button, dirty badge, etc. all subscribe.
 *
 * Caller contract (enforced by code review, not runtime):
 *   - ``setStructure`` REPLACES the canvas wholesale (no merge).  After
 *     this call, ``dirty`` is false (the structure as just loaded is
 *     the canonical version).  The dirty flag flips to true only when
 *     ``markDirty()`` is called (typically from a modifier panel).
 *   - ``markSaved(path)`` clears ``dirty`` and records ``last_save_to``.
 *     Call from the save-to-project handler on success only.
 *   - ``clear()`` wipes to empty + sets ``dirty=false``.  Use on
 *     explicit user Discard.
 *
 * History: this primitive was added when the Structure tab merged
 * Build's structure-generation paths.  Before the merge, generated
 * structures had to be saved + reloaded to be editable; the canvas
 * state collapses that round-trip.  See
 * ``docs/tabs/architecture.md`` § 5 for the full design.
 */
(function (root) {
    "use strict";

    // Empty canvas — the initial state and the post-clear() state.
    function _emptyState() {
        return {
            source_format: null,
            text:          null,
            periodicity:   null,
            annotations:   null,      // F1: opaque annotation-channels carrier
            source: {
                kind:             "blank",
                file:             null,
                generator_input:  null,
            },
            dirty:        false,
            last_save_to: null,
        };
    }

    // Normalise a periodicity object to the canonical shape, or null.
    // structure-periodicity.md: cell (3x3 | null), axis_kind
    // ([str,str,str] | null), vacuum ([n,n,n]), kgrid ([n,n,n]).  This
    // rides ALONGSIDE the geometry so a save writes the whole structure
    // (workspace-contract.md §4.0).
    function _normPeriodicity(p) {
        if (!p || typeof p !== "object") return null;
        return {
            cell:      p.cell || null,
            // §3a resolved (effective) cell from the ONE (server) resolver; the display
            // accessor surfaces it.  Carried for the render, never saved (save writes
            // the raw `cell`); refreshed by every server response.
            resolved_cell: p.resolved_cell || null,
            axis_kind: p.axis_kind || null,
            vacuum:    Array.isArray(p.vacuum) ? p.vacuum.slice(0, 3) : [0, 0, 0],
            kgrid:     Array.isArray(p.kgrid)  ? p.kgrid.slice(0, 3)  : [1, 1, 1],
        };
    }

    // Deep copy for the READ boundary (§1.2.1 concealment).  getStructure() hands callers
    // a periodicity they CANNOT use to mutate _state in place -- matrices copied row-wise,
    // flat arrays sliced.  Without this every `ws.getUnitCellInfo().value` / `.periodicity`
    // was a live reference into the store (the "defensive copies" the header promises).
    function _clonePeriodicity(p) {
        if (!p) return null;
        var mat = function (m) {
            return Array.isArray(m)
                ? m.map(function (r) { return Array.isArray(r) ? r.slice() : r; })
                : null;
        };
        return {
            cell:          mat(p.cell),
            resolved_cell: mat(p.resolved_cell),
            axis_kind:     Array.isArray(p.axis_kind) ? p.axis_kind.slice() : null,
            vacuum:        Array.isArray(p.vacuum) ? p.vacuum.slice() : null,
            kgrid:         Array.isArray(p.kgrid) ? p.kgrid.slice() : null,
        };
    }

    // Defensive DEEP copy of the opaque annotations payload (workspace-contract §1.2.1):
    // getStructure() must not hand out a LIVE reference into _state, or a consumer that
    // mutates the returned annotations writes straight into the canvas store.
    function _cloneAnnotations(a) {
        if (a == null) return null;
        try {
            return (typeof structuredClone === "function")
                ? structuredClone(a)
                : JSON.parse(JSON.stringify(a));
        } catch (_) {
            return a;   // non-serialisable opaque payload -> last-resort shared ref
        }
    }

    // Private state.  Initialised by ``_restoreFromSession`` on first
    // public-API call; further mutations go through the setters below.
    var _state = null;
    var _subscribers = [];

    function _restoreFromSession() {
        // Restore from the unified snapshot under ``molbuilder.workspace.v1``
        // (workspace-contract.md §4.1 — the sole persistence key), read via the shared
        // snapshot IO.  Empty state when nothing has been persisted yet.
        var restored = _restoreFromSnapshot();
        _state = restored || _emptyState();
    }

    /**
     * Map the persisted snapshot (read from the shared snapshot-io.js) into this module's
     * internal state shape.  Returns null when the snapshot is absent or carries no
     * structure.  Reads sessionStorage DIRECTLY (not through the dispatcher), so it does
     * not depend on the dispatcher having mounted -- an honest, deterministic reload.
     */
    function _restoreFromSnapshot() {
        // Read the persisted snapshot DIRECTLY from the shared IO (snapshot-io.js) -- NOT
        // "up" through the dispatcher.  sessionStorage is always available and needs no
        // module mounted, so the reload restore is deterministic: it no longer silently
        // drops your unsaved edits when a canvas read runs before the dispatcher has
        // mounted (an honest reload).
        var io = root.molbuilder && root.molbuilder.workspaceSnapshot;
        var snap = (io && typeof io.read === "function") ? io.read() : null;
        if (!snap || !snap.state) return null;
        var st = snap.state;
        var struct = st.structure;
        if (!struct || typeof struct.text !== "string") return null;
        var empty = _emptyState();
        return {
            source_format: struct.source_format || null,
            text:          struct.text,
            periodicity:   _normPeriodicity(struct.periodicity),
            annotations:   struct.annotations || null,   // F1: survive reload restore
            source:        Object.assign({}, empty.source, st.source || {}),
            dirty:         !!st.dirty,
            last_save_to:  st.last_save_to || null,
        };
    }

    function _notify() {
        var snapshot = _snapshot();
        for (var i = 0; i < _subscribers.length; i++) {
            try { _subscribers[i](snapshot); } catch (_) {}
        }
    }

    function _snapshot() {
        // Returns a deep-cloned snapshot so callers can't mutate
        // private state.  Cheap because the payload is small.
        return JSON.parse(JSON.stringify(_state));
    }

    function _ensureInit() {
        if (_state === null) _restoreFromSession();
    }

    // ---- Public API ----

    /**
     * Replace the canvas with a new structure.
     *
     * @param {object} structure - { source_format: "xyz"|"pdb", text: <body> }
     * @param {object} source    - { kind, file?, generator_input? }
     *
     * Resets dirty to false (the freshly-loaded structure IS the
     * canonical version; user modifications haven't happened yet).
     * Clears last_save_to (the loaded structure may match the file
     * on disk but the canvas doesn't claim to track the file's mtime
     * — explicit Save sets last_save_to).
     */
    function setStructure(structure, source) {
        _ensureInit();
        if (!structure || typeof structure !== "object") {
            throw new TypeError(
                "setStructure: structure must be an object");
        }
        var fmt = structure.source_format;
        if (fmt !== "xyz" && fmt !== "pdb") {
            throw new TypeError(
                "setStructure: source_format must be 'xyz' or 'pdb'; "
              + "got " + JSON.stringify(fmt));
        }
        if (typeof structure.text !== "string" || !structure.text) {
            throw new TypeError(
                "setStructure: text must be a non-empty string");
        }
        var src = source || {};
        if (typeof src.kind !== "string" || !src.kind) {
            throw new TypeError(
                "setStructure: source.kind must be a non-empty string");
        }
        _state = {
            source_format: fmt,
            text:          structure.text,
            periodicity:   _normPeriodicity(structure.periodicity),
            // F1: annotation channels ride opaquely (Modify doesn't edit them but
            // must re-emit them on Save so they aren't clobbered).
            annotations:   structure.annotations || null,
            source: {
                kind:            src.kind,
                file:            src.file != null ? String(src.file) : null,
                generator_input: src.generator_input != null
                                    ? src.generator_input : null,
            },
            dirty:        false,
            last_save_to: null,
        };
        _notify();
    }

    /**
     * Replace the canvas text in place — text only; the source
     * provenance + last_save_to stay put.  Marks the canvas dirty
     * (any post-load modifier op flips dirty).  Use after a
     * modifier op that returned new XYZ/PDB bytes; the canvas
     * still came from the same file / generator, but the bytes
     * have diverged.
     *
     * No-op on an empty canvas (nothing to replace).
     */
    function replaceContent(text, periodicity, annotations) {
        _ensureInit();
        if (_state.text == null) return;
        if (typeof text !== "string" || !text) {
            throw new TypeError(
                "replaceContent: text must be a non-empty string");
        }
        _state.text  = text;
        // A modifier op that changed the cell (e.g. add-electrodes captures a
        // lattice) passes the new periodicity; a text-only edit omits it and
        // the existing periodicity is kept.  Pass null to explicitly clear.
        if (periodicity !== undefined) {
            _state.periodicity = _normPeriodicity(periodicity);
        }
        // §19.3.2: a modify op reindexed the per-atom annotation channels WITH the geometry,
        // so the op response's annotations REPLACE the canvas's (else a delete/add would
        // leave the old, now-misaligned channels).  undefined = keep (a text/periodicity-only
        // edit doesn't touch them).
        if (annotations !== undefined) {
            _state.annotations = annotations || null;
        }
        _state.dirty = true;
        _notify();
    }

    /**
     * Merge a periodicity PATCH into the canvas (cell / axis_kind / vacuum / kgrid)
     * WITHOUT touching the geometry text — the write path for UI edits of the lattice
     * + k-grid (§1.2.1 write accessors).  Only the provided keys change; the rest are
     * kept.  Marks dirty + notifies.
     */
    function setPeriodicity(patch) {
        _ensureInit();
        if (_state.text == null) return;     // empty canvas — nothing to set on
        if (!patch || typeof patch !== "object") {
            throw new TypeError("setPeriodicity: patch must be an object");
        }
        var cur = _state.periodicity
            || { cell: null, resolved_cell: null, axis_kind: null,
                 vacuum: [0, 0, 0], kgrid: [1, 1, 1] };
        _state.periodicity = _normPeriodicity({
            cell:      "cell"      in patch ? patch.cell      : cur.cell,
            // resolved_cell is DERIVED -- written back after the server re-resolves an
            // edit (§3a); an explicit `cell` set clears the stale resolved value.
            resolved_cell: "cell" in patch ? null
                           : ("resolved_cell" in patch ? patch.resolved_cell
                              : cur.resolved_cell),
            axis_kind: "axis_kind" in patch ? patch.axis_kind : cur.axis_kind,
            vacuum:    "vacuum"    in patch ? patch.vacuum    : cur.vacuum,
            kgrid:     "kgrid"     in patch ? patch.kgrid     : cur.kgrid,
        });
        _state.dirty = true;
        _notify();
    }

    /**
     * Mark the canvas dirty.  Call after any modifier op (delete,
     * add, orient, region-tag, etc.) that changed the on-disk bytes.
     */
    function markDirty() {
        _ensureInit();
        if (_state.text == null) return;  // empty canvas — nothing to dirty
        if (_state.dirty) return;          // already dirty — no-op + no notify
        _state.dirty = true;
        _notify();
    }

    /**
     * Mark the canvas saved.  Clears dirty and records the path the
     * canvas was saved to.  Call from the save-to-project handler on
     * success.
     *
     * @param {string} path - absolute file path
     */
    function markSaved(path) {
        _ensureInit();
        if (typeof path !== "string" || !path) {
            throw new TypeError("markSaved: path must be a non-empty string");
        }
        _state.dirty = false;
        _state.last_save_to = path;
        _notify();
    }

    /**
     * Wipe the canvas back to empty.  Drops the in-memory text +
     * source provenance + dirty flag + last_save_to.  Use on
     * explicit user Discard.
     */
    function clear() {
        _ensureInit();
        _state = _emptyState();
        _notify();
    }

    function isEmpty()       { _ensureInit(); return _state.text == null; }
    function isDirty()       { _ensureInit(); return !!_state.dirty; }
    function getSource()     { _ensureInit(); return _snapshot().source; }
    function getStructure()  {
        _ensureInit();
        if (_state.text == null) return null;
        return {
            source_format: _state.source_format,
            text:          _state.text,
            periodicity:   _clonePeriodicity(_state.periodicity),   // defensive copy
            annotations:   _cloneAnnotations(_state.annotations),   // F1: opaque carry (defensive copy)
        };
    }
    function getLastSavedTo(){ _ensureInit(); return _state.last_save_to; }

    /**
     * Subscribe to state changes.  Callback fires AFTER every
     * setStructure / markDirty / markSaved / clear with a snapshot
     * of the new state.  Returns an unsubscribe function.
     */
    function onChange(cb) {
        if (typeof cb !== "function") {
            throw new TypeError("onChange: cb must be a function");
        }
        _subscribers.push(cb);
        return function () {
            var ix = _subscribers.indexOf(cb);
            if (ix >= 0) _subscribers.splice(ix, 1);
        };
    }

    /**
     * Force a re-read from sessionStorage.  Useful when another
     * tab on the same origin has modified the storage (rare in this
     * app's deployment model — one user, one window) or to reset
     * tests between runs.
     */
    function reloadFromStorage() {
        _restoreFromSession();
        _notify();
    }

    var api = {
        setStructure:      setStructure,
        replaceContent:    replaceContent,
        setPeriodicity:    setPeriodicity,
        markDirty:         markDirty,
        markSaved:         markSaved,
        clear:             clear,
        isEmpty:           isEmpty,
        isDirty:           isDirty,
        getSource:         getSource,
        getStructure:      getStructure,
        getLastSavedTo:    getLastSavedTo,
        onChange:          onChange,
        reloadFromStorage: reloadFromStorage,
    };

    // UMD-ish export.  In Node test contexts (canvas-state-only
    // unit tests) ``module.exports`` carries the api so tests can
    // ``require()`` it without a DOM.  In the browser the api is mounted on the PRIVATE
    // ``window.molbuilder.molview._canvasState`` slot — the legacy
    // ``window.molbuilder.structureCanvas`` global + the ``runtime.register(
    // "structure.canvas", ...)`` are gone.  data-model.js reads from
    // ``molview._canvasState`` (this mount) and also honours a legacy ``structureCanvas``
    // mount as a test-only escape hatch for harnesses that ``require()`` this file and
    // assign the return value manually.
    if (typeof module !== "undefined" && module.exports) {
        module.exports = api;
    } else {
        root.molbuilder = root.molbuilder || {};
        // Mount on the MolView namespace -- the canvas store is part of MolView's
        // concealed data model, NOT the (persistence-only) workspace (2026-07 carve).
        // data-model.js merges ``molview`` with ``|| {}``, so this survives regardless of
        // script-load order.
        root.molbuilder.molview = root.molbuilder.molview || {};
        root.molbuilder.molview._canvasState = api;
    }
})(typeof window !== "undefined" ? window : globalThis);
