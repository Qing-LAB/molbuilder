/* Workspace dispatcher — single client-side entry point for the
 * Molbuilder tab's workspace state.
 *
 * Per docs/protocols/workspace-state.md § 4.2 (Phase 4 thin
 * wrapper, 2026-06-07).  Public surface:
 *
 *   window.molbuilder.workspace.{subscribe, getState, getStructure,
 *     getSource, isDirty, isEmpty, getSelection,
 *     loadFromFile, generate, applyOp, save, discard, undo,
 *     selection.{toggle,set,add,remove,all,invert,clear,
 *                setMode,setFilters,setCombinator,
 *                applyFilter,writeLabel},
 *     view.{applyState,getState}}
 *
 * Phase 4 is a THIN WRAPPER over the three legacy stores:
 *
 *   - canvas-state (window.molbuilder.structureCanvas)
 *     - owns structure text + source provenance + dirty flag.
 *   - selection store (window.molbuilder.selection.store)
 *     - owns atoms list + selection + filters + mode.
 *   - modify-tab IIFE (exposed pieces on window.molbuilder.modify
 *     + runtime registry: modify.handle, modify.postOp,
 *     modify.applyUndo)
 *     - owns the 3Dmol embed model + history stack + per-op HTTP.
 *
 * The dispatcher synthesises a unified ``Workspace`` snapshot
 * from these three on every read and on every subscription
 * fanout.  Phases 5-9 migrate consumers off the legacy stores;
 * Phase 8 collapses persistence onto a single sessionStorage
 * key.  Phase 9 deletes the legacy stores entirely and the
 * dispatcher becomes the only state owner.
 *
 * On task tabs (structure-optimization / spectrum-calculation /
 * transport-calculation / results) the dispatcher mounts but is
 * effectively empty — those pages don't host the canvas or the
 * selection store, so reads return nulls and ops throw a
 * descriptive error.  ``window.molbuilder.workspace`` is still
 * present so cross-page navigation doesn't ReferenceError.
 *
 * Tests: tests/test_workspace_dispatcher_js.py.
 */
(function (root) {
    "use strict";

    // ─── Internal references resolved on demand ─────────────────── //

    function _canvas() {
        return (root.molbuilder && root.molbuilder.structureCanvas)
            ? root.molbuilder.structureCanvas : null;
    }
    function _store() {
        return (root.molbuilder
                && root.molbuilder.selection
                && root.molbuilder.selection.store)
            ? root.molbuilder.selection.store : null;
    }
    function _handle() {
        return (root.molbuilder
                && root.molbuilder.modify
                && root.molbuilder.modify.handle)
            ? root.molbuilder.modify.handle : null;
    }
    function _runtime() {
        return (root.molbuilder && root.molbuilder.runtime)
            ? root.molbuilder.runtime : null;
    }
    function _projects() {
        return (root.molbuilder && root.molbuilder.projects)
            ? root.molbuilder.projects : null;
    }
    function _molbuilderTab() {
        return (root.molbuilder && root.molbuilder.molbuilderTab)
            ? root.molbuilder.molbuilderTab : null;
    }

    function _missing(name) {
        return new Error(
            "workspace dispatcher: " + name + " not available on this "
            + "page.  Phase 4 wires the dispatcher on /molbuilder; "
            + "other tabs do not host the canvas.");
    }

    // ─── Reads: synthesise the unified state from legacy stores ─── //

    /** Return the structure slice (or null when nothing's loaded). */
    function getStructure() {
        var cs = _canvas();
        var st = _store();
        if (!cs || cs.isEmpty()) return null;
        var canvas = cs.getStructure();   // {source_format, text}
        if (!canvas) return null;
        var s = st ? st.getState() : null;
        return {
            text:          canvas.text,
            source_format: canvas.source_format,
            title:         (s && s.title) || "",
            n_atoms:       s ? s.atoms.length : 0,
            atoms:         s ? s.atoms.slice() : [],
            lattice:       null,    // Structure has no lattice today
        };
    }

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
        if (!st) return {indices: [], mode: "click", filters: [], combinator: "or"};
        var s = st.getState();
        return {
            indices:    s.selection.slice(),
            mode:       s.mode,
            filters:    s.filters.map(function (f) { return Object.assign({}, f); }),
            combinator: s.combinator,
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
                        "workspace dispatcher: subscriber threw", e);
                }
            }
        }
    }

    function _ensureSubscribed() {
        if (_wired) return;
        _wired = true;
        // canvas-state fires on every structure change + dirty toggle.
        var cs = _canvas();
        if (cs && typeof cs.onChange === "function") {
            cs.onChange(_notify);
        }
        // Selection store fires on atoms / selection / filter changes.
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
            throw new TypeError("workspace.subscribe(fn): function required");
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
        setFilters:      function (filters) {
            var s = _store(); if (!s) throw _missing("selection store");
            return s.setFilters(filters);
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
            return s.writeLabel(target, indices);
        },
    };

    // ─── View sub-namespace: passthrough to the 3Dmol embed ──────── //

    var view = {
        applyState: function (patch) {
            var h = _handle(); if (!h) throw _missing("modify embed");
            if (typeof h.applyState !== "function") {
                throw _missing("modify embed.applyState");
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

    // ─── Operations: delegate to existing legacy code ────────────── //

    /**
     * Load a project-saved file into the canvas (.xyz / .pdb).
     * Goes through the universal commit gate — warning modal fires
     * if the canvas is dirty.  Delegates to ``molbuilderTab.commitFile``
     * exposed in cd9655e.
     */
    function loadFromFile(path) {
        if (typeof path !== "string" || !path) {
            return Promise.reject(new TypeError(
                "workspace.loadFromFile(path): non-empty string required"));
        }
        var tab = _molbuilderTab();
        if (!tab || typeof tab.commitFile !== "function") {
            return Promise.reject(_missing(
                "molbuilderTab.commitFile (Phase 4 needs the Molbuilder "
                + "tab's selection-bootstrap mounted)"));
        }
        return Promise.resolve(tab.commitFile(path));
    }

    /**
     * Generate a structure via one of the Sources-card generators.
     * Dispatches by ``kind`` to the matching ``structure.{kind}``
     * module's ``generate(input, opts)`` method.  Each generator
     * already routes through ``structurePage.loadIntoCanvas`` +
     * ``loadStructureText`` so the warning-modal gate + Phase 2
     * extras (backend_used, etc.) are honoured automatically.
     */
    function generate(kind, input, opts) {
        var key = String(kind || "").toLowerCase();
        var moduleName = _GENERATOR_MODULE_BY_KIND[key];
        if (!moduleName) {
            return Promise.reject(new Error(
                "workspace.generate: unknown kind " + JSON.stringify(kind)
                + "; expected one of "
                + Object.keys(_GENERATOR_MODULE_BY_KIND).join(", ")));
        }
        var mod = root.molbuilder && root.molbuilder[moduleName];
        if (!mod || typeof mod.generate !== "function") {
            return Promise.reject(_missing(
                moduleName + " (generator not loaded on this page)"));
        }
        return mod.generate(input, opts || {});
    }

    var _GENERATOR_MODULE_BY_KIND = {
        smiles:  "structureSmiles",
        name:    "structureName",
        dna:     "structureDna",
        rna:     "structureRna",
        peptide: "structurePeptide",
        file:    "structureFile",
    };

    /**
     * Apply a modifier op (delete / add_atom / orient / translate /
     * rotate / electrode / symmetric_electrodes).  Server returns
     * the canonical workspace payload; the modify-tab's existing
     * postOp handles state replacement (canvas-state, selection
     * store, 3Dmol embed all updated atomically).  Delegates to
     * the runtime-registered ``modify.postOp`` (Phase 4 exposed it).
     */
    function applyOp(op, args) {
        if (typeof op !== "string" || !op) {
            return Promise.reject(new TypeError(
                "workspace.applyOp(op, args): op must be a non-empty string"));
        }
        var rt = _runtime();
        if (!rt || typeof rt.whenReady !== "function") {
            return Promise.reject(_missing(
                "runtime registry (cannot resolve modify.postOp)"));
        }
        return rt.whenReady("modify.postOp").then(function (postOp) {
            if (typeof postOp !== "function") {
                throw _missing("modify.postOp");
            }
            var label = _OP_LABELS[op] || op;
            return postOp("/api/modify/" + op, args || {}, label);
        });
    }

    // Human-readable labels for each modifier op — surfaces in the
    // edit-status line during the in-flight window.
    var _OP_LABELS = {
        delete:                "Deleted",
        add_atom:              "Added atom",
        orient:                "Oriented",
        rotate:                "Rotated",
        translate:             "Translated",
        electrode:             "Added electrode",
        symmetric_electrodes:  "Added junction",
    };

    /**
     * Save the workspace structure to disk.  Delegates to the
     * Sources-card Save panel's exported API.  Phase 5 may inline
     * the save here; for Phase 4 the existing save module is the
     * canonical write path.
     */
    function save(opts) {
        opts = opts || {};
        var saveMod = root.molbuilder && root.molbuilder.structureSave;
        if (!saveMod || typeof saveMod.save !== "function") {
            return Promise.reject(_missing("structureSave"));
        }
        return Promise.resolve(saveMod.save(opts));
    }

    /**
     * Discard the workspace structure — clears the canvas + selection.
     * Pure local action (no HTTP).  Fires the warning modal first if
     * the canvas is dirty (the canvas-state clear() handles its own
     * notification; the selection store clear is direct).
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
    function undo() {
        var rt = _runtime();
        if (!rt || typeof rt.whenReady !== "function") {
            return Promise.reject(_missing("runtime registry"));
        }
        return rt.whenReady("modify.applyUndo").then(function (applyUndo) {
            if (typeof applyUndo !== "function") {
                throw _missing("modify.applyUndo");
            }
            return applyUndo();
        });
    }

    // ─── Mount on window.molbuilder.workspace ───────────────────── //

    var api = {
        subscribe:      subscribe,
        getState:       getState,
        getStructure:   getStructure,
        getSource:      getSource,
        getSelection:   getSelection,
        isDirty:        isDirty,
        isEmpty:        isEmpty,
        loadFromFile:   loadFromFile,
        generate:       generate,
        applyOp:        applyOp,
        save:           save,
        discard:        discard,
        undo:           undo,
        selection:      selection,
        view:           view,
    };

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
    root.molbuilder.workspace = api;
    if (_runtime() && typeof _runtime().register === "function") {
        _runtime().register("workspace", api);
    }
    if (typeof module !== "undefined" && module.exports) {
        module.exports = api;
    }
})(typeof window !== "undefined" ? window : globalThis);
