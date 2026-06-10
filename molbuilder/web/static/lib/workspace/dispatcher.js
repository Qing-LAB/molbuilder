/* Workspace dispatcher — single client-side entry point for the
 * Molbuilder tab's workspace state.
 *
 * Per docs/protocols/workspace-state.md (§ 4.2 public surface,
 * § 4.4 wire shape, § 4.5 per-op selection rule).  Public
 * surface, current as of 2026-06-08:
 *
 *   window.molbuilder.workspace.{
 *     // Reads + subscription
 *     subscribe, getState, getStructure, getSource, getSelection,
 *     isDirty, isEmpty,
 *     // Operations (server round-trip + atomic state replacement)
 *     loadFromFile, generate, applyOp, save, discard, undo,
 *     // Internal pipeline exposed for the modify-tab loader
 *     applyPayload,
 *     // Persistence
 *     readPersistedSnapshot, STORAGE_KEY,
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
 *   * ``applyOp`` owns the modifier-op fetch + cross-store
 *     update pipeline (phase 5 self-sufficient).  Builds the
 *     request body via ``window.molbuilder.modify.currentStateBody``
 *     (modify-tab IIFE exposes this hook), POSTs
 *     ``/api/modify/<op>``, routes the response through
 *     ``_applyWorkspacePayload``.
 *   * ``_applyWorkspacePayload`` is THE single cross-store sync
 *     point: canvas-state.replaceContent + modify-tab
 *     applyStructure hook + selection store adoptAtoms +
 *     selection_remap (phase 3 wire shape).  Every entry point
 *     (applyOp, the modify-tab's loadStructureText) routes
 *     through it.
 *   * Persistence: debounced write to
 *     ``sessionStorage["molbuilder.workspace.v1"]`` on every
 *     state change + final flush on pagehide.  Legacy mirrors
 *     (``molbuilder.structure_canvas`` from canvas-state and
 *     ``modify-state`` from the modify viewer IIFE) still
 *     write in parallel during the phase 8→9 transition window;
 *     they're documented in the workspace-state.md migration
 *     table as scheduled for retirement but technically active.
 *
 * Underlying stores (still active; phase 9 marked them
 * "internal" without code deletion):
 *
 *   - canvas-state (``window.molbuilder.structureCanvas``)
 *     - owns structure text + source provenance + dirty flag.
 *   - selection store (``window.molbuilder.selection.store``)
 *     - owns atoms list + selection + filters + mode.
 *   - modify-tab IIFE
 *     (``window.molbuilder.modify.{handle, currentStateBody,
 *       applyStructure}`` + runtime registry
 *     ``modify.handle``, ``modify.applyUndo``)
 *     - owns the 3Dmol embed model, the IIFE's per-state mirror
 *       (state.xyz / state.elements / ...), and the undo history
 *       stack.
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
        // Phase 8 — persistence.  Debounced write to the unified
        // sessionStorage key on every state change.  Final flush
        // on ``pagehide`` (wired below in the mount block) so a
        // last-microtask change isn't lost when the user navigates.
        _schedulePersist();
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

    // ─── Operations ──────────────────────────────────────────────── //
    //
    // ``applyOp`` owns its pipeline end-to-end.  The other op methods
    // (loadFromFile, generate, save, undo) delegate to the existing
    // legacy modules — those are intentional thin wrappers that exist
    // so consumers can stay on the unified ``ws.*`` API while the
    // underlying implementations migrate at their own pace.

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
     * Apply a server-returned workspace payload to every store
     * atomically.  Phase 5 (2026-06-07) — the single cross-store
     * sync point.
     *
     * Side effects, in order:
     *   1. canvas-state.replaceContent(text)  when ``touchCanvas``
     *      (modifier-op flow; flips the dirty bit).  Generator + sidebar
     *      flows pass ``touchCanvas: false`` because canvas-state was
     *      already set via ``structurePage.loadIntoCanvas`` (dirty=false).
     *   2. modify-tab IIFE state + 3Dmol embed via the registered
     *      ``window.molbuilder.modify.applyStructure`` hook.  Synchronous;
     *      after this call state.* and the embed reflect the payload.
     *   3. ``selection_remap`` (Phase 3 wire shape) is applied to the
     *      existing selection — surviving indices remap, removed atoms
     *      drop, all in one ``setSelection`` call.
     *   4. ``opts.resetSelection`` clears the selection (sidebar /
     *      generator flow).  Mutually exclusive with ``selection_remap``.
     *   5. Dispatcher subscribers are notified.
     */
    function _applyWorkspacePayload(payload, opts) {
        opts = opts || {};
        var touchCanvas    = opts.touchCanvas !== false;
        var resetSelection = !!opts.resetSelection;
        var text = payload && (payload.text || payload.xyz);

        // Capture the PRE-op selection BEFORE any store mutation.
        // adoptAtoms below naively filters ``state.selection`` to
        // in-range indices for the new atom count; that filter
        // destroys the data the selection_remap step needs to
        // translate (selecting atom 2 + deleting atom 0: adoptAtoms
        // drops 2 since 2 >= 2 in the 2-atom result; without
        // capturing first, the remap reads from [] instead of [2]
        // and produces [] instead of the correct [1]).
        var st = _store();
        var preSelection = (st && typeof st.getState === "function")
            ? st.getState().selection.slice() : [];

        // 1. Canvas-state — text + dirty bit.
        var cs = _canvas();
        if (touchCanvas && cs && text
                && typeof cs.replaceContent === "function") {
            try { cs.replaceContent(text); } catch (_) { /* swallow */ }
        }

        // 2. modify-tab applyStructure hook (IIFE state.* + 3Dmol
        //    embed only).  This hook is the modify-tab's
        //    self-update; it does NOT touch the selection store
        //    or canvas-state (the dispatcher owns those).  When the
        //    hook is absent (task tabs without a modify IIFE) the
        //    call is a no-op.
        var modifyHook = root.molbuilder
                      && root.molbuilder.modify
                      && root.molbuilder.modify.applyStructure;
        if (typeof modifyHook === "function") {
            try { modifyHook(payload, opts); } catch (_) { /* swallow */ }
        }

        // 3. Selection store atoms — the BOMB-0 cross-store sync.
        //    Single source of truth: this is the ONLY place the
        //    dispatcher consults ``payload.atoms``.
        if (st && Array.isArray(payload.atoms)
                && typeof st.adoptAtoms === "function") {
            st.adoptAtoms(payload.atoms);
        }

        // 4. Selection mapping.  Reads from ``preSelection`` (captured
        //    before adoptAtoms' destructive filter) so a Delete op's
        //    selection_remap translates the user's ORIGINAL anchor
        //    rather than the post-filter empty set.
        var remap = payload && payload.extra
                 && payload.extra.selection_remap;
        if (Array.isArray(remap) && st
                && typeof st.setSelection === "function") {
            var newSel = [];
            for (var i = 0; i < preSelection.length; i++) {
                var idx = preSelection[i];
                var newIdx = (idx >= 0 && idx < remap.length)
                    ? remap[idx] : null;
                if (newIdx != null) newSel.push(newIdx);
            }
            st.setSelection(newSel);
        } else if (resetSelection && st
                && typeof st.clearSelection === "function") {
            st.clearSelection();
        }

        // 5. Notify dispatcher subscribers.
        _notify();
    }

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
    /**
     * Parse structure bytes (XYZ / PDB) through ``/api/build/load``
     * and install the result atomically.  Used by every "load
     * existing bytes into the workspace" flow — sidebar
     * commitFile (reads file from disk + passes text here), every
     * Sources-card generator (the engine returned text, we want
     * canonical metadata).  The previous IIFE-local
     * ``window.molbuilder.loadStructureText`` is now a thin alias
     * over this method.
     *
     * Returns the canonical workspace payload (text + atoms +
     * extras) so callers can use ``r.atoms`` for follow-up store
     * sync (e.g. selection-bootstrap's setSourceFile via
     * adoptSession).  Throws on network error or non-ok envelope —
     * symmetric with ``applyOp``.
     *
     * Caller-owned canvas-state: the apply call uses
     * ``touchCanvas: false`` because every documented call site
     * (sidebar commitFile via ``structurePage.loadIntoCanvas``,
     * the Sources-card generators that pre-set canvas-state before
     * invoking ``viewerLoader``) already drove the
     * canvas-state side-effect.  Forcing a second canvas write
     * here would clobber the dirty-bit handshake the caller
     * already negotiated.  See ``_applyWorkspacePayload`` for the
     * full canvas-state contract.
     */
    async function loadFromText(text, filename) {
        const resp = await root.fetch("/api/build/load", {
            method:  "POST",
            headers: { "Content-Type": "application/json" },
            body:    JSON.stringify({ text: text, filename: filename }),
        });
        const r = await resp.json();
        if (!r.ok) {
            throw new Error(r.error || "Load failed.");
        }
        _applyWorkspacePayload(r, {
            touchCanvas:    false,
            resetSelection: true,
        });
        return r;
    }

    function applyOp(op, args) {
        if (typeof op !== "string" || !op) {
            return Promise.reject(new TypeError(
                "workspace.applyOp(op, args): op must be a non-empty string"));
        }
        var csb = root.molbuilder
               && root.molbuilder.modify
               && root.molbuilder.modify.currentStateBody;
        if (typeof csb !== "function") {
            return Promise.reject(_missing("modify.currentStateBody"));
        }
        var body = Object.assign(csb(), args || {});
        return root.fetch("/api/modify/" + op, {
            method:  "POST",
            headers: { "Content-Type": "application/json" },
            body:    JSON.stringify(body),
        }).then(function (resp) {
            return resp.json().then(function (r) {
                return { httpOk: resp.ok, r: r };
            });
        }).then(function (env) {
            if (!env.httpOk || !env.r || !env.r.ok) {
                throw new Error(
                    (env.r && env.r.error) || "modify/" + op + " failed");
            }
            _applyWorkspacePayload(env.r, { touchCanvas: true });
            return env.r;
        });
    }

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
    function save(opts) {
        var saveMod = root.molbuilder && root.molbuilder.structureSave;
        if (!saveMod || typeof saveMod.save !== "function") {
            return Promise.reject(_missing("structureSave"));
        }
        return Promise.resolve(saveMod.save());
    }

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

    // ─── Persistence (Phase 8) ─────────────────────────────────── //

    /**
     * Phase 8 of the workspace-state migration (2026-06-07): the
     * dispatcher owns the unified sessionStorage mirror under
     * ``molbuilder.workspace.v1``.  The legacy per-store mirrors
     * (``molbuilder.structure_canvas`` and ``modify-state``) are
     * still actively writing in parallel — a follow-up commit
     * will retire them once ``restoreModifyState`` is migrated to
     * read from the unified key (workspace-state.md § 6 step 8
     * "what did NOT land" sub-list).  Until then a refresh
     * gracefully falls back to whichever mirror has data.
     *
     * Schema:
     *
     *   {
     *     v:        1,
     *     saved_at: "<ISO 8601>",
     *     state: {
     *       structure:   <ws.getStructure() snapshot, or null>,
     *       source:      <ws.getSource() snapshot>,
     *       dirty:       <ws.isDirty()>,
     *       last_save_to: <canvas-state.getLastSavedTo()>,
     *       selection:   <ws.getSelection()>,
     *       view:        <ws.view.getState()>,
     *     },
     *   }
     *
     * Restore policy mirrors the per-store dirty-gated heuristic
     * landed in cd9655e: when ``dirty=true`` OR ``source.file=null``
     * the saved snapshot is authoritative; otherwise the source
     * file on disk is.  This protects unsaved edits while letting
     * external file changes propagate.
     *
     * On the modify-tab restoreModifyState already runs at
     * ``DOMContentLoaded``; the dispatcher's restore is layered
     * on top: it reads ``molbuilder.workspace.v1`` first; falls
     * back to the legacy mirrors when absent.
     */
    // Shared classic-script constants — see lib/constants.js.
    // Fallback string keeps the dispatcher functional in test
    // contexts that don't load constants.js (e.g. the JS-unit
    // tests that boot a minimal window stub).
    var STORAGE_KEY = ((root.molbuilder || {}).constants || {})
        .SS_WORKSPACE
        || "molbuilder.workspace.v1";
    var _persistDeadline = null;

    function _serialise() {
        // The snapshot always carries the full structure (including
        // ``structure.atoms``).  The cd9655e dirty-gate
        // ("when canvas is clean AND has a source file, refetch
        // from disk on restore so external changes propagate") is
        // applied AT RESTORE TIME — the snapshot consumer reads
        // ``state.dirty`` + ``state.source.file`` and decides
        // whether to install the saved atoms or force a disk
        // refetch.  Earlier this gate lived here and nulled out
        // ``structure.atoms`` — but downstream consumers also
        // derive ``elements`` / ``atom_names`` / etc. from
        // ``atoms[]``, so nulling it broke the IIFE state restore
        // (empty Anchor readouts, broken refreshSelectionUI).
        return {
            v:        1,
            saved_at: new Date().toISOString(),
            state: {
                structure:    getStructure(),
                source:       getSource(),
                dirty:        isDirty(),
                last_save_to: _lastSavedTo(),
                selection:    getSelection(),
                view:         view.getState(),
            },
        };
    }

    function _lastSavedTo() {
        var cs = _canvas();
        return (cs && typeof cs.getLastSavedTo === "function")
            ? cs.getLastSavedTo() : null;
    }

    function _persistToSession() {
        if (!root.sessionStorage) return;
        try {
            root.sessionStorage.setItem(
                STORAGE_KEY, JSON.stringify(_serialise()));
        } catch (e) {
            // Quota exceeded or storage disabled.  Same handling
            // as canvas-state.js / modify-state — log + skip.
            if (root.console && root.console.warn) {
                root.console.warn(
                    "workspace dispatcher: could not persist:",
                    e && e.message);
            }
        }
    }

    function _schedulePersist() {
        if (!root.sessionStorage) return;
        if (_persistDeadline) clearTimeout(_persistDeadline);
        _persistDeadline = setTimeout(function () {
            _persistDeadline = null;
            _persistToSession();
        }, 100);
    }

    /**
     * Returns the parsed snapshot if the unified key has one,
     * else null.  Callers (modify-tab restoreModifyState +
     * canvas-state init) check this before falling back to their
     * legacy mirrors.  Public so the modify-tab + canvas-state
     * can read it during the migration window.
     */
    function readPersistedSnapshot() {
        if (!root.sessionStorage) return null;
        var raw;
        try { raw = root.sessionStorage.getItem(STORAGE_KEY); }
        catch (_) { return null; }
        if (!raw) return null;
        try {
            var parsed = JSON.parse(raw);
            if (!parsed || parsed.v !== 1) return null;
            return parsed;
        } catch (_) {
            return null;
        }
    }

    // ─── Mount on window.molbuilder.workspace ───────────────────── //

    var api = {
        subscribe:             subscribe,
        getState:              getState,
        getStructure:          getStructure,
        getSource:             getSource,
        getSelection:          getSelection,
        isDirty:               isDirty,
        isEmpty:               isEmpty,
        loadFromFile:          loadFromFile,
        loadFromText:          loadFromText,
        generate:              generate,
        applyOp:               applyOp,
        applyPayload:          _applyWorkspacePayload,
        save:                  save,
        discard:               discard,
        undo:                  undo,
        selection:             selection,
        view:                  view,
        // Phase 8 — persistence:
        readPersistedSnapshot: readPersistedSnapshot,
        STORAGE_KEY:           STORAGE_KEY,
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

    // Phase 8 — eagerly subscribe to the underlying stores so the
    // persistence pipeline fires on every state change, even when
    // no UI consumer has subscribed to ``ws.subscribe`` yet.  The
    // pre-Phase-8 behaviour was lazy-attach (only when someone
    // called ``ws.subscribe``); that meant a fresh page with no
    // panel mounted would never persist its workspace state.
    _ensureSubscribed();

    // Phase 8 — pagehide flush.  The debounced persist may have a
    // pending timer when the user navigates; force a final write
    // so the next page sees the latest state.  Mirrors what
    // modify/viewer.js does for its modify-state key today (the
    // legacy mirror co-exists during the migration window).
    if (root.addEventListener) {
        root.addEventListener("pagehide", function () {
            if (_persistDeadline) {
                clearTimeout(_persistDeadline);
                _persistDeadline = null;
            }
            _persistToSession();
        });
    }

    if (typeof module !== "undefined" && module.exports) {
        module.exports = api;
    }
})(typeof window !== "undefined" ? window : globalThis);
