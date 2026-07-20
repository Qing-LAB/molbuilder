/* **Workspace-internal as of Phase 9 (2026-06-13)** — this module
 * no longer mounts a public singleton on
 * ``window.molbuilder.molview.selection.store``.  The workspace
 * dispatcher (lib/workspace/dispatcher.js) creates + holds the
 * one process-wide instance via the ``_createStore`` factory
 * exported below.  Every external consumer goes through
 * ``window.molbuilder.workspace.selection.*`` (=``ws.selection.*``).
 *
 * The factory stays mounted under ``window.molbuilder.molview.selection.
 * _createStore`` so per-test isolation (Node L2 tests + a future
 * Playwright `_test` hook) can spin up fresh stores without
 * driving the dispatcher.
 *
 * Atom-selection store -- factory + state shape + mutators.
 *
 * THE state holder for the atom-selection module.  Consumers (panel,
 * viewer-adapter, page bootstrap) subscribe to the store and call its
 * mutators on user actions.  No module talks directly to any other;
 * every cross-module signal goes through this store.
 *
 * Full spec:  docs/protocols/molview-module.md
 *
 * State shape:
 *
 *   {
 *     sourceFile: null | string         // current structure path (.xyz / .pdb)
 *     atoms:      Atom[]                // current structure's atoms
 *     selection:  number[]              // THE selection set;
 *                                       // shared across modes;
 *                                       // kept sorted ascending
 *     pickOrder:  number[]              // same atom-indices as
 *                                       // ``selection`` but in
 *                                       // click order (vertex for
 *                                       // the 3-atom angle readout
 *                                       // = pickOrder[1]).  Always
 *                                       // a permutation of
 *                                       // ``selection``.
 *     mode:       "click" | "filter"    // which editor is visible
 *     filters:    Filter[]              // filter drafts (not
 *                                       // materialised until
 *                                       // applyFilter() runs)
 *     combinator: "or" | "and"
 *     loading:    boolean
 *     error:      null | string
 *   }
 *
 * Click mode edits ``selection`` atom-by-atom via ``toggleAtom``
 * (client-side; no server roundtrip).  Filter mode composes a query
 * that the user explicitly applies via ``applyFilter()`` -- the
 * server evaluates and replaces ``selection``.  Switching modes
 * does NOT touch ``selection``: any selection persists across mode
 * changes.  This is the "selection is the truth, modes are just
 * editors" design.
 *
 * Event protocol: one event type ("state changed").  Each
 * synchronous mutator fires subscribers once via microtask.  Async
 * mutators (the ones that hit the server) fire twice: at start
 * (loading=true) and end (loading=false).  Reentrance is safe.
 */
(function (root) {
    "use strict";

    // The channel/index FEATURES (knownChannels, the 1-based by_index shift)
    // depend on the L1 models (atom-channels.js / atom-index.js); the rest of
    // the store does not.  So this module stays loadable standalone (node
    // harnesses) — those two functions reference L1 at CALL time, and a caller
    // that uses them must have L1 loaded (modify.html does; a test that
    // exercises them supplies it).

    const EVAL_URL    = "/api/selection/eval";
    const ATOMS_URL   = "/api/selection/atoms";
    const FILES_READ  = "/api/files/read";

    function _initialState() {
        return {
            sourceFile: null,
            atoms:      [],
            selection:  [],
            // Pick-order shadow: the same atom-indices as
            // ``selection`` but kept in the order the user picked
            // them.  ``selection`` is sorted ascending (set
            // semantics, used by every consumer that doesn't care
            // about click history); ``pickOrder`` preserves the
            // click sequence so the 3-atom angle readout can use
            // the middle pick as the vertex.  Always a subset of
            // ``selection`` modulo permutation; mutators keep the
            // two in lock-step.  Reset to [] on every source-file
            // change, adoptAtoms, clearSelection, setSelection
            // (input order is the new pick order), selectAll, and
            // invertSelection (no user picks ⇒ atom-index order).
            pickOrder:  [],
            mode:       "click",
            // "Show selected only" -- a view flag that lives in the store (the
            // single source of truth) so the panel checkbox + the viewer
            // adapter both drive/read it through the store, not a cross-module
            // handle.  Isolate only takes visual effect with a non-empty
            // selection (the adapter render gates on selSet.size > 0).
            isolate:    false,
            // View-toggle flags -- the render engine (molview-render-streamline.md §7.2) reads
            // these from the store; the View menu / rail toggles WRITE them via setViewFlag.
            // The store is the single source of view state (task #64); the engine renders from it.
            showIndex:  false,     // atom-index labels
            showForces: false,     // force-vector overlay
            showCell:   true,      // unit-cell box
            showAxis:   true,      // axes
            forceScale: undefined, // Å per force unit (undefined -> engine default)
            filters:    [],
            combinator: "or",
            loading:    false,
            error:      null,
        };
    }
    // The view-toggle flag names (task #64). Booleans except forceScale (number|undefined).
    var VIEW_FLAG_NAMES = ["showIndex", "showForces", "showCell", "showAxis", "forceScale"];

    // --------------------------------------------------------------- //
    //  Rule translation: JS filters[] + combinator -> server rule     //
    // --------------------------------------------------------------- //

    // Translate one panel-side filter draft into a server rule.
    // Returns ``null`` for an "incomplete" filter (empty / placeholder
    // value); callers skip null operands so a half-typed filter row
    // doesn't silently wipe the result under AND combinator.  An
    // incomplete row under AND ought to behave like "no constraint
    // from this row yet" -- the user hasn't told us anything to
    // intersect with -- rather than "match nothing".
    function _filterToRule(f) {
        switch (f.kind) {
            case "by_element": {
                const elements = (f.value || "").split(",")
                    .map((s) => s.trim()).filter(Boolean);
                if (elements.length === 0) return null;
                return { op: "by_element", elements: elements };
            }
            case "by_index": {
                const raw = (f.value || "").trim();
                if (!raw) return null;
                // The user types 1-based indices (matching the display);
                // the server by_index_range rule is 0-based -- shift at this
                // boundary (data-vocabulary.md § 3.1).
                const expression =
                    root.molbuilder.atomIndexModel.shiftExpression(raw, -1);
                return { op: "by_index_range", expression: expression };
            }
            case "by_residue": {
                // Category channel (atom-annotations.md §5): comma-separated
                // residue names -> the existing server ByResidueName rule.
                const names = (f.value || "").split(",")
                    .map((s) => s.trim()).filter(Boolean);
                if (names.length === 0) return null;
                return { op: "by_residue_name", names: names };
            }
            case "by_label": {
                const name = (f.value || "").trim();
                if (!name) return null;
                return { op: "by_region", name: name };
            }
            default:
                throw new Error("unknown filter kind: " + f.kind);
        }
    }

    function _filtersToRule(filters, combinator) {
        // Drop incomplete rows so a half-typed filter doesn't poison
        // an AND with an empty-set operand.
        const operands = filters.map(_filterToRule)
            .filter((r) => r !== null);
        if (operands.length === 0) return null;   // signal "no filter"
        if (operands.length === 1) return operands[0];
        return { op: combinator, operands: operands };
    }

    // --------------------------------------------------------------- //
    //  Per-atom payload normalisation                                  //
    // --------------------------------------------------------------- //

    function _normaliseAtom(raw) {
        // 2026-06-09: idempotent normaliser — accepts EITHER the
        // server wire shape (raw.regions, raw.is_frozen, snake_case
        // metadata) OR an already-normalised atom object (labels,
        // isFrozen, camelCase).  The save flow's Save-as re-anchor
        // passes ws.getAtoms() output back through adoptSession;
        // pre-fix that silently wiped labels because the wire-shape
        // reads (raw.regions, raw.is_frozen) returned undefined on
        // already-normalised input.
        const out = {
            index:   raw.index,
            element: raw.element,
            labels:  Array.isArray(raw.labels)
                         ? raw.labels.slice()
                         : (Array.isArray(raw.regions)
                                ? raw.regions.slice() : []),
            isFrozen: raw.is_frozen !== undefined ? !!raw.is_frozen
                      : !!raw.isFrozen,
        };
        // Coordinates ride ON the atom (workspace-contract.md §1.2.1 -- the atom is
        // the geometric truth, not a re-parsed xyz string).  Kept as numbers.
        if (raw.x !== undefined && raw.x !== null) out.x = Number(raw.x);
        if (raw.y !== undefined && raw.y !== null) out.y = Number(raw.y);
        if (raw.z !== undefined && raw.z !== null) out.z = Number(raw.z);
        // Accept both snake_case (wire) and camelCase (in-memory)
        // metadata fields — same idempotence reasoning.
        if (raw.atom_name)    out.atomName    = raw.atom_name;
        else if (raw.atomName) out.atomName   = raw.atomName;
        if (raw.residue_name) out.residueName = raw.residue_name;
        else if (raw.residueName) out.residueName = raw.residueName;
        if (raw.chain_id)     out.chainId     = raw.chain_id;
        else if (raw.chainId) out.chainId     = raw.chainId;
        // residue_id is a NUMBER (1-based residue sequence) that can legitimately be 0,
        // so guard with != null, not truthiness.  It reaches the atom via the payload's
        // top-level residue_ids[] distributed per-atom in data-model _applyWorkspacePayload
        // (the wire has no per-atom residue_id).  Carrying it here makes molview.data the
        // COMPLETE single source -- modify ops no longer need a parallel state.* mirror.
        if (raw.residue_id != null)      out.residueId = raw.residue_id;
        else if (raw.residueId != null)  out.residueId = raw.residueId;
        return out;
    }

    // --------------------------------------------------------------- //
    //  HTTP helpers                                                    //
    // --------------------------------------------------------------- //

    function _parseJsonResponse(r) {
        return r.json()
            .then((j) => ({ ok: r.ok, body: j }))
            .catch(() => ({
                ok:   false,
                body: { error: "non-JSON response (status " + r.status + ")" },
            }));
    }

    function _postJson(url, body, signal) {
        return fetch(url, {
            method:  "POST",
            headers: {"Content-Type": "application/json"},
            body:    JSON.stringify(body),
            signal:  signal,
        }).then(_parseJsonResponse);
    }

    function _getJson(url, signal) {
        return fetch(url, { signal: signal }).then(_parseJsonResponse);
    }

    // --------------------------------------------------------------- //
    //  Store factory                                                   //
    // --------------------------------------------------------------- //

    function _create() {
        const state = _initialState();
        // Change-kinds for the dirty-bit set (see _notify). A mutating op passes ONLY its own kind.
        const CHANGE = { SELECTION: "selection", ATOMS: "atoms", COORDS: "coords",
                         LABELS: "labels", FILTERS: "filters", MODE: "mode", VIEW: "view",
                         SOURCE: "source", STATUS: "status" };
        const subscribers = new Set();
        let pending = false;
        let inflight = null;
        // Loader callback: ``async (text, filename) => void``.  The
        // page bootstrap injects the viewer's loader here via
        // ``setLoader`` so the store stays free of DOM / 3Dmol /
        // page-specific globals (spec §5 rule 3: "no DOM, no 3Dmol,
        // no Flask -- pure data + fetch").  Without a loader, the
        // store still fetches the atom list but does not attempt to
        // populate any viewer -- useful in headless contexts (tests,
        // tabs that just want the atom list).
        let structureLoader = null;

        function _snapshot() {
            return {
                sourceFile: state.sourceFile,
                atoms:      state.atoms.slice(),
                selection:  state.selection.slice(),
                // pickOrder is the click-order shadow consumed by
                // the selection panel's measurement readout
                // (vertex = pickOrder[1] for the 3-atom angle).
                // Pre-task-#304 the snapshot dropped it, so the
                // panel always read ``undefined`` and silently fell
                // back to the geometric-vertex heuristic — the
                // entire chemist's-pick semantic was dead end-to-
                // end despite being correctly maintained inside the
                // store.  Slice for the same defensive-copy reason
                // selection has.
                pickOrder:  state.pickOrder.slice(),
                mode:       state.mode,
                isolate:    state.isolate,
                showIndex:  state.showIndex,
                showForces: state.showForces,
                showCell:   state.showCell,
                showAxis:   state.showAxis,
                forceScale: state.forceScale,
                filters:    state.filters.slice(),
                combinator: state.combinator,
                loading:    state.loading,
                error:      state.error,
            };
        }

        // Dirty-bit set. Each mutating op passes ITS OWN change-kind to _notify(kind); the kinds
        // accumulate across the coalesced window and are handed to subscribers as a `changes`
        // array, then cleared. So every consumer does only the op that change needs -- a selection
        // click marks SELECTION -> the panel diffs highlight (not a 444-row rebuild) and the engine
        // re-applies halos (not a regen). An op that marks nothing yields an empty `changes`, which
        // consumers treat as "unknown -> do the safe full update" (never stale, just not optimal).
        let _dirty = {};
        // Mark a change-kind WITHOUT firing (accumulates into the next _notify). For a body that
        // mutates state but relies on a LATER _notify (e.g. _fetchAtoms runs inside _run, which
        // notifies at the end): the body marks its kind so the terminal notify carries it too.
        function _mark(kind) { if (kind) _dirty[kind] = true; }
        function _notify(kind) {
            if (kind) _dirty[kind] = true;
            if (pending) return;
            pending = true;
            Promise.resolve().then(() => {
                pending = false;
                const changes = Object.keys(_dirty);
                _dirty = {};
                const snap = _snapshot();
                subscribers.forEach((fn) => {
                    try { fn(snap, changes); }
                    catch (e) {
                        if (root.console) root.console.error(
                            "[selection.store] subscriber threw", e
                        );
                    }
                });
            });
        }

        function _abortInflight() {
            if (inflight) {
                try { inflight.abort(); } catch (e) { /* ignore */ }
                inflight = null;
            }
        }

        function _newSignal() {
            _abortInflight();
            inflight = new AbortController();
            return inflight.signal;
        }

        function getState() { return _snapshot(); }

        function subscribe(fn) {
            if (typeof fn !== "function") {
                throw new TypeError("subscribe(fn) expects a function");
            }
            subscribers.add(fn);
            try { fn(_snapshot()); }
            catch (e) {
                if (root.console) root.console.error(e);
            }
            return function unsubscribe() { subscribers.delete(fn); };
        }

        async function _fetchAtoms(signal) {
            // Every path here replaces state.atoms; the notify comes LATER from _run (marked
            // STATUS), so mark ATOMS now or the panel would DIFF (stale rows) instead of rebuild.
            _mark(CHANGE.ATOMS);
            if (!state.sourceFile) {
                state.atoms = [];
                return;
            }
            const { ok, body } = await _postJson(ATOMS_URL, {
                structure_path: state.sourceFile,
            }, signal);
            if (!ok) {
                state.error = (body && body.error) || "atom fetch failed";
                state.atoms = [];
                return;
            }
            state.atoms = (body.atoms || []).map(_normaliseAtom);
            state.error = null;
        }

        async function _loadViewer(signal) {
            if (!state.sourceFile) return true;
            // No injected loader == headless mode: skip viewer load,
            // fetch atoms only.
            if (typeof structureLoader !== "function") return true;
            const url = FILES_READ + "?path="
                      + encodeURIComponent(state.sourceFile);
            const { ok, body } = await _getJson(url, signal);
            if (!ok) {
                state.error = (body && body.error)
                    ? body.error : "file read failed";
                return false;
            }
            const filename = state.sourceFile.split("/").pop();
            try {
                await structureLoader(body.text, filename);
            } catch (e) {
                if (e && e.name === "AbortError") throw e;
                state.error = "loader failed: "
                            + (e && e.message ? e.message : String(e));
                return false;
            }
            return true;
        }

        // Async mutator runner: loading=true notify, run body, then
        // loading=false notify.  Cancels any in-flight previous async
        // mutator via AbortController.  Clears ``state.error`` at
        // the start so a recovered mutator (e.g. the user switches
        // to a clean file after a failed filter eval) doesn't leave
        // a stale red banner on screen; the body re-sets the error
        // if it fails again.
        async function _run(body) {
            const signal = _newSignal();
            state.loading = true;
            state.error   = null;
            _notify(CHANGE.STATUS);
            try {
                await body(signal);
            } catch (e) {
                if (e && e.name === "AbortError") return;
                state.error = e && e.message ? e.message : String(e);
            } finally {
                if (!signal.aborted) {
                    state.loading = false;
                    _notify(CHANGE.STATUS);
                }
            }
        }

        // ----------------------------------------------------------- //
        //  PUBLIC: source file                                        //
        // ----------------------------------------------------------- //

        function setSourceFile(path) {
            const next = path || null;
            if (next === state.sourceFile) return Promise.resolve();
            return _run(async (signal) => {
                state.sourceFile = next;
                state.atoms      = [];
                state.selection  = [];   // a fresh file starts empty
                state.pickOrder  = [];
                if (next === null) return;
                // If the viewer failed to load, don't pull atoms --
                // showing rows for a structure the user can't see in
                // 3D is more confusing than the visible error banner.
                const ok = await _loadViewer(signal);
                if (!ok) return;
                await _fetchAtoms(signal);
            });
        }

        // A SAVE just wrote the in-memory model to ``path``; re-anchor sourceFile so
        // later label edits target the new file.  SYNCHRONOUS + sourceFile-ONLY: a
        // save does NOT change the model, so this must NOT reload, refetch, or touch
        // atoms/selection (contrast setSourceFile, which is a fresh OPEN and clears
        // everything).  This is the door molview.data.markSaved routes through, so a
        // save flow never reaches into the store to hand-set the source (the old
        // save.js `adoptSession({sourceFile})` reach-around).
        function noteSavedTo(path) {
            const next = path || null;
            if (next === state.sourceFile) return;
            state.sourceFile = next;
            _notify(CHANGE.STATUS);
        }

        // Inject the structure loader.  Called by the page
        // bootstrap with the viewer-specific loader (e.g.
        // modify/viewer.js's ``loadStructureText``, which accepts
        // both XYZ and PDB content -- the server's /api/build/load
        // sniffs the format).  Pass
        // ``null`` to detach (the store falls back to atom-list-
        // only / "headless" mode).
        function setLoader(fn) {
            if (fn !== null && typeof fn !== "function") {
                throw new TypeError("setLoader(fn): function or null required");
            }
            structureLoader = fn;
        }

        // Rehydrate from a session snapshot WITHOUT re-loading the
        // viewer.  Used by /modify's sessionStorage restore path
        // where the viewer model has already been populated
        // synchronously via ``applyStructure(...)``.  Differs from
        // setSourceFile in two ways:
        //   1. it skips _loadViewer entirely (the viewer is already
        //      populated -- re-loading would discard the camera /
        //      indices and double-fetch over HTTP for no gain);
        //   2. it accepts a pre-validated selection that survives
        //      the structure swap, so the panel and adapter come
        //      back in sync without losing the user's pick.
        //
        // ``atoms``  (BOMB-0 follow-up, 2026-06-07): the canonical
        // per-atom payload to install.  When supplied, the disk
        // fetch is SKIPPED — saved snapshots carry the in-memory
        // post-op atoms which are authoritative until the user
        // saves to disk.  Without this knob a Modify-tab session
        // that did a Delete + navigated away would come back
        // showing the PRE-delete atom list because /api/selection/
        // atoms reads the disk (which still has the pre-op file).
        // Falls back to the disk fetch when ``atoms`` is absent
        // (cross-tab handoff path where the file IS the source of
        // truth, and pre-fix sessions still in storage).
        function adoptSession({ sourceFile, selection, atoms }) {
            if (sourceFile && typeof sourceFile !== "string") {
                return Promise.reject(
                    new TypeError("sourceFile must be a string or null")
                );
            }
            const sel = Array.isArray(selection)
                ? selection.filter((i) => typeof i === "number")
                : [];
            const preFetched = Array.isArray(atoms) ? atoms : null;
            return _run(async (signal) => {
                state.sourceFile = sourceFile || null;
                state.atoms      = preFetched
                    ? preFetched.map(_normaliseAtom) : [];
                state.selection  = sel.slice().sort((a, b) => a - b);
                // adoptSession carries no click history (it's a
                // session snapshot from disk); pickOrder mirrors
                // selection in ascending order so consumers don't
                // see a stale 3-atom angle vertex from before
                // the restore.
                state.pickOrder  = state.selection.slice();
                if (preFetched) {
                    // Drop selection indices that no longer exist
                    // in the adopted atoms list (mirrors adoptAtoms).
                    const n = state.atoms.length;
                    state.selection = state.selection.filter(
                        (i) => Number.isInteger(i) && i >= 0 && i < n);
                    const valid = new Set(state.selection);
                    state.pickOrder = state.pickOrder.filter(
                        (i) => valid.has(i));
                    return;
                }
                if (!state.sourceFile) return;
                await _fetchAtoms(signal);
            });
        }

        function refreshAtoms() {
            return _run(async (signal) => {
                await _fetchAtoms(signal);
            });
        }

        /**
         * Replace state.atoms with a server-provided list — for
         * modifier-op responses + cross-tab handoff payloads that
         * already carry the canonical per-atom shape (the same
         * /api/selection/atoms returns).  No HTTP roundtrip, no
         * signal abort — this is the in-memory sync path the BOMB-0
         * fix needs.
         *
         * ``sourceFile`` (optional): when a file-open drives this
         * call it names the loaded path so the store's sourceFile
         * (the "Loaded: X" readout + any later refreshAtoms) is set
         * in the SAME synchronous write as the atoms.  This is what
         * lets a file load install atoms+source+selection in ONE
         * store write (the load contract — see
         * ``_applyWorkspacePayload`` and molview-module.md §19.3):
         * pass it and the fresh molecule is fully settled before the
         * "ready" signal (getNAtoms) is observable, so nothing done
         * on the settled structure gets clobbered by a later reset.
         * Omitted -> sourceFile untouched (a modifier op keeps the
         * current file).
         *
         * Selection state is filtered: indices that no longer
         * exist (e.g. atoms removed by a Delete op) are dropped.
         * Per-op subtree state (filter drafts, mode) is preserved.
         *
         * Synchronous publication — subscribers see the new atoms
         * + the filtered selection in a single fanout.
         */
        function adoptAtoms(rawAtoms, sourceFile) {
            if (!Array.isArray(rawAtoms)) {
                throw new TypeError(
                    "adoptAtoms: rawAtoms must be an array");
            }
            if (sourceFile !== undefined) {
                state.sourceFile = sourceFile || null;
            }
            state.atoms = rawAtoms.map(_normaliseAtom);
            // Drop selection indices that no longer exist (the
            // op may have removed atoms; the array length is now
            // the upper bound).  Sorted-ascending invariant
            // preserved.  pickOrder is filtered against the same
            // surviving set so the click-order shadow stays in
            // lockstep with selection.
            const n = state.atoms.length;
            state.selection = state.selection.filter(
                (i) => Number.isInteger(i) && i >= 0 && i < n);
            const surviving = new Set(state.selection);
            state.pickOrder = state.pickOrder.filter(
                (i) => surviving.has(i));
            state.error = null;
            _notify(CHANGE.ATOMS);
        }

        /**
         * Swap the atoms' COORDINATES in place — the frame/time axis
         * (workspace-contract.md §1.5): a new frame is the SAME atoms at new
         * positions.  Keeps identity, labels, frozen, selection, filters, and
         * mode intact — only x/y/z change.  Requires exactly one [x,y,z] per
         * existing atom (the caller — the dispatcher's frame API — validates
         * the count against the structure first).  Synchronous publication.
         */
        function setCoords(coords) {
            if (!Array.isArray(coords) || coords.length !== state.atoms.length) {
                throw new Error(
                    "setCoords: one [x,y,z] per atom required (have "
                    + state.atoms.length + " atoms, got "
                    + (Array.isArray(coords) ? coords.length : typeof coords) + ")");
            }
            for (let i = 0; i < state.atoms.length; i++) {
                const p = coords[i];
                state.atoms[i].x = Number(p[0]);
                state.atoms[i].y = Number(p[1]);
                state.atoms[i].z = Number(p[2]);
            }
            _notify(CHANGE.COORDS);
        }

        // ----------------------------------------------------------- //
        //  PUBLIC: UI mode  (just controls which editor is visible)   //
        //  Switching modes does NOT touch state.selection.            //
        // ----------------------------------------------------------- //

        function setMode(mode) {
            if (mode !== "click" && mode !== "filter") {
                return Promise.reject(new Error("bad mode: " + mode));
            }
            if (mode === state.mode) return Promise.resolve();
            state.mode = mode;
            _notify(CHANGE.MODE);
            return Promise.resolve();
        }

        // "Show selected only".  Synchronous view-flag mutator (no server):
        // set the flag + notify; the viewer adapter re-renders from the
        // snapshot and the panel checkbox reflects state.isolate.
        function setIsolate(on) {
            const next = !!on;
            if (next === state.isolate) return;
            state.isolate = next;
            _notify(CHANGE.VIEW);
        }

        // Set a view-toggle flag (task #64). One setter for all of them: the View menu / rail
        // writes here; the render engine reads the flag from the snapshot and re-renders.
        // Booleans are coerced; forceScale takes a number (anything else -> undefined = default).
        function setViewFlag(name, value) {
            if (VIEW_FLAG_NAMES.indexOf(name) === -1) return;
            const next = (name === "forceScale")
                ? (typeof value === "number" ? value : undefined)
                : !!value;
            if (state[name] === next) return;
            state[name] = next;
            _notify(CHANGE.VIEW);
        }

        // Auto-clear "show selected only" when a selection change EMPTIES the set
        // (non-empty -> empty) while isolate is on: an isolate view with nothing
        // selected would show nothing.  This is a selection-STATE rule, so it lives
        // in the store (moved here from the old view-controls toggle).  It fires ONLY
        // on the empty TRANSITION -- setIsolate() does not route through here -- so
        // "check the box, then click an atom" still works (checking while empty keeps
        // the flag; the click that follows leaves the set non-empty).
        function _afterSelectionChange(prevLen) {
            if (state.isolate && prevLen > 0 && state.selection.length === 0) {
                state.isolate = false;
            }
            _notify(CHANGE.SELECTION);
        }

        // ----------------------------------------------------------- //
        //  PUBLIC: click-mode editing  (client-side, no HTTP)         //
        // ----------------------------------------------------------- //

        function toggleAtom(index) {
            if (typeof index !== "number") {
                return Promise.reject(new TypeError("index must be number"));
            }
            const prevLen = state.selection.length;
            const i = state.selection.indexOf(index);
            if (i === -1) {
                // Insert in sorted order so consumers always see a
                // sorted-ascending selection.  Splice-insert is O(n)
                // but n is the selection size, not the structure
                // size; fine for typical workflows.
                const insertAt = state.selection.findIndex((j) => j > index);
                const next = state.selection.slice();
                if (insertAt === -1) next.push(index);
                else                 next.splice(insertAt, 0, index);
                state.selection = next;
                // pickOrder is append-only on toggle-in so the
                // most-recently-clicked atom lands at the end —
                // that's the user's pick sequence the angle
                // readout reads.
                state.pickOrder = state.pickOrder.concat([index]);
            } else {
                const next = state.selection.slice();
                next.splice(i, 1);
                state.selection = next;
                state.pickOrder = state.pickOrder.filter(
                    (j) => j !== index);
            }
            _afterSelectionChange(prevLen);
            return Promise.resolve();
        }

        function setSelection(indices) {
            if (!Array.isArray(indices)) {
                return Promise.reject(new TypeError("indices must be array"));
            }
            // Two-pass dedup that preserves INPUT order for
            // pickOrder while keeping the sorted-ascending
            // invariant for ``selection``.  Test-set / batch
            // callers that pass [1, 0, 2] get
            // selection=[0,1,2] (set semantics) and
            // pickOrder=[1,0,2] (vertex = middle = 0).
            const prevLen = state.selection.length;
            const seen = new Set();
            const orderedIn = [];
            indices.forEach((x) => {
                if (typeof x !== "number") return;
                if (seen.has(x)) return;
                seen.add(x);
                orderedIn.push(x);
            });
            state.selection = orderedIn.slice().sort((a, b) => a - b);
            // Defensive copy so a caller mutating ``orderedIn``
            // post-return (or chaining off ``state``) can't bleed
            // into the store.  Symmetric with every other mutator.
            state.pickOrder = orderedIn.slice();
            _afterSelectionChange(prevLen);
            return Promise.resolve();
        }

        // Union the given indices into the current selection.
        // Sugar over ``setSelection`` for the common "Add N atoms"
        // workflow (avoids consumer boilerplate of read-merge-set).
        function addToSelection(indices) {
            if (!Array.isArray(indices)) {
                return Promise.reject(new TypeError("indices must be array"));
            }
            // pickOrder gets the NEW atoms appended in input order
            // — same semantic as a series of toggleAtom calls.
            const merged = new Set(state.selection);
            const additions = [];
            indices.forEach((i) => {
                if (typeof i !== "number") return;
                if (!merged.has(i) && additions.indexOf(i) === -1) {
                    additions.push(i);
                }
                merged.add(i);
            });
            state.selection = Array.from(merged).sort((a, b) => a - b);
            state.pickOrder = state.pickOrder.concat(additions);
            _notify(CHANGE.SELECTION);
            return Promise.resolve();
        }

        // Subtract the given indices from the current selection.
        // Preserves pickOrder of the SURVIVING atoms (the user's
        // sequence is "stable minus the removed ones").
        function removeFromSelection(indices) {
            if (!Array.isArray(indices)) {
                return Promise.reject(new TypeError("indices must be array"));
            }
            const prevLen = state.selection.length;
            const drop = new Set(indices.filter((i) => typeof i === "number"));
            const survivingPickOrder = state.pickOrder.filter(
                (i) => !drop.has(i));
            state.selection = state.selection.filter((i) => !drop.has(i));
            state.pickOrder = survivingPickOrder;
            _afterSelectionChange(prevLen);
            return Promise.resolve();
        }

        // Select every atom in the loaded structure.
        function selectAll() {
            return setSelection(state.atoms.map((a) => a.index));
        }

        // Invert the current selection.
        function invertSelection() {
            const sel = new Set(state.selection);
            const out = [];
            state.atoms.forEach((a) => {
                if (!sel.has(a.index)) out.push(a.index);
            });
            return setSelection(out);
        }

        // ----------------------------------------------------------- //
        //  PUBLIC: filter drafts  (client-side, no eval until apply)  //
        // ----------------------------------------------------------- //

        function setFilters(filters) {
            if (!Array.isArray(filters)) {
                return Promise.reject(new TypeError("filters must be array"));
            }
            state.filters = filters.slice();
            _notify(CHANGE.FILTERS);
            return Promise.resolve();
        }

        // Filter-editing conveniences -- panel code was doing
        // ``next = filters.slice(); next.push/splice; setFilters(next)``
        // for each of add / remove / update; collapse to one call.

        function addFilter(filter) {
            if (!filter || typeof filter !== "object") {
                return Promise.reject(new TypeError("filter object required"));
            }
            return setFilters(state.filters.concat([filter]));
        }

        function removeFilter(index) {
            if (typeof index !== "number"
                || index < 0
                || index >= state.filters.length) {
                return Promise.reject(new Error("filter index out of range"));
            }
            const next = state.filters.slice();
            next.splice(index, 1);
            return setFilters(next);
        }

        function updateFilter(index, filter) {
            if (typeof index !== "number"
                || index < 0
                || index >= state.filters.length) {
                return Promise.reject(new Error("filter index out of range"));
            }
            if (!filter || typeof filter !== "object") {
                return Promise.reject(new TypeError("filter object required"));
            }
            const next = state.filters.slice();
            next[index] = filter;
            return setFilters(next);
        }

        function setCombinator(c) {
            if (c !== "or" && c !== "and") {
                return Promise.reject(new Error("bad combinator: " + c));
            }
            if (c === state.combinator) return Promise.resolve();
            state.combinator = c;
            _notify(CHANGE.FILTERS);
            return Promise.resolve();
        }

        // ----------------------------------------------------------- //
        //  PUBLIC: applyFilter -- materialise filter into selection   //
        // ----------------------------------------------------------- //

        function applyFilter() {
            return _run(async (signal) => {
                if (!Array.isArray(state.atoms) || state.atoms.length === 0) {
                    state.error = "no atoms to filter";
                    return;
                }
                const rule = _filtersToRule(state.filters, state.combinator);
                if (rule === null) {
                    // No filters -- treat as "select nothing".
                    state.selection = [];
                    state.pickOrder = [];
                    state.error     = null;
                    return;
                }
                // A5b: evaluate against the IN-MEMORY workspace atoms (the store),
                // NOT a disk read -- so filters see unsaved labels/edits.  The
                // store's atoms carry element + labels + isFrozen + residue.
                const { ok, body } = await _postJson(EVAL_URL, {
                    atoms: state.atoms,
                    rule:  rule,
                }, signal);
                if (!ok) {
                    state.error = (body && body.error)
                                  || "filter eval failed";
                    return;
                }
                // The eval endpoint returns ``selected_indices``; do
                // NOT trust the shape blindly -- a misshapen response
                // (string elements, null, non-array) would poison
                // state.selection and break the next toggleAtom
                // .indexOf comparison.  Filter to safe ints, sort
                // for consumer convenience.
                const raw = (body && Array.isArray(body.selected_indices))
                    ? body.selected_indices : [];
                state.selection = raw
                    .filter((i) => Number.isInteger(i))
                    .sort((a, b) => a - b);
                // Filter-eval results have no user pick history;
                // mirror selection so the angle readout sees a
                // sensible deterministic order (ascending atom
                // index).  A filter-applied selection of 3 atoms
                // falls back to the geometric vertex via the
                // ordered-isnt-meaningful path.
                state.pickOrder = state.selection.slice();
                state.error     = null;
                // Race safety net.  applyFilter shares the _run
                // abort signal with setSourceFile, so a sequence
                //   1. user picks B in the sidebar
                //   2. setSourceFile(B) starts; state.atoms is
                //      synchronously cleared; the file-load + atom
                //      fetch are pending
                //   3. user clicks Apply filter before the load
                //      finishes
                // aborts setSourceFile mid-fetch -- state.atoms
                // never gets repopulated -- while the server happily
                // evaluates against the on-disk XYZ and we set
                // state.selection above.  The user ends up with
                // selection=[...] + atoms=[], the panel shows
                // "N / 0 atoms" + empty list, the viewer paints
                // halos.  Refetch atoms here so the panel re-syncs.
                // (No-op when the load did complete: state.atoms
                // is non-empty in the common case.)
                if (state.atoms.length === 0) {
                    await _fetchAtoms(signal);
                }
            });
        }

        // ----------------------------------------------------------- //
        //  PUBLIC: label writes (sidecar) -- selection unchanged      //
        // ----------------------------------------------------------- //

        // Gather the current per-atom labels back into the {regions, frozen}
        // shape, so writeLabel can apply a REPLACE-per-target change locally.
        function _currentLabels() {
            const regions = {};
            const frozen = [];
            for (let i = 0; i < state.atoms.length; i++) {
                const a = state.atoms[i];
                if (!a) continue;
                (a.labels || []).forEach((L) => {
                    if (!regions[L]) regions[L] = [];
                    regions[L].push(i);
                });
                if (a.isFrozen) frozen.push(i);
            }
            return { regions, frozen };
        }

        function writeLabel(target, indices) {
            if (!target || typeof target !== "string") {
                return Promise.reject(new TypeError("target required"));
            }
            if (!Array.isArray(indices)) {
                return Promise.reject(new TypeError("indices must be array"));
            }
            // §4.0 / save-flow §1: assigning a label is an IN-MEMORY edit, like
            // every other modify op -- it does NOT touch disk.  The
            // .molstruct.json is written only on explicit Save, which serialises
            // the whole store (regions + frozen + periodicity).  Previously this
            // POSTed /api/selection/save on every Assign, silently rewriting the
            // LOADED file's sidecar before the user ever saved.
            //
            // Compute the new state from the current atoms + this REPLACE-per-
            // target change, then apply in place (no HTTP, no disk read).
            const cur = _currentLabels();
            const idx = indices
                .filter((i) => Number.isInteger(i))
                .sort((a, b) => a - b);
            if (target === "frozen_atoms") {
                cur.frozen = idx;
            } else if (idx.length === 0) {
                delete cur.regions[target];    // empty -> remove the region
            } else {
                cur.regions[target] = idx;     // REPLACE that region's members
            }
            state.error = null;
            _applyLabelsFromSidecar(cur.regions, cur.frozen);
            return Promise.resolve({ ok: true });
        }

        /**
         * Refresh each atom's ``labels`` + ``isFrozen`` from a
         * server-returned sidecar payload — without re-reading
         * atoms from disk.  Used by ``writeLabel`` to update the
         * workspace's in-memory state after a successful sidecar
         * write, so the panel re-renders with the new tag column
         * without clobbering modifier-op atom additions.
         */
        function _applyLabelsFromSidecar(regions, frozenAtoms) {
            const byIndex = new Map();
            if (regions && typeof regions === "object") {
                Object.keys(regions).forEach((label) => {
                    const idxs = regions[label];
                    if (!Array.isArray(idxs)) return;
                    idxs.forEach((i) => {
                        if (!Number.isInteger(i)) return;
                        if (!byIndex.has(i)) byIndex.set(i, []);
                        byIndex.get(i).push(label);
                    });
                });
            }
            const frozenSet = new Set();
            if (Array.isArray(frozenAtoms)) {
                frozenAtoms.forEach((i) => {
                    if (Number.isInteger(i)) frozenSet.add(i);
                });
            }
            for (let i = 0; i < state.atoms.length; i++) {
                const a = state.atoms[i];
                a.labels   = byIndex.get(i) ? byIndex.get(i).slice() : [];
                a.isFrozen = frozenSet.has(i);
            }
            _notify(CHANGE.LABELS);
        }

        // ----------------------------------------------------------- //
        //  PUBLIC: clear selection                                    //
        // ----------------------------------------------------------- //

        function clearSelection() {
            if (state.selection.length === 0
                && state.pickOrder.length === 0) {
                return Promise.resolve();
            }
            state.selection = [];
            state.pickOrder = [];
            _notify(CHANGE.SELECTION);
            return Promise.resolve();
        }

        // Unified filterable-channel enumeration (atom-annotations.md §5, L2):
        // the live set of channels across the current atoms, via the pure L1
        // channel model.  Consumers enumerate this instead of special-casing
        // regions vs frozen.  A caller that uses this must have L1 loaded
        // (a feature-level dependency -- see the note at module top).
        function knownChannels() {
            return root.molbuilder.atomChannelModel.channelKinds(state.atoms);
        }

        return {
            // reads
            getState:           getState,
            knownChannels:      knownChannels,
            subscribe:          subscribe,
            // source file
            setSourceFile:      setSourceFile,
            noteSavedTo:        noteSavedTo,      // save-as source re-anchor (sync, no reload)
            refreshAtoms:       refreshAtoms,
            adoptAtoms:         adoptAtoms,
            setCoords:          setCoords,        // frame coord-swap (workspace §1.5)
            adoptSession:       adoptSession,
            setLoader:          setLoader,
            // mode
            setMode:            setMode,
            setIsolate:         setIsolate,
            setViewFlag:        setViewFlag,     // view-toggle flags (task #64; §7.2)
            // selection editing
            toggleAtom:         toggleAtom,
            setSelection:       setSelection,
            addToSelection:     addToSelection,
            removeFromSelection: removeFromSelection,
            selectAll:          selectAll,
            invertSelection:    invertSelection,
            clearSelection:     clearSelection,
            // filter drafts
            setFilters:         setFilters,
            addFilter:          addFilter,
            removeFilter:       removeFilter,
            updateFilter:       updateFilter,
            setCombinator:      setCombinator,
            applyFilter:        applyFilter,
            // sidecar writes
            writeLabel:         writeLabel,
        };
    }

    // Phase 9 (2026-06-13): the public singleton mount + the
    // matching ``runtime.register("selection.store", ...)`` call
    // are GONE.  The dispatcher owns the one process-wide
    // instance, created via the factory below.  Consumers reach
    // it through ``window.molbuilder.workspace.selection.*``.
    //
    // The factory stays mounted at the existing namespace so
    // test harnesses + a future per-instance test hook can spin
    // up isolated stores.
    root.molbuilder = root.molbuilder || {};
    root.molbuilder.molview = root.molbuilder.molview || {};
    root.molbuilder.molview.selection = root.molbuilder.molview.selection || {};
    root.molbuilder.molview.selection._createStore = _create;

    // Phase 5 (fused module): the ws.selection SURFACE, built around ANY raw
    // store.  selection-panel + viewer-adapter consume the renamed surface
    // (toggle/all/clear/invert/... + an ``indices``-shaped getState/subscribe)
    // so a readonly/ephemeral inspector can own an ISOLATED selection.
    //
    // This is the SHARED-CONSUMER subset.  The dispatcher's ws.selection is a
    // SUPERSET: it adds workspace-lifecycle methods (getAtoms, setSourceFile,
    // refreshAtoms) that the singleton needs but readonly inspectors do not —
    // the shared consumers (panel/adapter/mount-panel) call only the common set,
    // so the ephemeral surface intentionally omits those.  The snapshot shape is
    // a single source: ws.selection delegates to _surfaceSnapshot (below), so
    // there is no twin to keep in sync.
    // Defensive per-atom copy for the read surface (workspace-contract.md §1.2.1).  The
    // snapshot slices the atoms ARRAY, but the atom OBJECTS must be copies too -- otherwise a
    // consumer mutating ``getState().atoms[i].x`` (or ``getStructure().atoms[i]``, which
    // reads through this same snapshot) would leak straight into the store.  The atom shape
    // is flat scalars + a nested ``labels`` array (the only nested field), so a shallow
    // object copy + a ``labels`` slice is a full defensive copy.
    function _cloneAtom(a) {
        if (!a || typeof a !== "object") return a;
        var c = Object.assign({}, a);
        if (Array.isArray(a.labels)) c.labels = a.labels.slice();
        return c;
    }
    function _ephemeralSnapshot(st) {
        if (!st) {
            return { indices: [], mode: "click", isolate: false,
                     showIndex: false, showForces: false, showCell: true, showAxis: true,
                     forceScale: undefined,
                     filters: [],
                     combinator: "or", loading: false, error: null, atoms: [],
                     sourceFile: null, pickOrder: [] };
        }
        return {
            indices:    Array.isArray(st.selection) ? st.selection.slice() : [],
            mode:       st.mode || "click",
            isolate:    !!st.isolate,
            // View-toggle flags (task #64) -- the render engine reads these through this snapshot.
            showIndex:  !!st.showIndex,
            showForces: !!st.showForces,
            showCell:   !!st.showCell,
            showAxis:   !!st.showAxis,
            forceScale: st.forceScale,
            filters:    (st.filters || []).map(function (f) { return Object.assign({}, f); }),
            combinator: st.combinator || "or",
            loading:    !!st.loading,
            error:      st.error || null,
            atoms:      Array.isArray(st.atoms) ? st.atoms.map(_cloneAtom) : [],
            sourceFile: st.sourceFile || null,
            pickOrder:  Array.isArray(st.pickOrder) ? st.pickOrder.slice() : [],
        };
    }
    function _surface(s) {
        return {
            toggle:          function (i)     { return s.toggleAtom(i); },
            set:             function (ix)    { return s.setSelection(ix); },
            add:             function (ix)    { return s.addToSelection(ix); },
            remove:          function (ix)    { return s.removeFromSelection(ix); },
            all:             function ()      { return s.selectAll(); },
            invert:          function ()      { return s.invertSelection(); },
            clear:           function ()      { return s.clearSelection(); },
            setMode:         function (m)     { return s.setMode(m); },
            setIsolate:      function (on)    { return s.setIsolate(on); },
            setViewFlag:     function (n, v)  { return s.setViewFlag(n, v); },   // task #64
            setFilters:      function (f)     { return s.setFilters(f); },
            addFilter:       function (f)     { return s.addFilter(f); },
            removeFilter:    function (i)     { return s.removeFilter(i); },
            updateFilter:    function (i, p)  { return s.updateFilter(i, p); },
            setCombinator:   function (c)     { return s.setCombinator(c); },
            applyFilter:     function ()      { return s.applyFilter(); },
            writeLabel:      function (t, ix) { return s.writeLabel(t, ix); },
            setLoader:       function (fn)    { return s.setLoader(fn); },
            // Install atoms directly (readonly inspectors pass atoms built from
            // the viewer handle; providing ``atoms`` skips any server fetch).
            adoptSession:    function (o)     { return s.adoptSession(o); },
            getState:        function ()      { return _ephemeralSnapshot(s.getState()); },
            subscribe:       function (fn)    {
                return s.subscribe(function (st, changes) { fn(_ephemeralSnapshot(st), changes); });
            },
        };
    }
    // Public factory for a fresh, ISOLATED selection with the ws.selection
    // surface — pass it to selectionPanel.mount(host,{store}) +
    // viewerAdapter.attach(handle,{store}).
    root.molbuilder.molview.selection.createEphemeralStore = function () {
        return _surface(_create());
    };
    // The ONE surface-snapshot shaper (raw state -> the {indices,...} surface
    // shape).  The dispatcher's ws.selection.getState/subscribe delegate to THIS
    // (was a hand-maintained character-identical twin -- now a single source).
    root.molbuilder.molview.selection._surfaceSnapshot = _ephemeralSnapshot;
    // Exposed so the dispatcher's getStructure() (which reads the RAW store snapshot, not
    // this surface shaper) can apply the SAME defensive per-atom copy -- workspace-contract
    // §1.2.1 immutable reads, one shared helper.
    root.molbuilder.molview.selection._cloneAtom = _cloneAtom;
})(typeof window !== "undefined" ? window : globalThis);
