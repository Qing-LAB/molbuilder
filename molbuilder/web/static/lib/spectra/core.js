/* Spectra inspector core -- VibrationView mode viewer + Plotly chart + form.
 *
 * THE shared spectra-inspector implementation.  Two consumers:
 *
 *   * /spectrum-calculation -- the spectrum-calculation page (via
 *                  spectra/viewer.js, which contains only the
 *                  DOMContentLoaded bootstrap that calls into this
 *                  module).
 *   * /results  -- the unified post-merge inspector page (via
 *                  lib/inspectors/spectra.js, which mounts this
 *                  module's exported ``mount(host, opts)`` into the
 *                  registry-supplied ``#inspector-host``).
 *
 * Both consumers call ``window.molbuilder.spectraInspector.mount()``,
 * so a bug fix here lands in both consumers automatically -- no
 * fork.  Same lift pattern as ``lib/trajectory/core.js`` (see
 * docs/web/results.md).
 *
 * Wires the schema-driven SpectraConfig form (via the shared
 * ``molbuilder.formSchema`` helpers) to the three Spectra API
 * endpoints (spec § 10):
 *
 *   GET  /api/build/schema/pyscf?calculation=vibration -- the form
 *   POST /api/task-setup/handover       -- Send to Task setup
 *   POST /api/spectra/load              -- parse a results JSON
 *
 * --- DOM scoping convention ----------------------------------------
 * The inspector body lives inside ``mountInspector(rootEl)``, so DOM
 * queries via ``$(id)`` are scoped to rootEl.  On /spectra rootEl is
 * the document (full-page mount); on /results rootEl is the
 * inspector-partial container injected into ``#inspector-host``.
 *
 * No build step: ES2017+ JavaScript, no bundler.
 */
(function (root) {
    "use strict";

    /**
     * Mount the spectra inspector inside ``rootEl``.
     *
     * Parameters:
     *   rootEl -- DOM element (or document) that contains the
     *             spectra ids the inspector wires up.
     *   opts   -- what the mounting page hands in:
     *             ``file``               a results JSON to open (the
     *                                    Results tab's pick);
     *             ``mountVibrationView`` the page's door to the mode
     *                                    viewer;
     *             ``onFormsReady``       called when the forms are
     *                                    rendered (the chemistry card
     *                                    asks again).
     *
     * Returns a handle ``{ dispose() }`` so the registry can tear
     * down timers + Plotly listeners between inspector swaps.
     * /spectra's bootstrap holds the handle for completeness but
     * never disposes (the tab lives for the lifetime of the page).
     */

    function mountInspector(rootEl, opts) {
    rootEl = rootEl || document;
    opts = opts || {};

    const $ = (id) => rootEl.querySelector("#" + id);

    // ----- DOM refs (resolved once at startup) -----------------
    const els = {
        spectrumChart:  null,
        // The spectrum's whole section, its heading, its controls and the
        // sentence saying why a result with no strength draws mode
        // positions instead (web/spectra.md § 2, § 9b.3); the table the
        // route marks.
        spectrumSection: null,
        spectrumHeading: null,
        spectrumControls: null,
        spectrumAbsent: null,
        modesTable:     null,
        formContainer:  null,   // wraps both panels: the generate-side gate + the edit-listener scope
        form:           { pyscf: null, siesta: null },   // one schema-built form per engine
        engineStrip:    null,
        engineNote:     null,
        sendBtn:        null,
        sendStatus:     null,
        preflightPanel: null,
        resultsSummary: null,
        resultsMeta:    null,
        methodsBlock:   null,
        methodsText:    null,
        methodsCopy:    null,
        methodsNote:    null,
        displayFloor:   null,
        displayFloorOut: null,
        modesTbody:     null,
        // Selection sync + ES panel additions (§ 9.2.2 / § 9.2.4):
        modesFilter:    null,
        modesCsvBtn:    null,
        modesFilterCount: null,
        modesTheadRow:  null,
        esPanel:        null,
        esModeIdx:      null,
        esModeFreq:     null,
        esBarDiagram:   null,
        esSummary:      null,
        // The run's progress: one status line and the phase dots.  The
        // dropdown is the one route to a file, and a running result is
        // followed by itself (web/spectra.md § 7).
        watchStatus:    null,
        phaseIndicator: null,
        // Spectrum chart Lorentzian-broadening control.
        broadeningFwhm: null,
        // VibrationView mode-animation viewer (vibrationview.md; § 9.2.3).
        modeViewerWrap: null,
        modeViewer:     null,
        viewerStatus:   null,
        animAmplitude:  null,
        animAmplitudeVal: null,
        animSpeed:      null,
        animSpeedVal:   null,
        animToggle:     null,
    };

    // Last successful render payload + interactive state.
    // Bucketed state shape per docs/web/results.md, mirroring the
    // trajectory inspector's shape for cross-inspector consistency.  Five disjoint
    // buckets + a state-machine field; backward-compat aliases keep
    // existing render code working with the legacy flat ``state.X``
    // shape; transition() is the SINGLE entry-point for fileState /
    // lifecycle / derived writes.
    //
    // Form / calculation state (schema, lastScript, etc.) is OUTSIDE
    // the five buckets per contract § 11 ("calculation-tab form state
    // ... has its own workspace + form-dirty contracts").  Those
    // fields stay at the top level of `state` for backward compat;
    // they'll migrate to a future spectra-form contract.
    //
    // The state machine field is the contract's enforcement point:
    // bucket mutations OUTSIDE transition() are forbidden for
    // fileState / lifecycle / derived (matrix § 3); viewState and
    // uiPrefs CAN be mutated by event handlers (mode pick, filter
    // input, broadening slider, etc.).
    const state = {
        // Form / calculation state -- NOT covered by the results-state
        // contract.  Kept at the top level of `state` for backward
        // compat; migrates to a future workspace contract.
        schemas:        { pyscf: null, siesta: null },   // the catalogue narrowed to (engine, vibration)
        engine:         "pyscf",                          // the strip's choice: the description's engine
        engineChosenByUser: false,                         // a click beats the structure's default
        strip:              null,                          // the shared strip's handle (lib/tab-strip.js)
        lastJobName:    null,

        // Per contract § 2: IDLE / LOADING / LOADED / WATCHING / ERROR.
        machine: "IDLE",

        fileState: {
            // The file on screen, written by transition('LOADING') from
            // the path loadByPath was handed -- the dropdown's pick, a
            // Refresh, or the registry's hot swap.
            path:    null,
            // results: SpectraResults dict from /api/spectra/load.
            // Replaced atomically inside transition('APPLY').
            results: null,
            // HOW THE RUN THIS FILE BELONGS TO IS DOING -- `{state, detail,
            // live}`, the one door's answer, sent with every load
            // (web/results.md § 4.1).  The viewer follows while `live`.
            // null = the file belongs to no run.
            run: null,
        },

        viewState: {
            // 1-based index of the active mode, or null.  Survives
            // watchTick re-renders when the pick remains valid.
            selectedMode: null,
        },

        // uiPrefs: the knobs a person sets and expects to find again.
        // PERSISTED through the workspace, tag `results:spectra:ui`
        // (`workspace.md` § 4) -- see the lane below.  Trajectory leaves its
        // own bucket empty on purpose (§ 13) and is unaffected.
        uiPrefs: {
            modeFilter:     "",
            sortColumn:     "index_1based",
            sortDir:        "asc",
            broadeningFWHM: 20,
            animAmplitude:  0.15,
            animSpeed:      1.0,
            // WHICH pairing of eigenvector and amplitude (§ 12.2), and the
            // temperature the thermal one needs.  Preferences like the two above
            // them, so they belong in the bucket the others live in, which
            // persists.
            animAmplitudeMode: "display",   // or "zero-point" / "thermal"
            animTemperature:   298,         // K
        },

        lifecycle: {
            watchTimer:    null,
            watchInFlight: false,
            watchAbort:    null,
            loadAbort:     null,
            // Consecutive transient-error counter.  After
            // WATCH_MAX_ERRORS in a row, transition to ERROR.
            watchErrors:   0,
            // File-identity guard (contract § 4 Invariant 1).
            // Every fetch resolution checks (response.path,
            // its own requested path) before applying.  Late responses
            // from a prior file can never write into the current
            // file's view.
        },

        derived: {
            // Empty -- spectra has no per-iter rolling-window
            // derived state.  Kept present-but-empty so the
            // five-bucket contract shape holds; tests can pin it.
        },

        // The VibrationView handle (vibrationview.md) -- the concealed normal-mode viewer.
        // Cleared by renderResults on geometry change and by dispose() on unmount.  Lives at
        // the top level of `state` (not in any bucket) because it's a wrapper-managed
        // external resource.
        vib:            null,
        esResizeObserver:    null,   // the same, for the level diagram
        modeTab:             "table", // which of the three views is on screen
        vibMounting:    false,   // one build per mount, not one per mode click
        vibStructure:   null,    // which structure the viewer holds (§ 5.1)
        animPaused:     false,   // the USER's intent, not the viewer's state
        exporting:      null,    // the AbortController of a running export
    };

    /* ── How you were looking, kept where everything else is kept ──────────
     *
     * Contract: `spectra.md` § 7 (what a reload restores), `workspace.md` § 4
     *           (the tag) and § 5 (the only two calls), `molview.md` § 11.2b
     *           (the lane this copies -- "looking is not changing").
     * Owns:     one workspace slot, tag `results:spectra:ui`, state_index 0,
     *           holding the eight view knobs in `state.uiPrefs`.
     *
     * NOT sessionStorage.  `workspace.md` § 4 promises "there is one way to
     * save and one way to load", and the browser-storage half underneath the
     * workspace is deliberately not part of its surface.  A second store in
     * the Results tab would be a second home for tab state -- the thing the
     * tag exists to prevent.  `lib/molview/ui-context.js` already keeps a
     * viewer's camera, frame and switches in exactly this shape, so this is
     * that lane for the knobs MolView does not own.
     */
    var PREFS_TAG       = "results:spectra:ui";
    var PREFS_VERSION   = 1;
    var PREFS_DELAY_MS  = 400;      // the same coalescing ui-context uses
    var PREFS_DEFAULTS  = null;     // snapshotted from the bucket, below

    var _prefsArmed    = false;     // writes off until the restore has run
    var _prefsApplying = false;     // and off while the restore applies
    var _prefsTimer    = null;

    function _ws() {
        var w = root.molbuilder && root.molbuilder.workspace;
        if (!w || typeof w.workspaceId !== "function"
               || typeof w.persist !== "function"
               || typeof w.readState !== "function") return null;
        return w;
    }

    function _prefsIdentity(w) {
        return { workspace_id: w.workspaceId(PREFS_TAG), state_index: 0 };
    }

    function _prefsFlush() {
        _prefsTimer = null;
        if (!_prefsArmed || _prefsApplying) return;
        var w = _ws();
        if (!w) return;
        // persist() does not wait: true means sent, not saved.  A failure
        // arrives later on onPersistError, and a lost preference is not worth
        // interrupting anyone over -- the knobs still work for this session.
        try { w.persist(PREFS_TAG, { v: PREFS_VERSION, prefs: state.uiPrefs },
                        _prefsIdentity(w)); } catch (_) {}
    }

    function _prefsSchedule() {
        if (!_prefsArmed || _prefsApplying) return;
        if (_prefsTimer !== null) clearTimeout(_prefsTimer);
        _prefsTimer = setTimeout(_prefsFlush, PREFS_DELAY_MS);
    }

    /* The controls must SHOW what was restored.
     *
     * Every handler reads its value out of the DOM (`onBroadeningChange` does
     * `parseFloat(els.broadeningFwhm.value)`), so state alone is not enough:
     * the box would still read 20 while the spectrum was broadened by the
     * restored 42, and the first touch of any control would push the stale
     * default back into state AND persist it -- a feature that silently undoes
     * itself is worse than none.
     *
     * init() already primes these from the markup and re-runs
     * `onAmplitudeModeChange()` so the panel cannot contradict itself -- its
     * own comment names a restored "thermal" leaving the temperature box
     * hidden as the thing it prevents.  That priming runs BEFORE this async
     * read, so it has to happen again here, with the values that came back.
     */
    function _prefsToControls() {
        var u = state.uiPrefs;
        function put(el, v) { if (el && v !== undefined && v !== null) el.value = String(v); }
        put(els.modesFilter,      u.modeFilter);
        put(els.broadeningFwhm,   u.broadeningFWHM);
        put(els.animAmplitude,    u.animAmplitude);
        put(els.animSpeed,        u.animSpeed);
        put(els.animAmplitudeMode, u.animAmplitudeMode);
        put(els.animTemperature,  u.animTemperature);
        // The mode pairing owns more than its own value: it decides whether the
        // temperature box or the size slider is the one on screen.  Re-run the
        // handler init() runs, for the same reason init() runs it.
        if (els.animAmplitudeMode && typeof onAmplitudeModeChange === "function") {
            try { onAmplitudeModeChange(); } catch (_) {}
        }
    }

    /* Read the slot back and apply what is usable.
     *
     * A STORED VALUE IS NOT TRUSTED.  A slot outlives a rename and the file is
     * editable, so a knob is taken only when the key is one the bucket
     * declares AND its type matches the default's: a `broadeningFWHM` of
     * "abc" would otherwise reach the broadening maths and the viewer would
     * fail on a value nobody typed.  `v` guards the shape as a whole, the way
     * ui-context's VERSION does.
     */
    async function _prefsRestore() {
        var w = _ws();
        if (!w) { _prefsArmed = true; return; }
        var saved = null;
        try { saved = await w.readState(_prefsIdentity(w)); }
        catch (_) { /* readState already answers null for any failure */ }
        if (!saved || saved.v !== PREFS_VERSION
                || !saved.prefs || typeof saved.prefs !== "object") {
            _prefsArmed = true;
            return;
        }
        _prefsApplying = true;
        try {
            var got = saved.prefs, k;
            for (k in PREFS_DEFAULTS) {
                if (!Object.prototype.hasOwnProperty.call(PREFS_DEFAULTS, k)) continue;
                if (!Object.prototype.hasOwnProperty.call(got, k)) continue;
                if (typeof got[k] !== typeof PREFS_DEFAULTS[k]) continue;
                if (typeof got[k] === "number" && !isFinite(got[k])) continue;
                state.uiPrefs[k] = got[k];      // the bucket, not the alias
            }
            // Still inside the applying window: pushing values into the
            // controls fires their handlers, which assign to the same knobs,
            // and `_prefsApplying` is what stops that echoing into a write.
            _prefsToControls();
        } finally {
            _prefsApplying = false;
            _prefsArmed = true;
        }
        // The knobs are read while rendering, and the read above is async, so
        // a file already on screen was drawn with the defaults.  renderResults
        // is the door every watch tick already goes through, so re-entering it
        // is the ordinary path and not a special case.
        if (state.fileState.results) {
            try { renderResults(state.fileState.results, state.fileState.path); }
            catch (_) {}
        }
    }

    // Backward-compat aliases.  ~3000 lines of existing render +
    // event code reads/writes the legacy flat shape; the aliases
    // route through to the bucketed canonical home so the body keeps
    // working unchanged.  See trajectory/core.js for the same
    // pattern + rationale.
    (function _wireBackcompatAliases() {
        // The shared inspector helper (lib/inspectors/lifecycle.js).
        function alias(key, bucket) {
            root.molbuilder.inspectorLifecycle.alias(state, key, bucket);
        }
        alias("results",        "fileState");
        alias("selectedMode",   "viewState");
        // The uiPrefs knobs alias AND schedule a save: the alias setter is
        // the one door every write in the body already passes through, so the
        // lane needs nothing from the ~3000 lines that assign the flat names.
        function prefAlias(key) {
            Object.defineProperty(state, key, {
                get: function ()  { return state.uiPrefs[key]; },
                set: function (v) { state.uiPrefs[key] = v; _prefsSchedule(); },
                enumerable: true,
                configurable: true,
            });
        }
        prefAlias("modeFilter");
        prefAlias("sortColumn");
        prefAlias("sortDir");
        prefAlias("broadeningFWHM");
        prefAlias("animAmplitude");
        prefAlias("animSpeed");
        prefAlias("animAmplitudeMode");
        prefAlias("animTemperature");
        alias("watchTimer",     "lifecycle");
        alias("watchInFlight",  "lifecycle");
        alias("watchAbort",     "lifecycle");
        alias("loadAbort",      "lifecycle");
        alias("watchErrors",    "lifecycle");
    })();

    // The defaults ARE whatever the bucket was declared with -- snapshotted
    // rather than retyped, so a knob added to uiPrefs is persisted and
    // type-checked the day it appears and the two cannot disagree.
    PREFS_DEFAULTS = JSON.parse(JSON.stringify(state.uiPrefs));

    // Transition orchestrator (contract § 2).  Single entry-point
    // for state-machine transitions; mirrors trajectory's
    // transition() implementation.
    //
    // Targets (per contract § 2):
    //   'LOADING'  -> { path }: empty fileState, reset viewState,
    //                 abort in-flight controllers, clear timer,
    //   'LOADED'   -> {}:       stop watchTimer.  Used when
    //                 allPhasesComplete, and when following stops --
    //                 a tick's error, or WATCH_MAX_ERRORS consecutive
    //                 network failures (stopWatch).
    //   'WATCHING' -> {}:       start watchTimer.  Used after a load or
    //                 a tick when the run is still progressing.
    //   'ERROR'    -> {}:       stop watchTimer.  Used by loadByPath
    //                 alone, when a load fails (network, missing file,
    //                 wrong schema) -- there is nothing on screen to
    //                 keep.
    //   'IDLE'     -> {}:       full reset on dispose.
    //   'APPLY'    -> {path?, results?}: atomic fileState write.
    //                 Single canonical fileState writer per
    //                 contract § 2.
    function transition(target, payload) {
        payload = payload || {};
        if (target === "LOADING") {
            if (state.lifecycle.loadAbort) {
                try { state.lifecycle.loadAbort.abort(); } catch (_) {}
                state.lifecycle.loadAbort = null;
            }
            if (state.lifecycle.watchAbort) {
                try { state.lifecycle.watchAbort.abort(); } catch (_) {}
                state.lifecycle.watchAbort = null;
            }
            state.lifecycle.watchInFlight = false;
            if (state.lifecycle.watchTimer) {
                clearInterval(state.lifecycle.watchTimer);
                state.lifecycle.watchTimer = null;
            }
            // Empty fileState.  New path lands via the LOADING
            // payload; results reload comes through transition('APPLY')
            // when the fetch resolves.
            state.fileState.path    = payload.path || null;
            state.fileState.results = null;
            state.fileState.run     = null;
            // Reset viewState per matrix.  selectedMode = null
            // forces _pickDefaultMode on the next renderResults.
            state.viewState.selectedMode = null;
            // Clear transient lifecycle counters.
            state.lifecycle.watchErrors = 0;
            state.machine = "LOADING";
            return;
        }
        if (target === "IDLE") {
            if (state.lifecycle.loadAbort) {
                try { state.lifecycle.loadAbort.abort(); } catch (_) {}
                state.lifecycle.loadAbort = null;
            }
            if (state.lifecycle.watchAbort) {
                try { state.lifecycle.watchAbort.abort(); } catch (_) {}
                state.lifecycle.watchAbort = null;
            }
            state.lifecycle.watchInFlight = false;
            if (state.lifecycle.watchTimer) {
                clearInterval(state.lifecycle.watchTimer);
                state.lifecycle.watchTimer = null;
            }
            state.fileState.path    = null;
            state.fileState.results = null;
            state.viewState.selectedMode = null;
            state.lifecycle.watchErrors = 0;
            state.machine = "IDLE";
            return;
        }
        if (target === "LOADED") {
            // Run finished (allPhasesComplete true), or following
            // stopped.  Stop watchTimer if running.
            if (state.lifecycle.watchTimer) {
                clearInterval(state.lifecycle.watchTimer);
                state.lifecycle.watchTimer = null;
            }
            // ABORT THE TICK ALREADY ON THE WIRE, as LOADING and IDLE
            // both do.  Without it a tick mid-flight would resolve
            // with `signal.aborted` false and -- because LOADED keeps
            // `fileState.path` on purpose -- pass the path guard too,
            // render, and call `_settlePostLoad()`, which
            // transitions back to WATCHING and starts a NEW interval.
            //
            // Safe on the normal-completion path as well: `watchTick`
            // builds a fresh AbortController every tick, and the tick
            // that calls `_settlePostLoad` is already past its own
            // `signal.aborted` guard when this runs.
            if (state.lifecycle.watchAbort) {
                try { state.lifecycle.watchAbort.abort(); } catch (_) {}
                state.lifecycle.watchAbort = null;
            }
            state.lifecycle.watchInFlight = false;
            state.machine = "LOADED";
            return;
        }
        if (target === "WATCHING") {
            // Run is still progressing.  Start watchTimer
            // (idempotent: only sets a new interval if one isn't
            // already running).
            if (!state.lifecycle.watchTimer) {
                state.lifecycle.watchTimer =
                    setInterval(watchTick, WATCH_INTERVAL_MS);
            }
            state.machine = "WATCHING";
            return;
        }
        if (target === "ERROR") {
            if (state.lifecycle.watchTimer) {
                clearInterval(state.lifecycle.watchTimer);
                state.lifecycle.watchTimer = null;
            }
            // ABORT, as IDLE, LOADING and LOADED all do.  A no-op today:
            // both callers of transition('ERROR') live in `loadByPath`,
            // which passes through LOADING first, and LOADING aborts -- so
            // nothing is ever in flight by the time we get here.
            //
            // It is here because the door is what `watchTick`'s resolution guard
            // depends on: `signal.aborted` is the only thing that catches a
            // tick which has already SETTLED but not yet continued, and a
            // path check alone lets it through (see the note there).  ERROR
            // does not clear `fileState.path` either, so both guards would
            // pass -- the continuation would repaint the body and
            // `_settlePostLoad` would restart the very poll timer this
            // branch just stopped, wiping the error the user is reading.
            //
            // Unreachable today, one line from reachable the moment anyone
            // calls transition('ERROR') from a path that has not aborted.
            if (state.lifecycle.watchAbort) {
                try { state.lifecycle.watchAbort.abort(); } catch (_) {}
                state.lifecycle.watchAbort = null;
            }
            state.lifecycle.watchInFlight = false;
            state.machine = "ERROR";
            return;
        }
        if (target === "APPLY") {
            /* THE ONE WRITE: a name and the data that belongs to it,
             * together, or neither (`results.md` § 4: fileState is
             * "replaced atomically").
             *
             * A payload MUST say which file it is for.  A late answer
             * for a file we have moved off is dropped HERE, once, because
             * its own name no longer matches the one on screen.  That is
             * the guard: not a counter, the data's own identity. */
            if (payload.path === undefined) {
                throw new Error(
                    "APPLY without a path: results must be written with "
                    + "the file they came from, never onto the current one");
            }
            if (payload.path !== state.fileState.path) {
                return;          // an answer for a file we are not showing
            }
            state.fileState.path    = payload.path;
            // Each field is written when the payload carries it: the results
            // by `renderResults`, how the run is doing by the load and the
            // tick that fetched them.
            if (payload.results !== undefined)
                state.fileState.results = payload.results;
            if (payload.run !== undefined)
                state.fileState.run = payload.run;
            return;
        }
        // Unknown target: silent no-op.
    }

    // _settlePostLoad: after fileState has been populated by
    // transition('APPLY'), route to the state THE RUN calls for -- the one
    // door's answer the server sends with the file (`run`,
    // web/results.md § 4.1), for both callers, loadByPath and watchTick:
    // live -> transition('WATCHING') (the run is FOLLOWED,
    // web/spectra.md § 7), else -> transition('LOADED') (finished, failed,
    // never launched, or no run: nothing more will arrive).  The phase flags are the file's
    // facts, drawn as the dots; the follow is the run's.
    function _settlePostLoad() {
        const run = state.fileState.run;
        transition(run && run.live ? "WATCHING" : "LOADED");
    }

    // Poll interval for the live-watch loop.  2 s is the sweet spot:
    // long enough that the engine's atomic-replace writes don't get
    // caught mid-flight (they're sub-millisecond anyway) and short
    // enough that the UI feels live.  Not exposed as a user knob.
    const WATCH_INTERVAL_MS = 2000;

    // After this many consecutive transient errors (network down,
    // file mid-replace, etc.) the watcher gives up rather than
    // hammering the API forever.
    const WATCH_MAX_ERRORS = 5;

    /* THE PHYSICAL CONSTANTS THIS PAGE CONVERTS WITH ARE SERVED, not copied:
     * `/api/spectra/load` sends them with every result, from their one home
     * (`molbuilder.constants`; `web/blueprints/spectra.py::_page_constants`),
     * and `renderResults` takes them in before any panel draws
     * (`architecture.md` § 3; user: one source per fact). */
    const K = {};

    // ----- Listener bookkeeping ---------------------------------
    //
    // Every element-level addEventListener inside mountInspector goes
    // through _on(), which registers the teardown at the same moment as the
    // registration; dispose() hands the whole scope back in one call.
    //
    // On /results the host's innerHTML is cleared by the inspector
    // adapter after our dispose() returns, which on its own would
    // garbage-collect the listeners along with their nodes; doing the
    // explicit removal here means the dispose() contract holds even
    // when (a) a future caller forgets to clear the host, (b) a
    // listener was attached to an element OUTSIDE the host (e.g., a
    // future ``window.addEventListener`` for keyboard shortcuts), or
    // (c) the inspector grows a "remount in place" path that re-runs
    // init() and would otherwise leak the previous round of
    // listeners.  Mirrors lib/trajectory/core.js's dispose contract.
    //
    // THE SCOPE IS THE ONLY REGISTRY.
    var _listeners = root.molbuilder.inspectorLifecycle.listeners();
    function _on(target, event, handler, opts) {
        _listeners.on(target, event, handler, opts);
    }

    // ----- Status helper ----------------------------------------
    /* Through the ONE status writer (`lib/status.js`, loaded on both pages
     * that mount this module): the severity vocabulary is its, and a line
     * rewritten with the same words -- the follow's progress line, every
     * 2 s -- is left alone rather than re-announced to a screen reader.
     * A missing slot stays silent here: /spectrum-calculation has no
     * Results panel, and its absence is not a bug. */
    function setStatus(el, msg, kind) {
        if (!el) return;
        window.molbuilder.status.set(el, msg || "", kind || null);
    }

    /* A notice that REPLACES what a host holds.  The message is set
     * as text: it is an exception's or a server's words, and words are never
     * markup (ui-contract.md § 7). */
    function showNotice(host, msg, kind) {
        if (!host) return;
        const p = document.createElement("p");
        p.className = "status";
        setStatus(p, msg, kind);
        host.replaceChildren(p);
    }

    // ----- Form schema load + render ----------------------------
    //
    // The form is the CATALOGUE's vibration schema and depends on no
    // picked structure (frozen atoms are structure-side facts riding
    // the hand-over -- § 8 of web/spectra.md).

    // Monotonic counter for in-flight schema fetches.  The fetches
    // could race -- a
    // later request can finish AFTER an earlier one, and without
    // this guard the older response would overwrite the newer in
    // state.schema + the rendered form.  We snapshot the counter
    // before await and discard our continuation if a newer
    // request has been issued since.
    let _schemaFetchSeq = 0;

    async function initSchemaForm() {
        const fs = (window.molbuilder || {}).formSchema;
        if (!fs) {
            for (const host of Object.values(els.form)) {
                showNotice(host, "form-schema.js not loaded; check that "
                           + "lib/form-schema.js appears before this script "
                           + "in the template.", "error");
            }
            return;
        }
        _wireEngineStrip();
        await _reloadVibrationSchemas(fs);
        // No structure-commit subscription: the schemas do not depend on
        // the picked structure (frozen atoms are structure-side facts that
        // ride the hand-over, web/spectra.md § 8), so a sidebar pick never
        // wipes typed parameters.  What a pick DOES decide is the default
        // engine -- structureLoaded(), which the page calls.
    }

    async function _reloadVibrationSchemas(fs) {
        // BOTH engines' forms come from the CATALOGUE, narrowed to the
        // vibration kind (template.md § 6.3) -- the same door and the same
        // renderer the Structure-optimization tab uses for its two
        // sub-forms, so a parameter is defined once and rendered the same
        // on every tab.  Both stay mounted; the strip only shows one.
        const mySeq = ++_schemaFetchSeq;
        try {
            const [pyscf, siesta] = await Promise.all([
                fs.fetchSchema("pyscf",  { calculation: "vibration" }),
                fs.fetchSchema("siesta", { calculation: "vibration" }),
            ]);
            // Race guard: a rapid remount can issue a newer fetch during
            // our await; the older response must not overwrite the newer.
            if (mySeq !== _schemaFetchSeq) return;
            state.schemas.pyscf  = pyscf;
            state.schemas.siesta = siesta;
            for (const [engine, schema] of [["pyscf", pyscf], ["siesta", siesta]]) {
                const host = els.form[engine];
                if (!host) continue;
                host.innerHTML = "";
                fs.renderForm(host, schema);
            }
            wireCompatibilityListeners();
            applyCompatibility();
            // The forms exist now: whoever shows what they will carry (the
            // page's chemistry card) asks again.
            if (opts && typeof opts.onFormsReady === "function") {
                opts.onFormsReady();
            }
        } catch (exc) {
            if (mySeq !== _schemaFetchSeq) return;
            for (const host of Object.values(els.form)) {
                showNotice(host, "Could not load form schema: " + String(exc),
                           "error");
            }
        }
    }

    // ----- The engine strip -------------------------------------
    //
    // WHICH ENGINE is the description's `engine` (task.json), chosen here
    // the way the Structure-optimization tab chooses it -- a strip over two
    // mounted forms -- and carried by the hand-over.  It is not a parameter
    // of the deck (engines/vibration.md § 3.1).
    function _activeEngine() {
        return state.engine === "siesta" ? "siesta" : "pyscf";
    }

    function setEngine(name, opts) {
        const engine = name === "siesta" ? "siesta" : "pyscf";
        const o = opts || {};
        // The strip owns the buttons and the panels (lib/tab-strip.js); its
        // onChange records the choice here.  The note and the checks are
        // this tab's.
        if (state.strip) state.strip.select(engine, o);
        else { state.engine = engine; if (o.byUser) state.engineChosenByUser = true; }
        if (els.engineNote) {
            els.engineNote.textContent = o.reason || "";
            els.engineNote.hidden = !o.reason;
        }
        refreshPreflightDebounced();
    }

    function _wireEngineStrip() {
        const mb = window.molbuilder || {};
        if (!els.engineStrip || !mb.tabStrip) return;
        state.strip = mb.tabStrip.mount(els.engineStrip, {
            onChange: (name, meta) => {
                state.engine = name === "siesta" ? "siesta" : "pyscf";
                if (meta && meta.byUser) {
                    // A click beats the structure's default, and retires the
                    // note that explained the default.
                    state.engineChosenByUser = true;
                    if (els.engineNote) {
                        els.engineNote.textContent = "";
                        els.engineNote.hidden = true;
                    }
                    refreshPreflightDebounced();
                }
            },
        });
    }

    /* THE STRUCTURE DECIDES THE DEFAULT.  A structure that repeats or
     * continues along an axis is refused by PySCF's gate (an isolated-
     * molecule code) and is what SIESTA is for, so a periodic load switches
     * the strip to SIESTA and says why -- unless the person already chose.
     * The page calls this after every successful load. */
    function structureLoaded() {
        // THE MODEL'S OWN DOOR for the axis kinds (molview.md § 9.3) -- never
        // the wire envelope, whose metadata is nested where no caller should
        // know.
        const d = _viewer && _viewer.data;
        const ak = (d && typeof d.getAxisKind === "function") ? d.getAxisKind() : null;
        const periodic = Array.isArray(ak) && ak.some((k) => k && k !== "isolated");
        if (periodic && !state.engineChosenByUser && _activeEngine() !== "siesta") {
            setEngine("siesta", {
                reason: "This structure repeats or continues along an axis, "
                      + "which PySCF's gate refuses: SIESTA is the engine for "
                      + "a periodic system.  Switch back if you meant a cluster.",
            });
            return;
        }
        // An isolated load after a periodic one: the previous reason is not
        // this structure's.
        if (!periodic && els.engineNote && !els.engineNote.hidden) {
            els.engineNote.textContent = "";
            els.engineNote.hidden = true;
        }
        refreshPreflightDebounced();
    }

    /* THE FORMS THE CHEMISTRY CARD ANSWERS FOR (`lib/chemistry.js`): each
     * engine's rendered form and its schema, so the card can ask for the
     * charge and spin of exactly what each form says.  The page asks; it
     * never reaches into the containers (overview.md § 1). */
    function stateForms() {
        const out = {};
        for (const engine of ["pyscf", "siesta"]) {
            const host = els.form[engine], schema = state.schemas[engine];
            if (host && schema) out[engine] = { host: host, schema: schema };
        }
        return out;
    }

    // ----- Selector / compatibility (lock unused value fields) --
    //
    // The selector (skip/all/explicit) has three value fields: the
    // explicit list, live only under `explicit`, and the frequency
    // window's two bounds, live only under `all`.  Locking the others
    // matches the Build tab's pattern -- the user can't enter values
    // the selector would ignore.
    function _fieldIdByName(name) {
        // The catalogue owns id derivation (`_item_to_field`); this
        // module never spells an id.
        const schema = state.schemas.pyscf;
        const sections = (schema && schema.sections) || [];
        for (const sect of sections) {
            for (const f of (sect.fields || [])) {
                if (f.name === name) return f.id || "";
            }
        }
        return "";
    }

    function _esSelectionEl() {
        // The probe's selectors are the PySCF form's; SIESTA has none.
        const id = _fieldIdByName("es_mode_selection");
        return (id && els.form.pyscf) ? els.form.pyscf.querySelector("#" + id) : null;
    }

    function wireCompatibilityListeners() {
        _on(_esSelectionEl(), "change", applyCompatibility);
    }

    function applyCompatibility() {
        const sel = _esSelectionEl();
        if (!sel) return;
        const which = sel.value;
        // Map selector value -> the fields that are active for it.  Every
        // other one is disabled -- a lock on EDITING: the field keeps its
        // value and the form still sends it (`formSchema.collectForm` reads
        // no `disabled`), so a value typed before the switch reaches the
        // template, where prep says it enters nothing (validation/
        // spectra.py).  The window filters `all` alone -- `skip` selects
        // nothing and `explicit` names its modes -- so outside `all` it
        // enters nothing and is locked like the list is outside `explicit`
        // (web/spectra.md § 9a.1; user, 2026-09-28).
        const activeByMode = {
            "skip":      [],
            "all":       ["freq_min_cm1", "freq_max_cm1"],
            "explicit":  ["es_explicit_indices"],
        };
        const valueFields = ["es_explicit_indices", "freq_min_cm1",
                             "freq_max_cm1"];
        const active = activeByMode[which] || [];
        for (const name of valueFields) {
            const id = _fieldIdByName(name);
            const f = (id && els.form.pyscf)
                ? els.form.pyscf.querySelector("#" + id) : null;
            if (!f) continue;
            const isActive = active.includes(name);
            f.disabled = !isActive;
            // Visually fade the field set so it's obvious which one
            // is in play -- the disabled attr does some of this, but
            // a class lets us style the wrapping <label> too.
            const wrap = f.closest("label, .field");
            if (wrap) wrap.classList.toggle("is-locked", !isActive);
        }
    }

    // ----- Helpers: gather form values + xyz -------------------
    function collectParams(engine) {
        const fs = (window.molbuilder || {}).formSchema;
        const e = engine || _activeEngine();
        const host = els.form[e], schema = state.schemas[e];
        if (!fs || !host || !schema) return {};
        return fs.collectForm(host, schema);
    }

    /* THE VIEWER THIS PAGE MOUNTED, handed to us by the page that mounted it
     * (spectra/viewer.js). We do not look one up: there is nothing to look up in,
     * and a viewer belongs to whoever mounted it (molview.md § 5.6). */
    let _viewer = null;
    function useViewer(handle) { _viewer = (handle && handle.ok) ? handle : null; }

    function getTheStructure() {
        return _viewer ? _viewer.data.getStructure() : null;
    }

    /* THE STRUCTURE THIS TAB WOULD HAND OVER, as the envelope the server's
     * doors read -- ONE READ OF THE VIEWER (molview.md § 9.3): `exportFile()`
     * is the viewer's own producer, and it carries atoms, positions at the
     * displayed frame, labels, regions (frozen atoms), the cell and its axis
     * kinds, and the record of the run it came from.  The preflight, the
     * hand-over and the chemistry card all read it here, so the three cannot
     * be about different structures.  Null with nothing loaded. */
    function structureForRequest() {
        const out = _viewer ? _viewer.data.exportFile() : null;
        return (out && out.structure) ? out.structure : null;
    }

    // ----- Live preflight: gate ① for the vibration kind ---------
    //
    // The SAME verdict prep's settings gate gives later, surfaced
    // while the person is still at the form (Build's pattern,
    // structure-optimization/viewer.js::refreshPreflight): debounced
    // POST /api/build/preflight with calculation="vibration"; the
    // shared findings renderer places each finding on its
    // workflow-group card and the rest on the summary panel below
    // the form.  No structure loaded -> no call: warnings that
    // haven't been earned yet don't show.
    function _debouncePreflight(fn, wait) {
        let t = null;
        return function () {
            if (t) clearTimeout(t);
            t = setTimeout(fn, wait);
        };
    }

    async function refreshPreflight() {
        const _structure = structureForRequest();
        if (!_structure) return;
        const engine = _activeEngine();
        let params;
        try { params = collectParams(engine); }
        catch (_) { return; }   // the field's own caption says why
        try {
            const r = await fetch("/api/build/preflight", {
                method:  "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    structure:   _structure,
                    engine:      engine,
                    calculation: "vibration",
                    params:      params,
                }),
            }).then(x => x.json());
            if (Array.isArray(r.issues)) {
                live[engine] = r.issues;
                showFindings(engine, r.issues);
            }
        } catch (e) {
            // Network hiccup: the panel keeps its previous state; the
            // settings gate at prep is the canonical refusal anyway.
        }
    }
    const refreshPreflightDebounced = _debouncePreflight(refreshPreflight, 250);

    // What the live preflight said last, per engine -- the findings a Send
    // adds its own to (`handover-procedure.md` § 2.1).
    const live = { siesta: [], pyscf: [] };

    /* ONE PLACE this tab shows an engine's findings -- the live
     * preflight's and the hand-over's (the cell gate's notices) --
     * through the one renderer (lib/validation-findings.js); `reveal`
     * brings a Send's first error into view. */
    function showFindings(engine, issues, reveal) {
        const vf = (window.molbuilder || {}).validationFindings;
        if (!vf) return;
        /* fieldIds lands each finding beside its own control -- the same
         * map the optimization tab passes; without it a finding falls to
         * card-then-residual.  emptyText keeps the panel's no-findings copy alive:
         * the template's static row is destroyed by the first render. */
        const ids = {};
        const schema = state.schemas[engine];
        const sects = (schema && schema.sections) || [];
        for (const s of sects) {
            for (const f of (s.fields || [])) ids[f.name] = f.id;
        }
        // Both forms stay mounted (one is shown), so the OTHER form's
        // field rows are cleared before this engine's are drawn -- a scope
        // of one form would leave the hidden form's stale.
        for (const e of ["pyscf", "siesta"]) {
            if (e !== engine && els.form[e] && typeof vf.clear === "function") {
                vf.clear({ formScope: els.form[e] });
            }
        }
        vf.render(issues, { panel: els.preflightPanel,
                            formScope: els.form[engine],
                            fieldIds: ids,
                            reveal: !!reveal,
                            emptyText: "No findings yet — checks "
                                + "run live as you edit." });
    }

    // ----- Send to Task setup (the hand-over) -------------------
    //
    // This tab
    // DESCRIBES a vibration calculation and hands the description to
    // Task setup -- it renders no deck.  Guards, write order and
    // notice handling live in lib/task-handover.js, ONE door shared
    // with /structure-optimization; this tab contributes only what it
    // alone knows: its structure door, the engine, the kind, its form.
    async function sendToTaskSetup() {
        const mb = window.molbuilder || {};
        const say = (kind, msg) => setStatus(els.sendStatus, msg, kind);
        if (!mb.taskHandover) {
            say("error", "lib/task-handover.js is not loaded.");
            return;
        }
        /* The structure through the tab's one read of the viewer
         * (`structureForRequest`).  Frozen atoms REACH THE CALCULATION THIS
         * WAY: they ride the structure's own files, not a form field
         * (plan § 2). */
        const _structure = structureForRequest();
        /* The engine is THE STRIP'S CHOICE -- the description's engine, the
         * same fact the Structure-optimization tab sends from its own strip
         * -- and the params are that engine's form. */
        const engine = _activeEngine();
        // A VALUE THAT WILL NOT READ AS ITS TYPE is refused here, naming its
        // field -- beside which its caption already says why.
        let params;
        try { params = collectParams(engine); }
        catch (e) {
            say("error", e.message);
            return;
        }
        await mb.taskHandover.send({
            projects:    mb.projects,
            say:         say,
            // The hand-over's findings in this engine's panel, beside the
            // live preflight's, which are still true of the form it sends.
            standing:     live[engine],
            // ...and a refusal's first error brought into view.
            showFindings: (issues) => showFindings(engine, issues, true),
            structure:   _structure,
            engine:      engine,
            params:      params,
            calculation: "vibration",
        });
    }

    // ----- Load by server-side path ----------------------------
    //
    // /api/spectra/load with {path: "<server-side path>"} so the server
    // reads the file directly -- no re-upload after every phase write.
    /** THE one route to `/api/spectra/load` (`results.md` § 4):
     *  `loadByPath` and `watchTick` both come through here.
     *
     *  One door, and the answer comes back WITH the name it was asked for,
     *  so no caller is in a position to write it under another.
     */
    // NOTE: the returned `path` is the REQUESTED one, echoed straight back
    // from the argument.  `/api/spectra/load` does not send a path, and it
    // must not be made to: the route resolves through
    // `_resolve_within_roots` (~ and $VARS expanded, symlinks followed), so
    // a server-echoed path would not equal the string the caller holds and
    // APPLY would drop every payload.  One string feeds both sides.
    async function fetchResults(path, signal) {
        const r = await fetch("/api/spectra/load", {
            method:  "POST",
            headers: { "Content-Type": "application/json" },
            body:    JSON.stringify({ path: path }),
            signal:  signal,
        });
        return { path: path, body: await r.json() };
    }

    /** Load the file ``path`` names -- the dropdown's pick, a Refresh, a
     *  return to the tab, or the registry's hot swap -- and FOLLOW it while
     *  its run is live (web/spectra.md § 7): `_settlePostLoad` starts the
     *  poll when the run the server says the file belongs to is queued or
     *  running, and `watchTick` stops it when it no longer is. */
    async function loadByPath(path) {
        path = String(path || "").trim();
        if (!path) return;
        setStatus(els.watchStatus, "Loading " + path + "…", "muted");
        // Contract § 2: file-switch / Load -> transition('LOADING').
        // Aborts loadAbort + watchAbort, clears watchInFlight, stops
        // the watchTimer if running, empties fileState (sets path),
        // and resets viewState -- the same reset for a new file and
        // for a Refresh of this one (contract § 5: no partial resets).
        transition("LOADING", { path: path });
        state.lifecycle.loadAbort = new AbortController();
        const signal = state.lifecycle.loadAbort.signal;
        let body;
        try {
            // The JSON parse is inside `fetchResults`, which is called
            // inside this try, so a malformed / truncated body lands on
            // the same status-banner path as a network error.
            body = (await fetchResults(path, signal)).body;
        } catch (exc) {
            // AbortError: a newer loadByPath() superseded us, or
            // dispose() ran.  Silent -- the newer action owns the
            // status banner now.
            if (exc.name === "AbortError") return;
            setStatus(els.watchStatus,
                      "Network error: " + exc.message, "error");
            transition("ERROR");
            _announceReady({ error: exc.message });
            return;
        }
        // Contract § 4 Invariant 1, asked of the ANSWER'S OWN IDENTITY.
        // Two questions:
        //   * is this still the file on screen?  A newer LOADING moved
        //     `fileState.path`, and dispose set it to null;
        //   * was this request superseded?  `signal.aborted` says so, and
        //     it is the only thing that can tell two loads of the SAME
        //     file apart, which a path comparison cannot.
        if (signal.aborted) return;
        if (path !== state.fileState.path) return;
        if (!body.ok) {
            let msg = body.error || "Load failed.";
            if (body.kind === "schema_mismatch") {
                msg = "Schema version mismatch (expected "
                    + body.expected_version + ", got "
                    + body.actual_version + "). "
                    + "Update molbuilder or use a matching script version.";
            } else if (body.kind === "not_found") {
                // The dropdown lists files that exist, so this one went
                // between the listing and the read.
                msg = "File not found at " + path
                    + " -- it was removed after the folder was listed.";
            }
            setStatus(els.watchStatus, msg, "error");
            transition("ERROR");
            _announceReady({ error: msg });
            return;
        }
        // How the run is doing, from the answer, before anything is drawn.
        transition("APPLY", { path: path,
                              run: body.run !== undefined ? body.run : null });
        renderResults(body.results, path);
        updatePhaseIndicator(body.results);
        // WATCHING while the run is live, else LOADED: a run still going is
        // followed from here (web/spectra.md § 7).
        _settlePostLoad();
        _showRunStatus(body.results);
        _announceReady({});
    }

    /* THE LOAD HAS ENDED -- drawn, or refused with its reason on the status
     * line (the shared inspector door, lib/inspectors/lifecycle.js). */
    function _announceReady(detail) {
        window.molbuilder.inspectorLifecycle.announceReady("spectra", detail);
    }

    /** The status line for a result: finished, or where the run is now --
     *  and "following" only while the poll actually runs, read off the
     *  machine the settle just moved, never re-derived beside it.  By ROLE,
     *  like the rest of the viewer: it names the phases the file's own
     *  flags carry, never a switch only one engine has. */
    function _showRunStatus(results) {
        const run = state.fileState.run;
        if (state.machine === "WATCHING") {
            setStatus(els.watchStatus, _watchProgressLine(results)
                      + " — following, every "
                      + (WATCH_INTERVAL_MS / 1000) + " s.", "muted");
        } else if (run && run.state === "unreadable") {
            // A record that does not read, in its reader's words.
            setStatus(els.watchStatus, "Run record unreadable — "
                      + run.detail, "error");
        } else if (run && run.state === "failed") {
            // THE RUN STOPPED, in its own words -- between phases, the dot
            // for the one it was in still reads running.
            setStatus(els.watchStatus, "Run stopped — " + run.detail + ". "
                      + _watchProgressLine(results) + ".", "error");
        } else if (allPhasesComplete(results)) {
            const n = (results.modes || []).length;
            setStatus(els.watchStatus, "Run complete ✓ — " + n + " mode"
                      + (n === 1 ? "" : "s") + ".", "ok");
        } else {
            setStatus(els.watchStatus,
                      _watchProgressLine(results) + ".", "muted");
        }
    }

    // ----- Live-watch poller (spec § 6.1) -----------------------
    //
    // Polls /api/spectra/load { path: <...> } every WATCH_INTERVAL_MS
    // while a job is running -- started by `_settlePostLoad` when a load
    // finds a phase still to finish.  The engine writes <job>.spectra.json
    // atomically at each phase boundary, so each poll gets a parsed
    // SpectraResults and re-renders the UI with whatever phases are
    // populated so far.
    //
    // Stops by itself when the run is no longer live (the server's `run`,
    // web/results.md § 4.1), when the file goes away, or after
    // WATCH_MAX_ERRORS consecutive transient failures -- and with the
    // inspector, when another file is picked or it is disposed.
    function stopWatch(reason) {
        // Contract § 2: stop following -> LOADED (the file is no longer
        // being polled but the loaded snapshot remains visible, and
        // fileState.path is kept so the status names what it was).
        transition("LOADED");
        if (reason) setStatus(els.watchStatus, reason, "muted");
    }

    async function watchTick() {
        if (!state.fileState.path) return;
        // In-flight guard: skip overlap ticks entirely; dispose()
        // aborts via watchAbort.
        if (state.lifecycle.watchInFlight) return;
        let body;
        state.lifecycle.watchAbort = new AbortController();
        const signal = state.lifecycle.watchAbort.signal;
        state.lifecycle.watchInFlight = true;
        // THE FILE THIS TICK IS FOR, read once and carried.  Reading
        // `state.fileState.path` again after the await would be reading
        // the file the user has since moved to, which is how an answer
        // ends up painted under the wrong name -- and it is what every
        // guard below compares against.
        const myPath = state.fileState.path;
        try {
            body = (await fetchResults(myPath, signal)).body;
        } catch (exc) {
            if (exc.name === "AbortError") return;
            if (myPath !== state.fileState.path) return;
            state.lifecycle.watchErrors++;
            setStatus(els.watchStatus,
                      "Network error (" + state.lifecycle.watchErrors + "/"
                      + WATCH_MAX_ERRORS + "): " + exc.message, "error");
            if (state.lifecycle.watchErrors >= WATCH_MAX_ERRORS) {
                stopWatch("Stopped after " + WATCH_MAX_ERRORS
                          + " consecutive network errors.");
            }
            return;
        } finally {
            state.lifecycle.watchInFlight = false;
        }
        // TWO guards at resolution, the same pair `loadByPath` uses.
        //
        // `watchInFlight` guards CONCURRENCY -- it stops a second tick
        // starting while one is out.  It cannot say anything about a tick
        // that has already SETTLED:
        //
        //   watching A -> tick resolves, continuation queued
        //   -> Refresh fires for A (a reload of the same file)
        //      -> transition('LOADING') aborts us, stops the timer,
        //         empties results
        //   -> our continuation runs.  myPath === fileState.path, both
        //      "A", so a path check alone lets it through: it repaints
        //      the pre-Refresh body and _settlePostLoad restarts the
        //      poll timer the Refresh had just stopped.
        //
        // `signal.aborted` is what closes that, because the transition
        // aborted this request.  The path answers the OTHER question -- a
        // different file, or dispose, which sets it to null.  Both are
        // needed.
        if (signal.aborted) return;
        if (myPath !== state.fileState.path) return;
        if (!body.ok) {
            // The engine replaces the file atomically, so a poll never
            // finds it half-written or briefly missing: an error here --
            // the file gone included -- ends the follow, and says why.
            stopWatch("Stopped following: " + (body.error || "load failed"));
            return;
        }
        state.lifecycle.watchErrors = 0;
        // How the run is doing, then whatever phases are populated so far.
        transition("APPLY", { path: myPath,
                              run: body.run !== undefined ? body.run : null });
        renderResults(body.results, myPath);
        updatePhaseIndicator(body.results);
        // The same settle as a load: the run no longer live -> LOADED (the
        // timer stops), else WATCHING (keeps polling).
        _settlePostLoad();
        _showRunStatus(body.results);
    }

    function _watchProgressLine(results) {
        // One-line summary of where the run is RIGHT NOW so the
        // user knows what to expect.  Relaxation runs FIRST (it is
        // the in-deck precondition, plan D3): frequencies of an
        // unrelaxed structure would be wrong, so it gates the rest.
        const rx = results.relaxation || {};
        if (rx.enabled && results.phase_relaxation !== "complete") {
            const n = (rx.n_steps != null)
                ? " (step " + rx.n_steps + ")" : "";
            return "Relaxing the geometry" + n;
        }
        const f = results.phase_frequencies;
        const r = results.phase_raman;
        const e = results.phase_es;
        if (f !== "complete") return "Computing vibrational frequencies (Hessian)";
        // `not requested` is terminal (vibration.md § 4.9): nothing is
        // computing for a phase the description never asked for.
        const done = (v) => v === "complete" || v === "not requested";
        if (results.config && results.config.compute_raman && !done(r))
            return "Computing Raman activities (polarizability derivatives)";
        // A file from before the flag carries "" -- no record, not a phase
        // still to come (vibration.md § 4.9).
        const ir = results.phase_ir;
        if (results.config && results.config.compute_ir && ir && !done(ir))
            return "Computing infrared intensities (dipole derivatives)";
        const sel = results.config && results.config.es_mode_selection;
        if (sel && sel !== "skip" && !done(e)) {
            const haveES = (results.modes || [])
                .filter(m => m.electronic_structure).length;
            const planned = (results.selected_mode_idxs_1based || []).length;
            const planTxt = planned ? (" (" + haveES + " of " + planned + " modes done)")
                                    : "";
            return "Computing per-mode orbital energies (displaced SCFs)" + planTxt;
        }
        return "Still running";
    }

    function allPhasesComplete(results) {
        // A run is "complete" when every phase the CONFIG asked for
        // is complete.  L2 (frequencies) is always required.
        // Relaxation counts only when the deck ran it (v4 files and
        // `already_relaxed` runs carry enabled: false / no block).
        const rx = results.relaxation || {};
        if (rx.enabled && results.phase_relaxation !== "complete")
            return false;
        if (results.phase_frequencies !== "complete") return false;
        const cfg = results.config || {};
        const done = (v) => v === "complete" || v === "not requested";
        if (cfg.compute_raman && !done(results.phase_raman)) return false;
        // "" is a file written before the infrared flag: no record, and
        // no reason to wait (vibration.md § 4.9).
        if (cfg.compute_ir && results.phase_ir && !done(results.phase_ir))
            return false;
        if (cfg.es_mode_selection && cfg.es_mode_selection !== "skip"
            && !done(results.phase_es)) return false;
        return true;
    }

    function updatePhaseIndicator(results) {
        if (!els.phaseIndicator) return;
        els.phaseIndicator.hidden = false;
        const dots = els.phaseIndicator.querySelectorAll(".phase-dot");
        dots.forEach(dot => {
            const ph = dot.dataset.phase;   // relaxation|frequencies|raman|ir|es
            // A phase the file's route does not have shows no dot at all
            // (web/spectra.md § 3): SIESTA's has no Raman, no infrared and
            // no probe.  Infrared's flag is "" in a file written before it
            // existed -- no record, so no dot rather than one left empty.
            const has = ph === "raman" ? _routeHas(results, "compute_raman")
                      : ph === "ir" ? (_routeHas(results, "compute_ir")
                                       && results.phase_ir !== "")
                      : ph === "es" ? _routeHas(results, "es_mode_selection")
                      : true;
            const wrap = dot.closest(".phase");
            if (wrap) wrap.hidden = !has;
            const v  = results["phase_" + ph] || "empty";
            // the state is a phrase (`not requested`); the class is one token
            dot.className = "phase-dot phase-" + String(v).replace(/\s+/g, "-");
            dot.title = ph + ": " + v;
        });
    }

    /** Render an answer, TOGETHER WITH THE FILE IT CAME FROM.
     *
     *  `path` is not decoration and is not optional: it is the difference
     *  between "these are the results" and "these are the results FOR THIS
     *  FILE".  Every write below carries it, so a late answer cannot be
     *  painted under a name it does not belong to (`results.md` § 4).
     */
    // Copy the Methods paragraph to the clipboard.  The clipboard API
    // is unavailable on an insecure origin and can be refused by policy,
    // so the failure path selects the text instead of leaving the user
    // with a button that silently does nothing.
    /* The floor is a VIEW control: it redraws and touches nothing else,
     * which is why it can be dragged live.  The number is echoed beside
     * the slider because a bare range input tells the user where the
     * handle is, never what it means. */
    function onDisplayFloor() {
        const pct = Number(els.displayFloor && els.displayFloor.value);
        if (!Number.isFinite(pct)) return;
        if (els.displayFloorOut) {
            els.displayFloorOut.textContent =
                (Math.round(pct * 10) / 10) + "%";
        }
        _withChart((h) => h.setDisplayFloor(pct));
    }

    function copyMethodsText() {
        const text = (els.methodsText && els.methodsText.textContent) || "";
        if (!text) return;
        const note = (msg, cls) => {
            if (!els.methodsNote) return;
            els.methodsNote.textContent = msg;
            els.methodsNote.className = "status " + cls;
            window.setTimeout(() => {
                if (els.methodsNote) els.methodsNote.textContent = "";
            }, 4000);
        };
        if (navigator.clipboard && window.isSecureContext) {
            navigator.clipboard.writeText(text)
                .then(() => note("copied", "ok"))
                .catch(() => { selectMethodsText(); note("select and copy", "warn"); });
        } else {
            selectMethodsText();
            note("select and copy (clipboard needs a secure origin)", "warn");
        }
    }

    function selectMethodsText() {
        if (!els.methodsText || !window.getSelection) return;
        const range = document.createRange();
        range.selectNodeContents(els.methodsText);
        const sel = window.getSelection();
        sel.removeAllRanges();
        sel.addRange(range);
    }

    /* WHAT THIS FILE'S ROUTE CAN COMPUTE, read BY ROLE (web/spectra.md
     * § 9b.3; engines/vibration.md § 3.1): a channel the route has a switch
     * for sits in the file's own `config`, and a route with no switch for it
     * -- SIESTA for the strengths and the per-mode probe -- computed nothing
     * there and never could.  Asked of the file, never of the engine's name,
     * so the next engine is answered by its own file. */
    function _routeHas(r, item) {
        return !!(r && r.config && (item in r.config));
    }

    /* A SPECTRUM ONLY WHERE A STRENGTH WAS COMPUTED (web/spectra.md § 2):
     * with none, a height would mean nothing, so none is drawn.  The
     * chart's own rule, read off the same fields and the same flag this
     * viewer hands it -- a strength on a REAL mode (spectrumchart.md
     * § 6.2, § 6.4) -- so the heading and the controls always name the
     * picture the chart draws.  Counting an imaginary mode's strength
     * would title a positions picture "Spectrum". */
    function _anyStrength(r) {
        return ((r && r.modes) || []).some(m => !m.has_imag
            && (Number.isFinite(m.raman_activity_a4_amu)
                || Number.isFinite(m.ir_intensity_km_mol)));
    }

    /* The one sentence under the mode positions, saying why there are no
     * heights -- which of the four cases it is (web/spectra.md § 2): the
     * route computes no strengths; the run asked for none; they are still
     * being computed; the run asked and recorded none. */
    function _noSpectrumSentence(r) {
        if (!_routeHas(r, "compute_raman") && !_routeHas(r, "compute_ir")) {
            return "Each line marks where a mode is -- this route computes "
                 + "the frequencies and the mode shapes, not infrared or "
                 + "Raman intensities, so there are no heights.  Click a line "
                 + "to pick its mode.";
        }
        const cfg = r.config || {};
        if (!cfg.compute_raman && !cfg.compute_ir) {
            return "Each line marks where a mode is -- infrared and Raman "
                 + "intensities were not requested in this run, so there are "
                 + "no heights.";
        }
        const running = (v) => v === "empty" || v === "running";
        if (running(r.phase_frequencies) || running(r.phase_raman)
            || running(r.phase_ir)) {
            return "The intensities are still being computed; until the "
                 + "first ones land each line marks where a mode is.";
        }
        return "Each line marks where a mode is -- the run asked for "
             + "intensities and recorded none, so there are no heights; its "
             + "log says why.";
    }

    function renderResults(results, path) {
        if (results && results.constants) Object.assign(K, results.constants);
        if (!results) {
            els.resultsSummary.hidden = true;
            // Contract § 2: route fileState writes through
            // transition('APPLY').  selectedMode is viewState
            // (event-mutable per matrix § 3) so direct write is
            // allowed there.
            transition("APPLY", { path: path, results: null });
            state.selectedMode = null;
            return;
        }
        // Live-watch same-content guard.
        // ``watchTick`` polls /api/spectra/load every WATCH_INTERVAL_MS
        // and most ticks return identical results (Hessian phase still
        // running, ES phase still cooking).  Without this gate the
        // viewer-dispose block below tears down + rebuilds the VibrationView
        // mode viewer every 2s, which resets the user's camera angle and
        // pauses the vibration animation right when they're trying to
        // study a mode.  Fingerprint on the fields that drive what's
        // rendered: atom count, mode count + per-mode ES presence,
        // phase markers, and currently-selected mode.  Same fingerprint
        // = nothing to redraw, bail.
        const prev = state.results;
        const newFp = _resultsFingerprint(results, state.selectedMode);
        const prevFp = prev ? _resultsFingerprint(prev, state.selectedMode) : null;
        if (prevFp !== null && prevFp === newFp) {
            // Keep state.results pointing at the freshest object so any
            // downstream reads see the latest references (runtime_info
            // etc. can update even when the fingerprint is stable).
            // Contract § 2: route fileState writes through
            // transition('APPLY').
            transition("APPLY", { path: path, results: results });
            return;
        }
        transition("APPLY", { path: path, results: results });
        els.resultsSummary.hidden = false;

        // Top-of-summary meta dictionary.  ``runtime_info`` (added in
        // results-schema v4, 2026-05-22) carries the actual CPU/GPU
        // resources the run consumed -- so a user who saw load=40 on
        // a 20-core host can confirm here "yes, the script used 20
        // PySCF threads with BLAS=1, no oversubscription."  Missing
        // keys render as "—" so older results (without runtime_info)
        // degrade cleanly.
        const rt   = results.runtime_info || {};
        const cpu  = (rt.n_threads_pyscf != null)
            ? `${rt.n_threads_pyscf} PySCF (BLAS=${rt.n_threads_blas ?? "?"}, `
              + `physical=${rt.physical_cores ?? "?"}, logical=${rt.logical_cores ?? "?"})`
            : "—";
        const gpu  = (rt.gpu_used === true)
            ? `ON — ${rt.gpu_name || "?"}`
              + (rt.gpu_compute_capability ? ` (CC ${rt.gpu_compute_capability})` : "")
              + (rt.cuda_version            ? ` · CUDA ${rt.cuda_version}` : "")
            : (rt.gpu_requested === true
                ? `OFF — ${rt.gpu_name || "GPU requested but fell back to CPU"}`
                : (rt.gpu_used === false ? "OFF" : "—"));
        // PRESENT WHAT THE RUN ACTUALLY DID, never what the config asked
        // for.  The two dmu/dR routes cost wildly different amounts --
        // analytic rides on the Hessian's own CPHF solve, the sweep
        // spends 6N extra SCFs -- so a reader comparing two runs needs
        // to see which one they got.  An older sidecar predates the
        // field: that is an absence of record, reported as such rather
        // than guessed either way.
        /* R7's second half: what the harmonic analysis took out before
         * diagonalising, and over which atoms the Hessian ran, said beside
         * the result and not only in the file. */
        function _removedMotionsLabel(r) {
            const rm = r.removed_motions || {};
            if (rm.count == null) return "not recorded";
            const nFree = (r.free_atom_idxs || []).length;
            const modes = (r.modes || []).length;
            return rm.count + " (" + (3 * nFree) + " coordinates of the free atoms \u2192 "
                 + modes + (modes === 1 ? " vibration)" : " vibrations)");
        }
        function _hessianScopeLabel(r) {
            if (!r.hessian_scope) return "not recorded";
            const n = r.n_atoms_in_hessian, N = r.n_atoms_total;
            return r.hessian_scope === "free"
                ? "the free atoms only (" + n + " of " + N + ")"
                : "every atom (" + N + ")";
        }
        /* The Raman line by ROUTE, like the infrared line: a phase flag says
         * a step finished, not whether it computed anything, and a SIESTA
         * file's phase_raman is 'complete' with nothing behind it. */
        function _ramanRouteLabel(r) {
            // By ROLE, not by engine name: a file whose config never asked
            // (SIESTA carries no `compute_raman`) and whose route is `none`
            // computed nothing on that route; a PySCF file that asked and
            // was answered `none` was not requested.
            const asked = r.config && ("compute_raman" in r.config);
            if (!asked && r.raman_route === "none") return "not computed on this route";
            if (!r.config || !r.config.compute_raman) return "not requested";
            switch (r.raman_route) {
                case "finite-difference":
                    return "computed \u2014 finite-difference polarizabilities"
                         + (r.raman_fd_step_ang != null
                            ? " (\u00b1" + r.raman_fd_step_ang + " \u00c5)" : "");
                case "none":
                    return "not computed";
                case "":
                case undefined:
                    return r.phase_raman === "complete"
                         ? "computed \u2014 route not recorded" : r.phase_raman;
                default:
                    return "computed \u2014 " + r.raman_route;
            }
        }
        function _irRouteLabel(r) {
            const asked = r.config && ("compute_ir" in r.config);
            if (!asked && r.ir_route === "none") return "not computed on this route";
            if (!r.config || !r.config.compute_ir) return "not requested";
            switch (r.ir_route) {
                case "analytic":
                    return "computed — analytic \u2202\u03bc/\u2202R "
                         + "(no extra SCFs)";
                case "finite-difference":
                    return "computed — finite-difference dipoles";
                case "none":
                case "":
                case undefined:
                    return "computed — route not recorded";
                default:
                    return "computed — " + r.ir_route;
            }
        }
        /* The per-mode orbital energies by the same rule as the two
         * strengths: a route with no switch for the probe computed nothing
         * there, and names nothing it could not have asked for. */
        function _esRouteLabel(r) {
            if (!_routeHas(r, "es_mode_selection")) {
                return "not computed on this route (its electronic response "
                     + "along a mode, the projected DOS, is planned)";
            }
            if (r.config.es_mode_selection === "skip") return "not requested";
            return r.phase_es;
        }
        const meta = [
            ["Engine",            results.engine + " " + (results.engine_version || "?")],
            ["Atoms (total)",     results.n_atoms_total],
            ["Free / frozen",     (results.free_atom_idxs || []).length
                                    + " / "
                                    + (results.frozen_atom_idxs || []).length],
            // ABSENT IS NOT ZERO (engines/vibration.md § 6.5): a SIESTA
            // file carries no reference energy, and Number(null) is 0.
            ["Equilibrium E (Eh)", (results.equilibrium
                                    && results.equilibrium.scf_energy_eh != null)
                                    ? Number(results.equilibrium.scf_energy_eh).toFixed(8)
                                    : "\u2014"],
            ["Hessian over",       _hessianScopeLabel(results)],
            ["Whole-body motions removed", _removedMotionsLabel(results)],
            ["CPU / threads",      cpu],
            ["GPU",                gpu],
            ["Host",               rt.hostname || "—"],
            ["Relaxation",                _relaxSummary(results)],
            ["Frequencies (Hessian)",     results.phase_frequencies],
            ["IR intensities",            _irRouteLabel(results)],
            ["Raman activities",           _ramanRouteLabel(results)],
            ["Per-mode orbital energies",  _esRouteLabel(results)],
        ];
        // The Methods paragraph is composed during the run and grows as
        // phases complete, so it is rendered from whatever is present
        // rather than gated on the run being finished.  Empty means the
        // composer has not written anything yet -- show nothing, never
        // an empty box that reads like a missing result.
        if (els.methodsBlock) {
            const md = (results.methods_text || "").trim();
            els.methodsBlock.hidden = !md;
            if (md && els.methodsText) els.methodsText.textContent = md;
        }

        // Built, not written: the values are the results file's own strings
        // (engine, version, host), set as text.
        els.resultsMeta.replaceChildren(...meta.flatMap(([k, v]) => {
            const dt = document.createElement("dt");
            dt.textContent = String(k);
            const dd = document.createElement("dd");
            dd.textContent = String(v);
            return [dt, dd];
        }));

        // Table columns follow the file's ROUTE, never its data
        // (web/spectra.md § 9b.3): a column the route can compute stays
        // up with its cells empty until they land, so a run whose
        // per-mode orbitals are still cooking does not lose and regain
        // its headers; a column the route has no switch for
        // -- SIESTA's -- is not shown at all (the class toggles below).
        const anyES = (results.modes || []).some(m => !!m.electronic_structure);

        // Auto-select the highest-Raman-activity real mode so the
        // ES panel comes up populated (if any mode has ES).  If no
        // mode has ES, fall back to the lowest-index real mode.
        //
        // The user's existing selection is preserved if it is still
        // valid: a live-watch tick must not reset a pick to the
        // auto-default.  Only auto-pick when the prior selection is
        // null OR no longer exists in the current modes list.
        if (results.modes && results.modes.length) {
            const prior = state.selectedMode;
            const priorStillValid = (prior != null) && results.modes.some(
                m => m.index_1based === prior);
            if (!priorStillValid) {
                state.selectedMode = _pickDefaultMode(results.modes, anyES);
            }
        } else {
            state.selectedMode = null;
        }

        // A SPECTRUM WHERE A STRENGTH WAS COMPUTED, THE MODE POSITIONS WHERE
        // NONE WAS -- one line at each mode, picked like a stick, with no
        // width or floor control since there is no height for either to act
        // on -- and the columns only for what the route computes
        // (web/spectra.md § 2, § 9b.3; spectrumchart.md § 6.2).
        const drawn = _anyStrength(results);
        const haveModes = (results.modes || []).length > 0;
        if (els.spectrumSection) els.spectrumSection.hidden = !haveModes;
        if (els.spectrumHeading) {
            els.spectrumHeading.textContent = drawn ? "Spectrum" : "Mode positions";
        }
        if (els.spectrumControls) els.spectrumControls.hidden = !drawn;
        if (els.spectrumAbsent) {
            els.spectrumAbsent.hidden = drawn;
            els.spectrumAbsent.textContent = drawn ? ""
                : _noSpectrumSentence(results);
        }
        if (els.modesTable) {
            els.modesTable.classList.toggle("route-no-raman",
                                            !_routeHas(results, "compute_raman"));
            els.modesTable.classList.toggle("route-no-ir",
                                            !_routeHas(results, "compute_ir"));
            els.modesTable.classList.toggle("route-no-es",
                                            !_routeHas(results, "es_mode_selection"));
        }
        if (haveModes) renderSpectrumChart(results.modes || []);
        renderModesTable();
        renderESPanel();
        renderThermoPanel(results);
        // Geometry changed (new results loaded) -- discard the old
        // VibrationView mode viewer so the next render rebuilds with the
        // fresh structure.
        if (state.vib) {
            _stopAnimation();
            // Dispose the viewer so the next render builds one against the fresh
            // structure.  It draws no controls of its own to tear down -- only a
            // canvas, a caption and a clock (vibrationview.md § 5.4, § 8).
            try { state.vib.dispose(); }
            catch (_) {}
            state.vib = null;
            if (els.modeViewer) els.modeViewer.innerHTML = "";
        }
        renderModeViewer();
    }

    function _relaxSummary(results) {
        // The tracked relaxation phase (v5; plan D3/D4): what the deck or the
        // read-back recorded, in one line.  v4 files carry no block.
        const rx = results.relaxation || {};
        // THE JUDGED FORCE FIRST, whichever route measured it (vibration.md
        // § 4.3 for PySCF, § 5.5 for SIESTA), on every path that carries
        // it.  The key says its unit.
        let force = "";
        if (rx.max_force_eh_bohr != null) {
            force = " \u2014 max |F| on the free atoms "
                  + Number(rx.max_force_eh_bohr).toExponential(1) + " Eh/Bohr"
                  + (rx.converged === true ? " (within the criterion)"
                     : rx.converged === false ? " \u26a0 above the criterion"
                     : "");
        }
        if (!rx.enabled) {
            // Nothing ran: the person's assertion (either route), or the
            // force-constant route that relaxes nothing and says so in its
            // warning; a v4 file says nothing at all.
            if (rx.already_relaxed) {
                return "asserted relaxed"
                     + (rx.warning ? " \u2014 " + rx.warning : "") + force;
            }
            return (rx.warning ? "none \u2014 " + rx.warning
                               : (results.phase_relaxation || "\u2014")) + force;
        }
        let out = results.phase_relaxation || "empty";
        if (rx.n_steps != null) out += " \u2014 " + rx.n_steps + " steps";
        out += force;
        if (rx.warning) out += " \u26a0 " + rx.warning;
        return out;
    }

    function _resultsFingerprint(results, selectedMode) {
        // Compact key over the fields renderResults branches on.  Any
        // change here means "user-visible viewer state needs to update".
        // Stable string makes equality cheap — bail without rerendering
        // when the live-watch poll returned an unchanged snapshot.
        //
        // Includes per-mode Raman + IR activity values and frequencies:
        // activities populate mid-phase (same mode count, phase still
        // "running", no ES flip), and the bar heights ARE what the user
        // is watching.
        //
        // ``actBits`` is a folded checksum (sum of activity values
        // truncated to 3 decimals).  Sum is order-stable because we
        // walk modes[] in index order, and 3-decimal precision keeps
        // the string short while catching real activity changes.
        const modes = results.modes || [];
        const esBits = modes.map(m => m.electronic_structure ? "1" : "0").join("");
        function _actSum(field) {
            let s = 0;
            for (const m of modes) {
                const v = m[field];
                if (v != null && Number.isFinite(v)) s += v;
            }
            return s.toFixed(3);
        }
        function _freqSum() {
            let s = 0;
            for (const m of modes) {
                const v = m.frequency_cm1;
                if (v != null && Number.isFinite(v)) s += v;
            }
            return s.toFixed(2);
        }
        return [
            results.n_atoms_total,
            modes.length,
            esBits,
            _freqSum(),
            _actSum("raman_activity_a4_amu"),
            _actSum("ir_intensity_km_mol"),
            results.phase_frequencies || "",
            results.phase_raman || "",
            results.phase_ir || "",
            results.ir_route || "",
            results.raman_route || "",
            results.phase_es || "",
            results.phase_relaxation || "",
            String((results.relaxation
                    && results.relaxation.n_steps) || ""),
            (((results.thermo || {}).grid || {}).temperatures_K
                ? "thermo" : ""),
            selectedMode == null ? "-" : String(selectedMode),
        ].join("|");
    }

    function _pickDefaultMode(modes, preferES) {
        if (preferES) {
            // First mode with ES populated, sorted by Raman activity
            // descending if available.
            const withES = modes.filter(m => !!m.electronic_structure);
            if (withES.length) {
                withES.sort((a, b) =>
                    (b.raman_activity_a4_amu || 0) -
                    (a.raman_activity_a4_amu || 0)
                );
                return withES[0].index_1based;
            }
        }
        // Fallback: brightest real mode by Raman, else first real,
        // else first mode.
        const real = modes.filter(m => !m.has_imag);
        const pool = real.length ? real : modes;
        const ranked = pool
            .filter(m => m.raman_activity_a4_amu != null)
            .sort((a, b) => b.raman_activity_a4_amu - a.raman_activity_a4_amu);
        return (ranked[0] || pool[0]).index_1based;
    }

    // ----- Mode table: sort + filter + selection + CSV ----------
    //
    // The table is the tabular twin of the spectrum chart (§ 9.2.2).
    // Sort + filter + row click all run client-side against
    // state.results.modes; the table is re-rendered on each state
    // change.  Cheap (typical mode counts are <1000).
    function renderModesTable() {
        if (!state.results) return;
        const modes = _modesForTable();
        const anyES = (state.results.modes || []).some(m => !!m.electronic_structure);
        // Build rows via createElement instead of innerHTML+string concat:
        // a future free-text column on a mode would otherwise be an
        // interpolation hazard.  textContent / appendChild keeps the
        // surface trustworthy by construction.
        els.modesTbody.replaceChildren(
            ...modes.map(m => _renderModeRow(m, anyES)));

        // Update filter-result count.
        const total = (state.results.modes || []).length;
        if (state.modeFilter) {
            setStatus(els.modesFilterCount,
                      `${modes.length} of ${total} modes match`,
                      "muted");
        } else {
            setStatus(els.modesFilterCount, "", "muted");
        }

        // Re-apply the active-row highlight after the rebuild.
        _highlightActiveRow();
        _updateSortIndicators();
    }

    function _modesForTable() {
        const modes  = (state.results.modes || []).slice();
        // Filter: case-insensitive substring across all stringified
        // visible column values.
        const filt = (state.modeFilter || "").trim().toLowerCase();
        const filtered = filt
            ? modes.filter(m => _modeMatchesFilter(m, filt))
            : modes;
        // Sort.
        const col = state.sortColumn;
        const dir = state.sortDir === "desc" ? -1 : 1;
        const key = (m) => _modeKey(m, col);
        filtered.sort((a, b) => {
            const ka = key(a), kb = key(b);
            // null/undefined sort to the bottom regardless of dir
            // (a missing value isn't "smaller" than a real one --
            // it just has no value).
            if (ka == null && kb == null) return 0;
            if (ka == null) return 1;
            if (kb == null) return -1;
            if (ka < kb) return -dir;
            if (ka > kb) return dir;
            return 0;
        });
        return filtered;
    }

    function _modeMatchesFilter(m, filt) {
        // Match against the same fields the table shows, stringified.
        const es = m.electronic_structure;
        const vals = [
            String(m.index_1based),
            Number(m.frequency_cm1).toFixed(1),
            m.raman_activity_a4_amu != null
                ? Number(m.raman_activity_a4_amu).toFixed(2) : "",
            m.ir_intensity_km_mol != null
                ? Number(m.ir_intensity_km_mol).toFixed(2) : "",
            m.has_imag ? "imag" : "",
            es ? "es" : "",
        ];
        if (es) {
            const homo = es.mo_energies_eq_eh[es.homo_index_in_window];
            const lumo = es.mo_energies_eq_eh[es.homo_index_in_window + 1];
            if (homo != null) vals.push((homo * K.hartree_ev).toFixed(3));
            if (lumo != null) vals.push((lumo * K.hartree_ev).toFixed(3));
            if (homo != null && lumo != null)
                vals.push(((lumo - homo) * K.hartree_ev).toFixed(3));
        }
        return vals.some(v => v.toLowerCase().includes(filt));
    }

    function _modeKey(m, col) {
        switch (col) {
            case "index_1based":          return m.index_1based;
            case "frequency_cm1":         return m.frequency_cm1;
            case "raman_activity_a4_amu": return m.raman_activity_a4_amu;
            case "ir_intensity_km_mol":   return m.ir_intensity_km_mol;
            case "has_imag":              return m.has_imag ? 1 : 0;
            case "has_es":                return m.electronic_structure ? 1 : 0;
            case "homo_eq_ev":            return _homoEq(m);
            case "lumo_eq_ev":            return _lumoEq(m);
            case "gap_eq_ev":             return _gapEq(m);
            case "dgap_max_mev":          return _dgapMax(m);
            default:                      return m.index_1based;
        }
    }

    function _homoEq(m) {
        const es = m.electronic_structure;
        if (!es) return null;
        const e = es.mo_energies_eq_eh[es.homo_index_in_window];
        return e == null ? null : e * K.hartree_ev;
    }
    function _lumoEq(m) {
        const es = m.electronic_structure;
        if (!es) return null;
        const e = es.mo_energies_eq_eh[es.homo_index_in_window + 1];
        return e == null ? null : e * K.hartree_ev;
    }
    function _gapEq(m) {
        const h = _homoEq(m), l = _lumoEq(m);
        return h == null || l == null ? null : l - h;
    }
    function _dgapMax(m) {
        const es = m.electronic_structure;
        if (!es) return null;
        const h = es.mo_energies_eq_eh[es.homo_index_in_window];
        const l = es.mo_energies_eq_eh[es.homo_index_in_window + 1];
        const hp = es.mo_energies_plus_eh[es.homo_index_in_window];
        const lp = es.mo_energies_plus_eh[es.homo_index_in_window + 1];
        const hm = es.mo_energies_minus_eh[es.homo_index_in_window];
        const lm = es.mo_energies_minus_eh[es.homo_index_in_window + 1];
        if ([h, l, hp, lp, hm, lm].some(x => x == null)) return null;
        const dPlus  = ((lp - hp) - (l - h)) * K.hartree_ev * 1000;  // meV
        const dMinus = ((lm - hm) - (l - h)) * K.hartree_ev * 1000;
        return Math.max(Math.abs(dPlus), Math.abs(dMinus));
    }

    function _renderModeRow(m, anyES) {
        // Returns an HTMLTableRowElement; caller appends to tbody.
        const fmt = (v, dp) => v == null ? "—" : Number(v).toFixed(dp);
        const raman = (m.raman_activity_a4_amu == null)
            ? "—"
            : Number(m.raman_activity_a4_amu).toFixed(2);
        /* The Raman, IR and orbital cells carry their column's class, so a
         * column the route cannot compute goes with its header
         * (`.route-no-*`, web/spectra.md § 9b.3). */
        /* "—" IS NOT ZERO.  A null means the run did not compute this
         * channel (`compute_ir` is off by default); 0.00 means it did and the
         * mode is silent there.  CO2 needs both readings in one table: its
         * symmetric stretch is 0.00 km/mol by symmetry, while a Raman-only
         * run leaves every IR cell "—". */
        const ir = (m.ir_intensity_km_mol == null)
            ? "—"
            : Number(m.ir_intensity_km_mol).toFixed(2);
        const tr = document.createElement("tr");
        tr.dataset.mode = String(m.index_1based);
        if (m.has_imag) tr.className = "mode-imag";
        const addCell = (text, cls) => {
            const td = document.createElement("td");
            td.textContent = text;
            if (cls) td.className = cls;
            tr.appendChild(td);
        };
        addCell(String(m.index_1based));
        addCell(Number(m.frequency_cm1).toFixed(1));
        addCell(raman, "raman-col");
        addCell(ir, "ir-col");
        addCell(m.has_imag ? "✓" : "");
        addCell(m.electronic_structure ? "✓" : "", "es-col");
        /* ALWAYS THE FOUR ES VALUE CELLS, because the header always has
         * their four `es-col` <th> (`_spectra_inspector.html`, beside
         * the ES? column).  An empty cell says "no value"; a missing
         * cell shifts the whole row.  Where the ROUTE has no probe, the
         * table's `route-no-es` class hides the header and these cells
         * together (web/spectra.md § 9b.3), so the two still line up. */
        addCell(anyES ? fmt(_homoEq(m), 3)  : "", "es-col");
        addCell(anyES ? fmt(_lumoEq(m), 3)  : "", "es-col");
        addCell(anyES ? fmt(_gapEq(m),  3)  : "", "es-col");
        addCell(anyES ? fmt(_dgapMax(m), 1) : "", "es-col");
        return tr;
    }

    function _highlightActiveRow() {
        const rows = els.modesTbody.querySelectorAll("tr");
        let chosen = null;
        rows.forEach(r => {
            const active = Number(r.dataset.mode) === state.selectedMode;
            r.classList.toggle("active", active);
            r.setAttribute("aria-selected", active ? "true" : "false");
            if (active) chosen = r;
        });
        _scrollRowToTop(chosen);
    }

    /* Put the chosen row at the top of the table, just under the header.
     *
     * A click on the spectrum can pick a mode hundreds of rows down, and a
     * highlight you have to go looking for is not much of an answer.  The
     * obvious `scrollIntoView({block: "nearest"})` is not enough here: the head
     * is `position: sticky`, so "just in view" means the row sits UNDERNEATH it
     * and is only technically visible.  Hence the arithmetic -- scroll by
     * exactly the distance from the row to the top of the scroll box, less the
     * head that covers it -- and hence the same place every time, so the
     * selected mode is always where the eye already is. */
    function _scrollRowToTop(row) {
        if (!row) return;
        const wrap = row.closest && row.closest(".modes-table-wrap");
        if (!wrap || typeof wrap.scrollTo !== "function") return;
        const head = wrap.querySelector("thead");
        const headHeight = head ? head.getBoundingClientRect().height : 0;
        const offset = row.getBoundingClientRect().top
                     - wrap.getBoundingClientRect().top
                     - headHeight;
        if (Math.abs(offset) < 1) return;            // already there
        wrap.scrollTo({ top: Math.max(0, wrap.scrollTop + offset), behavior: "smooth" });
    }

    function _updateSortIndicators() {
        const headers = els.modesTheadRow.querySelectorAll("th");
        headers.forEach(th => {
            th.classList.remove("sort-asc", "sort-desc");
            th.removeAttribute("aria-sort");
            if (th.dataset.col === state.sortColumn) {
                th.classList.add(state.sortDir === "desc" ? "sort-desc" : "sort-asc");
                th.setAttribute(
                    "aria-sort",
                    state.sortDir === "desc" ? "descending" : "ascending"
                );
            }
        });
    }

    function onTableHeaderClick(ev) {
        const th = ev.target.closest("th");
        if (!th || !th.dataset.col) return;
        const col = th.dataset.col;
        if (state.sortColumn === col) {
            state.sortDir = state.sortDir === "asc" ? "desc" : "asc";
        } else {
            state.sortColumn = col;
            // Default sort direction: numeric columns descending
            // (so "biggest Raman activity first" is the natural reach
            // for the user), index ascending (so "first mode first").
            state.sortDir = (col === "index_1based") ? "asc"
                          : (th.dataset.numeric === "1") ? "desc"
                          : "asc";
        }
        renderModesTable();
    }

    function onTableRowClick(ev) {
        const tr = ev.target.closest("tr[data-mode]");
        if (!tr) return;
        selectMode(Number(tr.dataset.mode));
    }

    function onFilterInput() {
        state.modeFilter = els.modesFilter.value || "";
        renderModesTable();
    }

    function selectMode(idx) {
        if (!state.results) return;
        state.selectedMode = Number(idx) || null;
        _highlightActiveRow();
        renderESPanel();
        renderModeViewer();
        // The chart mirrors the selection through its cheap door: one mark
        // recoloured, no curve recomputed, no axis moved (spectrumchart § 5.1).
        //
        // Queued on the mount rather than guarded by `if (chart)`: the mount is
        // asynchronous, so a row clicked in the first moments after a result
        // loads would otherwise move the table and the viewer while the chart
        // kept the old highlight.
        _withChart(c => c.setSelected(state.selectedMode));
    }

    // ----- CSV export ------------------------------------------
    function exportCSV() {
        if (!state.results) return;
        const anyES = (state.results.modes || []).some(m => !!m.electronic_structure);
        // BOTH channels, beside each other.
        // THE EXPORT FOLLOWS THE TABLE: a column the route cannot compute
        // is left out of both (web/spectra.md § 9b.3).
        const hasRaman = _routeHas(state.results, "compute_raman");
        const hasIr = _routeHas(state.results, "compute_ir");
        const hasEs = _routeHas(state.results, "es_mode_selection");
        const headers = ["index_1based", "frequency_cm1"]
            .concat(hasRaman ? ["raman_activity_a4_amu"] : [])
            .concat(hasIr ? ["ir_intensity_km_mol"] : [])
            .concat(["has_imag"])
            .concat(hasEs ? ["has_es"] : []);
        if (anyES) headers.push("homo_eq_ev", "lumo_eq_ev",
                                 "gap_eq_ev", "dgap_max_mev");
        const lines = [headers.join(",")];
        for (const m of _modesForTable()) {
            const row = [m.index_1based, Number(m.frequency_cm1).toFixed(4)];
            if (hasRaman) row.push(m.raman_activity_a4_amu == null ? ""
                : Number(m.raman_activity_a4_amu).toFixed(4));
            if (hasIr) row.push(m.ir_intensity_km_mol == null ? ""
                : Number(m.ir_intensity_km_mol).toFixed(4));
            row.push(m.has_imag ? "1" : "0");
            if (hasEs) row.push(m.electronic_structure ? "1" : "0");
            if (anyES) {
                const fmt4 = v => v == null ? "" : Number(v).toFixed(4);
                row.push(fmt4(_homoEq(m)));
                row.push(fmt4(_lumoEq(m)));
                row.push(fmt4(_gapEq(m)));
                row.push(_dgapMax(m) == null ? "" : Number(_dgapMax(m)).toFixed(2));
            }
            lines.push(row.join(","));
        }
        const blob = new Blob([lines.join("\n") + "\n"],
                              { type: "text/csv" });
        const url  = URL.createObjectURL(blob);
        const a    = document.createElement("a");
        a.href     = url;
        a.download = "spectra-modes.csv";
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    }

    // ----- ES panel (§ 9.2.4) ----------------------------------
    //
    // MO bar diagram for the selected mode: three columns (-A, eq, +A),
    // each plotting MO energies in eV as horizontal bars.  HOMO and
    // LUMO are highlighted; the gap drift Δ(LUMO−HOMO) between
    // displaced and equilibrium geometries is annotated underneath.
    /* ---- the level diagram's own helpers ---------------------------------
     *
     * They serve the electronic-structure diagram alone (the spectrum is its
     * own module, docs/web/spectrumchart.md), so they live beside the diagram
     * and are named for it.
     */
    /* THE CHART PALETTE, READ FROM THE STYLESHEET.
     *
     * Plotly takes colours as JavaScript values, so a chart cannot inherit them
     * the way an element does.
     *
     * The tokens are the source of truth (lib/tokens.css), so this asks the
     * document for their computed values and hands Plotly the answer.  One read,
     * cached: they cannot change without a reload.
     */
    let _theme = null;
    function _esTheme() {
        if (_theme) return _theme;
        const css = getComputedStyle(document.documentElement);
        const tok = (name, fallback) =>
            (css.getPropertyValue(name) || "").trim() || fallback;
        _theme = {
            paper:  tok("--bg-card",         "#1d2128"),
            grid:   tok("--border-soft",     "#2c313a"),
            axis:   tok("--border-strong",   "#3a3f48"),
            ink:    tok("--text-secondary",  "#a8aebb"),
            dim:    tok("--text-muted",      "#959ba7"),
            homo:   tok("--accent",          "#6ba6ff"),
            lumo:   tok("--warn-soft",       "#d8a64b"),
            // The spectrum's sticks: a real mode, an imaginary one, the
            // selected one, and the envelope over them.
            stick:     tok("--accent-strong", "#4a8de0"),
            stickImag: tok("--error",         "#f87171"),
            stickSel:  tok("--warning",       "#fbbf24"),
            envelope:  tok("--accent-hover",  "#8ab8ff"),
        };
        return _theme;
    }

    /* Plotly's `responsive: true` listens to the WINDOW, and the window is not
     * what changes here -- the sidebar collapses, the inspector panel resizes,
     * the container query flips the layout, and the window never moves.  So each
     * chart watches its own box.
     *
     * One observer, installed once, for the one figure this tab still draws. */
    function _watchEsWidth(el) {
        const key = "esResizeObserver";
        const node = el;
        if (state[key] || !node) return;
        if (typeof ResizeObserver === "undefined") return;
        let last = 0;
        const obs = new ResizeObserver(function (entries) {
            const w = entries[0] && entries[0].contentRect.width;
            if (!w || Math.abs(w - last) < 1) return;   // ignore sub-pixel noise
            last = w;
            if (typeof Plotly === "undefined") return;
            try { Plotly.Plots.resize(node); } catch (_) {}
        });
        obs.observe(node);
        state[key] = obs;
    }

    function renderESPanel() {
        if (!els.esPanel) return;
        if (!state.results || state.selectedMode == null) {
            els.esPanel.hidden = true;
            return;
        }
        const m = (state.results.modes || []).find(
            x => x.index_1based === state.selectedMode
        );
        if (!m) {
            els.esPanel.hidden = true;
            return;
        }
        els.esPanel.hidden = false;
        els.esModeIdx.textContent  = String(m.index_1based);

        const es = m.electronic_structure;

        /* THE DISPLACEMENT THE LEVELS WERE COMPUTED AT belongs in the header,
         * beside the mode it describes: it is the INPUT every number
         * below depends on, and the one the coupling divides by.  Stated here it
         * labels the whole panel, which is what "±A" in the diagram means. */
        els.esModeFreq.textContent =
            Number(m.frequency_cm1).toFixed(1) + " cm⁻¹"
            + (m.has_imag ? " (imaginary)" : "")
            + (es ? "  ·  A = " + es.amplitude_ang.toFixed(3) + " Å" : "");
        if (!es) {
            /* PURGE BEFORE OVERWRITING.  A mode with no electronic structure
             * replaces the figure with a sentence, and the node may be holding a
             * Plotly figure from the previously selected mode -- clearing its
             * innerHTML underneath the library leaves that figure's internal
             * state attached to a node that no longer contains it. */
            if (typeof Plotly !== "undefined") {
                try { Plotly.purge(els.esBarDiagram); } catch (_) {}
            }
            // `.status` keeps newlines (white-space: pre-line), so the "\n"
            // breaks the line.
            /* WHY THERE IS NONE, by role (web/spectra.md § 9b.3): a route
             * with no probe is told what its electronic response will be,
             * never an item it has no way to set. */
            const r = state.results || {};
            const running = r.phase_es === "empty" || r.phase_es === "running";
            // The selection is in the file from the first probed mode on, so
            // a mode it left out is told so while the probe still runs.
            const sel = r.selected_mode_idxs_1based || [];
            const leftOut = sel.length > 0 && !sel.includes(m.index_1based);
            const why = !_routeHas(r, "es_mode_selection")
                ? "This route computes the modes, not the electrons' "
                  + "response along them.\nThat response is the projected "
                  + "density of states at structures displaced along a mode "
                  + "-- planned (engines/vibration.md § 5.10), not built yet."
                : r.config.es_mode_selection === "skip"
                ? "The per-mode orbital check was not requested in this run "
                  + "(Mode selection: skip)."
                : (running && !leftOut)
                ? "The per-mode orbital check has not reached this mode yet."
                : "This mode was not in the run's selection for the per-mode "
                  + "orbital check.";
            showNotice(els.esBarDiagram, why, "muted");
            els.esSummary.innerHTML = "";
            return;
        }

        // Convert MO arrays to eV.
        const eq    = es.mo_energies_eq_eh.map(e => e * K.hartree_ev);
        const minus = es.mo_energies_minus_eh.map(e => e * K.hartree_ev);
        const plus  = es.mo_energies_plus_eh.map(e => e * K.hartree_ev);
        const hi    = es.homo_index_in_window;
        const li    = hi + 1;

        // Y-range: include all three displaced + eq arrays.
        const all = eq.concat(minus, plus);
        const lo  = Math.min.apply(null, all);
        const up  = Math.max.apply(null, all);
        const pad = (up - lo) * 0.05 || 0.1;
        const yMin = lo - pad, yMax = up + pad;

        /* THE FULL RANGE IS THE STARTING VIEW, not the only one: the user zooms
         * from here.  Re-set on every mode change so switching modes always
         * lands on the whole picture rather than inside the previous mode's
         * zoom, which would show a different molecule's window without saying so. */
        const fig = _renderLevelDiagram({
            minus: minus, eq: eq, plus: plus,
            homo_idx: hi, lumo_idx: li,
        });
        fig.layout.yaxis.range = [yMin, yMax];
        if (typeof Plotly === "undefined") {
            els.esBarDiagram.innerHTML =
                '<p class="status muted">chart library unavailable on this page</p>';
        } else {
            Plotly.react(els.esBarDiagram, fig.traces, fig.layout, fig.config);
            _watchEsWidth(els.esBarDiagram);
        }

        /* THE NUMBERS, GROUPED BY THE QUESTION THEY ANSWER.
         *
         * Three groups
         * say what each number is FOR -- where the levels sit, how the gap
         * moves when the molecule does, and how strongly this mode couples --
         * and that is the order a reader asks them in.
         *
         * Value and unit are separate fields so the stylesheet can right-align
         * the digits into a column; "−6.1234 eV" as one string cannot line up
         * with "12.34 meV" below it. */
        const gap_eq    = eq[li]    - eq[hi];
        const gap_plus  = plus[li]  - plus[hi];
        const gap_minus = minus[li] - minus[hi];
        const dgap_plus_mev  = (gap_plus  - gap_eq) * 1000;
        const dgap_minus_mev = (gap_minus - gap_eq) * 1000;
        // Electron-phonon coupling magnitude per spec § 9.2.4:
        //   g_HOMO = ΔE_HOMO(+A→−A) / (2A)  (meV/Å -- approximate)
        // The full spec divides by √(ℏ/(2mω)) but that requires the
        // mass-weighted normal coordinate magnitude per mode, which
        // we don't currently emit.  Showing the simpler ΔE/(2A) form
        // gives the user a first-pass EPC magnitude they can scale
        // later.
        const g_HOMO_mev_A = ((plus[hi] - minus[hi]) / (2 * es.amplitude_ang)) * 1000;
        const g_LUMO_mev_A = ((plus[li] - minus[li]) / (2 * es.amplitude_ang)) * 1000;

        const groups = [
            ["Where the levels sit", [
                ["HOMO",       eq[hi].toFixed(4),  "eV"],
                ["LUMO",       eq[li].toFixed(4),  "eV"],
                ["Gap",        gap_eq.toFixed(4),  "eV"],
            ]],
            ["How the gap moves", [
                ["at −A",      gap_minus.toFixed(4),      "eV"],
                ["at +A",      gap_plus.toFixed(4),       "eV"],
                ["change, −A", dgap_minus_mev.toFixed(2), "meV"],
                ["change, +A", dgap_plus_mev.toFixed(2),  "meV"],
            ]],
            ["Coupling, ΔE/(2A)", [
                ["HOMO",       g_HOMO_mev_A.toFixed(1), "meV/Å"],
                ["LUMO",       g_LUMO_mev_A.toFixed(1), "meV/Å"],
            ]],
        ];

        const node = (tag, cls, text) => {
            const n = document.createElement(tag);
            if (cls) n.className = cls;
            if (text !== undefined) n.textContent = text;
            return n;
        };
        els.esSummary.replaceChildren(...groups.map(([title, rows]) => {
            const dl = node("dl");
            for (const [k, v, u] of rows) {
                dl.append(node("dt", "", k), node("dd", "es-val", v),
                          node("dd", "es-unit", u));
            }
            const group = node("div", "es-group");
            group.append(node("h4", "es-group-title", title), dl);
            return group;
        }));
    }


    /* THE LEVEL DIAGRAM, drawn by the same library as the spectrum above it.
     *
     * The level shifts this panel exists to show are tiny -- 0.018 meV
     * against an 11.4 eV span in the BDT result, which is 1/4000 of a pixel --
     * so they cannot be seen without zooming, and zoom means pan, range
     * memory, a reset control and a hover readout, which the chart library
     * already loaded on this page provides.
     *
     * WHAT IS FIXED AND WHAT MOVES.  The x axis is three geometries, not a
     * quantity -- there is nothing between −A and eq -- so it is categorical and
     * `fixedrange`, and dragging or scrolling only ever moves the ENERGY axis.
     * That is the one axis worth exploring, and locking the other means a stray
     * gesture cannot leave the figure in a state that has to be reasoned about.
     *
     * ONE TRACE PER ROLE, not per level: a scatter trace draws every segment it
     * is given if the runs are separated by nulls, so the whole crowd of
     * occupied levels is one trace, the tie lines another, and HOMO and LUMO
     * carry their own so the legend can name them.
     */
    function _levelSegments(cols, index, centre, half) {
        // The horizontal dash for one orbital, in each of the three columns.
        const x = [], y = [];
        cols.forEach((col, c) => {
            if (index >= col.arr.length) return;
            x.push(centre(c) - half, centre(c) + half, null);
            y.push(col.arr[index],   col.arr[index],   null);
        });
        return { x: x, y: y };
    }

    function _tieSegments(cols, index, centre, half) {
        // The bridge across each gap, joining one orbital to itself.
        const x = [], y = [];
        for (let c = 0; c < cols.length - 1; c++) {
            if (index >= cols[c].arr.length || index >= cols[c + 1].arr.length) continue;
            x.push(centre(c) + half, centre(c + 1) - half, null);
            y.push(cols[c].arr[index], cols[c + 1].arr[index], null);
        }
        return { x: x, y: y };
    }

    function _renderLevelDiagram(opts) {
        const th = _esTheme();
        const cols = [
            { label: "−A", arr: opts.minus },
            { label: "eq", arr: opts.eq    },
            { label: "+A", arr: opts.plus  },
        ];
        /* The three geometries sit at x = 0, 1, 2 and each level is drawn as a
         * dash of ±HALF around its column.  HALF under 0.5 is what leaves a gap
         * between columns for the tie lines to cross -- the columns are as wide
         * as they are apart, so nothing is spread across empty space. */
        const HALF   = 0.3;
        const centre = (c) => c;

        const nLevels = Math.max(opts.eq.length, opts.minus.length, opts.plus.length);
        const crowd = { x: [], y: [] }, ties = { x: [], y: [] };
        const traces = [];

        for (let i = 0; i < nLevels; i++) {
            const seg = _levelSegments(cols, i, centre, HALF);
            const tie = _tieSegments(cols, i, centre, HALF);
            const frontier = (i === opts.homo_idx) ? "homo"
                           : (i === opts.lumo_idx) ? "lumo" : null;
            if (!frontier) {
                crowd.x.push.apply(crowd.x, seg.x); crowd.y.push.apply(crowd.y, seg.y);
                ties.x.push.apply(ties.x, tie.x);   ties.y.push.apply(ties.y, tie.y);
                continue;
            }
            /* TWO TRACES, NOT ONE, and the difference is the whole readability
             * of the figure.  Drawn at the same weight, a level and its tie line
             * merge into a single bar spanning the plot -- which is exactly what
             * a mode with no shift looks like, so the reader cannot tell three
             * levels joined from one line that never moved.  A thin connector
             * between thick dashes keeps the three geometries legible, and a
             * shift then reads as what it is: a sloping link. */
            const colour = frontier === "homo" ? th.homo : th.lumo;
            traces.push({
                type: "scatter", mode: "lines", showlegend: false, hoverinfo: "skip",
                x: tie.x, y: tie.y,
                line: { color: colour, width: 1.1 }, opacity: 0.8,
            });
            traces.push({
                type: "scatter", mode: "lines", name: frontier.toUpperCase(),
                x: seg.x, y: seg.y,
                line: { color: colour, width: 3 },
                hovertemplate: frontier.toUpperCase() + ": %{y:.4f} eV<extra></extra>",
            });
        }

        // Behind the frontier pair: the tie lines, then the levels themselves.
        traces.unshift(
            { type: "scatter", mode: "lines", name: "other levels",
              x: crowd.x, y: crowd.y,
              line: { color: th.dim, width: 1.3 },
              hovertemplate: "%{y:.4f} eV<extra></extra>" },
        );
        traces.unshift(
            { type: "scatter", mode: "lines", showlegend: false, hoverinfo: "skip",
              x: ties.x, y: ties.y,
              line: { color: th.axis, width: 1 }, opacity: 0.55 },
        );

        const layout = {
            margin: { t: 8, r: 8, b: 30, l: 52 },
            /* NO `height` HERE.  The box owns the height (spectra.css
             * .es-bar-diagram) and the plot fills it, so the figure follows the
             * layout rather than fighting it -- setting both means the CSS box
             * and the library disagree about how tall the figure is. */
            xaxis: {
                // Three geometries, not a continuum: no grid, no zoom, and a
                // little slack so the outer columns are not clipped.
                tickmode: "array",
                tickvals: cols.map((_, c) => centre(c)),
                ticktext: cols.map(c => c.label),
                range: [-0.5, cols.length - 0.5],
                fixedrange: true,
                zeroline: false, showgrid: false,
                color: th.ink,
            },
            yaxis: {
                title: { text: "Energy (eV)", font: { size: 11 } },
                gridcolor: th.grid,
                zeroline: false,
                color: th.ink,
                // This axis is free.  Scroll to zoom,
                // drag to pan, double-click to come back.
                fixedrange: false,
            },
            plot_bgcolor: th.paper,
            paper_bgcolor: th.paper,
            font: { color: th.ink, size: 10 },
            hovermode: "closest",
            dragmode: "pan",
            legend: { orientation: "h", y: 1.14, font: { size: 10 } },
            showlegend: true,
        };

        const config = {
            displaylogo: false,
            responsive: true,
            scrollZoom: true,          // wheel over the plot zooms the energy axis
            modeBarButtonsToRemove: ["select2d", "lasso2d", "zoom2d", "toggleSpikelines"],
        };
        return { traces: traces, layout: layout, config: config };
    }

    // ----- Mode-animation viewer (§ 9.2.3) ---------------------
    //
    // The concealed VibrationView module (vibrationview.md) renders the
    // equilibrium structure inside #mode-viewer and animates the selected
    // mode -- it owns the loop that adds the eigenvector displacement times
    // cos(phase) to each atom's equilibrium position every frame.  Spectra
    // just hands it the geometry + mode via vib.showMode; it never touches a
    // raw viewer.
    //
    // Geometry source: results.equilibrium.elements + positions_ang
    // (works after page reload).
    //
    // The mode shape is faithful (eigenvector_display carries the
    // direction + relative amplitudes correctly, with max(|L|)=1 per
    // mode so every mode reaches the same peak amplitude on screen).
    // The display amplitude slider is a user-tunable visualisation
    // knob, not a physical quantity -- thermal RMS amplitudes are
    // typically < 0.05 Å and too small to see otherwise.  For
    // physical-amplitude work (Raman re-projection, etc.), the JSON
    // also ships eigenvector_canonical with the mass-weighted unit
    // norm Σ_k m_k|L_k|² = 1.

    /* ── How big a vibration actually is (vibrationview.md § 12.2) ──────────
     *
     * The eigenvector fixes the SHAPE of the motion and not its size: its overall
     * scale is arbitrary until something fixes it. Two things can:
     *
     *   DISPLAY   the largest-moving atom swings by whatever the slider says.
     *             A drawing choice, using the display-normalised eigenvector
     *             (max|L| = 1, dimensionless), so the amplitude is in angstrom.
     *
     *   PHYSICAL  the atoms swing by as much as they do. The size comes from the
     *             mode's own frequency, using the mass-weighted eigenvector
     *             (Σ mᵢ|Lᵢ|² = 1, so L is in 1/√mass) — and the amplitude is then
     *             in √amu·Å, which is why the two can never share a slider.
     *
     *         zero-point   Q = √(ħ / 2ω)
     *         thermal      Q = √(ħ / 2ω · coth(ħω / 2k_BT))
     *
     *     The thermal form reduces to the zero-point one as T → 0, which is the
     *     check that they are one expression and not two.
     *
     * THIS IS THE TAB'S ARITHMETIC, not the viewer's. VibrationView holds no
     * frequency, no temperature and no physical constant (§ 12.2); it is handed a
     * displacement array and a number and animates their product.
     */

    // The zero-point amplitude is READ FROM THE FILE, never recomputed here:
    // every mode carries `zero_point_amplitude_amu12_ang`, derived at every
    // serialisation from the one constant in constants.py (vibration.md
    // § 6.3, § 6.6).
    // ħω / k_B per cm⁻¹, in kelvin -- the temperature at which a mode's
    // quantum is comparable to kT -- is `K.cm1_kelvin`, served (see `K`).

    /* Above this, calling a nearest neighbour a "bond" would be a claim rather
     * than a label, so the readout says "nearest contact" instead.  Generous on
     * purpose: it has to cover the long ones this program actually builds, Au–Au
     * at 2.88 Å and Au–S near 2.4 Å, and it decides one word of wording -- not
     * chemistry, not what is drawn. */
    const BOND_LIKE_ANG = 3.0;

    function _physicalAmplitude(modeRow, mode, temperatureK) {
        const nu = Math.abs(Number(modeRow && modeRow.frequency_cm1));
        const qzp = Number(modeRow && modeRow.zero_point_amplitude_amu12_ang);
        // An imaginary or zero mode carries null, and a file older than the
        // key carries nothing: no physical amplitude, the display form stays.
        if (!isFinite(nu) || nu <= 0 || !isFinite(qzp) || qzp <= 0) return null;
        let q = qzp;
        if (mode === "thermal") {
            const t = Number(temperatureK);
            if (isFinite(t) && t > 0) {
                const x = K.cm1_kelvin * nu / (2 * t);
                // coth(x); at large x this is 1 and the mode is in its ground
                // state, which is why a stiff mode at room temperature comes back
                // barely different from zero-point.
                q *= Math.sqrt(1 / Math.tanh(x));
            }
        }
        return q;
    }

    /* THE ONE CUT (vibrationview.md § 11).
     *
     * A spectra result carries far more than an animation needs, and the three
     * things it does need used to be read from four places scattered through this
     * file.  This is the only function that knows both the shape of a
     * `.spectra.json` and the shape of a mode, which is what keeps VibrationView
     * from ever naming spectra and the server from ever naming VibrationView.
     *
     * When the result cannot be animated it says so, and the caller says so
     * rather than finding a structure somewhere else: whatever MolView holds
     * on /results is whatever the last inspection installed, and an atom
     * count alone cannot tell two molecules of the same size apart.
     */
    /* ONE SHAPE, ALWAYS, so a caller cannot forget a case:
     *
     *     { ready: false, why: null }      nothing is selected — say nothing
     *     { ready: false, why: "…" }       selected, but it cannot be drawn
     *     { ready: true, structure, mode, amplitude, norm }
     *
     * Checking `ready` is the only thing to remember.
     */
    function _animationInputs() {
        const nothing = { ready: false, why: null };
        const r = state.results;
        if (!r || state.selectedMode == null) return nothing;

        const eq = r.equilibrium;
        if (!eq || !Array.isArray(eq.elements) || !Array.isArray(eq.positions_ang)
                || !eq.elements.length
                || eq.positions_ang.length !== eq.elements.length) {
            // Not animatable, and WHY is worth saying: results written before the
            // geometry was stored are a real thing a user still has on disk, and
            // a mode-visualisation panel that simply vanishes reads as a bug.
            return { ready: false,
                     why: "this result has no stored geometry, so its modes "
                        + "cannot be animated — re-parse the run to add one" };
        }
        const mode = (r.modes || []).find(
            m => m.index_1based === state.selectedMode);
        if (!mode || !Array.isArray(mode.eigenvector_display)) return nothing;

        // The eigenvector is indexed by FREE-atom row, so its length is the size
        // of the free set -- not of the structure.  Disagreement here means the
        // result is internally inconsistent; the viewer would refuse it anyway
        // (§ 6.3), and refusing it here says so before anything is drawn.
        const free = Array.isArray(r.free_atom_idxs) ? r.free_atom_idxs : null;
        if (free && free.length !== mode.eigenvector_display.length) {
            return { ready: false,
                     why: "this mode does not match the structure it is stored "
                        + "with, so it cannot be animated" };
        }
        if (!free && mode.eigenvector_display.length !== eq.elements.length) {
            return { ready: false,
                     why: "this mode does not match the structure it is stored "
                        + "with, so it cannot be animated" };
        }

        const hz = Number(mode.frequency_cm1);

        /* WHICH PAIRING (§ 12.2). The array and the amplitude go together: a
         * display eigenvector with an amplitude in angstrom, or a canonical one
         * with an amplitude in √amu·Å. Crossing them would be a correctness
         * bug. */
        const wantPhysical = state.animAmplitudeMode !== "display";
        const physical = wantPhysical
            ? _physicalAmplitude(mode, state.animAmplitudeMode, state.animTemperature)
            : null;
        const usePhysical = physical !== null
            && Array.isArray(mode.eigenvector_canonical);

        return {
            ready:     true,
            amplitude: usePhysical ? physical : state.animAmplitude,
            norm:      usePhysical
                ? (state.animAmplitudeMode === "thermal"
                    ? "physical, thermal at " + state.animTemperature + " K"
                    : "physical, zero-point")
                : "display",
            /* THE SAME FACT IN THE OTHER REGISTER.  `norm` is a record: it is
             * stamped into exported files, so it is terse and stable and must not
             * be reworded for looks.  This is what a person reads on screen, and
             * "display" is not English.  Built here, beside its twin, because two
             * spellings of one fact built in two places drift apart. */
            saidPlainly: usePhysical
                ? (state.animAmplitudeMode === "thermal"
                    ? "real size at " + state.animTemperature + " K"
                    : "real size, at absolute zero")
                : "drawn exaggerated",
            structure: {
                elements:  eq.elements.slice(),
                positions: eq.positions_ang.map(row => row.slice()),
            },
            mode: {
                index:         mode.index_1based,
                displacements: usePhysical ? mode.eigenvector_canonical
                                           : mode.eigenvector_display,
                basis:         free,
                /* WHICH ATOMS THE MODE BELONGS TO -- {"C": 0.914, "H": 0.086}.
                 * Computed by the server at /api/spectra/load, because it needs
                 * atomic masses and neither this page nor the .spectra.json has
                 * any.  Absent on a result the server could not weigh (an element
                 * ASE does not know), so every reader treats it as optional. */
                share:         mode.motion_share_by_element || null,
                // TEXT, not a number (§ 12.3): the sign carries meaning -- a
                // negative frequency is a saddle point, not a small number -- and
                // deciding that is spectroscopy, not drawing.
                label: "Mode " + mode.index_1based + " · "
                     + (isFinite(hz) ? hz.toFixed(1) : "?") + " cm⁻¹"
                     + (mode.has_imag ? " (imag)" : ""),
            },
        };
    }

    function renderModeViewer() {
        // Top-level entry point.  Called whenever selection / results
        // change.  Shows / hides the viewer, mounts the VibrationView
        // module lazily (_ensureViewer), and starts (or stops) the
        // animation depending on whether a mode is selected with a
        // non-null eigenvector.
        if (!els.modeViewerWrap) return;

        const inputs = _animationInputs();
        if (!inputs.ready) {
            _stopAnimation();
            // No reason to give means nothing is selected: hide the panel, since
            // there is nothing to explain.  A reason means something IS selected
            // and cannot be drawn — so say why, where the molecule would have
            // been, rather than vanishing and leaving the user to wonder which
            // click did that.
            els.modeViewerWrap.hidden = !inputs.why;
            if (inputs.why) {
                setStatus(els.viewerStatus, inputs.why, "muted");
                if (els.modeViewer) els.modeViewer.innerHTML = "";
            }
            return;
        }
        els.modeViewerWrap.hidden = false;
        /* WHICH mode is showing is written into the picture itself (§ 12.3), so
         * it rides into every exported frame -- and repeating it here would be the
         * same sentence twice, one directly above the other.  The status line
         * carries what the caption cannot: how big the motion actually is, which
         * is the number a caption in a paper would have to quote. */
        _reportSwing(inputs);
        _showMode(inputs);
    }

    /* Mounting is ASYNCHRONOUS (vibrationview.md § 8), so this returns a promise
     * and the callers do not wait on it: the molecule appears when the viewer is
     * built, and a second mode click while that is happening finds the viewer
     * already there.  Nothing is deferred or queued -- the handle a mount returns
     * is live, so there is no not-ready state for a caller to get wrong. */
    async function _showMode(inputs) {
        if (!state.vib) {
            // A build is already running: this click will be picked up when it
            // finishes, because that path re-reads the selection rather than
            // using whatever was current when it started.
            if (state.vibMounting) return;
            const make = opts.mountVibrationView;
            if (typeof make !== "function") {
                // The page that mounted this inspector did not hand the viewer
                // in.  A module cannot be looked up in a global -- that is the
                // point of it -- so this is a wiring fault, not a missing file.
                setStatus(els.viewerStatus,
                          "mode animation unavailable on this page", "muted");
                return;
            }
            state.vibMounting = true;
            els.modeViewer.innerHTML = "";
            let handle = null;
            try {
                handle = await make(els.modeViewer, {
                    amplitude: state.animAmplitude,
                    cycleSec:  1 / state.animSpeed,
                });
            } catch (e) {
                handle = { ok: false, error: (e && e.message) || String(e) };
            }
            state.vibMounting = false;
            if (!handle || handle.ok === false) {
                setStatus(els.viewerStatus,
                          "mode animation unavailable"
                          + (handle && handle.error ? " (" + handle.error + ")" : ""),
                          "muted");
                return;
            }
            state.vib = handle;
            state.vibStructure = null;

            /* THE SELECTION MAY HAVE MOVED while the viewer was being built.
             * `inputs` was worked out before the await, and a mount is slow
             * enough to click through.  Using the stale one would show the mode
             * that was selected when the box first appeared rather than the one
             * chosen since — so the current truth is read again here. */
            const fresh = _animationInputs();
            if (fresh.ready) inputs = fresh;
        }

        /* TWO DOORS, and the slow one only when it is the slow fact
         * (§ 5.1): a new result installs a structure and costs a redraw and a
         * refit; clicking through the modes of one result costs neither, which is
         * why the camera stays put while you browse. */
        const key = JSON.stringify(inputs.structure.elements);
        if (state.vibStructure !== key) {
            state.vib.setStructure(inputs.structure);
            state.vibStructure = key;
        }
        /* The amplitude is set BEFORE the mode, and it belongs to the pairing the
         * cut chose: a display eigenvector wants angstrom, a canonical one wants
         * √amu·Å (§ 12.2).  Amplitude has one home on the handle, so the tab
         * decides the number and the viewer just multiplies. */
        state.vib.setAmplitude(inputs.amplitude);
        state.vib.showMode(Object.assign({ norm: inputs.norm }, inputs.mode));

        /* IT RUNS UNLESS THE USER STOPPED IT.
         *
         * A vibration is the content here, so a still molecule is a viewer showing
         * nothing: whenever there is a mode to animate, it animates.  The only
         * thing that stops it is someone pressing Pause, and that intent is the
         * TAB's to remember -- `state.animPaused` -- not something to read back
         * off the viewer, because the viewer's clock legitimately stops for
         * reasons that are not the user: installing a new structure ends the mode
         * running against the old one (§ 5.1), and a mode that cannot be shown
         * stops it too.
         *
         * The module still never touches play/pause on its own (§ 9.2).  Deciding
         * that a mode should be moving is policy, and policy is the tab's. */
        if (!state.animPaused) state.vib.play();
        _syncPlayButton();
    }

    function _syncPlayButton() {
        if (!els.animToggle) return;
        els.animToggle.textContent = state.animPaused ? "Play" : "Pause";
    }

    /* HOW BIG THE MOTION IS, said in a way that means something.
     *
     * "0.173 Å" is a number, not a fact anyone can picture — is that a tremble or
     * is the molecule coming apart?  What answers it is the yardstick every
     * chemist already carries: a bond.  A swing of 0.17 Å is a tenth of a C–C
     * bond, and stating it that way is the difference between a readout and a
     * measurement.
     *
     * So this finds the atom that moves most, how far it goes, and the distance
     * to its own nearest neighbour — which for any atom in a molecule is the bond
     * it is attached by.  The caller reads the swing as a fraction of that.
     *
     * The furthest atom, not an average, because it bounds the picture: every
     * other atom in the mode moves less than this.  The amplitude multiplies the
     * eigenvector row, so the pairing in force (display or canonical, § 12.2)
     * carries through without this needing to know which one it was.
     */
    function _swingReport(inputs) {
        const rows  = inputs.mode.displacements;
        const basis = Array.isArray(inputs.mode.basis) ? inputs.mode.basis : null;
        const out   = { swing: 0, element: null, bond: null, neighbour: null };

        let worst = -1, max = 0;
        for (let k = 0; k < rows.length; k++) {
            const r = rows[k];
            const m = Math.sqrt(r[0] * r[0] + r[1] * r[1] + r[2] * r[2]);
            if (m > max) { max = m; worst = k; }
        }
        out.swing = inputs.amplitude * max;
        if (worst < 0) return out;

        // A row is indexed by MOVING atom; the basis names which structure atom
        // that is (vibrationview _maths.scatter).  Without a basis every atom
        // moves and the two indices are the same.
        const at  = basis ? Math.floor(Number(basis[worst])) : worst;
        const pos = inputs.structure.positions;
        const el  = inputs.structure.elements;
        if (!Array.isArray(pos[at])) return out;
        out.element = el[at] || null;

        let near = Infinity, nearAt = -1;
        for (let j = 0; j < pos.length; j++) {
            if (j === at || !Array.isArray(pos[j])) continue;
            const dx = pos[j][0] - pos[at][0];
            const dy = pos[j][1] - pos[at][1];
            const dz = pos[j][2] - pos[at][2];
            const d  = Math.sqrt(dx * dx + dy * dy + dz * dz);
            if (d < near) { near = d; nearAt = j; }
        }
        if (nearAt >= 0 && isFinite(near) && near > 0) {
            out.bond      = near;
            out.neighbour = el[nearAt] || null;
        }
        return out;
    }


    /* ── Saving the animation (vibrationview.md § 12) ───────────────────────
     *
     * The module produces BYTES; where they go is the page's business, and here
     * that is a download.  The viewer is asked for a picture of what is on screen
     * at a size and background the user picked, and for nothing else — the maths,
     * the amplitude and the rate are whatever the animation is already using, so
     * the file cannot disagree with the screen.
     */
    async function onExportAnimation() {
        if (state.exporting) return;
        // NOT guarded on having a viewer: a button that silently does nothing is
        // worse than one that says why.  With no mode showing, the export door
        // answers "no mode is showing, so there is nothing to export", and the
        // catch below puts that where the user is looking.
        if (!state.vib) {
            setStatus(els.animExportStatus,
                      "no mode is showing, so there is nothing to export", "muted");
            return;
        }
        const controller = new AbortController();
        state.exporting = controller;
        if (els.animExportBtn)    els.animExportBtn.disabled = true;
        if (els.animExportCancel) els.animExportCancel.hidden = false;

        const px = (el, fallback) => {
            const v = parseInt(el && el.value, 10);
            return Number.isFinite(v) && v > 0 ? v : fallback;
        };
        try {
            const out = await state.vib.exportAnimation({
                format:     (els.animExportFormat && els.animExportFormat.value) || "png-zip",
                width:      px(els.animExportWidth, 1600),
                height:     px(els.animExportHeight, 1200),
                background: (els.animExportBackground && els.animExportBackground.value) || undefined,
                cycles:     px(els.animExportCycles, 1),
                signal:     controller.signal,
                onProgress: (fraction, label) => {
                    setStatus(els.animExportStatus,
                              Math.round(fraction * 100) + "% — " + label, "muted");
                },
            });
            _download(out.blob, out.filename);
            // Say what was actually saved. The amplitude means nothing without
            // the normalization beside it (§ 12.2), so both are shown or neither.
            setStatus(els.animExportStatus,
                      "saved " + out.filename + " — " + out.meta.frames + " frames, "
                      + out.meta.normalization, "ok");
        } catch (e) {
            // A cancel is the user getting what they asked for, so it is not an
            // error and should not be dressed as one.
            const msg = (e && e.message) || "the export failed";
            setStatus(els.animExportStatus, msg,
                      msg.indexOf("cancel") >= 0 ? "muted" : "error");
        } finally {
            state.exporting = null;
            if (els.animExportBtn)    els.animExportBtn.disabled = false;
            if (els.animExportCancel) els.animExportCancel.hidden = true;
        }
    }

    function onExportCancel() {
        if (state.exporting) state.exporting.abort();
    }

    /* The page owns the destination, so the page makes the link. The module never
     * touches one: it hands back bytes and a suggested name (§ 12). */
    function _download(blob, filename) {
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        a.remove();
        // Revoke on the next turn: revoking synchronously can beat the click.
        setTimeout(() => URL.revokeObjectURL(url), 0);
    }

    /* "91% C, 9% H" from {"C": 0.914, "H": 0.086}.
     *
     * BIGGEST FIRST, sorted here rather than trusted from the wire: the server
     * builds the object in that order, but JSON serialisation sorts keys
     * alphabetically, so C would lead H whatever the physics said.
     *
     * Anything under 1% is dropped rather than printed as "0%", which is what
     * rounding would make of the sulphur that barely moves in a C–H stretch.  A
     * dropped element is a truer statement than a zero: it says the mode is not
     * about that atom, which is exactly what a 0.2% share means.
     */
    function _sayComposition(share) {
        if (!share || typeof share !== "object") return "";
        const parts = Object.keys(share)
            .map(el => ({ el: el, pct: Math.round(100 * Number(share[el])) }))
            .filter(p => p.pct >= 1)
            .sort((a, b) => b.pct - a.pct)
            .map(p => p.pct + "% " + p.el);
        return parts.join(", ");
    }

    /* THE LINE UNDER THE VIEWER, in words rather than symbols.
     *
     * A quantity that does not say what it measured cannot be used, so the
     * line is a sentence, and each clause answers one question (spectra.md
     * § 4.2):
     *
     *   the motion is 91% C, 9% H · nothing moves further than 0.173 Å from
     *   rest, 16% of that atom's bond · drawn exaggerated
     *   └── whose mode ──────────┘   └──── how big, against a yardstick ───┘
     *                                                        └ is it real? ┘
     *
     * WHOSE MODE, FIRST.  A reader wants to know they are looking at a ring
     * stretch before they are told how far it swings.
     *
     * A BOUND, NOT A SUBJECT.  The size clause says "nothing moves further than"
     * rather than naming the busiest atom, and that wording is deliberate: the
     * furthest-moving atom is a hydrogen in almost every mode (32 of the 36 in
     * the BDT result this was read against), because a light atom travels
     * furthest for the same energy.  "H moves 0.17 Å" therefore reads as a claim
     * that the mode is a hydrogen motion -- which mode 30 disproves, being 91%
     * carbon while its hydrogens move furthest.  What the number honestly is, is
     * a ceiling: every atom in the picture stays inside it.
     *
     * FROM REST: to one extreme, not the sweep between them, so an atom covers
     * twice this between extremes.  THE YARDSTICK: a percentage of that atom's
     * own bond, because 0.17 Å is a number and "a sixth of a bond" is a picture
     * -- its own bond, since a C–H bond is short and an Au–Au contact is long.
     * IS IT REAL: exaggerated is a drawing convention and the two physical
     * settings are measurements, which is the one thing a reader must not get
     * wrong about a number quoted from this panel.
     */
    function _reportSwing(inputs) {
        if (!els.viewerStatus) return;
        const r = _swingReport(inputs);

        let text = "";
        let why  = "";

        /* WHAT THE MODE IS, before how big it is drawn -- and the order is the
         * point.  A reader wants to know they are looking at a ring stretch
         * before they are told how far it swings. */
        const composition = _sayComposition(inputs.mode.share);
        if (composition) {
            text += "the motion is " + composition + " · ";
            why  += "Each element's share of the mode: its part of the "
                  + "mass-weighted motion (mᵢ|Lᵢ|² over the total), which is how "
                  + "a mode is assigned to the atoms it belongs to. ";
        }

        text += "nothing moves further than " + r.swing.toFixed(3)
              + " Å from rest";
        why  += "The distance is a ceiling over every atom: the furthest-moving "
              + "one gets this far from its rest position and all the others move "
              + "less. It is measured to one extreme of the swing, so an atom "
              + "covers twice this between extremes.";

        if (r.bond) {
            const share = 100 * r.swing / r.bond;
            text += ", " + (share < 1 ? "<1" : Math.round(share))
                  + "% of that atom's "
                  + (r.bond <= BOND_LIKE_ANG ? "bond" : "nearest contact");
            why  += " The percentage compares it with that atom's own "
                  + "nearest-neighbour distance — the bond it hangs from — "
                  + "because a displacement means nothing except beside a bond "
                  + "length. The atom is deliberately not named: it is usually a "
                  + "hydrogen whatever the mode is, since the lightest atom "
                  + "travels furthest for a given energy, so naming it would "
                  + "suggest the mode belongs to it when it does not.";
        }

        text += " · " + inputs.saidPlainly;
        setStatus(els.viewerStatus, text, "muted");
        els.viewerStatus.title = why;
    }

    /* THE THREE VIEWS OF ONE SELECTION.
     *
     * The modes table, the mode animation and the electronic structure all
     * describe the same selected mode.  As tabs they share one position, and
     * the selection is what moves.
     *
     * SELECTING A MODE DOES NOT SWITCH TAB.  All three update underneath; the
     * reader stays where they were looking.  A click that yanks the view away is
     * the same as losing your place.
     *
     * WHY THIS IS NOT JUST TOGGLING `hidden`.  A box inside a hidden panel has
     * no size, and both a 3-D canvas and a Plotly figure take their size FROM
     * their box.  Mounted or drawn while their tab was hidden, they come back
     * with a zero-size drawing surface.  So becoming visible is an event, and each view is
     * told to re-measure: the viewer re-fits its camera to the box
     * (vibrationview.md § 8 `refit`), the chart re-runs Plotly's resize.
     */
    const MODE_TABS = ["table", "viewer", "es", "thermo"];

    function _activateModeTab(name) {
        if (MODE_TABS.indexOf(name) === -1) return;
        state.modeTab = name;
        for (const t of MODE_TABS) {
            const btn   = document.getElementById("mode-tabbtn-" + t);
            const panel = document.getElementById("mode-tab-" + t);
            const on    = (t === name);
            if (btn) {
                btn.classList.toggle("is-active", on);
                btn.setAttribute("aria-selected", on ? "true" : "false");
            }
            if (panel) panel.hidden = !on;
        }
        // Now that the box has a size again, let what draws into it catch up.
        if (name === "viewer" && state.vib) {
            try { state.vib.refit(); } catch (_) {}
        }
        if (name === "es" && typeof Plotly !== "undefined" && els.esBarDiagram) {
            try { Plotly.Plots.resize(els.esBarDiagram); } catch (_) {}
        }
        if (name === "thermo" && typeof Plotly !== "undefined") {
            for (const el of [els.thermoCurves, els.thermoDecomp]) {
                if (el) { try { Plotly.Plots.resize(el); } catch (_) {} }
            }
        }
    }

    function onModeTabClick(ev) {
        const btn = ev.target.closest ? ev.target.closest("[data-mode-tab]") : null;
        if (btn) _activateModeTab(btn.dataset.modeTab);
    }

    function _stopAnimation() {
        // Not the user's doing -- there is simply nothing to animate -- so the
        // Pause INTENT is left alone and the motion resumes when a mode returns.
        if (state.vib && typeof state.vib.pause === "function") {
            try { state.vib.pause(); } catch (_) {}
        }
    }

    function onAnimAmplitudeChange() {
        const v = parseFloat(els.animAmplitude.value);
        if (Number.isFinite(v)) state.animAmplitude = v;
        if (els.animAmplitudeVal)
            els.animAmplitudeVal.textContent = v.toFixed(2) + " Å";
        // A live knob (vibrationview.md § 9.2): a plain write the running loop
        // reads on its next frame, so dragging never stops the animation.
        if (state.vib) { try { state.vib.setAmplitude(v); } catch (_) {} }
    }
    /* Switching between the two ways of asking "how big" (§ 12.2).
     *
     * The slider disappears for the physical pairings, because there is nothing
     * to slide: the size follows from the frequency.  What replaces it is a
     * readout of what that size came out as, which is the number a caption would
     * have to quote. */
    function onAmplitudeModeChange() {
        state.animAmplitudeMode = els.animAmplitudeMode.value || "display";
        const physical = state.animAmplitudeMode !== "display";
        if (els.animAmplitudeRow)  els.animAmplitudeRow.hidden  = physical;
        if (els.animTemperatureRow)
            els.animTemperatureRow.hidden = state.animAmplitudeMode !== "thermal";
        renderModeViewer();
    }

    /* Only the SIZE changes with temperature -- the eigenvector is the same one
     * -- so this is a live amplitude write and not a new mode.  Re-showing the
     * mode would restart the cycle from its peak, which on a number box means
     * restarting once per keystroke while somebody types "298". */
    function onTemperatureChange() {
        const t = parseFloat(els.animTemperature.value);
        if (!Number.isFinite(t) || t <= 0) return;
        state.animTemperature = t;
        const inputs = _animationInputs();
        if (!inputs.ready || !state.vib) return;
        // The size and the label naming its setting go together, so a saved
        // animation says the temperature it was drawn at (vibrationview.md
        // § 6.1).  A drawing aid either way (§ 12.2), not a physical claim.
        state.vib.setAmplitude(inputs.amplitude, inputs.norm);
        _reportSwing(inputs);
    }

    /* Speed sets how long ONE OSCILLATION takes, not a frame rate: a cycle is a
     * second by default, so 2x is half a second (vibrationview.md § 10.1).
     * Smoothness is a separate knob the viewer owns, which is why asking for a
     * faster wobble here cannot make it stutter. */
    function onAnimSpeedChange() {
        const v = parseFloat(els.animSpeed.value);
        if (!Number.isFinite(v) || v <= 0) return;
        // ONE home: the multiplier the slider shows and the preferences persist.
        // How long a cycle takes is 1/that, worked out where it is handed over
        // rather than stored beside it as a second copy that can drift.
        state.animSpeed = v;
        if (els.animSpeedVal) els.animSpeedVal.textContent = v.toFixed(1) + "×";
        if (state.vib) { try { state.vib.setCycleSec(1 / v); } catch (_) {} }
    }
    function onAnimToggle() {
        if (!state.vib) return;
        // The button sets the INTENT; the viewer follows it.  Toggling off what
        // the viewer happens to be doing would make the button mean "resume
        // whatever state you drifted into" rather than "I want this stopped".
        state.animPaused = !state.animPaused;
        try {
            if (state.animPaused) state.vib.pause();
            else                  state.vib.play();
        } catch (_) {}
        _syncPlayButton();
    }

    // ----- Spectrum chart --------------------------------------
    //
    // Draws frequency (cm⁻¹) against EVERY channel the run computed --
    // one stacked panel each, sharing the frequency axis (the chart's
    // own contract, docs/web/spectrumchart.md).
    /* THE SPECTRUM CHART IS A MODULE.
     *
     * The traces, the palette, the envelope, the click tolerance and the
     * width watcher live behind one door in
     * lib/spectrumchart/, whose contract is docs/web/spectrumchart.md. What is
     * left in this file is what the CONTRACT says belongs to a tab: the modes,
     * the selection, and the broadening the user typed.
     *
     * The mount is asynchronous and happens once. `chartReady` is the promise of
     * it, so callers can queue work against a chart that is still arriving
     * without any of them having to know whether it has.
     */
    let chart = null;
    let chartReady = null;

    /* Do something with the chart once it exists, whether it exists yet or not. */
    function _withChart(fn) {
        if (chart) { fn(chart); return; }
        if (chartReady) chartReady.then(handle => { if (handle) fn(handle); });
    }

    /* The two channels the chart draws, declared once.  Raman in the upper
     * panel, IR ("down") in the lower one: both peak at the same
     * frequencies, so one panel would make them collide. */
    /* THE IR AXIS IS AN ABSORPTION INDICATOR ON AN ARBITRARY SCALE, and
     * that is the honest label rather than a compromise.
     *
     * An IR spectrum is read as absorption, not as the km/mol a
     * calculation reports.  A TRUE absorbance or transmittance axis is
     * Beer-Lambert -- it needs a concentration and a path length, which
     * a computed gas-phase spectrum does not have, so any calibrated
     * axis here would encode a sample nobody measured.  "Arbitrary
     * units" is the convention that says exactly this, and it is what
     * every uncalibrated computed spectrum in the literature carries:
     * the band POSITIONS and their RELATIVE heights are the result; the
     * absolute scale is not claimed.
     *
     * So the lane is normalised to its own strongest band and the axis
     * says arb. units.  The measured km/mol is untouched and still
     * appears in the readout, the modes table and the CSV export --
     * nothing about the number is lost, only the pretence that the
     * height means a transmittance. */
    const CHART_CHANNELS = [
        { key: "raman", label: "Raman activity",  unit: "Å⁴/amu", direction: "up" },
        { key: "ir",    label: "IR absorption",   unit: "arb. units",
          direction: "down", relative: true },
    ];

    function _chartModes(modes) {
        // The tab's record in the shape the chart takes: one value per
        // CHANNEL, plus the activity class decided server-side.  `cls` is
        // read, never computed here -- the threshold that separates a
        // band from numerical residue lives in spectra/activity.py, and a
        // second copy of it in the viewer could not be tested against a
        // real run.
        return (modes || []).map(m => ({
            index:     m.index_1based,
            freq:      m.frequency_cm1,
            values: {
                raman: Number.isFinite(m.raman_activity_a4_amu)
                    ? m.raman_activity_a4_amu : null,
                ir: Number.isFinite(m.ir_intensity_km_mol)
                    ? m.ir_intensity_km_mol : null,
            },
            cls:       m.activity_class || "partial",
            imaginary: !!m.has_imag,
        }));
    }

    // ----- Thermochemistry panel (v5 `thermo`; plan § 2b) -------
    //
    // The DECK computes, this panel draws, and it DERIVES NOTHING
    // (web/spectra.md § 3): the headline, the regime sentence and the
    // T-grid arrays are read off `results.thermo`, and the electronic
    // reference is the file's own equilibrium energy -- PySCF's SCF
    // energy, zero on a route that reports none (SIESTA), whose numbers
    // are then the vibrational contributions alone.
    //
    // THE LABELS FOLLOW THE REGIME: `rrho` is the full gas-phase answer
    // (H, S, G); `vibrational-only` -- any atom held, and every SIESTA
    // result -- is the vibrational part (ZPE + U_vib, S_vib, F_vib), and
    // no pressure enters it (engines/vibration.md § 4.7).
    //
    // Tab-owned Plotly, same rules as the ES level diagram: colours
    // from _esTheme()'s CSS tokens, NO `height` in the layout (the
    // .thermo-chart box owns it), and a plain degrade when Plotly is
    // not on the page (/results loads it; /spectra does not mount
    // this panel at all).
    // Hartree in kcal/mol is `K.hartree_kcal_mol`, served with the
    // results (see `K`).

    function renderThermoPanel(results) {
        const th   = (results && results.thermo) || {};
        const grid = th.grid || {};
        const T    = grid.temperatures_K || [];
        const has  = T.length > 0;
        if (els.thermoTabBtn) els.thermoTabBtn.hidden = !has;
        if (!has) {
            // A v4 file, or a run that has not reached the thermo
            // stage yet.  If the user was ON the tab when such a run
            // loaded, land them somewhere real.
            if (state.modeTab === "thermo") _activateModeTab("table");
            return;
        }
        const rrho = th.regime === "rrho";
        const eq = results.equilibrium || {};
        const eElec = Number.isFinite(eq.scf_energy_eh) ? eq.scf_energy_eh : 0;
        const kcal = (eh) => eh * K.hartree_kcal_mol;
        const calK = (ehk) => ehk * K.hartree_kcal_mol * 1000.0;

        // --- The words: headline numbers + the writer's own note --------
        if (els.thermoNote) {
            const bits = [];
            const zpe = "ZPE " + Number(th.zpe_eh).toFixed(6) + " Eh ("
                      + kcal(th.zpe_eh).toFixed(2) + " kcal/mol)";
            if (rrho) {
                bits.push("At T = " + th.temperature_K + " K, P = "
                    + th.pressure_atm + " atm (full RRHO: electronic + "
                    + "translational + rotational + vibrational): " + zpe
                    + (th.g_eh != null
                       ? "; H − E_elec " + kcal(th.h_eh - eElec).toFixed(2)
                         + " kcal/mol; S " + calK(th.s_eh_k).toFixed(2)
                         + " cal/mol/K; G − E_elec "
                         + kcal(th.g_eh - eElec).toFixed(2) + " kcal/mol"
                       : "") + ".");
            } else {
                bits.push("At T = " + th.temperature_K + " K (vibrational "
                    + "contributions only — no pressure enters them): " + zpe
                    + (th.g_eh != null
                       ? "; ZPE + U_vib " + kcal(th.h_eh - eElec).toFixed(2)
                         + " kcal/mol; S_vib " + calK(th.s_eh_k).toFixed(3)
                         + " cal/mol/K; F_vib = ZPE + U_vib − T·S_vib "
                         + kcal(th.g_eh - eElec).toFixed(2) + " kcal/mol"
                       : "") + ".");
            }
            if (th.note) bits.push(String(th.note) + ".");
            els.thermoNote.textContent = bits.join("  ");
        }
        if (typeof Plotly === "undefined") {
            if (els.thermoCurves) {
                els.thermoCurves.textContent =
                    "(chart library unavailable on this page)";
            }
            return;
        }

        const t = _esTheme();
        const hRel = grid.h_eh.map((h) => kcal(h - eElec));
        const gRel = grid.g_eh.map((g) => kcal(g - eElec));
        const ts   = T.map((Ti, i) => kcal(Ti * grid.s_eh_k[i]));
        const config = {
            displaylogo: false, responsive: true,
            modeBarButtonsToRemove: ["select2d", "lasso2d",
                                     "toggleSpikelines"],
        };
        Plotly.react(els.thermoCurves, [
            { x: T, y: gRel, name: rrho ? "G − E_elec" : "F_vib",
              mode: "lines", line: { color: t.homo, width: 2 } },
            { x: T, y: hRel, name: rrho ? "H − E_elec" : "ZPE + U_vib",
              mode: "lines", line: { color: t.stick, width: 2 } },
            { x: T, y: ts, name: rrho ? "T·S" : "T·S_vib", mode: "lines",
              line: { color: t.lumo, width: 2 } },
        ], {
            // NO height -- the .thermo-chart box owns it.
            margin: { t: 8, r: 8, b: 34, l: 52 },
            plot_bgcolor: t.paper, paper_bgcolor: t.paper,
            font: { color: t.ink, size: 10 },
            showlegend: true,
            legend: { orientation: "h",
                      font: { color: t.ink, size: 10 } },
            xaxis: { title: { text: "T (K)", font: { size: 10 } },
                     gridcolor: t.grid, zeroline: false },
            yaxis: { title: { text: rrho ? "kcal/mol above E_elec"
                                         : "kcal/mol, vibrational contributions",
                              font: { size: 10 } },
                     gridcolor: t.grid, zeroline: false },
        }, config);

        // --- The decomposition at the grid point nearest the headline T --
        // Every bar is read off the grid; the parts sum to the last bar.
        let i0 = 0, dmin = Infinity;
        for (let i = 0; i < T.length; i++) {
            const d = Math.abs(T[i] - th.temperature_K);
            if (d < dmin) { dmin = d; i0 = i; }
        }
        const zpe  = kcal(grid.zpe_eh[i0]);
        const uvib = kcal(grid.u_vib_eh[i0]);
        const mts  = -kcal(T[i0] * grid.s_eh_k[i0]);
        const gnet = kcal(grid.g_eh[i0] - eElec);
        // Full RRHO: what is left of H above E_elec, ZPE and U_vib is the
        // translational and rotational energy and pV -- read, not assumed.
        const rest = kcal(grid.h_eh[i0] - eElec) - zpe - uvib;
        const bars = rrho
            ? { x: ["ZPE", "U_vib", "trans + rot + pV", "−T·S", "G − E_elec"],
                y: [zpe, uvib, rest, mts, gnet],
                c: [t.stick, t.homo, t.axis, t.lumo, t.stickSel] }
            : { x: ["ZPE", "U_vib", "−T·S_vib", "F_vib"],
                y: [zpe, uvib, mts, gnet],
                c: [t.stick, t.homo, t.lumo, t.stickSel] };
        Plotly.react(els.thermoDecomp, [{
            type: "bar", x: bars.x, y: bars.y, marker: { color: bars.c },
        }], {
            margin: { t: 8, r: 8, b: 34, l: 52 },
            plot_bgcolor: t.paper, paper_bgcolor: t.paper,
            font: { color: t.ink, size: 10 },
            showlegend: false,
            xaxis: { gridcolor: t.grid, zeroline: false },
            yaxis: { title: { text: "kcal/mol (at "
                              + Math.round(T[i0]) + " K)",
                              font: { size: 10 } },
                     gridcolor: t.grid, zeroline: true,
                     zerolinecolor: t.axis },
        }, config);
    }

    function renderSpectrumChart(modes) {
        if (!els.spectrumChart) return;
        if (!chartReady) {
            chartReady = import("/static/lib/spectrumchart/index.js")
                .then(({ mount }) => mount(els.spectrumChart, {
                    // Declared at mount because they decide the LAYOUT --
                    // how many panels, and which one each occupies.
                    channels: CHART_CHANNELS,
                    // A click enters the tab and comes back as setSelected;
                    // the chart never highlights on its own.
                    onSelect: (index) => selectMode(index),
                }))
                .then((handle) => {
                    if (!handle.ok) {
                        showNotice(els.spectrumChart, handle.error, "muted");
                        return null;
                    }
                    chart = handle;
                    return handle;
                })
                .catch((err) => {
                    showNotice(els.spectrumChart, "spectrum chart unavailable: "
                               + (err && err.message ? err.message : err),
                               "muted");
                    return null;
                });
        }
        chartReady.then((handle) => {
            if (!handle) return;
            handle.setModes(_chartModes(modes));
            handle.setBroadening(state.broadeningFWHM || 0);
            handle.setSelected(state.selectedMode == null ? null : state.selectedMode);
        });
    }


    function onBroadeningChange() {
        const raw = parseFloat(els.broadeningFwhm.value);
        const v = Number.isFinite(raw) ? Math.max(0, raw) : 0;
        state.broadeningFWHM = v;
        if (state.results && _anyStrength(state.results)) {
            renderSpectrumChart(state.results.modes || []);
        }
    }

    // Refresh-button listener wiring (contract § 5).  Mirrors
    // trajectory's _wireRefreshListener: wired ONCE at mount; not
    // re-wired per-load.  The Refresh button does what § 5 says:
    // file-switch with the current path.
    function _wireRefreshListener() {
        const C = (window.molbuilder || {}).constants;
        if (!C || !C.EVENT_REFRESH_REQUESTED) return;
        // Route through the _on() helper so dispose() tears this down
        // with every other listener registered in this module.  Pinned by
        // tests/spectra/test_blueprint.py::
        // TestSpectraDisposeContract::
        // test_all_element_listeners_route_through_on_helper.
        _on(document, C.EVENT_REFRESH_REQUESTED, () => {
            const p = state.fileState.path;
            if (!p) return;     // not yet loaded; nothing to refresh
            loadByPath(p);
        });
    }

    // ----- Bootstrap -------------------------------------------
    function init() {
        els.formContainer  = $("spectra-form-container");
        els.form.pyscf     = $("spectra-form-pyscf");
        els.form.siesta    = $("spectra-form-siesta");
        els.engineStrip    = $("spectra-engine-strip");
        els.engineNote     = $("spectra-engine-note");
        els.sendBtn        = $("send-to-task-setup");
        els.sendStatus     = $("send-status");
        els.preflightPanel = $("spectra-issues");
        els.resultsSummary = $("results-summary");
        els.resultsMeta    = $("results-summary-list");
        els.methodsBlock   = $("methods-block");
        els.methodsText    = $("methods-text");
        els.methodsCopy    = $("methods-copy");
        els.methodsNote    = $("methods-copy-note");
        els.displayFloor   = $("display-floor");
        els.displayFloorOut = $("display-floor-out");
        els.modesTbody     = $("modes-tbody");
        els.spectrumChart  = $("spectrum-chart");
        els.spectrumSection = $("spectrum-section");
        els.spectrumHeading = $("spectrum-heading");
        els.spectrumControls = $("spectrum-controls");
        els.spectrumAbsent = $("spectrum-absent");
        els.modesTable     = $("modes-table");
        // Mode-table interactions + ES panel.
        els.modesFilter       = $("modes-filter");
        els.modesCsvBtn       = $("modes-csv-btn");
        els.modesFilterCount  = $("modes-filter-count");
        els.modesTheadRow     = $("modes-thead-row");
        els.esPanel           = $("es-panel");
        els.esModeIdx         = $("es-mode-idx");
        els.esModeFreq        = $("es-mode-freq");
        els.esBarDiagram      = $("es-bar-diagram");
        els.thermoTabBtn      = $("mode-tabbtn-thermo");
        els.thermoNote        = $("thermo-note");
        els.thermoCurves      = $("thermo-curves");
        els.thermoDecomp      = $("thermo-decomp");
        els.esSummary         = $("es-summary");
        // The run's progress: the status line and the phase dots.
        els.watchStatus       = $("watch-status");
        els.phaseIndicator    = $("phase-indicator");
        els.broadeningFwhm    = $("broadening-fwhm");
        // 3D mode-animation viewer.
        els.modeViewerWrap    = $("mode-viewer-wrap");
        // The tab strip: one listener on the strip, not three on the buttons.
        // Found by id like every other element here -- a class query would reach
        // across the whole document, and this inspector is mounted INTO a page
        // that may hold other panels.
        els.modeTabs          = $("mode-tabs");
        els.modeViewer        = $("mode-viewer");
        els.viewerStatus      = $("viewer-status");
        els.animAmplitude     = $("anim-amplitude");
        els.animAmplitudeVal  = $("anim-amplitude-val");
        els.animSpeed         = $("anim-speed");
        els.animSpeedVal      = $("anim-speed-val");
        els.animToggle        = $("anim-toggle");
        els.animAmplitudeMode  = $("anim-amplitude-mode");
        els.animAmplitudeRow   = $("anim-amplitude-row");
        els.animTemperature    = $("anim-temperature");
        els.animTemperatureRow = $("anim-temperature-row");
        els.animExportFormat    = $("anim-export-format");
        els.animExportWidth     = $("anim-export-width");
        els.animExportHeight    = $("anim-export-height");
        els.animExportBackground = $("anim-export-background");
        els.animExportCycles    = $("anim-export-cycles");
        els.animExportBtn       = $("anim-export-btn");
        els.animExportCancel    = $("anim-export-cancel");
        els.animExportStatus    = $("anim-export-status");

        // --- Generate-side wiring (only present on /spectra) -------
        //
        // The /results inspector partial mounts only the inspect-side
        // ids (results table, mode viewer, ES panel);
        // generate-side ids (form container, Send to Task setup)
        // live only in spectra.html.  Gate the
        // whole generate block on formContainer's presence so a
        // /results-side mount stays a clean inspect-only inspector.
        const hasGenerateSide = Boolean(els.formContainer);
        if (hasGenerateSide) {
            _on(els.sendBtn, "click", sendToTaskSetup);
            // Live science check on every edit.
            _on(els.formContainer, "input",  refreshPreflightDebounced);
            _on(els.formContainer, "change", refreshPreflightDebounced);

        }

        // --- Inspect-side wiring -----------------------------------
        //
        // Gated on the partial's Results panel.  Post-step 2.5 /spectra
        // drops the inspect-side partial entirely (the page becomes
        // generate-only), so this whole block must no-op when none of
        // the inspect-side ids exist; the same module mounts cleanly
        // into either consumer.
        const hasInspectSide = Boolean(els.resultsSummary);
        if (hasInspectSide) {
            // FWHM-controlled broadening re-renders the chart in
            // place.
            /* THE METHODS COPY BUTTON AND THE DISPLAY FLOOR ARE
             * INSPECT-SIDE: both controls live in the RESULTS panel. */
            _on(els.methodsCopy, "click", copyMethodsText);
            if (els.displayFloor) {
                _on(els.displayFloor, "input", onDisplayFloor);
                onDisplayFloor();   // adopt whatever the markup starts at
            }

            if (els.broadeningFwhm) {
                _on(els.broadeningFwhm, "input", onBroadeningChange);
                // Read initial value from the input so an
                // HTML-default-modified value (sessionStorage etc.)
                // propagates without needing a manual edit.
                const v = parseFloat(els.broadeningFwhm.value);
                if (Number.isFinite(v) && v >= 0) state.broadeningFWHM = v;
            }

            // 3D viewer control wiring.
            if (els.animAmplitude) {
                _on(els.animAmplitude, "input", onAnimAmplitudeChange);
                onAnimAmplitudeChange();
            }
            if (els.animSpeed) {
                _on(els.animSpeed, "input", onAnimSpeedChange);
                onAnimSpeedChange();
            }
            _on(els.animToggle, "click", onAnimToggle);
            if (els.animAmplitudeMode) {
                _on(els.animAmplitudeMode, "change", onAmplitudeModeChange);
                // Primed like the amplitude and speed handlers above: the markup
                // carries the starting value, and running the handler once makes
                // the state and the visible controls agree.  Without it a restored
                // "thermal" would leave the temperature box hidden and the size
                // slider showing -- the panel contradicting itself.
                onAmplitudeModeChange();
            }
            _on(els.animTemperature,   "input",  onTemperatureChange);
            _on(els.animExportBtn,      "click",  onExportAnimation);
            _on(els.animExportCancel,   "click",  onExportCancel);

            // Mode-table interactions.
            _on(els.modeTabs,      "click", onModeTabClick);
            _on(els.modesTheadRow, "click", onTableHeaderClick);
            _on(els.modesTbody,    "click", onTableRowClick);
            _on(els.modesFilter,   "input", onFilterInput);
            _on(els.modesCsvBtn,   "click", exportCSV);
        }

        if (hasGenerateSide) {
            initSchemaForm();
        }
    }

    init();

    // Put the view knobs back (`spectra.md` § 7).  Kicked here, once per
    // mount, for the reason ui-context is attached after the model exists:
    // the read is async and writes stay disarmed until it has finished, so a
    // default announced during init cannot overwrite the saved value on its
    // way in.  Nothing awaits it -- a slow read must not delay the form.
    _prefsRestore();

    // Contract § 5: wire Refresh ONCE at mount.  Mirrors
    // trajectory's _wireRefreshListener pattern.
    _wireRefreshListener();

    // The file the caller mounted us for -- /results passes the
    // dropdown's pick as opts.file, the one route to a file.  Guarded on
    // the inspect side because /spectra (generate-only after step 2.5)
    // mounts this module with no Results panel and passes no file.
    if (opts.file && els.resultsSummary) {
        loadByPath(opts.file);
    }

    // ---- pageshow / visibilitychange: force-refresh on tab re-entry //
    //
    // Same shape as the /results file-picker + trajectory
    // inspector: a bfcache restore or tab re-focus must
    // re-fetch the currently-loaded spectra file so a fresh result
    // generated in another tab actually appears, rather than the
    // cached snapshot from the previous visit.
    //
    // Guard on ``state.results !== null`` so a never-loaded inspector
    // doesn't fire spurious /api/spectra/load on every visibility
    // event; the file is the one on screen, `fileState.path`.
    function _onPageShow(_evt) {
        if (state.results !== null && state.fileState.path) {
            loadByPath(state.fileState.path);
        }
    }
    function _onVisibilityChange(_evt) {
        if (document.visibilityState === "visible"
            && state.results !== null && state.fileState.path) {
            loadByPath(state.fileState.path);
        }
    }
    _on(window,   "pageshow",          _onPageShow);
    _on(document, "visibilitychange",  _onVisibilityChange);

    // The handle the caller uses to dispose the mounted inspector.
    // /results' registry calls dispose() before mounting the next
    // inspector; /spectra's bootstrap holds it for completeness
    // but never disposes (the tab lives forever).
    return {
        /**
         * Tear down every long-lived resource this mount created: the
         * live-watch poller, the VibrationView mode viewer (its animation
         * loop + canvas, via state.vib.dispose()), the spectrum chart (which
         * takes its own surface, watcher and markup down, via chart.dispose())
         * and the level diagram's Plotly figure, which is this tab's own.
         * After dispose() the rootEl's contents are no longer owned by the
         * inspector; caller may clear/replace freely.
         */
        dispose() {
            // Hand the listener scope back (lib/inspectors/lifecycle.js).
            // It tears down in reverse, so the most recent registration goes
            // first -- the order they would be re-attached in on a remount.
            _listeners.disposeAll();
            // Contract § 2: dispose -> transition('IDLE').  Single
            // canonical site for the full reset matrix § 3 row
            // "dispose / unmount" (aborts loadAbort +
            // watchAbort, stops watchTimer, clears watchInFlight,
            // clears fileState + viewState, sets machine='IDLE').
            transition("IDLE");
            // The viewer is an external resource rather than bucket state, so
            // it is torn down explicitly: its own dispose stops the clock and
            // releases the drawing surface (vibrationview.md § 8).
            if (state.vib) {
                try { state.vib.dispose(); } catch (_) {}
                state.vib = null;
            }
            /* The spectrum chart takes itself down: one call, and its surface,
             * its box watcher and its markup go with it.  This tab neither
             * purges it nor knows what it was drawn with. */
            if (chart) { try { chart.dispose(); } catch (_) {} }
            chart = null;
            chartReady = null;
            /* The level diagram is still this tab's own figure, and a
             * purged-but-still-observed node leaks an observer per mount -- the
             * inspector is mounted and disposed every time the user switches
             * result files. */
            if (typeof Plotly !== "undefined" && els.esBarDiagram) {
                try { Plotly.purge(els.esBarDiagram); } catch (_) {}
            }
            if (state.esResizeObserver) {
                try { state.esResizeObserver.disconnect(); } catch (_) {}
                state.esResizeObserver = null;
            }
        },
        /**
         * Swap the displayed spectra results to ``path`` without
         * re-mounting the inspector.  Mirrors lib/trajectory/core.js's
         * load(path) so the /results registry can hot-swap files
         * between dispatch ticks instead of dispose → remount.  The
         * same door as the first load (POST /api/spectra/load, abort +
         * render + status update, and a run still going followed).
         */
        load(path) {
            return loadByPath(path);
        },

        /* THE VIEWER THIS PAGE MOUNTED, handed over by the page that mounted
         * it.  On the handle, because the viewer it stores is per-mount state:
         * `_viewer` lives inside this function, so a module-level door would be
         * writing into whichever mount happened to run last. */
        useViewer: useViewer,
        /* The engine strip's door, for the page that mounted this: which
         * engine is active, set it, tell the inspector a structure landed
         * (so the structure can pick the default), and hand over what the
         * chemistry card answers for -- the forms, and the structure the
         * tab would hand over.  The page never reaches into the containers
         * (overview.md § 1). */
        activeEngine:    _activeEngine,
        setEngine:       setEngine,
        structureLoaded: structureLoaded,
        stateForms:      stateForms,
        structureForRequest: structureForRequest,
    };

    }   // ----- end of mountInspector(rootEl, opts) -----

    // The free-row -> global-atom eigenvector scatter (`web/spectra.md` § 8)
    // belongs to VibrationView and is reached only through its one door: this file
    // hands over a mode and the module reads its own basis (vibrationview.md § 6.3).

    // Export for both consumers (spectra/viewer.js bootstrap on
    // /spectra, lib/inspectors/spectra.js on /results).  Each
    // consumer is responsible for picking when + where to mount;
    // this module does NOT self-bootstrap on page load.
    root.molbuilder = root.molbuilder || {};
    root.molbuilder.spectraInspector = {
        mount: mountInspector,
        // `useViewer` is on what `mount` RETURNS, not here: it stores the
        // viewer for one mounted inspector, and this object is shared by all of
        // them.  (No structure setter either: the structure is read off the
        // viewer the page mounted, so there is no in-memory holder to feed.)
    };

})(typeof window !== "undefined" ? window : this);
