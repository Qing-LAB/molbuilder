/* Spectra inspector core -- VibrationView mode viewer + Plotly chart + form.
 *
 * THE shared spectra-inspector implementation.  Two consumers:
 *
 *   * /spectra  -- the legacy single-purpose page (via
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
 *   (the old schema/spectra + spectra/render routes retired at P3)
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
     *   opts   -- reserved for future use (e.g. ``{file?: string}``
     *             to auto-load a results JSON on /results-side
     *             mount).
     *
     * Returns a handle ``{ dispose() }`` so the registry can tear
     * down timers + Plotly listeners between inspector swaps.
     * /spectra's bootstrap holds the handle for completeness but
     * never disposes (the tab lives for the lifetime of the page).
     */
    // Module-level holder for the structure bytes the user loaded
    // via the Inspect-structure card's "Load from sidebar selection"
    // button (task #309, 2026-06-09 follow-up to #296).  Pre-#309
    // the bytes were stored in a hidden ``<textarea id="structure-
    // text">``; then in a module-level ``_loadedStructureText`` holder.
    // BOTH are gone (2026-07): the structure is read from the viewer
    // OFF THE MODEL (molview.data) at call time -- no second copy.

    function mountInspector(rootEl, opts) {
    rootEl = rootEl || document;
    opts = opts || {};

    const $ = (id) => rootEl.querySelector("#" + id);

    // ----- DOM refs (resolved once at startup) -----------------
    const els = {
        spectrumChart:  null,
        // ``structureText`` slot removed 2026-06-10 along with the
        // ``$("structure-text")`` lookup in init() — the underlying
        // ``<textarea id="structure-text">`` was retired in task #309;
        // the structure is read off the viewer (getTheStructure ->
        // molview.data) at call time.
        // xyzFile / xyzLoadBtn / xyzStatus removed 2026-05-18: the
        // in-template <input type="file" id="xyz-file"> + sibling
        // load-button were dropped when the projects sidebar took
        // over file selection.  loadXyzFile() and its event-listener
        // wiring are gone too (init had to null-check ``els.xyzLoadBtn``
        // since the id no longer exists; the whole branch was dead).
        formContainer:  null,
        sendBtn:        null,
        sendStatus:     null,
        preflightPanel: null,
        // resultsFile / loadResultsBtn / resultsStatus removed for the
        // same reason as xyz* above; loadResults() also gone.  The
        // /api/spectra/load endpoint stays -- it's still used by the
        // path-based watch-path loader (els.loadPathBtn) below.
        resultsSummary: null,
        resultsMeta:    null,
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
        // Load / live-watch by server-side path:
        watchPath:      null,
        loadPathBtn:    null,
        watchBtn:       null,
        watchStopBtn:   null,
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
    // Bucketed state shape per docs/web/results.md
    // PR 3 mirrors the trajectory inspector's shape (PR 2 + PR 2.1 +
    // PR 2.2 + PR 2.3) for cross-inspector consistency.  Five disjoint
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
        schema:         null,
        lastJobName:    null,

        // Per contract § 2: IDLE / LOADING / LOADED / WATCHING / ERROR.
        machine: "IDLE",

        fileState: {
            // ``path`` was state.watchPath pre-PR-3.  The contract
            // § 7 spectra mapping settled on fileState.path being
            // canonical; the legacy `watchPath` name lives only as
            // a backward-compat alias.  Both Load and Start-watching
            // read els.watchPath.value into this field via
            // transition('LOADING').
            path:    null,
            // results: SpectraResults dict from /api/spectra/load.
            // Replaced atomically inside transition('APPLY').
            results: null,
        },

        viewState: {
            // 1-based index of the active mode, or null.  Survives
            // watchTick re-renders when the pick remains valid
            // (PR 2 audit follow-up D, 2026-06-17).
            selectedMode: null,
        },

        // uiPrefs: per-session knobs.  Contract § 3 reserves this
        // bucket for sessionStorage-persisted values.  Spectra
        // populates from day one (unlike trajectory which leaves
        // it empty per § 13).  TODO(PR 3.1?): wire the
        // sessionStorage roundtrip under key
        // `molbuilder.results.spectra.uiPrefs.v1`.
        uiPrefs: {
            modeFilter:     "",
            sortColumn:     "index_1based",
            sortDir:        "asc",
            broadeningFWHM: 20,
            animAmplitude:  0.15,
            animSpeed:      1.0,
            // WHICH pairing of eigenvector and amplitude (§ 12.2), and the
            // temperature the thermal one needs.  Preferences like the two above
            // them, so they belong in the bucket the others live in -- otherwise
            // they are simply left behind when this bucket learns to persist.
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
            // derived state today (trajectory has scfPollHistory
            // for the per-iter time estimate; spectra has nothing
            // equivalent).  Kept present-but-empty so the
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

    // Backward-compat aliases.  ~3000 lines of existing render +
    // event code reads/writes the legacy flat shape; the aliases
    // route through to the bucketed canonical home so the body keeps
    // working unchanged.  See trajectory/core.js for the same
    // pattern + rationale.
    (function _wireBackcompatAliases() {
        // The shared inspector helper (lib/inspectors/lifecycle.js): both
        // cores spelled this out byte-identically.
        function alias(key, bucket) {
            root.molbuilder.inspectorLifecycle.alias(state, key, bucket);
        }
        // fileState: legacy `watchPath` -> canonical `path`.
        Object.defineProperty(state, "watchPath", {
            get: function ()  { return state.fileState.path; },
            set: function (v) { state.fileState.path = v; },
            enumerable: true,
            configurable: true,
        });
        alias("results",        "fileState");
        alias("selectedMode",   "viewState");
        alias("modeFilter",     "uiPrefs");
        alias("sortColumn",     "uiPrefs");
        alias("sortDir",        "uiPrefs");
        alias("broadeningFWHM", "uiPrefs");
        alias("animAmplitude",  "uiPrefs");
        alias("animSpeed",      "uiPrefs");
        alias("animAmplitudeMode", "uiPrefs");
        alias("animTemperature",   "uiPrefs");
        alias("watchTimer",     "lifecycle");
        alias("watchInFlight",  "lifecycle");
        alias("watchAbort",     "lifecycle");
        alias("loadAbort",      "lifecycle");
        alias("watchErrors",    "lifecycle");
    })();

    // Transition orchestrator (contract § 2).  Single entry-point
    // for state-machine transitions; mirrors trajectory's
    // transition() implementation.
    //
    // Targets (per contract § 2):
    //   'LOADING'  -> { path }: empty fileState, reset viewState,
    //                 abort in-flight controllers, clear timer,
    //   'LOADED'   -> {}:       stop watchTimer.  Used when
    //                 allPhasesComplete OR after a Load-once.
    //   'WATCHING' -> {}:       start watchTimer.  Used by
    //                 startWatch and by watchTick when the run is
    //                 still progressing.
    //   'ERROR'    -> {}:       stop watchTimer.  Used after
    //                 WATCH_MAX_ERRORS consecutive failures or a
    //                 fatal schema mismatch.
    //   'IDLE'     -> {}:       full reset on dispose.
    //   'APPLY'    -> {path?, results?}: atomic fileState write.
    //                 Single canonical fileState writer per
    //                 contract § 2 (closed by trajectory's PR 2.3;
    //                 spectra mirrors that here).
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
            // Run finished (allPhasesComplete true) OR Load-once.
            // Stop watchTimer if running.
            if (state.lifecycle.watchTimer) {
                clearInterval(state.lifecycle.watchTimer);
                state.lifecycle.watchTimer = null;
            }
            // ABORT THE TICK ALREADY ON THE WIRE, as LOADING and IDLE
            // both do.  Without it "Stop" did not stop: `stopWatch`
            // cleared the timer, but a tick mid-flight then resolved
            // with `signal.aborted` false and -- because LOADED keeps
            // `fileState.path` on purpose -- passed the path guard too,
            // so it rendered and called `_settlePostLoad(true)`, which
            // transitions back to WATCHING and starts a NEW interval.
            // The buttons said stopped while the poll ran on, and the
            // only way out was Start-then-Stop.
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
            state.lifecycle.watchInFlight = false;
            state.machine = "ERROR";
            return;
        }
        if (target === "APPLY") {
            /* THE ONE WRITE: a name and the data that belongs to it,
             * together, or neither.  `results.md` § 4 has always said
             * fileState is "replaced atomically"; until 2026-09-03 it was
             * not.  Every caller passed `{results}` alone, so the answer
             * landed under whatever path happened to be sitting there, and
             * A `fetchSeq` counter existed to survive that -- snapshotted
             * before each fetch and re-checked after, in five places, to
             * notice it had written into the wrong file.  It is gone
             * (2026-09-04): once the answer carries its own name, the
             * question "is this still the file on screen?" is asked of the
             * data, and dispose -- which never bumped the counter -- is
             * caught too.
             *
             * A payload MUST now say which file it is for.  A late answer
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
            state.fileState.results = payload.results;
            return;
        }
        // Unknown target: silent no-op.
    }

    // _settlePostLoad: after fileState has been populated by
    // transition('APPLY'), inspect the fresh results and route to
    // the appropriate post-LOADING state.
    //
    // Two callers:
    //   * loadByPath (Load-once button) calls with start_watch=false.
    //     Run is rendered but no timer starts; we go to LOADED
    //     regardless of completion.
    //   * watchTick / startWatch call with start_watch=true.  If
    //     allPhasesComplete -> transition('LOADED') (run done; stop
    //     polling).  Else -> transition('WATCHING') (keep polling).
    //
    // Unlike trajectory there is NO 2-tick buffer here: spectra's
    // allPhasesComplete is a sticky monotonic flag (phase_*
    // markers progress forward through "running" -> "complete"
    // and never flap back).  One "complete" tick is sufficient.
    function _settlePostLoad(startWatch) {                            // eslint-disable-line no-unused-vars
        const results = state.fileState.results;
        if (!startWatch) {
            transition("LOADED");
            return;
        }
        if (results && allPhasesComplete(results)) {
            transition("LOADED");
        } else {
            transition("WATCHING");
        }
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

    // Hartree-to-eV conversion factor.  Used by the ES panel to
    // present MO energies in user-friendly units instead of Eh.
    // CODATA 2018 value.
    const EH_TO_EV = 27.211386245988;

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
    // THE SCOPE IS THE ONLY REGISTRY.  A local ``_cleanups`` array stood
    // beside it for one commit on 2026-08-23 -- the leftover of the array
    // ``_on`` used to push into before the scope was extracted -- and
    // dispose() drained THAT while every listener sat in the scope, so a
    // mount/dispose cycle removed nothing at all.  Two registries is the
    // same defect as two readers: one of them is the one that gets used.
    var _listeners = root.molbuilder.inspectorLifecycle.listeners();
    function _on(target, event, handler, opts) {
        _listeners.on(target, event, handler, opts);
    }

    // ----- Status helper ----------------------------------------
    function setStatus(el, msg, kind) {
        if (!el) return;
        el.textContent = msg || "";
        el.classList.remove("ok", "error", "muted", "warn");
        if (kind) el.classList.add(kind);
    }

    // ----- Form schema load + render ----------------------------
    //
    // The form is the CATALOGUE's vibration schema and depends on no
    // picked structure (frozen atoms are structure-side facts riding
    // the hand-over -- § 8 of web/spectra.md; the sidecar pre-fill
    // narration that stood here described the retired flow the two
    // functions below explicitly say is gone).

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
            els.formContainer.innerHTML =
                '<p class="status error">form-schema.js not loaded; '
                + 'check that <code>lib/form-schema.js</code> appears '
                + 'before this script in the template.</p>';
            return;
        }
        await _reloadVibrationSchema(fs);
        // No structure-commit subscription any more: the schema does
        // not depend on the picked structure (the sidecar pre-fill it
        // re-fetched for is gone -- see _reloadVibrationSchema), so a
        // sidebar pick no longer wipes typed parameters, and the
        // discard-confirm that guarded the wipe has nothing to guard.
    }

    async function _reloadVibrationSchema(fs) {
        // The vibration form comes from the CATALOGUE, narrowed to the
        // vibration calculation kind (template.md § 6.3) -- the same
        // door the Build tab reads, so a parameter is defined once and
        // rendered the same on every tab (spectra-migration-plan.md
        // P2's substitution).
        //
        // The schema does NOT depend on the picked structure any more:
        // the old per-structure re-fetch existed to let the server
        // seed ``frozen_indices`` from the .molstruct.json sidecar,
        // and frozen atoms are STRUCTURE-side facts now -- they ride
        // the hand-over's structure files straight off the model
        // (plan § 2), never a form field.
        const mySeq = ++_schemaFetchSeq;
        try {
            const schema = await fs.fetchSchema(
                "pyscf", { calculation: "vibration" },
            );
            // Race guard: a rapid remount can issue a newer fetch
            // during our await; the older response must not
            // overwrite the newer one in state.schema.
            if (mySeq !== _schemaFetchSeq) return;
            state.schema = schema;
            els.formContainer.innerHTML = "";
            fs.renderForm(els.formContainer, schema);
            wireCompatibilityListeners();
            applyCompatibility();
        } catch (exc) {
            // Same race guard for the failure path.
            if (mySeq !== _schemaFetchSeq) return;
            els.formContainer.innerHTML =
                '<p class="status error">Could not load form schema: '
                + escapeHtml(String(exc)) + '</p>';
        }
    }

    function escapeHtml(s) {
        return String(s).replace(/[&<>"']/g, (c) => ({
            "&": "&amp;", "<": "&lt;", ">": "&gt;",
            "\"": "&quot;", "'": "&#39;",
        }[c]));
    }

    // ----- Selector / compatibility (lock unused value fields) --
    //
    // The Model 2 selector (none/all/top_n/threshold/explicit) picks
    // exactly ONE active value field.  Locking the rest matches the
    // Build tab's pattern -- the user can't enter a top_n value
    // when threshold is selected, etc.
    function _fieldIdByName(name) {
        // The catalogue owns id derivation (`_item_to_field`); this
        // module never spells an id.  The previous hardcoded
        // spellings ("s-es_top_n") had drifted from the real ids
        // ("s-es-top-n"), so the lock below had never fired.
        const sections = (state.schema && state.schema.sections) || [];
        for (const sect of sections) {
            for (const f of (sect.fields || [])) {
                if (f.name === name) return f.id || "";
            }
        }
        return "";
    }

    function _esSelectionEl() {
        const id = _fieldIdByName("es_mode_selection");
        return id ? els.formContainer.querySelector("#" + id) : null;
    }

    function wireCompatibilityListeners() {
        _on(_esSelectionEl(), "change", applyCompatibility);
    }

    function applyCompatibility() {
        const sel = _esSelectionEl();
        if (!sel) return;
        const which = sel.value;
        // Map selector value -> the field that's active for it.  All
        // other Electronic-structure value fields get the disabled
        // attribute so the form coercion drops them.
        const activeByMode = {
            "skip":      null,
            "all":       null,
            "top_n":     "es_top_n",
            "threshold": "es_threshold",
            "explicit":  "es_explicit_indices",
        };
        const valueFields = [
            "es_top_n", "es_threshold", "es_explicit_indices",
        ];
        const active = activeByMode[which] || null;
        for (const name of valueFields) {
            const id = _fieldIdByName(name);
            const f = id
                ? els.formContainer.querySelector("#" + id) : null;
            if (!f) continue;
            const isActive = (name === active);
            f.disabled = !isActive;
            // Visually fade the field set so it's obvious which one
            // is in play -- the disabled attr does some of this, but
            // a class lets us style the wrapping <label> too.
            const wrap = f.closest("label, .field");
            if (wrap) wrap.classList.toggle("is-locked", !isActive);
        }
    }

    // ----- Helpers: gather form values + xyz -------------------
    function collectParams() {
        const fs = (window.molbuilder || {}).formSchema;
        if (!fs || !state.schema) return {};
        return fs.collectForm(els.formContainer, state.schema);
    }

    /* THE VIEWER THIS PAGE MOUNTED, handed to us by the page that mounted it
     * (spectra/viewer.js). We do not look one up: there is nothing to look up in,
     * and a viewer belongs to whoever mounted it (molview.md § 5.6). */
    let _viewer = null;
    function useViewer(handle) { _viewer = (handle && handle.ok) ? handle : null; }

    function getTheStructure() {
        return _viewer ? _viewer.data.getStructure() : null;
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
        const _out = _viewer ? _viewer.data.exportFile() : null;
        if (!_out || !_out.structure) return;
        try {
            const r = await fetch("/api/build/preflight", {
                method:  "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    structure:   _out.structure,
                    engine:      "pyscf",
                    calculation: "vibration",
                    params:      collectParams(),
                }),
            }).then(x => x.json());
            const vf = (window.molbuilder || {}).validationFindings;
            if (vf && Array.isArray(r.issues)) {
                /* fieldIds lands each finding beside its own control --
                 * the same map the optimization tab passes; omitting it
                 * degraded every Spectrum finding to card-then-residual
                 * (validation-findings.js says so).  emptyText keeps
                 * the panel's no-findings copy alive: the template's
                 * static row is destroyed by the first render. */
                const ids = {};
                const sects = (state.schema && state.schema.sections) || [];
                for (const s of sects) {
                    for (const f of (s.fields || [])) ids[f.name] = f.id;
                }
                vf.render(r.issues, { panel: els.preflightPanel,
                                      formScope: els.formContainer,
                                      fieldIds: ids,
                                      emptyText: "No findings yet — checks "
                                          + "run live as you edit." });
            }
        } catch (e) {
            // Network hiccup: the panel keeps its previous state; the
            // settings gate at prep is the canonical refusal anyway.
        }
    }
    const refreshPreflightDebounced = _debouncePreflight(refreshPreflight, 250);

    // ----- Send to Task setup (the hand-over) -------------------
    //
    // The P2 substitution (spectra-migration-plan.md § 4): this tab
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
        /* ONE READ OF THE VIEWER (molview.md § 9.3): `exportFile()` is
         * the viewer's own producer and emits the exact envelope the
         * hand-over door reads -- atoms, positions at the displayed
         * frame, labels, regions (frozen atoms) and cell in one read.
         * Frozen atoms REACH THE CALCULATION THIS WAY: they ride the
         * structure's own files, not a form field (plan § 2). */
        const _out = _viewer ? _viewer.data.exportFile() : null;
        await mb.taskHandover.send({
            projects:    mb.projects,
            say:         say,
            structure:   (_out && _out.structure) ? _out.structure : null,
            engine:      "pyscf",
            params:      collectParams(),
            calculation: "vibration",
        });
    }

    // The Generate / Save / Methods-modal half of this module was
    // REMOVED at P3 (spectra-migration plan, 2026-08-21) after the
    // P2 substitution made it unreachable: the tab DESCRIBES a
    // vibration calculation and hands it to Task setup
    // (sendToTaskSetup above); the deck is written by `prep`.

    // (loadResults removed 2026-05-18: the in-template multipart-
    // upload affordance is gone, replaced by the server-side path
    // loader below (els.loadPathBtn) and the sidebar's "Load from
    // current selection" path published via spectra/page.js.  The
    // /api/spectra/load endpoint still accepts multipart upload --
    // it just has no client today.)

    // ----- Load once by server-side path -----------------------
    //
    // Same /api/spectra/load endpoint as the file-upload path, but
    // with {path: "<server-side path>"} so the server reads the
    // file directly.  This is the primary path for users running
    // molbuilder on the same machine as their spectra.py job --
    // no re-upload after every phase write.
    /** THE one route to `/api/spectra/load` (`results.md` § 4).
     *
     *  There were two, and that was the defect underneath everything else:
     *  `loadByPath` read the filename out of the DOM box
     *  (`els.watchPath.value`) and `watchTick` read it out of the state
     *  (`state.fileState.path`).  Two sources of truth for *which file is
     *  this*, each with its own copy of the abort + sequence-guard dance.
     *
     *  One door, and the answer comes back WITH the name it was asked for,
     *  so no caller is in a position to write it under another.
     */
    // NOTE: the returned `path` is the REQUESTED one, echoed straight back
    // from the argument.  `/api/spectra/load` does not send a path, and it
    // must not be made to: the route resolves through
    // `_resolve_within_roots` (~ and $VARS expanded, symlinks followed), so
    // a server-echoed path would not equal the string in the path box and
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

    async function loadByPath() {
        const path = (els.watchPath.value || "").trim();
        if (!path) {
            setStatus(els.watchStatus, "Enter a path first.", "error");
            return;
        }
        setStatus(els.watchStatus, "Loading " + path + "…", "muted");
        // Contract § 2: file-switch / Load -> transition('LOADING').
        // Aborts loadAbort + watchAbort, clears watchInFlight, stops
        // the watchTimer if running, empties fileState (sets path),
        // and resets viewState.  Pre-PR-3 the inline aborts + path write
        // lived here;
        // PR 3 centralizes them in transition() for a single source
        // of truth.
        //
        // Subtle: pre-PR-3 the watchTimer was NOT cleared by Load-
        // once (K2 2026-06-14: "user still wants live updates").
        // PR 3 inverts: Load-once stops the timer.  Rationale: the
        // user clicking "Load once" while a watch is running is an
        // explicit "stop watching, show me this snapshot" gesture;
        // they can click "Start watching" again if they want polls.
        // This also matches contract § 5 ("Refresh = file-switch")
        // which forbids partial resets.
        transition("LOADING", { path: path });
        state.lifecycle.loadAbort = new AbortController();
        const signal = state.lifecycle.loadAbort.signal;
        let body;
        try {
            // Phase 6e seventh-review LANDMINE-4: the JSON parse used
            // to live OUTSIDE this try, so a malformed / truncated body
            // became an unhandled promise rejection.  It is inside
            // `fetchResults`, which is called here, so the SyntaxError
            // still lands on the same status-banner path as a network
            // error.
            body = (await fetchResults(path, signal)).body;
        } catch (exc) {
            // AbortError: a newer loadByPath() superseded us, or
            // dispose() ran.  Silent -- the newer action owns the
            // status banner now.
            if (exc.name === "AbortError") return;
            setStatus(els.watchStatus,
                      "Network error: " + exc.message, "error");
            transition("ERROR");
            return;
        }
        // Contract § 4 Invariant 1, asked of the ANSWER'S OWN IDENTITY.
        // Two questions, and the pair is what a sequence counter used to
        // approximate:
        //   * is this still the file on screen?  A newer LOADING moved
        //     `fileState.path`, and dispose set it to null -- which the
        //     counter never caught, because IDLE did not bump it;
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
                msg = "File not found at " + path
                    + ".  If the run is still in equilibrium SCF, "
                    + "click 'Start watching' to poll until the first "
                    + "phase checkpoint appears.";
            }
            setStatus(els.watchStatus, msg, "error");
            transition("ERROR");
            return;
        }
        renderResults(body.results, path);
        updatePhaseIndicator(body.results);
        setStatus(els.watchStatus, "Loaded.", "ok");
        // Load-once: no polling.  Transition to LOADED regardless
        // of completion -- the user explicitly asked for a snapshot.
        _settlePostLoad(false);
        // Signal "first render visible" so the /results tab-level
        // picker drops its "Parsing…" status.  Deferred via
        // double-rAF so the browser paints the spectra chart +
        // mode-table content before the picker meta clears -- see
        // ``lib/trajectory/core.js`` for the reasoning behind the
        // double-tick wait.
        try {
            const dispatch = () => document.dispatchEvent(new CustomEvent(
                window.molbuilder.constants.EVENT_INSPECTOR_READY,
                { detail: { inspector: "spectra" } }
            ));
            if (typeof requestAnimationFrame === "function") {
                requestAnimationFrame(() => requestAnimationFrame(dispatch));
            } else {
                dispatch();
            }
        } catch (_) { /* see lib/trajectory/core.js for context */ }
    }

    // ----- Live-watch poller (spec § 6.1) -----------------------
    //
    // Polls /api/spectra/load { path: <...> } every WATCH_INTERVAL_MS
    // while a job is running.  The engine writes <job>.spectra.json
    // atomically at each phase boundary, so each poll either:
    //   * gets a 404 (file not written yet -- equilibrium SCF still
    //     in flight); shows "Waiting..." and keeps polling.
    //   * gets a parsed SpectraResults; re-renders the UI with
    //     whatever phases are populated so far.
    //
    // Auto-stops when allPhasesComplete() returns true, when the
    // user clicks Stop, or after WATCH_MAX_ERRORS consecutive
    // transient failures.
    function startWatch() {
        const path = (els.watchPath.value || "").trim();
        if (!path) {
            setStatus(els.watchStatus, "Enter a path first.", "error");
            return;
        }
        if (state.lifecycle.watchTimer) return;  // already watching
        // Contract § 2: Start-watching = transition('LOADING') with
        // the user's path (full reset including any prior fileState
        // from a previous Load-once), then transition('WATCHING')
        // which starts the timer.  Same reset matrix as Load-once;
        // the difference is the post-load transition target.
        transition("LOADING", { path: path });
        els.watchBtn.disabled     = true;
        els.watchStopBtn.disabled = false;
        els.watchPath.disabled    = true;
        setStatus(els.watchStatus,
                  "Watching " + path + " every "
                  + (WATCH_INTERVAL_MS / 1000) + " s...", "muted");
        // First tick immediately so the user doesn't wait
        // WATCH_INTERVAL_MS before seeing any feedback.  The
        // immediate watchTick() will call _settlePostLoad(true) on
        // success, which transitions to WATCHING or LOADED.
        watchTick();
        // If the immediate tick didn't already transition us to
        // WATCHING (e.g. the file 404'd and watchTick returned
        // before _settlePostLoad), explicitly start the timer here
        // so polling continues.  transition('WATCHING') is
        // idempotent.
        if (state.machine !== "WATCHING" && state.machine !== "LOADED") {
            transition("WATCHING");
        }
    }

    function stopWatch(reason) {
        // Contract § 2: Stop-watching transitions to LOADED (the
        // file is no longer being polled but the loaded snapshot
        // remains visible).  Pre-PR-3 this nulled watchPath which
        // wiped the fileState.path; PR 3 keeps the path so the
        // user can see what they were watching.
        transition("LOADED");
        els.watchBtn.disabled     = false;
        els.watchStopBtn.disabled = true;
        els.watchPath.disabled    = false;
        if (reason) {
            setStatus(els.watchStatus, reason,
                      reason.startsWith("Run complete") ? "ok" : "muted");
        }
    }

    async function watchTick() {
        if (!state.fileState.path) return;
        // 2026-06-14 G4 in-flight guard: skip overlap ticks entirely.
        // PR 3 keeps this guard; dispose() aborts via watchAbort.
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
        // TWO guards at resolution, the same pair `loadByPath` uses, and
        // the reasoning that once justified only one of them was wrong.
        //
        // `watchInFlight` guards CONCURRENCY -- it stops a second tick
        // starting while one is out.  It cannot say anything about a tick
        // that has already SETTLED, and the deletion of `fetchSeq`
        // (2026-09-04) briefly rested on it doing both.  It does not:
        //
        //   watching A -> tick resolves, continuation queued
        //   -> Refresh / "Load once" fires for A (neither button is
        //      disabled during a watch, unlike Start)
        //      -> transition('LOADING') aborts us, stops the timer,
        //         empties results
        //   -> our continuation runs.  myPath === fileState.path, both
        //      "A", so a path check alone lets it through: it repaints
        //      the pre-Refresh body and _settlePostLoad restarts the
        //      poll timer the Refresh had just stopped.
        //
        // `signal.aborted` is what closes that, because the transition
        // aborted this request; the counter used to catch it by being
        // bumped.  The path answers the OTHER question -- a different
        // file, or dispose, which sets it to null and never bumped the
        // counter at all.  Both, or the pair is not equivalent.
        if (signal.aborted) return;
        if (myPath !== state.fileState.path) return;
        if (!body.ok) {
            if (body.kind === "not_found") {
                setStatus(els.watchStatus,
                          "Waiting for first checkpoint (equilibrium "
                          + "SCF still running)...", "muted");
                return;
            }
            stopWatch("Stopped: " + (body.error || "load failed"));
            return;
        }
        state.lifecycle.watchErrors = 0;
        // Render whatever phases are populated so far.
        renderResults(body.results, myPath);
        updatePhaseIndicator(body.results);
        // _settlePostLoad(true): if allPhasesComplete -> LOADED
        // (stops timer); else -> WATCHING (keeps polling).  This
        // replaces the inline allPhasesComplete -> stopWatch dance
        // with the unified post-load settle helper used by both
        // loadByPath and watchTick.
        _settlePostLoad(true);
        // Status banner: completion-message OR progress-line.
        if (allPhasesComplete(body.results)) {
            setStatus(els.watchStatus,
                      "Run complete ✓  ("
                      + (body.results.modes || []).length
                      + " modes; "
                      + (body.results.config && body.results.config.compute_raman
                         ? "Raman ✓ " : "")
                      + (body.results.config
                         && body.results.config.es_mode_selection
                         && body.results.config.es_mode_selection !== "skip"
                         ? "ES ✓ " : "")
                      + ")", "ok");
            // Restore button enablement that stopWatch normally does.
            els.watchBtn.disabled     = false;
            els.watchStopBtn.disabled = true;
            els.watchPath.disabled    = false;
        } else {
            setStatus(els.watchStatus, _watchProgressLine(body.results), "muted");
        }
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
        if (results.config && results.config.compute_raman && r !== "complete")
            return "Computing Raman activities (polarizability derivatives)";
        const sel = results.config && results.config.es_mode_selection;
        if (sel && sel !== "skip" && e !== "complete") {
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
        if (cfg.compute_raman && results.phase_raman !== "complete") return false;
        if (cfg.es_mode_selection && cfg.es_mode_selection !== "skip"
            && results.phase_es !== "complete") return false;
        return true;
    }

    function updatePhaseIndicator(results) {
        if (!els.phaseIndicator) return;
        els.phaseIndicator.hidden = false;
        const dots = els.phaseIndicator.querySelectorAll(".phase-dot");
        dots.forEach(dot => {
            const ph = dot.dataset.phase;   // relaxation|frequencies|raman|es
            const v  = results["phase_" + ph] || "empty";
            dot.className = "phase-dot phase-" + v;
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
    function renderResults(results, path) {
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
        // 2026-06-12 (audit #352): live-watch same-content guard.
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
            // transition('APPLY') so this function is no longer a
            // fileState writer in disguise (mirrors trajectory PR 2.3).
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
        const meta = [
            ["Engine",            results.engine + " " + (results.engine_version || "?")],
            ["Atoms (total)",     results.n_atoms_total],
            ["Free / frozen",     (results.free_atom_idxs || []).length
                                    + " / "
                                    + (results.frozen_atom_idxs || []).length],
            ["Equilibrium E (Eh)", (results.equilibrium &&
                                    Number(results.equilibrium.scf_energy_eh)
                                        .toFixed(8)) || "—"],
            ["CPU / threads",      cpu],
            ["GPU",                gpu],
            ["Host",               rt.hostname || "—"],
            ["Relaxation",                _relaxSummary(results)],
            ["Frequencies (Hessian)",     results.phase_frequencies],
            ["Raman activities",           results.phase_raman],
            ["Per-mode orbital energies",  results.phase_es],
        ];
        els.resultsMeta.innerHTML = meta
            .map(([k, v]) => "<dt>" + escapeHtml(String(k)) + "</dt>"
                           + "<dd>" + escapeHtml(String(v)) + "</dd>")
            .join("");

        // ES-derived table columns: ALWAYS visible.  Pre-fix the
        // ES column headers vanished when no mode had electronic_
        // structure populated -- same disease as the hide-frozen-
        // row case (UI presence tied to data).  Users would see
        // the column headers disappear on first results-load and
        // wonder where they went; subsequent runs of a different
        // job with ES data would have the columns reappear,
        // breaking column-position muscle memory.
        //
        // Contract: column presence is a stable affordance.  When
        // no mode has ES data, the cells render empty (per
        // ``renderModesTable`` below) -- that's the honest UX.
        // See 2026-06-14 hide-frozen-row precedent + the same-day
        // audit findings for context.
        const anyES = (results.modes || []).some(m => !!m.electronic_structure);

        // Auto-select the highest-Raman-activity real mode so the
        // ES panel comes up populated (if any mode has ES).  If no
        // mode has ES, fall back to the lowest-index real mode.
        //
        // 2026-06-17 Fix D: preserve the user's existing selection if
        // it's still valid in the new modes list.  Pre-fix every
        // renderResults call (including every live-watch tick that
        // passed the fingerprint guard) unconditionally overwrote
        // state.selectedMode via _pickDefaultMode -- so a user
        // browsing mode 5 would have their pick silently reset to
        // the auto-default the moment a new mode finished ES.  Only
        // auto-pick when the prior selection is null OR no longer
        // exists in the current modes list.
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

        renderSpectrumChart(results.modes || []);
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
        // The tracked relaxation phase (v5; plan D3/D4): what the
        // deck recorded, in one line.  v4 files carry no block.
        const rx = results.relaxation || {};
        if (rx.already_relaxed) return "skipped (already relaxed)";
        if (!rx.enabled) return results.phase_relaxation || "\u2014";
        let out = results.phase_relaxation || "empty";
        if (rx.n_steps != null) out += " \u2014 " + rx.n_steps + " steps";
        if (rx.max_force_eh_a != null) {
            out += ", max |F| "
                 + Number(rx.max_force_eh_a).toExponential(1)
                 + " Eh/\u00c5";
        }
        if (rx.warning) out += " \u26a0 " + rx.warning;
        return out;
    }

    function _resultsFingerprint(results, selectedMode) {
        // Compact key over the fields renderResults branches on.  Any
        // change here means "user-visible viewer state needs to update".
        // Stable string makes equality cheap — bail without rerendering
        // when the live-watch poll returned an unchanged snapshot.
        //
        // 2026-06-17 Fix C: include per-mode Raman + IR activity values
        // and frequencies.  Pre-fix the fingerprint covered modes.length
        // + ES bits + phase markers only; when Raman activities
        // populated mid-phase (same mode count, phase still "running",
        // no ES flip) the fingerprint was identical -> renderResults
        // bailed -> the spectrum chart bar heights never refreshed
        // until either modes grew or a phase marker changed.  Bar
        // heights ARE what the user is watching; treating them as
        // "no change" was the bug.
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
        // Build rows via createElement instead of innerHTML+string concat
        // (audit #354 follow-up).  The data values are server-supplied
        // numerics + booleans run through Number(...).toFixed() which is
        // safe today, but the pattern violated the XSS audit's "prefer
        // createElement" rule and a future schema change (e.g. a free-
        // text "notes" column on a mode) would silently re-introduce
        // an interpolation hazard.  textContent / appendChild keeps the
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
            m.has_imag ? "imag" : "",
            es ? "es" : "",
        ];
        if (es) {
            const homo = es.mo_energies_eq_eh[es.homo_index_in_window];
            const lumo = es.mo_energies_eq_eh[es.homo_index_in_window + 1];
            if (homo != null) vals.push((homo * EH_TO_EV).toFixed(3));
            if (lumo != null) vals.push((lumo * EH_TO_EV).toFixed(3));
            if (homo != null && lumo != null)
                vals.push(((lumo - homo) * EH_TO_EV).toFixed(3));
        }
        return vals.some(v => v.toLowerCase().includes(filt));
    }

    function _modeKey(m, col) {
        switch (col) {
            case "index_1based":          return m.index_1based;
            case "frequency_cm1":         return m.frequency_cm1;
            case "raman_activity_a4_amu": return m.raman_activity_a4_amu;
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
        return e == null ? null : e * EH_TO_EV;
    }
    function _lumoEq(m) {
        const es = m.electronic_structure;
        if (!es) return null;
        const e = es.mo_energies_eq_eh[es.homo_index_in_window + 1];
        return e == null ? null : e * EH_TO_EV;
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
        const dPlus  = ((lp - hp) - (l - h)) * EH_TO_EV * 1000;  // meV
        const dMinus = ((lm - hm) - (l - h)) * EH_TO_EV * 1000;
        return Math.max(Math.abs(dPlus), Math.abs(dMinus));
    }

    function _renderModeRow(m, anyES) {
        // Returns an HTMLTableRowElement; caller appends to tbody.
        // Pre-2026-06-13 this returned an interpolated <tr>...</tr>
        // string that callers concatenated into ``innerHTML`` —
        // worked because the values are numerics + booleans run
        // through Number(...).toFixed(), but it violated the project-
        // wide "prefer createElement" rule and would silently allow a
        // future free-text column to bypass escaping.
        const fmt = (v, dp) => v == null ? "—" : Number(v).toFixed(dp);
        const raman = (m.raman_activity_a4_amu == null)
            ? "—"
            : Number(m.raman_activity_a4_amu).toFixed(2);
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
        addCell(raman);
        addCell(m.has_imag ? "✓" : "");
        addCell(m.electronic_structure ? "✓" : "");
        if (anyES) {
            addCell(fmt(_homoEq(m), 3),  "es-col");
            addCell(fmt(_lumoEq(m), 3),  "es-col");
            addCell(fmt(_gapEq(m),  3),  "es-col");
            addCell(fmt(_dgapMax(m), 1), "es-col");
        }
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
        // This used to redraw the whole spectrum for every click.
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
        const headers = ["index_1based", "frequency_cm1",
                         "raman_activity_a4_amu", "has_imag", "has_es"];
        if (anyES) headers.push("homo_eq_ev", "lumo_eq_ev",
                                 "gap_eq_ev", "dgap_max_mev");
        const lines = [headers.join(",")];
        for (const m of _modesForTable()) {
            const row = [
                m.index_1based,
                Number(m.frequency_cm1).toFixed(4),
                m.raman_activity_a4_amu == null ? "" :
                    Number(m.raman_activity_a4_amu).toFixed(4),
                m.has_imag ? "1" : "0",
                m.electronic_structure ? "1" : "0",
            ];
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
    //
    // We deliberately use plain SVG (no Plotly) for the bar diagram:
    // it's a small static-ish picture and the SVG markup is easier
    // to read in the page source than a Plotly trace soup.
    /* ---- the level diagram's own helpers ---------------------------------
     *
     * Both of these served two figures until the spectrum became a module of
     * its own (docs/web/spectrumchart.md), which took its palette and its box
     * watcher with it.  What is left serves the electronic-structure diagram
     * alone, so it lives beside the diagram and is named for it.
     */
    /* THE CHART PALETTE, READ FROM THE STYLESHEET.
     *
     * Plotly takes colours as JavaScript values, so a chart cannot inherit them
     * the way an element does -- which is how both charts on this tab ended up
     * carrying their own copies of #1d2128, #2c313a and #cfd3da.  Three
     * literals, in two places, that a theme change would silently leave behind.
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
            dim:    tok("--text-muted",      "#6c7280"),
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
     * what changes.  This chart lives in a grid that reflows when the box it is
     * in gets wider or narrower -- the projects sidebar collapsing, the results
     * panel resizing, the mode viewer moving from beside the chart to below it
     * -- and none of those resize the window.  So the chart kept whatever width
     * it was first drawn at and either overflowed its box or left a gap.
     *
     * One observer on the container, redrawing at its new size. */
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
         * beside the mode it describes.  It was a row in the numbers list, where
         * it read as a result among results -- it is the INPUT every number
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
            els.esBarDiagram.innerHTML =
                '<p class="status muted">'
                + 'No electronic-structure data for this mode.<br>'
                + 'Re-run with es_mode_selection covering this mode '
                + '(or pick \'all\') to see HOMO/LUMO drift here.'
                + '</p>';
            els.esSummary.innerHTML = "";
            return;
        }

        // Convert MO arrays to eV.
        const eq    = es.mo_energies_eq_eh.map(e => e * EH_TO_EV);
        const minus = es.mo_energies_minus_eh.map(e => e * EH_TO_EV);
        const plus  = es.mo_energies_plus_eh.map(e => e * EH_TO_EV);
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
         * They were ten flat rows in one list, which made the reader do the
         * sorting: an equilibrium energy, a displaced gap and a coupling
         * constant sat side by side looking equally important.  Three groups
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

        els.esSummary.innerHTML = groups.map(([title, rows]) =>
            '<div class="es-group">'
            + '<h4 class="es-group-title">' + escapeHtml(title) + "</h4>"
            + "<dl>"
            + rows.map(([k, v, u]) =>
                  "<dt>" + escapeHtml(k) + "</dt>"
                + '<dd class="es-val">'  + escapeHtml(v) + "</dd>"
                + '<dd class="es-unit">' + escapeHtml(u) + "</dd>").join("")
            + "</dl></div>"
        ).join("");
    }


    /* THE LEVEL DIAGRAM, drawn by the same library as the spectrum above it.
     *
     * It was hand-rolled SVG, on the reasoning that a small static picture is
     * easier to read as markup than a Plotly trace.  That held until the picture
     * stopped being static: the level shifts this panel exists to show are tiny
     * -- 0.018 meV against an 11.4 eV span in the BDT result, which is 1/4000 of
     * a pixel -- so they cannot be seen without zooming, and zoom means pan,
     * range memory, a reset control and a hover readout.  Writing all four by
     * hand, next to a chart library already loaded on this very page and already
     * drawing the spectrum, would be inventing a wheel in view of the wheel.
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
                // THE POINT OF THE REWRITE: this axis is free.  Scroll to zoom,
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
    // Geometry source priority:
    //   1. results.equilibrium.elements + positions_ang
    //      (preferred; works after page reload).
    //   2. Read off the viewer (getTheStructure -- the structure it holds,
    //      molview.data.getStructure().text).
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

    // Q for the zero-point RMS, in √amu·Å, from a wavenumber in cm⁻¹:
    //     √(ħ / 4πcν̃) expressed in those units. At 1000 cm⁻¹ this is 0.1298,
    //     so a mode entirely on one hydrogen swings 0.13 Å — the textbook figure.
    const ZERO_POINT_Q = 4.105804;
    // ħω / k_B per cm⁻¹, in kelvin: the temperature at which a mode's quantum
    // is comparable to kT.
    const CM1_IN_KELVIN = 1.438777;

    /* Above this, calling a nearest neighbour a "bond" would be a claim rather
     * than a label, so the readout says "nearest contact" instead.  Generous on
     * purpose: it has to cover the long ones this program actually builds, Au–Au
     * at 2.88 Å and Au–S near 2.4 Å, and it decides one word of wording -- not
     * chemistry, not what is drawn. */
    const BOND_LIKE_ANG = 3.0;

    function _physicalAmplitude(waveNumberCm1, mode, temperatureK) {
        const nu = Math.abs(Number(waveNumberCm1));
        if (!isFinite(nu) || nu <= 0) return null;   // an imaginary or zero mode
        let q = ZERO_POINT_Q / Math.sqrt(nu);
        if (mode === "thermal") {
            const t = Number(temperatureK);
            if (isFinite(t) && t > 0) {
                const x = CM1_IN_KELVIN * nu / (2 * t);
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
     * It returns null when the result cannot be animated, and the caller says so
     * rather than finding a structure somewhere else.  There USED to be a
     * fallback here: when a result carried no stored geometry, it read the
     * structure off whatever MolView happened to be holding.  On /results that is
     * "whatever the last inspection installed" — so a mode could be animated
     * against a molecule it was not computed for, guarded only by an atom count,
     * which any two molecules of the same size pass.  It was also unreachable and
     * broken in two independent ways, so deleting it removed no working
     * behaviour: it returned its coordinates under a key nothing read, and the
     * viewer it read from is never handed to this page.
     */
    /* ONE SHAPE, ALWAYS, so a caller cannot forget a case:
     *
     *     { ready: false, why: null }      nothing is selected — say nothing
     *     { ready: false, why: "…" }       selected, but it cannot be drawn
     *     { ready: true, structure, mode, amplitude, norm }
     *
     * It used to answer three different shapes — null, an object carrying only a
     * message, and a full result — and a caller that checked two of the three
     * read a mode off the one that has none. Checking `ready` is now the only
     * thing to remember, and there is no fourth case to miss.
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
         * with an amplitude in √amu·Å. Crossing them is the correctness bug the
         * backend's own schema history records — v1 shipped one field used for
         * both, and splitting it is why there are two. */
        const wantPhysical = state.animAmplitudeMode !== "display";
        const physical = wantPhysical
            ? _physicalAmplitude(hz, state.animAmplitudeMode, state.animTemperature)
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
         * Reading playback back as if it were the intent is what an earlier draft
         * did, and it froze the molecule on the SECOND result you opened: the
         * structure install stopped the clock, nothing had asked for that, and
         * nothing started it again.
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
     * It read "≤ 0.050 Å · display", then "max ±0.050 Å", and both were unreadable
     * for the same reason: they gave a quantity without saying what it measured.
     * Less than WHAT?  The maximum over WHAT?  A reader who has to ask cannot use
     * the number, and a number nobody can use is decoration.
     *
     * So the line is a sentence now, and each clause answers one of those
     * questions (spectra.md § 4.2):
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
     * describe the same selected mode.  They used to be three bands stacked down
     * the page, so comparing a mode's motion against its level shifts meant
     * scrolling between two places and holding one in memory.  As tabs they
     * share one position, and the selection is what moves.
     *
     * SELECTING A MODE DOES NOT SWITCH TAB.  All three update underneath; the
     * reader stays where they were looking.  A click that yanks the view away is
     * the same as losing your place.
     *
     * WHY THIS IS NOT JUST TOGGLING `hidden`.  A box inside a hidden panel has
     * no size, and both a 3-D canvas and a Plotly figure take their size FROM
     * their box.  Mounted or drawn while their tab was hidden, they come back
     * with a zero-size drawing surface -- the same collapse that left the level
     * diagram a 10px strip.  So becoming visible is an event, and each view is
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
        state.vib.setAmplitude(inputs.amplitude);
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

    // ----- Spectrum chart (Plotly) -----------------------------
    //
    // Draws frequency (cm⁻¹) vs Raman activity (Å⁴/amu) as a
    // stem-style bar plot.  Imaginary modes (frequency < 0) get a
    // distinct red colour + a separate trace so a saddle-point
    // geometry is visually obvious without consulting the table.
    // Modes whose Raman activity isn't computed (cfg.compute_raman
    // = False on the producing run) are shown at activity 0 with a
    // grey marker so the user sees the mode density but understands
    // there's no intensity data.
    /* THE SPECTRUM CHART IS A MODULE NOW.
     *
     * Everything that used to be here -- the traces, the palette, the envelope,
     * the click tolerance, the width watcher -- lives behind one door in
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

    function _chartModes(modes) {
        // The tab's record, in the four fields the chart takes (§ 6.1).
        return (modes || []).map(m => ({
            index:     m.index_1based,
            freq:      m.frequency_cm1,
            activity:  Number.isFinite(m.raman_activity_a4_amu)
                ? m.raman_activity_a4_amu : null,
            imaginary: !!m.has_imag,
        }));
    }

    // ----- Thermochemistry panel (v5 `thermo`; plan § 2b) -------
    //
    // The DECK computes, this panel draws: the headline (T, P)
    // numbers, the regime sentence and the T-grid arrays are all read
    // off `results.thermo`.  The ONE derived quantity is the
    // electronic reference E_elec, recovered exactly from the grid's
    // own construction  h = E_elec + zpe + u_vib + kB*T  (how the
    // deck builds `h_eh`), so no thermochemistry formula lives here.
    //
    // Tab-owned Plotly, same rules as the ES level diagram: colours
    // from _esTheme()'s CSS tokens, NO `height` in the layout (the
    // .thermo-chart box owns it), and a plain degrade when Plotly is
    // not on the page (/results loads it; /spectra does not mount
    // this panel at all).
    var _KB_EH      = 3.166811563e-6;   // Boltzmann, Eh/K -- the deck's value
    var _EH_TO_KCAL = 627.509474;

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

        // --- The words: headline numbers + the deck's regime note --
        if (els.thermoNote) {
            const bits = [];
            let head = "At T = " + th.temperature_K + " K, P = "
                     + th.pressure_atm + " atm: ZPE "
                     + Number(th.zpe_eh).toFixed(6) + " Eh ("
                     + (th.zpe_eh * _EH_TO_KCAL).toFixed(1) + " kcal/mol)";
            if (th.g_eh != null) {
                head += "; H " + Number(th.h_eh).toFixed(6) + " Eh"
                      + "; S " + Number(th.s_eh_k).toExponential(4) + " Eh/K"
                      + "; G " + Number(th.g_eh).toFixed(6)
                      + " Eh (full RRHO)";
            }
            bits.push(head + ".");
            if (th.note) bits.push(String(th.note) + ".");
            bits.push("Curves show the harmonic VIBRATIONAL contributions "
                + "above the electronic minimum"
                + (th.regime === "rrho"
                   ? "; rotational/translational terms enter only the "
                     + "headline RRHO numbers above."
                   : "."));
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
        // E_elec off the grid identity -- exact, not a fit.
        const eRef = grid.h_eh[0] - grid.zpe_eh[0] - grid.u_vib_eh[0]
                   - _KB_EH * T[0];
        const gRel = grid.g_eh.map((g) => (g - eRef) * _EH_TO_KCAL);
        const hRel = grid.h_eh.map((h) => (h - eRef) * _EH_TO_KCAL);
        const ts   = T.map((Ti, i) => Ti * grid.s_eh_k[i] * _EH_TO_KCAL);
        const config = {
            displaylogo: false, responsive: true,
            modeBarButtonsToRemove: ["select2d", "lasso2d",
                                     "toggleSpikelines"],
        };
        Plotly.react(els.thermoCurves, [
            { x: T, y: gRel, name: "G − E_elec", mode: "lines",
              line: { color: t.homo, width: 2 } },
            { x: T, y: hRel, name: "H − E_elec", mode: "lines",
              line: { color: t.stick, width: 2 } },
            { x: T, y: ts, name: "T·S", mode: "lines",
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
            yaxis: { title: { text: "kcal/mol above E_elec",
                              font: { size: 10 } },
                     gridcolor: t.grid, zeroline: false },
        }, config);

        // --- Decomposition at the grid point nearest the headline T --
        let i0 = 0, dmin = Infinity;
        for (let i = 0; i < T.length; i++) {
            const d = Math.abs(T[i] - th.temperature_K);
            if (d < dmin) { dmin = d; i0 = i; }
        }
        const zpe  = grid.zpe_eh[i0] * _EH_TO_KCAL;
        const uth  = (grid.u_vib_eh[i0] + _KB_EH * T[i0]) * _EH_TO_KCAL;
        const mts  = -T[i0] * grid.s_eh_k[i0] * _EH_TO_KCAL;
        const gnet = (grid.g_eh[i0] - eRef) * _EH_TO_KCAL;
        Plotly.react(els.thermoDecomp, [{
            type: "bar",
            x: ["ZPE", "U_vib + kT", "−T·S",
                "G − E_elec"],
            y: [zpe, uth, mts, gnet],
            marker: { color: [t.stick, t.homo, t.lumo, t.stickSel] },
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
                    // A click enters the tab and comes back as setSelected;
                    // the chart never highlights on its own.
                    onSelect: (index) => selectMode(index),
                }))
                .then((handle) => {
                    if (!handle.ok) {
                        els.spectrumChart.innerHTML =
                            '<p class="status muted">' + handle.error + '</p>';
                        return null;
                    }
                    chart = handle;
                    return handle;
                })
                .catch((err) => {
                    els.spectrumChart.innerHTML =
                        '<p class="status muted">spectrum chart unavailable: '
                        + (err && err.message ? err.message : err) + '</p>';
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
        if (state.results) {
            renderSpectrumChart(state.results.modes || []);
        }
    }

    // Refresh-button listener wiring (contract § 5).  Mirrors
    // trajectory's _wireRefreshListener: wired ONCE at mount; not
    // re-wired per-load.  Spectra historically did not listen for
    // EVENT_REFRESH_REQUESTED (the file picker's "Refresh" button
    // fired into the void); PR 3 closes that gap so the Refresh
    // button does what § 5 says: file-switch with the current path.
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
            // Match the els.watchPath input so loadByPath picks up
            // the current value (loadByPath reads from the DOM).
            if (els.watchPath) els.watchPath.value = p;
            loadByPath();
        });
    }

    // ----- Bootstrap -------------------------------------------
    function init() {
        // ``xyz-file`` / ``xyz-load-btn`` / ``xyz-status`` lookups
        // dropped 2026-05-18: those template ids no longer exist
        // (sidebar took over file selection).  loadXyzFile() is also
        // gone; see the comment at the els declaration above.
        els.formContainer  = $("spectra-form-container");
        els.sendBtn        = $("send-to-task-setup");
        els.sendStatus     = $("send-status");
        els.preflightPanel = $("spectra-issues");
        // results-file / load-results-btn / results-status lookups
        // dropped for the same reason as xyz* above.
        els.resultsSummary = $("results-summary");
        els.resultsMeta    = $("results-summary-list");
        els.modesTbody     = $("modes-tbody");
        els.spectrumChart  = $("spectrum-chart");
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
        // Load-by-path + live-watch.
        els.watchPath         = $("watch-path");
        els.loadPathBtn       = $("load-path-btn");
        els.watchBtn          = $("watch-btn");
        els.watchStopBtn      = $("watch-stop-btn");
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
        // ids (load controls, results table, mode viewer, ES panel);
        // generate-side ids (form container, generate-btn, methods
        // modal, download/copy) live only in spectra.html.  Gate the
        // whole generate block on formContainer's presence so a
        // /results-side mount stays a clean inspect-only inspector.
        const hasGenerateSide = Boolean(els.formContainer);
        if (hasGenerateSide) {
            _on(els.sendBtn, "click", sendToTaskSetup);
            // The old form-dirty tracking left with its one reader
            // (the discard-confirm died at P2); the P3 sweep removed
            // the declaration but left three writers, which threw a
            // strict-mode ReferenceError on every edit -- caught by
            // the 2026-08-21 full-text review.
            // Live science check on every edit (and on auto-detect's
            // programmatic fills -- setValues dispatches input).
            _on(els.formContainer, "input",  refreshPreflightDebounced);
            _on(els.formContainer, "change", refreshPreflightDebounced);

        }

        // --- Inspect-side wiring -----------------------------------
        //
        // Gated on the partial's ``watch-path`` input.  Post-step 2.5
        // /spectra drops the inspect-side partial entirely (the page
        // becomes generate-only), so this whole block must no-op when
        // none of the inspect-side ids exist; the same module mounts
        // cleanly into either consumer.
        const hasInspectSide = Boolean(els.watchPath);
        if (hasInspectSide) {
            _on(els.loadPathBtn,  "click", loadByPath);
            _on(els.watchBtn,     "click", startWatch);
            _on(els.watchStopBtn, "click", function () { stopWatch("Stopped."); });
            // FWHM-controlled broadening re-renders the chart in
            // place.
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

    // PR 3 contract § 5: wire Refresh ONCE at mount.  Mirrors
    // trajectory's _wireRefreshListener pattern (which fixed the
    // per-load listener-pile-up bug there).
    _wireRefreshListener();

    // If the caller asked for an initial file (the /results-side
    // mount passes the sidebar's current selection via opts.file),
    // load it now.  /spectra's bootstrap doesn't pass opts.file --
    // the user types into watch-path or click-and-loads via the
    // sidebar handoff there.  Without this call /results would
    // mount the inspector but leave it empty + force the user to
    // re-pick the same file in the inspector's loader bar, which
    // is exactly the UX confusion the registry dispatch is meant
    // to eliminate.
    if (opts.file && els.watchPath) {
        // Pre-fill the watch-path input so the inspector matches
        // the visual state it'd be in if the user had typed the
        // path themselves and clicked Load (the existing
        // loadByPath reads from els.watchPath.value).  Guarded
        // on els.watchPath because /spectra (generate-only after
        // step 2.5) has no inspect-side ids and the assignment
        // would NPE; /results always has the partial mounted.
        els.watchPath.value = opts.file;
        loadByPath();
    }

    // ---- pageshow / visibilitychange: force-refresh on tab re-entry //
    //
    // Same shape as the /results file-picker (#192) + trajectory
    // inspector (#194): a bfcache restore or tab re-focus must
    // re-fetch the currently-loaded spectra file so a fresh result
    // generated in another tab actually appears.  Without these
    // handlers the user sees the cached snapshot from the previous
    // visit until they manually re-pick the file in the dropdown
    // -- the exact UX confusion #192 was filed for.
    //
    // Guard on ``state.results !== null`` so a never-loaded inspector
    // doesn't fire spurious /api/spectra/load on every visibility
    // event.  ``loadByPath()`` is path-driven (reads from
    // ``els.watchPath.value``), so we don't need to track the path
    // separately -- it's already pinned in the DOM and survives
    // bfcache.
    function _onPageShow(_evt) {
        if (state.results !== null && els.watchPath
            && els.watchPath.value) {
            loadByPath();
        }
    }
    function _onVisibilityChange(_evt) {
        if (document.visibilityState === "visible"
            && state.results !== null
            && els.watchPath && els.watchPath.value) {
            loadByPath();
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
            // "dispose / unmount" (aborts loadAbort + renderAbort +
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
         * between dispatch ticks instead of dispose → remount.
         * Equivalent to typing into ``watch-path`` and clicking
         * "Load once"; uses the same loadByPath path internally
         * (POST /api/spectra/load, abort + render + status update).
         */
        load(path) {
            if (els.watchPath) {
                els.watchPath.value = path;
            }
            return loadByPath();
        },

        /* THE VIEWER THIS PAGE MOUNTED, handed over by the page that mounted
         * it.  On the handle, because the viewer it stores is per-mount state:
         * `_viewer` lives inside this function, so a module-level door would be
         * writing into whichever mount happened to run last.
         *
         * It WAS module-level, in the export block below -- referencing a name
         * declared in here, which does not exist out there.  The whole export
         * threw `ReferenceError: useViewer is not defined` at load, so
         * `molbuilder.spectraInspector` was NEVER ASSIGNED: the page could not
         * find `.mount`, logged that core.js must be missing, and the entire
         * Generate side of /spectrum-calculation never started. */
        useViewer: useViewer,
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
