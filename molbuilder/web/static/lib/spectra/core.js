/* Spectra inspector core -- 3Dmol mode viewer + Plotly chart + form.
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
 * docs/protocols/results-tab.md).
 *
 * Wires the schema-driven SpectraConfig form (via the shared
 * ``molbuilder.formSchema`` helpers) to the three Spectra API
 * endpoints (spec § 10):
 *
 *   GET  /api/build/schema/spectra      -- form schema
 *   POST /api/spectra/render            -- generate spectra.py
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
    // text">``; the textarea was the only carrier and double-served
    // as the only DOM contract.  Task #309 retires the textarea —
    // ``setStructureText`` writes here, ``getStructureText`` reads
    // from here first and falls back to the textarea ONLY in test
    // contexts that still mount the DOM with the old shape.
    let _loadedStructureText = "";

    function mountInspector(rootEl, opts) {
    rootEl = rootEl || document;
    opts = opts || {};

    const $ = (id) => rootEl.querySelector("#" + id);

    // ----- DOM refs (resolved once at startup) -----------------
    const els = {
        spectrumChart:  null,
        structureText:        null,
        // xyzFile / xyzLoadBtn / xyzStatus removed 2026-05-18: the
        // in-template <input type="file" id="xyz-file"> + sibling
        // load-button were dropped when the projects sidebar took
        // over file selection.  loadXyzFile() and its event-listener
        // wiring are gone too (init had to null-check ``els.xyzLoadBtn``
        // since the id no longer exists; the whole branch was dead).
        formContainer:  null,
        generateBtn:    null,
        methodsBtn:     null,
        generateStatus: null,
        priorPath:      null,
        issuesPanel:    null,
        downloadBtn:    null,
        copyBtn:        null,
        scriptPreview:  null,
        scriptSummary:  null,
        // resultsFile / loadResultsBtn / resultsStatus removed for the
        // same reason as xyz* above; loadResults() also gone.  The
        // /api/spectra/load endpoint stays -- it's still used by the
        // path-based watch-path loader (els.loadPathBtn) below.
        resultsSummary: null,
        resultsMeta:    null,
        modesTbody:     null,
        methodsModal:   null,
        methodsBody:    null,
        methodsClose:   null,
        methodsCopy:    null,
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
        // 3Dmol mode-animation viewer (§ 9.2.3).
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
    const state = {
        schema:         null,
        lastScript:     null,
        lastMethodsMd:  null,
        lastJobName:    null,
        // Live results + selection state (interactive layer).
        results:        null,     // SpectraResults dict from /api/spectra/load
        selectedMode:   null,     // 1-based index of active mode, or null
        modeFilter:     "",       // current filter string
        sortColumn:     "index_1based",
        sortDir:        "asc",    // 'asc' | 'desc'
        // Live-watch poller state.
        watchTimer:     null,     // setInterval handle, or null
        watchPath:      null,     // server-side path being polled
        watchErrors:    0,        // consecutive transient-error counter
        // Spectrum chart -- Lorentzian broadening FWHM in cm⁻¹.
        // 0 disables the overlay (sticks only).
        broadeningFWHM: 20,
        // 3Dmol mode-animation viewer.
        // viewer:        null (removed by #232) — state.handle is
        //                       the only viewer reference now.
        // animTimer / animPhase / animLastTs removed by #231 Part B
        // — the embed owns the vibration rAF loop now.  Phase 5h I-3:
        // animPaused mirror retired too — read the live state from
        // _handle.isAnimationPlaying() (Phase 5g B-1 unified
        // store).  Mirror would drift on mode-switch (we force-play
        // via setAnimation({paused: false}) without touching the
        // shadow).
        animAmplitude:  0.15,    // peak Cartesian amplitude in Å (see partial)
        animSpeed:      1.0,     // cycle-rate multiplier (1.0 = ~1 Hz)
        // AbortControllers for in-flight HTTP requests.  ``loadAbort``
        // covers /api/spectra/load (Load + watch ticks); ``renderAbort``
        // covers /api/spectra/render (Generate script).  dispose()
        // aborts both so /results' registry can swap inspectors mid-
        // request without leaving an in-flight fetch hanging.
        loadAbort:      null,
        renderAbort:    null,
    };

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

    // ANIM_FRAMES_PER_CYCLE constant removed by #231 Part B — the
    // embed's vibration animation (handle.setAnimation({kind:
    // "vibration", displacements})) drives the rAF loop at the
    // browser's native cadence; no pre-computed frame count needed.

    // ----- Listener bookkeeping ---------------------------------
    //
    // Every element-level addEventListener inside mountInspector goes
    // through _on() so the registered teardown closure is captured in
    // _cleanups[].  dispose() walks _cleanups in reverse to
    // removeEventListener every one of them.
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
    const _cleanups = [];
    function _on(target, event, handler, opts) {
        if (!target) return;
        target.addEventListener(event, handler, opts);
        _cleanups.push(function () {
            target.removeEventListener(event, handler, opts);
        });
    }

    // Form-dirty primitive (B.5.3) — flipped true on any
    // user-driven input/change inside #spectra-form-container.
    // Used by the commit handler to fire the shared warning
    // modal before a sidebar dblclick rebuilds the schema for
    // a different structure, which would silently discard the
    // user's parameter edits.  Cleared on commit-accept,
    // Generate, and Save (the values have been emitted into a
    // script or written to disk).
    let _formDirty = false;

    // ----- Status helper ----------------------------------------
    function setStatus(el, msg, kind) {
        if (!el) return;
        el.textContent = msg || "";
        el.classList.remove("ok", "error", "muted");
        if (kind) el.classList.add(kind);
    }

    // ----- Form schema load + render ----------------------------
    //
    // The three-stage contract (design.md "Sidecar-driven boundary
    // conditions"): when the user picks a .xyz that has a sidecar,
    // the server pre-fills the form's structure-dependent defaults
    // (today: ``frozen_indices`` from sidecar's ``frozen_atoms``)
    // so the boundary condition is VISIBLE before Generate.  We
    // pass the current sidebar selection (if any, and if it's a
    // .xyz) as ``structurePath`` so the schema endpoint can read
    // the sidecar.  We also subscribe to the sidebar's onChange so
    // the form re-renders with fresh pre-fill when the user picks
    // a different structure.

    // Monotonic counter for in-flight schema fetches.  Rapid
    // sidebar clicks fire onChange multiple times; each fires a
    // _reloadSchemaForCurrentStructure().  The fetches race -- a
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
        await _reloadSchemaForCurrentStructure(fs);

        // Re-fetch + re-render when the sidebar's pick changes to
        // a different .xyz, so the sidecar's frozen_atoms surfaces
        // in the form for the new structure.  Non-xyz picks
        // (peeking a .log, .fdf, etc.) don't touch the form.
        //
        // ``projects.onChange`` fires its callback IMMEDIATELY on
        // subscribe (its contract -- gives subscribers a free
        // initial fire so they don't need a separate
        // getCurrentFile() call).  If the current file is a .xyz
        // we'll end up doing a duplicate render right after the
        // ``await`` above; the ``_schemaFetchSeq`` race guard
        // makes that correct but wastes one fetch.  Track the
        // unsubscribe in ``_cleanups`` so dispose() actually
        // breaks the subscription (parity with every other
        // long-lived hook in this file).
        // Subscribe via the module-init contract (design.md "Module
        // init contract").  The projects-sidebar module loads as
        // ``<script type="module">`` (deferred), so its API isn't
        // available at DOMContentLoaded -- ``runtime.whenReady`` is
        // the structural way to await it without polling.  Falls
        // back to a direct capture when the runtime isn't loaded
        // (legacy / isolation tests) so this code still works
        // without it.
        const _rt = (window.molbuilder || {}).runtime;
        const _projP = (_rt && typeof _rt.whenReady === "function")
            ? _rt.whenReady("projects")
            : Promise.resolve((window.molbuilder || {}).projects);
        _projP.then((proj) => {
        if (!proj) return;
        // The "use this file in Spectra" handler — schema reload
        // for the picked structure.  Defined once and wired to
        // both the universal commit channel AND the page-mount
        // cross-tab handoff (a structure file already in
        // sessionStorage when /spectrum-calculation mounts should
        // be treated as a commit; the user picked it on another
        // tab and navigated here to configure spectra for it).
        async function _commitStructureForSpectra(sel) {
            const f = (sel && sel.file) ? String(sel.file) : "";
            const lc = f.toLowerCase();
            if (!lc.endsWith(".xyz") && !lc.endsWith(".pdb")) return;
            // Form-dirty gate: if the user has typed parameter
            // edits since the last commit/Generate, ask before
            // discarding via the shared warning modal.
            const modal = window.molbuilder
                       && window.molbuilder.warningModal;
            if (_formDirty && modal
                && typeof modal.confirmDiscardUnsaved === "function") {
                const proceed = await modal.confirmDiscardUnsaved();
                if (!proceed) return;
            }
            _formDirty = false;
            await _reloadSchemaForCurrentStructure(fs).catch(() => {});
        }
        // Universal commit subscription — dblclick on a structure
        // file in the sidebar.  Falls back to onChange for older
        // deployments without onCommit.
        const _subscribe = (typeof proj.onCommit === "function")
            ? proj.onCommit.bind(proj)
            : proj.onChange.bind(proj);
        const unsub = _subscribe(_commitStructureForSpectra);
        if (typeof unsub === "function") {
            _cleanups.push(unsub);
        }
        // Cross-tab handoff: if a structure is already in
        // sessionStorage when this tab mounts, treat it as a commit
        // so the user doesn't have to re-click in the sidebar.
        // Only fires once on mount; onCommit itself doesn't fire-
        // on-subscribe.
        const _initFile = (typeof proj.getCurrentFile === "function")
            ? proj.getCurrentFile() : "";
        if (_initFile) {
            const _initDir = (typeof proj.getCurrentDir === "function")
                ? proj.getCurrentDir() : "";
            _commitStructureForSpectra({ dir: _initDir, file: _initFile });
        }
        });   // close _projP.then(...)
    }

    async function _reloadSchemaForCurrentStructure(fs) {
        // Resolve the current sidebar .xyz, if any.  The schema
        // endpoint tolerates a missing / invalid path (it returns
        // the static schema with a ``_notice`` describing why the
        // pre-fill was skipped), so this branch is just "best-
        // effort hint to the server"; nothing here can fail the
        // form mount.
        const proj = (window.molbuilder || {}).projects;
        let structurePath = "";
        if (proj && typeof proj.getCurrentFile === "function") {
            const f = proj.getCurrentFile() || "";
            const lc = f.toLowerCase();
            if (lc.endsWith(".xyz") || lc.endsWith(".pdb")) {
                structurePath = f;
            }
        }
        const mySeq = ++_schemaFetchSeq;
        try {
            const schema = await fs.fetchSchema(
                "spectra", { structurePath: structurePath },
            );
            // Race guard: if a newer fetch was issued during our
            // await, drop our continuation -- otherwise an older
            // response would overwrite the newer one in
            // state.schema + the rendered form, and the user
            // would see the wrong structure's pre-fill.
            if (mySeq !== _schemaFetchSeq) return;
            state.schema = schema;
            els.formContainer.innerHTML = "";
            fs.renderForm(els.formContainer, schema);
            // Surface the optional server-side notice (e.g. "sidecar
            // atom count differs -- re-export from /modify").
            // Non-enumerable property; only set when the server
            // sent one.
            const notice = schema._notice;
            if (notice && els.generateStatus) {
                setStatus(els.generateStatus, notice, "warn");
            }
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
    function wireCompatibilityListeners() {
        const sel = els.formContainer.querySelector("#s-es-selection");
        _on(sel, "change", applyCompatibility);
    }

    function applyCompatibility() {
        const sel = els.formContainer.querySelector("#s-es-selection");
        if (!sel) return;
        const which = sel.value;
        // Map selector value -> the field that's active for it.  All
        // other Electronic-structure value fields get the disabled
        // attribute so the form coercion drops them.
        const activeByMode = {
            "skip":      null,
            "all":       null,
            "top_n":     "s-es_top_n",
            "threshold": "s-es_threshold",
            "explicit":  "s-es_explicit_indices",
        };
        const allValueIds = [
            "s-es_top_n", "s-es_threshold", "s-es_explicit_indices",
        ];
        const active = activeByMode[which] || null;
        for (const id of allValueIds) {
            const f = $(id);
            if (!f) continue;
            const isActive = (id === active);
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

    function getStructureText() {
        // Source of truth: the module-level ``_loadedStructureText``
        // populated by the Inspect-structure card's Load handler
        // (spectra/viewer.js _commitStructure → setStructureText).
        // The legacy ``<textarea id="structure-text">`` was retired
        // 2026-06-09 (task #309).  Fall through to it ONLY when
        // ``_loadedStructureText`` is empty AND the DOM still
        // carries the textarea — covers test contexts that mount
        // the pre-#296 DOM shape.  XYZ or PDB content both
        // accepted; the server sniffs the format.
        if (_loadedStructureText) {
            return _loadedStructureText.trim();
        }
        if (els.structureText && els.structureText.value) {
            return els.structureText.value.trim();
        }
        return "";
    }

    // ----- Render button: POST /api/spectra/render -------------
    async function generateScript() {
        const structureText = getStructureText();
        if (!structureText) {
            setStatus(els.generateStatus,
                      "Paste an XYZ or PDB block, or pick a structure file first.",
                      "error");
            return;
        }
        setStatus(els.generateStatus, "Rendering…");
        clearOutputs();

        const body = {
            // ``structure_text`` (renamed 2026-05-22 from the
            // misleading ``xyz``): server sniffs XYZ vs PDB from
            // the first line of the content.
            structure_text: structureText,
            params:         collectParams(),
        };
        // Pass the current sidebar XYZ path (if any) so the server
        // can apply the .molstruct.json sidecar to the structure
        // BEFORE preflight runs.  This is what activates stage-3
        // of the boundary-condition contract (design.md
        // "Sidecar-driven boundary conditions") end-to-end:
        // preflight's Pattern A (divergence) + Pattern B
        // (unrecognized labels) need struct.frozen_atoms +
        // struct.regions populated, which only happens via the
        // sidecar.  Pasted-XYZ-only flows (no sidebar pick) just
        // skip the sidecar and lose stage-3 guards -- the right
        // behavior when there genuinely is no sidecar to honor.
        //
        // Variable name: ``projForStruct`` (not ``proj``) because
        // the lower saveToWorkspace block already has a ``const
        // proj`` -- a second ``const proj`` in the same function
        // scope is a SyntaxError that breaks the whole module.
        const projForStruct = (window.molbuilder || {}).projects;
        if (projForStruct && typeof projForStruct.getCurrentFile === "function") {
            const f = (projForStruct.getCurrentFile() || "");
            // Both .xyz and .pdb are accepted on the server -- send
            // the path either way so the sidecar's frozen_atoms +
            // regions flow into preflight Pattern A/B regardless
            // of source format.
            const lc = f.toLowerCase();
            if (lc.endsWith(".xyz") || lc.endsWith(".pdb")) {
                body.structure_path = f;
            }
        }
        const priorPath = (els.priorPath.value || "").trim();
        if (priorPath) body.prior_path = priorPath;

        // Supersede any in-flight render (rapid click on Generate
        // while a previous render is still on the wire).  dispose()
        // aborts on unmount.
        if (state.renderAbort) state.renderAbort.abort();
        state.renderAbort = new AbortController();
        const signal = state.renderAbort.signal;
        let r;
        try {
            r = await fetch("/api/spectra/render", {
                method:  "POST",
                headers: { "Content-Type": "application/json" },
                body:    JSON.stringify(body),
                signal:  signal,
            }).then(x => x.json().then(b => ({ status: x.status, body: b })));
        } catch (exc) {
            // AbortError: a newer Generate click superseded this
            // call, or dispose() ran.  Silent.
            if (exc.name === "AbortError") return;
            setStatus(els.generateStatus,
                      "Network error: " + exc.message, "error");
            return;
        }

        // The /api/spectra/render endpoint includes issues even on
        // 400; render them so the user sees structured feedback.
        renderIssues(r.body.issues || []);

        if (!r.body.ok) {
            setStatus(els.generateStatus,
                      r.body.error || "Render failed.", "error");
            return;
        }

        // Happy path: stash script + methods_md, enable buttons.
        state.lastScript    = r.body.script;
        state.lastMethodsMd = r.body.methods_md || "";
        state.lastJobName   = r.body.job_name || "spectra";

        els.scriptPreview.textContent = state.lastScript;
        els.downloadBtn.disabled = false;
        els.copyBtn.disabled     = false;

        const nLines = state.lastScript.split("\n").length;
        setStatus(els.scriptSummary,
                  `${nLines} lines · ${state.lastJobName}.spectra.py`,
                  "muted");

        const warns = (r.body.issues || []).filter(i => i.severity === "warn");
        const summary = warns.length
            ? `Generated with ${warns.length} warn(s).`
            : "Generated.";
        setStatus(els.generateStatus, summary, "ok");

        // 2026-05-26: Generate is now render-only.  The Save-to-current-
        // dir button (see saveSpectraToCurrentDir below) is the explicit
        // disk-commit action.  Mirrors the Build SIESTA / PySCF split
        // shipped 2026-05-24 (abc8a7c).
        refreshSaveSpectraButtonAvailability();
    }


    /**
     * Re-evaluate enable state on the Save-spectra button.  Enabled
     * when (a) the user has clicked Generate (state.lastScript is
     * populated) AND (b) the Projects sidebar has a non-root subdir
     * selected.  Wired to projects.onChange so a sidebar pick toggles
     * the button live without a page reload.
     */
    function refreshSaveSpectraButtonAvailability() {
        const btn = els.saveBtn;
        if (!btn) return;
        const proj = (window.molbuilder || {}).projects;
        // hasDir uses projects.atRoot() (added 2026-05-26) so the
        // button doesn't enable at the projects root (where
        // saveToWorkspace silently returns null + click would
        // produce a confusing "no current_dir" error).
        const hasDir = proj && typeof proj.atRoot === "function"
            ? !proj.atRoot()
            : !!(proj && proj.getCurrentDir());
        btn.disabled = !(state.lastScript && hasDir);
    }


    /**
     * Commit step.  Saves <job>.spectra.py to the sidebar dir + drops
     * a sibling <job>.spectra.run.sh.  Reentrancy guard prevents
     * double-click pipelines.  Mirrors the SIESTA + PySCF Save
     * handlers in viewer.js (commit 084df7e).
     */
    async function saveSpectraToCurrentDir() {
        if (state.savingSpectra) return;
        const proj = (window.molbuilder || {}).projects;
        const btn = els.saveBtn;
        if (!state.lastScript) {
            setStatus(els.saveStatus, "Click Generate first.", "error");
            return;
        }
        if (!proj) {
            setStatus(els.saveStatus, "Projects sidebar not loaded.", "error");
            return;
        }
        const destDir = proj.getCurrentDir() || "";
        if (!destDir) {
            setStatus(els.saveStatus,
                "Pick a project subdir in the sidebar first.",
                "error");
            return;
        }
        state.savingSpectra = true;
        const _origText = btn.textContent;
        btn.disabled = true;
        btn.textContent = "Saving…";
        // Sidebar lock: same rationale as the SIESTA/PySCF Save
        // handlers in viewer.js -- a mid-pipeline directory change
        // would retarget the wrapper install away from where we
        // just wrote the .spectra.py.
        state.saveSpectraAbort = new AbortController();
        const _saveSignal = state.saveSpectraAbort.signal;
        const _hasLock = typeof proj.lock === "function";
        if (_hasLock) {
            try {
                proj.lock("Saving Spectra + wrapper…",
                          [() => state.saveSpectraAbort.abort()]);
            } catch (_) { /* already locked: continue without */ }
        }
        try {
            setStatus(els.saveStatus, "Saving to " + destDir + " …", "muted");
            // safeSave threads the signal end-to-end and renames
            // the abort envelope's ``aborted`` flag to
            // ``cancelled`` — see projects/state.js safeSave.
            // Without it, callers used to forget either the signal
            // (BOMB-A, fifth review) or the filter (BOMB-1, fourth
            // review).  The helper makes both invariants the
            // default shape.
            const wrote = await proj.safeSave(
                state.lastScript, state.lastJobName + ".spectra.py",
                { overwrite: true, signal: _saveSignal });
            if (wrote && wrote.cancelled) {
                setStatus(els.saveStatus, "Save cancelled.", "muted");
                return;
            }
            if (!wrote || !wrote.ok) {
                setStatus(els.saveStatus,
                    "Save failed: " + (wrote && wrote.error || "no current_dir"),
                    "error");
                return;
            }
            let wrapperMsg = "";
            let wrapperCancelled = false;
            try {
                const wr = await fetch("/api/run/install-wrapper", {
                    method:  "POST",
                    headers: { "Content-Type": "application/json" },
                    body:    JSON.stringify({
                        script_path:   wrote.path,
                        mpi_np:        null,
                        omp_threads:   null,
                        max_memory_mb: null,
                    }),
                    signal: _saveSignal,
                }).then(x => x.json());
                if (wr.ok) {
                    const verb = wr.overwritten ? "overwrote" : "wrote";
                    wrapperMsg = " · " + verb + " " + wr.wrapper_name;
                }
            } catch (e) {
                if (proj.isCancelError(e)) {
                    wrapperCancelled = true;
                }
                /* other failures stay non-fatal */
            }
            if (wrapperCancelled) {
                setStatus(els.saveStatus,
                    "Save cancelled after writing " + wrote.relPath
                  + " (wrapper not installed).", "muted");
                return;
            }
            setStatus(els.saveStatus,
                "Wrote " + wrote.relPath + wrapperMsg, "ok");
            // Refresh sidebar so the .spectra.py + .run.sh both appear.
            try { if (proj.refresh) await proj.refresh(); } catch (_) { /* non-fatal */ }
        } finally {
            state.savingSpectra = false;
            btn.disabled = false;
            btn.textContent = _origText;
            refreshSaveSpectraButtonAvailability();
            if (_hasLock) proj.unlock();
        }
    }

    function clearOutputs() {
        state.lastScript    = null;
        state.lastMethodsMd = null;
        els.scriptPreview.textContent = "";
        els.downloadBtn.disabled = true;
        els.copyBtn.disabled     = true;
        setStatus(els.scriptSummary, "", "muted");
    }

    // ----- Issues panel ----------------------------------------
    //
    // Show error → warn → info, in that order, so the user reads
    // blockers first and advisory notices last.  Pre-task-#304 the
    // renderer concatenated only ``errs + warns`` and dropped
    // ``info`` silently — the Pattern-B regions notice that
    // /api/build/{fdf,pyscf} emits as INFO would never reach the
    // user on the Spectra panel even though the API delivered it.
    function renderIssues(issues) {
        const panel = els.issuesPanel;
        if (!issues || issues.length === 0) {
            panel.innerHTML =
                '<p class="status muted">No issues.</p>';
            return;
        }
        const errs  = issues.filter(i => i.severity === "error");
        const warns = issues.filter(i => i.severity === "warn");
        const infos = issues.filter(i => i.severity === "info");
        const html = [];
        for (const i of errs.concat(warns).concat(infos)) {
            const cls = (i.severity === "error" ? "issue error"
                : i.severity === "warn"  ? "issue warn"
                                         : "issue info");
            html.push(
                '<div class="' + cls + '">'
                + '<span class="badge">' + escapeHtml(i.severity) + '</span> '
                + '<span class="msg">' + escapeHtml(i.message) + '</span>'
                + (i.where ? ' <code class="where">' + escapeHtml(i.where) + '</code>' : '')
                + '</div>'
            );
        }
        panel.innerHTML = html.join("\n");
    }

    // ----- Download / copy script ------------------------------
    function downloadScript() {
        if (!state.lastScript) return;
        const blob = new Blob([state.lastScript], { type: "text/x-python" });
        const url  = URL.createObjectURL(blob);
        const a    = document.createElement("a");
        a.href     = url;
        a.download = (state.lastJobName || "spectra") + ".spectra.py";
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    }

    async function copyScript() {
        if (!state.lastScript) return;
        try {
            await navigator.clipboard.writeText(state.lastScript);
            setStatus(els.generateStatus, "Copied.", "ok");
        } catch (exc) {
            setStatus(els.generateStatus,
                      "Copy failed (browser blocked clipboard).", "error");
        }
    }

    // ----- Methods modal ---------------------------------------
    function openMethodsModal() {
        const md = state.lastMethodsMd
            || "Run *Generate* to see the Methods preview.";
        // Minimal Markdown -> HTML: paragraphs separated by blank
        // lines.  We deliberately don't load a full Markdown lib --
        // the content is plain prose with bibliography keys in
        // backticks, no headings / lists.
        const html = md
            .split(/\n\n+/)
            .map(p => "<p>" + escapeHtml(p).replace(/\n/g, "<br>") + "</p>")
            .join("\n");
        els.methodsBody.innerHTML = html;
        if (typeof els.methodsModal.showModal === "function") {
            els.methodsModal.showModal();
        } else {
            // Fallback for browsers without <dialog>.
            els.methodsModal.setAttribute("open", "");
        }
    }

    function closeMethodsModal() {
        if (typeof els.methodsModal.close === "function") {
            els.methodsModal.close();
        } else {
            els.methodsModal.removeAttribute("open");
        }
    }

    async function copyMethods() {
        if (!state.lastMethodsMd) return;
        try {
            await navigator.clipboard.writeText(state.lastMethodsMd);
        } catch (_) {
            // Fall through silently -- the user can still select the
            // text from the modal manually.
        }
    }

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
    async function loadByPath() {
        const path = (els.watchPath.value || "").trim();
        if (!path) {
            setStatus(els.watchStatus, "Enter a path first.", "error");
            return;
        }
        setStatus(els.watchStatus, "Loading " + path + "…", "muted");
        // Cancel any previous in-flight load (rapid file-switching);
        // start a fresh controller for this request.  dispose()
        // aborts it on unmount.
        if (state.loadAbort) state.loadAbort.abort();
        state.loadAbort = new AbortController();
        const signal = state.loadAbort.signal;
        let body;
        try {
            const r = await fetch("/api/spectra/load", {
                method:  "POST",
                headers: { "Content-Type": "application/json" },
                body:    JSON.stringify({ path: path }),
                signal:  signal,
            });
            // Phase 6e seventh-review LANDMINE-4: r.json() used to
            // live OUTSIDE this try, so a malformed / truncated
            // JSON body became an unhandled promise rejection.
            // Inside the try the SyntaxError lands on the same
            // status-banner path as the network error.
            body = await r.json();
        } catch (exc) {
            // AbortError: a newer loadByPath() superseded us, or
            // dispose() ran.  Silent -- the newer action owns the
            // status banner now.
            if (exc.name === "AbortError") return;
            setStatus(els.watchStatus,
                      "Network error: " + exc.message, "error");
            return;
        }
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
            return;
        }
        renderResults(body.results);
        updatePhaseIndicator(body.results);
        setStatus(els.watchStatus, "Loaded.", "ok");
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
        if (state.watchTimer) return;  // already watching
        state.watchPath   = path;
        state.watchErrors = 0;
        els.watchBtn.disabled     = true;
        els.watchStopBtn.disabled = false;
        els.watchPath.disabled    = true;
        setStatus(els.watchStatus,
                  "Watching " + path + " every "
                  + (WATCH_INTERVAL_MS / 1000) + " s...", "muted");
        // First tick immediately so the user doesn't wait WATCH_INTERVAL_MS
        // before seeing any feedback.
        watchTick();
        state.watchTimer = setInterval(watchTick, WATCH_INTERVAL_MS);
    }

    function stopWatch(reason) {
        if (state.watchTimer) clearInterval(state.watchTimer);
        state.watchTimer       = null;
        state.watchPath        = null;
        state.watchErrors      = 0;
        els.watchBtn.disabled     = false;
        els.watchStopBtn.disabled = true;
        els.watchPath.disabled    = false;
        if (reason) {
            setStatus(els.watchStatus, reason,
                      reason.startsWith("Run complete") ? "ok" : "muted");
        }
    }

    async function watchTick() {
        if (!state.watchPath) return;
        let body;
        try {
            const r = await fetch("/api/spectra/load", {
                method:  "POST",
                headers: { "Content-Type": "application/json" },
                body:    JSON.stringify({ path: state.watchPath }),
            });
            // Phase 6e seventh-review LANDMINE-4: r.json() inside
            // the try so malformed JSON (proxy drop, SIESTA
            // mid-write race) doesn't become an unhandled
            // rejection in the setInterval that drives the watch.
            body = await r.json();
        } catch (exc) {
            state.watchErrors++;
            setStatus(els.watchStatus,
                      "Network error (" + state.watchErrors + "/"
                      + WATCH_MAX_ERRORS + "): " + exc.message, "error");
            if (state.watchErrors >= WATCH_MAX_ERRORS) {
                stopWatch("Stopped after " + WATCH_MAX_ERRORS
                          + " consecutive network errors.");
            }
            return;
        }
        if (!body.ok) {
            // 404 (file not yet written) is the COMMON case during
            // the equilibrium SCF -- treat it as "still warming up",
            // not as a hard error.  Other kinds (malformed,
            // schema_mismatch, field) stop the watcher.
            if (body.kind === "not_found") {
                setStatus(els.watchStatus,
                          "Waiting for first checkpoint (equilibrium "
                          + "SCF still running)...", "muted");
                return;
            }
            stopWatch("Stopped: " + (body.error || "load failed"));
            return;
        }
        state.watchErrors = 0;
        // Render whatever phases are populated so far.
        renderResults(body.results);
        updatePhaseIndicator(body.results);
        // Auto-stop when all configured phases are done.
        if (allPhasesComplete(body.results)) {
            stopWatch("Run complete ✓  ("
                      + (body.results.modes || []).length
                      + " modes; "
                      + (body.results.config && body.results.config.compute_raman
                         ? "Raman ✓ " : "")
                      + (body.results.config
                         && body.results.config.es_mode_selection
                         && body.results.config.es_mode_selection !== "skip"
                         ? "ES ✓ " : "")
                      + ")");
        } else {
            setStatus(els.watchStatus, _watchProgressLine(body.results), "muted");
        }
    }

    function _watchProgressLine(results) {
        // One-line summary of where the run is RIGHT NOW so the
        // user knows what to expect.
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
            const ph = dot.dataset.phase;            // "frequencies"|"raman"|"es"
            const v  = results["phase_" + ph] || "empty";
            dot.className = "phase-dot phase-" + v;
            dot.title = ph + ": " + v;
        });
    }

    function renderResults(results) {
        if (!results) {
            els.resultsSummary.hidden = true;
            state.results      = null;
            state.selectedMode = null;
            return;
        }
        state.results = results;
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
            ["Frequencies (Hessian)",     results.phase_frequencies],
            ["Raman activities",           results.phase_raman],
            ["Per-mode orbital energies",  results.phase_es],
        ];
        els.resultsMeta.innerHTML = meta
            .map(([k, v]) => "<dt>" + escapeHtml(String(k)) + "</dt>"
                           + "<dd>" + escapeHtml(String(v)) + "</dd>")
            .join("");

        // Show/hide ES-derived table columns based on whether any
        // mode has electronic_structure populated.
        const anyES = (results.modes || []).some(m => !!m.electronic_structure);
        rootEl.querySelectorAll(".modes-table .es-col").forEach(th => {
            th.hidden = !anyES;
        });

        // Auto-select the highest-Raman-activity real mode so the
        // ES panel comes up populated (if any mode has ES).  If no
        // mode has ES, fall back to the lowest-index real mode.
        if (results.modes && results.modes.length) {
            state.selectedMode = _pickDefaultMode(results.modes, anyES);
        } else {
            state.selectedMode = null;
        }

        renderSpectrumChart(results.modes || []);
        renderModesTable();
        renderESPanel();
        // Geometry changed (new results loaded) -- discard the old
        // 3Dmol viewer so the next render rebuilds with the fresh
        // structure.
        if (state.handle) {
            _stopAnimation();
            // #206 / #232 cleanup: handle.dispose() tears down the
            // standard knob bar DOM + ResizeObserver + the embed's
            // vibration loop.  state.viewer was a redundant alias
            // for state.handle._viewer3dmol() left over from the
            // pre-Part-B path; #232 review collapses to the handle.
            try { state.handle.dispose(); }
            catch (_) {}
            state.handle = null;
            if (els.modeViewer) els.modeViewer.innerHTML = "";
        }
        renderModeViewer();
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
        const rows = modes.map(m => _renderModeRow(m, anyES));
        els.modesTbody.innerHTML = rows.join("");

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
        const raman = (m.raman_activity_a4_amu == null)
            ? "—"
            : Number(m.raman_activity_a4_amu).toFixed(2);
        const fmt = (v, dp) => v == null ? "—" : Number(v).toFixed(dp);
        const hev = anyES ? fmt(_homoEq(m), 3) : "";
        const lev = anyES ? fmt(_lumoEq(m), 3) : "";
        const gev = anyES ? fmt(_gapEq(m),  3) : "";
        const dgmev = anyES ? fmt(_dgapMax(m), 1) : "";
        const esCols = anyES
            ? `<td class="es-col">${hev}</td>`
              + `<td class="es-col">${lev}</td>`
              + `<td class="es-col">${gev}</td>`
              + `<td class="es-col">${dgmev}</td>`
            : "";
        const imagClass = m.has_imag ? ' class="mode-imag"' : "";
        return `<tr data-mode="${m.index_1based}"${imagClass}>`
            + `<td>${m.index_1based}</td>`
            + `<td>${Number(m.frequency_cm1).toFixed(1)}</td>`
            + `<td>${raman}</td>`
            + `<td>${m.has_imag ? "✓" : ""}</td>`
            + `<td>${m.electronic_structure ? "✓" : ""}</td>`
            + esCols
            + `</tr>`;
    }

    function _highlightActiveRow() {
        const rows = els.modesTbody.querySelectorAll("tr");
        rows.forEach(r => {
            const active = Number(r.dataset.mode) === state.selectedMode;
            r.classList.toggle("active", active);
            r.setAttribute("aria-selected", active ? "true" : "false");
        });
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
        // Also highlight the corresponding stick in the chart by
        // re-rendering it (Plotly's selectedpoints API is per-trace,
        // and we have three; cleanest is a full react()).
        renderSpectrumChart((state.results.modes || []));
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
        els.esModeFreq.textContent =
            Number(m.frequency_cm1).toFixed(1) + " cm⁻¹"
            + (m.has_imag ? " (imaginary)" : "");

        const es = m.electronic_structure;
        if (!es) {
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

        els.esBarDiagram.innerHTML = _renderBarDiagramSVG({
            minus: minus, eq: eq, plus: plus,
            homo_idx: hi, lumo_idx: li,
            yMin: yMin, yMax: yMax,
            amplitude: es.amplitude_ang,
        });

        // Summary dict: Gap @ eq / ±A, ΔGap, ES SCF energies.
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

        const summary = [
            ["Amplitude A",            es.amplitude_ang.toFixed(3) + " Å"],
            ["HOMO @ eq",              eq[hi].toFixed(4)    + " eV"],
            ["LUMO @ eq",              eq[li].toFixed(4)    + " eV"],
            ["Gap @ eq",               gap_eq.toFixed(4)    + " eV"],
            ["Gap @ +A",               gap_plus.toFixed(4)  + " eV"],
            ["Gap @ −A",               gap_minus.toFixed(4) + " eV"],
            ["ΔGap (+A)",              dgap_plus_mev.toFixed(2)  + " meV"],
            ["ΔGap (−A)",              dgap_minus_mev.toFixed(2) + " meV"],
            ["g_HOMO ≈ ΔE/(2A)",       g_HOMO_mev_A.toFixed(1) + " meV/Å"],
            ["g_LUMO ≈ ΔE/(2A)",       g_LUMO_mev_A.toFixed(1) + " meV/Å"],
        ];
        els.esSummary.innerHTML = summary
            .map(([k, v]) => "<dt>" + escapeHtml(String(k)) + "</dt>"
                           + "<dd>" + escapeHtml(String(v)) + "</dd>")
            .join("");
    }

    function _renderBarDiagramSVG(opts) {
        // Three columns: -A, 0, +A.  Each column has horizontal
        // bars for every MO energy.  HOMO/LUMO are coloured;
        // others are grey lines.
        const W = 520, H = 220;
        const margin = { top: 20, right: 16, bottom: 36, left: 56 };
        const innerW = W - margin.left - margin.right;
        const innerH = H - margin.top  - margin.bottom;
        const yScale = (e) =>
            margin.top + innerH * (1 - (e - opts.yMin) / (opts.yMax - opts.yMin));

        const cols = [
            { label: "−A", x: 0,            arr: opts.minus },
            { label: "eq", x: innerW / 2,   arr: opts.eq    },
            { label: "+A", x: innerW,       arr: opts.plus  },
        ];
        const barW = 80;

        const svgParts = [
            `<svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="xMidYMid meet"`,
            ` width="100%" role="img" aria-label="MO energy bar diagram">`,
            // y-axis line
            `<line x1="${margin.left}" y1="${margin.top}"`,
            `      x2="${margin.left}" y2="${margin.top + innerH}"`,
            `      stroke="#3a3f48" />`,
            `<text x="6" y="${margin.top + innerH / 2}"`,
            `      transform="rotate(-90 6 ${margin.top + innerH / 2})"`,
            `      fill="#cfd3da" font-size="11" text-anchor="middle">`,
            `  Energy (eV)`,
            `</text>`,
        ];

        // y-axis ticks (5 even).
        for (let i = 0; i <= 5; i++) {
            const v = opts.yMin + (opts.yMax - opts.yMin) * i / 5;
            const y = yScale(v);
            svgParts.push(
                `<line x1="${margin.left - 4}" y1="${y}"`,
                `      x2="${margin.left}"     y2="${y}"`,
                `      stroke="#3a3f48" />`,
                `<text x="${margin.left - 8}" y="${y + 3}"`,
                `      fill="#cfd3da" font-size="10" text-anchor="end">`,
                `  ${v.toFixed(2)}`,
                `</text>`
            );
        }

        for (const col of cols) {
            const cx = margin.left + col.x;
            // x-axis label.
            svgParts.push(
                `<text x="${cx}" y="${margin.top + innerH + 20}"`,
                `      fill="#cfd3da" font-size="11" text-anchor="middle">`,
                `  ${col.label}</text>`
            );
            for (let i = 0; i < col.arr.length; i++) {
                const y = yScale(col.arr[i]);
                const isHomo = (i === opts.homo_idx);
                const isLumo = (i === opts.lumo_idx);
                const color = isHomo ? "#4a90d9"
                            : isLumo ? "#e0a070"
                            : "#666";
                const sw    = (isHomo || isLumo) ? 2.5 : 1;
                svgParts.push(
                    `<line x1="${cx - barW / 2}" y1="${y}"`,
                    `      x2="${cx + barW / 2}" y2="${y}"`,
                    `      stroke="${color}" stroke-width="${sw}" />`
                );
            }
        }

        // Legend.
        svgParts.push(
            `<g transform="translate(${margin.left + 12}, ${margin.top - 4})">`,
            `  <line x1="0" y1="0" x2="14" y2="0" stroke="#4a90d9" stroke-width="2.5" />`,
            `  <text x="18" y="3" fill="#cfd3da" font-size="10">HOMO</text>`,
            `  <line x1="58" y1="0" x2="72" y2="0" stroke="#e0a070" stroke-width="2.5" />`,
            `  <text x="76" y="3" fill="#cfd3da" font-size="10">LUMO</text>`,
            `</g>`,
            `</svg>`
        );
        return svgParts.join("\n");
    }

    // ----- Mode-animation viewer (§ 9.2.3) ---------------------
    //
    // 3Dmol.js renders the equilibrium structure inside #mode-viewer
    // and we animate the selected mode by adding the eigenvector
    // displacement times sin(phase) to each atom's equilibrium
    // position on every animation frame.
    //
    // Geometry source priority:
    //   1. results.equilibrium.elements + positions_ang
    //      (preferred; works after page reload).
    //   2. Parsed from els.structureText.value
    //      (fallback; only works while the user keeps the XYZ in
    //      the input form).
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

    function _equilibriumGeometry() {
        // Return { elements, positions } or null if neither source
        // has a usable structure.  Positions are Å.
        const r = state.results;
        if (r && r.equilibrium
                && Array.isArray(r.equilibrium.elements)
                && Array.isArray(r.equilibrium.positions_ang)
                && r.equilibrium.elements.length
                && r.equilibrium.positions_ang.length
                   === r.equilibrium.elements.length) {
            return {
                elements:  r.equilibrium.elements.slice(),
                positions: r.equilibrium.positions_ang.map(row => row.slice()),
            };
        }
        // Fallback: parse the XYZ the user loaded via the
        // Inspect-structure card (module-level
        // ``_loadedStructureText``; the legacy textarea fallback
        // continues to work for test contexts that still mount
        // the pre-#296 DOM shape).
        const structureText = getStructureText();
        if (!structureText) return null;
        try {
            return window.molbuilder.xyz.parse(structureText);
        } catch (_) {
            return null;
        }
    }

    function renderModeViewer() {
        // Top-level entry point.  Called whenever selection / results
        // change.  Shows / hides the viewer, builds the 3Dmol
        // instance lazily, and starts (or stops) the animation
        // depending on whether a mode is selected with a non-null
        // eigenvector.
        if (!els.modeViewerWrap) return;

        const geom = _equilibriumGeometry();
        if (!geom || state.selectedMode == null || !state.results) {
            els.modeViewerWrap.hidden = true;
            _stopAnimation();
            return;
        }
        const mode = (state.results.modes || []).find(
            m => m.index_1based === state.selectedMode
        );
        if (!mode || !Array.isArray(mode.eigenvector_display)) {
            els.modeViewerWrap.hidden = true;
            _stopAnimation();
            return;
        }
        els.modeViewerWrap.hidden = false;

        if (typeof window.$3Dmol === "undefined") {
            els.modeViewer.innerHTML =
                '<p class="status muted" style="padding:1rem">'
                + '3Dmol.js failed to load; mode animation '
                + 'unavailable.</p>';
            setStatus(els.viewerStatus, "3Dmol not loaded", "muted");
            return;
        }
        setStatus(els.viewerStatus,
                  `Mode ${mode.index_1based} · `
                  + Number(mode.frequency_cm1).toFixed(1)
                  + " cm⁻¹"
                  + (mode.has_imag ? " (imag)" : ""),
                  "muted");

        _ensureViewer();
        _startAnimation(geom, mode);
    }

    function _ensureViewer() {
        // Create the embed handle ONCE per mount.  Owns the canvas
        // + standard knob bar + vibration animation; mode swaps
        // re-run setStructure + setAnimation without re-mounting.
        if (state.handle) return;
        els.modeViewer.innerHTML = "";
        // #206 Part A migration: mount via the standard embed so
        // the knob bar (Style / Labels / Axes / Reset / PNG /
        // Background / Export) appears above the mode-viewer
        // canvas, matching every other tab.  ``_viewer3dmol()``
        // returns the underlying 3Dmol viewer so the existing
        // vibration machinery (addModelsAsFrames + setFrame +
        // amplitude/speed loop in _startAnimation) continues
        // untouched.
        //
        // Part B (#231, landed 2026-06-03): vibration playback now
        // goes through the embed's ``animation: {kind: "vibration",
        // displacements}`` per § 3.9 — the bespoke
        // ANIM_FRAMES_PER_CYCLE pre-computed sine cycle + raw
        // setFrame loop are gone.  Frozen-atom greying goes through
        // ``handle.setOverlays({atoms: [{indices, style}]})`` per
        // § 3.12.
        state.handle = window.molbuilder.viewer.embed(
            els.modeViewer, {
                // Phase 6: ``#1d2128`` is now the embed-wide default
                // background; spectra no longer needs to override it
                // explicitly (or widen the preset list — the default
                // list is ["#1d2128", "#ffffff", "transparent"]).
                style:  { rep: "ball-and-stick", radiusScale: 1.0 },
                pick:   { mode: "none" },
                card:   { title: "Vibrational mode",
                          showInfoLine: false,
                          height: "100%" },
                axes:   true,
                export: { defaultName: "vibration" },
                onError(err) {
                    try { console.warn("[spectra.inspector]",
                                        err.code, err.message); }
                    catch (_) {}
                },
            }
        );
    }

    // _buildFrameMovie removed by #231 Part B.  The embed's
    // vibration animation (handle.setAnimation({kind: "vibration",
    // displacements})) replaces the bespoke ANIM_FRAMES_PER_CYCLE
    // pre-computed sine cycle.  The embed's loop computes
    // pos_i(φ) = baseline_i + amplitude · cos(φ) · displacement_i
    // and applies it natively at requestAnimationFrame cadence;
    // amplitude / speedHz changes are partial-update opts on
    // setAnimation, no frame rebuild needed.

    function _applyModeViewerStyle() {
        // #231 Part B: base style flows through the embed's
        // handle.setStyle (ball-and-stick) -- already set at
        // mount time.  Frozen atoms grey out via the OverlaySpec
        // per-atom style override per § 3.12.  Replaces the raw
        // viewer.setStyle({serial: idx+1}, ...) per-atom loop.
        const frozen = (state.results.frozen_atom_idxs || []).map(Number);
        if (frozen.length) {
            state.handle.setOverlays({
                atoms: [{
                    indices: frozen,
                    style:   { color: "#555" },
                }],
            });
        } else {
            // Clear any prior frozen-atom overlay so a mode swap
            // doesn't leave stale greys when the new geometry has
            // no frozen atoms.
            state.handle.setOverlays(null);
        }
    }

    function _startAnimation(geom, mode) {
        // #231 Part B: vibration playback is the embed's
        // ``animation: {kind: "vibration", displacements,
        // amplitude, speedHz, paused}`` contract per § 3.9.
        // The embed's loop computes pos_i(φ) = baseline_i +
        // amplitude · cos(φ) · displacement_i; the previous
        // bespoke ANIM_FRAMES_PER_CYCLE pre-computed sine cycle
        // produces the same visible oscillation (sin vs cos is
        // a 90° phase shift the user can't see).
        _stopAnimation();
        if (els.animToggle) els.animToggle.textContent = "Pause";

        // Build the per-atom displacement vector.  Free atoms
        // get the mode's eigenvector entry; frozen atoms stay
        // at zero (no motion, anchors the visualisation).
        const free       = state.results.free_atom_idxs || [];
        const evec_free  = mode.eigenvector_display;
        const nAtoms     = geom.elements.length;
        const displacements = new Array(nAtoms);
        for (let i = 0; i < nAtoms; i++) displacements[i] = [0, 0, 0];
        for (let k = 0; k < free.length; k++) {
            const atomIdx = free[k];
            if (atomIdx >= 0 && atomIdx < nAtoms) {
                displacements[atomIdx] = evec_free[k].slice();
            }
        }

        // Build the equilibrium-structure xyz text and mount it
        // as the embed's baseline.  The vibration loop applies
        // amplitude · cos(φ) · displacement on top of these
        // baseline coordinates.
        const lines = [String(nAtoms), ""];
        for (let i = 0; i < nAtoms; i++) {
            const p = geom.positions[i];
            lines.push(
                geom.elements[i]
                + " " + p[0].toFixed(6)
                + " " + p[1].toFixed(6)
                + " " + p[2].toFixed(6)
            );
        }
        state.handle.setStructure({ xyz: lines.join("\n") });
        _applyModeViewerStyle();
        state.handle.refit();

        // Hand the vibration to the embed.  Amplitude + speedHz
        // changes after this point just call ``setAnimation`` with
        // a partial payload (see onAnimAmplitudeChange + speed
        // listener); the displacements stay until a new mode is
        // selected, when the whole vibration is re-set with the
        // new eigenvector.
        state.handle.setAnimation({
            kind:          "vibration",
            displacements: displacements,
            amplitude:     state.animAmplitude,
            speedHz:       state.animSpeed,
            paused:        false,
        });
    }

    function _stopAnimation() {
        // Cancel embed-driven vibration playback.  The handle's
        // pauseAnimation tears down the rAF loop the embed runs.
        if (state.handle
            && typeof state.handle.pauseAnimation === "function") {
            try { state.handle.pauseAnimation(); } catch (_) {}
        }
    }

    function onAnimAmplitudeChange() {
        const v = parseFloat(els.animAmplitude.value);
        if (Number.isFinite(v)) state.animAmplitude = v;
        if (els.animAmplitudeVal)
            els.animAmplitudeVal.textContent = v.toFixed(2) + " Å";
        // #231 Part B: amplitude is a partial-update field on the
        // embed's vibration animation.  No frame rebuild needed --
        // the embed's loop reads amplitude live.
        if (state.handle && state.results && state.selectedMode != null) {
            try {
                state.handle.setAnimation({ amplitude: v });
            } catch (_) {}
        }
    }
    function onAnimSpeedChange() {
        const v = parseFloat(els.animSpeed.value);
        if (Number.isFinite(v)) state.animSpeed = v;
        if (els.animSpeedVal)
            els.animSpeedVal.textContent = v.toFixed(1) + "×";
        if (state.handle && state.results && state.selectedMode != null) {
            try {
                state.handle.setAnimation({ speedHz: v });
            } catch (_) {}
        }
    }
    function onAnimToggle() {
        // Phase 5h I-3: read the live runtime state from the embed
        // (Phase 5g B-1 unified store) rather than a host-side
        // mirror.  Without B-1 the mirror would silently drift
        // every time _setMode forced playback via
        // setAnimation({paused: false}).
        if (!state.handle) return;
        const wasPlaying = state.handle.isAnimationPlaying();
        try {
            if (wasPlaying) state.handle.pauseAnimation();
            else            state.handle.playAnimation();
        } catch (_) {}
        if (els.animToggle) {
            els.animToggle.textContent = wasPlaying ? "Play" : "Pause";
        }
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
    function renderSpectrumChart(modes) {
        if (!els.spectrumChart) return;
        if (typeof Plotly === "undefined") {
            // Plotly is loaded via CDN; if a slow network hasn't
            // delivered it yet the modes table still renders.  Show
            // a one-line fallback rather than failing silently.
            els.spectrumChart.innerHTML =
                '<p class="status muted">Plotly not loaded; spectrum chart unavailable.</p>';
            return;
        }
        if (!modes.length) {
            Plotly.purge(els.spectrumChart);
            els.spectrumChart.innerHTML =
                '<p class="status muted">No modes yet.</p>';
            return;
        }

        // Two display modes:
        //
        //   ACTIVITY MODE -- at least one mode has a Raman activity;
        //     y-axis is the activity in Å⁴/amu.  Modes without
        //     activity (partial L3) plot at y=0 with a "Raman not
        //     yet computed" hover hint so the user knows what's
        //     missing.
        //
        //   DENSITY MODE  -- no mode has a Raman activity yet (L2
        //     done but L3 hasn't started, or compute_raman=False).
        //     Every stick gets unit height so the frequency
        //     distribution is visible; y-axis title tells the user
        //     intensities are missing.  Otherwise the user just
        //     sees a flat x-axis line with nothing on it.
        const anyRaman   = modes.some(m =>
            m.raman_activity_a4_amu !== null
            && m.raman_activity_a4_amu !== undefined
        );
        const densityMode = !anyRaman;

        // Bucket the modes into traces by real / imaginary so each
        // gets its own hover + legend entry.  Modes pending Raman
        // also get their own trace so the legend says "pending"
        // explicitly rather than mixing into "Real" / "Imaginary".
        const real    = { x: [], y: [], text: [], idx: [], color: [] };
        const imag    = { x: [], y: [], text: [], idx: [], color: [] };
        const pending = { x: [], y: [], text: [], idx: [] };
        const sel     = state.selectedMode;
        for (const m of modes) {
            const f      = Number(m.frequency_cm1);
            const hasIm  = !!m.has_imag || f < 0;
            const raman  = m.raman_activity_a4_amu;
            const isSel  = (m.index_1based === sel);
            const ramanText = (raman === null || raman === undefined)
                ? "Raman: not yet computed"
                : "Raman = " + Number(raman).toFixed(2) + " Å⁴/amu";
            const txt = "Mode " + m.index_1based
                      + "<br>ω = " + f.toFixed(1) + " cm⁻¹"
                      + "<br>" + ramanText;

            if (raman === null || raman === undefined) {
                if (densityMode) {
                    // No intensity data anywhere -- show the stick
                    // at unit height in the appropriate real/imag
                    // bucket so the legend still works.
                    if (hasIm) {
                        imag.x.push(f); imag.y.push(1); imag.text.push(txt);
                        imag.idx.push(m.index_1based);
                        imag.color.push(isSel ? "#ffd454" : "#e07070");
                    } else {
                        real.x.push(f); real.y.push(1); real.text.push(txt);
                        real.idx.push(m.index_1based);
                        real.color.push(isSel ? "#ffd454" : "#4a90d9");
                    }
                } else {
                    // Partial L3 (some modes have activity, this one
                    // doesn't yet) -- mark with a separate "pending"
                    // trace so the user sees there are uncomputed
                    // modes beyond the visible spectrum.
                    pending.x.push(f);
                    pending.y.push(0);
                    pending.text.push(txt);
                    pending.idx.push(m.index_1based);
                }
            } else if (hasIm) {
                imag.x.push(f);
                imag.y.push(Number(raman));
                imag.text.push(txt);
                imag.idx.push(m.index_1based);
                // Highlight the selected stick by colour-overriding
                // its bar in the per-point marker.color array.
                imag.color.push(isSel ? "#ffd454" : "#e07070");
            } else {
                real.x.push(f);
                real.y.push(Number(raman));
                real.text.push(txt);
                real.idx.push(m.index_1based);
                real.color.push(isSel ? "#ffd454" : "#4a90d9");
            }
        }

        const traces = [];
        if (real.x.length) traces.push({
            type:        "bar",
            name:        densityMode ? "Real (freq only)" : "Real",
            x:           real.x,
            y:           real.y,
            text:        real.text,
            hoverinfo:   "text",
            marker:      { color: real.color, line: { width: 0 } },
            width:       6,
            // Stash the mode-index list on the trace so the click
            // handler can look up which mode was hit.
            customdata:  real.idx,
        });
        if (imag.x.length) traces.push({
            type:        "bar",
            name:        densityMode ? "Imaginary (freq only)" : "Imaginary",
            x:           imag.x,
            y:           imag.y,
            text:        imag.text,
            hoverinfo:   "text",
            marker:      { color: imag.color, line: { width: 0 } },
            width:       6,
            customdata:  imag.idx,
        });
        // Lorentzian-broadened envelope.  Active in activity mode
        // (sticks visible) when FWHM > 0; rendered as a line trace
        // overlaid on the sticks.  Sum of Lorentzians centered at
        // each mode's frequency, normalised so peak height = the
        // mode's Raman activity.
        if (!densityMode && state.broadeningFWHM > 0) {
            const envelope = _lorentzianEnvelope(
                modes, state.broadeningFWHM
            );
            if (envelope.x.length) {
                traces.push({
                    type: "scatter",
                    mode: "lines",
                    name: `Lorentzian (FWHM ${state.broadeningFWHM} cm⁻¹)`,
                    x:    envelope.x,
                    y:    envelope.y,
                    hoverinfo: "skip",
                    line: { color: "#8ab9e6", width: 1.5 },
                    // Bars on top of the line.
                });
            }
        }
        if (pending.x.length) traces.push({
            type:        "scatter",
            mode:        "markers",
            name:        "Raman pending",
            x:           pending.x,
            y:           pending.y,
            text:        pending.text,
            hoverinfo:   "text",
            marker:      { color: "#888", symbol: "x", size: 7 },
            customdata:  pending.idx,
        });

        const layout = {
            margin:    { t: 28, r: 16, b: 44, l: 56 },
            xaxis:     {
                title: "Frequency (cm⁻¹)",
                zeroline: false,
                gridcolor: "#2c313a",
                color: "#cfd3da",
            },
            yaxis:     {
                title: "Raman activity (Å⁴/amu)",
                rangemode: "tozero",
                gridcolor: "#2c313a",
                color: "#cfd3da",
            },
            plot_bgcolor:  "#1d2128",
            paper_bgcolor: "#1d2128",
            font:          { color: "#cfd3da" },
            barmode:       "overlay",
            legend:        { orientation: "h", y: 1.12 },
            height:        260,
        };

        const config = {
            displaylogo: false,
            responsive:  true,
            modeBarButtonsToRemove: [
                "select2d", "lasso2d", "autoScale2d",
            ],
        };

        Plotly.react(els.spectrumChart, traces, layout, config)
            .then(() => {
                // Wire (or re-wire) the click handler.  Plotly's
                // .react() preserves event listeners across calls,
                // but we attach idempotently for safety -- the .on()
                // de-dupes on the same handler reference.
                els.spectrumChart.removeAllListeners
                    && els.spectrumChart.removeAllListeners("plotly_click");
                els.spectrumChart.on("plotly_click", _onChartClick);
            });
    }

    function _onChartClick(ev) {
        // Plotly's click event carries `points[]`; each point has
        // `customdata` = our mode index for the clicked stick.
        if (!ev || !ev.points || !ev.points.length) return;
        const idx = ev.points[0].customdata;
        if (idx != null) selectMode(Number(idx));
    }

    /* Sum-of-Lorentzians envelope for the spectrum chart.
     *
     * For each mode with finite raman_activity_a4_amu, adds a
     * Lorentzian centered at its frequency, normalised so the
     * peak value equals the mode's activity:
     *
     *     L_i(x) = A_i · γ² / ((x - x_i)² + γ²)
     *
     * where γ = FWHM / 2 is the half-width at half-maximum.
     * Total spectrum is the sum of all L_i.
     *
     * Returns {x, y} arrays sampled on a grid that spans the mode
     * range with a few-cm⁻¹ resolution.  Empty input -> empty
     * arrays so the caller can skip the trace.
     */
    function _lorentzianEnvelope(modes, fwhm) {
        const bright = modes.filter(m =>
            m.raman_activity_a4_amu != null
            && Number.isFinite(Number(m.raman_activity_a4_amu))
        );
        if (!bright.length || fwhm <= 0) return { x: [], y: [] };

        const gamma = fwhm / 2;
        // Grid: extend a few HWHMs past each end of the spectrum so
        // the envelope returns to ~0 at the edges.  Sample density:
        // ~0.2·FWHM, capped to >=1 cm⁻¹.
        let xmin = Infinity, xmax = -Infinity;
        for (const m of bright) {
            const f = Number(m.frequency_cm1);
            if (f < xmin) xmin = f;
            if (f > xmax) xmax = f;
        }
        xmin -= 5 * gamma;
        xmax += 5 * gamma;
        const step = Math.max(1, Math.round(fwhm / 5));
        const n = Math.max(1, Math.floor((xmax - xmin) / step) + 1);
        const x = new Array(n);
        const y = new Array(n).fill(0);
        for (let k = 0; k < n; k++) {
            x[k] = xmin + k * step;
        }
        const gamma2 = gamma * gamma;
        for (const m of bright) {
            const x0 = Number(m.frequency_cm1);
            const A  = Number(m.raman_activity_a4_amu);
            for (let k = 0; k < n; k++) {
                const dx = x[k] - x0;
                y[k] += A * gamma2 / (dx * dx + gamma2);
            }
        }
        return { x: x, y: y };
    }

    function onBroadeningChange() {
        const raw = parseFloat(els.broadeningFwhm.value);
        const v = Number.isFinite(raw) ? Math.max(0, raw) : 0;
        state.broadeningFWHM = v;
        if (state.results) {
            renderSpectrumChart(state.results.modes || []);
        }
    }

    // ----- Bootstrap -------------------------------------------
    function init() {
        els.structureText        = $("structure-text");
        // ``xyz-file`` / ``xyz-load-btn`` / ``xyz-status`` lookups
        // dropped 2026-05-18: those template ids no longer exist
        // (sidebar took over file selection).  loadXyzFile() is also
        // gone; see the comment at the els declaration above.
        els.formContainer  = $("spectra-form-container");
        els.generateBtn    = $("generate-btn");
        els.methodsBtn     = $("methods-preview-btn");
        els.generateStatus = $("generate-status");
        els.priorPath      = $("prior-path");
        els.issuesPanel    = $("issues-panel");
        els.downloadBtn    = $("download-script-btn");
        els.copyBtn        = $("copy-script-btn");
        // Save-to-current-dir button (2026-05-26 Generate/Save split).
        els.saveBtn        = $("save-spectra-btn");
        els.saveStatus     = $("save-status");
        els.scriptPreview  = $("script-preview");
        els.scriptSummary  = $("script-summary");
        // results-file / load-results-btn / results-status lookups
        // dropped for the same reason as xyz* above.
        els.resultsSummary = $("results-summary");
        els.resultsMeta    = $("results-summary-list");
        els.modesTbody     = $("modes-tbody");
        els.spectrumChart  = $("spectrum-chart");
        els.methodsModal   = $("methods-modal");
        els.methodsBody    = $("methods-modal-body");
        els.methodsClose   = $("methods-close-btn");
        els.methodsCopy    = $("methods-copy-btn");
        // Mode-table interactions + ES panel.
        els.modesFilter       = $("modes-filter");
        els.modesCsvBtn       = $("modes-csv-btn");
        els.modesFilterCount  = $("modes-filter-count");
        els.modesTheadRow     = $("modes-thead-row");
        els.esPanel           = $("es-panel");
        els.esModeIdx         = $("es-mode-idx");
        els.esModeFreq        = $("es-mode-freq");
        els.esBarDiagram      = $("es-bar-diagram");
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
        els.modeViewer        = $("mode-viewer");
        els.viewerStatus      = $("viewer-status");
        els.animAmplitude     = $("anim-amplitude");
        els.animAmplitudeVal  = $("anim-amplitude-val");
        els.animSpeed         = $("anim-speed");
        els.animSpeedVal      = $("anim-speed-val");
        els.animToggle        = $("anim-toggle");

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
            _on(els.generateBtn,  "click", generateScript);
            _on(els.methodsBtn,   "click", openMethodsModal);
            _on(els.downloadBtn,  "click", downloadScript);
            _on(els.copyBtn,      "click", copyScript);
            _on(els.methodsClose, "click", closeMethodsModal);
            _on(els.methodsCopy,  "click", copyMethods);
            if (els.saveBtn) {
                _on(els.saveBtn, "click", saveSpectraToCurrentDir);
            }
            // Form-dirty tracking (B.5.3): user-driven input on
            // the param form flips the flag; Generate / Save
            // clear it (params have been emitted).  Delegated
            // listeners on the container so re-renders of the
            // form schema don't need to re-wire each field.
            _on(els.formContainer, "input",
                () => { _formDirty = true; });
            _on(els.formContainer, "change",
                () => { _formDirty = true; });
            _on(els.generateBtn, "click",
                () => { _formDirty = false; });
            if (els.saveBtn) {
                _on(els.saveBtn, "click",
                    () => { _formDirty = false; });
            }
            // Live re-evaluate save-button enable state on sidebar pick.
            const proj = (window.molbuilder || {}).projects;
            if (proj && typeof proj.onChange === "function") {
                const unsub = proj.onChange(refreshSaveSpectraButtonAvailability);
                if (typeof unsub === "function") _cleanups.push(unsub);
            }
            refreshSaveSpectraButtonAvailability();
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

            // Mode-table interactions.
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
         * Tear down every long-lived resource this mount created:
         * the live-watch poller, the mode-animation rAF loop, the
         * 3Dmol viewer's bookkeeping, and Plotly's resize/event
         * observers on the spectrum chart.  After dispose() the
         * rootEl's contents are no longer owned by the inspector;
         * caller may clear/replace freely.
         */
        dispose() {
            // Walk listener teardowns in reverse so the most recent
            // registration tears down first (LIFO -- matches the
            // order they'd be attached in a remount cycle).  Per-
            // cleanup catch keeps one buggy teardown from blocking
            // the rest.
            while (_cleanups.length) {
                try { _cleanups.pop()(); } catch (_) {}
            }
            // Abort in-flight HTTP requests so their then-handlers
            // don't fire against torn-down DOM (registry remount or
            // unmount mid-fetch).
            if (state.loadAbort)   { state.loadAbort.abort();   state.loadAbort = null; }
            if (state.renderAbort) { state.renderAbort.abort(); state.renderAbort = null; }
            if (state.watchTimer) {
                clearInterval(state.watchTimer);
                state.watchTimer = null;
            }
            // #231 Part B: bespoke animation loop (state.animTimer +
            // rAF) is gone -- handle.dispose() tears down the embed's
            // own vibration loop.
            if (state.handle) {
                try { state.handle.dispose(); } catch (_) {}
                state.handle = null;
            }
            if (typeof Plotly !== "undefined" && els.spectrumChart) {
                try { Plotly.purge(els.spectrumChart); } catch (_) {}
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
    };

    }   // ----- end of mountInspector(rootEl, opts) -----

    // Export for both consumers (spectra/viewer.js bootstrap on
    // /spectra, lib/inspectors/spectra.js on /results).  Each
    // consumer is responsible for picking when + where to mount;
    // this module does NOT self-bootstrap on page load.
    root.molbuilder = root.molbuilder || {};
    root.molbuilder.spectraInspector = {
        mount: mountInspector,
        /**
         * Push raw structure bytes into the module's in-memory
         * holder so Generate / Methods read them via
         * ``getStructureText`` without depending on the legacy
         * hidden ``<textarea id="structure-text">``.  Called by
         * spectra/viewer.js's Load-from-sidebar handler after a
         * successful ``/api/files/read``.  Pass an empty string
         * to clear (e.g. on dispose or before a fresh load).
         */
        setStructureText(text) {
            _loadedStructureText = typeof text === "string" ? text : "";
        },
    };

})(typeof window !== "undefined" ? window : this);
