/* Spectra tab front-end controller.
 *
 * Wires the schema-driven SpectraConfig form (via the shared
 * `molbuilder.formSchema` helpers) to the three Spectra API
 * endpoints (spec § 10):
 *
 *   GET  /api/build/schema/spectra      -- form schema
 *   POST /api/spectra/render            -- generate spectra.py
 *   POST /api/spectra/load              -- parse a results JSON
 *
 * v1 scope: structure + form + render + load + Methods-preview
 * modal + Issues panel + results summary table.  Plotly spectrum
 * chart, 3Dmol viewer, and live-watch poller are explicit
 * follow-ups (spec § 9.2 / § 9.5).
 *
 * No build step: ES2017+ JavaScript, no bundler.  Mirrors the
 * structure of static/viewer.js for the Build tab so future
 * readers don't have to learn two layouts.
 */
(function () {
    "use strict";

    const $ = (id) => document.getElementById(id);

    // ----- DOM refs (resolved once at startup) -----------------
    const els = {
        spectrumChart:  null,
        xyzText:        null,
        xyzFile:        null,
        xyzLoadBtn:     null,
        xyzStatus:      null,
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
        resultsFile:    null,
        loadResultsBtn: null,
        resultsStatus:  null,
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
        viewer:         null,    // 3Dmol GLViewer instance (lazy-built)
        animTimer:      null,    // requestAnimationFrame handle
        animPaused:     false,
        animAmplitude:  0.3,     // peak Cartesian amplitude in Å
        animSpeed:      1.0,     // cycle-rate multiplier (1.0 = ~1 Hz)
        animPhase:      0.0,     // current phase in radians
        animLastTs:     null,    // last frame timestamp for dt
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

    // ----- Status helper ----------------------------------------
    function setStatus(el, msg, kind) {
        if (!el) return;
        el.textContent = msg || "";
        el.classList.remove("ok", "error", "muted");
        if (kind) el.classList.add(kind);
    }

    // ----- Form schema load + render ----------------------------
    async function initSchemaForm() {
        const fs = (window.molbuilder || {}).formSchema;
        if (!fs) {
            els.formContainer.innerHTML =
                '<p class="status error">form-schema.js not loaded; '
                + 'check that <code>lib/form-schema.js</code> appears '
                + 'before this script in the template.</p>';
            return;
        }
        try {
            const schema = await fs.fetchSchema("spectra");
            state.schema = schema;
            els.formContainer.innerHTML = "";
            fs.renderForm(els.formContainer, schema);
            wireCompatibilityListeners();
            applyCompatibility();
        } catch (exc) {
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
        if (sel) sel.addEventListener("change", applyCompatibility);
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
            const f = document.getElementById(id);
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

    function getXyz() {
        return (els.xyzText.value || "").trim();
    }

    // ----- XYZ load (file -> textarea) -------------------------
    async function loadXyzFile() {
        const files = els.xyzFile.files;
        if (!files.length) {
            setStatus(els.xyzStatus, "Pick a file first.", "error");
            return;
        }
        const file = files[0];
        try {
            const text = await file.text();
            els.xyzText.value = text;
            setStatus(els.xyzStatus, `Loaded ${file.name}.`, "ok");
        } catch (exc) {
            setStatus(els.xyzStatus, "Read error: " + exc.message, "error");
        }
    }

    // ----- Render button: POST /api/spectra/render -------------
    async function generateScript() {
        const xyz = getXyz();
        if (!xyz) {
            setStatus(els.generateStatus,
                      "Paste an XYZ block or load a file first.",
                      "error");
            return;
        }
        setStatus(els.generateStatus, "Rendering…");
        clearOutputs();

        const body = {
            xyz:    xyz,
            params: collectParams(),
        };
        const priorPath = (els.priorPath.value || "").trim();
        if (priorPath) body.prior_path = priorPath;

        let r;
        try {
            r = await fetch("/api/spectra/render", {
                method:  "POST",
                headers: { "Content-Type": "application/json" },
                body:    JSON.stringify(body),
            }).then(x => x.json().then(b => ({ status: x.status, body: b })));
        } catch (exc) {
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

        // If a Projects sidebar dir is selected (and not the projects/
        // root), also write the script to <current_dir>/<job>.spectra.py.
        // Strict no-overwrite: if the file already exists, the 409
        // message surfaces verbatim.  Download + Copy stay enabled as
        // fallback for the "no dir selected" case.
        //
        // saveToWorkspace is the single source of truth for the
        // generate-and-save flow (lib/projects-sidebar.js); each tab
        // calls it instead of duplicating fetch + refresh logic.
        const proj = (window.molbuilder || {}).projects;
        if (!proj) return;
        const r = await proj.saveToWorkspace(
            state.lastScript, state.lastJobName + ".spectra.py");
        if (!r) return;     // no current_dir / at root -- skip silently
        if (r.ok) {
            setStatus(els.generateStatus, "Wrote " + r.relPath, "ok");
        } else {
            setStatus(els.generateStatus,
                "Generated, but " + r.error
                + " Use Download / Copy below as fallback.",
                "warn");
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
    function renderIssues(issues) {
        const panel = els.issuesPanel;
        if (!issues || issues.length === 0) {
            panel.innerHTML =
                '<p class="status muted">No issues.</p>';
            return;
        }
        // Sort errors first so the user reads the blockers up top.
        const errs  = issues.filter(i => i.severity === "error");
        const warns = issues.filter(i => i.severity === "warn");
        const html = [];
        for (const i of errs.concat(warns)) {
            const cls = (i.severity === "error" ? "issue error"
                                                : "issue warn");
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

    // ----- Load results: POST /api/spectra/load ---------------
    async function loadResults() {
        const files = els.resultsFile.files;
        if (!files.length) {
            setStatus(els.resultsStatus, "Pick a file first.", "error");
            return;
        }
        setStatus(els.resultsStatus, "Parsing…");
        const fd = new FormData();
        fd.append("file", files[0]);
        let r;
        try {
            r = await fetch("/api/spectra/load", { method: "POST", body: fd })
                  .then(x => x.json().then(b => ({ status: x.status, body: b })));
        } catch (exc) {
            setStatus(els.resultsStatus,
                      "Network error: " + exc.message, "error");
            return;
        }
        if (!r.body.ok) {
            // The blueprint includes a `kind` field that maps to the
            // parser's exception class.  Schema mismatches carry
            // expected + actual versions; pass them along.
            let msg = r.body.error || "Load failed.";
            if (r.body.kind === "schema_mismatch") {
                msg = "Schema version mismatch (expected "
                    + r.body.expected_version + ", got "
                    + r.body.actual_version + ").  Update molbuilder "
                    + "or use a matching script version.";
            }
            setStatus(els.resultsStatus, msg, "error");
            return;
        }
        renderResults(r.body.results);
        updatePhaseIndicator(r.body.results);
        setStatus(els.resultsStatus, "Loaded.", "ok");
    }

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
        let r;
        try {
            r = await fetch("/api/spectra/load", {
                method:  "POST",
                headers: { "Content-Type": "application/json" },
                body:    JSON.stringify({ path: path }),
            });
        } catch (exc) {
            setStatus(els.watchStatus,
                      "Network error: " + exc.message, "error");
            return;
        }
        const body = await r.json();
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
        let r;
        try {
            r = await fetch("/api/spectra/load", {
                method:  "POST",
                headers: { "Content-Type": "application/json" },
                body:    JSON.stringify({ path: state.watchPath }),
            });
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
        const body = await r.json();
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

        // Top-of-summary meta dictionary.
        const meta = [
            ["Engine",            results.engine + " " + (results.engine_version || "?")],
            ["Atoms (total)",     results.n_atoms_total],
            ["Free / fixed",      (results.free_atom_idxs || []).length
                                    + " / "
                                    + (results.fixed_atom_idxs || []).length],
            ["Equilibrium E (Eh)", (results.equilibrium &&
                                    Number(results.equilibrium.scf_energy_eh)
                                        .toFixed(8)) || "—"],
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
        document.querySelectorAll(".modes-table .es-col").forEach(th => {
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
        if (state.viewer) {
            _stopAnimation();
            try { state.viewer.clear(); } catch (_) {}
            state.viewer = null;
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
    //   2. Parsed from els.xyzText.value
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
        // Fallback: parse the XYZ in the input form.
        const xyzText = (els.xyzText && els.xyzText.value || "").trim();
        if (!xyzText) return null;
        try {
            return _parseXyz(xyzText);
        } catch (_) {
            return null;
        }
    }

    function _parseXyz(text) {
        // Minimal XYZ parser: line 1 = atom count, line 2 = title,
        // remaining lines = "<element> <x> <y> <z>".  Tolerates
        // extra whitespace / blank trailing lines.
        const lines = text.split(/\r?\n/);
        if (lines.length < 3) throw new Error("xyz too short");
        const n = parseInt(lines[0].trim(), 10);
        if (!Number.isFinite(n) || n < 1) throw new Error("bad atom count");
        const elements = [];
        const positions = [];
        for (let i = 0; i < n; i++) {
            const parts = (lines[i + 2] || "").trim().split(/\s+/);
            if (parts.length < 4) throw new Error("bad atom line " + (i + 2));
            elements.push(parts[0]);
            positions.push([
                parseFloat(parts[1]),
                parseFloat(parts[2]),
                parseFloat(parts[3]),
            ]);
        }
        return { elements: elements, positions: positions };
    }

    function _geomToXyz(elements, positions) {
        // Format an Elements + Positions pair as an XYZ block
        // 3Dmol can ingest.  No title line content is required;
        // a single space placeholder suffices.
        const lines = [String(elements.length), ""];
        for (let i = 0; i < elements.length; i++) {
            const [x, y, z] = positions[i];
            lines.push(`${elements[i]} ${x.toFixed(8)} ${y.toFixed(8)} ${z.toFixed(8)}`);
        }
        return lines.join("\n");
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

        _ensureViewer(geom);
        _startAnimation(geom, mode);
    }

    function _ensureViewer(geom) {
        // Build the 3Dmol viewer once; reuse on subsequent renders.
        // The viewer's internal model is updated per-frame via
        // setAtomCoordinates (cheaper than rebuilding).
        if (state.viewer) return;
        // Clear any "Plotly not loaded"-style fallback content.
        els.modeViewer.innerHTML = "";
        state.viewer = window.$3Dmol.createViewer(els.modeViewer, {
            backgroundColor: "#1d2128",
        });
        const xyz = _geomToXyz(geom.elements, geom.positions);
        state.viewer.addModel(xyz, "xyz");
        state.viewer.setStyle({}, {
            stick:  { radius: 0.15 },
            sphere: { scale: 0.25 },
        });
        // Grey out fixed atoms so the user sees the static anchor.
        const fixed = new Set(
            (state.results.fixed_atom_idxs || []).map(Number)
        );
        if (fixed.size) {
            // 3Dmol atom serial is 1-based; our indices are 0-based.
            for (const idx of fixed) {
                state.viewer.setStyle(
                    { serial: idx + 1 },
                    { sphere: { scale: 0.25, color: "#555" },
                      stick:  { radius: 0.15, color: "#555" } }
                );
            }
        }
        state.viewer.zoomTo();
        state.viewer.render();
    }

    function _startAnimation(geom, mode) {
        // Cancel any previous frame loop, then kick a fresh one.
        _stopAnimation();
        state.animPaused = false;
        state.animPhase = 0;
        state.animLastTs = null;
        if (els.animToggle) els.animToggle.textContent = "Pause";

        // Pre-compute the per-atom displacement vector in (n_atoms, 3)
        // shape.  Free atoms get the mode eigenvector; fixed atoms
        // get zero.  free_atom_idxs maps eigenvector row -> atom.
        const free = state.results.free_atom_idxs || [];
        const evec_free = mode.eigenvector_display;
        const nAtoms = geom.elements.length;
        const displacement = new Array(nAtoms);
        for (let i = 0; i < nAtoms; i++) displacement[i] = [0, 0, 0];
        for (let k = 0; k < free.length; k++) {
            const atomIdx = free[k];
            if (atomIdx >= 0 && atomIdx < nAtoms) {
                displacement[atomIdx] = evec_free[k].slice();
            }
        }

        const eqPos = geom.positions;

        function tick(ts) {
            if (!state.viewer) return;
            if (state.animPaused) {
                state.animLastTs = ts;
                state.animTimer = requestAnimationFrame(tick);
                return;
            }
            if (state.animLastTs != null) {
                const dt = (ts - state.animLastTs) / 1000;   // seconds
                // 1.0× speed = 1 cycle / second = 2π rad/s.
                state.animPhase += 2 * Math.PI * state.animSpeed * dt;
            }
            state.animLastTs = ts;
            const s = Math.sin(state.animPhase);
            const A = state.animAmplitude;
            // 3Dmol's setAtomCoordinates wants {serial, x, y, z}
            // batch update; iterate atoms one at a time using the
            // public ``atoms.x = ...`` mutation pattern instead.
            const atoms = state.viewer.selectedAtoms({});
            for (let i = 0; i < atoms.length && i < nAtoms; i++) {
                atoms[i].x = eqPos[i][0] + A * s * displacement[i][0];
                atoms[i].y = eqPos[i][1] + A * s * displacement[i][1];
                atoms[i].z = eqPos[i][2] + A * s * displacement[i][2];
            }
            state.viewer.render();
            state.animTimer = requestAnimationFrame(tick);
        }
        state.animTimer = requestAnimationFrame(tick);
    }

    function _stopAnimation() {
        if (state.animTimer) cancelAnimationFrame(state.animTimer);
        state.animTimer = null;
        state.animLastTs = null;
    }

    function onAnimAmplitudeChange() {
        const v = parseFloat(els.animAmplitude.value);
        if (Number.isFinite(v)) state.animAmplitude = v;
        if (els.animAmplitudeVal)
            els.animAmplitudeVal.textContent = v.toFixed(2) + " Å";
    }
    function onAnimSpeedChange() {
        const v = parseFloat(els.animSpeed.value);
        if (Number.isFinite(v)) state.animSpeed = v;
        if (els.animSpeedVal)
            els.animSpeedVal.textContent = v.toFixed(1) + "×";
    }
    function onAnimToggle() {
        state.animPaused = !state.animPaused;
        if (els.animToggle)
            els.animToggle.textContent = state.animPaused ? "Play" : "Pause";
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
        els.xyzText        = $("xyz-text");
        els.xyzFile        = $("xyz-file");
        els.xyzLoadBtn     = $("xyz-load-btn");
        els.xyzStatus      = $("xyz-status");
        els.formContainer  = $("spectra-form-container");
        els.generateBtn    = $("generate-btn");
        els.methodsBtn     = $("methods-preview-btn");
        els.generateStatus = $("generate-status");
        els.priorPath      = $("prior-path");
        els.issuesPanel    = $("issues-panel");
        els.downloadBtn    = $("download-script-btn");
        els.copyBtn        = $("copy-script-btn");
        els.scriptPreview  = $("script-preview");
        els.scriptSummary  = $("script-summary");
        els.resultsFile    = $("results-file");
        els.loadResultsBtn = $("load-results-btn");
        els.resultsStatus  = $("results-status");
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

        // xyzLoadBtn + loadResultsBtn were the in-template file-upload
        // buttons that used to flank an <input type=file>.  The
        // Projects sidebar replaces both; the file inputs were dropped
        // from spectra.html.  Null-guard so this initialisation block
        // still runs on the trimmed page.
        if (els.xyzLoadBtn) {
            els.xyzLoadBtn.addEventListener("click", loadXyzFile);
        }
        els.generateBtn.addEventListener("click", generateScript);
        els.methodsBtn.addEventListener("click", openMethodsModal);
        els.downloadBtn.addEventListener("click", downloadScript);
        els.copyBtn.addEventListener("click", copyScript);
        if (els.loadResultsBtn) {
            els.loadResultsBtn.addEventListener("click", loadResults);
        }
        els.loadPathBtn.addEventListener("click", loadByPath);
        els.watchBtn.addEventListener("click", startWatch);
        els.watchStopBtn.addEventListener("click", () => stopWatch("Stopped."));
        // FWHM-controlled broadening re-renders the chart in place.
        if (els.broadeningFwhm) {
            els.broadeningFwhm.addEventListener("input", onBroadeningChange);
            // Read initial value from the input so an
            // HTML-default-modified value (sessionStorage etc.)
            // propagates without needing a manual edit.
            const v = parseFloat(els.broadeningFwhm.value);
            if (Number.isFinite(v) && v >= 0) state.broadeningFWHM = v;
        }

        // 3D viewer control wiring.
        if (els.animAmplitude) {
            els.animAmplitude.addEventListener("input", onAnimAmplitudeChange);
            onAnimAmplitudeChange();
        }
        if (els.animSpeed) {
            els.animSpeed.addEventListener("input", onAnimSpeedChange);
            onAnimSpeedChange();
        }
        if (els.animToggle) {
            els.animToggle.addEventListener("click", onAnimToggle);
        }

        // Mode-table interactions.
        els.modesTheadRow.addEventListener("click", onTableHeaderClick);
        els.modesTbody.addEventListener("click", onTableRowClick);
        els.modesFilter.addEventListener("input", onFilterInput);
        els.modesCsvBtn.addEventListener("click", exportCSV);
        els.methodsClose.addEventListener("click", closeMethodsModal);
        els.methodsCopy.addEventListener("click", copyMethods);

        initSchemaForm();
    }

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", init);
    } else {
        init();
    }
})();
