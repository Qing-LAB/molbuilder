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
    };

    // Last successful render payload.  Holds script + methods_md
    // so the modal + download / copy buttons don't have to re-request.
    const state = {
        schema:         null,
        lastScript:     null,
        lastMethodsMd:  null,
        lastJobName:    null,
    };

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
            "none":      null,
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
        setStatus(els.resultsStatus, "Loaded.", "ok");
    }

    function renderResults(results) {
        if (!results) {
            els.resultsSummary.hidden = true;
            return;
        }
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
            ["Phase: frequencies", results.phase_frequencies],
            ["Phase: Raman",       results.phase_raman],
            ["Phase: ES (L4)",     results.phase_es],
        ];
        els.resultsMeta.innerHTML = meta
            .map(([k, v]) => "<dt>" + escapeHtml(String(k)) + "</dt>"
                           + "<dd>" + escapeHtml(String(v)) + "</dd>")
            .join("");

        // Modes table.
        const rows = (results.modes || []).map(m => {
            const raman = (m.raman_activity_a4_amu === null
                           || m.raman_activity_a4_amu === undefined)
                ? "—"
                : Number(m.raman_activity_a4_amu).toFixed(2);
            return "<tr>"
                + "<td>" + m.index_1based + "</td>"
                + "<td>" + Number(m.frequency_cm1).toFixed(1) + "</td>"
                + "<td>" + raman + "</td>"
                + "<td>" + (m.has_imag ? "✓" : "") + "</td>"
                + "<td>" + (m.electronic_structure ? "✓" : "") + "</td>"
                + "</tr>";
        });
        els.modesTbody.innerHTML = rows.join("");
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
        els.methodsModal   = $("methods-modal");
        els.methodsBody    = $("methods-modal-body");
        els.methodsClose   = $("methods-close-btn");
        els.methodsCopy    = $("methods-copy-btn");

        els.xyzLoadBtn.addEventListener("click", loadXyzFile);
        els.generateBtn.addEventListener("click", generateScript);
        els.methodsBtn.addEventListener("click", openMethodsModal);
        els.downloadBtn.addEventListener("click", downloadScript);
        els.copyBtn.addEventListener("click", copyScript);
        els.loadResultsBtn.addEventListener("click", loadResults);
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
