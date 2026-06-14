/* molbuilder web UI client.
 *
 * Three concerns:
 *   1. POST /api/build/molecule with the user's input -> get back XYZ + meta.
 *   2. Render the XYZ in 3Dmol with style controls.
 *   3. POST /api/build/fdf with the XYZ + form values -> get back FDF text,
 *      offer it as a Blob download.
 */
(function () {
    "use strict";

    const $ = (id) => document.getElementById(id);

    // ----- State ------------------------------------------------------
    const state = {
        xyz: null,            // last successful build's XYZ string
        pdb: null,
        title: null,
        labels: [],           // 3Dmol label objects so we can clear them
        fdf: null,
        pyscf: null,
        // 2026-06-14 viewer-is-truth contract: hold the loaded
        // structure's labels in memory at Load time so the Generate
        // POST can ship them DIRECTLY in the body -- no path-pointer
        // indirection on the server.  Empty defaults mean "no labels
        // on this structure" (an explicit user claim, not a hint to
        // the server to consult disk).  Populated by sidecarLabels
        // .fetch() inside _commitStructure on each successful Load.
        frozen_atoms: [],
        regions: {},
    };

    // Embedded MolViewer (#198, 2026-06-02; contract:
    // docs/protocols/embedded-viewer.md).  Site migration #202
    // landed 2026-06-03 — Build now uses the standard knob bar
    // (Style / Labels / Axes / Reset / PNG / Background / Export)
    // owned by the embed.  The bespoke <details> Style block in
    // index.html is gone; getCamera/setCamera handle the cross-
    // session camera persistence (was raw viewer.getView/setView);
    // the embed's internal ResizeObserver tracks #viewer's
    // resizable container box (was a bespoke RO + window-resize
    // listener).  The handle is the only viewer touchpoint.
    const _handle = window.molbuilder.viewer.embed(
        $("viewer"),
        {
            // No xyz at mount -- Build loads structures asynchronously
            // (via the Build / Load forms or the sidebar handoff);
            // the viewer renders an empty canvas until the first
            // setStructure call inside renderStructure() below.
            style: {
                rep:         "stick",
                radiusScale: 1.0,
            },
            // No pick (the Build viewer is read-only -- the user
            // edits via the build form, not via clicks on atoms).
            pick:   { mode: "none" },
            // Default canonical chrome (knob bar visible above
            // the canvas) per § 7.1 usage pattern.
            card:   { title: "Structure", showInfoLine: true,
                      height: "100%" },
            // Axes default OFF (2026-06-13) — explicit here for the
            // cross-tab consistency rule that every embed mount
            // states the axes default rather than relying on the
            // embed's implicit "axes: opts.axes !== false" treatment
            // (which defaults to TRUE).  Opt in via the knob bar.
            axes:   false,
            // Default export filename when the user clicks Save
            // to project / Download from the Export popover.
            export: { defaultName: "structure" },
            // Surface viewer errors via the existing setStatus
            // helper (defined below) instead of silently logging.
            onError(err) {
                try { setStatus("status", err.message || err.code,
                                "err"); }
                catch (_) {}
            },
        }
    );

    // ----- Status helpers --------------------------------------------
    function setStatus(elId, msg, kind) {
        const el = $(elId);
        el.textContent = msg;
        el.className = "status" + (kind ? " " + kind : "");
    }

    // Render a structured issues panel from a list of {severity,
    // message, where} dicts.  Hides the panel when there are no
    // issues; otherwise emits one <li class="issue-item"
    // data-severity="..."> per issue, with the where-tag floated to
    // the right.  Used both by the live preflight loop and by the
    // post-Generate response handler.
    // Render one Issue <li> into the supplied <ul>.  Shared between
    // the residual panel + the per-card panels.
    function _appendIssueItem(ul, issue) {
        const li = document.createElement("li");
        li.className = "issue-item";
        li.setAttribute("data-severity", issue.severity || "info");
        const msg = document.createElement("span");
        msg.className = "issue-msg";
        msg.textContent = issue.message || "";
        li.appendChild(msg);
        if (issue.where) {
            const tag = document.createElement("span");
            tag.className = "issue-where";
            tag.textContent = issue.where;
            li.appendChild(tag);
        }
        ul.appendChild(li);
    }

    // Per docs/protocols/web-ui-coherence.md Rule 2, issues that
    // carry a ``workflow_group`` tag (profile / stage / budget)
    // render INSIDE the relevant workflow-group card; un-tagged
    // issues (geometry / cell / polymer / legacy untagged config
    // fields) render in the legacy residual panel referenced by
    // ``panelId``.  ``formContainerId`` scopes the per-card render
    // to one engine's form so siesta + pyscf issues don't cross-
    // pollinate when the two forms share the page.
    function renderIssues(panelId, issues, formContainerId) {
        const list = (issues || []).filter(Boolean);
        // Fan grouped issues into per-card panels first.  Always
        // clear every per-card panel (whether or not we'll write to
        // it) so a stale finding from a previous render disappears.
        const form = formContainerId ? $(formContainerId) : null;
        const cardPanels = form
            ? form.querySelectorAll(".card-issues[data-workflow-group]")
            : [];
        for (const cp of cardPanels) {
            cp.innerHTML = "";
            cp.hidden = true;
        }
        const residual = $(panelId);
        if (residual) {
            residual.innerHTML = "";
            residual.hidden = true;
        }
        if (!list.length) return;
        // Split: per-group buckets + residual.
        const buckets = new Map();   // role → array
        const residualList = [];
        for (const issue of list) {
            const g = issue.workflow_group;
            if (g && form) {
                if (!buckets.has(g)) buckets.set(g, []);
                buckets.get(g).push(issue);
            } else {
                residualList.push(issue);
            }
        }
        // Write per-card buckets.
        for (const cp of cardPanels) {
            const role = cp.getAttribute("data-workflow-group");
            const bucket = buckets.get(role);
            if (!bucket || !bucket.length) continue;
            for (const issue of bucket) _appendIssueItem(cp, issue);
            cp.hidden = false;
        }
        // Write residual.
        if (residual && residualList.length) {
            for (const issue of residualList) _appendIssueItem(residual, issue);
            residual.hidden = false;
        }
    }

    function clearIssues(panelId, formContainerId) {
        const ul = $(panelId);
        if (ul) { ul.innerHTML = ""; ul.hidden = true; }
        const form = formContainerId ? $(formContainerId) : null;
        if (form) {
            for (const cp of form.querySelectorAll(
                                ".card-issues[data-workflow-group]")) {
                cp.innerHTML = "";
                cp.hidden = true;
            }
        }
    }

    // Debounce helper -- collapses a burst of form-input events into
    // a single call after `wait` ms of quiet.  Used to throttle the
    // live /api/build/preflight calls so dragging a slider doesn't
    // spam the endpoint.
    function debounce(fn, wait) {
        let t = null;
        return function (...args) {
            if (t) clearTimeout(t);
            t = setTimeout(() => fn.apply(this, args), wait);
        };
    }

    // Live preflight: when the user has already built a structure
    // (state.xyz exists) and they tweak any tab's params, hit
    // /api/build/preflight to refresh the issues panel before they
    // click Generate.  No-op until the first successful build, so a
    // user who just landed on the page and is fiddling with form
    // defaults doesn't see warnings that haven't been earned yet.
    async function refreshPreflight(engine) {
        if (!state.xyz) return;
        const params = (engine === "siesta")
            ? collectFdfParams()
            : collectPyscfParams();
        const panelId = (engine === "siesta") ? "fdf-issues" : "pyscf-issues";
        const formContainerId = (engine === "siesta")
            ? "siesta-form-container" : "pyscf-form-container";
        try {
            const r = await fetch("/api/build/preflight", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ xyz: state.xyz, engine, params }),
            }).then(x => x.json());
            if (r.ok) renderIssues(panelId, r.issues, formContainerId);
        } catch (e) {
            // Network error during preflight is not surfaced -- the
            // panel stays in its previous state.  The Generate path
            // (which gets the same issues from the render endpoint)
            // is the canonical source of truth.
        }
    }
    const refreshPreflightDebounced = {
        siesta: debounce(() => refreshPreflight("siesta"), 250),
        pyscf:  debounce(() => refreshPreflight("pyscf"),  250),
    };

    // The Build form (kind / input-text / backend / form / terminal /
    // add-hydrogens / source-mode toggle) was retired on 2026-06-08
    // with task #295 — the Optimization tab is file-driven now, and
    // structures arrive via the Projects sidebar (the "Load from
    // sidebar selection" button + mount-time sessionStorage handoff
    // wired further below).  The placeholderFor / toggleNucleicOptions
    // / applySourceMode / BACKEND_LABEL / build click / load-file
    // upload helpers that lived in this block were deleted in the
    // same pass; the Molbuilder tab carries the equivalent generators
    // (see molbuilder/web/static/modify/viewer.js).

    // Take a structure response from /api/build/load and populate
    // the viewer + info panel + enable the FDF section.
    // A new structure invalidates any previously-generated FDF / PySCF
    // outputs -- we clear those and disable their download buttons so
    // the user can't accidentally download stale text from the prior
    // structure.
    /* Compute the BlockSize molbuilder's backend would auto-pick for a
       structure of n_atoms.  Mirrors `_auto_block_size` in
       molbuilder/siesta.py -- if either side changes the rule, the
       other must follow.  Used only to label the BlockSize textbox's
       placeholder; the actual value still comes from the backend. */
    function autoBlockSize(n) {
        if (n >= 16) return 8;
        if (n >= 8)  return 4;
        if (n >= 4)  return 2;
        return 1;
    }

    function clearStructureInfo(reason) {
        // Wipe the info readout when a load attempt FAILS or when
        // the user picks a non-loadable file in the sidebar.  Without
        // this, the readout keeps showing the LAST successful load
        // ("3 atoms, formula H2O") even after the user picked a
        // junction.pdb that failed to parse -- a lie to the user
        // about what's currently loaded.  ``reason`` is the
        // short string to show in #info-title (e.g. "load failed",
        // "no structure built").  The atoms/residues/formula slots
        // go to the em-dash placeholder so the user can't read
        // numbers off them.
        const title = $("info-title");
        const atoms = $("info-atoms");
        const resi  = $("info-residues");
        const form  = $("info-formula");
        if (title) title.textContent = reason || "no structure built";
        if (atoms) atoms.textContent = "—";   // em-dash
        if (resi)  resi.textContent  = "—";
        if (form)  form.textContent  = "—";
    }

    function applyStructureResult(r) {
        state.xyz = r.xyz;
        state.pdb = r.pdb;
        state.title = r.title;
        state.fdf = null;
        state.pyscf = null;
        // Stash the full response in case a future feature needs
        // to replay it without re-fetching (e.g. an in-tab undo
        // step on the Generate-input form).  Pre-task-#306 this
        // was load-bearing for the now-retired ``builder-structure``
        // sessionStorage round-trip; today it's an inexpensive
        // bookkeeping slot.
        state.last_response = r;
        $("info-title").textContent     = r.title;
        $("info-atoms").textContent     = r.n_atoms;
        $("info-residues").textContent  = r.n_residues || "—";
        $("info-formula").textContent   = formula(r.elements);
        // Update the BlockSize textbox's placeholder to show the
        // auto-picked value for this structure.  Empty input still
        // means "use auto"; the placeholder just makes it visible.
        // Defensive: the schema-driven SIESTA form may not have
        // rendered yet on the very first applyStructureResult()
        // call after a navigation.  Skip silently; the placeholder
        // is just a hint, not load-bearing.
        const bs = $("p-block-size");
        if (bs) {
            bs.placeholder =
                "auto (" + autoBlockSize(r.n_atoms) + ", n=" + r.n_atoms + ")";
        }
        $("generate-fdf").disabled = false;
        $("generate-pyscf").disabled = false;
        // Stale outputs / status / download buttons -> reset
        $("dl-fdf").disabled = true;
        $("dl-pyscf").disabled = true;
        $("fdf-output").hidden = true;
        $("fdf-output").textContent = "";
        $("pyscf-output").hidden = true;
        $("pyscf-output").textContent = "";
        setStatus("fdf-status", "");
        setStatus("pyscf-status", "");
        // A new structure invalidates the previous run's issue panels;
        // refresh both with a preflight tick so the user sees only
        // issues against the new geometry + current params.
        clearIssues("fdf-issues", "siesta-form-container");
        clearIssues("pyscf-issues", "pyscf-form-container");
        refreshPreflightDebounced.siesta();
        refreshPreflightDebounced.pyscf();
        renderStructure();
    }

    // ----- Tabs (SIESTA / PySCF) -------------------------------------
    document.querySelectorAll(".tab-btn").forEach(btn => {
        btn.addEventListener("click", () => {
            const target = btn.dataset.tab;
            document.querySelectorAll(".tab-btn").forEach(b => {
                const active = (b === btn);
                b.classList.toggle("active", active);
                b.setAttribute("aria-selected", active ? "true" : "false");
            });
            document.querySelectorAll(".tab-panel").forEach(p => {
                p.hidden = (p.id !== "tab-" + target);
            });
        });
    });

    // ----- Parameter compatibility rules -----------------------------
    //
    // When one input's value makes another field meaningless or
    // forbidden (e.g. method=RKS forces spin=0), the dependent field
    // gets disabled with a "(locked: ...)" hint explaining why.  The
    // hints update live as the user changes options.
    //
    // Each rule is a function that reads triggering inputs and calls
    // setLock(elementId, lockReason | null) to toggle the lock state.
    // setLock(id, null) unlocks; setLock(id, "<text>") locks + hint.

    function setLock(elId, reason) {
        const el = $(elId);
        if (!el) return;
        const label = el.closest("label");
        // Find or create the .lock-reason span as the last child of
        // the enclosing <label>.
        let hint = label && label.querySelector(":scope > .lock-reason");
        if (!hint && label) {
            hint = document.createElement("span");
            hint.className = "lock-reason";
            hint.hidden = true;
            label.appendChild(hint);
        }
        if (reason === null) {
            el.disabled = false;
            if (label) label.classList.remove("is-locked");
            if (hint)  hint.hidden = true;
        } else {
            el.disabled = true;
            if (label) label.classList.add("is-locked");
            if (hint) {
                hint.textContent = reason;
                hint.hidden = false;
            }
        }
    }

    // ---- PySCF rules -------------------------------------------------
    function applyPyscfCompatibility() {
        // Method <-> Spin: restricted methods (RKS/RHF) require spin=0.
        const method = $("py-method") ? $("py-method").value : null;
        const restricted = (method === "RKS" || method === "RHF");
        if (restricted) {
            $("py-spin").value = "0";
            setLock("py-spin",
                "Restricted methods (RKS/RHF) require spin=0. Switch to "
                + "UKS/UHF for open-shell systems.");
        } else {
            setLock("py-spin", null);
        }

        // optimize=false -> entire optimization + pre-opt sections moot.
        const optimize = $("py-optimize") && $("py-optimize").checked;
        const optReason = optimize ? null
            : "Geometry optimization is disabled (set 'Optimize geometry' on).";
        ["py-optimizer", "py-geom-max-steps",
         "py-geom-conv-energy", "py-geom-conv-grms",
         "py-geom-conv-gmax"].forEach(id => setLock(id, optReason));

        // Pre-opt fields: depend on optimize=true AND preopt=true.
        const preopt = $("py-preopt") && $("py-preopt").checked;
        let preoptReason;
        if (!optimize) {
            preoptReason = "Geometry optimization is disabled.";
        } else if (!preopt) {
            preoptReason =
                "Pre-optimization is disabled (tick 'Enable pre-optimization').";
        } else {
            preoptReason = null;
        }
        ["py-preopt-functional", "py-preopt-basis",
         "py-preopt-max-steps", "py-preopt-grms"].forEach(id =>
            setLock(id, preoptReason));
        // The 'Enable pre-optimization' checkbox itself depends only on optimize.
        setLock("py-preopt", optimize ? null
                                       : "Geometry optimization is disabled.");

        // Solvent <-> solvent_method: method only meaningful when a
        // solvent is selected.
        const solv = $("py-solvent") && $("py-solvent").value;
        setLock("py-solvent-method",
                (!solv || solv === "")
                    ? "No solvent selected (gas phase)."
                    : null);
    }

    // ---- SIESTA rules ------------------------------------------------
    function applySiestaCompatibility() {
        // SpinTotal only meaningful when SpinPolarized is on.
        const spinPol = $("p-spin-polarized")
            && $("p-spin-polarized").checked;
        setLock("p-spin-total",
                spinPol ? null
                        : "Tick 'Spin polarized' first; SpinTotal is "
                          + "ignored without spin polarisation.");

        // Relaxation type "none" -> per-step relaxation params moot.
        const relax = $("p-relax") && $("p-relax").value;
        const noneReason =
            (relax === "none")
                ? "Single-point only (no MD block emitted in the FDF)."
                : null;
        ["p-relax-steps", "p-force-tol", "p-max-displ"]
            .forEach(id => setLock(id, noneReason));
    }

    function applyCompatibility() {
        applyPyscfCompatibility();
        applySiestaCompatibility();
    }

    // The compat-engine wiring + initial run move into
    // initFormsFromSchema() below: the form's inputs only exist
    // AFTER the schema-driven renderer fills the containers, so
    // attaching listeners at module-load (when the containers are
    // empty <div>s) wouldn't find anything.  Same reason
    // restoreFormState() and the first applyCompatibility() are
    // deferred.
    function wireCompatibilityListeners() {
        [
            "py-method", "py-optimize", "py-preopt", "py-solvent",
            "p-spin-polarized", "p-relax",
        ].forEach(id => {
            const el = $(id);
            if (el) el.addEventListener("change", applyCompatibility);
        });
    }

    // Module-level cache of the form schemas, populated once on
    // page load by initFormsFromSchema().  collectFdfParams() and
    // collectPyscfParams() read from this; getFormIds() walks both
    // schemas to build the persistence ID list.
    const formSchemas = { siesta: null, pyscf: null };

    async function initFormsFromSchema() {
        const fs = (window.molbuilder || {}).formSchema;
        if (!fs) {
            console.error(
                "form-schema.js not loaded; build form will not appear"
            );
            return;
        }
        try {
            const [siesta, pyscf] = await Promise.all([
                fs.fetchSchema("siesta"),
                fs.fetchSchema("pyscf"),
            ]);
            formSchemas.siesta = siesta;
            formSchemas.pyscf  = pyscf;
            const siestaC = $("siesta-form-container");
            const pyscfC  = $("pyscf-form-container");
            if (siestaC) fs.renderForm(siestaC, siesta);
            if (pyscfC)  fs.renderForm(pyscfC,  pyscf);
        } catch (exc) {
            console.error("could not load build-form schema:", exc);
            // Surface a visible failure so the user knows the
            // server's /api/build/schema endpoint is unreachable.
            for (const id of ["siesta-form-container", "pyscf-form-container"]) {
                const c = $(id);
                if (c) c.innerHTML =
                    '<p class="status error">Could not load form schema: '
                    + String(exc).replace(/[&<>]/g, c =>
                        ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' }[c]))
                    + '</p>';
            }
            return;
        }
        // restoreFormState BEFORE applyCompatibility so compat rules
        // see the restored values when computing locks.
        restoreFormState();
        wireCompatibilityListeners();
        wirePreflightListeners();
        applyCompatibility();
    }

    // ----- Sidebar-driven loading (Projects sidebar -> Build) ------- //
    //
    // The Projects sidebar publishes its current pick via
    // ``window.molbuilder.projects.onChange``.  Each onChange fire
    // brings ``{dir, file}``; when ``file`` ends with ``.xyz`` or
    // ``.pdb`` we fetch its content via /api/files/read and POST to
    // /api/build/load (same endpoint the file-upload button uses),
    // so picking a structure in the sidebar auto-loads it into
    // Build without making the user re-upload via the OS dialog.
    //
    // The previous behaviour was "Build doesn't listen to the
    // sidebar at all" -- which broke the natural workflow of
    // "navigate to my project, click my .pdb, see it in Build".
    // /modify already had this wiring; this brings Build to parity.
    //
    // Race safety: every subscribe fire takes a monotonic seq; if a
    // later pick supersedes this one before the two-step fetch
    // finishes, we discard the older response.  ``lastLoadedFile``
    // also debounces against same-file refires (onChange publishes
    // on every set, not only on diffs).
    let _sidebarLoadSeq = 0;
    let _sidebarLastFile = "";

    // Subscribe via the module-init contract (design.md "Module init
    // contract"): the projects-sidebar module loads as
    // ``<script type="module">`` (deferred), so it's NOT available
    // when this classic-script viewer.js's IIFE runs.  We wait for
    // ``runtime.whenReady("projects")`` to resolve -- replaces the
    // earlier polling hack with a structural answer.  If the runtime
    // isn't loaded (legacy / test-isolation path), fall back to a
    // simple "skip the wiring" so the rest of viewer.js still works.
    // ----- Form-dirty tracking (B.5.3) ------------------------------
    //
    // Per the universal sidebar interaction model: sidebar commit
    // (dblclick) on a structure file should rebuild this form's
    // structure section.  But if the user has already typed
    // SIESTA/PySCF parameters, a silent rebuild discards those.
    // Track "the user has edited the param section since the last
    // commit/Generate" and gate the next commit through a warning.
    //
    // Listens on both engine containers — the `input` event covers
    // text inputs + textareas + sliders; `change` covers selects +
    // checkboxes.  Each container's inputs are repopulated by
    // form-schema rendering on engine switch, which we deliberately
    // do NOT treat as a user edit (the events fire synchronously
    // during the programmatic .value assignments and would set
    // _formDirty=true spuriously); we reset _formDirty AFTER each
    // rebuild via the same mechanism as a successful sidebar commit.
    let _formDirty = false;
    let _ignoreFormChanges = false;
    function _wireFormDirtyTracking() {
        const cIds = ["siesta-form-container", "pyscf-form-container"];
        for (const id of cIds) {
            const c = document.getElementById(id);
            if (!c) continue;
            const handler = () => { if (!_ignoreFormChanges) _formDirty = true; };
            c.addEventListener("input",  handler);
            c.addEventListener("change", handler);
        }
    }
    document.addEventListener("DOMContentLoaded", _wireFormDirtyTracking);

    function _resetFormDirty() { _formDirty = false; }
    // Generate buttons consume the form values — clear dirty so the
    // next sidebar commit doesn't warn for params the user already
    // emitted into a script.
    document.addEventListener("DOMContentLoaded", () => {
        for (const id of ["generate-fdf", "generate-pyscf"]) {
            const b = document.getElementById(id);
            if (b) b.addEventListener("click", _resetFormDirty);
        }
    });

    if (window.molbuilder && window.molbuilder.runtime
        && typeof window.molbuilder.runtime.whenReady === "function") {
        window.molbuilder.runtime.whenReady("projects").then((_proj) => {
        // The "use this file in Build" handler.  Defined once and
        // wired to both the universal commit channel AND the page-
        // mount cross-tab handoff (a structure file already in
        // sessionStorage when /structure-optimization mounts should
        // be treated as a commit; the user picked it on another tab
        // and navigated here to configure SIESTA / PySCF for it).
        async function _commitStructure(sel) {
            const f = (sel && sel.file) ? String(sel.file) : "";
            const ext = f.toLowerCase().split(".").pop();
            if (ext !== "xyz" && ext !== "pdb") {
                // Commit on a non-structure file — tell the user
                // what they double-clicked isn't loadable into
                // Build; leave the existing structure alone.
                if (f) {
                    const filename = f.split("/").pop();
                    setStatus("load-status",
                        `${filename} is not a structure file `
                        + `(Build accepts .xyz / .pdb only).`,
                        "warn");
                }
                return;
            }
            if (f === _sidebarLastFile) return;
            // Form-dirty gate: if the user has typed parameter
            // edits since the last commit/Generate, ask before
            // discarding them via the shared warning modal.
            const modal = window.molbuilder
                       && window.molbuilder.warningModal;
            if (_formDirty && modal
                && typeof modal.confirmDiscardUnsaved === "function") {
                const proceed = await modal.confirmDiscardUnsaved();
                if (!proceed) return;
            }
            _formDirty = false;
            _sidebarLastFile = f;
            const mySeq = ++_sidebarLoadSeq;
            const filename = f.split("/").pop();
            setStatus("load-status", `Loading ${filename}…`);
            let text;
            try {
                const rr = await fetch(
                    "/api/files/read?path=" + encodeURIComponent(f)
                );
                const rb = await rr.json();
                if (!rr.ok || !rb.ok) {
                    setStatus("load-status",
                        rb.error || "Read failed.", "error");
                    clearStructureInfo("load failed: " + filename);
                    return;
                }
                text = rb.text;
            } catch (e) {
                setStatus("load-status",
                    "Network error reading " + filename + ": " + e.message,
                    "error");
                clearStructureInfo("load failed: " + filename);
                return;
            }
            if (mySeq !== _sidebarLoadSeq) return;     // superseded
            try {
                const r = await fetch("/api/build/load", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({
                        text: text,
                        filename: filename,
                        format: ext,
                    }),
                }).then((x) => x.json());
                if (mySeq !== _sidebarLoadSeq) return;
                if (!r.ok) {
                    setStatus("load-status",
                        r.error || "Load failed.", "error");
                    clearStructureInfo("load failed: " + filename);
                    return;
                }
                applyStructureResult(r);
                // 2026-06-14 contract: pull the file's sidecar
                // labels into in-memory state.  These travel with
                // the bytes when Generate fires; no server-side
                // disk re-read.  ``sidecarLabels.fetch`` always
                // resolves (missing sidecar -> empty defaults), so
                // this Promise can't reject the Load.
                const _sl = window.molbuilder
                         && window.molbuilder.sidecarLabels;
                if (_sl && typeof _sl.fetch === "function") {
                    try {
                        const labels = await _sl.fetch(f);
                        state.frozen_atoms = labels.frozen_atoms || [];
                        state.regions     = labels.regions || {};
                    } catch (_) {
                        // Defensive -- helper is documented as never
                        // throwing; if it does, default to empty
                        // rather than silently keep stale labels
                        // from the prior load.
                        state.frozen_atoms = [];
                        state.regions     = {};
                    }
                } else {
                    state.frozen_atoms = [];
                    state.regions     = {};
                }
                setStatus("load-status",
                    `Loaded ${r.n_atoms}-atom ${ext.toUpperCase()} `
                    + `from ${filename}.`, "ok");
                // Flip the load-bar readout to "Loaded: <name>" so
                // the user sees the pick is in the viewer (not just
                // selected in the sidebar).  Defined further below in
                // the same scope; reachable via function hoisting.
                if (typeof _refreshLoadButton === "function") {
                    _refreshLoadButton();
                }
                if (typeof _refreshAutoDetectButton === "function") {
                    _refreshAutoDetectButton();
                }
                // Phase 3 (2026-06-10): auto-fire the analyzer so
                // the chemistry rationale is visible by default,
                // not gated behind a button click.  Forms are NOT
                // pre-filled — the user must click Auto-detect to
                // apply suggestions.  This closes the gap where a
                // user who ignored the button saw no chemistry
                // analysis until Generate-time (when the validator
                // fired).  See scientific-validation.md § 2.5.
                if (typeof _autoAnalyzeOnLoad === "function") {
                    _autoAnalyzeOnLoad(_sidebarLastFile);
                }
            } catch (e) {
                if (mySeq !== _sidebarLoadSeq) return;
                setStatus("load-status",
                    "Network error: " + e.message, "error");
                clearStructureInfo("load failed: " + filename);
            }
        }   // close _commitStructure

        // Universal commit subscription — dblclick on a structure
        // file in the sidebar.  Falls back to onChange for older
        // deployments without onCommit so the tab stays usable.
        const subscribe = (typeof _proj.onCommit === "function")
            ? _proj.onCommit.bind(_proj)
            : _proj.onChange.bind(_proj);
        subscribe(_commitStructure);

        // Cross-tab handoff: if the user arrived here with a
        // sessionStorage selection already in place (the Molbuilder
        // tab's save-first Send button, task #294, or the user
        // having picked a file in the sidebar before navigating),
        // commit it once on mount so the viewer + Generate buttons
        // come up populated.  The "Load from sidebar selection"
        // button below is the EXPLICIT entry point for subsequent
        // re-loads after a sidebar pick; the mount-time commit is
        // the cross-tab handoff path.
        const _initialFile = (typeof _proj.getCurrentFile === "function")
            ? _proj.getCurrentFile() : "";
        if (_initialFile) {
            const _initialDir = (typeof _proj.getCurrentDir === "function")
                ? _proj.getCurrentDir() : "";
            _commitStructure({ dir: _initialDir, file: _initialFile });
        }

        // ----- Load-from-sidebar button (task #295, 2026-06-08) ---- //
        //
        // The structure-optimization tab's sole structure entry
        // point.  Reads the current Projects-sidebar pick and
        // commits it through ``_commitStructure``.  Click is the
        // explicit user gesture; the button surfaces the current
        // pick + enables only when it's a loadable .xyz/.pdb.
        const _isLoadable = (name) => {
            const n = String(name || "").toLowerCase();
            return n.endsWith(".xyz") || n.endsWith(".pdb");
        };
        const _basename = (p) => {
            const ix = String(p || "").lastIndexOf("/");
            return ix >= 0 ? p.slice(ix + 1) : p;
        };
        let _candidatePath = "";
        // Workspace dirty state — surfaced in the readout so the user
        // sees "Loaded: X · unsaved changes" after they modify the
        // structure but before saving.  Read lazily so this code
        // works during early init when ws may not be wired yet.
        const _wsDirty = () => {
            const w = (window.molbuilder || {}).workspace;
            try { return !!(w && w.getState && w.getState().dirty); }
            catch (_e) { return false; }
        };
        function _refreshLoadButton() {
            const btn = $("load-from-sidebar-btn");
            const readout = $("load-source-readout");
            if (!btn) return;
            const loadable = _isLoadable(_candidatePath);
            btn.disabled = !loadable;
            if (readout) {
                const isLoaded = loadable
                    && _sidebarLastFile === _candidatePath;
                if (isLoaded) {
                    readout.textContent = _wsDirty()
                        ? `Loaded: ${_basename(_candidatePath)} · unsaved changes`
                        : `Loaded: ${_basename(_candidatePath)}`;
                } else if (loadable) {
                    readout.textContent =
                        `Selected: ${_basename(_candidatePath)}`;
                } else if (_candidatePath) {
                    readout.textContent =
                        `Selected: ${_basename(_candidatePath)} (not loadable)`;
                } else {
                    readout.textContent =
                        "Pick a .xyz / .pdb in the Projects sidebar.";
                }
            }
        }
        _candidatePath = _initialFile;
        _refreshLoadButton();
        if (typeof _proj.onChange === "function") {
            _proj.onChange((sel) => {
                _candidatePath = (sel && sel.file) ? sel.file : "";
                _refreshLoadButton();
            });
        }
        // Re-render the readout whenever workspace dirty state flips.
        // Subscribe defensively — ws may not exist on pages that don't
        // mount the canvas (e.g. spectra-only views).
        (function _subscribeWsDirty() {
            const w = (window.molbuilder || {}).workspace;
            if (w && typeof w.subscribe === "function") {
                w.subscribe(_refreshLoadButton);
            }
        })();
        const _loadBtn = $("load-from-sidebar-btn");
        if (_loadBtn) {
            _loadBtn.addEventListener("click", () => {
                if (!_isLoadable(_candidatePath)) return;
                const _dir = (typeof _proj.getCurrentDir === "function")
                    ? _proj.getCurrentDir() : "";
                _commitStructure({ dir: _dir, file: _candidatePath });
            });
        }

        // -------- Auto-detect button (Phase 2 of the chemistry
        // middle-layer work; see
        // docs/protocols/scientific-validation.md § 2.5).
        //
        // Posts the currently-loaded structure to
        // /api/structure/analyze, then applies the engine-agnostic
        // ChemistryAnalysis's per-engine translation onto BOTH the
        // SIESTA + PySCF sub-forms in a single click.  Rationale +
        // warnings are surfaced in the auto-detect-panel <details>
        // so the user can see what was decided before generating.
        //
        // Disabled when no structure is loaded.  The endpoint
        // accepts the same file path used by /api/build/load above,
        // so we read it from _sidebarLastFile (the path the user
        // most-recently committed via the Load button or sidebar
        // double-click).
        //
        // Concurrency safety: _autoDetectSeq mirrors _sidebarLoadSeq.
        // If the user clicks Auto-detect, then loads a different
        // structure while the request is in flight, the in-flight
        // response would otherwise apply to the new structure's
        // form.  We snapshot the seq at request time and discard
        // the response if a newer load (or another auto-detect) has
        // happened since.
        function _refreshAutoDetectButton() {
            const btn = $("auto-detect-btn");
            if (!btn) return;
            btn.disabled = !_sidebarLastFile;
        }
        _refreshAutoDetectButton();
        let _autoDetectSeq = 0;

        const _autoBtn = $("auto-detect-btn");
        if (_autoBtn) {
            _autoBtn.addEventListener("click", async () => {
                if (!_sidebarLastFile) return;
                // Snapshot BOTH the path AND the load-seq so a
                // mid-flight structure swap is detectable.
                const mySeq      = ++_autoDetectSeq;
                const myLoadSeq  = _sidebarLoadSeq;
                const myPath     = _sidebarLastFile;
                _autoBtn.disabled = true;
                setStatus("auto-detect-status", "Analyzing…");
                let body;
                try {
                    const r = await fetch("/api/structure/analyze", {
                        method:  "POST",
                        headers: { "Content-Type": "application/json" },
                        body:    JSON.stringify({
                            structure_path: myPath,
                        }),
                    });
                    body = await r.json();
                    if (mySeq !== _autoDetectSeq
                        || myLoadSeq !== _sidebarLoadSeq) {
                        // Superseded — a newer click or load happened
                        // while this request was in flight.  Drop the
                        // response silently; the newer request owns
                        // the UI now.
                        return;
                    }
                    if (!r.ok || !body.ok) {
                        setStatus("auto-detect-status",
                            body && body.error
                                ? body.error
                                : `Analyze failed (HTTP ${r.status}).`,
                            "error");
                        return;
                    }
                } catch (e) {
                    if (mySeq !== _autoDetectSeq
                        || myLoadSeq !== _sidebarLoadSeq) return;
                    setStatus("auto-detect-status",
                        "Network error: "
                        + (e && e.message ? e.message : String(e)),
                        "error");
                    return;
                } finally {
                    // Re-enable only if this is still the latest
                    // click — otherwise the newer one owns the
                    // button state.
                    if (mySeq === _autoDetectSeq
                        && myLoadSeq === _sidebarLoadSeq) {
                        _refreshAutoDetectButton();
                    }
                }
                _applyAutoDetectToForms(body);
                _renderAutoDetectPanel(body);
                setStatus("auto-detect-status",
                    "Applied to both forms.  Review rationale below.",
                    "ok");
            });
        }

        /**
         * Phase 3 auto-analyze (fired automatically from
         * _commitStructure on every successful load).  Hits the
         * same /api/structure/analyze endpoint as the button, but
         * does NOT apply the result to the forms — only renders
         * the rationale panel + warnings so the user sees the
         * chemistry conclusions without lifting a finger.
         *
         * Form-fill stays gated behind the explicit button click
         * so we never silently mutate the user's params; we just
         * surface the science.
         */
        async function _autoAnalyzeOnLoad(path) {
            if (!path) return;
            const mySeq      = ++_autoDetectSeq;
            const myLoadSeq  = _sidebarLoadSeq;
            try {
                const r = await fetch("/api/structure/analyze", {
                    method:  "POST",
                    headers: { "Content-Type": "application/json" },
                    body:    JSON.stringify({ structure_path: path }),
                });
                const body = await r.json();
                if (mySeq !== _autoDetectSeq
                    || myLoadSeq !== _sidebarLoadSeq) return;
                if (!r.ok || !body.ok) {
                    // Silent failure — the user can still click
                    // the button to retry; we don't want to
                    // flash an error for a background analyze
                    // they didn't ask for.
                    return;
                }
                _renderAutoDetectPanel(body);
                setStatus("auto-detect-status",
                    "Chemistry analyzed — click Auto-detect to "
                    + "apply suggested defaults to the forms.",
                    null);
            } catch (_) {
                // Same silent-failure rationale — background fire.
            }
        }

        /**
         * Spread the analyze response's suggested.<engine> blocks
         * onto both engine sub-forms via formSchema.setValues.  The
         * setter dispatches input/change events so the form-dirty
         * tracker sees the programmatic edit.
         */
        function _applyAutoDetectToForms(resp) {
            const fs = (window.molbuilder || {}).formSchema;
            if (!fs || typeof fs.setValues !== "function") return;
            const sug = (resp && resp.suggested) || {};
            const siestaEl = $("siesta-form-container");
            const pyscfEl  = $("pyscf-form-container");
            if (siestaEl && formSchemas.siesta && sug.siesta) {
                fs.setValues(siestaEl, formSchemas.siesta, sug.siesta);
            }
            if (pyscfEl && formSchemas.pyscf && sug.pyscf) {
                fs.setValues(pyscfEl, formSchemas.pyscf, sug.pyscf);
            }
        }

        function _renderAutoDetectPanel(resp) {
            const panel = $("auto-detect-panel");
            if (!panel) return;
            panel.hidden = false;
            panel.open = true;
            const ratEl  = $("auto-detect-rationale");
            const warnEl = $("auto-detect-warnings");
            const metEl  = $("auto-detect-metals");
            if (ratEl) {
                // The engine-agnostic rationale lives on each
                // adapter's response (echoed from the analyzer).
                // Either engine's value is the same; pick PySCF.
                const sug = (resp.suggested || {}).pyscf || {};
                ratEl.textContent = sug.rationale || "";
            }
            if (warnEl) {
                warnEl.textContent = "";
                for (const w of (resp.warnings || [])) {
                    const li = document.createElement("li");
                    li.textContent = w;
                    warnEl.appendChild(li);
                }
                warnEl.hidden = (resp.warnings || []).length === 0;
            }
            if (metEl) {
                metEl.textContent = "";
                for (const h of (resp.metal_hints || [])) {
                    const dt = document.createElement("dt");
                    dt.textContent = h.element;
                    metEl.appendChild(dt);
                    for (const c of (h.common_spins || [])) {
                        const dd = document.createElement("dd");
                        dd.textContent =
                            `spin=${c.spin} — ${c.label}`;
                        metEl.appendChild(dd);
                    }
                }
                metEl.hidden = (resp.metal_hints || []).length === 0;
            }
            // Inject / refresh the .workflow-detection-chip in every
            // .workflow-group--profile + .workflow-group--budget card
            // header so the user sees the analyzer's key conclusion
            // attached to the panel where they'll act on it.  Stage-
            // tagged fields don't get a chip — the staged-relaxation
            // recipe is system-agnostic and the user doesn't override
            // those values based on chemistry.
            _renderWorkflowGroupChips(resp);
        }

        // Detection-chip helpers live in lib/detection-chip.js so
        // every engine tab that wants chips (SIESTA, PySCF, Transport,
        // …) reads from one implementation.  Pre-2026-06-13 these
        // helpers were closure-private here, which silently denied
        // Transport (Au-junction users!) any chip surface.  See
        // docs/protocols/web-ui-coherence.md Rule 1.
        const _detectionChip = (window.molbuilder
                                && window.molbuilder.detectionChip)
            || { buildText: () => ({ profile: "", budget: "" }),
                 render: () => 0 };
        const _buildDetectionChipText = _detectionChip.buildText;
        const _renderWorkflowGroupChips = _detectionChip.render;
        });   // close runtime.whenReady("projects").then(...)
    }

    // formula() lives in static/lib/mol-format.js; loaded by the
    // template above.  Local alias keeps callers below readable.
    const formula = (window.molbuilder && window.molbuilder.fmt
                     ? window.molbuilder.fmt.formula
                     : (els) => (els && els.length ? els.join("") : "—"));

    // ----- 2. Render --------------------------------------------------
    // The standard knob bar (style picker / labels popover / axes /
    // reset / background / export) owns all the per-viewer chrome;
    // the Optimization tab's host page only owns its tab-specific
    // controls (Load-from-sidebar button, Generate buttons,
    // Download row).  The bespoke <details> Style block that used
    // to live in index.html is gone; rep / radius / background /
    // labels are reached via the knob bar so every consumer site
    // shares the same UX.

    function renderStructure() {
        if (!state.xyz) {
            // No structure to mount.  setStructure with an empty
            // xyz/pdb is a no-op in the embed; the empty state is
            // rare (only at first page load before the user
            // clicks Load from sidebar selection, or before the
            // cross-tab handoff auto-loads a file from
            // sessionStorage).  Leaves whatever was previously
            // rendered until the user loads something.
            return;
        }
        _handle.setStructure({ xyz: state.xyz });
        _handle.refit();
    }

    // ----- Downloads --------------------------------------------------
    function downloadAs(text, filename, mime = "text/plain") {
        const blob = new Blob([text], { type: mime });
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url; a.download = filename;
        document.body.appendChild(a);
        a.click();
        a.remove();
        setTimeout(() => URL.revokeObjectURL(url), 0);
    }

    // Phase 6: #dl-xyz / #dl-pdb retired — the embed's Export menu
    // (View / Export → Download → .xyz | .pdb) owns structure
    // download now.  ``downloadAs`` is still used below for .fdf
    // and .py generator outputs.

    function safeName(s) {
        return (s || "molecule").replace(/[^A-Za-z0-9._-]+/g, "_");
    }

    // Save-to-current-dir bookkeeping (Round 1 of the 2026-05-24
    // Generate/Save split).  Generate is now PURELY render-for-preview;
    // a separate "Save to current dir" button commits the rendered
    // artifact + the .run.sh wrapper + pseudo copies.  We cache the
    // most-recent-render's filename + the params used so the Save
    // handler can post them without re-rendering.  Cleared on Generate
    // failure (so a stale fdf doesn't pretend to be savable).
    state.lastFdfSave  = null;     // { filename, params, systemLabel }
    state.lastPyscfSave = null;    // { filename, params, jobName }

    /**
     * Enable / disable the Save buttons based on:
     *   (a) state.fdf / state.pyscf has been Generate-d, AND
     *   (b) the Projects sidebar has a non-root subdir selected.
     * Subscribed to projects.onChange below so a sidebar pick toggles
     * the buttons live.
     */
    function refreshSaveButtonAvailability() {
        const proj = (window.molbuilder || {}).projects;
        // hasDir = "the sidebar points at a non-root subdir where
        // saveToWorkspace can actually land a file".  Uses
        // projects.atRoot() (added 2026-05-26) instead of the raw
        // ``!!dir`` truthy check that incorrectly enabled Save at
        // the projects root -- the click would then fall through
        // saveToWorkspace's atProjectsRoot()->null path and show a
        // confusing "no current_dir" error.
        const hasDir = proj && typeof proj.atRoot === "function"
            ? !proj.atRoot()
            : !!(proj && proj.getCurrentDir());
        const sfdf = $("save-fdf");
        if (sfdf) sfdf.disabled = !(state.fdf  && hasDir);
        const spy = $("save-pyscf");
        if (spy)  spy.disabled  = !(state.pyscf && hasDir);
    }
    // Live re-evaluate the buttons when the sidebar selection changes.
    (function () {
        const proj = (window.molbuilder || {}).projects;
        if (proj && typeof proj.onChange === "function") {
            proj.onChange(refreshSaveButtonAvailability);
        }
    }());

    /**
     * Live caption for the SIESTA psml_lib field.  Drops a small
     * one-liner directly below the input that shows the user the
     * absolute path the server will resolve their entry to:
     *
     *   ""                    -> "(set a path)"
     *   "pseudopotential"     -> "→ /home/.../projects/pseudopotential"
     *   "../../../pseudo"     -> "→ /home/.../projects/../../../pseudo"
     *   "/abs/path"           -> "→ /abs/path"  (absolute, no change)
     *   "~/pseudos"           -> "→ ~/pseudos"  (expanded server-side)
     *
     * Addresses the UX gap that the implicit ``projects/`` anchor
     * isn't visible in the field itself.  Mirrors the server's
     * pseudos.resolve_psml_lib() rule (minus the dest_dir hop,
     * which only kicks in at Save time and isn't relevant here).
     */
    function installPsmlLibLiveCaption() {
        const input = $("p-psml-lib");
        if (!input || input.dataset.captionInstalled === "1") return;
        input.dataset.captionInstalled = "1";
        // Caption element rendered immediately after the input.  The
        // form-schema renderer wraps each field in a <label>; we
        // append into that label so the caption rides along with
        // wherever the field lives in the DOM.
        const cap = document.createElement("small");
        // .schema-field-hint is styled in lib/form-schema.css using the
        // real --text-muted token (replaces the prior inline --muted
        // fallback which silently used #888 because --muted isn't a
        // defined token).
        cap.className = "schema-field-hint";
        cap.id = "p-psml-lib-resolved";
        const parent = input.parentElement || input;
        parent.appendChild(cap);
        const proj = (window.molbuilder || {}).projects;
        function render() {
            const v = (input.value || "").trim();
            if (!v) {
                cap.textContent = "(unset -- SIESTA will fail to start without pseudos)";
                return;
            }
            // Absolute or ~/ -> echo (no anchor applied).
            if (v.charAt(0) === "/" || v.startsWith("~")) {
                cap.textContent = "→ " + v;
                return;
            }
            // Relative -> show projects/-anchored form.  The Save
            // step also tries dest-relative first, but at type-time
            // we don't know dest yet; show the projects/ anchor.
            const root = (proj && proj.getProjectsRoot && proj.getProjectsRoot()) || "";
            if (root) {
                cap.textContent = "→ " + root.replace(/\/$/, "") + "/" + v;
            } else {
                cap.textContent = "→ projects/" + v;
            }
        }
        input.addEventListener("input", render);
        // Initial paint AND re-paint when the projects root resolves
        // (the sidebar bootstrap is async; root might not be ready
        // when the form first mounts).
        render();
        if (proj && typeof proj.onChange === "function") {
            proj.onChange(render);
        }
    }
    // Form fields render asynchronously (form-schema fetches the
    // schema from /api/build/schema/siesta then renders).  Retry the
    // installation a few times so we hit the post-render DOM.
    (function pollForField(tries) {
        if (tries <= 0) return;
        if ($("p-psml-lib")) {
            installPsmlLibLiveCaption();
        } else {
            setTimeout(() => pollForField(tries - 1), 150);
        }
    }(40));   // ~6s budget

    // ----- 3. Generate FDF (render-only preview) ---------------------
    // Pure render: validate + render + populate the preview pane +
    // enable Download/Save.  Does NOT touch disk.  The user clicks
    // "Save to current dir" once they're happy with the preview.
    $("generate-fdf").addEventListener("click", async () => {
        if (!state.xyz) {
            setStatus("fdf-status", "Build a structure first.", "error");
            return;
        }
        setStatus("fdf-status", "Rendering FDF…");
        const params = collectFdfParams();
        // Send the sidebar dir as a dest hint so the server's
        // validator can resolve dest-relative ``cfg.psml_lib`` paths
        // (the form holds dest-relative strings after a prior Save).
        // Optional: validator falls back to projects/-relative when
        // omitted; the file-existence check downgrades to WARN in
        // that case so render proceeds.
        const _proj = (window.molbuilder || {}).projects;
        const _destDir = (_proj && _proj.getCurrentDir()) || "";
        // structure_path lets the server apply the .molstruct.json sidecar
        // next to the LOADED XYZ/PDB.  Without this hop /modify's
        // frozen_atoms list NEVER reaches render_fdf and SIESTA relaxes
        // every atom even when the user explicitly froze some.  Mirrors
        // /spectra; was the silent gap that prompted the 2026-05-25 fix.
        //
        // MUST read ``_sidebarLastFile`` (the file the user actually
        // committed into the viewer), NOT ``_proj.getCurrentFile()``
        // (the LIVE sidebar pick, which changes on every click while
        // the user navigates to pick a dest_dir or browse).  The
        // 2026-06-14 BDT-stage-2 incident: user loaded an .xyz with a
        // sidecar carrying 50 frozen atoms, then clicked into a
        // sibling directory to set dest_dir, then Generate.  Pre-fix
        // ``getCurrentFile()`` returned the LAST-CLICKED-IN-SIDEBAR
        // path (the dest dir's stage-1 .fdf), the server looked for a
        // sidecar next to that file, found none, emitted a
        // constraints-free stage-2 .fdf.  ``_sidebarLastFile`` stays
        // pinned to the committed load, so the server reads the
        // correct sidecar regardless of subsequent sidebar navigation.
        const _structPath = _sidebarLastFile || "";
        // Abort any in-flight Generate so two rapid clicks don't
        // race -- the LATER click should win, but without
        // AbortController the response that arrives LAST writes
        // state.fdf / state.lastFdfSave (which might be the OLDER
        // click's response if its render took longer).  Pattern
        // mirrors lib/spectra/core.js's renderAbort / loadAbort.
        if (state.fdfAbort) state.fdfAbort.abort();
        state.fdfAbort = new AbortController();
        const _signal = state.fdfAbort.signal;
        try {
            const r = await fetch("/api/build/fdf", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    xyz:            state.xyz,
                    params,
                    dest_dir:       _destDir || null,
                    // structure_path remains a back-compat fallback
                    // for the server (kicks in when frozen_atoms /
                    // regions keys aren't sent).  In-body labels
                    // below are the authoritative source under the
                    // 2026-06-14 viewer-is-truth contract.
                    structure_path: _structPath || null,
                    // In-body labels (viewer-is-truth contract):
                    // the server consumes these directly and skips
                    // sidecar re-read when the keys are present
                    // (even when empty).  See _shared.apply_labels_
                    // to_struct docstring.  Pulled into state at
                    // Load time inside _commitStructure.
                    frozen_atoms: state.frozen_atoms || [],
                    regions:      state.regions || {},
                }),
                signal: _signal,
            }).then(x => x.json());
            if (!r.ok) {
                setStatus("fdf-status", r.error || "FDF render failed.", "error");
                state.fdf = null;
                state.lastFdfSave = null;
                // Clear the Save-in-flight flag too: if the user clicked
                // Generate WHILE a Save was running, the Save's finally
                // would otherwise re-enable the button against a now-
                // null state.fdf.  Caught by the 2026-05-26 review.
                state.savingFdf = false;
                refreshSaveButtonAvailability();
                return;
            }
            state.fdf = r.fdf;
            $("fdf-output").textContent = r.fdf;
            $("fdf-output").hidden = false;
            $("dl-fdf").disabled = false;
            // Cache what Save needs: filename basename (label + stage
            // suffix), the params dict (mpi/omp/psml_lib live here),
            // and the system_label for downstream display.
            const fdfLabel = (r.system_label || "siesta").replace(
                /[^A-Za-z0-9._-]+/g, "_");
            const _stage = stageNumberFromPreset();
            const _stageSuffix = _stage ? `-stage${_stage}` : "";
            state.lastFdfSave = {
                filename:    fdfLabel + _stageSuffix + ".fdf",
                params:      params,
                systemLabel: r.system_label,
            };
            refreshSaveButtonAvailability();
            const fdfMsg = `OK — ${r.fdf.split("\n").length} lines, label "${r.system_label}".`;
            const issues = r.issues || [];
            renderIssues("fdf-issues", issues, "siesta-form-container");
            setStatus("fdf-status", fdfMsg,
                      issues.some(i => i.severity === "warn") ? "warn" : "ok");
        } catch (e) {
            // AbortError = a newer Generate click superseded this one;
            // the newer click's handler owns state -- don't clobber.
            if (e && e.name === "AbortError") return;
            setStatus("fdf-status", "Network error: " + e.message, "error");
            state.fdf = null;
            state.lastFdfSave = null;
            state.savingFdf = false;
            refreshSaveButtonAvailability();
        }
    });

    // ----- 3b. Save FDF + .run.sh + .psml files to current dir ------
    // Commit step.  Requires (a) state.fdf populated by a prior
    // Generate, (b) Projects sidebar pointing at a non-root subdir.
    // Writes the .fdf, copies the matching .psml files (if psml_lib
    // is set), drops the .run.sh wrapper, then rewrites the
    // ``psml_lib`` form field to a path RELATIVE TO dest_dir.  The
    // rewrite is the portability-+-privacy fix: avoids storing the
    // absolute /home/<user>/... path in form state / future sidecars,
    // and survives copying the whole project tree elsewhere.
    // Reentrancy guard for both Save buttons.  A 4-step async pipeline
    // (write .fdf / .py, install pseudos, install wrapper, refresh
    // sidebar) can be 100s of ms; a double-click triggers TWO
    // pipelines into the same dest with the same content.  Set the
    // flag at click-time, disable the button, restore in finally so
    // an exception path doesn't lock the button permanently.
    state.savingFdf   = false;
    state.savingPyscf = false;

    $("save-fdf").addEventListener("click", async () => {
        if (state.savingFdf) return;          // ignore double-click
        state.savingFdf = true;
        const _btn = $("save-fdf");
        const _origText = _btn.textContent;
        _btn.disabled = true;
        _btn.textContent = "Saving…";
        // Sidebar lock: prevents the user from changing current_dir
        // mid-pipeline (which would retarget step 2's pseudo install
        // to a different directory than where step 1 wrote the .fdf).
        // The AbortController collects every fetch's signal so the
        // lock banner's Cancel can abort an in-flight request and
        // unwind through the finally below.
        const proj = (window.molbuilder || {}).projects;
        state.saveFdfAbort = new AbortController();
        const _signal = state.saveFdfAbort.signal;
        const _hasLock = !!proj && typeof proj.lock === "function";
        if (_hasLock) {
            try {
                proj.lock("Saving FDF + pseudos + wrapper…",
                          [() => state.saveFdfAbort.abort()]);
            } catch (_) { /* if already locked (shouldn't happen
                            without overlapping handlers), proceed
                            without acquiring */ }
        }
        try {
            return await _runSaveFdfPipeline(_signal);
        } finally {
            state.savingFdf = false;
            _btn.disabled = false;
            _btn.textContent = _origText;
            refreshSaveButtonAvailability();
            if (_hasLock) proj.unlock();
        }
    });

    async function _runSaveFdfPipeline(abortSignal) {
        const meta = state.lastFdfSave;
        const proj = (window.molbuilder || {}).projects;
        if (!state.fdf || !meta) {
            setStatus("fdf-status", "Click Generate first.", "error");
            return;
        }
        if (!proj) {
            setStatus("fdf-status", "Projects sidebar not loaded.", "error");
            return;
        }
        const destDir = proj.getCurrentDir() || "";
        if (!destDir) {
            setStatus("fdf-status",
                "Pick a project subdir in the sidebar first.",
                "error");
            return;
        }
        setStatus("fdf-status", "Saving to " + destDir + " …");
        const { filename, params } = meta;
        // Step 1: write the .fdf.  2026-05-30: thread the lock's
        // AbortSignal so the Cancel button can interrupt this fetch
        // (#174 sidebar gap M1 -- writeFile now honours opts.signal).
        let written;
        try {
            const w = await proj.safeSave(
                state.fdf, filename,
                { overwrite: true, signal: abortSignal });
            // safeSave contract: branch on r.cancelled BEFORE
            // !r.ok so user-initiated Cancel doesn't render as a
            // red "Save failed: …" banner.  See state.js safeSave.
            if (w && w.cancelled) {
                setStatus("fdf-status", "Save cancelled.", "muted");
                return;
            }
            if (!w || !w.ok) {
                setStatus("fdf-status",
                    "Save failed: " + (w && w.error || "no current_dir"),
                    "error");
                return;
            }
            const dir = (w.path || "").replace(/\/[^/]*$/, "");
            written = { path: w.path, dir, relPath: w.relPath };
        } catch (e) {
            setStatus("fdf-status",
                "Save network error: " + e.message, "error");
            return;
        }
        // Step 2: copy .psml files into dest dir.
        const psmlLib = (params || {}).psml_lib;
        let installedOk = true;
        let psmlMsg = "";
        if (psmlLib) {
            try {
                const ip = await fetch("/api/siesta/install-pseudos", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({
                        psml_lib:        psmlLib,
                        dest_dir:        written.dir,
                        structure_text:  state.xyz,
                    }),
                    signal: abortSignal,
                }).then(x => x.json());
                if (ip.ok) {
                    const nNew  = (ip.copied || []).length;
                    const nOver = (ip.overwritten || []).length;
                    const nSkip = (ip.skipped || []).length;
                    const missing = ip.missing || [];
                    const parts = [];
                    if (nNew)  parts.push(`copied ${nNew} new`);
                    if (nOver) parts.push(`overwrote ${nOver}`);
                    if (nSkip) parts.push(`${nSkip} already present`);
                    psmlMsg = ".psml: " + (parts.join(", ") || "no-op");
                    if (missing.length) {
                        psmlMsg += " · MISSING: " + missing.join(", ")
                                + " — SIESTA will refuse to start";
                        installedOk = false;
                    }
                } else {
                    psmlMsg = "pseudo install failed: " + (ip.error || "?");
                    installedOk = false;
                }
            } catch (e) {
                // Phase 6e fifth-review BOMB-B: user-initiated
                // Cancel reaches this catch as AbortError.  Bail
                // the whole pipeline rather than continue into
                // the wrapper-install step under a cancelled
                // intent.  proj.isCancelError centralises the
                // predicate (sixth-review follow-up).
                if (proj.isCancelError(e)) {
                    setStatus("fdf-status",
                        "Save cancelled. (Wrote " + written.relPath
                      + " before cancel; pseudos and .run.sh not "
                      + "installed.)",
                        "muted");
                    try {
                        if (proj.refresh) await proj.refresh();
                    } catch (_) {}
                    return;
                }
                psmlMsg = "pseudo install network error: " + e.message;
                installedOk = false;
            }
        }
        // Step 3: drop the .run.sh next to the .fdf -- BUT ONLY IF the
        // pseudo install succeeded.  If install-pseudos missed elements
        // (installedOk = false), a wrapper that references the (still-
        // missing) pseudos would launch SIESTA which aborts at startup
        // with an opaque "pseudo_read: ERROR: Pseudopotential file not
        // found" -- the user sees a green "wrote .run.sh" and only
        // discovers the breakage after running the wrapper.  Skip the
        // wrapper write when pseudos are incomplete; surface the gap
        // in the status text instead.  Caught by the 2026-05-26 review.
        const _n = (k) => {
            const v = (params || {})[k];
            const n = Number(v);
            return Number.isFinite(n) && n > 0 ? n : null;
        };
        let wrapperMsg = "";
        let wrapperCancelled = false;
        if (!installedOk) {
            wrapperMsg = "skipped .run.sh (pseudos incomplete; "
                       + "fix the install-pseudos errors above and "
                       + "click Save again)";
        } else {
            try {
                const wr = await fetch("/api/run/install-wrapper", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({
                        script_path:   written.path,
                        mpi_np:        _n("mpi_np"),
                        omp_threads:   _n("omp_threads"),
                        max_memory_mb: _n("max_memory_mb"),
                    }),
                    signal: abortSignal,
                }).then(x => x.json());
                if (wr.ok) {
                    const verb = wr.overwritten ? "overwrote" : "wrote";
                    wrapperMsg = `${verb} ${wr.wrapper_name}`;
                }
            } catch (e) {
                if (proj.isCancelError(e)) {
                    wrapperCancelled = true;
                }
                /* other failures stay non-fatal */
            }
        }
        if (wrapperCancelled) {
            setStatus("fdf-status",
                "Save cancelled after writing " + written.relPath
              + " (.run.sh not installed).", "muted");
            try {
                if (proj.refresh) await proj.refresh();
            } catch (_) {}
            return;
        }

        // Step 4: rewrite the psml_lib form field to relative-to-dest
        // (the portability fix).  Only when the field currently holds
        // an absolute path AND we actually copied pseudos -- otherwise
        // we'd silently corrupt a form the user is still editing.
        if (psmlLib && installedOk && psmlLib.charAt(0) === "/") {
            const pathLib = ((window.molbuilder || {}).path) || {};
            const rel = (typeof pathLib.relativeFromDir === "function")
                ? pathLib.relativeFromDir(psmlLib, written.dir)
                : psmlLib;
            // Field's DOM id comes from id_prefix "p" + id from the
            // schema ("p-psml-lib").  Find it + update.  We don't go
            // through form-schema.collectForm/dispatch -- a direct
            // DOM write is fine because the field is a plain text input.
            const input = $("p-psml-lib");
            if (input && rel !== psmlLib) {
                input.value = rel;
                // Fire input + change so any listeners (live preflight
                // / dirty-flag tracking) see the new value.
                input.dispatchEvent(new Event("input",  { bubbles: true }));
                input.dispatchEvent(new Event("change", { bubbles: true }));
            }
        }

        // Compose final status: write target + psml + wrapper line.
        const segs = ["Wrote " + written.relPath];
        if (psmlMsg)    segs.push(psmlMsg);
        if (wrapperMsg) segs.push(wrapperMsg);
        setStatus("fdf-status", segs.join(" · "),
            !installedOk ? "error" : "ok");

        // Refresh the sidebar so the newly-dropped .fdf + .psml + .run.sh
        // all appear in the directory listing.  saveToWorkspace's
        // writeFile triggers ONE refresh (for the .fdf only) -- the
        // /api/siesta/install-pseudos + /api/run/install-wrapper
        // endpoints write files directly via shutil/write_run_wrapper
        // without going through writeFile, so they don't auto-refresh.
        // One explicit refresh at the end covers all three steps.
        try { if (proj.refresh) await proj.refresh(); } catch (_) { /* non-fatal */ }
    }
    // (close of _runSaveFdfPipeline)

    $("dl-fdf").addEventListener("click", () => {
        if (!state.fdf) return;
        const label = ($("p-system-label").value.trim() || "siesta").replace(
            /[^A-Za-z0-9._-]+/g, "_");
        // Stage-aware filename: <name>-stage<N>.fdf when the user
        // picked a non-Custom preset, so saving each stage's FDF
        // alongside the previous ones in one directory doesn't
        // overwrite (and matches the "Run with: ... <name>-stage<N>.fdf"
        // line emitted in the FDF body).
        const stage = stageNumberFromPreset();
        const suffix = stage ? `-stage${stage}` : "";
        downloadAs(state.fdf, label + suffix + ".fdf");
    });

    // Map the Generate-input form's stage-preset selector value to
    // the SiestaConfig / PySCFConfig ``stage`` integer.  Custom and
    // single-run modes pass null so the unsuffixed ``.molwatch.log``
    // filename is used.
    function stageNumberFromPreset() {
        const v = ($("p-stage-preset") || {}).value || "custom";
        if (v === "coarse") return 1;
        if (v === "medium") return 2;
        if (v === "tight")  return 3;
        return null;
    }

    function collectFdfParams() {
        // The schema-driven collector returns one entry per dataclass
        // field with a "section": metadata key.  The "Relaxation
        // stage" preset is a UI shortcut, not a dataclass field
        // rendered by the schema; we layer the stage-number on top of
        // the collected params.  (2026-05-27: dropped the legacy
        // system_name = system_label fold -- the dataclass no longer
        // has system_name.)
        if (!formSchemas.siesta) return {};
        const fs = (window.molbuilder || {}).formSchema;
        const params = fs.collectForm(
            $("siesta-form-container"), formSchemas.siesta
        );
        params.stage = stageNumberFromPreset();
        return params;
    }

    // ----- 4. Generate PySCF script (render-only preview) -----------
    $("generate-pyscf").addEventListener("click", async () => {
        if (!state.xyz) {
            setStatus("pyscf-status", "Build a structure first.", "error");
            return;
        }
        setStatus("pyscf-status", "Rendering PySCF script…");
        const params = collectPyscfParams();
        // structure_path: see /api/build/fdf handler for rationale --
        // lets the server apply /modify's .molstruct.json sidecar
        // (frozen_atoms + regions) before render_script runs.
        //
        // Reads ``_sidebarLastFile`` (committed-load path), NOT
        // ``_proj.getCurrentFile()`` (live sidebar pick).  Same bug
        // class as the SIESTA path above: the live pick changes
        // every time the user clicks ANYWHERE in the sidebar, so by
        // the time Generate fires the path may point at whatever
        // file the user last browsed -- breaking sidecar discovery.
        const _structPathPy = _sidebarLastFile || "";
        // Abort any in-flight Generate -- see /api/build/fdf above
        // for the rationale.  Two rapid clicks otherwise race.
        if (state.pyscfAbort) state.pyscfAbort.abort();
        state.pyscfAbort = new AbortController();
        const _signalPy = state.pyscfAbort.signal;
        try {
            const r = await fetch("/api/build/pyscf", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    xyz:            state.xyz,
                    params,
                    // structure_path: back-compat fallback when
                    // in-body labels aren't present.
                    structure_path: _structPathPy || null,
                    // In-body labels (viewer-is-truth contract);
                    // pulled into state at Load time.  Server
                    // prefers these over sidecar re-read.
                    frozen_atoms: state.frozen_atoms || [],
                    regions:      state.regions || {},
                }),
                signal: _signalPy,
            }).then(x => x.json());
            if (!r.ok) {
                setStatus("pyscf-status",
                    r.error || "PySCF render failed.", "error");
                state.pyscf = null;
                state.lastPyscfSave = null;
                state.savingPyscf = false;
                refreshSaveButtonAvailability();
                return;
            }
            state.pyscf = r.script;
            $("pyscf-output").textContent = r.script;
            $("pyscf-output").hidden = false;
            $("dl-pyscf").disabled = false;
            const jobName = (r.job_name || "pyscf").replace(
                /[^A-Za-z0-9._-]+/g, "_");
            state.lastPyscfSave = {
                filename: jobName + ".py",
                params:   params,
                jobName:  r.job_name,
            };
            refreshSaveButtonAvailability();
            const pyMsg = `OK — ${r.script.split("\n").length} lines, job "${r.job_name}".`;
            const issues = r.issues || [];
            renderIssues("pyscf-issues", issues, "pyscf-form-container");
            setStatus("pyscf-status", pyMsg,
                      issues.some(i => i.severity === "warn") ? "warn" : "ok");
        } catch (e) {
            if (e && e.name === "AbortError") return;  // superseded
            setStatus("pyscf-status", "Network error: " + e.message, "error");
            state.pyscf = null;
            state.lastPyscfSave = null;
            state.savingPyscf = false;
            refreshSaveButtonAvailability();
        }
    });

    // ----- 4b. Save PySCF .py + .run.sh to current dir --------------
    $("save-pyscf").addEventListener("click", async () => {
        if (state.savingPyscf) return;
        state.savingPyscf = true;
        const _btnPy = $("save-pyscf");
        const _origTextPy = _btnPy.textContent;
        _btnPy.disabled = true;
        _btnPy.textContent = "Saving…";
        // Sidebar lock: same rationale as the SIESTA Save handler -- a
        // mid-pipeline current_dir change would retarget the wrapper
        // install to the wrong directory.
        const _projPyL = (window.molbuilder || {}).projects;
        state.savePyscfAbort = new AbortController();
        const _signalPy = state.savePyscfAbort.signal;
        const _hasLockPy = !!_projPyL && typeof _projPyL.lock === "function";
        if (_hasLockPy) {
            try {
                _projPyL.lock("Saving PySCF + wrapper…",
                              [() => state.savePyscfAbort.abort()]);
            } catch (_) { /* already locked: continue without */ }
        }
        try {
            return await _runSavePyscfPipeline(_signalPy);
        } finally {
            state.savingPyscf = false;
            _btnPy.disabled = false;
            _btnPy.textContent = _origTextPy;
            refreshSaveButtonAvailability();
            if (_hasLockPy) _projPyL.unlock();
        }
    });

    async function _runSavePyscfPipeline(abortSignal) {
        const meta = state.lastPyscfSave;
        const proj = (window.molbuilder || {}).projects;
        if (!state.pyscf || !meta) {
            setStatus("pyscf-status", "Click Generate first.", "error");
            return;
        }
        if (!proj) {
            setStatus("pyscf-status", "Projects sidebar not loaded.", "error");
            return;
        }
        const destDir = proj.getCurrentDir() || "";
        if (!destDir) {
            setStatus("pyscf-status",
                "Pick a project subdir in the sidebar first.",
                "error");
            return;
        }
        setStatus("pyscf-status", "Saving to " + destDir + " …");
        // 2026-05-30: thread the lock's AbortSignal so the Cancel
        // button can interrupt this fetch (#174 sidebar gap M1).
        let written;
        try {
            const w = await proj.safeSave(
                state.pyscf, meta.filename,
                { overwrite: true, signal: abortSignal });
            // safeSave contract: see SIESTA mirror above + state.js.
            if (w && w.cancelled) {
                setStatus("pyscf-status", "Save cancelled.", "muted");
                return;
            }
            if (!w || !w.ok) {
                setStatus("pyscf-status",
                    "Save failed: " + (w && w.error || "no current_dir"),
                    "error");
                return;
            }
            const dir = (w.path || "").replace(/\/[^/]*$/, "");
            written = { path: w.path, dir, relPath: w.relPath };
        } catch (e) {
            setStatus("pyscf-status",
                "Save network error: " + e.message, "error");
            return;
        }
        // PySCF wrapper: no mpi_np / no omp / no memory cap (the
        // in-script runtime block handles those).
        let wrapperMsg = "";
        let wrapperCancelled = false;
        try {
            const wr = await fetch("/api/run/install-wrapper", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    script_path:   written.path,
                    mpi_np:        null,
                    omp_threads:   null,
                    max_memory_mb: null,
                }),
                signal: abortSignal,
            }).then(x => x.json());
            if (wr.ok) {
                const verb = wr.overwritten ? "overwrote" : "wrote";
                wrapperMsg = `${verb} ${wr.wrapper_name}`;
            }
        } catch (e) {
            if (proj.isCancelError(e)) {
                wrapperCancelled = true;
            }
            /* other failures stay non-fatal */
        }
        if (wrapperCancelled) {
            setStatus("pyscf-status",
                "Save cancelled after writing " + written.relPath
              + " (.run.sh not installed).", "muted");
            return;
        }
        const segs = ["Wrote " + written.relPath];
        if (wrapperMsg) segs.push(wrapperMsg);
        setStatus("pyscf-status", segs.join(" · "), "ok");
        // Refresh sidebar so the freshly-dropped .py + .run.sh show
        // up in the listing.  install-wrapper writes outside the
        // writeFile path, so a manual refresh is needed.
        try { if (proj.refresh) await proj.refresh(); } catch (_) { /* non-fatal */ }
    }
    // (close of _runSavePyscfPipeline)

    $("dl-pyscf").addEventListener("click", () => {
        if (!state.pyscf) return;
        const label = ($("py-job-name").value.trim() || "pyscf_relax")
            .replace(/[^A-Za-z0-9._-]+/g, "_");
        // Stage-aware filename for the same reason as dl-fdf above.
        const stage = stageNumberFromPreset();
        const suffix = stage ? `-stage${stage}` : "";
        downloadAs(state.pyscf, label + suffix + ".py", "text/x-python");
    });

    function collectPyscfParams() {
        // Schema-driven collector + three post-processing tweaks:
        //   1. The "Relaxation stage" preset is a UI shortcut (lives
        //      in the SIESTA panel but flows through to PySCFConfig
        //      so PySCF runs also write stage-suffixed molwatch logs).
        //   2. dispersion = "none" comes off the select as the literal
        //      string "none"; server's config_from_params normalises
        //      it to None, but we drop it client-side too so the
        //      validation panel never sees the placeholder.
        //   3. Drop null-valued keys so the dataclass uses its
        //      declared default rather than getting None where it
        //      expects an int / float (e.g. charge=None should leave
        //      the dataclass alone; the server's auto-detect path
        //      runs only when no charge key is present).
        if (!formSchemas.pyscf) return {};
        const fs = (window.molbuilder || {}).formSchema;
        const params = fs.collectForm(
            $("pyscf-form-container"), formSchemas.pyscf
        );
        params.stage = stageNumberFromPreset();
        if (params.dispersion === "none") params.dispersion = null;
        // Drop nulls.
        Object.keys(params).forEach(k => {
            if (params[k] === null) delete params[k];
        });
        return params;
    }

    // ----- Session state: persist Generate-input form values across tab navigation -----
    // Navigating to another tab and back is a full page reload;
    // sessionStorage survives same-tab navigations so the user's
    // typed parameter values aren't lost.

    // Static IDs that aren't part of the schema-driven SIESTA /
    // PySCF forms but DO need session-storage persistence.  Post
    // task-295 the in-tab Build form is gone, so the Relaxation-
    // stage preset selector (a UI shortcut, not a dataclass field)
    // is the only static survivor.  All other persistent IDs are
    // derived at save/restore time by walking the rendered
    // schemas.
    const STATIC_FORM_IDS = [
        "p-stage-preset",
    ];

    function getFormIds() {
        const ids = STATIC_FORM_IDS.slice();
        for (const sch of [formSchemas.siesta, formSchemas.pyscf]) {
            if (!sch) continue;
            for (const sect of sch.sections) {
                for (const f of sect.fields) {
                    if (f.kind === "int-triple") {
                        // Tuple field renders as three sub-inputs;
                        // each has its own id and needs persistence.
                        for (const lab of f.labels) {
                            ids.push(f.id + "-" + lab);
                        }
                    } else {
                        ids.push(f.id);
                    }
                }
            }
        }
        return ids;
    }

    function saveFormState() {
        const saved = {};
        getFormIds().forEach(id => {
            const el = $(id);
            if (!el) return;
            saved[id] = el.type === "checkbox" ? el.checked : el.value;
        });
        sessionStorage.setItem("builder-form", JSON.stringify(saved));
    }

    function restoreFormState() {
        let saved;
        try { saved = JSON.parse(sessionStorage.getItem("builder-form") || "null"); }
        catch (_) { return; }
        if (!saved) return;
        getFormIds().forEach(id => {
            const el = $(id);
            if (!el || !(id in saved)) return;
            if (el.type === "checkbox") el.checked = saved[id];
            else el.value = saved[id];
        });
    }

    // The Relaxation-stage preset selector is static HTML, so the
    // first restore pass can run synchronously; the schema-driven
    // SIESTA / PySCF fields get restored a second time inside
    // initFormsFromSchema() after the renderer fills the containers.
    restoreFormState();
    window.addEventListener("pagehide", saveFormState);

    // Kick off the async schema fetch + form render.  Everything
    // that depends on the form's inputs (compatibility engine,
    // change listeners, full restoreFormState walk) happens inside
    // this function once the renderer has populated the DOM.
    initFormsFromSchema();

    // ----- Cross-tab structure handoff (post-task-#295) --------------
    //
    // The legacy "builder-structure" sessionStorage save/restore that
    // used to live here was retired 2026-06-09 (task #306): post-task
    // #295 the Optimization tab is file-driven, so the cross-tab
    // structure handoff goes through the Projects-sidebar pointer
    // (``sessionStorage.molbuilder.current_file``) + the mount-time
    // auto-load in the runtime.whenReady("projects") block above.
    // That path also re-reads bytes from disk on every restore, so
    // the snapshot-the-XYZ-into-sessionStorage approach was both
    // redundant and a quota-pressure liability on large structures.
    // No saveStructureState / pagehide listener is needed any more.

    // ----- Staged-relaxation presets ----------------------------- //
    // The Watch tab carries the full workflow guide; the Build tab's
    // job is to make it one-click to fill the SIESTA convergence
    // params for each stage.  Same SystemLabel + same directory ->
    // SIESTA reads the previous stage's .XV and .DM automatically.
    //
    // Values are the ones documented in the Watch tab's recipe table.
    // The ``custom`` option does NOT reset anything -- it just stops
    // auto-filling so the user can fine-tune individual fields.
    // STAGE_PRESETS restructured 2026-06-13.  PRE-FIX the preset
    // wrote to mixing-weight (a SYSTEM characteristic — depends on
    // metallic / organic chemistry, not stage) and to relax-steps
    // (a BUDGET cap — depends on system size, not stage).  Switching
    // stages silently mutated those values away from the user's
    // intent — the Au-BDT-Au bug class.
    //
    // After 2026-06-13: only fields tagged ``workflow_group:
    // "stage"`` in the schema are written.  Budget (max_scf_iter,
    // relax_steps) and system characteristics (mixing_weight,
    // electronic_temperature, spin_*) are NEVER touched by a stage
    // switch; the user manages those via the dedicated Resource
    // budget + System characteristics workflow-group cards.
    //
    // The presets are the same convergence-target values that
    // appear in the published staged-relaxation literature:
    // coarse = fast descent (loose tols, big steps), medium =
    // refine, tight = production-grade.  Each successive stage
    // tightens; none of them touches "how much patience I'm
    // willing to spend on this run."
    const STAGE_PRESETS = {
        coarse: {
            "p-mesh-cutoff":         200,
            "p-pao-energy-shift":    0.02,
            "p-dm-tolerance":        1e-3,
            "p-dm-energy-tolerance": 1e-3,
            "p-force-tol":           0.04,
            "p-max-displ":           0.20,
        },
        medium: {
            "p-mesh-cutoff":         300,
            "p-pao-energy-shift":    0.01,
            "p-dm-tolerance":        1e-4,
            "p-dm-energy-tolerance": 1e-4,
            "p-force-tol":           0.02,
            "p-max-displ":           0.10,
        },
        tight: {
            "p-mesh-cutoff":         400,
            "p-pao-energy-shift":    0.005,
            "p-dm-tolerance":        1e-5,
            "p-dm-energy-tolerance": 1e-5,
            "p-force-tol":           0.01,
            "p-max-displ":           0.05,
        },
    };

    // Document the IDs we KNOW are stage-tagged on the SIESTA side.
    // viewer.js doesn't have direct access to the schema metadata
    // (the form-schema renderer holds it), so we maintain an
    // explicit list here AND a test in
    // tests/test_live_poll_invariants_audit.py pins that every ID
    // in this list corresponds to a field tagged
    // ``workflow_group: "stage"`` in molbuilder/config/siesta.py.
    // Add a new stage field → tag in siesta.py AND add the ID here;
    // the test catches the half-done case.
    const _STAGE_PRESET_KEYS_SIESTA = new Set([
        "p-mesh-cutoff", "p-pao-energy-shift",
        "p-dm-tolerance", "p-dm-energy-tolerance",
        "p-force-tol",   "p-max-displ",
    ]);

    function applyStagePreset(stage) {
        const preset = STAGE_PRESETS[stage];
        if (!preset) return;
        Object.entries(preset).forEach(([id, value]) => {
            // Defense in depth: only write to IDs that the
            // _STAGE_PRESET_KEYS_SIESTA allowlist confirms are
            // stage-tagged.  A typo / drift between the preset dict
            // and the schema metadata gets caught here at runtime
            // rather than silently mutating a budget / system field.
            if (!_STAGE_PRESET_KEYS_SIESTA.has(id)) {
                if (console && console.warn) {
                    console.warn(
                        "[stage-preset] refusing to write " + id
                        + " — not in stage-key allowlist.  Update "
                        + "_STAGE_PRESET_KEYS_SIESTA in viewer.js AND "
                        + "tag the field workflow_group: \"stage\" in "
                        + "molbuilder/config/siesta.py."
                    );
                }
                return;
            }
            const el = $(id);
            if (!el) return;
            el.value = String(value);
            // Fire 'change' so the preflight + sessionStorage handlers
            // see the new value as if the user had typed it.
            el.dispatchEvent(new Event("change", { bubbles: true }));
            el.dispatchEvent(new Event("input", { bubbles: true }));
        });
    }

    const stageSel = $("p-stage-preset");
    if (stageSel) {
        stageSel.addEventListener("change", () => {
            applyStagePreset(stageSel.value);
        });
    }

    // Wire each engine-scoped form input to the debounced preflight
    // refresh so the issues panel updates live as the user adjusts
    // settings.  p-* IDs feed SIESTA's panel; py-* IDs feed PySCF's.
    // No-op until the user has built a structure (state.xyz set) --
    // see refreshPreflight().
    //
    // CRITICAL: must run AFTER initFormsFromSchema renders the form,
    // because the schema-driven inputs don't exist at module load.
    // wirePreflightListeners is called from initFormsFromSchema's
    // tail.  It also walks getFormIds() rather than a hard-coded
    // list so adding a new dataclass field auto-wires preflight.
    function wirePreflightListeners() {
        getFormIds().forEach(id => {
            const el = $(id);
            if (!el) return;
            const which = id.startsWith("p-")  ? "siesta"
                        : id.startsWith("py-") ? "pyscf"
                                               : null;
            if (!which) return;
            // Skip the static stage-preset selector -- it has its own
            // change handler that bulk-fills sibling inputs (each of
            // which fires their own change event and gets caught
            // below).
            if (id === "p-stage-preset") return;
            const event = (el.type === "checkbox" || el.tagName === "SELECT")
                ? "change" : "input";
            el.addEventListener(event, () => refreshPreflightDebounced[which]());
        });
    }
})();
