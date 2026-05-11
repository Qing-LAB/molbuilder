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
    };

    const viewer = $3Dmol.createViewer("viewer", {
        backgroundColor: "white",
        defaultcolors: $3Dmol.elementColors.Jmol,
    });

    // Keep the 3Dmol canvas in sync with the user-resizable container.
    // 3Dmol's WebGL context doesn't auto-track its parent box; we have
    // to call viewer.resize() whenever the container's dimensions change
    // (CSS resize handle, window resize, layout reflow).
    function syncViewerSize() {
        viewer.resize();
        viewer.render();
    }
    const wrapEl = $("viewer-wrap");
    if (wrapEl && typeof ResizeObserver !== "undefined") {
        new ResizeObserver(syncViewerSize).observe(wrapEl);
    }
    window.addEventListener("resize", syncViewerSize);

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
    function renderIssues(panelId, issues) {
        const ul = $(panelId);
        if (!ul) return;
        ul.innerHTML = "";
        const list = (issues || []).filter(Boolean);
        if (!list.length) {
            ul.hidden = true;
            return;
        }
        for (const i of list) {
            const li = document.createElement("li");
            li.className = "issue-item";
            li.setAttribute("data-severity", i.severity || "info");
            const msg = document.createElement("span");
            msg.className = "issue-msg";
            msg.textContent = i.message || "";
            li.appendChild(msg);
            if (i.where) {
                const tag = document.createElement("span");
                tag.className = "issue-where";
                tag.textContent = i.where;
                li.appendChild(tag);
            }
            ul.appendChild(li);
        }
        ul.hidden = false;
    }

    function clearIssues(panelId) {
        const ul = $(panelId);
        if (!ul) return;
        ul.innerHTML = "";
        ul.hidden = true;
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
        try {
            const r = await fetch("/api/build/preflight", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ xyz: state.xyz, engine, params }),
            }).then(x => x.json());
            if (r.ok) renderIssues(panelId, r.issues);
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

    function placeholderFor(kind) {
        switch (kind) {
            case "peptide": return "ARNDC  or  AR[SEP]C";
            // Sequences are read 5'->3' by default; explicit
            // 5'-...-3' or 3'-...-5' labels are accepted.
            case "dna":     return "ATGCATGCAT  or  5'-ATGC-3'";
            case "rna":     return "AUGCAUGCAU  or  5'-AUGC-3'";
            case "smiles":  return "c1ccccc1   or  Sc1ccc(S)cc1";
            case "name":    return "benzene   or  1,4-benzenedithiol";
            default:        return "";
        }
    }

    function toggleNucleicOptions() {
        const k = $("kind").value;
        $("nucleic-options").hidden = !(k === "dna" || k === "rna");
        // RNA's natural form is A; flip the default when switching
        if (k === "rna" && $("form").value === "B") $("form").value = "A";
        if (k === "dna" && $("form").value === "A") $("form").value = "B";
    }
    $("kind").addEventListener("change", (e) => {
        $("input-text").placeholder = placeholderFor(e.target.value);
        toggleNucleicOptions();
    });
    $("input-text").placeholder = placeholderFor($("kind").value);
    toggleNucleicOptions();
    $("input-text").addEventListener("keydown", (e) => {
        if (e.key === "Enter") $("build-btn").click();
    });

    // Map the canonical (lowercase) backend identifier returned by
    // /api/backends to the user-facing label.  X3DNA is the product
    // name from x3dna.org -- not "3DNA" or "threedna".  Used for the
    // dropdown's "auto" relabel, the hint line, and the post-build
    // "via <name>" message.
    const BACKEND_LABEL = {
        threedna: "X3DNA",
        amber:    "Amber",
        rdkit:    "RDKit",
    };
    const labelFor = (name) => BACKEND_LABEL[name] || name;

    // Detect installed backends, grey out unavailable ones in the
    // dropdown, label the "auto" option with the resolved backend
    // name so the user sees what would actually run, and surface a
    // visible warning in #backend-hint when X3DNA (the highest-
    // quality backend) isn't installed.  One-shot fetch on page load.
    fetch("/api/backends").then(r => r.json()).then(r => {
        if (!r || !r.ok) return;
        const sel = $("backend");
        for (const opt of sel.options) {
            const name = opt.value;
            if (name === "auto") {
                if (r.auto_name) {
                    opt.text = `auto  (→ ${labelFor(r.auto_name)})`;
                } else {
                    opt.text = "auto  (no backend installed)";
                    opt.disabled = true;
                }
                continue;
            }
            opt.disabled = !r.available[name];
            if (!r.available[name]) {
                opt.text = opt.text + "  (not installed)";
            }
        }
        // Hint line below the dropdown -- always present so the user
        // can read what's installed without expanding the dropdown.
        const hint = $("backend-hint");
        if (hint) {
            const parts = [];
            if (r.auto_name) {
                parts.push(`auto → <b>${labelFor(r.auto_name)}</b>`);
            } else {
                parts.push("no nucleic-acid backend is installed");
            }
            if (!r.available.threedna) {
                parts.push(
                    "X3DNA not detected (canonical B/A/Z helices unavailable; " +
                    'install from <a href="http://x3dna.org/" target="_blank" rel="noopener">x3dna.org</a> ' +
                    "to enable)"
                );
            }
            hint.innerHTML = parts.join(" &middot; ");
            hint.className = r.auto_name && r.available.threedna
                ? "status ok"
                : "status warn";
        }
    }).catch(() => { /* /api/backends optional */ });

    // ----- 1. Build ---------------------------------------------------
    $("build-btn").addEventListener("click", async () => {
        const kind = $("kind").value;
        const input = $("input-text").value.trim();
        if (!input) { setStatus("build-status", "Enter a sequence first.", "error"); return; }
        setStatus("build-status", "Building…");

        const body = { kind, input };
        if (kind === "dna" || kind === "rna") {
            body.backend  = $("backend").value;
            body.form     = $("form").value;
            body.terminal = $("terminal").value;
            // Tri-state add_hydrogens select: "auto" / "on" / "off".
            // Sent as a string; the build endpoint also accepts bool
            // for legacy callers but the form posts the string form.
            body.add_hydrogens        = $("add-hydrogens").value;
            body.protonate_phosphates = $("protonate-phosphates").checked;
        }
        try {
            const r = await fetch("/api/build/molecule", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(body),
            }).then(x => x.json());
            if (!r.ok) {
                setStatus("build-status", r.error || "Build failed.", "error");
                return;
            }
            applyStructureResult(r);
            // Include the backend that ran -- users picking "auto"
            // need to know whether they got X3DNA, Amber, or RDKit
            // because the geometry differs substantially (canonical
            // helix vs extended chain vs folded conformer).
            const via = r.backend_used
                ? ` via ${labelFor(r.backend_used)}`
                : "";
            const built = `Built ${r.n_atoms}-atom structure${via}.`;
            // Surface validation warnings from build-time geometry
            // checks (most importantly H/heavy ratio when the user
            // opted out of add_hydrogens for an X3DNA build).  Errors
            // would have come back as r.ok === false above; here we
            // only see warnings.
            const warns = (r.issues || []).filter(i => i.severity === "warn");
            if (warns.length) {
                const tail = warns.map(i => i.message).join(" • ");
                setStatus("build-status",
                    `${built}  ⚠ ${tail}`, "warn");
            } else {
                setStatus("build-status", built, "ok");
            }
        } catch (e) {
            setStatus("build-status", "Network error: " + e.message, "error");
        }
    });

    // Take a structure response (either /api/build/molecule or /api/build/load) and
    // populate the viewer + info panel + enable the FDF section.
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

    function applyStructureResult(r) {
        state.xyz = r.xyz;
        state.pdb = r.pdb;
        state.title = r.title;
        state.fdf = null;
        state.pyscf = null;
        // Stash the full response so saveStructureState can persist
        // a faithful re-render payload to sessionStorage; the same
        // shape feeds back into applyStructureResult on restore.
        state.last_response = r;
        $("info-title").textContent     = r.title;
        $("info-atoms").textContent     = r.n_atoms;
        $("info-residues").textContent  = r.n_residues || "—";
        $("info-formula").textContent   = formula(r.elements);
        // Update the BlockSize textbox's placeholder to show the
        // auto-picked value for this structure.  Empty input still
        // means "use auto"; the placeholder just makes it visible.
        // Defensive: the schema-driven SIESTA form may not have
        // rendered yet on the very first restoreStructureState()
        // call after a navigation.  Skip silently; the placeholder
        // is just a hint, not load-bearing.  initFormsFromSchema()
        // will re-apply the structure result anyway via
        // restoreStructureState's second invocation path.
        const bs = $("p-block-size");
        if (bs) {
            bs.placeholder =
                "auto (" + autoBlockSize(r.n_atoms) + ", n=" + r.n_atoms + ")";
        }
        $("dl-xyz").disabled = false;
        $("dl-pdb").disabled = false;
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
        clearIssues("fdf-issues");
        clearIssues("pyscf-issues");
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

    // ----- Load existing .xyz / .pdb ----------------------------------
    $("load-file").addEventListener("change", () => {
        $("load-btn").disabled = !$("load-file").files.length;
        setStatus("load-status", "");
    });
    $("load-btn").addEventListener("click", async () => {
        const files = $("load-file").files;
        if (!files.length) {
            setStatus("load-status", "Pick a file first.", "error"); return;
        }
        const file = files[0];
        setStatus("load-status", `Loading ${file.name}…`);
        const fd = new FormData();
        fd.append("file", file);
        try {
            const r = await fetch("/api/build/load", { method: "POST", body: fd })
                            .then(x => x.json());
            if (!r.ok) {
                setStatus("load-status", r.error || "Load failed.", "error");
                return;
            }
            applyStructureResult(r);
            setStatus("load-status",
                `Loaded ${r.n_atoms}-atom ${r.source_format.toUpperCase()} from ${file.name}.`,
                "ok");
        } catch (e) {
            setStatus("load-status", "Network error: " + e.message, "error");
        }
    });

    // formula() lives in static/lib/mol-format.js; loaded by the
    // template above.  Local alias keeps callers below readable.
    const formula = (window.molbuilder && window.molbuilder.fmt
                     ? window.molbuilder.fmt.formula
                     : (els) => (els && els.length ? els.join("") : "—"));

    // ----- 2. Render --------------------------------------------------
    // Sizing math lives in molbuilder/web/static/lib/mol-style.js so the
    // Build and Watch viewers stay in lock-step on representation
    // numerics.  The Build form has no colorscheme picker -- pass null
    // and 3Dmol falls back to the viewer-level Jmol defaults set at
    // createViewer() time above.
    function styleSpec() {
        return molbuilder.style.spec({
            rep:         $("rep").value,
            scale:       parseFloat($("radius").value),
            colorscheme: null,
        });
    }

    function clearLabels() {
        for (const l of state.labels) viewer.removeLabel(l);
        state.labels = [];
    }

    function drawLabels() {
        clearLabels();
        if (!$("show-labels").checked || !state.xyz) {
            viewer.render(); return;
        }
        const lines = state.xyz.split("\n");
        const n = parseInt(lines[0], 10);
        for (let i = 0; i < n; i++) {
            const parts = (lines[i + 2] || "").trim().split(/\s+/);
            if (parts.length < 4) continue;
            const x = parseFloat(parts[1]),
                  y = parseFloat(parts[2]),
                  z = parseFloat(parts[3]);
            const lbl = viewer.addLabel(String(i + 1), {
                position: { x, y, z },
                backgroundColor: "black",
                backgroundOpacity: 0.55,
                fontColor: "white",
                fontSize: 9,
                inFront: true,
                showBackground: true,
            });
            state.labels.push(lbl);
        }
        viewer.render();
    }

    function applyStyle() {
        viewer.setStyle({}, styleSpec());
        viewer.render();
    }

    function renderStructure() {
        viewer.removeAllModels();
        viewer.removeAllLabels();
        state.labels = [];
        if (!state.xyz) return;
        viewer.addModel(state.xyz, "xyz");
        applyStyle();
        viewer.zoomTo();
        drawLabels();
    }

    $("rep").addEventListener("change", applyStyle);
    $("radius").addEventListener("input", applyStyle);
    $("show-labels").addEventListener("change", drawLabels);
    $("bg").addEventListener("change", (e) => {
        viewer.setBackgroundColor(e.target.value);
        viewer.render();
    });

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

    $("dl-xyz").addEventListener("click", () => {
        if (!state.xyz) return;
        downloadAs(state.xyz, safeName(state.title) + ".xyz",
                   "chemical/x-xyz");
    });
    $("dl-pdb").addEventListener("click", () => {
        if (!state.pdb) return;
        downloadAs(state.pdb, safeName(state.title) + ".pdb",
                   "chemical/x-pdb");
    });

    function safeName(s) {
        return (s || "molecule").replace(/[^A-Za-z0-9._-]+/g, "_");
    }

    // ----- 3. Generate FDF -------------------------------------------
    $("generate-fdf").addEventListener("click", async () => {
        if (!state.xyz) {
            setStatus("fdf-status", "Build a structure first.", "error");
            return;
        }
        setStatus("fdf-status", "Rendering FDF…");
        const params = collectFdfParams();
        try {
            const r = await fetch("/api/build/fdf", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ xyz: state.xyz, params }),
            }).then(x => x.json());
            if (!r.ok) {
                setStatus("fdf-status", r.error || "FDF render failed.", "error");
                return;
            }
            state.fdf = r.fdf;
            $("fdf-output").textContent = r.fdf;
            $("fdf-output").hidden = false;
            $("dl-fdf").disabled = false;
            const fdfMsg = `OK — ${r.fdf.split("\n").length} lines, label "${r.system_label}".`;
            // Status line stays terse ("OK -- ... lines").  The
            // structured issues panel below the action row carries
            // the per-issue detail; setStatus's `warn` colour fades
            // in only when there's at least one warning to draw the
            // eye downward to the panel.
            const issues = r.issues || [];
            renderIssues("fdf-issues", issues);
            setStatus("fdf-status", fdfMsg,
                      issues.some(i => i.severity === "warn") ? "warn" : "ok");
        } catch (e) {
            setStatus("fdf-status", "Network error: " + e.message, "error");
        }
    });

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

    // Map the Build form's stage-preset selector value to the
    // SiestaConfig / PySCFConfig ``stage`` integer.  Custom and
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
        // field with a "section": metadata key.  Two post-processing
        // tweaks on top:
        //   1. The web form has ONE user-visible "Job name" input
        //      (system_label).  The Python API has BOTH system_name
        //      and system_label.  We fold them here so the generated
        //      FDF carries a matched pair without exposing two
        //      near-identical fields to the user.
        //   2. The "Relaxation stage" preset is a UI shortcut, not a
        //      dataclass field rendered by the schema; we layer the
        //      stage-number on top of the collected params.
        if (!formSchemas.siesta) return {};
        const container = $("siesta-form-container");
        const params = window.molbuilder.formSchema.collectForm(
            container, formSchemas.siesta
        );
        params.system_name = params.system_label;
        params.stage       = stageNumberFromPreset();
        return params;
    }

    // ----- 4. Generate PySCF script -----------------------------------
    $("generate-pyscf").addEventListener("click", async () => {
        if (!state.xyz) {
            setStatus("pyscf-status", "Build a structure first.", "error");
            return;
        }
        setStatus("pyscf-status", "Rendering PySCF script…");
        const params = collectPyscfParams();
        try {
            const r = await fetch("/api/build/pyscf", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ xyz: state.xyz, params }),
            }).then(x => x.json());
            if (!r.ok) {
                setStatus("pyscf-status",
                    r.error || "PySCF render failed.", "error");
                return;
            }
            state.pyscf = r.script;
            $("pyscf-output").textContent = r.script;
            $("pyscf-output").hidden = false;
            $("dl-pyscf").disabled = false;
            const pyMsg = `OK — ${r.script.split("\n").length} lines, job "${r.job_name}".`;
            const issues = r.issues || [];
            renderIssues("pyscf-issues", issues);
            setStatus("pyscf-status", pyMsg,
                      issues.some(i => i.severity === "warn") ? "warn" : "ok");
        } catch (e) {
            setStatus("pyscf-status", "Network error: " + e.message, "error");
        }
    });

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
        const container = $("pyscf-form-container");
        const params = window.molbuilder.formSchema.collectForm(
            container, formSchemas.pyscf
        );
        params.stage = stageNumberFromPreset();
        if (params.dispersion === "none") params.dispersion = null;
        // Drop nulls.
        Object.keys(params).forEach(k => {
            if (params[k] === null) delete params[k];
        });
        return params;
    }

    // ----- Session state: persist form values across Build↔Watch navigation -----
    // Navigating to /watch and back is a full page reload; sessionStorage
    // survives same-tab navigations so the user's input isn't lost.

    // Static IDs that aren't part of the schema-driven SIESTA /
    // PySCF forms but DO need session-storage persistence: the
    // build-section inputs (kind, sequence, backend, etc.) and the
    // Relaxation-stage preset selector (a UI shortcut, not a
    // dataclass field).  All other persistent IDs are derived at
    // save/restore time by walking the rendered schemas.
    const STATIC_FORM_IDS = [
        "kind", "input-text", "backend", "form", "terminal",
        "add-hydrogens", "protonate-phosphates",
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

    // Map legacy (pre-schema-driven cutover) input IDs to the
    // current schema-derived ones.  Users with sessionStorage from
    // before the 2026-05-11 cutover get one-shot migration of the
    // few fields whose IDs changed: basis_size went from p-basis ->
    // p-basis-size, and the kgrid sub-inputs went from p-kx/p-ky/p-kz
    // to p-k-x/p-k-y/p-k-z.  Add an entry here if a future rename
    // would otherwise drop a field's value on first reload.
    const LEGACY_ID_MIGRATION = {
        "p-basis": "p-basis-size",
        "p-kx":    "p-k-x",
        "p-ky":    "p-k-y",
        "p-kz":    "p-k-z",
    };

    function restoreFormState() {
        let saved;
        try { saved = JSON.parse(sessionStorage.getItem("builder-form") || "null"); }
        catch (_) { return; }
        if (!saved) return;
        // Apply legacy-key migration: copy values stored under the
        // old id to the new one BEFORE the per-id restore loop.
        // Don't overwrite a value already saved under the new id
        // (the user has clearly used the new form since the cutover).
        for (const [oldId, newId] of Object.entries(LEGACY_ID_MIGRATION)) {
            if (oldId in saved && !(newId in saved)) {
                saved[newId] = saved[oldId];
            }
        }
        getFormIds().forEach(id => {
            const el = $(id);
            if (!el || !(id in saved)) return;
            if (el.type === "checkbox") el.checked = saved[id];
            else el.value = saved[id];
        });
        // Re-run nucleic toggle so the nucleic-options row reflects
        // the restored kind value.
        toggleNucleicOptions();
        $("input-text").placeholder = placeholderFor($("kind").value);
    }

    // Build-section restore can run synchronously since those fields
    // are static HTML; the schema-driven SIESTA / PySCF fields get
    // restored a second time inside initFormsFromSchema() after the
    // renderer fills the containers.
    restoreFormState();
    window.addEventListener("pagehide", saveFormState);

    // Kick off the async schema fetch + form render.  Everything
    // that depends on the form's inputs (compatibility engine,
    // change listeners, full restoreFormState walk) happens inside
    // this function once the renderer has populated the DOM.
    initFormsFromSchema();

    // ----- Session state: persist the BUILT / LOADED structure too ---
    // The form-state restore above brings back what the user TYPED;
    // this complementary pair brings back what they actually
    // BUILT.  Without it, clicking Watch and coming back nukes the
    // 3-D viewer's molecule -- the form fields survive but the
    // structure does not, forcing the user to rebuild before they
    // can keep editing.  Phase 1 of the cross-tab persistence work
    // recorded in docs/spec/modify-tab.md.
    //
    // The same key ("builder-structure") is the destination of the
    // upcoming Modify -> Build "Send to Build" handoff (M5): the
    // Modify tab writes the finished junction here, navigates to /,
    // and Build's restore picks it up identically.

    const STRUCTURE_KEY = "builder-structure";
    const STRUCTURE_SCHEMA_VERSION = 1;

    function saveStructureState() {
        if (!state.xyz || !state.last_response) return;
        let camera = null;
        try {
            camera = viewer.getView();
        } catch (_e) {
            // 3Dmol's getView is synchronous; defensive on teardown.
        }
        const payload = {
            v: STRUCTURE_SCHEMA_VERSION,
            saved_at: new Date().toISOString(),
            response: state.last_response,
            camera:   camera,
        };
        try {
            sessionStorage.setItem(STRUCTURE_KEY, JSON.stringify(payload));
        } catch (e) {
            // QuotaExceededError on a structure that doesn't fit
            // (sessionStorage cap is ~5-10 MB).  Skip without
            // crashing -- the user just loses persistence on this
            // particular load.
            console.warn("builder: could not save structure state:",
                         e && e.message);
        }
    }

    function restoreStructureState() {
        let saved = null;
        try {
            saved = JSON.parse(sessionStorage.getItem(STRUCTURE_KEY) || "null");
        } catch (_e) {
            return false;
        }
        if (!saved || saved.v !== STRUCTURE_SCHEMA_VERSION) return false;
        const r = saved.response;
        if (!r || !r.xyz) return false;
        // Replay the build/load through the same path a fresh
        // /api/build/molecule response would take.  This keeps the
        // info panel, atom counts, and the disabled/enabled state of
        // the Generate buttons consistent with a fresh build.
        applyStructureResult(r);
        // Restore the camera last so it doesn't fight the zoomTo()
        // inside renderStructure().
        if (Array.isArray(saved.camera)) {
            try {
                viewer.setView(saved.camera);
                viewer.render();
            } catch (_e) {
                viewer.zoomTo();
                viewer.render();
            }
        }
        return true;
    }

    // Restore AFTER the form-state restore above so the Build form
    // fields match the structure if the user had built (e.g. a DNA
    // sequence is in the input box AND the helix is in the viewer).
    restoreStructureState();
    window.addEventListener("pagehide", saveStructureState);

    // ----- Staged-relaxation presets ----------------------------- //
    // The Watch tab carries the full workflow guide; the Build tab's
    // job is to make it one-click to fill the SIESTA convergence
    // params for each stage.  Same SystemLabel + same directory ->
    // SIESTA reads the previous stage's .XV and .DM automatically.
    //
    // Values are the ones documented in the Watch tab's recipe table.
    // The ``custom`` option does NOT reset anything -- it just stops
    // auto-filling so the user can fine-tune individual fields.
    const STAGE_PRESETS = {
        coarse: {
            "p-mesh-cutoff":         200,
            "p-pao-energy-shift":    0.02,
            "p-mixing-weight":       0.05,
            "p-dm-tolerance":        1e-3,
            "p-dm-energy-tolerance": 1e-3,
            "p-relax-steps":         80,
            "p-force-tol":           0.04,
            "p-max-displ":           0.20,
        },
        medium: {
            "p-mesh-cutoff":         300,
            "p-pao-energy-shift":    0.01,
            "p-mixing-weight":       0.02,
            "p-dm-tolerance":        1e-4,
            "p-dm-energy-tolerance": 1e-4,
            "p-relax-steps":         40,
            "p-force-tol":           0.02,
            "p-max-displ":           0.10,
        },
        tight: {
            "p-mesh-cutoff":         400,
            "p-pao-energy-shift":    0.005,
            "p-mixing-weight":       0.01,
            "p-dm-tolerance":        1e-5,
            "p-dm-energy-tolerance": 1e-5,
            "p-relax-steps":         30,
            "p-force-tol":           0.01,
            "p-max-displ":           0.05,
        },
    };

    function applyStagePreset(stage) {
        const preset = STAGE_PRESETS[stage];
        if (!preset) return;
        Object.entries(preset).forEach(([id, value]) => {
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
