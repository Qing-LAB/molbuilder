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

    // ----- Source-mode toggle (Build vs Load) -----
    //
    // The two input paths are mutually exclusive: the user is
    // EITHER entering a sequence/SMILES/name to build a structure,
    // OR loading an external .xyz / .pdb file.  Both panels can't
    // be visible at the same time -- having two competing inputs
    // (sequence vs file) on screen confuses the "did the wrong one
    // just fire?" diagnosis when something fails.
    //
    // Switching modes does NOT clear state.xyz: a structure built
    // moments ago stays in the viewer even after the user flips to
    // load mode to compare it with a saved file.  Switching just
    // changes what the next "produce a structure" action does.
    function applySourceMode(mode) {
        const buildOn = (mode === "build");
        $("build-panel").hidden = !buildOn;
        $("load-panel").hidden  =  buildOn;
        // Status messages from the OTHER panel are stale once we
        // switch -- clear them so a "Loaded foo.xyz" message doesn't
        // hang around while the user is mid-build.
        if (buildOn) {
            setStatus("load-status",  "");
        } else {
            setStatus("build-status", "");
        }
    }
    document.querySelectorAll('input[name="source-mode"]').forEach(r => {
        r.addEventListener("change", (e) => applySourceMode(e.target.value));
    });
    // Initial state mirrors the HTML `checked` attribute on the
    // build radio; this call is for tests / restored sessions where
    // the radio state may have been set programmatically.
    applySourceMode(
        (document.querySelector('input[name="source-mode"]:checked') || {}).value
        || "build"
    );
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
                clearStructureInfo("load failed");
                return;
            }
            applyStructureResult(r);
            setStatus("load-status",
                `Loaded ${r.n_atoms}-atom ${r.source_format.toUpperCase()} from ${file.name}.`,
                "ok");
        } catch (e) {
            setStatus("load-status", "Network error: " + e.message, "error");
            clearStructureInfo("load failed");
        }
    });

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
    if (window.molbuilder && window.molbuilder.runtime
        && typeof window.molbuilder.runtime.whenReady === "function") {
        window.molbuilder.runtime.whenReady("projects").then((_proj) => {
        _proj.onChange(async (sel) => {
            const f = (sel && sel.file) ? String(sel.file) : "";
            const ext = f.toLowerCase().split(".").pop();
            if (ext !== "xyz" && ext !== "pdb") {
                // User picked a non-structure file (or no file).  The
                // load-status line still carries whatever the last
                // successful load said -- leaving it would lie to the
                // user ("Loaded 3-atom XYZ from water.xyz" while
                // they're now looking at junction.log).  Sync BOTH
                // the status line AND the info readout so the user
                // sees a consistent state: their pick did NOT load.
                if (f) {
                    const filename = f.split("/").pop();
                    setStatus("load-status",
                        `${filename} is not a structure file `
                        + `(Build accepts .xyz / .pdb only).`,
                        "warn");
                    // Don't clear info -- the previous valid load is
                    // still showing in the viewer + the structure
                    // panel; surfacing a "no structure" placeholder
                    // would over-correct.  The status line tells the
                    // user this pick wasn't a load attempt.
                } else {
                    setStatus("load-status", "");
                }
                _sidebarLastFile = f;
                return;
            }
            if (f === _sidebarLastFile) return;
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
                setStatus("load-status",
                    `Loaded ${r.n_atoms}-atom ${ext.toUpperCase()} `
                    + `from ${filename}.`, "ok");
            } catch (e) {
                if (mySeq !== _sidebarLoadSeq) return;
                setStatus("load-status",
                    "Network error: " + e.message, "error");
                clearStructureInfo("load failed: " + filename);
            }
        });
        });   // close runtime.whenReady("projects").then(...)
    }

    // formula() lives in static/lib/mol-format.js; loaded by the
    // template above.  Local alias keeps callers below readable.
    const formula = (window.molbuilder && window.molbuilder.fmt
                     ? window.molbuilder.fmt.formula
                     : (els) => (els && els.length ? els.join("") : "—"));

    // ----- 2. Render --------------------------------------------------
    // The standard knob bar (style picker / labels popover / axes /
    // reset / background / export) owns all the per-viewer chrome
    // post-migration; Build's host page only owns its tab-specific
    // controls (Build form, Generate buttons, Download row).  The
    // bespoke <details> Style block in index.html is gone with this
    // migration; rep / radius / background / labels move into the
    // knob bar where every consumer site shares the same UX.

    function renderStructure() {
        if (!state.xyz) {
            // No structure to mount.  setStructure with an empty
            // xyz/pdb is a no-op in the embed; Build's empty state
            // is rare (only at first page load before any build /
            // load).  Leaves whatever was previously rendered
            // until the user builds something.
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
        // next to the loaded XYZ/PDB.  Without this hop /modify's
        // frozen_atoms list NEVER reaches render_fdf and SIESTA relaxes
        // every atom even when the user explicitly froze some.  Mirrors
        // /spectra; was the silent gap that prompted the 2026-05-25 fix.
        const _structPath = (_proj && _proj.getCurrentFile()) || "";
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
                    structure_path: _structPath || null,
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
            renderIssues("fdf-issues", issues);
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
            const w = await proj.saveToWorkspace(
                state.fdf, filename,
                { overwrite: true, signal: abortSignal });
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
            } catch (_) { /* non-fatal; user can run the .fdf manually */ }
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
        const _projPy = (window.molbuilder || {}).projects;
        const _structPathPy = (_projPy && _projPy.getCurrentFile()) || "";
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
                    structure_path: _structPathPy || null,
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
            renderIssues("pyscf-issues", issues);
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
            const w = await proj.saveToWorkspace(
                state.pyscf, meta.filename,
                { overwrite: true, signal: abortSignal });
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
        } catch (_) { /* non-fatal */ }
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
    // recorded in docs/tabs/modify.md.
    //
    // The same key ("builder-structure") is the destination of the
    // upcoming Modify -> Build "Send to Build" handoff (M5): the
    // Modify tab writes the finished junction here, navigates to /,
    // and Build's restore picks it up identically.

    const STRUCTURE_KEY = "builder-structure";
    const STRUCTURE_SCHEMA_VERSION = 1;

    function saveStructureState() {
        if (!state.xyz || !state.last_response) return;
        // Per § 3.13: getCamera returns an opaque CameraState
        // {_viewer, _version, data} we round-trip through
        // setCamera at restore time.  Drops the raw viewer.getView
        // touchpoint that #202 migration eliminated.
        const camera = _handle.getCamera();
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
        // inside renderStructure().  Per § 3.13 setCamera no-ops
        // on _viewer / _version mismatch (forward-compat).
        if (saved.camera) {
            _handle.setCamera(saved.camera);
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
            // 150 outer geometry steps for the final tight stage.
            // Earlier value (30) was BACKWARDS for a tight stage:
            // tight = small displacement cap (0.05 Å) + strict
            // force tolerance (0.01 eV/Å) means each step covers
            // ~4× less ground AND we're chasing a 4× smaller
            // residual force.  30 steps would converge only if
            // the previous stage already left us very close.
            "p-relax-steps":         150,
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
