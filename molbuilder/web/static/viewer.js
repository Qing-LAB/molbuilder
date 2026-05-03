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
            body.add_hydrogens        = $("add-hydrogens").checked;
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
        $("info-title").textContent     = r.title;
        $("info-atoms").textContent     = r.n_atoms;
        $("info-residues").textContent  = r.n_residues || "—";
        $("info-formula").textContent   = formula(r.elements);
        // Update the BlockSize textbox's placeholder to show the
        // auto-picked value for this structure.  Empty input still
        // means "use auto"; the placeholder just makes it visible.
        $("p-block-size").placeholder =
            "auto (" + autoBlockSize(r.n_atoms) + ", n=" + r.n_atoms + ")";
        $("dl-xyz").disabled = false;
        $("dl-pdb").disabled = false;
        $("generate-fdf").disabled = false;
        $("generate-pyscf").disabled = false;
        // Stale outputs / status / download / handoff buttons -> reset
        $("dl-fdf").disabled = true;
        $("dl-pyscf").disabled = true;
        $("watch-fdf").disabled = true;
        $("watch-pyscf").disabled = true;
        $("fdf-output").hidden = true;
        $("fdf-output").textContent = "";
        $("pyscf-output").hidden = true;
        $("pyscf-output").textContent = "";
        setStatus("fdf-status", "");
        setStatus("pyscf-status", "");
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

    // Wire change events for every input that triggers a rule.  We
    // listen on `change` rather than `input` so rapid typing in a
    // number field doesn't thrash the DOM; the rules only depend on
    // dropdown values and checkbox states anyway.
    [
        "py-method", "py-optimize", "py-preopt", "py-solvent",
        "p-spin-polarized", "p-relax",
    ].forEach(id => {
        const el = $(id);
        if (el) el.addEventListener("change", applyCompatibility);
    });

    // Initial state on page load.
    applyCompatibility();

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

    function formula(elements) {
        if (!elements || !elements.length) return "—";
        const counts = {};
        elements.forEach(e => counts[e] = (counts[e] || 0) + 1);
        const order = ["C", "H", "N", "O", "P", "S"];
        const parts = [];
        order.forEach(e => {
            if (counts[e]) {
                parts.push(counts[e] > 1 ? `${e}${counts[e]}` : e);
                delete counts[e];
            }
        });
        Object.keys(counts).sort().forEach(e => {
            parts.push(counts[e] > 1 ? `${e}${counts[e]}` : e);
        });
        return parts.join("");
    }

    // ----- 2. Render --------------------------------------------------
    function styleSpec() {
        const rep = $("rep").value;
        const scale = parseFloat($("radius").value) || 1.0;
        switch (rep) {
            case "sphere":
                return { sphere: { scale: 1.0 * scale } };
            case "stick":
                return { stick: { radius: 0.16 * scale },
                         sphere: { scale: 0.18 * scale } };
            case "line":
                return { line: { linewidth: 1 + 2 * scale } };
            case "ballstick":
            default:
                return { stick: { radius: 0.12 * scale },
                         sphere: { scale: 0.32 * scale } };
        }
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
            state.fdf_label = r.system_label;
            $("fdf-output").textContent = r.fdf;
            $("fdf-output").hidden = false;
            $("dl-fdf").disabled = false;
            $("watch-fdf").disabled = false;
            const fdfMsg = `OK — ${r.fdf.split("\n").length} lines, label "${r.system_label}".`;
            const fdfWarns = (r.issues || []).filter(i => i.severity === "warn");
            if (fdfWarns.length) {
                setStatus("fdf-status",
                    `${fdfMsg}  ⚠ ${fdfWarns.map(i => i.message).join(" • ")}`,
                    "warn");
            } else {
                setStatus("fdf-status", fdfMsg, "ok");
            }
        } catch (e) {
            setStatus("fdf-status", "Network error: " + e.message, "error");
        }
    });

    $("dl-fdf").addEventListener("click", () => {
        if (!state.fdf) return;
        const label = ($("p-system-label").value.trim() || "siesta").replace(
            /[^A-Za-z0-9._-]+/g, "_");
        downloadAs(state.fdf, label + ".fdf");
    });

    // Watch this run -- open /watch with the predicted molwatch.log
    // filename pre-filled.  The user edits the path to absolute (the
    // server doesn't know where they'll run the calculation), then
    // clicks Load on the Watch tab.
    $("watch-fdf").addEventListener("click", () => {
        if (!state.fdf_label) return;
        const path = `${state.fdf_label}.molwatch.log`;
        window.open(`/watch?path=${encodeURIComponent(path)}`, "_blank");
    });

    function collectFdfParams() {
        const num  = (id) => parseFloat($(id).value);
        const int  = (id) => parseInt($(id).value, 10);
        const bool = (id) => $(id).checked;
        return {
            system_name:            $("p-system-name").value.trim(),
            system_label:           $("p-system-label").value.trim(),

            // Basis & grid
            basis_size:             $("p-basis").value,
            mesh_cutoff:            num("p-mesh-cutoff"),
            pao_energy_shift:       num("p-pao-energy-shift"),

            // XC
            xc_functional:          $("p-xc-functional").value,
            xc_authors:             $("p-xc-authors").value.trim(),

            // SCF
            solution_method:        $("p-solution-method").value,
            mixing_weight:          num("p-mixing-weight"),
            pulay_history:          int("p-pulay-history"),
            dm_tolerance:           num("p-dm-tolerance"),
            dm_energy_tolerance:    num("p-dm-energy-tolerance"),
            max_scf_iter:           int("p-max-scf-iter"),
            electronic_temperature: num("p-temperature"),

            // Spin
            spin_polarized:         bool("p-spin-polarized"),
            spin_total:             $("p-spin-total").value === ""
                                    ? null : num("p-spin-total"),

            // k-grid
            kgrid:                  [int("p-kx"), int("p-ky"), int("p-kz")],

            // Parallel execution -- empty BlockSize / "auto" ParallelOverK
            // mean "let the backend auto-pick", which is the recommended
            // path; the textbox is here only for power users who want a
            // specific value.
            parallel_block_size:    ($("p-block-size").value === ""
                                    ? null : int("p-block-size")),
            parallel_over_k:        ($("p-parallel-over-k").value === "auto"
                                    ? null : $("p-parallel-over-k").value === "true"),

            // Relaxation
            relax_type:             $("p-relax").value,
            relax_steps:            int("p-relax-steps"),
            relax_force_tol:        num("p-force-tol"),
            relax_max_displ:        num("p-max-displ"),

            // Output + positioning + comments
            write_coor_xmol:        bool("p-write-coor-xmol"),
            write_md_history:       bool("p-write-md-history"),
            write_hs:               bool("p-write-hs"),
            wrap_into_cell:         bool("p-wrap-into-cell"),
            center_in_vacuum:       bool("p-center-in-vacuum"),
            verbose_comments:       bool("p-verbose-comments"),
        };
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
            state.pyscf_label = r.job_name;
            $("pyscf-output").textContent = r.script;
            $("pyscf-output").hidden = false;
            $("dl-pyscf").disabled = false;
            $("watch-pyscf").disabled = false;
            const pyMsg = `OK — ${r.script.split("\n").length} lines, job "${r.job_name}".`;
            const pyWarns = (r.issues || []).filter(i => i.severity === "warn");
            if (pyWarns.length) {
                setStatus("pyscf-status",
                    `${pyMsg}  ⚠ ${pyWarns.map(i => i.message).join(" • ")}`,
                    "warn");
            } else {
                setStatus("pyscf-status", pyMsg, "ok");
            }
        } catch (e) {
            setStatus("pyscf-status", "Network error: " + e.message, "error");
        }
    });

    $("dl-pyscf").addEventListener("click", () => {
        if (!state.pyscf) return;
        const label = ($("py-job-name").value.trim() || "pyscf_relax")
            .replace(/[^A-Za-z0-9._-]+/g, "_");
        downloadAs(state.pyscf, label + ".py", "text/x-python");
    });

    $("watch-pyscf").addEventListener("click", () => {
        if (!state.pyscf_label) return;
        const path = `${state.pyscf_label}.molwatch.log`;
        window.open(`/watch?path=${encodeURIComponent(path)}`, "_blank");
    });

    function collectPyscfParams() {
        const str  = (id) => $(id).value;
        const trim = (id) => $(id).value.trim();
        const num  = (id) => {
            const v = $(id).value.trim();
            return v === "" ? null : parseFloat(v);
        };
        const int  = (id) => {
            const v = $(id).value.trim();
            return v === "" ? null : parseInt(v, 10);
        };
        const bool = (id) => $(id).checked;
        const params = {
            // System
            job_name:           trim("py-job-name") || "pyscf_relax",
            charge:             int("py-charge"),       // null -> auto-detect
            spin:               int("py-spin")  || 0,
            symmetry:           bool("py-symmetry"),

            // Method
            method:             str("py-method"),
            functional:         trim("py-functional"),
            basis:              trim("py-basis"),
            dispersion:         str("py-dispersion"),   // "none" -> server maps to null
            density_fit:        bool("py-density-fit"),

            // SCF
            scf_conv_tol:       num("py-scf-conv-tol"),
            scf_max_cycle:      int("py-scf-max-cycle"),
            scf_init_guess:     str("py-init-guess"),
            grid_level:         int("py-grid-level"),
            level_shift:        num("py-level-shift"),

            // Pre-opt
            preopt:             bool("py-preopt"),
            preopt_functional:  trim("py-preopt-functional"),
            preopt_basis:       trim("py-preopt-basis"),
            preopt_max_steps:   int("py-preopt-max-steps"),
            preopt_grms:        num("py-preopt-grms"),

            // Main opt
            optimize:           bool("py-optimize"),
            optimizer:          str("py-optimizer"),
            geom_max_steps:     int("py-geom-max-steps"),
            geom_conv_energy:   num("py-geom-conv-energy"),
            geom_conv_grms:     num("py-geom-conv-grms"),
            geom_conv_gmax:     num("py-geom-conv-gmax"),

            // Solvent
            solvent:            str("py-solvent"),      // "" -> server maps to null
            solvent_method:     str("py-solvent-method"),

            // Runtime / output
            max_memory_mb:      int("py-max-memory"),
            threads:            int("py-threads"),
            verbose:            int("py-verbose"),
            chkfile:            bool("py-chkfile"),
            log_file:           bool("py-log-file"),
            verbose_comments:   bool("py-verbose-comments"),
        };
        // Drop null-valued keys so the server-side dataclass uses its
        // declared default rather than getting None where it expects an
        // int / float.
        Object.keys(params).forEach(k => {
            if (params[k] === null) delete params[k];
        });
        return params;
    }
})();
