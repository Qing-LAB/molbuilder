/* molbuilder web UI client — the structure-optimization tab page controller.
 *
 * WHAT THIS TAB DOES, and it is less than it used to:
 *   1. Show the structure the sidebar loaded, through the MolView component.
 *   2. Render the parameter form from the CATALOGUE (`/api/build/schema`,
 *      fetched by the shared form library) and collect what the user sets.
 *   3. Validate it live — POST /api/build/preflight — and place each finding
 *      beside the control it is about.
 *   4. Offer the auto-detected charge / spin — POST /api/structure/analyze.
 *
 * IT PRODUCES NO ARTIFACT.  It renders no deck, writes no file and hands
 * nothing to `prep`.  Those two POSTs are the only calls it makes.
 *
 * This header described a third concern until 2026-08-15 — *"POST
 * /api/build/fdf with the XYZ + form values -> get back FDF text, offer it as
 * a Blob download"* — and that flow was removed when script generation and
 * staging left this tab (user: the tab collects parameters, the staging
 * surface owns the rest).  The route still exists in `build.py`; this UI no
 * longer exposes it.  A file header that describes a removed feature is the
 * worst kind of stale, because it is the first thing a reader trusts.
 *
 * MolView is consumed through its ONE public door (molview-esm-finalization.md): the ES-module
 * import below.  No `window.molbuilder.molview` / `.fmt` global reads — those are the transitional
 * shims we are dumping.  Workspace (`window.molbuilder.workspace`) is a SEPARATE module, still a
 * classic global, read at call time.
 */
import { mount as mvMount, formula as mvFormula }
    from "/static/lib/molview/index.js";

(function () {
    "use strict";
    /* WHO THESE BYTES BELONG TO (workspace.md § 4) — the one string used BOTH as the
     * viewer's `owner` and as the tag on every workspace call, so the two cannot
     * drift into naming different slots. */
    const WORKSPACE_TAG = "structure-opt";

    const $ = (id) => document.getElementById(id);

    /**
     * Format a fetch / response-parse exception into a
     * user-friendly status banner string.  Distinguishes:
     *
     *   * ``SyntaxError`` -- ``r.json()`` failed because the server
     *     returned non-JSON (5xx with an HTML error page, 501
     *     stub, proxy plain-text drop).  Surfacing as "network
     *     error" misleads the user into checking their connection
     *     when the server itself crashed.
     *   * Anything else -- genuine network failure (TypeError
     *     "Failed to fetch", DNS issue, CORS preflight rejection),
     *     surface as "Network error: <message>".
     *
     * AbortError is caller's responsibility -- this helper is
     * called AFTER the caller has filtered abort.
     *
     * L2 round-4 fix (R4-A finding #4): same pattern as
     * ``projects/api.js::_fetchEnvelope`` lines 74-83.
     */
    function _formatFetchError(e) {
        if (e && e.name === "SyntaxError") {
            return "Server returned non-JSON response "
                 + "(likely a 5xx error page).  Check the server "
                 + "log for the actual failure.";
        }
        return "Network error: "
             + (e && e.message ? e.message : String(e));
    }

    // ----- State ------------------------------------------------------
    const state = {
        pdb: null,
        title: null,
        labels: [],           // 3Dmol label objects so we can clear them
        fdf: null,
        pyscf: null,
        // NOTHING structural is held here.  Coordinates, labels and periodicity
        // are read LIVE off the viewer at request time, in ONE read
        // (getStructure) -- contract F1 (docs/science/validation.md 4.1).  ``state.xyz`` used to live here,
        // filled once at load, so a request could carry fresh labels + fresh
        // periodicity + STALE coordinates and validation judged a structure the
        // viewer was not showing.  There is no mirror to desync now.
    };

    /* (`_facts()` was deleted 2026-08-15.)  A one-line wrapper over
     * `_data().getStructure()` with no callers left.  Its own comment records
     * how it got here: it replaced `factsForRequest()`, *"a second door whose
     * only job was to hand back the same facts in a different shape"* — and
     * then `_structureForRequest()` below became the one door, leaving this
     * as a third.  The rule it stated is worth keeping and now lives where it
     * is used: read the structure ONCE, so the facts that leave together were
     * read together (F1, docs/science/validation.md § 4.1). */

    /* THE STRUCTURE AS THE SERVER TAKES IT — ASKED FOR, NOT ASSEMBLED.
     *
     * `exportFile()` is the viewer's own producer: the atoms, their positions
     * at the frame on screen, and the facts beside them, in the one envelope
     * every structure door reads (molview.md § 9.3 — the facts that leave
     * together were read together).
     *
     * IT USED TO BE BUILT HERE, by hand, and got the shape wrong: the cell went
     * in as `metadata: {periodicity: …}`, a key the envelope does not define,
     * so the receiver refused the whole body. That was invisible while the
     * doors still read the legacy `xyz` text field and ignored the envelope
     * entirely — the moment they started reading it, every Generate and every
     * preflight on this tab answered 400.
     *
     * Before that it was an XYZ DOCUMENT, asked of a viewer that has none and
     * writes none (§ 11.7), so every request carried an empty string. Three
     * shapes, one door: the door's own producer is the answer. */
    function _structureForRequest() {
        const d = _data();
        const out = d ? d.exportFile() : null;
        return (out && out.structure) ? out.structure : null;
    }

    // Embedded MolViewer (#198, 2026-06-02; contract:
    // docs/web/molview.md).  Site migration #202
    // landed 2026-06-03 — Build now uses the standard knob bar
    // (Style / Labels / Axes / Reset / PNG / Background / Export)
    // owned by the embed.  The bespoke <details> Style block in
    // index.html is gone; getCamera/setCamera handle the cross-
    // session camera persistence (was raw viewer.getView/setView);
    // the embed's internal ResizeObserver tracks #viewer's
    // resizable container box (was a bespoke RO + window-resize
    // listener).  The handle is the only viewer touchpoint.
    // The FULL concealed MolView component — the SAME rich card Modify (/molbuilder)
    // mounts (fused viewer + selection/cell panel + view toggles) — but mode:"readonly"
    // because this tab READS the structure (to generate SIESTA/PySCF scripts), it does
    // not edit geometry.  The structure lives in molview.data as the single source of
    // truth.  Mounted lazily on the first renderStructure().
    // MolView is imported through its one door (top of file): `mvMount` and
    // `mvFormula`.  `_data()` is THE VIEWER THIS PAGE MOUNTED — the handle
    // `mvMount` handed back, held in `_mvHandle` — and not a name looked up in a
    // global, because a viewer belongs to whoever mounted it (molview.md § 5.6).
    // Workspace is a separate module, read at call time.
    const _ws   = () => (window.molbuilder && window.molbuilder.workspace) || null;
    const _data = () => ((_mvHandle && _mvHandle.ok) ? _mvHandle.data : null);
    let _mvHandle = null;

    /* ── WHAT THIS TAB REMEMBERS ──────────────────────────────────────────
     *
     * Two things, and they are the two a user would be annoyed to lose on a
     * tab switch: THE MOLECULE THEY LOADED, and THE PARAMETERS THEY TYPED.
     * The parameters already survive (see the form-state block near the end of
     * this file); this is the molecule.
     *
     * IT IS THE TAB'S SAVE, NOT THE VIEWER'S. This viewer is read-only, which
     * means its timeline does nothing -- no saved points, no Retract, nothing
     * written of its own accord. That is about the SEQUENCE, not about the
     * page: what the tab put on screen is the tab's own fact, kept under the
     * tab's own tag beside its form values (workspace.md § 4 -- several savers
     * on one page, each deciding what and when).
     *
     * WHAT IS SAVED IS THE STRUCTURE, NOT THE PATH. A path would be re-read on
     * the way back, so a file changed on disk while you were away would quietly
     * replace what you had; the structure comes back as you left it, and the
     * Load button is what goes and gets the new bytes.
     */
    const _SAVED = () => ({ workspace_id: _ws().workspaceId(WORKSPACE_TAG),
                            state_index:  0 });

    function _rememberStructure() {
        const ws = _ws();
        const d  = _data();
        if (!ws || !d) return;
        // What the viewer hands out as data -- atoms, labels, cell -- which is
        // exactly what it can be handed back (molview.md § 9.3).
        const out = d.exportFile();
        if (!out || !out.structure) return;
        ws.persist(WORKSPACE_TAG,
                   { v: 1, structure: out.structure, loadedFrom: _sidebarLastFile || "" },
                   _SAVED());
    }

    /* Put it back, or answer null when there is nothing to put back.
     *
     * `installMolecule` into a viewer that holds nothing is allowed in
     * read-only -- what read-only refuses is EDITING what is on screen, and
     * this is a page arriving with no structure at all. */
    async function _restoreStructure() {
        const ws = _ws();
        const d  = _data();
        if (!ws || !d) return null;
        let saved;
        try {
            saved = await ws.readState(_SAVED());
        } catch (_e) {
            return null;          // unreachable storage reads as "nothing saved"
        }
        // Bytes from a layout this build does not know are not something to
        // guess at; an empty canvas is the honest answer.
        if (!saved || saved.v !== 1 || !saved.structure) return null;
        await d.installMolecule({ structure: saved.structure });
        return saved;
    }

    // ----- Status helpers --------------------------------------------
    function setStatus(elId, msg, kind) {
        const el = $(elId);
        // Null-guarded: a missing status slot must never turn the REPORT of a
        // failure into a throw of its own (that made a MolView mount failure
        // completely invisible -- the catch handler died on a phantom id).
        if (!el) {
            if (window.console) console.warn("[structure-opt] no #" + elId
                + " status slot for: " + msg);
            return;
        }
        el.textContent = msg;
        el.className = "status" + (kind ? " " + kind : "");
    }

    // Findings rendering lives in ONE module for the whole app
    // (lib/validation-findings.js, contract R2 in
    // docs/science/validation.md 4.1).  The three per-tab copies this
    // page used to own each silently dropped a finding whose
    // workflow_group named a card the schema had not rendered; the
    // shared module iterates the FINDINGS instead of the panels, so
    // an unroutable one lands in the residual panel (R3).  These two
    // thin wrappers keep the existing call sites readable.
    /* {field name -> the DOM id the form gave it}, from the schema this page
     * actually rendered.  Read from the schema rather than derived, because
     * the schema is where that rule lives; a copy of it here would be a
     * second answer to a question with one. */
    function _fieldIds(engine) {
        const sch = formSchemas[engine];
        if (!sch || !Array.isArray(sch.sections)) return null;
        const out = {};
        for (const sect of sch.sections) {
            for (const fld of sect.fields) out[fld.name] = fld.id;
        }
        return out;
    }

    function renderIssues(panelId, issues, formContainerId) {
        const f = (window.molbuilder || {}).validationFindings;
        if (!f) return { total: 0, residual: 0, byGroup: {}, byField: {},
                         counts: { error: 0, warn: 0, info: 0 } };
        // The engine is the container's own name -- the two are one choice.
        const engine = (formContainerId || "").startsWith("pyscf")
            ? "pyscf" : "siesta";
        return f.render(issues, {
            panel:     $(panelId),
            formScope: formContainerId ? $(formContainerId) : null,
            fieldIds:  _fieldIds(engine),
        });
    }

    function clearIssues(panelId, formContainerId) {
        const f = (window.molbuilder || {}).validationFindings;
        if (!f) return;
        f.clear({ panel:     $(panelId),
                  formScope: formContainerId ? $(formContainerId) : null });
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
    // (a structure is loaded) and they tweak any tab's params, hit
    // /api/build/preflight to refresh the issues panel before they
    // click Generate.  No-op until the first successful build, so a
    // user who just landed on the page and is fiddling with form
    // defaults doesn't see warnings that haven't been earned yet.
    async function refreshPreflight(engine) {
        if (!_structureForRequest()) return;
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
                /* The cell rides inside `structure`, from the same read as the
                 * atoms -- the preflight has to judge the structure the user is
                 * looking at, not a cell fetched a moment later. */
                body: JSON.stringify({ structure: _structureForRequest(),
                                       engine, params }),
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
       structure of n_atoms.  Mirrors the SIZE-ONLY branch of
       `_auto_block_size` in molbuilder/siesta/input.py (the one taken
       when no rank count is set) -- if either side changes the rule,
       the other must follow.  Used only to label the BlockSize
       textbox's placeholder; the actual value still comes from the
       backend, which also knows mpi_np and the GPU toggle. */
    /* (autoBlockSize deleted 2026-08-15.)  It hand-copied the size-only
       branch of `_auto_block_size` in molbuilder/siesta/input.py to label
       the BlockSize placeholder, and its own comment conceded the cost:
       "if either side changes the rule, the other must follow."  It never
       knew the rank count -- and since the rank count moved to the staging
       surface it never could -- so the number it showed was not the one the
       backend would pick.  The placeholder is the catalogue's "(auto)" now,
       and the rule has one home again. */

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

    // Populate the page state + info panel FROM THE MODEL (the single source of truth
    // the ONE load door installed) -- NOT from a hand-rolled /api/build/load response +
    // a separate sidecar fetch.  The Generate/preflight POST reads the live facts + the
    // sidecar labels; the info panel reads the counts; all come off molview.data.
    function _applyLoadedModel(filename) {
        const d = _data();
        const s = d ? d.getStructure() : null;
        /* THE COUNT COMES OFF THE ELEMENTS, because that is what the structure
         * carries. This read `s.atoms` — a key the envelope has never had
         * (`elements`, `annotations`, `periodicity`, `frames`, `forcesPerFrame`)
         * — so it was `undefined`, the count fell to the empty array, and the
         * header said "0 atoms" beside a drawn molecule. The FORMULA was right
         * the whole time, because it took the `getElements()` branch: one line
         * reading the master copy properly, one line guessing at it. */
        const elements = (d && typeof d.getElements === "function")
            ? (d.getElements() || []) : [];
        const n_atoms = elements.length;
        state.title = filename;
        // Labels (regions / frozen) are NOT mirrored into state -- whatever
        // consumes them reads them FRESH off the model (getRegions / getFrozen)
        // so they can never desync (unified-API access, no state.* copy).
        $("info-title").textContent     = filename;
        $("info-atoms").textContent     = n_atoms;
        $("info-residues").textContent  = "—";
        $("info-formula").textContent   = formula(elements);
        // The BlockSize placeholder is left as the catalogue's own
        // ``null_label`` -- "(auto)".  It read
        // ``auto (<size>, n=<atoms>)`` until 2026-08-15, and the parenthesis
        // told the user nothing they could act on: ``n`` was the ATOM COUNT,
        // which in a parallel-execution field reads as a rank count, and the
        // size shown was only the no-ranks branch of the backend rule -- so
        // on any real run it was not the number that would be used.  A
        // prediction that is usually wrong is worse than no prediction
        // (user, 2026-08-15).
        // A new structure invalidates the previous run's issue panels;
        // refresh both with a preflight tick so the user sees only
        // issues against the new geometry + current params.
        clearIssues("fdf-issues", "siesta-form-container");
        clearIssues("pyscf-issues", "pyscf-form-container");
        refreshPreflightDebounced.siesta();
        refreshPreflightDebounced.pyscf();
        _ensureMounted();
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

        // optimize=false -> optimizer choice + per-stage ladder moot.
        // The stage-table widget renders its own enabled-stage rows;
        // we only need to lock the optimizer dropdown here, since the
        // stage-table can still be edited (and a future ``optimize=
        // true`` flip would carry those edits forward).
        const optimize = $("py-optimize") && $("py-optimize").checked;
        const optReason = optimize ? null
            : "Geometry optimization is disabled (set 'Optimize geometry' on).";
        setLock("py-optimizer", optReason);

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
        // SpinTotal is only meaningful once the spin treatment is something
        // other than non-polarized.
        //
        // THIS GATE WAS BROKEN, not merely stale (found 2026-08-15).  It read
        // ``$("p-spin-polarized").checked`` -- a CHECKBOX that stopped
        // existing when `spin_polarized` became the four-state
        // `spin_treatment` enum.  ``$()`` returned null, the `&&` made the
        // gate permanently false, and so `p-spin-total` was locked FOREVER
        // with a message telling the user to tick a control that is not on
        // the page.  A dead id does not throw; it quietly answers "no".
        const spinSel = $("p-spin-treatment");
        const polarised = !!spinSel && spinSel.value !== "non-polarized";
        setLock("p-spin-total",
                polarised ? null
                          : "Set 'Spin treatment' to something other than "
                            + "non-polarized; SpinTotal is ignored without "
                            + "spin polarisation.");

        // Relaxation type "none" -> per-step relaxation params moot.
        const relax = $("p-relax-type") && $("p-relax-type").value;
        const noneReason =
            (relax === "none")
                ? "Single-point only (no MD block emitted in the FDF)."
                : null;
        ["p-relax-steps", "p-relax-force-tol", "p-relax-max-displ"]
            .forEach(id => setLock(id, noneReason));

        // (The GPU/diagonaliser auto-switch was deleted 2026-08-15.)
        //
        // It read ``$("p-enable-gpu").checked`` and, if ScaLAPACK was
        // selected, silently rewrote the user's diagonaliser to
        // ELPA-1STAGE.  Three things were wrong with keeping it:
        //
        //   * ``p-enable-gpu`` is not on this form any more -- the GPU flag
        //     is a bench axis answered on the staging surface -- so the gate
        //     was dead code that could never fire;
        //   * its stated reason, "ScaLAPACK has no GPU path (siesta.md
        //     § 13)", is an obsolete citation.  ELPA runs on CPU *and* GPU,
        //     and ScaLAPACK here is SIESTA's built-in Divide-and-Conquer;
        //     the corrected account is in the `diag_algorithm` help;
        //   * it silently changed a value the user chose.  Reconciling a
        //     solver with the hardware belongs to `prep`, which is the layer
        //     that knows the hardware -- and which records what it changed
        //     (tuning.md § 2.11's realignment rule).
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
            "py-method", "py-optimize", "py-solvent",
            // ``p-spin-polarized`` -> ``p-spin-treatment`` (the checkbox
            // became a four-state enum) and ``p-enable-gpu`` is gone from
            // this form entirely.  Listening on a dead id is silent: the
            // listener simply never attaches, so the gate it drives stops
            // updating and nothing says so.
            "p-spin-treatment", "p-relax-type", "p-diag-algorithm",
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


    /* ---------- what is not at the recommended value ----------------
     *
     * A single "reset everything" button cannot tell a deliberate 4x4x1
     * k-grid from a value that arrived with an older session, and both
     * live in the same form.  So this LISTS the differences and resets
     * only what is ticked.
     *
     * The comparison is `formSchema.diffFromDefaults` -- the module that
     * already owns "what the DOM holds" and "what the schema says".
     * Reading the inputs here would mean a second reader for every kind
     * that module handles.
     */
    function mountRecommended(engine, container, schema) {
        const panel = $(engine + "-recommend");
        if (!panel) return;
        const listEl  = panel.querySelector(".rec-diff-list");
        const countEl = panel.querySelector(".rec-diff-count");
        const resetEl = panel.querySelector(".rec-diff-reset");
        const allEl   = panel.querySelector(".rec-diff-all");
        const fs = (window.molbuilder || {}).formSchema;
        if (!listEl || !fs || typeof fs.diffFromDefaults !== "function") return;

        function show(v) {
            if (Array.isArray(v)) return v.join(", ");
            if (v === true) return "on";
            if (v === false) return "off";
            if (v === null || v === undefined || v === "") return "(empty)";
            return String(v);
        }

        function ticked() {
            return Array.from(listEl.querySelectorAll("input:checked"))
                        .map((c) => c.value);
        }

        function refresh() {
            let diffs = [];
            try { diffs = fs.diffFromDefaults(container, schema); }
            catch (_) { return; }          // a half-rendered form is not an error
            if (!diffs.length) {
                panel.hidden = true;
                listEl.textContent = "";
                return;
            }
            panel.hidden = false;
            countEl.textContent = diffs.length === 1
                ? "1 parameter is not at its recommended value"
                : diffs.length + " parameters are not at their recommended values";
            listEl.textContent = "";
            for (const d of diffs) {
                const li = document.createElement("li");
                const lab = document.createElement("label");
                const cb = document.createElement("input");
                cb.type = "checkbox"; cb.value = d.name;
                const name = document.createElement("span");
                name.className = "rec-diff-name";
                name.textContent = d.label;
                const now = document.createElement("span");
                now.className = "rec-diff-now";
                now.textContent = show(d.current) + (d.unit ? " " + d.unit : "");
                const rec = document.createElement("span");
                rec.className = "rec-diff-rec";
                rec.textContent = show(d.recommended) + (d.unit ? " " + d.unit : "");
                lab.title = [d.name, d.help].filter(Boolean).join("\n\n");
                lab.append(cb, name, now, rec);
                li.append(lab);
                listEl.append(li);
            }
            resetEl.disabled = true;
        }

        listEl.addEventListener("change", () => {
            resetEl.disabled = ticked().length === 0;
        });

        allEl.addEventListener("click", () => {
            const boxes = listEl.querySelectorAll("input[type=checkbox]");
            const turnOn = ticked().length !== boxes.length;
            boxes.forEach((c) => { c.checked = turnOn; });
            resetEl.disabled = !turnOn;
        });

        resetEl.addEventListener("click", () => {
            const want = new Set(ticked());
            if (!want.size) return;
            let diffs = [];
            try { diffs = fs.diffFromDefaults(container, schema); }
            catch (_) { return; }
            const values = {};
            for (const d of diffs) if (want.has(d.name)) values[d.name] = d.recommended;
            fs.setValues(container, schema, values);
            refresh();
        });

        // The form is also written by the compatibility engine and by
        // session restore, so the panel follows the DOM rather than only
        // the user's typing.
        container.addEventListener("change", refresh);
        container.addEventListener("input", refresh);
        refresh();
    }

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
        // AFTER the restore, and this ordering is the whole of whether the
        // panel works: restoreFormState assigns `el.value` directly and
        // dispatches nothing, so a panel mounted before it measures a form
        // that is still at its defaults and then never hears the values
        // arrive.  Mounted here, its first reading is the real one.
        for (const [engine, hostId] of [["siesta", "siesta-form-container"],
                                       ["pyscf",  "pyscf-form-container"]]) {
            const host = $(hostId);
            if (host && formSchemas[engine]) {
                mountRecommended(engine, host, formSchemas[engine]);
            }
        }
    }

    // ----- Sidebar-driven loading (Projects sidebar -> Build) ------- //
    //
    // The Projects sidebar publishes commits via
    // ``window.molbuilder.projects.onCommit``.  On a commit of a
    // ``.xyz`` / ``.pdb`` we load it through the ONE file door
    // (``projects.parser.openMolecule(path)``), which reads the .xyz +
    // its .molstruct.json sidecar and installs the model -- so picking a
    // structure in the sidebar auto-loads it without a re-upload.
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
            /* A commit carries the file's FULL path.  A caller that passed a
             * bare name got it resolved against the server process's own
             * directory and refused as "outside every configured root" — a
             * message that names neither the caller nor which argument was
             * wrong.  The commit also carries the directory, so the one case
             * that used to fail is the one case we can simply complete. */
            let f = (sel && sel.file) ? String(sel.file) : "";
            if (f && !f.includes("/") && sel && sel.dir) {
                f = String(sel.dir).replace(/\/+$/, "") + "/" + f;
            }
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
            if (f === _sidebarLastFile) {
                // H3 2026-06-14: same file as last commit -- skip
                // the structure reload (cached) but RE-FIRE the
                // auto-detect chip refresh.  The user may have
                // edited the file on /molbuilder, saved under the
                // same path, and navigated back to this tab; the
                // structure-cache still matches the prior bytes,
                // but ``/api/structure/analyze`` reads from disk,
                // so the chip CAN refresh to reflect the on-disk
                // change.  Pre-fix the bare early-return left the
                // chip showing the verdict from before the edit
                // (e.g. "closed-shell singlet" after the user just
                // removed the metal atom).  Cheap fix: just re-
                // fire the analyzer — the per-tab structure cache
                // stays correct because the on-disk file might or
                // might not have changed; the chip is the user-
                // visible surface that needs to stay honest.
                if (typeof _autoAnalyzeOnLoad === "function") {
                    _autoAnalyzeOnLoad(_sidebarLastFile);
                }
                return;
            }
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
            // ONE load door (structure-load-save-contract.md): read the .xyz + its
            // .molstruct.json sidecar via the concealed projects parser and install the
            // MODEL -- labels/regions/frozen ride along, so the read-only viewer shows
            // them.  This replaces the old SECOND load path (hand-rolled /api/files/read
            // + /api/build/load with NO sidecar + a separate sidecarLabels.fetch), which
            // dropped the labels the Molbuilder tab shows -- the exact inconsistency.
            const _proj2 = window.molbuilder && window.molbuilder.projects;
            if (!_proj2 || !_proj2.parser
                    || typeof _proj2.parser.openMolecule !== "function") {
                setStatus("load-status",
                    "Projects file package unavailable "
                    + "(structure-load-save-contract).", "error");
                clearStructureInfo("load failed: " + filename);
                return;
            }
            const _viewer = await _ensureMounted();
            if (!_viewer) {
                setStatus("load-status",
                    "Viewer unavailable: nothing to load the structure into.",
                    "error");
                clearStructureInfo("load failed: " + filename);
                return;
            }
            let res;
            try {
                res = await _proj2.parser.openMolecule(_viewer, f);
            } catch (e) {
                if (mySeq !== _sidebarLoadSeq) return;
                setStatus("load-status", _formatFetchError(e), "error");
                clearStructureInfo("load failed: " + filename);
                return;
            }
            if (mySeq !== _sidebarLoadSeq) return;     // superseded
            if (!res || res.ok === false) {
                setStatus("load-status",
                    (res && res.error) || "Load failed.", "error");
                clearStructureInfo("load failed: " + filename);
                return;
            }
            // Populate page state + info panel FROM THE MODEL (single source of truth) +
            // mount the read-only card.  The Generate/preflight POST + info panel now
            // read the model, not a parallel hand-rolled response.
            _applyLoadedModel(filename);
            /* KEEP IT, so coming back to this tab shows what you were working
             * on.  Saved HERE, at the one moment the tab acquires a structure,
             * rather than on a timer or on the way out: `pagehide` does not
             * fire reliably on a mobile tab switch, and this page has exactly
             * one event worth recording. */
            _rememberStructure();
            const _d = _data();
            const _atoms = _d ? (_d.getStructure() || {}).elements : null;
            setStatus("load-status",
                `Loaded ${(Array.isArray(_atoms) ? _atoms.length : "")}-atom `
                + `${ext.toUpperCase()} from ${filename}.`, "ok");
            // Flip the load-bar readout to "Loaded: <name>", refresh the auto-detect
            // button, and auto-fire the analyzer (chemistry rationale visible by default;
            // see scientific-validation.md § 2.5).  All hoisted, defined below in scope.
            if (typeof _refreshLoadButton === "function") _refreshLoadButton();
            if (typeof _refreshAutoDetectButton === "function") _refreshAutoDetectButton();
            if (typeof _autoAnalyzeOnLoad === "function") _autoAnalyzeOnLoad(_sidebarLastFile);
        }   // close _commitStructure

        // Universal commit subscription — dblclick on a structure
        // file in the sidebar.  Falls back to onChange for older
        // deployments without onCommit so the tab stays usable.
        const subscribe = (typeof _proj.onCommit === "function")
            ? _proj.onCommit.bind(_proj)
            : _proj.onChange.bind(_proj);
        subscribe(_commitStructure);

        /* WHAT HAPPENS WHEN YOU COME BACK TO THIS TAB.
         *
         * The molecule you had, and the parameters you typed.  The parameters
         * are handled at the end of this file (form state); this is the
         * molecule.
         *
         * THE TAB SAVES IT, NOT THE VIEWER.  This viewer is read-only, which
         * settles two things and neither of them is "the tab remembers
         * nothing": its timeline does nothing -- no saved points, no Retract --
         * and `installMolecule` is allowed only into a viewer holding nothing,
         * or when the caller says `enforce`.  Both of those are about the
         * SEQUENCE and about EDITING.  What this tab put on screen is the tab's
         * own fact, kept under the tab's own tag beside its form values
         * (workspace.md § 4: several savers on one page, each deciding what and
         * when).
         *
         * TWO DEAD RESTORES STOOD HERE BEFORE, and both looked convincing.  The
         * first read the saved bytes back out of the workspace and opened them
         * (`readPersistedSnapshot`, a door the workspace no longer has, behind
         * a `typeof` guard, so it answered "nothing saved" every visit).  The
         * second called `data.load(0)`, copied from the Modify tab -- which is
         * EDITABLE, and where that call IS the restore.  Here it is a gate that
         * returns null.  Neither was measured against this viewer's mode.
         *
         * THE VIEWER IS MOUNTED FIRST, because a structure needs somewhere to
         * go.  That was a real defect on its own: this block used to run
         * against whatever `_data()` happened to answer, and on a fresh page
         * that was null -- nothing had mounted yet, so the restore branch could
         * never be taken and the sidebar seeded over the session every time.
         *
         * THE SIDEBAR ONLY SEEDS AN EMPTY CANVAS.  A pick left highlighted from
         * last time is not an instruction to load it: loading a file is the
         * Load button or a double-click, never a side effect of arriving.  What
         * is asked is "did anything come back?", never "does the sidebar's file
         * match" -- a generated structure has atoms and no file at all, and the
         * file comparison read that as empty and wiped it.
         */
        const _initialFile = (typeof _proj.getCurrentFile === "function")
            ? _proj.getCurrentFile() : "";
        const _restored = (async function _restoreThenSeed() {
            await _ensureMounted();
            const saved = await _restoreStructure();
            if (saved) {
                /* The file it came from is the TAB's note, saved beside the
                 * structure -- the viewer tracks contents, not files
                 * (molview.md § 6.7), so it does not come back with the atoms. */
                _sidebarLastFile = saved.loadedFrom || "";
                const _name = _sidebarLastFile
                    ? _basename(_sidebarLastFile) : "restored structure";
                _applyLoadedModel(_name);
                /* AND THE PAGE FURNITURE, or the tab contradicts itself: the
                 * molecule is back on the canvas while the load bar still reads
                 * "Selected: x.xyz" with the Load button live, as though nothing
                 * had been loaded.  Same refreshers a fresh load runs -- there
                 * is one way for this page to say "a structure is loaded", and
                 * a restore has to use it rather than half of it. */
                setStatus("load-status",
                    `Restored ${_name} — ${saved.structure.elements.length} atoms.`,
                    "ok");
                if (typeof _refreshLoadButton === "function") _refreshLoadButton();
                if (typeof _refreshAutoDetectButton === "function") _refreshAutoDetectButton();
                return true;
            }
            if (!_initialFile) return false;
            // Empty canvas + a sidebar pick: the genuine first arrival, or a
            // handoff from another tab.
            const _initialDir = (typeof _proj.getCurrentDir === "function")
                ? _proj.getCurrentDir() : "";
            _commitStructure({ dir: _initialDir, file: _initialFile });
            return false;
        })().catch(function (e) {
            /* NOBODY IS WAITING ON THIS PROMISE, so a throw would land nowhere.
             * This is the first thing the page does; failing at it in silence
             * leaves an empty canvas that reads as "there was nothing here". */
            setStatus("load-status",
                "Could not restore this tab's structure: "
                + ((e && e.message) || "unknown error")
                + ". Pick a file and press Load.", "error");
            return false;
        });

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
        // "Is there work here that is not on the sequence yet" is `uncommitted`,
        // a value the viewer holds rather than a question asked of it. It is
        // raised inside the gate, after a change lands, so nothing outside marks
        // it (molview.md § 11.2). Read lazily: there is no viewer until the first
        // load mounts one.
        const _modelDirty = () => {
            const d = _data();
            return !!(d && d.uncommitted);
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
                    readout.textContent = _modelDirty()
                        ? `Loaded: ${_basename(_candidatePath)} · unsaved changes`
                        : `Loaded: ${_basename(_candidatePath)}`;
                } else if (loadable) {
                    readout.textContent =
                        `Selected: ${_basename(_candidatePath)}`;
                } else if (_candidatePath) {
                    // SAY WHY, not just that.  The reason used to exist only
                    // on the commit path (double-click), so a single click
                    // left "not loadable" standing alone with no way to find
                    // out what would be loadable.
                    readout.textContent =
                        `Selected: ${_basename(_candidatePath)} `
                        + `(not loadable — .xyz / .pdb only)`;
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
        // Re-render the readout whenever the model's dirty state flips.
        // Subscribe defensively — molview.data may not be mounted on pages
        // that don't mount the canvas (e.g. spectra-only views).
        (function _subscribeModelDirty() {
            const d = _data();
            if (d && typeof d.subscribe === "function") {
                d.subscribe(_refreshLoadButton);
            }
        })();
        const _loadBtn = $("load-from-sidebar-btn");
        if (_loadBtn) {
            _loadBtn.addEventListener("click", () => {
                if (!_isLoadable(_candidatePath)) return;
                const _dir = (typeof _proj.getCurrentDir === "function")
                    ? _proj.getCurrentDir() : "";
                // Explicit Load = "load the current file NOW".  Always re-read
                // from disk, even if it is the same path the tab already holds
                // (restored on mount, or loaded earlier this session): the file
                // may have changed on disk and the user asked for a fresh load.
                // Clear the same-file dedup guard so _commitStructure does not
                // short-circuit.  (The auto/subscribe path keeps the dedup.)
                _sidebarLastFile = "";
                _commitStructure({ dir: _dir, file: _candidatePath });
            });
        }

        // -------- Auto-detect button (Phase 2 of the chemistry
        // middle-layer work; see
        // docs/science/validation.md).
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
        // J3 2026-06-14: shared AbortController so spam-clicks (or a
        // mount-time auto-fire racing a manual click) cancel the
        // prior request instead of letting it land on the server.
        // The ``_autoDetectSeq`` gate above prevents stale responses
        // from updating the DOM, but the request still completes
        // server-side -- wasted CPU + bytes on every spam-click.
        // The AbortController kills the in-flight fetch cleanly so
        // the server stops parsing on supersede.
        let _autoDetectAbort = null;

        const _autoBtn = $("auto-detect-btn");
        if (_autoBtn) {
            _autoBtn.addEventListener("click", async () => {
                if (!_sidebarLastFile) return;
                // Snapshot BOTH the path AND the load-seq so a
                // mid-flight structure swap is detectable.
                const mySeq      = ++_autoDetectSeq;
                const myLoadSeq  = _sidebarLoadSeq;
                const myPath     = _sidebarLastFile;
                // J3: abort any prior analyze request still on the
                // wire (manual click + background fire share the
                // same controller so either supersedes the other).
                if (_autoDetectAbort) _autoDetectAbort.abort();
                _autoDetectAbort = new AbortController();
                const mySignal = _autoDetectAbort.signal;
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
                        signal: mySignal,
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
                    // AbortError = superseded by another click /
                    // load fire.  Silent; the new request owns
                    // the UI now.
                    if (e && e.name === "AbortError") return;
                    if (mySeq !== _autoDetectSeq
                        || myLoadSeq !== _sidebarLoadSeq) return;
                    setStatus("auto-detect-status",
                        _formatFetchError(e), "error");
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
            // J3: abort any prior in-flight analyze.  Mount-time
            // auto-fires can race the user's manual click on the
            // SAME path (H3 same-file re-fire path); aborting the
            // first lets the second take ownership cleanly.
            if (_autoDetectAbort) _autoDetectAbort.abort();
            _autoDetectAbort = new AbortController();
            const mySignal = _autoDetectAbort.signal;
            try {
                const r = await fetch("/api/structure/analyze", {
                    method:  "POST",
                    headers: { "Content-Type": "application/json" },
                    body:    JSON.stringify({ structure_path: path }),
                    signal:  mySignal,
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
                // Includes AbortError from supersede; nothing to do.
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
        // docs/web/ui-contract.md Rule 1.
        const _detectionChip = (window.molbuilder
                                && window.molbuilder.detectionChip)
            || { buildText: () => ({ profile: "", budget: "" }),
                 render: () => 0 };
        const _buildDetectionChipText = _detectionChip.buildText;
        const _renderWorkflowGroupChips = _detectionChip.render;
        });   // close runtime.whenReady("projects").then(...)
    }

    // formula() is imported from MolView's door (mvFormula, top of file).  Imported,
    // so it is always the real function (no global-read, no load-order fallback,
    // no "OHH" full-expansion bug).
    const formula = mvFormula;

    // ----- 2. Render --------------------------------------------------
    // The standard knob bar (style picker / labels popover / axes /
    // reset / background / export) owns all the per-viewer chrome;
    // the Optimization tab's host page only owns its tab-specific
    // controls (Load-from-sidebar button, Generate buttons,
    // Download row).  The bespoke <details> Style block that used
    // to live in index.html is gone; rep / radius / background /
    // labels are reached via the knob bar so every consumer site
    // shares the same UX.

    // The model is loaded by the ONE door (projects.parser.openMolecule); this ONLY
    // ensures the read-only MolView card is mounted on first load.  The mounted render
    // reacts to the model on its own -- we must NOT re-install bare text here (that was
    // the old second load path, and it would drop the sidecar labels the door loaded).
    /* AWAITABLE, because the load door needs a viewer to put the file into and
     * this is what makes one. A viewer mounts before it has a structure
     * (molview.md § 8), so this runs first and the load follows.
     *
     * It used to be fire-and-forget, called AFTER the load — which worked only
     * while the load door could find a viewer by name in a global. */
    async function _ensureMounted() {
        const ws = _ws();
        // NOT gated on a viewer existing: this IS what creates it, and testing
        // for one first is what stopped three pages mounting at all.
        if (!ws) return null;
        if (_mvHandle) return _mvHandle;
        return mvMount($("viewer-host"), ws,
                { mode: "readonly", owner: WORKSPACE_TAG })
            .then(function (h) { _mvHandle = (h && h.ok) ? h : null; return _mvHandle; })
            .catch(function (e) {
                // #load-status is the page's real load/viewer status slot (there is
                // no #status element -- the old id was a phantom that made this
                // handler throw into its own catch, hiding every mount failure).
                // "error", not "err": the severity is a CSS class
                // (.status.ok/.error/.warn/.muted, page-shell.css), and "err"
                // matches none of them -- so the one message that says the
                // viewer is not there was drawn in ordinary body text.
                setStatus("load-status",
                    "Viewer failed to mount: " + ((e && e.message) || "render failed"),
                    "error");
            });
    }

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
    // KNOWN GAP (audit 2026-07): polling is the anti-pattern -- form-schema
    // exposes no render-complete callback today, so this is the documented
    // interim.  If form-schema grows an onRendered seam, replace this loop.
    (function pollForField(tries) {
        if (tries <= 0) return;
        if ($("p-psml-lib")) {
            installPsmlLibLiveCaption();
        } else {
            setTimeout(() => pollForField(tries - 1), 150);
        }
    }(40));   // ~6s budget

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
        // `continue_retries` needs NO lifting: it is an ordinary SiestaConfig
        // field (section "Compute & budget"), so `collectForm` above already
        // returned it, and `engines/stages.md § 3` is explicit that it is "an
        // ordinary shared field; what made it look special is only where it
        // lands".
        //
        // A block here used to read `params.stages[<seq>].on_nonconvergence`
        // and copy the retry budget up.  Both halves of that are gone:
        // `SiestaConfig` has had no `stages` field since P2 deleted
        // `SiestaStageSpec`, so `params.stages` was `undefined` and the gate
        // could never fire; and `on_nonconvergence` left the SIESTA producer
        // with the inter-stage edges (P7 unit 2).  Deleting it is what makes
        // the retry budget actually arrive -- it was being gated on a lookup
        // that always failed.
        return params;
    }

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
    ];

    function getFormIds() {
        const ids = STATIC_FORM_IDS.slice();
        for (const sch of [formSchemas.siesta, formSchemas.pyscf]) {
            if (!sch) continue;
            for (const sect of sch.sections) {
                for (const f of sect.fields) {
                    if (f.kind === "int-triple"
                            || f.kind === "float-triple") {
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

    // Wire each engine-scoped form input to the debounced preflight
    // refresh so the issues panel updates live as the user adjusts
    // settings.  p-* IDs feed SIESTA's panel; py-* IDs feed PySCF's.
    // No-op until the user has built a structure (the model is non-empty) --
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
            const event = (el.type === "checkbox" || el.tagName === "SELECT")
                ? "change" : "input";
            el.addEventListener(event, () => refreshPreflightDebounced[which]());
        });
    }
    /* ---------------------------------------------------------------- *
     *  Send to Task setup — the hand-off                                *
     * ---------------------------------------------------------------- *
     * This tab collects parameters and produces no artifact, so until this
     * existed the form's work had nowhere to go at all.  The button writes
     * two files into the folder the projects sidebar has selected, then
     * opens Task setup there.
     *
     * THROUGH DISK, NEVER IN MEMORY.  `web/tabs.md` § 1 forbids an in-memory
     * "send to tab" hand-off and lists four costs: a result depending on
     * hidden state, re-running silently producing something different, two
     * people getting different answers, export losing information.  Writing
     * the files first keeps all four bought.
     *
     * It lives HERE rather than in its own module because the two things it
     * needs -- `_structureForRequest()` and the collect*Params() pair -- are
     * private to this file.  A separate module would have to re-derive them,
     * which is the duplication this codebase keeps paying for.
     */
    function _handoverSay(kind, text) {
        const n = document.getElementById("handover-status");
        if (!n) return;
        n.hidden = false;
        n.className = "status " + kind;      // page-shell owns the severities
        n.textContent = text;
    }

    async function _sendToTaskSetup() {
        const projects = window.molbuilder && window.molbuilder.projects;
        const dest = (projects && typeof projects.getCurrentDir === "function")
            ? projects.getCurrentDir() : "";
        if (!dest) {
            _handoverSay("warn", "Pick the folder to write into, in the "
                + "sidebar first — this button writes into the selected "
                + "folder and creates none of its own.");
            return;
        }
        const structure = _structureForRequest();
        if (!structure) {
            _handoverSay("warn", "Load a structure first — a description is "
                + "of something.");
            return;
        }
        /* A CALCULATION LIVES UNDER A TOPIC (`job-contracts.md` § 2.5): the
         * tree is project / topic / calculation, and the three levels above
         * the calculation are organisational.  `safeSave` refuses only at the
         * projects ROOT, so without this a Send into a bare project or topic
         * folder would put a calculation where one may not live.
         *
         * Measured against the projects root rather than by counting slashes,
         * because the root is configurable — `relativeToProjects` is the
         * sidebar's own answer to "where am I". */
        const rel = (typeof projects.relativeToProjects === "function")
            ? String(projects.relativeToProjects(dest) || "") : "";
        const depth = rel.split("/").filter(Boolean).length;
        if (depth < 3) {
            _handoverSay("warn",
                "That folder is too high in the tree — a calculation lives "
                + "under a topic (projects / project / topic / your folder). "
                + "Pick or make one further down in the sidebar.");
            return;
        }

        /* ONE JOB PER FOLDER (`job-contracts.md` § 2.1 Rule 1).  Send writes
         * with `overwrite: true`, so without this it would silently replace
         * another calculation's template.  The check goes through the file
         * layer like every other read -- `missingOk` makes "no description
         * here" a 200 rather than a logged 404, which is the normal case.
         *
         * This guard EXISTED as a 409 in the hand-over endpoint and was lost
         * on 2026-08-16 when that endpoint stopped resolving the destination
         * to become render-only.  Restored on the side that chooses where to
         * write, which is where it belongs. */
        try {
            const prior = await projects.readFile(dest + "/task.json",
                                                  { missingOk: true });
            if (prior && prior.ok !== false && prior.exists !== false
                && typeof prior.text === "string" && prior.text.trim()) {
                _handoverSay("error",
                    "That folder already holds a task.json — it is a described "
                    + "calculation. Sending here would overwrite it. Pick or "
                    + "make another folder in the sidebar; one job per folder.");
                return;
            }
        } catch (e) {
            _handoverSay("error", "Could not check the folder: "
                + (e && e.message ? e.message : e));
            return;
        }

        const engine = _activeEngine();
        const params = (engine === "siesta")
            ? collectFdfParams() : collectPyscfParams();

        _handoverSay("muted", "Rendering…");
        let out;
        try {
            const r = await fetch("/api/task-setup/handover", {
                method:  "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    structure, engine, params,
                    name: (dest.split("/").filter(Boolean).pop() || ""),
                    // No `structure_path`.  It used to send the projects
                    // sidebar's selected file, which is a SECOND fact read at a
                    // second moment -- `molview.md` § 9.3a's rule is that the
                    // facts which leave together were read together, and this
                    // one was read from a cursor rather than from the
                    // structure.  The server names the files it writes from the
                    // structure that is right here in this body.
                }),
            });
            out = await r.json();
            if (!r.ok || !out || out.ok === false) {
                _handoverSay("error",
                    (out && out.error) || ("render failed (" + r.status + ")"));
                return;
            }
        } catch (e) {
            _handoverSay("error", "Could not reach the server: "
                + (e && e.message ? e.message : e));
            return;
        }

        /* The BYTES go through the content-blind file layer, never a fetch of
         * our own (`web/projects.md` § 1).  `safeSave` writes into the folder
         * the user selected, refuses at the projects root, re-lists the
         * sidebar afterwards, and returns the four-way shape below -- the
         * canonical caller pattern, as `lib/spectra/core.js` uses it. */
        _handoverSay("muted", "Writing…");
        /* The STRUCTURE goes first, and the order is deliberate: the hand-over
         * NAMES those files, so writing it before them would leave a folder
         * whose description points at nothing if the next write fails. */
        const parts = (out.structure_files || []).map((f) => [f.name, f.text]);
        parts.push([out.template_name, out.template_text],
                   [out.handover_name, out.handover_text]);
        for (const [name, text] of parts) {
            // safeSave(TEXT, FILENAME, opts) -- text first.
            const wrote = await projects.safeSave(text, name, { overwrite: true })
                .catch((e) => ({ ok: false, error: String(e && e.message || e) }));
            if (wrote && wrote.cancelled) {
                _handoverSay("muted", "Cancelled — nothing written.");
                return;
            }
            if (!wrote || !wrote.ok) {
                _handoverSay("error", "Could not write " + name + ": "
                    + ((wrote && wrote.error) || "no folder selected"));
                return;
            }
        }
        const body = { files: parts.map(([n]) => n) };
        _handoverSay("ok", "Wrote " + (body.files || []).join(" and ")
            + " — opening Task setup…");
        // The sidebar's selection is shared through sessionStorage, so Task
        // setup opens on the same folder without this handing it anything.
        window.location.href = "/task-setup";
    }

    /** Which engine's form is showing — the one the user has been filling. */
    function _activeEngine() {
        const pyscf = document.getElementById("tab-pyscf");
        const showing = pyscf && !pyscf.hidden
            && getComputedStyle(pyscf).display !== "none";
        return showing ? "pyscf" : "siesta";
    }

    {
        const _btn = document.getElementById("send-to-task-setup");
        if (_btn) _btn.addEventListener("click", _sendToTaskSetup);
    }
})();
