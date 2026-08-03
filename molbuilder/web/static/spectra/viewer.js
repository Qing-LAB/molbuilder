/* /spectra page bootstrap.
 *
 * Two responsibilities, both /spectra-specific (the generic
 * /results flow already lives in lib/spectra/core.js + the
 * spectra inspector adapter):
 *
 *   1. Mount the spectra-inspector core against ``document`` so
 *      the schema form + Generate / Methods / Issues / script-
 *      preview / Save handlers wire up.  This is the unchanged
 *      pre-task-#296 behaviour.
 *
 *   2. Wire the Inspect-structure card (task #296, 2026-06-09):
 *      mount a 3Dmol embed in ``#viewer-wrap`` and hook the
 *      ``#load-from-sidebar-btn`` so the user can pick a
 *      structure file in the Projects sidebar and click Load to
 *      see it in the viewer.  The spectra inspector's Generate POST
 *      reads the structure OFF THE MODEL at send time
 *      (core.js reads the structure off the viewer this page mounted)
 *      -- no push, no in-memory holder (the old setStructureText
 *      seam + the pre-#309 hidden ``<textarea id="structure-text">``
 *      are both gone).
 *
 * Mirrors the Optimization tab's pattern in static/viewer.js
 * (the Load button, the info readout, the sidebar onCommit
 * subscription).  Where the two pages diverge today:
 *   * Optimization carries a Build form + Generate buttons +
 *     SIESTA/PySCF schema; spectra carries the spectra schema
 *     and its own Generate/Methods/script-preview machinery in
 *     core.js.
 *   * Optimization's commit also POSTs /api/build/load to get
 *     the canonical workspace payload (atom list, residue ids,
 *     etc.) so the SIESTA validators have it.  Spectra doesn't
 *     consume that payload today; we only need the raw bytes
 *     for the form's structure_text field.  Skipping the build/
 *     load roundtrip keeps the load fast.
 */
import { mount, formula as mvFormula } from "/static/lib/molview/index.js";

// Who this tab's saved work belongs to (workspace.md § 4): the viewer's `owner`
// and the tag on any workspace call, so the two cannot name different slots.
const WORKSPACE_TAG = "spectra";
(function () {
    "use strict";

    function _$(id) { return document.getElementById(id); }

    function setStatus(elId, msg, kind) {
        const el = _$(elId);
        if (!el) return;
        el.textContent = msg || "";
        el.className = "status"
            + (kind === "ok"    ? " ok"    : "")
            + (kind === "warn"  ? " warn"  : "")
            + (kind === "error" ? " error" : "");
    }

    /* THE INSPECTOR THIS PAGE MOUNTED, kept so the page can hand it the viewer
     * it also mounted.  One page, two things it owns, and it introduces them --
     * rather than either of them looking the other up. */
    let _inspector = null;

    function _bootstrapSpectraCore() {
        // Defensive: if core.js failed to load (e.g. a CDN-block in
        // some weird CSP edge case), fail loudly in the console
        // rather than silently rendering an empty <main>.
        const api = (window.molbuilder || {}).spectraInspector;
        if (!api || typeof api.mount !== "function") {
            console.error(
                "[spectra] spectraInspector.mount not found.  "
                + "Check that lib/spectra/core.js is loaded BEFORE "
                + "spectra/viewer.js (the script tag order in "
                + "spectra.html is load-bearing)."
            );
            return;
        }
        // Mount with ``document`` as the root so $() lookups find
        // /spectra's generate-side form ids (which live OUTSIDE any
        // partial; full-page mount).
        _inspector = api.mount(document);
    }

    /**
     * Update the static info readout below the viewer header.  Title shows the
     * basename of the loaded file; atom count + formula are READ FROM THE MODEL
     * (molview.data.getElements -- the structure the Load just installed), never
     * re-parsed from text: MolView already parsed it, and the inspector rule is
     * "no structure parsing in a consumer" (lib/inspectors/structure.js).
     * Runs synchronously after every successful Load (the model is settled).
     */
    /* THE VIEWER IS HANDED IN, because this function does not have one.
     *
     * It read `mvHandle` off a `let` declared inside the bootstrap below --
     * a name that does not exist in this scope -- so every call threw
     * `ReferenceError: mvHandle is not defined` and the readout was never
     * written. The throw happened inside the load path, after the structure was
     * already on screen, so the page looked like it had loaded a molecule and
     * then simply refused to say what it was. */
    function _updateInfo(viewer, filename) {
        const title   = _$("info-title");
        const atomsEl = _$("info-atoms");
        const formula = _$("info-formula");
        const elements = (viewer && viewer.ok
            && viewer.data.getElements()) || [];
        if (!elements.length) {
            if (title)   title.textContent   = "no structure loaded";
            if (atomsEl) atomsEl.textContent = "—";
            if (formula) formula.textContent = "—";
            return;
        }
        if (title)   title.textContent   = filename || "loaded";
        if (atomsEl) atomsEl.textContent = String(elements.length);
        // The Hill formula() is imported from MolView's door (mvFormula = mol-format.js).  `formula`
        // here is the DOM readout node; mvFormula is the formatter function.
        if (formula) formula.textContent = mvFormula(elements);
    }

    /**
     * Bootstrap the Inspect-structure card: mount the read-only MolView
     * component (lazily, on first load), wire the Load button, and subscribe to
     * ``onCommit`` so a sidebar dblclick on a .xyz/.pdb loads it via
     * ``projects.parser.openMolecule``.
     */
    function _bootstrapInspectCard() {
        const host = _$("spectra-molview-host");
        if (!host) return;
        // Read-only MolView — the SAME concealed component Modify/Transport mount,
        // in mode:"readonly" for structure demonstration only (view toggles +
        // selection/cell panel, no editing).  Mounted lazily on the first load.
        const ws   = window.molbuilder && window.molbuilder.workspace;
        const proj = window.molbuilder && window.molbuilder.projects;
        // NOT gated on a viewer: there is none until a load mounts one, and
        // testing for one here is what stopped this page mounting at all.
        if (!ws || typeof mount !== "function"
                || !proj || !proj.parser
                || typeof proj.parser.openMolecule !== "function") {
            setStatus("load-status",
                "Viewer unavailable: the MolView / projects module failed to load "
                + "(check the template script tags).", "error");
            return;
        }
        let mvHandle = null;   // mounted on first structure load

        let _sidebarLastFile = "";
        let _loadSeq         = 0;

        // ---- Load button + sidebar onCommit subscription -------- //
        const _isLoadable = (name) => {
            const n = String(name || "").toLowerCase();
            return n.endsWith(".xyz") || n.endsWith(".pdb");
        };
        const _basename = (p) => {
            const ix = String(p || "").lastIndexOf("/");
            return ix >= 0 ? p.slice(ix + 1) : p;
        };

        let _candidatePath = "";
        function _refreshLoadButton() {
            const btn = _$("load-from-sidebar-btn");
            const readout = _$("load-source-readout");
            if (!btn) return;
            const loadable = _isLoadable(_candidatePath);
            btn.disabled = !loadable;
            if (readout) {
                const isLoaded = loadable
                    && _sidebarLastFile === _candidatePath;
                readout.textContent = isLoaded
                    ? `Loaded: ${_basename(_candidatePath)}`
                    : loadable
                        ? `Selected: ${_basename(_candidatePath)}`
                        : (_candidatePath
                            ? `Selected: ${_basename(_candidatePath)} (not loadable)`
                            : "Pick a .xyz / .pdb in the Projects sidebar.");
            }
        }

        async function _commitStructure(sel) {
            const f = (sel && sel.file) ? String(sel.file) : "";
            const ext = f.toLowerCase().split(".").pop();
            if (ext !== "xyz" && ext !== "pdb") {
                if (f) {
                    setStatus("load-status",
                        `${_basename(f)} is not a structure file `
                        + `(.xyz / .pdb only).`, "warn");
                }
                return;
            }
            if (f === _sidebarLastFile) return;
            const mySeq = ++_loadSeq;
            setStatus("load-status",
                `Loading ${_basename(f)}…`, null);
            try {
                // THE VIEWER FIRST, THEN THE FILE: a viewer mounts before it has
                // a structure (molview.md § 8), and the load door needs somewhere
                // to put what it reads. This ran the other way round, which only
                // worked while the door could find a viewer in a global.
                if (!mvHandle || !mvHandle.ok) {
                    // Cache ONLY a live handle (mount contract: failure ->
                    // {ok:false}); a failed mount must not stick, so the next
                    // structure load retries instead of staying viewer-less.
                    const _h = await mount(host, ws,
                        { mode: "readonly", owner: WORKSPACE_TAG });
                    mvHandle = (_h && _h.ok) ? _h : null;
                    if (!mvHandle) throw new Error("the viewer could not be built");
                    /* The page mounted the viewer, so the page hands it on:
                     * the Generate panel needs this same one.
                     *
                     * Handed to THE INSPECTOR THIS PAGE MOUNTED, not to a
                     * module-wide door. The viewer belongs to whoever mounted
                     * it (molview.md § 5.6), and so does the inspector holding
                     * it -- a module-level setter would be one viewer for every
                     * mount on the page. */
                    if (_inspector && typeof _inspector.useViewer === "function") {
                        _inspector.useViewer(mvHandle);
                    }
                }
                // The format-aware sidebar door reads the .xyz + its
                // .molstruct.json and installs both into THIS viewer in one write.
                const r = await proj.parser.openMolecule(mvHandle, f);
                if (r && r.ok === false) {
                    throw new Error(r.error || ("Could not load " + f));
                }
            } catch (e) {
                setStatus("load-status",
                    "Load failed: " + (e && e.message ? e.message : e), "error");
                return;
            }
            if (mySeq !== _loadSeq) return;  // superseded by a newer load
            _sidebarLastFile = f;
            // (Nothing else to push into spectra/core.js: it was handed the
            // viewer at mount and reads the structure off it at send time --
            // so there is no second in-memory copy to feed or drift.)
            _updateInfo(mvHandle, _basename(f));
            setStatus("load-status",
                `Loaded ${_basename(f)}.`, "ok");
            _refreshLoadButton();
            // Phase 3 (2026-06-10): auto-fire the analyzer so the
            // chemistry rationale is visible by default, not gated
            // behind the Auto-detect button.  Forms are NOT pre-
            // filled — the button still owns that explicit step.
            // See scientific-validation.md § 2.5; same hook on
            // /structure-optimization in static/viewer.js.
            if (typeof _autoAnalyzeOnLoad === "function") {
                _autoAnalyzeOnLoad(f);
            }
        }

        // Sidebar onChange / onCommit subscription + initial
        // candidate-path tracking.  Mirrors the Optimization
        // tab's pattern in static/viewer.js.
        const rt = (window.molbuilder || {}).runtime;
        const projP = (rt && typeof rt.whenReady === "function")
            ? rt.whenReady("projects")
            : Promise.resolve((window.molbuilder || {}).projects);
        projP.then((proj) => {
            if (!proj) return;
            // Initial mount-time auto-load (cross-tab handoff via
            // sessionStorage.molbuilder.current_file).
            const initialFile = (typeof proj.getCurrentFile === "function")
                ? proj.getCurrentFile() : "";
            if (initialFile) {
                _candidatePath = initialFile;
                _refreshLoadButton();
                _commitStructure({ file: initialFile });
            } else {
                _refreshLoadButton();
            }
            // Subscribe to sidebar changes for the candidate-path
            // readout (single-click → "Selected: foo.xyz" hint).
            if (typeof proj.onChange === "function") {
                proj.onChange((sel) => {
                    _candidatePath = (sel && sel.file) ? sel.file : "";
                    _refreshLoadButton();
                });
            }
            // Dblclick commits the file for loading (universal
            // interaction model — task #301 same channel).
            const subscribe = (typeof proj.onCommit === "function")
                ? proj.onCommit.bind(proj)
                : proj.onChange.bind(proj);
            subscribe(_commitStructure);
        });

        // Explicit Load button click → load whatever the sidebar
        // currently highlights.
        const loadBtn = _$("load-from-sidebar-btn");
        if (loadBtn) {
            loadBtn.addEventListener("click", () => {
                if (!_isLoadable(_candidatePath)) return;
                _commitStructure({ file: _candidatePath });
            });
        }

        // -------- Auto-detect chemistry (Card 2 of the post-2026-06-10
        // vertical workflow on /spectrum-calculation; matching the
        // Optimization tab pattern from static/viewer.js).
        //
        // POST /api/structure/analyze with the currently-loaded
        // structure path, then apply the PySCF adapter's
        // (charge, spin, method) translation onto the spectra form
        // (SpectraConfig has the same three field names as
        // PySCFConfig, so the wire shape matches 1:1).
        //
        // Why a separate handler from /structure-optimization's:
        // single form to fill (not SIESTA + PySCF), no SIESTA
        // adapter needed.  Concurrency safety mirrors the
        // _loadSeq pattern used by _commitStructure above.
        function _refreshAutoDetectButton() {
            const btn = _$("auto-detect-btn");
            if (!btn) return;
            btn.disabled = !_sidebarLastFile;
        }
        let _autoDetectSeq = 0;
        // J3 2026-06-14: shared AbortController across the manual
        // click + background _autoAnalyzeOnLoad fires.  See
        // viewer.js (structure-opt) for the same pattern.
        let _autoDetectAbort = null;
        const _autoBtn = _$("auto-detect-btn");
        if (_autoBtn) {
            _autoBtn.addEventListener("click", async () => {
                if (!_sidebarLastFile) return;
                const mySeq     = ++_autoDetectSeq;
                const myLoadSeq = _loadSeq;
                const myPath    = _sidebarLastFile;
                if (_autoDetectAbort) _autoDetectAbort.abort();
                _autoDetectAbort = new AbortController();
                const mySignal = _autoDetectAbort.signal;
                _autoBtn.disabled = true;
                setStatus("auto-detect-status", "Analyzing…", null);
                let body;
                try {
                    const r = await fetch("/api/structure/analyze", {
                        method:  "POST",
                        headers: { "Content-Type": "application/json" },
                        body:    JSON.stringify({ structure_path: myPath }),
                        signal:  mySignal,
                    });
                    body = await r.json();
                    if (mySeq !== _autoDetectSeq
                        || myLoadSeq !== _loadSeq) return;
                    if (!r.ok || !body.ok) {
                        setStatus("auto-detect-status",
                            body && body.error
                                ? body.error
                                : `Analyze failed (HTTP ${r.status}).`,
                            "error");
                        return;
                    }
                } catch (e) {
                    if (e && e.name === "AbortError") return;
                    if (mySeq !== _autoDetectSeq
                        || myLoadSeq !== _loadSeq) return;
                    setStatus("auto-detect-status",
                        "Network error: "
                        + (e && e.message ? e.message : String(e)),
                        "error");
                    return;
                } finally {
                    if (mySeq === _autoDetectSeq
                        && myLoadSeq === _loadSeq) {
                        _refreshAutoDetectButton();
                    }
                }
                await _applyAutoDetectToSpectraForm(body);
                _renderAutoDetectPanel(body);
                setStatus("auto-detect-status",
                    "Applied to the parameter form.  Review rationale below.",
                    "ok");
            });
        }

        /**
         * Phase 3 auto-analyze (fired from _commitStructure on
         * every successful load).  Same shape as the Optimization
         * tab's helper in static/viewer.js: hits the analyzer,
         * renders the rationale panel, does NOT touch the
         * parameter form (the explicit button click still owns
         * the form-fill).
         */
        async function _autoAnalyzeOnLoad(path) {
            if (!path) return;
            const mySeq     = ++_autoDetectSeq;
            const myLoadSeq = _loadSeq;
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
                    || myLoadSeq !== _loadSeq) return;
                if (!r.ok || !body.ok) return;  // silent — see Optimization helper
                _renderAutoDetectPanel(body);
                setStatus("auto-detect-status",
                    "Chemistry analyzed — click Auto-detect to "
                    + "apply suggested defaults to the form.",
                    null);
            } catch (_) {
                // Silent — background fire.
            }
        }

        /**
         * Fetch the spectra schema (so we don't have to reach
         * into the inspector's private state) and call
         * formSchema.setValues with the PySCF adapter's output.
         * SpectraConfig has charge / spin / method matching the
         * PySCF adapter output 1:1.
         */
        async function _applyAutoDetectToSpectraForm(resp) {
            const fs = (window.molbuilder || {}).formSchema;
            if (!fs || typeof fs.setValues !== "function") return;
            const container = _$("spectra-form-container");
            if (!container) return;
            const sug = (resp.suggested || {}).pyscf;
            if (!sug) return;
            let schema;
            try {
                schema = await fs.fetchSchema("spectra");
            } catch (e) {
                console.warn("[spectra] auto-detect: schema fetch failed:",
                              e);
                return;
            }
            fs.setValues(container, schema, sug);
        }

        function _renderAutoDetectPanel(resp) {
            const panel = _$("auto-detect-panel");
            if (!panel) return;
            panel.hidden = false;
            panel.open = true;
            const ratEl  = _$("auto-detect-rationale");
            const warnEl = _$("auto-detect-warnings");
            const metEl  = _$("auto-detect-metals");
            if (ratEl) {
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
        }

        // Refresh the Auto-detect button state every time the
        // load handler finishes (success or failure path) by
        // wrapping _refreshLoadButton with a sibling call.
        const _origRefreshLoad = _refreshLoadButton;
        _refreshLoadButton = function () {   // eslint-disable-line no-func-assign
            _origRefreshLoad();
            _refreshAutoDetectButton();
        };
        _refreshAutoDetectButton();
    }

    function bootstrapSpectraPage() {
        _bootstrapInspectCard();
        _bootstrapSpectraCore();
    }

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", bootstrapSpectraPage);
    } else {
        bootstrapSpectraPage();
    }
})();
