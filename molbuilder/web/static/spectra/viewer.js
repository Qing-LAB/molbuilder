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
import { molviewFiles } from "../lib/projects/molview-doors.js";
(function () {
    "use strict";

    function _$(id) { return document.getElementById(id); }

    /** The shared `.status` writer (lib/status.js).  This copy allowed only
     *  ok|warn|error and silently DROPPED anything else, so `muted` -- which
     *  page-shell declares and other tabs use -- rendered as nothing here. */
    function setStatus(elId, msg, kind) {
        window.molbuilder.status.set(elId, msg, kind);
    }

    /* THE INSPECTOR THIS PAGE MOUNTED, kept so the page can hand it the viewer
     * it also mounted.  One page, two things it owns, and it introduces them --
     * rather than either of them looking the other up. */
    let _inspector = null;
    // THE CHEMISTRY CARD (`lib/chemistry.js`): the charge and spin each
    // engine form's vibration will carry, resolved by the one
    // electronic-state class for exactly what the form says, with each
    // value's reason (`science/chemistry-correctness.md` § 2a) -- about the
    // structure the tab would hand over, read by the inspector's one read
    // of the viewer.  It asks on every structure load and every edit to a
    // charge or spin field, and fills nothing in.  It replaced an
    // Auto-detect button that spread a suggestion onto the forms,
    // overwriting them.
    const _chemistry = window.molbuilder.chemistry.attach({
        kind: "vibration",
        forms: () => (_inspector && typeof _inspector.stateForms === "function"
                      ? _inspector.stateForms() : {}),
        structure: () => (_inspector
                          && typeof _inspector.structureForRequest === "function"
                          ? _inspector.structureForRequest() : null),
    });

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
        _inspector = api.mount(document, {
            // The forms exist now: the chemistry card answers for them.
            onFormsReady: () => { if (_chemistry) _chemistry.refresh(); },
        });
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
                        // SAY WHY, not just that.  The reason used to exist
                        // only on the commit path (double-click), so a single
                        // click left "not loadable" standing alone with no
                        // way to find out what would be loadable.
                        : (_candidatePath
                            ? `Selected: ${_basename(_candidatePath)} `
                              + `(not loadable — .xyz / .pdb only)`
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
            if (f === _sidebarLastFile) {
                // Same file: the structure on screen stays, and so does the
                // card's answer about it.  The Load button is what fetches
                // new bytes (it clears this guard, below).
                return;
            }
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
                        { mode: "readonly", owner: WORKSPACE_TAG,
                          files: molviewFiles });
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
            // The inspector reads the structure off the viewer it was handed
            // at mount -- no second copy -- and is told a load landed so the
            // structure can pick the default engine (a periodic structure is
            // SIESTA's) and the live checks rerun.
            if (_inspector && typeof _inspector.structureLoaded === "function") {
                _inspector.structureLoaded();
            }
            _updateInfo(mvHandle, _basename(f));
            setStatus("load-status",
                `Loaded ${_basename(f)}.`, "ok");
            _refreshLoadButton();
            // The chemistry card: this structure's charge and spin, for
            // exactly what the forms say.
            _chemistry.refresh();
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
                // Explicit Load = "load the current file NOW", even the one
                // already on screen: it may have changed on disk.  Clearing
                // the same-file guard is how the Build tab's Load does it;
                // this one skipped the reload until the M6 review.  (The
                // sidebar's double-click path keeps the guard.)
                _sidebarLastFile = "";
                _commitStructure({ file: _candidatePath });
            });
        }

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
