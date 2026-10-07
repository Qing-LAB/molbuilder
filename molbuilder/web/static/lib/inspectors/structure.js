/* Structure-preview inspector: read-only 3-D view of a .xyz / .pdb
 * result file.
 *
 * The read-only viewer in the Results tab (web/molview.md § 4 + § 12.3).
 * It mounts the WHOLE MolView module read-only with one
 * call -- ``molview.mount(host, workspace, {mode:"readonly", owner})``
 * -- which builds the fused card, embeds the viewer, and wires the
 * selection panel + measurement overlay + view-controls
 * for free.  The molecule is opened through the ONE file door,
 * ``projects.parser.openMolecule(path)`` (reads the .xyz + its .molstruct.json
 * sidecar and installs the model -- labels + cell ride along).
 *
 * READ-ONLY means no EDIT controls (no modifier ops / Save-state).  Like
 * every other consumer, the Results view uses the REAL workspace,
 * namespaced by ``owner`` so it never mixes with the Modify tab's session.
 *
 * The inspector still owns its OWN card chrome (title + the "Open in
 * Molbuilder" link the user expects on /results + the status note); the
 * MolView fused card mounts inside that card, in an empty host div (the
 * host is NOT a .molviewer-card, so molview.mount takes its empty-host build
 * path and owns the whole assembly).
 */
import { mount } from "/static/lib/molview/index.js";
/* WHO THESE BYTES BELONG TO (workspace.md § 4) — the one string used BOTH as the
 * viewer's `owner` and as the tag on every workspace call, so the two cannot
 * drift into naming different slots. */
const WORKSPACE_TAG = "results:structure";
import { molviewFiles } from "../projects/molview-doors.js";

(function (root) {
    "use strict";

    // NOTE: this inspector is a VIEWER glue layer -- it must NOT parse structure files
    // or .fdf for the cell.  The ONE file door (projects.parser.openMolecule) reads the
    // .xyz + its .molstruct.json sidecar, and the model DEDUCES the cell from that data
    // (the .xyz's own lattice + the sidecar).  The inspector passes no cell / no
    // periodicity -- there is no load-time override.

    const inspector = {
        name:        "structure",
        displayName: "Structure preview",
        // PySCF / geomeTRIC writes multi-frame ``*_optim.xyz`` files
        // and SIESTA-side helpers may dump intermediate/final ``.xyz``
        // / ``.pdb`` -- both are user-meaningful results in a project
        // dir, so the file picker on /results should surface them.
        isResult:    true,
        match:       (file) => {
            const lower = file.toLowerCase();
            return lower.endsWith(".xyz") || lower.endsWith(".pdb");
        },
        resultCategory: (_file) => "Structure",

        mount(host, file, ctx) {
            host.innerHTML = "";

            // -- Outer card scaffold (per-inspector chrome) ------- //
            const card = document.createElement("section");
            card.className = "card inspector-card structure-card";

            const header = document.createElement("header");
            header.className = "inspector-card-header";
            const title = document.createElement("h2");
            title.className = "inspector-card-title";
            title.textContent = "Structure — " + _basename(file);
            header.appendChild(title);
            const actions = document.createElement("div");
            actions.className = "inspector-card-actions";
            const modifyLink = document.createElement("a");
            modifyLink.href = "/molbuilder";
            modifyLink.textContent = "Open in Molbuilder";
            modifyLink.className = "inspector-card-link";
            modifyLink.title = (
                "Loads the structure into the Molbuilder tab so you "
                + "can rotate / orient / add electrodes / etc."
            );
            modifyLink.addEventListener("click", () => {
                // Through the door (projects.md § 5): the selection slots
                // are per-tab, so the handoff must write the TARGET
                // page's slot -- raw key writes only feed the fallback a
                // page's own memory shadows.  Failure is non-fatal: the
                // link still navigates and /molbuilder keeps its state.
                const proj = (root.molbuilder || {}).projects;
                if (proj && typeof proj.handOffSelection === "function") {
                    const i = file.lastIndexOf("/");
                    proj.handOffSelection("/molbuilder",
                        i >= 0 ? file.slice(0, i) : "", file);
                }
            });
            actions.appendChild(modifyLink);
            header.appendChild(actions);
            card.appendChild(header);

            const status = document.createElement("p");
            status.className = "inspector-card-note structure-status";
            status.textContent = "Loading…";
            card.appendChild(status);

            // -- Empty host the MolView module mounts into --------- //
            // A plain div (NOT a .molviewer-card): molview.mount takes its
            // empty-host build path and BUILDS the fused card (viewer +
            // panel + fold + view-controls + measurement) inside.
            const molviewHost = document.createElement("div");
            // `molviewer-host`: below MolView's own floor the viewer scrolls in
            // its box rather than dragging the page sideways (molview.css).
            molviewHost.className = "structure-viewer-slot molviewer-host";
            card.appendChild(molviewHost);

            host.appendChild(card);

            let handle   = null;
            let disposed = false;
            /* THE LOAD HAS ENDED -- drawn, or refused with its reason on the
             * status line.  One signal for both, ONCE, so the tab's loading
             * cover and the picker's "Parsing…" line go when the answer is on
             * screen, whichever it is. */
            let readyFired = false;
            const signalReady = (detail) => {
                if (readyFired || disposed) return;
                readyFired = true;
                try {
                    document.dispatchEvent(new CustomEvent(
                        ((root.molbuilder || {}).constants || {})
                            .EVENT_INSPECTOR_READY || "molbuilder:inspector:ready",
                        { detail: Object.assign({ inspector: "structure" },
                                                detail) }));
                } catch (_) { /* see core.js for context */ }
            };
            const fail = (msg) => {
                status.textContent = msg;
                status.classList.add("inspector-inline-error");
                signalReady({ error: msg });
            };

            // The ONE door (projects.parser.openMolecule) reads the .xyz + its
            // .molstruct.json sidecar and installs the model (labels + cell ride along
            // -- MolView never parses).  No upfront ctx.readFile: that would be a
            // second read of the same file.  The cell is DEDUCED
            // from the actual data (the .xyz's own lattice + the sidecar); there is no
            // load-time cell override (edit it on the Cell page if a change is needed).
            (async () => {
                if (disposed) return;
                // NOT gated on a viewer existing: the mount below is what
                // creates one.
                if (typeof mount !== "function") {
                    fail("Viewer unavailable: the MolView module is missing "
                         + "from the template script tags.");
                    return;
                }

                // The REAL workspace persistence layer.  The workspace namespaces
                // by ``owner`` so this inspector's session never mixes with the
                // Modify tab's or another inspector's.
                const ws = root.molbuilder && root.molbuilder.workspace;
                if (!ws) {
                    fail("Viewer unavailable: the persistence layer "
                         + "(workspace/dispatcher.js) is missing from the template.");
                    return;
                }

                try {
                    // ONE call mounts the whole read-only component.  The panel is
                    // wired read-only (no assign/write controls); the measurement
                    // overlay and view-controls (Show selected only) all come
                    // for free through molview.mount.
                    handle = await mount(molviewHost, ws, {
                        mode:  "readonly",
                        owner: WORKSPACE_TAG,
                        files: molviewFiles,
                    });
                    if (disposed) {
                        if (handle && typeof handle.dispose === "function") {
                            try { handle.dispose(); } catch (_) {}
                        }
                        return;
                    }
                    if (!handle || !handle.ok) {
                        fail("Viewer failed: "
                             + ((handle && handle.error) || "molview.mount failed."));
                        return;
                    }

                    // ALWAYS A FRESH OPEN.  There is no restore branch, and there
                    // must not be one: this viewer is mounted `mode:"readonly"`, and
                    // § 9.4's gate makes every truth-changing door a NO-OP there --
                    // including `load`, which returns `Promise.resolve(null)` without
                    // touching the master copy (model.js: `load: gated(...)`).
                    //
                    // Re-opening is also what the contract says: a read-only tab
                    // keeps its structure by RELOADING it (the tab owns that, not the
                    // viewer -- molview.md § 12.3).  The camera and selection are not
                    // preserved.
                    //
                    // The registry only dispatches .xyz / .pdb
                    // to this inspector (see `match`), so the picked file IS the structure
                    // path -- no sidecar-path rewrite.  (Clicking the paired
                    // .molstruct.json shows its JSON via the `source` inspector: it is a
                    // metadata file; open the .xyz to view the structure.)
                    const structPath = file;
                    {
                        // The format-aware sidebar door reads the .xyz +
                        // .molstruct.json (labels/regions/frozen + periodicity) and
                        // installs the model -- the sidecar rides along.
                        const _proj = root.molbuilder && root.molbuilder.projects;
                        if (!_proj || !_proj.parser
                                || typeof _proj.parser.openMolecule !== "function") {
                            fail("Viewer unavailable: the projects "
                                 + "file package is missing from the template.");
                            return;
                        }
                        const res = await _proj.parser.openMolecule(handle, structPath);
                        if (res && res.ok === false) {
                            fail("Error: "
                                 + (res.error || "could not load " + structPath));
                            return;
                        }
                    }
                    /* RECORD WHAT THE RUN DIRECTORY SAYS ABOUT THIS
                     * STRUCTURE (structure-info-plan.md I5; model/parse.md
                     * 5b, 5b.1): the electronic contract its deck states
                     * (`calculation`) and what the run did to the geometry
                     * it left (`relaxation`).  Recorded through the info
                     * door, each shows on the Metadata page and travels
                     * with any export -- which is what lets a transport
                     * citation of the pair run sealed (4.1b), and lets a
                     * vibration calculation check a structure stated
                     * relaxed against its own record (vibration.md 2.2).
                     * Best-effort: the viewer works without either. */
                    try {
                        const cr = await fetch("/api/results/contract?path="
                            + encodeURIComponent(structPath));
                        const cb = await cr.json();
                        if (!disposed && cb && cb.ok
                                && handle.data && handle.data.info
                                && typeof handle.data.info.set === "function") {
                            for (const key of ["calculation", "relaxation"]) {
                                if (cb[key]) handle.data.info.set(key, cb[key]);
                            }
                        }
                    } catch (_) { /* no record is a real answer */ }
                    if (disposed) return;

                    const elems = handle.data.getElements() || [];
                    status.textContent = elems.length > 0
                        ? "Loaded " + elems.length + " atoms."
                        : "Loaded.";

                    // Test hook (no production reader): stash the handle on the host
                    // so Playwright e2e can drive the read-only view.  The SELECTION +
                    // structure are read off the global molview.data singleton
                    // (molview conceals its internals; the owner has no store ref).
                    molviewHost.__molview_results_handle = handle;

                    // Signal "first render visible" so the /results tab-level picker
                    // drops its "Parsing…" status.  Deferred via double-rAF so the
                    // browser paints the 3Dmol canvas before the picker meta clears
                    // -- matches the trajectory inspector's pattern (core.js).
                    try {
                        // ONCE, whichever of rAF and the timer wins.
                        const dispatch = function () { signalReady({}); };
                        // Prefer a post-paint dispatch (double-rAF) so the 3Dmol
                        // canvas is on screen before the picker drops its "parsing…"
                        // overlay -- no flash of empty viewer.  BUT rAF is paused in a
                        // BACKGROUNDED tab, which would leave the overlay stuck until
                        // the picker's 15s fallback ("parsing for a long time").  A
                        // short timer guarantees the ready signal fires regardless of
                        // paints; whichever wins, ``dispatch`` runs exactly once.
                        if (typeof requestAnimationFrame === "function") {
                            requestAnimationFrame(
                                () => requestAnimationFrame(dispatch));
                            setTimeout(dispatch, 250);
                        } else {
                            dispatch();
                        }
                    } catch (_) { /* see core.js for context */ }
                } catch (e) {
                    fail("Viewer failed: "
                         + (e && e.message ? e.message : String(e)));
                }
            })();

            return {
                dispose() {
                    disposed = true;
                    // molview.mount's handle tears down the whole assembly (viewer,
                    // panel, controls, overlays, subscriptions).
                    if (handle && typeof handle.dispose === "function") {
                        try { handle.dispose(); }
                        catch (_) { /* already torn down */ }
                    }
                    host.innerHTML = "";
                },
            };
        },
    };

    const _basename = (window.molbuilder
                       && window.molbuilder.path
                       && window.molbuilder.path.basename)
                    || ((p) => p || "");

    root.molbuilder = root.molbuilder || {};
    root.molbuilder.inspectors = root.molbuilder.inspectors || {};
    root.molbuilder.inspectors.structureInspector = inspector;
    if (root.molbuilder.inspectors.register) {
        root.molbuilder.inspectors.register(inspector);
    }
})(typeof window !== "undefined" ? window : this);
