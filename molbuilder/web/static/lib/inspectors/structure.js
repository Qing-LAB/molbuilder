/* Structure-preview inspector: read-only 3-D view of a .xyz / .pdb
 * result file.
 *
 * This is the FIRST Results -> MolView conversion (molview-module.md §18,
 * §14.5.3).  The inspector no longer hand-assembles a viewer + an
 * ephemeral selection store + a panel + measurement/view controls
 * controllers.  It mounts the WHOLE MolView module read-only with one
 * call -- ``molview.mount(host, workspace, {mode:"readonly", owner})``
 * -- which builds the fused card, embeds the viewer, and wires the
 * selection panel + measurement overlay + view-controls
 * for free.  The molecule is opened through the ONE file door,
 * ``projects.parser.openMolecule(path)`` (reads the .xyz + its .molstruct.json
 * sidecar and installs the model -- labels + cell ride along).
 *
 * READ-ONLY means no EDIT controls (no modifier ops / Save-state) -- it has
 * NOTHING to do with persistence.  Like every other consumer, the Results
 * view uses the REAL workspace: its session state (opened file, current
 * frame, selection, camera) persists and RESTORES on reload, namespaced by
 * ``owner`` so it never mixes with the Modify tab's session.
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

/* NO NOTE ANY MORE (2026-08-03).
 *
 * This tab kept one fact under its own tag -- which file is on screen -- to
 * decide between restoring and re-opening.  That decision is gone (a read-only
 * viewer cannot restore; it always re-opens), and with the reader deleted the
 * writer was persisting a state file to disk on every open that NOTHING would
 * ever read.  A write with no reader is not harmless: it is disk I/O, a file in
 * the workspace states directory, and a fact a later reader might believe.
 *
 * If something needs to know this again, it can be added back WITH its reader. */

/* NO READER FOR THE NOTE ANY MORE.  `_readShowing` was removed with the restore
 * branch it fed (2026-08-03): this viewer is read-only, so it cannot restore a
 * session, and the note's only job now is to say what was last shown for anyone
 * who asks later.  A reader kept "just in case" is how the dead branch survived
 * long enough to blank the tab. */

// molview.data is MolView's live internal state -> LOOK IT UP at read time (molview-module.md
// §D.0), never import it. Returns whatever MolView currently has (null = nothing loaded).
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
            card.className = "inspector-card structure-card";

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
            // Hand the current file off to /molbuilder via the
            // sessionStorage keys the Projects sidebar uses (see
            // ``lib/projects/state.js`` SS_FILE / SS_DIR).  Closes #117.
            modifyLink.addEventListener("click", () => {
                try {
                    const C = (root.molbuilder || {}).constants || {};
                    root.sessionStorage.setItem(
                        C.SS_FILE || "molbuilder.current_file", file);
                    // Derive the parent dir from the file path so
                    // the sidebar lands on the right folder.
                    const i = file.lastIndexOf("/");
                    if (i >= 0) {
                        root.sessionStorage.setItem(
                            C.SS_DIR || "molbuilder.current_dir",
                            file.slice(0, i));
                    }
                } catch (_) {
                    // sessionStorage may throw under private-browsing
                    // / quota-exceeded; the link still navigates and
                    // /modify falls back to its previous state.
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
            molviewHost.className = "structure-viewer-slot";
            card.appendChild(molviewHost);

            host.appendChild(card);

            let handle   = null;
            let disposed = false;

            // The ONE door (projects.parser.openMolecule) reads the .xyz + its
            // .molstruct.json sidecar and installs the model (labels + cell ride along
            // -- MolView never parses).  No upfront ctx.readFile, no /api/selection/atoms
            // prefetch -- those were second reads of the same file.  The cell is DEDUCED
            // from the actual data (the .xyz's own lattice + the sidecar); there is no
            // load-time cell override (edit it on the Cell page if a change is needed).
            (async () => {
                if (disposed) return;
                // NOT gated on a viewer existing: the mount below is what
                // creates one, and testing for one first is what stopped three
                // pages mounting at all.
                if (typeof mount !== "function") {
                    status.textContent = (
                        "Viewer unavailable: the MolView module is missing "
                        + "from the template script tags."
                    );
                    status.classList.add("inspector-inline-error");
                    return;
                }

                // The REAL workspace persistence layer.  The Results view is a
                // session like any other consumer: its state (opened file, current
                // frame, selection, camera) persists and RESTORES on reload.
                // "Read-only" is about the absence of EDIT controls, NOT persistence.
                // The workspace namespaces by ``owner`` so this inspector's session
                // never mixes with the Modify tab's or another inspector's.
                const ws = root.molbuilder && root.molbuilder.workspace;
                if (!ws) {
                    status.textContent = (
                        "Viewer unavailable: the persistence layer "
                        + "(workspace/dispatcher.js) is missing from the template.");
                    status.classList.add("inspector-inline-error");
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
                    });
                    if (disposed) {
                        if (handle && typeof handle.dispose === "function") {
                            try { handle.dispose(); } catch (_) {}
                        }
                        return;
                    }
                    if (!handle || !handle.ok) {
                        status.textContent = "Viewer failed: "
                            + ((handle && handle.error) || "molview.mount failed.");
                        status.classList.add("inspector-inline-error");
                        return;
                    }

                    // ALWAYS A FRESH OPEN.  There is no restore branch, and there
                    // must not be one: this viewer is mounted `mode:"readonly"`, and
                    // § 9.4's gate makes every truth-changing door a NO-OP there --
                    // including `load`, which returns `Promise.resolve(null)` without
                    // touching the master copy (model.js: `load: gated(...)`).
                    //
                    // THE BUG THAT WAS (found in a browser 2026-08-03).  This used to
                    // branch: if this owner's note named the same file, restore with
                    // `handle.data.load(0)` "to bring back the selection / camera you
                    // left" instead of re-opening.  On a read-only viewer that call did
                    // NOTHING -- so the FIRST time you opened a file it worked, and on
                    // every visit after, the panel said a bare "Loaded." over an empty
                    // viewer and an empty atom list.  No request, no error: a no-op does
                    // not throw, and the status line's own empty branch printed the
                    // reassuring word.  Nothing was restored more cheaply; the structure
                    // was simply gone.
                    //
                    // Re-opening is also what the contract already says: a read-only tab
                    // keeps its structure by RELOADING it (the tab owns that, not the
                    // viewer -- molview.md § 12.3).  The camera and selection are not
                    // preserved, which is what was really happening all along.
                    //
                    // The registry only dispatches .xyz / .pdb
                    // to this inspector (see `match`), so the picked file IS the structure
                    // path -- no sidecar-path rewrite.  (Clicking the paired
                    // .molstruct.json shows its JSON via the `source` inspector: it is a
                    // metadata file; open the .xyz to view the structure.)
                    const structPath = file;
                    /* WHICH FILE THIS INSPECTOR IS SHOWING — this tab's own note,
                     * saved under this tab's own tag.
                     *
                     * It is not MolView's to remember. MolView tracks contents:
                     * the atoms, the labels, the cell. Which file they came out
                     * of is a fact about a file operation THIS tab performed, so
                     * this tab keeps it, next to the viewer's state and not
                     * inside it (workspace.md § 4 — several savers, one page).
                     *
                     * It used to be dug out of the viewer's saved bytes, which
                     * meant a path from the projects world was riding inside the
                     * structure's own saved state and being read back out by
                     * whoever needed it. */
                    {
                        // The format-aware sidebar door reads the .xyz +
                        // .molstruct.json (labels/regions/frozen + periodicity) and
                        // installs the model -- the sidecar rides along, which is what
                        // fixed the label-less atom list bug.
                        const _proj = root.molbuilder && root.molbuilder.projects;
                        if (!_proj || !_proj.parser
                                || typeof _proj.parser.openMolecule !== "function") {
                            status.textContent = "Viewer unavailable: the projects "
                                + "file package is missing from the template.";
                            status.classList.add("inspector-inline-error");
                            return;
                        }
                        const res = await _proj.parser.openMolecule(handle, structPath);
                        if (res && res.ok === false) {
                            status.textContent = "Error: "
                                + (res.error || "could not load " + structPath);
                            status.classList.add("inspector-inline-error");
                            return;
                        }
                    }
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
                        var _readyFired = false;
                        const dispatch = function () {
                            if (_readyFired) return;   // fire ONCE (rAF + timer race)
                            _readyFired = true;
                            document.dispatchEvent(
                                new CustomEvent(
                                    ((root.molbuilder || {}).constants || {})
                                        .EVENT_INSPECTOR_READY
                                    || "molbuilder:inspector:ready",
                                    { detail: { inspector: "structure" } }
                                )
                            );
                        };
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
                    status.textContent = "Viewer failed: "
                                       + (e && e.message ? e.message : String(e));
                    status.classList.add("inspector-inline-error");
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
