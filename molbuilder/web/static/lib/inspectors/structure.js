/* Structure-preview inspector: read-only 3-D view of a .xyz / .pdb
 * result file.
 *
 * This is the FIRST Results -> MolView conversion (molview-module.md §18,
 * §14.5.3).  The inspector no longer hand-assembles a viewer + an
 * ephemeral selection store + a panel + measurement/view/k-grid
 * controllers.  It mounts the WHOLE MolView module read-only with one
 * call -- ``molview.mount(host, workspace, {mode:"readonly", owner})``
 * -- which builds the fused card, embeds the viewer, and wires the
 * selection panel + measurement overlay + view-controls + k-grid render
 * for free.  The molecule is opened through the module's single data door,
 * ``molview.data.openMolecule({text, filename, periodicity})``.
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
 * host is NOT a .molview-card, so molview.mount takes its empty-host build
 * path and owns the whole assembly).
 */
(function (root) {
    "use strict";

    // NOTE: this inspector is a VIEWER glue layer -- it must NOT parse structure
    // files or .fdf for the cell.  molbuilder/parse/ already extracts that
    // (StructureResult.cell); the results tab is
    // responsible for concentrating them and passing them in as params
    // (ctx.viewParams).  MolView only cares whether a periodicity was handed to
    // it (via openMolecule) or not.

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
            // A plain div (NOT a .molview-card): molview.mount takes its
            // empty-host build path and BUILDS the fused card (viewer +
            // panel + fold + view-controls + measurement + k-grid) inside.
            const molviewHost = document.createElement("div");
            molviewHost.className = "structure-viewer-slot";
            card.appendChild(molviewHost);

            host.appendChild(card);

            let handle   = null;
            let disposed = false;

            ctx.readFile(file).then(async (r) => {
                if (disposed) return;
                if (!r.ok) {
                    status.textContent = "Error: " + (r.error || "unknown");
                    status.classList.add("inspector-inline-error");
                    return;
                }
                const fmt = file.toLowerCase().endsWith(".pdb") ? "pdb" : "xyz";

                // Periodicity (structure-periodicity.md): the cell comes from the
                // dataset, read server-side and handed to MolView via openMolecule
                // -- MolView NEVER parses.  ctx.viewParams wins if the host already
                // supplied it; otherwise fetch the sidecar's full periodicity
                // ({cell, axis_kind, vacuum}) from /api/selection/atoms.  Absent ->
                // null: no unit-cell box, Cell page shows defaults.
                const vp = (ctx && ctx.viewParams) || {};
                let periodicity = vp.periodicity || (vp.cell ? { cell: vp.cell } : null);
                if (!periodicity && fmt === "xyz") {
                    try {
                        const cr = await fetch("/api/selection/atoms", {
                            method:  "POST",
                            headers: { "Content-Type": "application/json" },
                            body:    JSON.stringify({ structure_path: file }),
                        });
                        if (cr.ok) {
                            const cj = await cr.json();
                            if (cj && cj.ok) {
                                periodicity = cj.periodicity
                                    || (cj.cell ? { cell: cj.cell } : null);
                            }
                        }
                    } catch (_) { /* no cell -> no box, k-grid inert */ }
                    if (disposed) return;
                }

                const mv = (root.molbuilder && root.molbuilder.molview) || null;
                if (!mv || typeof mv.mount !== "function" || !mv.data) {
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
                    // overlay, view-controls (Show selected only / Show k-grid) and
                    // the k-grid render all come for free through molview.mount.
                    handle = await mv.mount(molviewHost, ws, {
                        mode:  "readonly",
                        owner: "results:structure",
                    });
                    if (disposed) {
                        if (handle && typeof handle.dispose === "function") {
                            try { handle.dispose(); } catch (_) {}
                        }
                        return;
                    }
                    if (!handle) {
                        status.textContent = "Viewer failed: molview.mount returned null.";
                        status.classList.add("inspector-inline-error");
                        return;
                    }

                    // Restore vs. fresh open (single-authority mount race,
                    // workspace-contract §4.5 -- same pattern as Modify's
                    // selection-bootstrap).  If this owner's persisted session is for
                    // the SAME file we're about to show, RESTORE it (``load(0)`` brings
                    // back the selection / camera you left on reload) instead of a fresh
                    // ``openMolecule`` (which resets the timeline and drops that state).
                    // A different file (you picked a new one) -> open fresh.
                    const restoreTarget =
                        (typeof ws.mountRestoreTarget === "function")
                            ? ws.mountRestoreTarget() : null;
                    if (restoreTarget && restoreTarget === file) {
                        await mv.data.load(0);
                    } else {
                        // Open the molecule through the ONE data door.  Periodicity rides
                        // along so the k-grid + Cell page work.  This is a data change the
                        // render reacts to on its own (molview owns the render loop).
                        await mv.data.openMolecule({
                            text:        r.text,
                            filename:    file,
                            periodicity: periodicity || null,
                        });
                    }
                    if (disposed) return;

                    const elems = (typeof mv.data.getElements === "function"
                        && mv.data.getElements()) || [];
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
                        const dispatch = () => document.dispatchEvent(
                            new CustomEvent(
                                ((root.molbuilder || {}).constants || {})
                                    .EVENT_INSPECTOR_READY
                                || "molbuilder:inspector:ready",
                                { detail: { inspector: "structure" } }
                            )
                        );
                        if (typeof requestAnimationFrame === "function") {
                            requestAnimationFrame(
                                () => requestAnimationFrame(dispatch));
                        } else {
                            dispatch();
                        }
                    } catch (_) { /* see core.js for context */ }
                } catch (e) {
                    status.textContent = "Viewer failed: "
                                       + (e && e.message ? e.message : String(e));
                    status.classList.add("inspector-inline-error");
                }
            });

            return {
                dispose() {
                    disposed = true;
                    // molview.mount's handle tears down the whole assembly (viewer,
                    // panel, controls, overlays, k-grid, subscriptions).
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
