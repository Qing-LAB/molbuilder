/* Structure-preview inspector: read-only 3-D view of a .xyz / .pdb
 * file.  Loads the file via ``ctx.readFile`` and hands the text +
 * format to the standard embedded MolViewer (lib/mol-viewer-embed.js,
 * contract: docs/protocols/molview-module.md).
 *
 * No frame playback, no force overlay -- that's the trajectory
 * inspector's job.  This is a static-structure peek.  An atom-pick
 * + measurement chip (xyz / distance / angle) was added 2026-06-09
 * (task #300) so users reading a result file can read off bond
 * geometry without opening it in Molbuilder; the chip uses the
 * shared ``lib/selection/measurements.js`` math + display.
 *
 * The inspector owns its OWN card (with the "Open in Molbuilder"
 * link the user expects on /results); the embedded viewer mounts
 * inside that card with ``card.title: ""`` so the embed doesn't
 * render a second header — the host's card-header is the title
 * strip.  The ``card.bare`` opt was retired 2026-06-03 along with
 * the .mol-viewer-bare CSS class; ``card.title: ""`` is the
 * canonical way to skip the embed's header.
 */
(function (root) {
    "use strict";

    // NOTE: this inspector is a VIEWER glue layer -- it must NOT parse structure
    // files or .fdf for the cell / k-grid.  molbuilder/parse/ already extracts
    // those (StructureResult.cell + the fdf kgrid diagonal); the results tab is
    // responsible for concentrating them and passing them in as params
    // (ctx.viewParams).  The viewer only cares whether a cell / kgrid was handed
    // to it or not.  See Slice C.

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
            // ``lib/projects/state.js`` SS_FILE / SS_DIR).
            // The Molbuilder tab's selection-bootstrap reads SS_FILE
            // on mount via ``projects.getCurrentFile()`` and
            // dispatches the auto-load.  Without setting these the
            // user would land on the tab with whatever file was
            // previously active — or an empty viewer.  Setting both
            // keys also makes the sidebar open to the correct folder
            // with the file highlighted.  Closes #117.
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

            // -- Slot the embedded viewer will mount into --------- //
            // Wrapper anchors the absolute-positioned measurement
            // chip to the viewer's actual canvas area (the embed's
            // card body fills the slot at 420 px height per the
            // ``card.height`` opt below).
            const viewerSlot = document.createElement("div");
            viewerSlot.className = "structure-viewer-slot molview-viewer";
            // The measurement readout is now the module overlay (§ 6.4),
            // mounted into viewerSlot in onReady -- no bespoke chip here.

            // -- Phase 5 (fused module): a readonly selection panel driven by
            // an ISOLATED ephemeral store -- the inspector owns its own
            // selection/filter; nothing touches the Molbuilder workspace.  If
            // the selection modules aren't on the page (createEphemeralStore
            // absent), the inspector degrades to the plain viewer + chip.
            const selApi   = (root.molbuilder && root.molbuilder.selection) || {};
            const selStore = (typeof selApi.createEphemeralStore === "function")
                ? selApi.createEphemeralStore() : null;
            let   panelMount = null;
            let   measOverlay = null;
            let   kgCtl = null;
            const panelHost  = document.createElement("div");
            panelHost.className = "structure-selection-host molview-panel";

            if (selStore) {
                // ONE fused card: viewer + foldable selection panel.  Side
                // (wide) / bottom (narrow) is CSS (fused-layout.css container
                // query); the fold chevron is local UI layout state, not the
                // store.  DOM built BEFORE embedding so the 3Dmol canvas is never
                // reparented (see § 6).
                card.classList.add("molview-card");
                const body = document.createElement("div");
                body.className = "molview-body";
                body.appendChild(viewerSlot);
                const foldBtn = document.createElement("button");
                foldBtn.type = "button";
                foldBtn.className = "molview-fold-btn";
                foldBtn.setAttribute("aria-label", "Fold or unfold the selection panel");
                foldBtn.setAttribute("aria-expanded", "true");
                foldBtn.textContent = "›";   // › -- light chevron; CSS rotates it on fold
                foldBtn.addEventListener("click", () => {
                    const folded = card.classList.toggle("is-folded");
                    foldBtn.setAttribute("aria-expanded", String(!folded));
                });
                body.appendChild(foldBtn);
                body.appendChild(panelHost);
                card.appendChild(body);
            } else {
                card.appendChild(viewerSlot);   // no selection modules -> plain viewer
            }

            host.appendChild(card);

            let viewerHandle = null;
            let disposed     = false;

            ctx.readFile(file).then(async (r) => {
                if (disposed) return;
                if (!r.ok) {
                    status.textContent = "Error: " + (r.error || "unknown");
                    status.classList.add("inspector-inline-error");
                    return;
                }
                const fmt = file.toLowerCase().endsWith(".pdb")
                    ? "pdb" : "xyz";
                // Phase 1 (structure-periodicity.md): the cell comes from the
                // dataset, read server-side and handed to the viewer -- the viewer
                // NEVER parses.  ctx.viewParams.cell wins if the host already
                // supplied it; otherwise fetch it from the .xyz's sidecar via
                // /api/selection/atoms (which loads the structure + sidecar).
                // Absent -> null: no unit-cell box, k-grid stays inert.
                let lattice = (ctx && ctx.viewParams && ctx.viewParams.cell) || null;
                if (!lattice && fmt === "xyz") {
                    try {
                        const cr = await fetch("/api/selection/atoms", {
                            method:  "POST",
                            headers: { "Content-Type": "application/json" },
                            body:    JSON.stringify({ structure_path: file }),
                        });
                        if (cr.ok) {
                            const cj = await cr.json();
                            if (cj && cj.ok && cj.cell) lattice = cj.cell;
                        }
                    } catch (_) { /* no cell -> no box, k-grid inert */ }
                    if (disposed) return;
                }
                const embedApi = (root.molbuilder
                                  && root.molbuilder.viewer
                                  && root.molbuilder.viewer.embed);
                if (typeof embedApi !== "function") {
                    status.textContent = (
                        "Viewer unavailable: lib/mol-viewer-embed.js "
                        + "missing from the template script tags."
                    );
                    status.classList.add("inspector-inline-error");
                    return;
                }
                const opts = {
                    // Source data flows in through the API; the
                    // viewer doesn't fetch.
                    [fmt]: r.text,
                    // Cell from the dataset (fetched above) -> the unit-cell
                    // wireframe can show + k-grid can tile getLattice().
                    lattice: lattice || undefined,
                    // Style matches the legacy structure inspector's
                    // ball-and-stick rendering.
                    style: { rep: "ball-and-stick", radiusScale: 1.0 },
                    // Post-#204: standard knob bar visible above the
                    // canvas (Style / Labels / Axes / Reset / PNG /
                    // Background / Export).  Title suppressed because
                    // the inspector's parent .inspector-pane already
                    // shows the file name + status note; the embed's
                    // info-line stays off for the same reason.
                    card: {
                        title:        "",
                        showInfoLine: false,
                        height:       "420px",
                    },
                    // Axes default OFF (2026-06-13) — consistent
                    // across all viewer mount sites; opt in via the
                    // knob bar's Axes button.
                    axes:   false,
                    export: { defaultName: r.basename || "structure" },
                    // No embed pick here: the viewer-adapter wires clicks ->
                    // store.toggle (single pick), and the measurement overlay
                    // (§ 6.4) + selection halos derive from the store.
                    onReady: function (handle) {
                        if (disposed) return;
                        const n = handle.getAtomCount();
                        status.textContent = n > 0
                            ? "Loaded " + n + " atoms."
                            : "Loaded.";
                        // Phase 5: feed the isolated store the loaded atoms so
                        // the panel's list renders.  Providing ``atoms`` skips
                        // any server fetch (readonly result files aren't
                        // selection sources server-side).
                        if (selStore && typeof selStore.adoptSession === "function") {
                            try {
                                const elems = handle.getElements() || [];
                                selStore.adoptSession({
                                    // sourceFile is REQUIRED for the panel's
                                    // filter: applyFilter posts {structure_path}
                                    // to /api/selection/eval and refuses without
                                    // it ("no source file").  The server reads
                                    // the result file directly (by-element etc.;
                                    // by-residue needs a sidecar it won't have).
                                    sourceFile: file,
                                    atoms: elems.map((el, i) => ({ index: i, element: el || "X" })),
                                    selection: [],
                                });
                            } catch (_) { /* panel just shows an empty list */ }
                        }
                        // Phase 5 (§ 6.4): the measurement overlay -- it derives
                        // position/distance/angle from the SELECTION (viewer
                        // clicks + panel both feed the store), with coords from
                        // this static structure's viewer handle.
                        const mvApi = (root.molbuilder && root.molbuilder.molview) || {};
                        if (selStore && typeof mvApi.mountMeasurementOverlay === "function") {
                            try {
                                measOverlay = mvApi.mountMeasurementOverlay(viewerSlot, {
                                    store: selStore,
                                    coordsProvider: () => handle.getAtomCoords(),
                                });
                            } catch (_) { /* overlay absent -> no readout */ }
                        }
                        // Test hook: expose the ephemeral store so e2e can drive
                        // the selection (replaces the retired chip pick hooks).
                        viewerSlot.__molbuilder_test_store = selStore;

                        // k-grid render: THE module controller (molview-module.md
                        // § 7) -- no inline loop here.  Capture the UNIT-CELL
                        // coords/elements/xyz ONCE (before any tiling; getAtomCoords
                        // reads back the supercell after setStructure), and hand the
                        // module the cell (host supplies it, § 8) + this unit.  The
                        // module subscribes, tiles on enable, restores on disable.
                        if (selStore && lattice
                                && typeof mvApi.mountKgridRender === "function") {
                            const _unit = {
                                coords:   handle.getAtomCoords(),
                                elements: handle.getElements(),
                                xyz:      r.text,
                            };
                            kgCtl = mvApi.mountKgridRender(handle, selStore, {
                                getUnit: function () { return _unit; },
                                getCell: function () { return lattice; },
                            });
                        }
                        // Signal "first render visible" so the
                        // /results tab-level picker drops its
                        // "Parsing…" status.  Deferred via double-rAF
                        // so the browser paints the 3Dmol canvas
                        // before the picker meta clears -- matches
                        // the trajectory inspector's pattern; see
                        // ``lib/trajectory/core.js`` for the reasoning
                        // behind the double-tick wait.
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
                    },
                };
                try {
                    viewerHandle = embedApi(viewerSlot, opts);
                    // Phase 5: mount the readonly selection panel against the
                    // isolated store + this viewer handle.  ``mode: readonly``
                    // makes the adapter PAINT selection onto the viewer while
                    // leaving the triple-pick measurement clicks untouched, and
                    // hides the panel's assign (write) controls.
                    if (selStore && typeof selApi.mountPanel === "function") {
                        selApi.mountPanel(panelHost, {
                            store:        selStore,
                            viewerHandle: viewerHandle,
                            mode:         "readonly",
                        }).then((m) => {
                            panelMount = m;
                            // Disposed before the async mount resolved -> tear
                            // the just-mounted panel down immediately.
                            if (disposed && m && m.panel
                                && typeof m.panel.dispose === "function") {
                                try { m.panel.dispose(); } catch (_) {}
                            }
                        });
                    }
                    // Test hook (no production reader): stash the embed handle on
                    // the slot so Playwright e2e can drive the viewer without
                    // canvas-pixel clicks.  The store hook (set in onReady) lets
                    // e2e drive the SELECTION, which now drives the measurement
                    // overlay (§ 6.4).  Double-underscore = test-only; hung off
                    // the slot (not window) to avoid multi-inspector collisions.
                    viewerSlot.__molbuilder_test_handle = viewerHandle;
                } catch (e) {
                    status.textContent = "Viewer failed: "
                                       + (e && e.message ? e.message : String(e));
                    status.classList.add("inspector-inline-error");
                }
            });

            return {
                dispose() {
                    disposed = true;
                    if (kgCtl) { try { kgCtl.dispose(); } catch (_) {} }
                    if (measOverlay && typeof measOverlay.dispose === "function") {
                        try { measOverlay.dispose(); } catch (_) {}
                    }
                    if (panelMount && panelMount.panel
                        && typeof panelMount.panel.dispose === "function") {
                        try { panelMount.panel.dispose(); } catch (_) {}
                    }
                    if (viewerHandle && typeof viewerHandle.dispose === "function") {
                        try { viewerHandle.dispose(); }
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
