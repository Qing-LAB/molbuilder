/* Structure-preview inspector: read-only 3-D view of a .xyz / .pdb
 * file.  Loads the file via ``ctx.readFile`` and hands the text +
 * format to the standard embedded MolViewer (lib/mol-viewer-embed.js,
 * contract: docs/protocols/embedded-viewer.md).
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
            viewerSlot.className = "structure-viewer-slot";
            card.appendChild(viewerSlot);
            const chip = document.createElement("div");
            chip.id = "structure-measurement";
            chip.className = "selection-measurement-overlay";
            chip.hidden = true;
            viewerSlot.appendChild(chip);

            host.appendChild(card);

            let viewerHandle = null;
            let disposed     = false;

            ctx.readFile(file).then((r) => {
                if (disposed) return;
                if (!r.ok) {
                    status.textContent = "Error: " + (r.error || "unknown");
                    status.classList.add("inspector-inline-error");
                    return;
                }
                const fmt = file.toLowerCase().endsWith(".pdb")
                    ? "pdb" : "xyz";
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
                const _meas = (root.molbuilder
                               && root.molbuilder.selection
                               && root.molbuilder.selection.measurements);
                const _updateChip = (handle) => {
                    if (!handle || disposed) return;
                    const pa = handle.getPickedIndices();
                    if (!_meas || pa.length === 0) {
                        chip.hidden = true;
                        chip.textContent = "";
                        return;
                    }
                    const positions = handle.getAtomCoords();
                    const elements  = handle.getElements();
                    // Build a tiny atomsMeta array on the fly —
                    // the inspector doesn't have residue context
                    // (that lives in a .molstruct.json sidecar,
                    // which Results doesn't fetch by design); the
                    // element + 1-based index is the right label
                    // for a static-structure peek.
                    const atomsMeta = elements.map(
                        (el, i) => ({ index: i, element: el || "?" }));
                    // pickedIndices preserves click order in the
                    // embed (pair / triple modes append in order);
                    // pass it through as both selection AND
                    // pickOrder so the angle vertex matches the
                    // user's 2nd click.
                    const result = _meas.compute(
                        pa, atomsMeta, positions, pa);
                    if (result) {
                        chip.hidden = false;
                        chip.dataset.kind = result.kind;
                        chip.textContent  = result.display;
                    } else {
                        chip.hidden = true;
                        chip.textContent = "";
                    }
                };
                const opts = {
                    // Source data flows in through the API; the
                    // viewer doesn't fetch.
                    [fmt]: r.text,
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
                    // 3-atom pick with halos drives the measurement
                    // chip — click 1 atom for xyz, 2 for distance,
                    // 3 for ∠A-B-C (vertex = 2nd click).  Halo is
                    // the same yellow sphere overlay the trajectory
                    // inspector uses.
                    pick: {
                        mode:  "triple",
                        halo:  true,
                        label: false,
                        onPick() {
                            _updateChip(viewerHandle);
                        },
                    },
                    onReady: function (handle) {
                        if (disposed) return;
                        const n = handle.getAtomCount();
                        status.textContent = n > 0
                            ? "Loaded " + n + " atoms."
                            : "Loaded.";
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
                    // Test hooks (no production reader): stash the
                    // embed handle + chip-update closure on the
                    // viewer slot DOM node so Playwright e2e can
                    // drive setPickedIndices + refresh the chip
                    // without simulating canvas-pixel clicks.  The
                    // embed deliberately suppresses ``onPick`` on
                    // ``setPickedIndices`` (avoids feedback loops),
                    // so test code that drives picks externally
                    // must trigger the chip update directly via
                    // ``__molbuilder_test_refreshChip()``.  The
                    // double-underscore prefix marks "test only";
                    // properties hang off the slot rather than
                    // ``window`` to avoid global collisions when
                    // multiple inspectors mount in the same run.
                    // Disposed implicitly when the inspector tears
                    // the host down.
                    viewerSlot.__molbuilder_test_handle = viewerHandle;
                    viewerSlot.__molbuilder_test_chip   = chip;
                    viewerSlot.__molbuilder_test_refreshChip =
                        () => _updateChip(viewerHandle);
                } catch (e) {
                    status.textContent = "Viewer failed: "
                                       + (e && e.message ? e.message : String(e));
                    status.classList.add("inspector-inline-error");
                }
            });

            return {
                dispose() {
                    disposed = true;
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
