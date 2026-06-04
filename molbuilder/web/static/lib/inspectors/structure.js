/* Structure-preview inspector: read-only 3-D view of a .xyz / .pdb
 * file.  Loads the file via ``ctx.readFile`` and hands the text +
 * format to the standard embedded MolViewer (lib/mol-viewer-embed.js,
 * contract: docs/protocols/embedded-viewer.md).
 *
 * No frame playback, no atom-pick, no force overlay -- that's the
 * trajectory inspector's job.  This is a static-structure peek.
 *
 * The inspector owns its OWN card (with the "Open in Modify" link
 * the user expects on /results); the embedded viewer mounts inside
 * that card with ``card.title: ""`` so the embed doesn't render a
 * second header — the host's card-header is the title strip.  The
 * ``card.bare`` opt was retired 2026-06-03 along with the
 * .mol-viewer-bare CSS class; ``card.title: ""`` is the canonical
 * way to skip the embed's header.
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
            modifyLink.href = "/modify";
            modifyLink.textContent = "Open in Modify";
            modifyLink.className = "inspector-card-link";
            modifyLink.title = (
                "Loads the structure into the Modify tab so you can "
                + "rotate / orient / add electrodes / etc."
            );
            actions.appendChild(modifyLink);
            header.appendChild(actions);
            card.appendChild(header);

            const status = document.createElement("p");
            status.className = "inspector-card-note structure-status";
            status.textContent = "Loading…";
            card.appendChild(status);

            // -- Slot the embedded viewer will mount into --------- //
            const viewerSlot = document.createElement("div");
            viewerSlot.className = "structure-viewer-slot";
            card.appendChild(viewerSlot);

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
                    axes:   true,
                    export: { defaultName: r.basename || "structure" },
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
                                    "molbuilder:inspector:ready",
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
