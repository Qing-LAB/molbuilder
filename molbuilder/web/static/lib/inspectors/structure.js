/* Structure-preview inspector: read-only 3-D view of a .xyz / .pdb
 * file.  Mounts a 3Dmol viewer inside the host element, loads the
 * file via /api/files/read, and feeds it to the viewer in the
 * matching format.  Provides a small "Open in Modify" affordance
 * for the natural next step.
 *
 * No frame playback, no atom-pick, no force overlay -- that's the
 * trajectory inspector's job.  This is a static-structure peek.
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

            const viewerEl = document.createElement("div");
            viewerEl.className = "structure-viewer";
            // The 3Dmol viewer needs a non-zero size BEFORE init.
            // Inline style keeps the per-inspector concern in one
            // file (no CSS dependency for this small inspector).
            viewerEl.style.width  = "100%";
            viewerEl.style.height = "420px";
            card.appendChild(viewerEl);

            host.appendChild(card);

            let viewerInstance = null;
            let disposed = false;

            ctx.readFile(file).then((r) => {
                if (disposed) return;
                if (!r.ok) {
                    status.textContent = "Error: " + (r.error || "unknown");
                    status.classList.add("inspector-inline-error");
                    return;
                }
                const fmt = file.toLowerCase().endsWith(".pdb")
                    ? "pdb" : "xyz";
                try {
                    viewerInstance = root.molbuilder.viewer.create(viewerEl);
                    viewerInstance.addModel(r.text, fmt);
                    viewerInstance.setStyle({}, {
                        stick:  { radius: 0.18 },
                        sphere: { scale: 0.28 },
                    });
                    viewerInstance.zoomTo();
                    viewerInstance.render();
                    status.textContent = "Loaded " + _atomCount(r.text, fmt)
                                       + " atoms.";
                    // Signal "first render visible" so the /results
                    // tab-level picker drops its "Parsing…" status.
                    // Mirrors the trajectory inspector's dispatch.
                    try {
                        document.dispatchEvent(new CustomEvent(
                            "molbuilder:inspector:ready",
                            { detail: { inspector: "structure" } }
                        ));
                    } catch (_) { /* see core.js for context */ }
                } catch (e) {
                    status.textContent = "3Dmol failed: "
                                       + (e && e.message ? e.message : String(e));
                    status.classList.add("inspector-inline-error");
                }
            });

            return {
                dispose() {
                    disposed = true;
                    if (viewerInstance) {
                        try { viewerInstance.removeAllModels(); }
                        catch (_) { /* viewer already torn down */ }
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

    function _atomCount(text, fmt) {
        // Cheap heuristic so the user gets immediate feedback.
        // Best-effort: an unparseable file still shows the viewer.
        try {
            if (fmt === "xyz") {
                const first = text.split(/\r?\n/, 1)[0].trim();
                const n = parseInt(first, 10);
                return Number.isFinite(n) && n > 0 ? n : "?";
            }
            // PDB: count ATOM + HETATM lines.
            const lines = text.split(/\r?\n/);
            let n = 0;
            for (const ln of lines) {
                if (ln.startsWith("ATOM ") || ln.startsWith("HETATM")) n++;
            }
            return n > 0 ? n : "?";
        } catch (_) {
            return "?";
        }
    }

    root.molbuilder = root.molbuilder || {};
    root.molbuilder.inspectors = root.molbuilder.inspectors || {};
    root.molbuilder.inspectors.structureInspector = inspector;
    if (root.molbuilder.inspectors.register) {
        root.molbuilder.inspectors.register(inspector);
    }
})(typeof window !== "undefined" ? window : this);
