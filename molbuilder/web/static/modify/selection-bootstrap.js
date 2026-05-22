/* /modify atom-selection bootstrap -- page glue only.
 *
 * Responsibilities (and ONLY these):
 *
 *   1. Fetch the _selection_panel.html partial and insert it into
 *      the page's #selection-host.
 *   2. Mount the panel against the singleton store.
 *   3. Attach the viewer-adapter once modify/viewer.js has exposed
 *      its 3Dmol viewer.
 *   4. Forward the Projects sidebar's onChange (filtered to .xyz)
 *      into store.setSourceFile.
 *
 * That's it.  All HTTP wiring, rule translation, save dispatch and
 * label bookkeeping lives in the store.  See
 * docs/protocols/atom-selection.md for the full architecture.
 */
(function () {
    "use strict";

    const PARTIAL_URL = "/partials/selection-panel";
    const HOST_ID     = "selection-host";

    function _renderFailure(host, message) {
        // Surface the failure inline so the user doesn't stare at a
        // silently-blank column.  textContent keeps user-visible
        // strings (and the URL) out of the HTML parser.
        const banner = document.createElement("div");
        banner.className = "selection-bootstrap-error";
        banner.setAttribute("role", "alert");
        banner.textContent = "Selection panel failed to load: " + message;
        host.innerHTML = "";
        host.appendChild(banner);
    }

    async function bootstrap() {
        const host = document.getElementById(HOST_ID);
        if (!host) {
            console.warn("[selection-bootstrap] #" + HOST_ID
                       + " missing; skipping.");
            return;
        }

        // 1. Fetch + insert the partial.  Same-origin, autoescaped
        // Jinja render, no user data -> innerHTML is safe.
        let html;
        try {
            const r = await fetch(PARTIAL_URL);
            if (!r.ok) {
                console.error("[selection-bootstrap] partial fetch returned "
                            + r.status);
                _renderFailure(host, "HTTP " + r.status + " from " + PARTIAL_URL);
                return;
            }
            html = await r.text();
        } catch (e) {
            console.error("[selection-bootstrap] partial fetch threw", e);
            _renderFailure(host, (e && e.message) ? e.message : String(e));
            return;
        }
        host.innerHTML = html;

        // 2. Mount the panel.
        if (!window.molbuilder || !window.molbuilder.selectionPanel) {
            console.error("[selection-bootstrap] selectionPanel module missing");
            return;
        }
        window.molbuilder.selectionPanel.mount(host);

        // 2b. Inject the viewer-specific XYZ loader into the store so
        // the store doesn't reach into ``window.molbuilder`` to do
        // its own file-load (spec §5 rule 3: the store has no DOM
        // and no 3Dmol; the page wires those in).  modify/viewer.js
        // exposes ``window.molbuilder.loadStructureText`` once its
        // DOMContentLoaded runs; if missing, the store falls back
        // to atom-list-only mode -- still functional, just no
        // viewer.
        const _store0 = window.molbuilder.selection
                      && window.molbuilder.selection.store;
        if (_store0 && typeof _store0.setLoader === "function"
                    && window.molbuilder.loadStructureText) {
            _store0.setLoader(window.molbuilder.loadStructureText);
        }

        // 3. Forward sidebar selection changes into the store.  The
        // store's setSourceFile is the single entry for "switch
        // structure" -- it loads the XYZ into the viewer + refetches
        // the atom list, atomically.  Selection clears on a new
        // file; filter drafts persist (the user must explicitly
        // ``applyFilter()`` against the new structure).
        const projects = window.molbuilder && window.molbuilder.projects;
        const store    = window.molbuilder.selection.store;
        // Both .xyz and .pdb are loadable into /modify -- the
        // server's selection blueprint dispatches by extension
        // (see web/blueprints/selection.py
        // ``_SUPPORTED_STRUCTURE_SUFFIXES``).  A pick of .log /
        // .fdf etc. is "view this file" not "swap the /modify
        // structure" -- those filter out so the user's atom
        // selection isn't silently wiped when they peek at a log.
        function _isLoadableStructure(name) {
            const lc = String(name || "").toLowerCase();
            return lc.endsWith(".xyz") || lc.endsWith(".pdb");
        }
        if (projects) {
            // Seed from whatever the sidebar already has selected.
            const initial = projects.getCurrentFile() || "";
            if (_isLoadableStructure(initial)) {
                store.setSourceFile(initial);
            }
            projects.onChange((sel) => {
                const f = (sel && sel.file) ? sel.file : "";
                if (_isLoadableStructure(f)) {
                    store.setSourceFile(f);
                }
            });
        }

        // 4. Attach the viewer-adapter once modify/viewer.js has
        // exposed its viewer.  Both this bootstrap and viewer.js
        // run on DOMContentLoaded; in practice viewer.js executes
        // first (script order in modify.html), but we poll briefly
        // to be robust against future ordering changes.
        const adapterModule =
            (window.molbuilder.selection && window.molbuilder.selection.viewerAdapter)
                ? window.molbuilder.selection.viewerAdapter : null;
        if (!adapterModule) {
            console.warn("[selection-bootstrap] viewerAdapter module missing");
            return;
        }
        let tries = 0;
        const timer = setInterval(() => {
            const v = (window.molbuilder.modify
                       && window.molbuilder.modify.viewer)
                      ? window.molbuilder.modify.viewer : null;
            if (v) {
                adapterModule.attach(v);
                clearInterval(timer);
            } else if (++tries >= 10) {
                clearInterval(timer);
                console.warn(
                    "[selection-bootstrap] viewer never appeared; "
                    + "click integration disabled"
                );
            }
        }, 100);
    }

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", bootstrap);
    } else {
        bootstrap();
    }
})();
