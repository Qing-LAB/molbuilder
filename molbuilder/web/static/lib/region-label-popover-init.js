/* Region-label-definitions popover initialiser.
 *
 * Bridges the library at lib/region-label-definitions.js (which
 * exposes renderPopover / init via window.molbuilder) to MolView's
 * current structure (molview.data) on DOMContentLoaded.
 *
 * Lives in its own file so the modify.html template stays free
 * of inline <script> blocks — those are silently blocked by the
 * site's CSP (`script-src 'self'`) and the wiring would never
 * run in production.  Caught by tests/test_no_inline_scripts.py
 * 2026-06-19 round-4 sweep.
 *
 * Robustness:
 *  * readyState check — if this script is loaded AFTER
 *    DOMContentLoaded has already fired (dynamic injection,
 *    SPA late-load), init runs immediately instead of never.
 *  * try/catch around getStructure() — a workspace not yet
 *    initialised (or in any unexpected state) returns an
 *    empty Set instead of throwing through the closure.
 */
"use strict";

(function () {
    function wire() {
        var ns = window.molbuilder;
        var defs = ns && ns.regionLabelDefinitions;
        if (!defs || typeof defs.init !== "function") return;
        defs.init(function () {
            try {
                // The per-atom region-label map lives on MolView's data model
                // (molview.data.getRegions), NOT the workspace -- the in-memory data
                // model was carved out of the workspace (tasks #41/#42); the workspace
                // is persistence-only and never had getRegions, so the old
                // `ns.workspace.getRegions` was always undefined and the popover always
                // showed "no labels".  getRegions() is the single per-atom-label gatherer.
                var data = ns && ns.molview && ns.molview.data;
                var regions = (data && typeof data.getRegions === "function")
                    ? data.getRegions() : {};
                return new Set(Object.keys(regions || {}));
            } catch (_err) {
                // Data model not yet up (or in any unexpected state) — the popover
                // degrades to "no labels present", strictly better than crashing.
                return new Set();
            }
        });
    }

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", wire);
    } else {
        // DOMContentLoaded already fired (script loaded late);
        // run on the current tick so callers don't see init
        // run during their own setup.
        setTimeout(wire, 0);
    }
})();
