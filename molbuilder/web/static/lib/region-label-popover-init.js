/* Region-label-definitions popover initialiser.
 *
 * Bridges the library at lib/region-label-definitions.js (which
 * exposes renderPopover / init via window.molbuilder) to the
 * workspace's current structure on DOMContentLoaded.
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
                var ws = ns && ns.workspace;
                // Use the dedicated accessor: getStructure() carries no `regions` field
                // (its `s.regions` was always undefined -> the popover always showed "no
                // labels").  getRegions() is the single per-atom-label gatherer.
                var regions = (ws && typeof ws.getRegions === "function")
                    ? ws.getRegions() : {};
                return new Set(Object.keys(regions || {}));
            } catch (_err) {
                // Workspace not yet up, or getStructure threw —
                // popover degrades to "no labels present" which is
                // strictly better than crashing.
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
