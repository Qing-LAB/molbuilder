/* panel-deps.js — the builder panels' dependency wiring, once.
 *
 * Five panels (smiles · name · peptide · rna · dna) each carried a
 * byte-identical `configure()` and `_lazyResolve()`, together with the same
 * eleven-line comment explaining the second one — about eighty lines of copy,
 * five places to edit, and five places for a fix to miss.
 *
 * WHAT THE COPIES WERE FOR, kept here because the reason is real:
 *
 *   * `configure()` is the test door.  A Node unit test drives the panel's
 *     state machine with a fake fetch and a fake structure page, without a
 *     DOM or an HTTP roundtrip.
 *   * `resolve()` is the LANDMINE-2 fix.  Each panel's IIFE used to capture
 *     `window.molbuilder.*` once at script-eval time, so a template that
 *     loaded a panel before `page.js` finished registering its globals left
 *     those slots null forever and the first generate() hit the
 *     "not configured" branch.  Re-reading on every call means a later
 *     script-load cannot silently degrade the panel.
 *
 * Values a test injected are never overwritten by the production lookup —
 * that is what makes the two doors coexist.
 *
 * Exports (on window.molbuilder.panelDeps): make(root) -> a per-panel slot.
 */
(function (root) {
    "use strict";

    function make(host) {
        var deps = { fetch: null, structurePage: null };

        return {
            /** The test door: explicit fakes win, and keep winning. */
            configure: function (opts) {
                opts = opts || {};
                if (opts.fetch) deps.fetch = opts.fetch;
                if (opts.structurePage) deps.structurePage = opts.structurePage;
            },
            /** The production door: re-read every call (LANDMINE-2). */
            resolve: function () {
                if (typeof host === "undefined" || !host || !host.molbuilder) {
                    return;
                }
                if (!deps.fetch && host.fetch) {
                    deps.fetch = host.fetch.bind(host);
                }
                if (!deps.structurePage && host.molbuilder.structurePage) {
                    deps.structurePage = host.molbuilder.structurePage;
                }
            },
            get fetch() { return deps.fetch; },
            get structurePage() { return deps.structurePage; },
        };
    }

    var api = { make: make };
    if (typeof module !== "undefined" && module.exports) {
        module.exports = api;
    }
    root.molbuilder = root.molbuilder || {};
    root.molbuilder.panelDeps = api;
})(typeof globalThis !== "undefined" ? globalThis : this);
