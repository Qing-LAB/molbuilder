/* panel-deps.js — the builder panels' dependency wiring, once, for the five
 * panels (smiles · name · peptide · rna · dna).
 *
 *   * `configure()` is the test door.  A Node unit test drives the panel's
 *     state machine with a fake fetch and a fake structure page, without a
 *     DOM or an HTTP roundtrip.
 *   * `resolve()` re-reads `window.molbuilder.*` on every call, so a
 *     template that loads a panel before `page.js` registers its globals
 *     still finds them: a later script-load cannot silently degrade the panel.
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
            /** The production door: re-read every call. */
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
