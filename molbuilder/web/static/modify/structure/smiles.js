/* SMILES generator panel — Molbuilder-tab Source.
 *
 * Wires the Sources card's "Generate from SMILES" panel:
 *
 *   #smiles-input            -- the user's SMILES string
 *   #smiles-generate-btn     -- click → POST to /api/build/molecule
 *   #smiles-status           -- inline progress / error readout
 *
 * The flow:
 *
 *   1. Read the SMILES input; refuse empty.
 *   2. POST {kind: "smiles", input: <smiles>} to /api/build/molecule.
 *      RDKit on the server returns {ok, xyz, n_atoms, ...}.
 *   3. Route the generated structure through ``structurePage.loadIntoCanvas``,
 *      which installs it into an empty viewer or appends it to the open one.
 *
 * Errors / cancellation surface in #smiles-status (network drop,
 * 4xx from RDKit).
 *
 * Test seam: ``configure(opts)`` lets tests inject a fake fetch +
 * structurePage so the Node-only unit tests can
 * drive the state machine without a real DOM or HTTP roundtrip.
 *
 * Design ref: docs/web/tabs.md § 2 (Creating a structure — the in-gate).
 */

(function (root) {
    "use strict";

    var BUILD_URL = root.molbuilder.panelDeps.BUILD_URL;

    // The panel's dependency slots, wired once for all five panels
    // (`panel-deps.js`): `configure` is the test door; the chain resolves
    // and checks them at each call.
    var _deps = root.molbuilder.panelDeps.make(root);
    var configure = _deps.configure;


    /**
     * Generate a structure from ``smiles`` and route it through the
     * page's load gate.
     *
     * @param {string} smiles
     * @returns {Promise<{ok: boolean,
     *                    cancelled?: boolean,
     *                    error?: string,
     *                    n_atoms?: number}>}
     */
    function generate(smiles) {
        return _deps.generateText(smiles, "smiles",
            "Enter a SMILES string first.",
            "Generating 3-D structure from SMILES\u2026");
    }

    /**
     * Wire the Sources-card SMILES panel: the input, Generate
     * button, and status readout.  Idempotent — calling twice is
     * a no-op on the second call.
     *
     * @param {object} [opts]
     * @param {Document} [opts.doc]   - the document to query (test seam)
     */
    var _wired = false;
    function wirePanel(opts) {
        opts = opts || {};
        var doc = opts.doc || root.document;
        if (!doc) return;
        if (_wired) return;
        _wired = true;

        var input  = doc.getElementById("smiles-input");
        var button = doc.getElementById("smiles-generate-btn");
        var status = doc.getElementById("smiles-status");
        if (!input || !button) return;

        /* The shared `.status` writer (lib/status.js): `.status` is the
         * app's one severity surface and its `error` IS red; the busy state
         * is the neutral line. */
        function setStatus(msg, kind) {
            window.molbuilder.status.set(
                status, msg, kind === "error" ? "error" : null);
        }

        button.addEventListener("click", function () {
            // Capture the SMILES at click time so the success
            // status reports what was BUILT, not whatever the user
            // typed while the request was in flight.
            var echo = input.value.trim();
            button.disabled = true;
            setStatus("Generating…", "generating");
            generate(echo).then(function (r) {
                button.disabled = false;
                if (r.ok) {
                    setStatus(
                        "Generated " + (r.n_atoms != null
                            ? r.n_atoms + " atoms" : "")
                        + " from " + echo
                        // Provenance: name the engine that built the geometry so
                        // the user knows when they're on the lower-fidelity
                        // OpenBabel fallback (RDKit-first, OpenBabel-fallback).
                        + (r.backend_used ? " · " + r.backend_used : ""));
                } else if (r.cancelled) {
                    // Not applied — tell them their workspace is untouched so
                    // they don't think Generate silently failed.
                    setStatus("Kept existing workspace.");
                } else {
                    setStatus(r.error || "Generation failed.",
                              "error");
                }
            });
        });

        // Enter inside the input triggers Generate too — keyboard
        // users shouldn't have to mouse over to the button.
        input.addEventListener("keydown", function (ev) {
            if (ev.key === "Enter" && !button.disabled) {
                ev.preventDefault();
                button.click();
            }
        });
    }

    var api = {
        configure: configure,
        generate:  generate,
        wirePanel: wirePanel,
        BUILD_URL: BUILD_URL,
    };

    if (typeof module !== "undefined" && module.exports) {
        module.exports = api;
    } else {
        root.molbuilder = root.molbuilder || {};
        root.molbuilder.structureSmiles = api;
        // Auto-configure against the production singletons.
        configure({
            fetch:         root.fetch
                            ? root.fetch.bind(root)
                            : undefined,
            structurePage: root.molbuilder.structurePage,
        });
        // Wire the panel on DOMContentLoaded.
        if (root.document) {
            if (root.document.readyState === "loading") {
                root.document.addEventListener(
                    "DOMContentLoaded", function () { wirePanel(); });
            } else {
                wirePanel();
            }
        }
        if (root.molbuilder.runtime
            && typeof root.molbuilder.runtime.register === "function") {
            root.molbuilder.runtime.register(
                "structure.smiles", api);
        }
    }
})(typeof window !== "undefined" ? window : globalThis);
