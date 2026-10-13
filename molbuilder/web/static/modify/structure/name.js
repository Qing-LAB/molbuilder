/* Generate-from-name panel — Molbuilder-tab Source.
 *
 * Wires the Sources card's "Generate from name" panel.  Identical
 * shape to ``smiles.js``; the only differences are the input
 * field id (`#name-input`), the button id
 * (`#name-generate-btn`), the status id (`#name-status`), and
 * the request body's ``kind`` ("name" instead of "smiles").  The
 * backend dispatches to ``build_from_name`` which hits PubChem
 * for the IUPAC / common name.
 *
 * Flow:
 *   1. Read the name input; refuse empty.
 *   2. POST {kind: "name", input: <name>} to /api/build/molecule.
 *   3. Route the generated structure through ``structurePage.loadIntoCanvas``,
 *      which installs it into an empty viewer or appends it to the open one.
 *
 * Errors / cancellation surface in #name-status.
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
     * Generate a structure from ``name`` (IUPAC / common /
     * trade) and route it through the page's load gate.
     *
     * @param {string} name
     * @returns {Promise<{ok: boolean,
     *                    cancelled?: boolean,
     *                    error?: string,
     *                    n_atoms?: number}>}
     */
    function generate(name) {
        return _deps.generateText(name, "name", "Enter a name first.",
            "Generating the structure from its name\u2026");
    }

    var _wired = false;
    function wirePanel(opts) {
        opts = opts || {};
        var doc = opts.doc || root.document;
        if (!doc) return;
        if (_wired) return;
        _wired = true;

        var input  = doc.getElementById("name-input");
        var button = doc.getElementById("name-generate-btn");
        var status = doc.getElementById("name-status");
        if (!input || !button) return;

        /* The shared `.status` writer (lib/status.js): `.status` is the
         * app's one severity surface and its `error` IS red; the busy state
         * is the neutral line. */
        function setStatus(msg, kind) {
            window.molbuilder.status.set(
                status, msg, kind === "error" ? "error" : null);
        }

        button.addEventListener("click", function () {
            // Capture the name at click time so the success status
            // reports what was BUILT, not whatever the user typed
            // while the request was in flight.
            var echo = input.value.trim();
            button.disabled = true;
            setStatus("Generating…", "generating");
            generate(echo).then(function (r) {
                button.disabled = false;
                if (r.ok) {
                    setStatus(
                        "Generated " + (r.n_atoms != null
                            ? r.n_atoms + " atoms" : "")
                        + " from " + echo);
                } else if (r.cancelled) {
                    setStatus("Kept existing workspace.");
                } else {
                    setStatus(r.error || "Generation failed.",
                              "error");
                }
            });
        });

        // Enter inside the input triggers Generate too.
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
        root.molbuilder.structureName = api;
        configure({
            fetch:         root.fetch
                            ? root.fetch.bind(root)
                            : undefined,
            structurePage: root.molbuilder.structurePage,
        });
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
                "structure.name", api);
        }
    }
})(typeof window !== "undefined" ? window : globalThis);
