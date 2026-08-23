/* Generate-from-name panel — Molbuilder-tab Source.
 *
 * Wires the Sources card's "Generate from name" panel.  Identical
 * shape to ``smiles.js``; the only differences are the input
 * field id (`#name-input`), the button id
 * (`#name-generate-btn`), the status id (`#name-status`), and
 * the request body's ``kind`` ("name" instead of "smiles").  The
 * backend dispatches to ``build_from_name`` which hits PubChem
 * (or local lookup) for the IUPAC / common name.
 *
 * Flow:
 *   1. Read the name input; refuse empty.
 *   2. POST {kind: "name", input: <name>} to /api/build/molecule.
 *   3. Route the generated XYZ through ``structurePage.loadIntoCanvas``
 *      — the dirty-canvas warning-modal fires there, and on accept it
 *      installs + renders via the MolView door (``molview.data.installMolecule``).
 *
 * Errors / cancellation surface in #name-status.
 *
 * Test seam: ``configure(opts)`` lets tests inject a fake fetch +
 * structurePage + viewer-loader so the Node-only unit tests can
 * drive the state machine without a real DOM or HTTP roundtrip.
 *
 * Design ref: docs/web/tabs.md (panel 5: name
 * lookup, part of the "others" generator group).
 */

(function (root) {
    "use strict";

    var BUILD_URL = "/api/build/molecule";

    // The panel's dependency slots, wired once for all five panels
    // (`panel-deps.js`): `configure` is the test door, `_lazyResolve` the
    // production re-read that LANDMINE-2 needs.  This was eighty lines of
    // byte-identical copy across smiles/name/peptide/rna/dna.
    var _deps = root.molbuilder.panelDeps.make(root);
    var configure = _deps.configure;
    var _lazyResolve = _deps.resolve;


    /**
     * Generate a structure from ``name`` (IUPAC / common /
     * trade) and route it through the canvas-state gate.
     *
     * @param {string} name
     * @returns {Promise<{ok: boolean,
     *                    cancelled?: boolean,
     *                    error?: string,
     *                    n_atoms?: number}>}
     */
    function generate(name) {
        if (typeof name !== "string" || !name.trim()) {
            return Promise.resolve({
                ok: false, error: "Enter a name first." });
        }
        // Lazy-resolve dependencies in case the script-load
        // order put us above page.js / lib/* (LANDMINE-2 fix).
        _lazyResolve();
        if (!_deps.fetch) {
            return Promise.reject(new Error(
                "name: fetch not configured"));
        }
        if (!_deps.structurePage) {
            return Promise.reject(new Error(
                "name: structurePage not configured"));
        }
        var trimmed = name.trim();
        return _deps.fetch(BUILD_URL, {
            method:  "POST",
            headers: { "Content-Type": "application/json" },
            body:    JSON.stringify({ kind: "name", input: trimmed }),
        })
        .then(function (r) {
            return r.json().then(function (body) {
                return { httpOk: r.ok, body: body };
            });
        })
        .then(function (env) {
            var body = env.body || {};
            if (!env.httpOk || !body.ok) {
                return {
                    ok:    false,
                    error: body.error
                            || ("HTTP error from " + BUILD_URL),
                };
            }
            return _deps.structurePage.loadIntoCanvas(
                { source_format: "xyz", text: body.xyz },
                { kind: "name",
                  generator_input: { name: trimmed } }
            ).then(function (gate) {
                if (!gate.ok) {
                    return { ok: false, cancelled: true };
                }
                // loadIntoCanvas routes through molview.data.installMolecule
                // (the MODEL primitive for generated text; the FILE door is
                // projects.parser.openMolecule -- not used here).
                return { ok: true, n_atoms: body.n_atoms };
            });
        })
        .catch(function (err) {
            return {
                ok:    false,
                error: "Could not reach " + BUILD_URL + ": "
                     + (err && err.message ? err.message
                                            : String(err)),
            };
        });
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

        /* The shared `.status` writer (lib/status.js).
         *
         * These seven panels each spelled this out, writing `.muted` with
         * `is-error` / `is-generating` / `is-loading` -- modifiers NO
         * stylesheet defined.  So a refused SMILES reported itself in the
         * same muted grey as a hint, on every builder panel, and had done
         * since they were written.  `.status` is the app's one severity
         * surface and its `error` IS red.
         *
         * The busy state maps to the neutral line: it had no appearance
         * before either (its class answered nothing), so this is the same
         * rendering with one fewer class that means nothing. */
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
