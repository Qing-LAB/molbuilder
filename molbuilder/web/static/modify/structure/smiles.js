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

    var BUILD_URL = "/api/build/molecule";

    // The panel's dependency slots, wired once for all five panels
    // (`panel-deps.js`): `configure` is the test door, `_lazyResolve` the
    // production re-read.
    var _deps = root.molbuilder.panelDeps.make(root);
    var configure = _deps.configure;
    var _lazyResolve = _deps.resolve;


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
        if (typeof smiles !== "string" || !smiles.trim()) {
            return Promise.resolve({
                ok: false, error: "Enter a SMILES string first." });
        }
        // Lazy-resolve dependencies in case the script-load
        // order put us above page.js / lib/*.
        _lazyResolve();
        if (!_deps.fetch) {
            return Promise.reject(new Error(
                "smiles: fetch not configured"));
        }
        if (!_deps.structurePage) {
            return Promise.reject(new Error(
                "smiles: structurePage not configured"));
        }
        var trimmed = smiles.trim();
        // Page busy fence (ui-contract.md § 10): the embed is real
        // server work, and while it runs, a tab switch or sidebar
        // click would abandon or retarget the result.  Cover the
        // window; Cancel aborts the request; the finally releases.
        var busy = (root.molbuilder && root.molbuilder.pageBusy) || null;
        var ctl  = (typeof AbortController === "function")
                       ? new AbortController() : null;
        if (busy) {
            busy.claim("Generating 3-D structure from SMILES…",
                       ctl ? [function () { ctl.abort(); }] : []);
        }
        return _deps.fetch(BUILD_URL, {
            method:  "POST",
            headers: { "Content-Type": "application/json" },
            body:    JSON.stringify({ kind: "smiles", input: trimmed }),
            signal:  ctl ? ctl.signal : undefined,
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
            // Hand off to the page's load gate.
            return _deps.structurePage.loadIntoCanvas(
                { structure: body.structure },
                { kind: "smiles",
                  generator_input: { smiles: trimmed } }
            ).then(function (gate) {
                if (!gate.ok) {
                    // Not applied — leave the workspace alone.
                    return { ok: false, cancelled: true };
                }
                // loadIntoCanvas installs into an empty viewer or appends to
                // the open structure (the MODEL door; the FILE door is
                // projects.parser.openMolecule -- not used here).
                return { ok: true, n_atoms: body.n_atoms,
                         backend_used: body.backend_used };
            });
        })
        .catch(function (err) {
            // Network drop / JSON parse failure — surface as a
            // single error envelope so the UI doesn't need to
            // branch on exception types.
            return {
                ok:    false,
                error: "Could not reach " + BUILD_URL + ": "
                     + (err && err.message ? err.message
                                            : String(err)),
            };
        })
        .finally(function () {
            // The recovery contract (ui-contract.md § 10): the fence releases on
            // every path -- success, refusal, cancel, network drop.
            if (busy) busy.release();
        });
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
