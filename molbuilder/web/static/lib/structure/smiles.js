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
 *   3. Route the generated XYZ through
 *      ``structurePage.loadIntoCanvas`` — that's where the dirty-
 *      canvas warning-modal fires, so a user who's mid-edit doesn't
 *      lose work to a stray Generate click.
 *   4. On canvas-accept, hand the XYZ to the viewer via
 *      ``window.molbuilder.loadStructureText`` so 3Dmol actually
 *      renders the molecule.
 *
 * Errors / cancellation surface in #smiles-status (network drop,
 * 4xx from RDKit, user-cancel on the warning modal).
 *
 * Test seam: ``configure(opts)`` lets tests inject a fake fetch +
 * structurePage + viewer-loader so the Node-only unit tests can
 * drive the state machine without a real DOM or HTTP roundtrip.
 *
 * Design ref: docs/tabs/architecture.md § 5.1 (panel 2: SMILES
 * generator) + § 5.4 (warning-modal gates Load + Generate).
 */
(function (root) {
    "use strict";

    var BUILD_URL = "/api/build/molecule";

    // Injected at bind() time (production) or via configure() (tests).
    var _fetch         = null;
    var _structurePage = null;
    var _viewerLoader  = null;

    function configure(opts) {
        opts = opts || {};
        if (opts.fetch)         _fetch         = opts.fetch;
        if (opts.structurePage) _structurePage = opts.structurePage;
        if (opts.viewerLoader)  _viewerLoader  = opts.viewerLoader;
    }

    /**
     * Lazy-resolve production singletons from window.molbuilder.
     * Pre-fix this module's IIFE captured them once at script-eval
     * time (LANDMINE-2): if a future template change loaded this
     * script BEFORE page.js / mol-viewer.js finished
     * registering their globals, those slots stayed null and the
     * first generate() call hit the "not configured" branch.
     * Re-reads on every call so a later script-load doesn't
     * silently degrade.  Test contexts that called configure()
     * with explicit fakes are unaffected (their values stay).
     */
    function _lazyResolve() {
        if (typeof root === "undefined" || !root.molbuilder) return;
        if (!_fetch && root.fetch)
            _fetch = root.fetch.bind(root);
        if (!_structurePage && root.molbuilder.structurePage)
            _structurePage = root.molbuilder.structurePage;
        if (!_viewerLoader && root.molbuilder.loadStructureText)
            _viewerLoader = root.molbuilder.loadStructureText;
    }

    /**
     * Generate a structure from ``smiles`` and route it through the
     * canvas-state gate.
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
        // order put us above page.js / lib/* (LANDMINE-2 fix).
        _lazyResolve();
        if (!_fetch) {
            return Promise.reject(new Error(
                "smiles: fetch not configured"));
        }
        if (!_structurePage) {
            return Promise.reject(new Error(
                "smiles: structurePage not configured"));
        }
        var trimmed = smiles.trim();
        return _fetch(BUILD_URL, {
            method:  "POST",
            headers: { "Content-Type": "application/json" },
            body:    JSON.stringify({ kind: "smiles", input: trimmed }),
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
            // Hand off to the canvas-state gate.  This fires the
            // warning modal if the canvas is dirty.
            return _structurePage.loadIntoCanvas(
                { source_format: "xyz", text: body.xyz },
                { kind: "smiles",
                  generator_input: { smiles: trimmed } }
            ).then(function (gate) {
                if (!gate.ok) {
                    // Cancelled — leave the viewer alone.
                    return { ok: false, cancelled: true };
                }
                // Render in 3Dmol via the existing viewer loader.
                // The loader is optional in test contexts.
                if (typeof _viewerLoader === "function") {
                    var fname = "smiles-" + trimmed.replace(
                        /[^A-Za-z0-9_-]/g, "_") + ".xyz";
                    try {
                        var maybe = _viewerLoader(body.xyz, fname);
                        if (maybe && typeof maybe.then === "function") {
                            return maybe.then(function () {
                                return { ok: true,
                                         n_atoms: body.n_atoms };
                            });
                        }
                    } catch (e) {
                        return {
                            ok:    false,
                            error: "Viewer failed to render: "
                                 + (e && e.message ? e.message
                                                   : String(e)),
                        };
                    }
                }
                return { ok: true, n_atoms: body.n_atoms };
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

        function setStatus(msg, kind) {
            if (!status) return;
            status.textContent = msg || "";
            // Two states: generating (in-flight) or error.  Empty
            // string clears both.
            status.className = "muted"
                + (kind === "error"      ? " is-error"      : "")
                + (kind === "generating" ? " is-generating" : "");
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
                        + " from " + echo);
                } else if (r.cancelled) {
                    // User clicked Cancel on the warning modal —
                    // tell them their workspace is untouched so
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
            viewerLoader:  root.molbuilder.loadStructureText,
        });
        // Wire the panel on DOMContentLoaded — the orchestrator's
        // auto-bind ran via canvas-state + warning-modal loading
        // first; by now structurePage is ready to receive calls.
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
