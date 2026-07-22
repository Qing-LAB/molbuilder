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
 *   3. Route the generated XYZ through
 *      ``structurePage.loadIntoCanvas`` — that's where the dirty-
 *      canvas warning-modal fires.
 *   4. On canvas-accept, hand the XYZ to the viewer via
 *      ``window.molbuilder.loadStructureText``.
 *
 * Errors / cancellation surface in #name-status.
 *
 * Test seam: ``configure(opts)`` lets tests inject a fake fetch +
 * structurePage + viewer-loader so the Node-only unit tests can
 * drive the state machine without a real DOM or HTTP roundtrip.
 *
 * Design ref: docs/tabs/architecture.md § 5.1 (panel 5: name
 * lookup, part of the "others" generator group).
 */
import { data as mvData } from "/static/lib/molview/index.js";
// Sources-card loader adapter -> MolView's ONE door (data.installMolecule); no window.* global.
const _loadText = (text, filename) => mvData.installMolecule({ text: text, filename: filename });

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
        if (!_viewerLoader)
            _viewerLoader = _loadText;
    }

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
        if (!_fetch) {
            return Promise.reject(new Error(
                "name: fetch not configured"));
        }
        if (!_structurePage) {
            return Promise.reject(new Error(
                "name: structurePage not configured"));
        }
        var trimmed = name.trim();
        return _fetch(BUILD_URL, {
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
            return _structurePage.loadIntoCanvas(
                { source_format: "xyz", text: body.xyz },
                { kind: "name",
                  generator_input: { name: trimmed } }
            ).then(function (gate) {
                if (!gate.ok) {
                    return { ok: false, cancelled: true };
                }
                // loadIntoCanvas now routes through molview.data.openMolecule,
                // which parses + renders the structure itself.  The old
                // viewerLoader second load is removed — it would
                // double-apply the same bytes.
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

        function setStatus(msg, kind) {
            if (!status) return;
            status.textContent = msg || "";
            status.className = "muted"
                + (kind === "error"      ? " is-error"      : "")
                + (kind === "generating" ? " is-generating" : "");
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
            viewerLoader:  _loadText,
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
