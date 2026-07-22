/* Peptide-builder generator panel — Molbuilder-tab Source.
 *
 * Wires the Sources card's "Generate peptide" panel.  Identical
 * shape to smiles.js + name.js; only the request kind ("peptide"
 * instead of "smiles" / "name") and the field IDs differ.  The
 * backend dispatches to ``build_peptide`` which expects a one-
 * letter amino-acid sequence (e.g. "ACDEF") and returns a 3-D
 * structure built via AmberTools' tleap (extended chain by
 * default).
 *
 * Flow:
 *   1. Read the sequence input; refuse empty / illegal codes.
 *   2. POST {kind: "peptide", input: <sequence>} to
 *      /api/build/molecule.
 *   3. Route through structurePage.loadIntoCanvas.
 *   4. Hand to ``loadStructureText`` for viewer render.
 *
 * Errors surface in #peptide-status (illegal codes server-side,
 * tleap failure, network drop).  Cancellation on the dirty-
 * canvas warning modal surfaces as "Kept existing workspace."
 *
 * Test seam: ``configure(opts)`` lets tests inject a fake
 * fetch + structurePage + viewer-loader.
 *
 * Design ref: docs/tabs/architecture.md § 5.1 (panel 4: peptide
 * builder).
 */
import { data as mvData } from "/static/lib/molview/index.js";
// Sources-card loader adapter -> MolView's ONE door (data.installMolecule); no window.* global.
const _loadText = (text, filename) => mvData.installMolecule({ text: text, filename: filename });

(function (root) {
    "use strict";

    var BUILD_URL = "/api/build/molecule";
    // One-letter amino-acid codes the backend's tleap path
    // recognises.  Used for client-side validation so the user
    // gets a clear inline error before the request hits the
    // network instead of waiting for tleap to fail.
    var VALID_AA = /^[ACDEFGHIKLMNPQRSTVWY]+$/i;

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
     * Generate a peptide from a one-letter amino-acid sequence.
     *
     * @param {string} sequence
     * @returns {Promise<{ok: boolean,
     *                    cancelled?: boolean,
     *                    error?: string,
     *                    n_atoms?: number}>}
     */
    function generate(sequence) {
        if (typeof sequence !== "string" || !sequence.trim()) {
            return Promise.resolve({
                ok: false,
                error: "Enter a peptide sequence first (one-letter codes).",
            });
        }
        var trimmed = sequence.trim().toUpperCase().replace(/\s+/g, "");
        if (!VALID_AA.test(trimmed)) {
            return Promise.resolve({
                ok:    false,
                error: "Sequence must use one-letter amino-acid codes "
                     + "(ACDEFGHIKLMNPQRSTVWY).  Got: "
                     + JSON.stringify(sequence),
            });
        }
        // Lazy-resolve dependencies in case the script-load
        // order put us above page.js / lib/* (LANDMINE-2 fix).
        _lazyResolve();
        if (!_fetch) {
            return Promise.reject(new Error(
                "peptide: fetch not configured"));
        }
        if (!_structurePage) {
            return Promise.reject(new Error(
                "peptide: structurePage not configured"));
        }
        return _fetch(BUILD_URL, {
            method:  "POST",
            headers: { "Content-Type": "application/json" },
            body:    JSON.stringify({
                kind: "peptide", input: trimmed,
            }),
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
                { kind: "peptide",
                  generator_input: { sequence: trimmed } }
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
        if (!doc || _wired) return;
        _wired = true;

        var input  = doc.getElementById("peptide-input");
        var button = doc.getElementById("peptide-generate-btn");
        var status = doc.getElementById("peptide-status");
        if (!input || !button) return;

        function setStatus(msg, kind) {
            if (!status) return;
            status.textContent = msg || "";
            status.className = "muted"
                + (kind === "error"      ? " is-error"      : "")
                + (kind === "generating" ? " is-generating" : "");
        }

        button.addEventListener("click", function () {
            var echo = input.value.trim().toUpperCase().replace(/\s+/g, "");
            button.disabled = true;
            setStatus("Generating peptide…", "generating");
            generate(echo).then(function (r) {
                button.disabled = false;
                if (r.ok) {
                    setStatus(
                        "Generated " + (r.n_atoms != null
                            ? r.n_atoms + " atoms" : "")
                        + " from sequence " + echo);
                } else if (r.cancelled) {
                    setStatus("Kept existing workspace.");
                } else {
                    setStatus(r.error || "Generation failed.",
                              "error");
                }
            });
        });

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
        VALID_AA:  VALID_AA,
    };

    if (typeof module !== "undefined" && module.exports) {
        module.exports = api;
    } else {
        root.molbuilder = root.molbuilder || {};
        root.molbuilder.structurePeptide = api;
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
                "structure.peptide", api);
        }
    }
})(typeof window !== "undefined" ? window : globalThis);
