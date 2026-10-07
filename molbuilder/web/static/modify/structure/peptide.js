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
 *   3. Route through ``structurePage.loadIntoCanvas``, which installs
 *      into an empty viewer or appends to the open structure.
 *
 * Errors surface in #peptide-status (illegal codes server-side,
 * tleap failure, network drop).  A load the gate did not apply
 * surfaces as "Kept existing workspace."
 *
 * Test seam: ``configure(opts)`` lets tests inject a fake
 * fetch + structurePage.
 *
 * Design ref: docs/web/tabs.md § 2 (Creating a structure — the in-gate).
 */

(function (root) {
    "use strict";

    var BUILD_URL = "/api/build/molecule";
    // One-letter amino-acid codes the backend's tleap path
    // recognises.  Used for client-side validation so the user
    // gets a clear inline error before the request hits the
    // network instead of waiting for tleap to fail.
    var VALID_AA = /^[ACDEFGHIKLMNPQRSTVWY]+$/i;

    // The panel's dependency slots, wired once for all five panels
    // (`panel-deps.js`): `configure` is the test door, `_lazyResolve` the
    // production re-read.
    var _deps = root.molbuilder.panelDeps.make(root);
    var configure = _deps.configure;
    var _lazyResolve = _deps.resolve;

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
        // order put us above page.js / lib/*.
        _lazyResolve();
        if (!_deps.fetch) {
            return Promise.reject(new Error(
                "peptide: fetch not configured"));
        }
        if (!_deps.structurePage) {
            return Promise.reject(new Error(
                "peptide: structurePage not configured"));
        }
        return _deps.fetch(BUILD_URL, {
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
            return _deps.structurePage.loadIntoCanvas(
                { structure: body.structure },
                { kind: "peptide",
                  generator_input: { sequence: trimmed } }
            ).then(function (gate) {
                if (!gate.ok) {
                    return { ok: false, cancelled: true };
                }
                // loadIntoCanvas installs into an empty viewer or appends to
                // the open structure (the MODEL door; the FILE door is
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
        if (!doc || _wired) return;
        _wired = true;

        var input  = doc.getElementById("peptide-input");
        var button = doc.getElementById("peptide-generate-btn");
        var status = doc.getElementById("peptide-status");
        if (!input || !button) return;

        /* The shared `.status` writer (lib/status.js): `.status` is the
         * app's one severity surface and its `error` IS red; the busy state
         * is the neutral line. */
        function setStatus(msg, kind) {
            window.molbuilder.status.set(
                status, msg, kind === "error" ? "error" : null);
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
