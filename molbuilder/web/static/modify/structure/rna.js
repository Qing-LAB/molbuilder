/* RNA-builder generator panel — Molbuilder-tab Source.
 *
 * Wires the Sources card's "Generate RNA" panel.  POSTs
 * ``{kind: "rna", input: <sequence>, form: <A|B|Z>}`` to
 * /api/build/molecule.  Backend dispatches to ``build_rna`` which
 * picks the best installed backend (3DNA preferred, AmberTools
 * fallback, RDKit last) for an ssRNA from a 1-letter sequence.
 *
 * Differences from dna.js:
 *   * Alphabet — ACGU (uracil) instead of ACGT (thymine).
 *   * Default form — A (canonical right-handed RNA helix)
 *     instead of B.
 *
 * Everything else (flow shape, status messages, cancel
 * semantics) is identical to dna.js / smiles.js / name.js /
 * peptide.js.
 *
 * Design ref: docs/web/tabs.md (panel 4: 3DNA
 * helix builder — RNA variant).
 */

(function (root) {
    "use strict";

    var BUILD_URL = "/api/build/molecule";
    var VALID_RNA = /^[ACGU]+$/i;
    var VALID_FORMS = ["A", "B", "Z"];
    var VALID_BACKENDS = ["auto", "threedna", "amber", "rdkit"];

    var _fetch         = null;
    var _structurePage = null;

    function configure(opts) {
        opts = opts || {};
        if (opts.fetch)         _fetch         = opts.fetch;
        if (opts.structurePage) _structurePage = opts.structurePage;
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
    }

    function generate(sequence, opts) {
        opts = opts || {};
        if (typeof sequence !== "string" || !sequence.trim()) {
            return Promise.resolve({
                ok: false,
                error: "Enter an RNA sequence first (ACGU).",
            });
        }
        var trimmed = sequence.trim().toUpperCase().replace(/\s+/g, "");
        if (!VALID_RNA.test(trimmed)) {
            return Promise.resolve({
                ok:    false,
                error: "Sequence must use ACGU only.  Got: "
                     + JSON.stringify(sequence),
            });
        }
        var form = opts.form || "A";
        if (VALID_FORMS.indexOf(form) < 0) {
            return Promise.resolve({
                ok:    false,
                error: "Helix form must be one of "
                     + VALID_FORMS.join(", ") + ".  Got: "
                     + JSON.stringify(form),
            });
        }
        var backend = opts.backend || "auto";
        if (VALID_BACKENDS.indexOf(backend) < 0) {
            return Promise.resolve({
                ok:    false,
                error: "Backend must be one of "
                     + VALID_BACKENDS.join(", ") + ".  Got: "
                     + JSON.stringify(backend),
            });
        }
        // Lazy-resolve dependencies in case the script-load
        // order put us above page.js / lib/* (LANDMINE-2 fix).
        _lazyResolve();
        if (!_fetch) {
            return Promise.reject(new Error(
                "rna: fetch not configured"));
        }
        if (!_structurePage) {
            return Promise.reject(new Error(
                "rna: structurePage not configured"));
        }
        return _fetch(BUILD_URL, {
            method:  "POST",
            headers: { "Content-Type": "application/json" },
            body:    JSON.stringify({
                kind:    "rna",
                input:   trimmed,
                form:    form,
                backend: backend,
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
                { kind: "rna",
                  generator_input: {
                      sequence: trimmed, form: form,
                      backend: backend,
                  } }
            ).then(function (gate) {
                if (!gate.ok) {
                    return { ok: false, cancelled: true };
                }
                // loadIntoCanvas routes through molview.data.installMolecule
                // (the MODEL primitive for generated text; the FILE door is
                // projects.parser.openMolecule -- not used here).
                return { ok: true, n_atoms: body.n_atoms,
                         backend_used: body.backend_used };
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

        var input   = doc.getElementById("rna-input");
        var formSel = doc.getElementById("rna-form-select");
        var backendSel = doc.getElementById("rna-backend-select");
        var button  = doc.getElementById("rna-generate-btn");
        var status  = doc.getElementById("rna-status");
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
            var formChoice = (formSel && formSel.value) || "A";
            var backendChoice = (backendSel && backendSel.value) || "auto";
            button.disabled = true;
            setStatus("Generating RNA…", "generating");
            generate(echo, {
                form: formChoice, backend: backendChoice,
            }).then(function (r) {
                button.disabled = false;
                if (r.ok) {
                    var backendNote = r.backend_used
                        && r.backend_used !== backendChoice
                        ? " (" + r.backend_used + ")"
                        : "";
                    setStatus(
                        "Generated " + (r.n_atoms != null
                            ? r.n_atoms + " atoms" : "")
                        + " from " + formChoice + "-form "
                        + echo + backendNote);
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
        configure:   configure,
        generate:    generate,
        wirePanel:   wirePanel,
        BUILD_URL:   BUILD_URL,
        VALID_RNA:   VALID_RNA,
        VALID_FORMS: VALID_FORMS,
    };

    if (typeof module !== "undefined" && module.exports) {
        module.exports = api;
    } else {
        root.molbuilder = root.molbuilder || {};
        root.molbuilder.structureRna = api;
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
                "structure.rna", api);
        }
    }
})(typeof window !== "undefined" ? window : globalThis);
