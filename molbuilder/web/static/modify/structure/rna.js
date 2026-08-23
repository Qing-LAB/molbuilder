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

    // The panel's dependency slots, wired once for all five panels
    // (`panel-deps.js`): `configure` is the test door, `_lazyResolve` the
    // production re-read that LANDMINE-2 needs.  This was eighty lines of
    // byte-identical copy across smiles/name/peptide/rna/dna.
    var _deps = root.molbuilder.panelDeps.make(root);
    var configure = _deps.configure;
    var _lazyResolve = _deps.resolve;

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
        if (!_deps.fetch) {
            return Promise.reject(new Error(
                "rna: fetch not configured"));
        }
        if (!_deps.structurePage) {
            return Promise.reject(new Error(
                "rna: structurePage not configured"));
        }
        return _deps.fetch(BUILD_URL, {
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
            return _deps.structurePage.loadIntoCanvas(
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
