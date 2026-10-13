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
 * Design ref: docs/web/tabs.md § 2 (Creating a structure — the in-gate).
 */

(function (root) {
    "use strict";

    var BUILD_URL = root.molbuilder.panelDeps.BUILD_URL;
    var VALID_RNA = /^[ACGU]+$/i;
    var VALID_FORMS = ["A", "B", "Z"];
    var VALID_BACKENDS = ["auto", "threedna", "amber", "rdkit"];

    // The panel's dependency slots, wired once for all five panels
    // (`panel-deps.js`): `configure` is the test door; the chain resolves
    // and checks them at each call.
    var _deps = root.molbuilder.panelDeps.make(root);
    var configure = _deps.configure;

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
        return _deps.generate(
            { kind: "rna", input: trimmed, form: form, backend: backend },
            { kind: "rna",
              generator_input: { sequence: trimmed, form: form,
                                 backend: backend } },
            "Building the RNA\u2026");
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

        /* The shared `.status` writer (lib/status.js): `.status` is the
         * app's one severity surface and its `error` IS red; the busy state
         * is the neutral line. */
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
