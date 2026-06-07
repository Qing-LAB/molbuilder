/* DNA-builder generator panel — Molbuilder-tab Source.
 *
 * Wires the Sources card's "Generate DNA" panel.  POSTs
 * ``{kind: "dna", input: <sequence>, form: <B|A|Z>}`` to
 * /api/build/molecule.  Backend dispatches to ``build_dna`` which
 * picks the best installed backend (3DNA preferred, AmberTools
 * fallback, RDKit last) for an ssDNA from a 1-letter sequence.
 *
 * Other knobs the underlying ``build_dna`` accepts (terminal,
 * add_hydrogens, protonate_phosphates) keep the Python defaults
 * here — the panel sticks to the load-bearing knobs (sequence +
 * form).  Advanced knobs can land later as additional fields if
 * users ask.
 *
 * Flow mirrors smiles.js / name.js / peptide.js exactly: validate
 * client-side → POST → loadIntoCanvas gate → viewer render.
 *
 * Sequence validation: 4-letter DNA alphabet ACGT (case-
 * insensitive; lowercased input gets uppercased before
 * submission).  Anything else gets an actionable inline error
 * before the network call.
 *
 * Design ref: docs/tabs/architecture.md § 5.1 (panel 3: 3DNA
 * helix builder).
 */
(function (root) {
    "use strict";

    var BUILD_URL = "/api/build/molecule";
    var VALID_DNA = /^[ACGT]+$/i;
    var VALID_FORMS = ["B", "A", "Z"];

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
     * Generate ssDNA from a 1-letter sequence.
     *
     * @param {string} sequence  ACGT, case-insensitive
     * @param {object} [opts]
     * @param {"B"|"A"|"Z"} [opts.form="B"]  helix form
     * @returns {Promise<envelope>}
     */
    function generate(sequence, opts) {
        opts = opts || {};
        if (typeof sequence !== "string" || !sequence.trim()) {
            return Promise.resolve({
                ok: false,
                error: "Enter a DNA sequence first (ACGT).",
            });
        }
        var trimmed = sequence.trim().toUpperCase().replace(/\s+/g, "");
        if (!VALID_DNA.test(trimmed)) {
            return Promise.resolve({
                ok:    false,
                error: "Sequence must use ACGT only.  Got: "
                     + JSON.stringify(sequence),
            });
        }
        var form = opts.form || "B";
        if (VALID_FORMS.indexOf(form) < 0) {
            return Promise.resolve({
                ok:    false,
                error: "Helix form must be one of "
                     + VALID_FORMS.join(", ") + ".  Got: "
                     + JSON.stringify(form),
            });
        }
        if (!_fetch) {
            return Promise.reject(new Error(
                "dna: fetch not configured"));
        }
        if (!_structurePage) {
            return Promise.reject(new Error(
                "dna: structurePage not configured"));
        }
        return _fetch(BUILD_URL, {
            method:  "POST",
            headers: { "Content-Type": "application/json" },
            body:    JSON.stringify({
                kind: "dna", input: trimmed, form: form,
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
                { kind: "dna",
                  generator_input: {
                      sequence: trimmed, form: form,
                  } }
            ).then(function (gate) {
                if (!gate.ok) {
                    return { ok: false, cancelled: true };
                }
                if (typeof _viewerLoader === "function") {
                    var fname = "dna-" + form + "-" + trimmed + ".xyz";
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

        var input  = doc.getElementById("dna-input");
        var formSel = doc.getElementById("dna-form-select");
        var button = doc.getElementById("dna-generate-btn");
        var status = doc.getElementById("dna-status");
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
            var formChoice = (formSel && formSel.value) || "B";
            button.disabled = true;
            setStatus("Generating DNA…", "generating");
            generate(echo, { form: formChoice }).then(function (r) {
                button.disabled = false;
                if (r.ok) {
                    setStatus(
                        "Generated " + (r.n_atoms != null
                            ? r.n_atoms + " atoms" : "")
                        + " from " + formChoice + "-form "
                        + echo);
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
        VALID_DNA:   VALID_DNA,
        VALID_FORMS: VALID_FORMS,
    };

    if (typeof module !== "undefined" && module.exports) {
        module.exports = api;
    } else {
        root.molbuilder = root.molbuilder || {};
        root.molbuilder.structureDna = api;
        configure({
            fetch:         root.fetch
                            ? root.fetch.bind(root)
                            : undefined,
            structurePage: root.molbuilder.structurePage,
            viewerLoader:  root.molbuilder.loadStructureText,
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
                "structure.dna", api);
        }
    }
})(typeof window !== "undefined" ? window : globalThis);
