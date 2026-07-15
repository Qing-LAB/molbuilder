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
    var VALID_BACKENDS = ["auto", "threedna", "amber", "rdkit"];

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
     * Parse the DNA-panel notation CLIENT-SIDE (mirrors the server's
     * nucleic._parse_dna_notation) so the ACGT check runs on the CORE
     * sequence, not the raw "ds,ATGC" / "5'-ATGC-3'" text:
     *   bare "ATGC"     -> { mode:"ss", core:"ATGC" }
     *   "ds,ATGC"       -> { mode:"ds", core:"ATGC" }   (canonical duplex)
     *   5'/3' markers accepted; a 3'->5' strand is reversed to 5'->3'.
     *   two comma-separated strands -> { twoStrand:true } (the server owns the
     *   "not yet supported" message for arbitrary duplexes).
     * The RAW notation is sent to the server, which re-parses authoritatively.
     */
    function _parseDnaNotation(raw) {
        var s = String(raw == null ? "" : raw).trim();
        var mode = "ss";
        if (/^ds[,:]/i.test(s)) { mode = "ds"; s = s.slice(3).trim(); }
        if (s.indexOf(",") >= 0) {
            return { mode: mode, core: "", twoStrand: true };
        }
        var m = s.match(/^\s*([53])'\s*-\s*([\s\S]*?)\s*-\s*([53])'\s*$/);
        var core;
        if (m) {
            core = m[2].replace(/[^A-Za-z]/g, "").toUpperCase();
            if (m[1] === "3") core = core.split("").reverse().join("");
        } else {
            core = s.replace(/[^A-Za-z]/g, "").toUpperCase();
        }
        return { mode: mode, core: core, twoStrand: false };
    }

    /**
     * Generate ssDNA from a 1-letter sequence.
     *
     * @param {string} sequence  ACGT, case-insensitive
     * @param {object} [opts]
     * @param {"B"|"A"|"Z"} [opts.form="B"]  helix form
     * @param {"auto"|"threedna"|"amber"|"rdkit"} [opts.backend="auto"]
     *        backend selector (default auto = best installed)
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
        var parsed = _parseDnaNotation(sequence);
        if (!parsed.twoStrand && !VALID_DNA.test(parsed.core)) {
            return Promise.resolve({
                ok:    false,
                error: "Sequence must use ACGT only (optionally prefixed with "
                     + "\"ds,\" for a canonical double strand).  Got: "
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
        var backend = opts.backend || "auto";
        if (VALID_BACKENDS.indexOf(backend) < 0) {
            return Promise.resolve({
                ok:    false,
                error: "Backend must be one of "
                     + VALID_BACKENDS.join(", ") + ".  Got: "
                     + JSON.stringify(backend),
            });
        }
        // Z-DNA via fiber requires alternating poly-d(GC).  Catch the
        // mismatch client-side with an actionable error so the user
        // doesn't wait for the server-side 3DNA backend to reject
        // (and a 60 s subprocess timeout if fiber slips into its
        // interactive "Number of repeats" prompt).  Bug #2 fix
        // (2026-06-07).
        // ``(GC)+`` and ``(CG)+`` cover both strand orientations;
        // length is implicitly even because each repeat is 2 bases.
        // Matches the server-side ``_is_alternating_gc`` predicate
        // exactly so client-side acceptance == server-side acceptance.
        if (form === "Z" && !parsed.twoStrand
                && (backend === "auto" || backend === "threedna")
                && !/^(GC)+$/i.test(parsed.core)
                && !/^(CG)+$/i.test(parsed.core)) {
            return Promise.resolve({
                ok:    false,
                error: "Z-DNA via 3DNA only supports alternating "
                     + "poly-d(GC) sequences (e.g. GCGC, CGCGCG).  "
                     + "Use B-form or A-form for " + parsed.core + ".",
            });
        }
        // Lazy-resolve dependencies in case the script-load
        // order put us above page.js / lib/* (LANDMINE-2 fix).
        _lazyResolve();
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
                kind:    "dna",
                // Send the RAW notation ("ds,ATGC", "5'-…-3'"): the server parses
                // it authoritatively (nucleic._parse_dna_notation).
                input:   sequence.trim(),
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
                { kind: "dna",
                  generator_input: {
                      sequence: sequence.trim(), form: form,
                      backend: backend,
                  } }
            ).then(function (gate) {
                if (!gate.ok) {
                    return { ok: false, cancelled: true };
                }
                // loadIntoCanvas now routes through molview.data.openMolecule,
                // which parses + renders the structure itself.  The old
                // viewerLoader second load is removed — it would
                // double-apply the same bytes.
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

        var input  = doc.getElementById("dna-input");
        var formSel = doc.getElementById("dna-form-select");
        var backendSel = doc.getElementById("dna-backend-select");
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
            var backendChoice = (backendSel && backendSel.value) || "auto";
            button.disabled = true;
            setStatus("Generating DNA…", "generating");
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
