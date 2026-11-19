/* panel-deps.js — the builder panels' dependency wiring, once, for the five
 * panels (smiles · name · peptide · rna · dna).
 *
 *   * `configure()` is the test door.  A Node unit test drives the panel's
 *     state machine with a fake structure page, without a DOM; the request
 *     goes through the page's one reader (`lib/server.js`).
 *   * `resolve()` re-reads `window.molbuilder.*` on every call, so a
 *     template that loads a panel before `page.js` registers its globals
 *     still finds them: a later script-load cannot silently degrade the panel.
 *
 * Values a test injected are never overwritten by the production lookup —
 * that is what makes the two doors coexist.
 *
 * The slot also carries THE ONE GENERATE CHAIN the five panels share
 * (`generate`): its slots resolved and checked, the request, the page's load
 * gate and the answer's envelope, under the page's fence -- said once, so no
 * panel can skip the fence or word a refusal its own way; the one door for a
 * panel whose input is a line of text (`generateText`: SMILES, name,
 * peptide); the one nucleic-acid panel, DNA's and RNA's (`generateNucleic`);
 * the one wiring of a panel's page, all five's (`wirePanel`); and the one
 * panel skeleton (`panel`): a panel file is its own `generate` and nothing
 * else.  A panel keeps what is its own: saying an empty field and naming its
 * request -- never the builders' rules, which the builder's refusal says.
 *
 * Exports (on window.molbuilder.panelDeps): panel(root, module, spec) -> a
 * panel's api, its slot made by `make`.  No panel checks the builders' own
 * rules (`engines/builders.md` § 1).
 */
(function (root) {
    "use strict";

    var BUILD_URL = "/api/build/molecule";

    function make(host) {
        /* `findings`: the one findings renderer
         * (`molbuilder.validationFindings`); `panel`: the list under the
         * panel's status its build's findings are drawn in, found when the
         * panel is wired (`wirePanel`). */
        var deps = { structurePage: null, findings: null, panel: null };

        /** THE BUILD'S FINDINGS, drawn under the panel through the one
         *  renderer (`science/validation.md` § 4.1 R2, R5) -- ``[]`` clears
         *  the list.  False when there are findings and nowhere to draw
         *  them: R3, nothing dropped without a word. */
        function draw(rows) {
            if (deps.panel && deps.findings) {
                deps.findings.render(rows, { panel: deps.panel });
                return true;
            }
            return rows.length === 0;
        }

        /** POST ``request`` to the molecule builder, then hand its structure
         *  to the page's load gate (`structurePage.loadIntoCanvas`) with
         *  ``generator`` -- ``{kind, generator_input}``, the record of what
         *  made it.  UNDER THE PAGE'S FENCE (`ui-contract.md` § 10): an embed
         *  is real server work, and a tab switch or a sidebar click while it
         *  runs would abandon or retarget the result; Cancel aborts the
         *  request, and the fence releases on every path.  What the build
         *  says of what it built -- its notices -- is drawn under the panel
         *  (its ``<kind>-findings`` list), the panel's earlier findings
         *  cleared at every build.  Resolves the panel's envelope: ``{ok: true, n_atoms,
         *  backend_used}``,
         *  ``{ok: false, cancelled: true}`` when the gate declines or Cancel
         *  is pressed, ``{ok: false, error}`` otherwise. */
        function generate(request, generator, reason) {
            // THE SLOTS, read at the call: a panel may load before the page
            // that fills them; one still empty -- the load door, or the
            // page's one reader -- is said, naming the panel.
            resolve();
            if (!deps.structurePage || !host.molbuilder.server) {
                return Promise.resolve({ ok: false, error: generator.kind
                    + ": the Molbuilder page did not load; reload it" });
            }
            draw([]);
            var ctl = (typeof AbortController === "function")
                ? new AbortController() : null;
            var chain = function () {
                // THE ONE READER (ui-contract.md § 12).
                return host.molbuilder.server.ask(BUILD_URL, {
                    method: "POST",
                    body:   request,
                    signal: ctl ? ctl.signal : undefined,
                })
                .then(function (a) {
                    // CANCEL IS AN ANSWER, not a failure: the request was
                    // aborted by the person, and the server was reachable.
                    if (a.aborted) return { ok: false, cancelled: true };
                    if (!a.ok) return { ok: false, error: a.error };
                    var body = a.body;
                    // The MODEL door: loadIntoCanvas installs into an empty
                    // viewer or appends to the open structure (the FILE
                    // door, projects.parser.openMolecule, is not this one).
                    return deps.structurePage.loadIntoCanvas(
                        { structure: body.structure }, generator
                    ).then(function (gate) {
                        // THE GATE'S ANSWER, its envelope this one's.
                        if (!gate.ok) return gate;
                        // WHAT THE BUILD FOUND, under the panel; ONE ANSWER
                        // for every panel: the atoms and the backend that
                        // built them.
                        if (!draw(Array.isArray(body.notices)
                                  ? body.notices : [])) {
                            return { ok: false, error: generator.kind
                                + ": built, but its findings have nowhere "
                                + "to be drawn -- the panel's findings list "
                                + "or the renderer is missing" };
                        }
                        // The count is the structure's own (`structure`,
                        // `web-api.md` § 1), one shape for every answer.
                        return { ok: true,
                                 n_atoms: body.structure.elements.length,
                                 backend_used: body.backend_used };
                    });
                })
                .catch(function (err) {
                    // A STEP OF THE CHAIN FAILING (the gate answers its own
                    // envelope): one envelope, so the panel need not branch
                    // on exceptions.
                    return { ok: false,
                             error: (err && err.message) ? err.message
                                                         : String(err) };
                });
            };
            var busy = host && host.molbuilder && host.molbuilder.pageBusy;
            return busy
                ? busy.run(reason, ctl ? [function () { ctl.abort(); }] : [],
                           chain)
                : chain();
        }

        /** What was typed, trimmed -- or ``null`` when nothing was: the one
         *  check every panel makes of its own before a request. */
        function typedOf(text) {
            return (typeof text === "string" && text.trim()) ? text.trim()
                                                             : null;
        }

        /** A panel whose input is one line of text -- a SMILES, a name, a
         *  peptide sequence: the text refused with ``empty`` when blank,
         *  else trimmed and built as ``kind`` (the request's ``input``, the
         *  record's ``generator_input[key]``, ``key`` the kind's own name
         *  for it, ``kind`` when none) under ``reason``. */
        function generateText(text, kind, empty, reason, key) {
            var typed = typedOf(text);
            if (typed === null) {
                return Promise.resolve({ ok: false, error: empty });
            }
            var said = {};
            said[key || kind] = typed;
            return generate({ kind: kind, input: typed },
                            { kind: kind, generator_input: said }, reason);
        }

        /** A NUCLEIC-ACID PANEL -- DNA's or RNA's, one shape: the sequence
         *  as typed (refused with ``empty`` when blank), the form and
         *  backend its menus hold (``opts.form``, ``opts.backend``, auto when
         *  none), and the kind's own further fields (``extra``: DNA's clash
         *  relief), built as ``kind`` under ``reason``. */
        function generateNucleic(kind, sequence, opts, empty, reason, extra) {
            opts = opts || {};
            var typed = typedOf(sequence);
            if (typed === null) {
                return Promise.resolve({ ok: false, error: empty });
            }
            var backend = opts.backend || "auto";
            var request = { kind: kind, input: typed, form: opts.form,
                            backend: backend };
            Object.keys(extra || {}).forEach(function (k) {
                request[k] = extra[k];
            });
            return generate(request,
                            { kind: kind,
                              generator_input: { sequence: typed,
                                                 form: opts.form,
                                                 backend: backend } },
                            reason);
        }

        /** Wire a builder panel's page, ONE SHAPE FOR ALL FIVE -- its
         *  ``<kind>-input``, Generate button, status and findings list, and
         *  the menus and clash-relief box where the kind has them -- to
         *  ``build(text, opts)``, once.  The status line says what was typed
         *  and what built it -- in the Backend menu's own words where the
         *  panel has one, else as the answer names it; what the build did
         *  not build of what was asked is the findings' to say, under it. */
        var wired = false;
        function wirePanel(doc, kind, build, label) {
            if (!doc || wired) return;
            wired = true;
            var el = function (part) {
                return doc.getElementById(kind + "-" + part);
            };
            var input = el("input"), button = el("generate-btn");
            var formSel = el("form-select"), backendSel = el("backend-select");
            var relax = el("relax-clashes"), status = el("status");
            deps.panel = el("findings") || null;
            if (!input || !button) return;
            var say = function (msg, severity) {
                host.molbuilder.status.set(
                    status, msg, severity === "error" ? "error" : null);
            };
            var wordsOf = function (backend) {
                var opts = backendSel ? backendSel.options || [] : [];
                for (var i = 0; i < opts.length; i++) {
                    if (opts[i].value === backend) return opts[i].text;
                }
                return backend;
            };
            button.addEventListener("click", function () {
                // THE TEXT AS TYPED: the builder reads it and says what it
                // cannot; captured at the click, so the line names what was
                // built, not what was typed while it ran.
                var typed = input.value.trim();
                button.disabled = true;
                say("Generating " + label + "\u2026", "generating");
                build(typed, {
                    form: formSel ? formSel.value : undefined,
                    backend: backendSel ? backendSel.value : undefined,
                    relaxClashes: !!(relax && relax.checked),
                }).then(function (r) {
                    button.disabled = false;
                    if (r.ok) {
                        say("Generated " + r.n_atoms + " atoms from " + typed
                            + (r.backend_used
                               ? " on " + wordsOf(r.backend_used) : ""));
                    } else if (r.cancelled) {
                        // Not applied: the workspace is untouched, said so
                        // the person does not think Generate failed.
                        say("Kept existing workspace.");
                    } else {
                        say(r.error || "Generation failed.", "error");
                    }
                });
            });
            // Enter inside the input is Generate too.
            input.addEventListener("keydown", function (ev) {
                if (ev.key === "Enter" && !button.disabled) {
                    ev.preventDefault();
                    button.click();
                }
            });
        }

        /** The production door: re-read every call. */
        function resolve() {
            if (typeof host === "undefined" || !host || !host.molbuilder) {
                return;
            }
            if (!deps.structurePage && host.molbuilder.structurePage) {
                deps.structurePage = host.molbuilder.structurePage;
            }
            if (!deps.findings && host.molbuilder.validationFindings) {
                deps.findings = host.molbuilder.validationFindings;
            }
        }

        return {
            generateText: generateText,
            generateNucleic: generateNucleic,
            wirePanel: wirePanel,
            /** The test door: explicit fakes win, and keep winning. */
            configure: function (opts) {
                opts = opts || {};
                if (opts.structurePage) deps.structurePage = opts.structurePage;
                if (opts.findings) deps.findings = opts.findings;
            },
        };
    }

    /** ONE BUILDER PANEL, its skeleton said once for all five: its slot
     *  (`make`), its ``generate`` made from the slot (``spec.generate``),
     *  and its api -- ``configure`` (the test door), ``generate``,
     *  and ``wirePanel`` (its page, by ``spec.kind``'s ids, its status
     *  naming ``spec.label``).  Under node (``mod``, the panel
     *  file's own module) the api is the file's export; in the page it is
     *  ``molbuilder[spec.global]``, wired once the document has loaded and
     *  registered with the runtime as ``structure.<kind>``.  The slots are
     *  read at each call (`resolve`), so nothing is configured at load. */
    function panel(host, mod, spec) {
        var slot = make(host);
        var generate = spec.generate(slot);
        var api = {
            configure: slot.configure,
            generate:  generate,
            wirePanel: function (opts) {
                slot.wirePanel((opts || {}).doc || host.document, spec.kind,
                               generate, spec.label);
            },
        };
        if (mod && mod.exports) {
            mod.exports = api;
            return api;
        }
        host.molbuilder[spec.global] = api;
        if (host.document) {
            if (host.document.readyState === "loading") {
                host.document.addEventListener(
                    "DOMContentLoaded", function () { api.wirePanel(); });
            } else {
                api.wirePanel();
            }
        }
        if (host.molbuilder.runtime
            && typeof host.molbuilder.runtime.register === "function") {
            host.molbuilder.runtime.register("structure." + spec.kind, api);
        }
        return api;
    }

    var api = { panel: panel };
    if (typeof module !== "undefined" && module.exports) {
        module.exports = api;
    }
    root.molbuilder = root.molbuilder || {};
    root.molbuilder.panelDeps = api;
})(typeof globalThis !== "undefined" ? globalThis : this);
