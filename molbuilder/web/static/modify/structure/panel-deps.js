/* panel-deps.js — the builder panels' dependency wiring, once, for the five
 * panels (smiles · name · peptide · rna · dna).
 *
 *   * `configure()` is the test door.  A Node unit test drives the panel's
 *     state machine with a fake fetch and a fake structure page, without a
 *     DOM or an HTTP roundtrip.
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
 * panel can skip the fence or word a refusal its own way; and the one door
 * for a panel whose input is a line of text (`generateText`: SMILES, name).
 * A panel keeps what is its own: checking its input and naming its request.
 *
 * Exports (on window.molbuilder.panelDeps): make(root) -> a per-panel slot.
 */
(function (root) {
    "use strict";

    var BUILD_URL = "/api/build/molecule";

    function make(host) {
        var deps = { fetch: null, structurePage: null };

        /** POST ``request`` to the molecule builder, then hand its structure
         *  to the page's load gate (`structurePage.loadIntoCanvas`) with
         *  ``generator`` -- ``{kind, generator_input}``, the record of what
         *  made it.  UNDER THE PAGE'S FENCE (`ui-contract.md` § 10): an embed
         *  is real server work, and a tab switch or a sidebar click while it
         *  runs would abandon or retarget the result; Cancel aborts the
         *  request, and the fence releases on every path.  Resolves the
         *  panel's envelope: ``{ok: true, n_atoms, backend_used, warnings}``,
         *  ``{ok: false, cancelled: true}`` when the gate declines or Cancel
         *  is pressed, ``{ok: false, error}`` otherwise. */
        function generate(request, generator, reason) {
            // THE SLOTS, read at the call: a panel may load before the page
            // that fills them; one still empty is said, naming the panel.
            resolve();
            if (!deps.fetch) {
                return Promise.reject(new Error(
                    generator.kind + ": fetch not configured"));
            }
            if (!deps.structurePage) {
                return Promise.reject(new Error(
                    generator.kind + ": structurePage not configured"));
            }
            var ctl = (typeof AbortController === "function")
                ? new AbortController() : null;
            var chain = function () {
                return deps.fetch(BUILD_URL, {
                    method:  "POST",
                    headers: { "Content-Type": "application/json" },
                    body:    JSON.stringify(request),
                    signal:  ctl ? ctl.signal : undefined,
                })
                .then(function (r) {
                    return r.json().then(function (body) {
                        return { httpOk: r.ok, body: body };
                    });
                })
                .then(function (env) {
                    var body = env.body || {};
                    if (!env.httpOk || !body.ok) {
                        return { ok: false,
                                 error: body.error
                                     || ("HTTP error from " + BUILD_URL) };
                    }
                    // The MODEL door: loadIntoCanvas installs into an empty
                    // viewer or appends to the open structure (the FILE
                    // door, projects.parser.openMolecule, is not this one).
                    return deps.structurePage.loadIntoCanvas(
                        { structure: body.structure }, generator
                    ).then(function (gate) {
                        if (!gate.ok) return { ok: false, cancelled: true };
                        // ONE ANSWER for every panel: the atoms, the backend
                        // that built them, and the builder's warnings (a
                        // mismatched-duplex clash) for the panel's status.
                        return { ok: true, n_atoms: body.n_atoms,
                                 backend_used: body.backend_used,
                                 warnings: Array.isArray(body.build_warnings)
                                     ? body.build_warnings : [] };
                    });
                })
                .catch(function (err) {
                    // CANCEL IS AN ANSWER, not a failure: the request was
                    // aborted by the person, and the server was reachable.
                    if (ctl && ctl.signal.aborted) {
                        return { ok: false, cancelled: true };
                    }
                    // A network drop or an answer that is not JSON: one
                    // envelope, so the panel need not branch on exceptions.
                    return { ok: false,
                             error: "Could not reach " + BUILD_URL + ": "
                                 + (err && err.message ? err.message
                                                       : String(err)) };
                });
            };
            var busy = host && host.molbuilder && host.molbuilder.pageBusy;
            return busy
                ? busy.run(reason, ctl ? [function () { ctl.abort(); }] : [],
                           chain)
                : chain();
        }

        /** A panel whose input is one line of text -- a SMILES, a name: the
         *  text refused with ``empty`` when blank, else trimmed and built as
         *  ``kind`` (the request's ``input``, the record's
         *  ``generator_input[kind]``) under ``reason``. */
        function generateText(text, kind, empty, reason) {
            if (typeof text !== "string" || !text.trim()) {
                return Promise.resolve({ ok: false, error: empty });
            }
            var trimmed = text.trim();
            var said = {};
            said[kind] = trimmed;
            return generate({ kind: kind, input: trimmed },
                            { kind: kind, generator_input: said }, reason);
        }

        /** The production door: re-read every call. */
        function resolve() {
            if (typeof host === "undefined" || !host || !host.molbuilder) {
                return;
            }
            if (!deps.fetch && host.fetch) {
                deps.fetch = host.fetch.bind(host);
            }
            if (!deps.structurePage && host.molbuilder.structurePage) {
                deps.structurePage = host.molbuilder.structurePage;
            }
        }

        return {
            generate: generate,
            generateText: generateText,
            /** The test door: explicit fakes win, and keep winning. */
            configure: function (opts) {
                opts = opts || {};
                if (opts.fetch) deps.fetch = opts.fetch;
                if (opts.structurePage) deps.structurePage = opts.structurePage;
            },
            resolve: resolve,
            get fetch() { return deps.fetch; },
            get structurePage() { return deps.structurePage; },
        };
    }

    var api = { make: make, BUILD_URL: BUILD_URL };
    if (typeof module !== "undefined" && module.exports) {
        module.exports = api;
    }
    root.molbuilder = root.molbuilder || {};
    root.molbuilder.panelDeps = api;
})(typeof globalThis !== "undefined" ? globalThis : this);
