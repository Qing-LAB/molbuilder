/* molbuilder web UI client — the structure-optimization tab page controller.
 *
 * WHAT THIS TAB DOES:
 *   1. Show the structure the sidebar loaded, through the MolView component.
 *   2. Render the parameter form from the CATALOGUE (`/api/build/schema`,
 *      fetched by the shared form library) and collect what the user sets.
 *   3. Validate it live — POST /api/build/preflight — and place each finding
 *      beside the control it is about.
 *   4. Show the charge and spin each form's calculation will carry, and why
 *      — the chemistry card (`lib/chemistry.js`, POST /api/structure/analyze).
 *   5. HAND THE DESCRIPTION OVER — POST /api/task-setup/handover, which
 *      writes the calculation's files into the sidebar's folder, every file
 *      or none (`handover-procedure.md` § 2).  Among them `task.1st.json`
 *      (`stages.md` § 6.5a): a partial description the Task Setup tab
 *      finishes.
 *
 * IT RENDERS NO DECK.  A deck is written by `prep`, on the machine that will
 * run it (`project-layout.md` § 2.2) — never here.  What this tab produces is
 * a DESCRIPTION, and step 5 is where it leaves.
 *
 * THE INSPECT CARD -- the viewer, the Load button, the structure kept as the
 * tab's own draft and put back when the tab is shown again -- is the one card
 * the Spectrum tab shares (`lib/structure-card.js`, `web/tabs.md` § 3).
 */
import { attachStructureCard } from "/static/lib/structure-card.js";
import { ask } from "../lib/server.js";

(function () {
    "use strict";
    const $ = (id) => document.getElementById(id);

    /* THE INSPECT CARD (`lib/structure-card.js`): under this tab's own tag
     * (workspace.md § 4), the viewer's owner and its draft's.  A structure
     * that lands -- loaded, or put back on arriving -- invalidates the last
     * findings and is checked and answered for again. */
    const card = attachStructureCard({
        host: "viewer-host",
        tag:  "structure-opt",
        onLoaded: () => {
            clearIssues("fdf-issues", "siesta-form-container");
            clearIssues("pyscf-issues", "pyscf-form-container");
            refreshPreflightDebounced.siesta();
            refreshPreflightDebounced.pyscf();
            _chemistry.refresh();
        },
    });
    // The engine strip is a SHARED widget (ui-contract.md § 1): the sheet in
    // form-components.css, the behaviour in lib/tab-strip.js, mounted below.
    let _engineStrip = null;

    // ----- Findings ----------------------------------------------------
    // Findings rendering lives in ONE module for the whole app
    // (lib/validation-findings.js, contract R2 in
    // docs/science/validation.md 4.1).  These two
    // thin wrappers keep the call sites readable.
    /* {field name -> the DOM id the form gave it}, from the schema this page
     * actually rendered, by the renderer's one map (`formSchema.fieldIds`). */
    function _fieldIds(engine) {
        const sch = formSchemas[engine];
        const fs = (window.molbuilder || {}).formSchema;
        return sch && fs ? fs.fieldIds(sch) : null;
    }

    function clearIssues(panelId, formContainerId) {
        const f = (window.molbuilder || {}).validationFindings;
        if (!f) return;
        f.clear({ panel:     $(panelId),
                  formScope: formContainerId ? $(formContainerId) : null });
    }

    // Debounce helper -- collapses a burst of form-input events into
    // a single call after `wait` ms of quiet.  Used to throttle the
    // live /api/build/preflight calls so dragging a slider doesn't
    // spam the endpoint.
    function debounce(fn, wait) {
        let t = null;
        return function (...args) {
            if (t) clearTimeout(t);
            t = setTimeout(() => fn.apply(this, args), wait);
        };
    }

    // Live preflight: when the user has already built a structure
    // (a structure is loaded) and they tweak any tab's params, hit
    // /api/build/preflight to refresh the issues panel before they
    // click Generate.  No-op until the first successful build, so a
    // user who just landed on the page and is fiddling with form
    // defaults doesn't see warnings that haven't been earned yet.
    /* ONE PLACE this tab shows an engine's findings -- the live preflight's
     * and the hand-over's (the cell gate's notices) -- through the one
     * renderer, in that engine's panel beside its form. */
    function _showFindings(engine, issues, reveal) {
        const f = (window.molbuilder || {}).validationFindings;
        if (!f) return { total: 0, residual: 0, byCard: {}, byField: {},
                         counts: { error: 0, warn: 0, info: 0 } };
        return f.render(issues, {
            panel:     $((engine === "siesta") ? "fdf-issues" : "pyscf-issues"),
            formScope: $(engine + "-form-container"),
            fieldIds:  _fieldIds(engine),
            reveal:    !!reveal,
        });
    }

    // What the live preflight said last, per engine -- the findings a Send
    // adds its own to (`handover-procedure.md` § 2.1).
    const _live = { siesta: [], pyscf: [] };

    // The newest check asked, per engine: an answer to an older one -- an
    // edit since, or a structure since -- is not drawn, as on the Spectrum
    // tab (`lib/spectra/core.js`).
    const _checkSeq = { siesta: 0, pyscf: 0 };

    async function refreshPreflight(engine) {
        const structure = card.structure();
        if (!structure) return;
        let params;
        try { params = collectParams(engine); }
        catch (_) { return; }   // the field's own caption says why
        const mine = ++_checkSeq[engine];
        // THE ONE READER (ui-contract.md § 12).
        const a = await ask("/api/build/preflight", {
            method: "POST",
            /* The cell rides inside `structure`, from the same read as the
             * atoms -- the preflight has to judge the structure the user is
             * looking at, not a cell fetched a moment later. */
            body: { structure, engine, params,
                    calculation: "optimization",
                    // THE FOLDER A SEND WOULD WRITE INTO (tabs.md § 3).
                    dest: window.molbuilder.taskHandover.where(
                        window.molbuilder.projects).into || undefined },
        });
        if (mine !== _checkSeq[engine]) return;
        /* Rendered whenever issues came back, ok or not (E-A3): the
         * endpoint's bad-params branch answers ok:false WITH the parse
         * error as an issue precisely so this panel can show it.
         * Same rule as the Spectrum tab (lib/spectra/core.js).  Any other
         * failure is not surfaced -- the panel stays in its previous state. */
        const issues = a.body && a.body.issues;
        if (Array.isArray(issues)) {
            _live[engine] = issues;
            _showFindings(engine, issues);
        }
    }
    
    const refreshPreflightDebounced = {
        siesta: debounce(() => refreshPreflight("siesta"), 250),
        pyscf:  debounce(() => refreshPreflight("pyscf"),  250),
    };

    // WHERE A SEND WRITES, said beside it, and each live check asked again
    // of the folder as the sidebar moves (`tabs.md` § 3).
    window.molbuilder.runtime.whenReady("projects").then((proj) =>
        window.molbuilder.taskHandover.followWhere(
            proj, document.getElementById("send-where"), () => {
                refreshPreflightDebounced.siesta();
                refreshPreflightDebounced.pyscf();
            }));

    // ----- Tabs (SIESTA / PySCF) -------------------------------------
    // One switcher for both tabs that offer two engine forms
    // (lib/tab-strip.js); the buttons name their panels by aria-controls.
    _engineStrip = window.molbuilder.tabStrip.mount(
        document.querySelector("#generate-card .tabs")
            || document.querySelector(".tabs"), {});

    // ----- What greys a field out -------------------------------------
    // The catalogue's `active_when` (engines/template.md § 5), drawn by the
    // one renderer (`lib/form-schema.js`): a relaxation's step settings wait
    // on a relaxer, the PCM model on a solvent.  The page keeps no rule of
    // its own about which field locks which (form-schema.md § 1.3).

    // Module-level cache of the form schemas, populated once on
    // page load by initFormsFromSchema().  collectParams() reads from
    // this, and the findings are placed by its ids (`_fieldIds`).
    const formSchemas = { siesta: null, pyscf: null };

    // THE CHEMISTRY CARD (`lib/chemistry.js`): the charge and spin each
    // engine form's calculation will carry, resolved by the one
    // electronic-state class for exactly what that form says, with each
    // value's reason (`science/chemistry-correctness.md` § 2a).  It is asked
    // about THE STRUCTURE THIS TAB WOULD HAND OVER -- the viewer's, through
    // the same door as the preflight and Generate -- on every load and
    // restore, and on every edit to a charge or spin field (the module
    // listens).  It fills nothing in: a blank field is already the
    // instruction "work it out".
    const _chemistry = window.molbuilder.chemistry.attach({
        kind: "optimization",
        forms: () => ({
            siesta: [{ host: $("siesta-form-container"),
                       schema: formSchemas.siesta }],
            pyscf:  [{ host: $("pyscf-form-container"),
                       schema: formSchemas.pyscf }],
        }),
        structure: () => card.structure(),
    });


    /* ---------- what is not at its default ---------------------------
     *
     * A single "reset everything" button cannot tell a deliberate 4x4x1
     * k-grid from a value that arrived with an older session, and both
     * live in the same form.  So this LISTS the differences and resets
     * only what is ticked.
     *
     * The comparison is `formSchema.diffFromDefaults` -- the module that
     * already owns "what the DOM holds" and "what the schema says".
     * Reading the inputs here would mean a second reader for every kind
     * that module handles.
     */
    function mountRecommended(engine, container, schema) {
        const panel = $(engine + "-recommend");
        if (!panel) return;
        const listEl  = panel.querySelector(".rec-diff-list");
        const countEl = panel.querySelector(".rec-diff-count");
        const resetEl = panel.querySelector(".rec-diff-reset");
        const allEl   = panel.querySelector(".rec-diff-all");
        const fs = (window.molbuilder || {}).formSchema;
        if (!listEl || !fs || typeof fs.diffFromDefaults !== "function") return;

        function show(v) {
            // An EMPTY list is empty, as an empty string is.
            if (Array.isArray(v)) return v.length ? v.join(", ") : "(empty)";
            if (v === true) return "on";
            if (v === false) return "off";
            if (v === null || v === undefined || v === "") return "(empty)";
            return String(v);
        }

        function ticked() {
            return Array.from(listEl.querySelectorAll("input:checked"))
                        .map((c) => c.value);
        }

        function refresh() {
            let diffs = [];
            try { diffs = fs.diffFromDefaults(container, schema); }
            catch (_) { return; }          // a half-rendered form is not an error
            if (!diffs.length) {
                panel.hidden = true;
                listEl.textContent = "";
                return;
            }
            panel.hidden = false;
            countEl.textContent = diffs.length === 1
                ? "1 setting is not at its default"
                : diffs.length + " settings are not at their defaults";
            listEl.textContent = "";
            for (const d of diffs) {
                const li = document.createElement("li");
                const lab = document.createElement("label");
                const cb = document.createElement("input");
                cb.type = "checkbox"; cb.value = d.name;
                const name = document.createElement("span");
                name.className = "rec-diff-name";
                name.textContent = d.label;
                const now = document.createElement("span");
                now.className = "rec-diff-now";
                now.textContent = show(d.current) + (d.unit ? " " + d.unit : "");
                const rec = document.createElement("span");
                rec.className = "rec-diff-rec";
                rec.textContent = show(d.recommended) + (d.unit ? " " + d.unit : "");
                lab.title = [d.name, d.help].filter(Boolean).join("\n\n");
                lab.append(cb, name, now, rec);
                li.append(lab);
                listEl.append(li);
            }
            resetEl.disabled = true;
        }

        /* THE BUTTON SAYS WHAT ITS CLICK DOES: every box ticked, it unticks
         * them all. */
        const allSays = () => {
            const n = listEl.querySelectorAll("input[type=checkbox]").length;
            allEl.textContent = (n && ticked().length === n)
                ? "Untick all" : "Tick all";
        };
        listEl.addEventListener("change", () => {
            resetEl.disabled = ticked().length === 0;
            allSays();
        });

        allEl.addEventListener("click", () => {
            const boxes = listEl.querySelectorAll("input[type=checkbox]");
            const turnOn = ticked().length !== boxes.length;
            boxes.forEach((c) => { c.checked = turnOn; });
            resetEl.disabled = !turnOn;
            allSays();
        });

        resetEl.addEventListener("click", () => {
            const want = new Set(ticked());
            if (!want.size) return;
            let diffs = [];
            try { diffs = fs.diffFromDefaults(container, schema); }
            catch (_) { return; }
            // BACK TO NOT CHOSEN, not to a copy of the default: a
            // blank field runs the kind's default and the
            // template records it as nobody's choice (form-schema.md § 1.1);
            // typing the recommended value in would record it as yours.
            const values = {};
            for (const d of diffs) if (want.has(d.name)) values[d.name] = null;
            fs.setValues(container, schema, values);
            refresh();
        });

        // The form is also written by session restore, so the panel follows
        // the DOM rather than only the user's typing.
        container.addEventListener("change", refresh);
        container.addEventListener("input", refresh);
        refresh();
    }

    async function initFormsFromSchema() {
        const fs = (window.molbuilder || {}).formSchema;
        if (!fs) {
            console.error(
                "form-schema.js not loaded; build form will not appear"
            );
            return;
        }
        try {
            const [siesta, pyscf] = await Promise.all([
                fs.fetchSchema("siesta", { calculation: "optimization" }),
                fs.fetchSchema("pyscf",  { calculation: "optimization" }),
            ]);
            formSchemas.siesta = siesta;
            formSchemas.pyscf  = pyscf;
            const siestaC = $("siesta-form-container");
            const pyscfC  = $("pyscf-form-container");
            if (siestaC) fs.renderForm(siestaC, siesta);
            if (pyscfC)  fs.renderForm(pyscfC,  pyscf);
        } catch (exc) {
            console.error("could not load build-form schema:", exc);
            // Surface a visible failure so the user knows the
            // server's /api/build/schema endpoint is unreachable.
            for (const id of ["siesta-form-container", "pyscf-form-container"]) {
                const c = $(id);
                if (!c) continue;
                // The server's words, set as text (ui-contract.md § 7).
                const p = document.createElement("p");
                p.className = "status error";
                p.textContent = "Could not load form schema: " + String(exc);
                c.replaceChildren(p);
            }
            return;
        }
        // The restore writes through `setValues`, which fires `input`, so
        // the renderer's greying follows the restored values.
        _forms.restore();
        wirePreflightListeners();
        installPsmlLibLiveCaption();
        // AFTER the restore: mounted here, the panel's first reading is the
        // form as restored.  (The restore writes through `setValues`, which
        // fires `input`, so each field's caption follows it.)
        for (const [engine, hostId] of [["siesta", "siesta-form-container"],
                                       ["pyscf",  "pyscf-form-container"]]) {
            const host = $(hostId);
            if (host && formSchemas[engine]) {
                mountRecommended(engine, host, formSchemas[engine]);
            }
        }
        // The forms exist now, so the chemistry card can answer for what
        // they say -- a structure loaded before the schemas arrived was
        // answered with every item blank.  A no-op with no structure yet.
        _chemistry.refresh();
    }

    /**
     * Live caption for the SIESTA psml_lib field: a one-liner below the
     * input saying where a typed folder is looked for, since the
     * projects folder it is read in is not visible in the field itself:
     *
     *   ""                    -> ""  (the field's own caption says the blank)
     *   "pseudopotential"     -> "→ pseudopotential, in the projects folder"
     *   "/abs/path", "~/x"    -> "→ /abs/path", "→ ~/x"  (as typed)
     *
     * It says no more than that: whether the server takes the path
     * (`pseudos.resolve_psml_lib`, which refuses one that climbs out with
     * `..`) is said where the description is checked.  Installed once,
     * from initFormsFromSchema's tail, after the forms are drawn and
     * restored; the field is found by the schema's own id (`_fieldIds`),
     * never by a prefix.
     */
    function installPsmlLibLiveCaption() {
        const id = (_fieldIds("siesta") || {}).psml_lib;
        const input = id ? $(id) : null;
        if (!input) return;
        // Caption element rendered immediately after the input.  The
        // form-schema renderer wraps each field in a <label>; we
        // append into that label so the caption rides along with
        // wherever the field lives in the DOM.
        const cap = document.createElement("small");
        // .schema-field-hint is styled in lib/form-schema.css using the
        // --text-muted token.
        cap.className = "schema-field-hint";
        const parent = input.parentElement || input;
        parent.appendChild(cap);
        function render() {
            const v = (input.value || "").trim();
            // A BLANK ONE IS SAID BY THE FIELD'S OWN CAPTION, in the item's
            // words (`null_label`); this one says where a typed path goes.
            if (!v) {
                cap.textContent = "";
                return;
            }
            // Absolute or ~/ -> echo (no anchor applied).
            if (v.charAt(0) === "/" || v.startsWith("~")) {
                cap.textContent = "→ " + v;
                return;
            }
            // Relative -> show projects/-anchored form.  The Save
            // step also tries dest-relative first, but at type-time
            // we don't know dest yet; show the projects/ anchor.
            cap.textContent = "→ " + v + ", in the projects folder";
        }
        input.addEventListener("input", render);
        render();
    }

    /* ONE collector, both engines.
     *
     * A blank is sent as `null`, for both engines: not chosen, which the
     * server's one door reads as "what lies under it" for every field
     * (`template.stated_config`, form-schema.md § 1.1) -- a blank charge
     * or spin among them, whose blank is "work it out"
     * (`science/chemistry-correctness.md` § 2a).  `dispersion = "none"`
     * is sent as itself: it is the value for no correction.  Throws, naming
     * the field, on a value that will not read as its type.
     */
    function collectParams(engine) {
        const schema = formSchemas[engine];
        if (!schema) return {};
        const fs = (window.molbuilder || {}).formSchema;
        return fs.collectForm($(engine + "-form-container"), schema);
    }


    /* KEPT IN THIS BROWSER TAB (`formSchema.keepInTab`, `runtime.md` § 3):
     * each engine's form as it stands, saved as the page is left and put
     * back after the forms are drawn (the user, 2026-11-11). */
    const _forms = window.molbuilder.formSchema.keepInTab(
        "structure_optimization_forms",
        () => ["siesta", "pyscf"].map((engine) => ({
            name: engine, host: $(engine + "-form-container"),
            schema: formSchemas[engine] })));

    // Kick off the async schema fetch + form render.  Everything that
    // needs the drawn forms -- the restore, the live check's listeners on
    // their containers, the pseudopotential caption -- happens inside this
    // function once the renderer has populated the DOM.
    initFormsFromSchema();

    // EACH ENGINE'S FORM refreshes its own preflight as the person adjusts
    // it -- its container listens for `input` and `change` from any field
    // there, the renderer's own writes included (they bubble), so the page
    // keeps no list of ids nor the server's prefixes.  No-op until the user
    // has built a structure (the model is non-empty) -- see
    // refreshPreflight().  Called from initFormsFromSchema's tail, after
    // the restore, so the restored values do not each ask again.
    function wirePreflightListeners() {
        for (const engine of ["siesta", "pyscf"]) {
            const host = $(engine + "-form-container");
            if (!host) continue;
            for (const event of ["input", "change"]) {
                host.addEventListener(event,
                    () => refreshPreflightDebounced[engine]());
            }
        }
    }
    /* ---------------------------------------------------------------- *
     *  Send to Task setup — the hand-off                                *
     * ---------------------------------------------------------------- *
     * The button writes
     * the files into the folder the projects sidebar has selected, and
     * stops there -- opening Task setup is the person's move (2026-09-19).
     *
     * THROUGH DISK, NEVER IN MEMORY.  `web/tabs.md` § 1 forbids an in-memory
     * "send to tab" hand-off and lists four costs: a result depending on
     * hidden state, re-running silently producing something different, two
     * people getting different answers, export losing information.  Writing
     * the files first keeps all four bought.
     *
     * It lives HERE rather than in its own module because the two things it
     * needs -- the card's structure (`card.structure()`) and
     * `collectParams()` -- are this page's.  A separate module would have to re-derive them,
     * which is the duplication this codebase keeps paying for.
     */
    function _handoverSay(kind, text) {
        const n = document.getElementById("handover-status");
        if (!n) return;
        n.hidden = false;
        // THE ONE STATUS WRITER (`lib/status.js`); page-shell owns the look.
        window.molbuilder.status.set(n, text, kind);
    }

    async function _sendToTaskSetup() {
        /* The guards, the call and the notice handling live in
         * lib/task-handover.js -- ONE door shared with the spectra tab
         * (spectra-migration-plan.md P2).  The dest guards are CONTRACT
         * (`job-contracts.md` § 2.1 / § 2.5), and two copies of a
         * contract drift.  This
         * tab contributes only what it alone knows: its structure door,
         * its engine, its form. */
        const mb = window.molbuilder || {};
        if (!mb.taskHandover) {
            _handoverSay("error", "lib/task-handover.js is not loaded.");
            return;
        }
        const engine = _activeEngine();
        // A VALUE THAT WILL NOT READ AS ITS TYPE is refused here, naming its
        // field -- beside which its caption already says why.
        let params;
        try { params = collectParams(engine); }
        catch (e) {
            _handoverSay("error", e.message);
            return;
        }
        await mb.taskHandover.send({
            projects:  mb.projects,
            say:       _handoverSay,
            // The hand-over's findings in the engine's panel, beside the
            // live preflight's, which are still true of the form it sends.
            standing:     _live[engine],
            // ...and a refusal's first error brought into view.
            showFindings: (issues) => _showFindings(engine, issues, true),
            door:      mb.taskHandover.HANDOVER,
            fields:    { structure: card.structure(), params: params },
            frame:     card.frame(),
            engine:    engine,
            calculation: "optimization",
        });
    }

    /** Which engine's form is showing — the one the user has been filling. */
    function _activeEngine() {
        return (_engineStrip && _engineStrip.active()) || "siesta";
    }

    {
        const _btn = document.getElementById("send-to-task-setup");
        if (_btn) _btn.addEventListener("click", _sendToTaskSetup);
    }
})();
