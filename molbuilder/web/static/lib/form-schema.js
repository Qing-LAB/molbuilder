/* Schema-driven form rendering for the Build tab.
 *
 * Consumes the JSON schema produced by the server-side
 * ``molbuilder.web.blueprints._shared.dataclass_to_form_schema``
 * (see GET /api/build/schema/<engine>) and renders an HTML form
 * inside a container element, then later collects the user's
 * values back into a flat object whose keys match the dataclass
 * field names.
 *
 * Public API (via window.molbuilder.formSchema):
 *
 *   * renderForm(container, schema) -- replaces container's
 *     contents with a stack of <fieldset> sections holding the
 *     schema's fields.  Each input's id matches schema field.id
 *     (typically "<prefix>-<field-name>"), so the existing
 *     compatibility engine + sessionStorage persistence keep
 *     working unchanged.
 *
 *   * collectForm(container, schema) -- walks the schema and
 *     reads the current DOM values back, returning a dict like
 *     ``{system_label: "siesta", kgrid: [1,1,1], spin_total: null, ...}``
 *     that the existing build endpoints accept verbatim.
 *
 *   * diffFromDefaults(container, schema) -- which fields are not
 *     at the schema's recommended value, as
 *     ``[{name, label, current, recommended, unit, help}]``.  Fields
 *     with no default are skipped; there is nothing to reset them to.
 *
 *   * fetchSchema(engine) -- thin wrapper around
 *     GET /api/build/schema/<engine> that throws on error and
 *     returns the schema body.
 *
 * Kinds handled (mirrors _shared.py::_field_to_schema):
 *
 *   checkbox    : <input type=checkbox>
 *   int         : <input type=number step=1>          (with null option if optional)
 *   number      : <input type=number step=any>        (with null option if optional)
 *   text        : <input type=text>                   (pattern= attribute respected)
 *   select      : <select> with one <option> per choice
 *                 (with null option if optional)
 *   tri-select  : <select> auto / true / false        (Optional[bool])
 *   int-triple  : three <input type=number step=1>    (Tuple[int,int,int], e.g. kgrid)
 *   float-triple : three <input type=number step=any> (Tuple[float,float,float],
 *                  e.g. kgrid_displacement — 0.5 must survive)
 *                  (List[<dataclass>], e.g. PySCFConfig.stages)
 *   comma-floats : comma-separated list of floats
 *                  (List[float], e.g. bias voltages)
 *
 * The renderer never invents a kind; if the server adds a new
 * one we fall through to a plain text input and log a warning so
 * the missing case surfaces at integration time rather than
 * silently producing a broken control.
 */
(function (root) {
    "use strict";

    /* ---------- internal helpers ---------- */

    function el(tag, attrs, ...children) {
        const e = document.createElement(tag);
        if (attrs) {
            for (const k in attrs) {
                // Defense in depth: refuse keys that would set an
                // event-handler attribute (onclick / onerror / ...)
                // or open a code-injection sink (innerHTML /
                // outerHTML / srcdoc).  All current callers pass
                // hardcoded keys like "id", "type", "value" -- the
                // refusal here is a tripwire for future misuse.
                if (/^on/i.test(k)
                        || k === "innerHTML"
                        || k === "outerHTML"
                        || k === "srcdoc") {
                    console.error(
                        "[form-schema.el] refusing dangerous attr key: "
                        + k
                    );
                    continue;
                }
                if (k === "class") {
                    e.className = attrs[k];
                } else if (k === "for") {
                    e.setAttribute("for", attrs[k]);
                } else if (k in e) {
                    // direct DOM property where supported (avoids
                    // attribute / property mismatch for booleans like
                    // .disabled and .checked).
                    e[k] = attrs[k];
                } else {
                    e.setAttribute(k, attrs[k]);
                }
            }
        }
        for (const c of children) {
            if (c == null) continue;
            e.appendChild(
                typeof c === "string" ? document.createTextNode(c) : c
            );
        }
        return e;
    }

    function labelText(f) {
        return f.unit ? `${f.label} (${f.unit})` : f.label;
    }

    /**
     * Render the source-of-truth engine-keyword tag next to a label,
     * if the field carries an ``engine_key`` from the schema.  The
     * tag is the actual keyword (or block name) the field writes
     * into the generated input file -- gives the user a direct map
     * from UI to generated script to the engine's manual + error
     * messages.  Skipped when ``engine_key`` matches ``label`` (the
     * label IS already the keyword, e.g. "MeshCutoff") to avoid
     * duplicate noise.  Returns null when no tag is warranted.
     */
    function engineKeyBadge(f) {
        const key = (f.engine_key || "").trim();
        if (!key) return null;
        const lbl = (f.label || "").trim();
        // Skip the badge when the label IS the keyword.  Two forms
        // count as "is the keyword" -- (a) exact match
        // ("MeshCutoff") and (b) match with a unit suffix
        // ("MeshCutoff (Ry)").  Without (b) the label-text rendered
        // by labelText() includes the unit, and the comparison
        // ``key.toLowerCase() === lbl.toLowerCase()`` would never
        // hit for any unit-bearing field -- so MeshCutoff /
        // PAO.EnergyShift / DM.Tolerance etc. all showed a duplicate
        // badge of the same text right of the label.  Caught by the
        // 2026-05-26 review.
        const lblBare = lbl.replace(/\s*\([^)]*\)\s*$/, "").trim();
        if (key.toLowerCase() === lbl.toLowerCase()) return null;
        if (key.toLowerCase() === lblBare.toLowerCase()) return null;
        const code = document.createElement("code");
        code.className = "schema-engine-key";
        // ``(molbuilder ...)`` markers tell the user "no engine equivalent
        // -- this knob only affects molbuilder's preprocessing / wrapper
        // / filename".  Tag with a class so the stylesheet can render
        // them differently (dashed border, muted text) and the user
        // doesn't go looking for them in the SIESTA / PySCF manual.
        if (key.startsWith("(molbuilder")) {
            code.classList.add("is-molbuilder-only");
            code.title = "molbuilder-only knob -- no engine keyword";
        } else {
            code.title = "Writes this keyword into the generated input file";
        }
        code.textContent = key;
        return code;
    }

    function makeNumber(f, isInt) {
        // type=number with step=any handles both ints and floats.
        // step=1 for ints so browser spinners go in integer steps.
        const inp = el("input", {
            id:   f.id,
            type: "number",
            step: isInt ? "1" : (f.step || "any"),
        });
        if (f.min !== undefined) inp.min = f.min;
        if (f.max !== undefined) inp.max = f.max;
        if (f.default !== null && f.default !== undefined) {
            inp.value = f.default;
        } else if (f.optional) {
            // Empty input means null for Optional[int]/Optional[float].
            inp.value = "";
            inp.placeholder = f.null_label || "(default)";
        }
        return inp;
    }

    function makeSelect(f) {
        const sel = el("select", { id: f.id });
        if (f.optional || f.null_option) {
            // First option is the "null" sentinel; value="" → null on collect.
            sel.appendChild(el("option", { value: "" }, f.null_label || "(default)"));
        }
        for (const c of f.choices) {
            const opt = el("option", { value: String(c) }, String(c));
            if (c === f.default) opt.selected = true;
            sel.appendChild(opt);
        }
        return sel;
    }

    function makeTriSelect(f) {
        // Optional[bool] tri-state: auto/true/false; default
        // mirrors the dataclass default (None → "auto").
        const sel = el("select", { id: f.id });
        const defStr = f.default === null || f.default === undefined
            ? "auto"
            : (f.default ? "true" : "false");
        for (const c of f.choices) {       // ["auto", "true", "false"]
            const opt = el("option", { value: c }, c);
            if (c === defStr) opt.selected = true;
            sel.appendChild(opt);
        }
        return sel;
    }

    function makeCheckbox(f) {
        return el("input", {
            id: f.id, type: "checkbox", checked: Boolean(f.default),
        });
    }

    function makeText(f) {
        const attrs = {
            id: f.id, type: "text",
            value: f.default == null ? "" : String(f.default),
            autocomplete: "off",
        };
        if (f.pattern) attrs.pattern = f.pattern;
        return el("input", attrs);
    }

    /* (The stage-table field kind -- makeStageTable, its presets, the
     * section wrapper and the collect/setValues arms -- retired at the
     * U6 close, 2026-08-22.  Its Python producer
     * ``_stagespec_to_field_schemas`` died when stages.md § 1.1a made a
     * PySCF ladder N decks, so no schema could carry the kind; the
     * renderer was recorded as reached-by-nothing in
     * tests/test_stage_vocabulary.py until the user's cleanup ask.
     * The live stage table is Task setup's own, hand-rolled in
     * task-setup/viewer.js over task.json.) */


    // The two triple kinds, so the places that special-case a triple ask one
    // question instead of listing both.
    const TRIPLE_KINDS = ["int-triple", "float-triple"];
    function isTriple(kind) { return TRIPLE_KINDS.indexOf(kind) !== -1; }

    function makeTriple(f, isInt) {
        // Three labelled number inputs sharing one id prefix.  Each
        // cell carries its own sub-label so kgrid (Tuple[int,int,int])
        // reads as "kx 1  ky 1  kz 1" instead of three anonymous boxes.
        // Sub-ids: f.id + "-" + label, e.g. "p-k-x" / "p-k-y" / "p-k-z";
        // collectForm reassembles into [int, int, int].
        // ``isInt`` splits the step exactly as makeNumber does for the
        // scalars.  A float triple stepping by 1 makes the browser call 0.5
        // invalid before any JS runs, and parseInt then reads it back as 0 --
        // which is the Gamma-centred grid the user was moving off.
        const wrap = el("span", { class: "schema-int-triple" });
        const defaults = Array.isArray(f.default) ? f.default : [0, 0, 0];
        f.labels.forEach((lab, i) => {
            const cell = el("span", { class: "schema-int-triple-cell" });
            cell.appendChild(el("span", {
                class: "schema-int-triple-label",
            }, lab));
            const cellInput = el("input", {
                id: `${f.id}-${lab}`, type: "number",
                step: isInt ? "1" : "any",
                value: defaults[i] != null ? defaults[i] : "",
            });
            // Bounds apply PER COMPONENT -- a triple's ``range`` bounds each
            // axis, not their sum.  Missing until 2026-08-15: makeNumber
            // honoured f.min/f.max and this did not, so kgrid accepted 0 and
            // -4 (a Monkhorst-Pack count is a COUNT) and the displacement
            // accepted anything at all, while both declared no range to
            // honour either.  Same two lines as the scalar path, so the two
            // controls cannot drift on what a bound means.
            if (f.min !== undefined) cellInput.min = f.min;
            if (f.max !== undefined) cellInput.max = f.max;
            cell.appendChild(cellInput);
            wrap.appendChild(cell);
        });
        return wrap;
    }

    /**
     * Long help-text strings (psml_lib at ~39 lines, basis_size's
     * convergence advice, etc.) used to live in ``title=`` -- browsers
     * truncate native tooltips to ~one OS-dependent line and the
     * paragraph-length contents were unreadable.  For multi-line help
     * we now render a click-to-expand ``<details>`` element with the
     * full text in a styled ``.schema-help-body``.  Short help still
     * goes into ``title=`` (single-line tooltip is fine for one-liners).
     * Threshold: 80 chars or first newline.
     */
    function helpIsLong(help) {
        if (!help) return false;
        if (help.indexOf("\n") !== -1) return true;
        return help.length > 80;
    }

    function makeHelpDetails(help, refs) {
        const det = document.createElement("details");
        det.className = "schema-help";
        const sum = document.createElement("summary");
        sum.textContent = "ⓘ help";
        sum.className = "schema-help-summary";
        det.appendChild(sum);
        // Preserve the source's line breaks (browser default for
        // <pre> would also work; div with white-space:pre-wrap reads
        // a bit nicer + lets us style border/background).
        const body = document.createElement("div");
        body.className = "schema-help-body";
        body.textContent = help;
        det.appendChild(body);
        // References -- resolved server-side from the one bibliography
        // (docs/science/references.bib); each renders as title + a DOI
        // link the user can follow to the paper.
        if (Array.isArray(refs) && refs.length) {
            const list = document.createElement("ul");
            list.className = "schema-help-refs";
            for (const c of refs) {
                const li = document.createElement("li");
                li.textContent = (c.title ? c.title + " — " : "") + (c.text || "");
                if (c.doi) {
                    const a = document.createElement("a");
                    a.href = "https://doi.org/" + c.doi;
                    a.target = "_blank";
                    a.rel = "noopener";
                    a.textContent = "doi:" + c.doi;
                    li.appendChild(document.createTextNode("  "));
                    li.appendChild(a);
                }
                list.appendChild(li);
            }
            det.appendChild(list);
        }
        // Click-anywhere-on-summary toggles the details; stop the
        // event from bubbling to the parent <label> (which would
        // forward clicks to the input -- e.g. a checkbox label
        // would flip the checkbox just because the user wanted to
        // read help).
        sum.addEventListener("click", (e) => e.stopPropagation());
        return det;
    }

    function renderField(f) {
        // Build a single <label> wrapping the input.  Checkbox lays
        // out as "[x] Label" -- the checkbox comes BEFORE the label
        // text; everything else lays out as "Label: <input>".
        const labelEl = el("label", {
            class: "schema-field schema-field-" + f.kind,
            // Short help in title= (single-line native tooltip); long
            // help moves below the input via <details>.
            title: helpIsLong(f.help) ? "" : (f.help || ""),
        });
        if (f.tier === "advanced") {
            labelEl.classList.add("is-advanced");
        }
        let input;
        switch (f.kind) {
            case "checkbox":   input = makeCheckbox(f);  break;
            case "int":        input = makeNumber(f, true);  break;
            case "number":     input = makeNumber(f, false); break;
            case "text":       input = makeText(f);      break;
            case "select":     input = makeSelect(f);    break;
            case "tri-select": input = makeTriSelect(f); break;
            case "int-triple":   input = makeTriple(f, true);  break;
            case "float-triple": input = makeTriple(f, false); break;
            case "comma-floats":
                // Variable-length List[float] field (Transport's
                // bias_voltages_v).  Render as a plain text input with
                // a placeholder hinting the comma-separated format;
                // the server-side coercer (``coerce_to_field_type``'s
                // ``Sequence[float]`` branch in _shared.py) parses the
                // string back into a list before the dataclass sees it.
                input = makeText(f);
                input.setAttribute("placeholder", "0.0, 0.5, 1.0");
                input.classList.add("schema-input-comma-floats");
                break;
            default:
                // Unknown kind: log + fallback to text so the form
                // still renders and the missing case is visible.
                if (root.console && root.console.warn) {
                    root.console.warn(
                        "form-schema: unknown kind",
                        f.kind, "for field", f.name
                    );
                }
                input = makeText(f);
        }
        if (f.kind === "checkbox") {
            labelEl.appendChild(input);
            labelEl.appendChild(document.createTextNode(" " + labelText(f)));
        } else {
            labelEl.appendChild(document.createTextNode(labelText(f) + " "));
            labelEl.appendChild(input);
        }
        const badge = engineKeyBadge(f);
        if (badge) labelEl.appendChild(badge);
        // Long help: append the click-to-expand <details> AFTER the
        // input + badge so it doesn't push them out of the layout grid.
        // f.refs rides along (U5, 2026-08-21): this is the path most
        // fields take, and dropping the parameter here meant the
        // catalogue's citations rendered nowhere reachable.
        if (helpIsLong(f.help)) {
            labelEl.appendChild(makeHelpDetails(f.help, f.refs));
        }
        return labelEl;
    }

    /* ---------- public API ---------- */

    // Workflow-group metadata (2026-06-13).  Each .workflow-group--<role>
    // card gets a label + a subtitle explaining "what changes when".
    // The roles come from the field-level ``workflow_group`` metadata
    // emitted by ``_shared.dataclass_to_form_schema``.  Fields whose
    // section contains only UNTAGGED fields render bare (no workflow-
    // group wrapper).
    const WORKFLOW_GROUP_META = {
        // Added 2026-08-15 (user).  These two are what a calculation cannot
        // be built without -- it needs a name for its output files and a
        // directory to find pseudopotentials in -- and they were the two
        // hardest things on the page to find: a card orders its contents by
        // `category`, so the label sorted under *procedure* near the bottom
        // of Run profile and the pseudopotential directory under *method* in
        // the middle, while Run profile's own subtitle promised both.
        "setup": {
            title:    "Setup",
            subtitle: "Start here.  What this run is CALLED, and where its "
                    + "pseudopotentials come from.  Nothing can be built "
                    + "until both are answered, and every output file is "
                    + "named after the first.",
        },
        "profile": {
            title:    "Run profile",
            subtitle: "WHAT you're computing — the physical character of "
                    + "the system: charge, spin, metallic vs organic, "
                    + "smearing, and the functional.  Set once per run; "
                    + "doesn't change between stages.",
        },
        "stage": {
            title:    "Convergence targets",
            subtitle: "What counts as converged — the knobs a staged "
                    + "sequence TIGHTENS as it goes.  This is the set the "
                    + "staging surface steps; nothing on this page steps it.",
        },
        "budget": {
            title:    "Compute & budget",
            subtitle: "How much compute am I willing to spend?  "
                    + "Iteration caps + parallel layout (MPI ranks, "
                    + "OMP threads, memory).  Scales with system size; "
                    + "does NOT change what counts as converged.",
        },
        // Added 2026-08-15.  Not a home for leftovers: FOUR of these were
        // already on the form, mis-filed under "what you're computing"
        // (write-coor-xmol, write-md-history, write-hs, verbose-comments on
        // SIESTA; chkfile, log-file, verbose on PySCF), and seven more had no
        // card at all and rendered loose below the three.  The three cards
        // answer *what am I computing*, *how tight*, and *how much compute* —
        // there were always four questions and only three cards.
        "output": {
            title:    "Output files",
            subtitle: "What the run WRITES — trajectories, logs, geometry "
                    + "snapshots, and which files are staged beside the "
                    + "input.  Changes what you get back, never the answer.",
        },
    };

    // Render-order of the three workflow-group cards (2026-06-13
    // reorder, after user feedback):
    //   1. Run profile — "what is this run?" identity + character
    //   2. Stage       — "what am I converging to right now?"
    //   3. Budget      — "how much patience?"
    // Reads naturally top-to-bottom on first encounter; profile is
    // the foundation that the other two iterate against.  Untagged
    // sections render in their original schema order AFTER the
    // three cards.
    //   4. Output      — "what do I get back?"  Last because it is the
    //                    only one you can decide after the physics.
    //   0. Setup       — "what is it called, and where are the pseudos?"
    //                    First because nothing downstream can be answered
    //                    without it (2026-08-15).
    const WORKFLOW_GROUP_ORDER = ["setup", "profile", "stage", "budget",
                                  "output"];

    function renderForm(container, schema) {
        if (!container || !schema || !Array.isArray(schema.sections)) {
            throw new Error("form-schema.renderForm: bad container/schema");
        }
        // Fresh render -> schema and DOM are presumed to match, so
        // clear the stale-warning cache.  Any actual mismatch on
        // the next collectForm will re-warn.
        _staleWarnings.clear();
        container.innerHTML = "";

        // Two-pass strategy (2026-06-13 restructure):
        //
        //   PASS 1: walk every field once, bucketing into
        //     - tagged fields → one of three role buckets, keyed by
        //       (role, original_section_name) so we can render with
        //       a legend like "SCF" inside the "Stage convergence
        //       target" card.
        //     - untagged fields → original section, rendered bare
        //       AFTER the three workflow-group cards.
        //
        //   PASS 2: render in fixed order (stage → budget → system →
        //     untagged sections) so the visual hierarchy makes the
        //     "switching the stage selector touches the stage card
        //     only" claim self-evident at a glance.
        //
        // Pre-2026-06-13 the form mixed stage / budget / system
        // fields inside the same SCF + Relaxation fieldsets, so
        // switching the stage preset silently rewrote budget +
        // system fields too.  That was the bug class the user
        // reported on Au-BDT-Au.
        const tagged = {};
        for (const role of WORKFLOW_GROUP_ORDER) {
            tagged[role] = new Map();
        }
        const untagged = [];

        for (const sect of schema.sections) {
            const remainingFields = [];
            for (const f of sect.fields) {
                const role = f.workflow_group;
                if (role && WORKFLOW_GROUP_META[role]) {
                    if (!tagged[role].has(sect.name)) {
                        tagged[role].set(sect.name, []);
                    }
                    tagged[role].get(sect.name).push(f);
                } else {
                    remainingFields.push(f);
                }
            }
            if (remainingFields.length > 0) {
                // Carry the section metadata + the leftover untagged
                // fields so we can render the section bare with its
                // original description.
                untagged.push({
                    name:        sect.name,
                    description: sect.description,
                    fields:      remainingFields,
                });
            }
        }

        // PASS 2 — Render workflow-group cards in fixed order.
        for (const role of WORKFLOW_GROUP_ORDER) {
            const sectMap = tagged[role];
            if (sectMap.size === 0) continue;
            const meta = WORKFLOW_GROUP_META[role];
            const card = el("section",
                { class: "workflow-group workflow-group--" + role });
            const header = el("header", { class: "workflow-group-header" });
            header.appendChild(el("h3",
                { class: "workflow-group-title" }, meta.title));
            card.appendChild(header);
            card.appendChild(el(
                "p",
                { class: "workflow-group-subtitle" },
                meta.subtitle,
            ));
            // Render each original section's tagged-field subset as
            // a mini-fieldset inside the card.  The legend keeps the
            // user's mental map ("the DM.Tolerance field belongs to
            // SCF") while moving it into the workflow-group context.
            for (const [sectName, fields] of sectMap.entries()) {
                const fs = el("fieldset", { class: "schema-section" });
                fs.appendChild(el("legend", null, sectName));
                for (const f of fields) {
                    fs.appendChild(renderField(f));
                }
                card.appendChild(fs);
            }
            // Per-card issues panel — appended at the bottom of the
            // card so validator findings tagged with this workflow-
            // group land WITH the fields they concern.  Per
            // docs/web/ui-contract.md Rule 2.  Hidden
            // until ``renderIssues`` populates it; tagged with the
            // role so the JS render path can find it via
            // ``[data-workflow-group="<role>"]``.
            card.appendChild(el(
                "ul",
                { "class":                "issues-panel card-issues",
                  "data-workflow-group":  role,
                  "hidden":               "",
                  "aria-live":            "polite" },
            ));
            container.appendChild(card);
        }

        // PASS 2 (cont.) — Render untagged sections in their
        // original schema order, bare (no workflow-group wrapper).
        for (const sect of untagged) {
            const fs = el("fieldset", { class: "schema-section" });
            fs.appendChild(el("legend", null, sect.name));
            if (sect.description) {
                fs.appendChild(el(
                    "p",
                    { class: "schema-section-desc" },
                    sect.description,
                ));
            }
            for (const f of sect.fields) {
                fs.appendChild(renderField(f));
            }
            container.appendChild(fs);
        }
    }

    /* Tracks fields we've already warned about per-load so a stale
     * schema doesn't spam the console with one warning per call to
     * collectForm.  Cleared whenever renderForm runs (a fresh render
     * is presumed to match the schema). */
    const _staleWarnings = new Set();

    function _warnStale(fieldName, reason) {
        if (_staleWarnings.has(fieldName)) return;
        _staleWarnings.add(fieldName);
        if (root.console && root.console.warn) {
            root.console.warn(
                "form-schema.collectForm: field '" + fieldName +
                "' " + reason + " (stale schema?)"
            );
        }
    }

    function collectField(f, container) {
        const elx = container.querySelector("#" + cssEsc(f.id));
        const optional = !!f.optional;
        // Schema/DOM mismatch: schema lists a field whose id has no
        // matching element.  Warn once, fall back to the schema's
        // declared default so the rest of the form still submits
        // sensibly.  int-triple handles this per-sub-input below.
        if (!elx && !isTriple(f.kind)) {
            _warnStale(f.name, "has id '" + f.id + "' but no DOM element");
            return f.default !== undefined ? f.default : null;
        }
        switch (f.kind) {
            case "checkbox":
                return !!elx.checked;
            case "int": {
                const v = elx.value.trim();
                if (v === "" && optional) return null;
                if (v === "") return null;
                const n = parseInt(v, 10);
                return Number.isFinite(n) ? n : null;
            }
            case "number": {
                const v = elx.value.trim();
                if (v === "" && optional) return null;
                if (v === "") return null;
                const n = parseFloat(v);
                return Number.isFinite(n) ? n : null;
            }
            case "text":
                return String(elx.value).trim();
            case "comma-floats":
                // Send the raw string; the server-side coercer
                // (_shared.py Sequence[float] branch) parses it into
                // List[float] before the dataclass sees it.
                return String(elx.value).trim();
            case "select": {
                const v = elx.value;
                // Empty value on an Optional select -> null.
                if (v === "" && (optional || f.null_option)) return null;
                return v;
            }
            case "tri-select": {
                const v = elx.value;
                if (v === "auto" || v === "") return null;
                return v === "true";
            }
            case "int-triple":
            case "float-triple": {
                // Read each component back with the parser its type asks for.
                // parseInt on a float triple truncates silently.
                const isInt = f.kind === "int-triple";
                const parse = isInt ? (s) => parseInt(s, 10) : parseFloat;
                const labs = f.labels || ["x", "y", "z"];
                const defaults = Array.isArray(f.default)
                    ? f.default : [0, 0, 0];
                const out = [];
                let anyMissing = false;
                labs.forEach((lab, i) => {
                    const subEl = container.querySelector(
                        "#" + cssEsc(f.id + "-" + lab)
                    );
                    if (!subEl) {
                        anyMissing = true;
                        out.push(defaults[i] != null ? defaults[i] : 0);
                        return;
                    }
                    const v = subEl.value.trim();
                    const n = v === "" ? null : parse(v);
                    out.push(Number.isFinite(n) ? n
                             : (defaults[i] != null ? defaults[i] : 0));
                });
                if (anyMissing) {
                    _warnStale(f.name, "has missing int-triple sub-input(s)");
                }
                return out;
            }
            default:
                return elx.value;
        }
    }

    function collectForm(container, schema) {
        if (!container || !schema || !Array.isArray(schema.sections)) {
            throw new Error("form-schema.collectForm: bad container/schema");
        }
        const out = {};
        for (const sect of schema.sections) {
            for (const f of sect.fields) {
                out[f.name] = collectField(f, container);
            }
        }
        return out;
    }

    async function fetchSchema(engine, opts) {
        // ``opts.calculation`` (optional) narrows the form to the
        // parameters that apply to that calculation KIND
        // (template.md § 6.3's `calculations` key); absent means
        // optimization, exactly as the server defaults it.
        // (A ``structurePath`` forward and a ``body.notice`` hook
        // stood here for the retired sidecar-prefill flow -- the
        // server dropped the query silently and no route ever
        // emitted the notice; both retired at the U6 close.)
        const calculation = (opts && opts.calculation) || "";
        let url = "/api/build/schema/" + encodeURIComponent(engine);
        if (calculation) {
            url += "?calculation=" + encodeURIComponent(calculation);
        }
        const r = await fetch(url);
        const body = await r.json();
        if (!r.ok || !body.ok) {
            throw new Error(
                "form-schema.fetchSchema: server returned "
                + r.status + " — " + (body.error || "")
            );
        }
        return body.schema;
    }

    /* CSS.escape() polyfill for older browsers; modern Chrome /
     * Firefox / Safari already ship it natively. */
    function cssEsc(s) {
        if (typeof CSS !== "undefined" && typeof CSS.escape === "function") {
            return CSS.escape(s);
        }
        return String(s).replace(/[^a-zA-Z0-9_-]/g, (c) => "\\" + c);
    }

    /**
     * Apply a values object to the rendered form.  Keys in
     * ``values`` match schema field ``name``s; missing keys leave
     * existing values alone.  Fires an ``input`` event on each
     * changed control so dirty-tracking listeners observe the
     * programmatic change.
     *
     * Used by the Auto-detect button (Optimization tab) to populate
     * (charge, spin, method, ...) from
     * ``/api/structure/analyze``'s ``suggested.<engine>`` block in
     * one call.  See docs/science/validation.md
     */
    function setValues(container, schema, values) {
        if (!container || !schema || !Array.isArray(schema.sections)) {
            throw new Error("form-schema.setValues: bad container/schema");
        }
        if (!values || typeof values !== "object"
            || Array.isArray(values)) return;
        for (const sect of schema.sections) {
            for (const f of sect.fields) {
                if (!(f.name in values)) continue;
                const v = values[f.name];
                // int-triple uses sub-ids ``<f.id>-<label>`` — there
                // is no parent element with ``f.id`` (makeIntTriple
                // wraps the three sub-inputs in an unidentified
                // <span>), so the standard ``#f.id`` lookup below
                // would return null and silently skip the field.
                // Handle int-triple via its own sub-id loop.
                if (isTriple(f.kind)) {
                    if (!Array.isArray(v) || v.length !== 3) continue;
                    const labs = (Array.isArray(f.labels)
                        && f.labels.length === 3)
                        ? f.labels
                        : ["x", "y", "z"];
                    for (let i = 0; i < 3; i++) {
                        const sub = container.querySelector(
                            "#" + cssEsc(f.id + "-" + labs[i]));
                        if (!sub) continue;
                        sub.value = String(v[i]);
                        try {
                            sub.dispatchEvent(new Event("input",
                                { bubbles: true }));
                            sub.dispatchEvent(new Event("change",
                                { bubbles: true }));
                        } catch (_) { /* legacy browser */ }
                    }
                    continue;
                }
                const elx = container.querySelector("#" + cssEsc(f.id));
                if (!elx) continue;
                if (f.kind === "checkbox") {
                    elx.checked = Boolean(v);
                } else {
                    // Numbers, selects, tri-selects, text — all
                    // accept .value as the canonical writer.
                    elx.value = v === null || v === undefined
                        ? "" : String(v);
                }
                // Notify dirty-trackers / live-preview consumers.
                try {
                    elx.dispatchEvent(new Event("input", { bubbles: true }));
                    elx.dispatchEvent(new Event("change", { bubbles: true }));
                } catch (_) { /* old browsers without Event ctor */ }
            }
        }
    }


    /* ---- diffFromDefaults(container, schema) ------------------------
     * Which fields are NOT at the catalogue's recommended value, and
     * what each would go back to.
     *
     * This belongs beside collectForm/setValues rather than in a tab,
     * because it is the same pair of facts those two already own: what
     * the DOM currently holds, and what the schema says.  A tab that
     * compared them itself would need its own reader for every kind
     * this module already handles.
     *
     * A field with no `default` is SKIPPED -- there is nothing to
     * recommend, so offering to reset it would mean blanking a value
     * on the user's behalf.
     */
    function diffFromDefaults(container, schema) {
        const current = collectForm(container, schema);
        const out = [];
        for (const sec of (schema.sections || [])) {
            for (const f of (sec.fields || [])) {
                if (f.default === undefined || f.default === null) continue;
                const now = current[f.name];
                if (same(now, f.default)) continue;
                out.push({
                    name: f.name,
                    label: f.label || f.name,
                    current: now,
                    recommended: f.default,
                    unit: f.unit || "",
                    help: f.help || "",
                });
            }
        }
        return out;
    }

    /* Values arrive typed (numbers, arrays, booleans), so comparing the
     * JSON is right for the composite kinds and safe for the scalars.
     * The one trap is 300 vs "300" -- a field the user typed into may
     * read back as text -- so numbers are compared numerically first. */
    function same(a, b) {
        if (typeof b === "number" && a !== null && a !== "" && !isNaN(Number(a))) {
            return Number(a) === b;
        }
        return JSON.stringify(a) === JSON.stringify(b);
    }

    root.molbuilder = root.molbuilder || {};
    root.molbuilder.formSchema = {
        renderForm:  renderForm,
        collectForm: collectForm,
        fetchSchema: fetchSchema,
        setValues:   setValues,
        diffFromDefaults: diffFromDefaults,
    };
})(typeof window !== "undefined" ? window : this);
