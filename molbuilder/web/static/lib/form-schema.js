/* Schema-driven form rendering for the Build tab.
 *
 * Consumes the JSON schema produced by the server-side
 * ``molbuilder.web.blueprints._shared.catalogue_to_form_schema``
 * (GET /api/build/schema/<engine>, GET /api/transport/schema) and
 * renders an HTML form
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
 *   * collectForm(container, schema) -- what the form HOLDS, one entry
 *     per field: ``{kgrid: [4,4,1], mesh_cutoff: null, ...}``.  A blank is
 *     ``null`` -- not chosen (form-schema.md § 1.1).  A value that will not
 *     read as its type is refused: it throws an Error naming the field
 *     (``err.field``), a fractional count included.  A field the rung
 *     fixes (``locked``, echoed read-only) is never collected.  An
 *     optional third argument names the fields to read, and only those.
 *
 *   * diffFromDefaults(container, schema) -- which fields hold a value
 *     that is not the kind's recommended one, as
 *     ``[{name, label, current, recommended, unit, help}]``.  A blank
 *     field is skipped (the recommendation already applies), and so is a
 *     field with no default; there is nothing to reset either to.
 *
 *   * fetchSchema(engine) -- thin wrapper around
 *     GET /api/build/schema/<engine> that throws on error and
 *     returns the schema body.
 *
 * Kinds handled (the server's are `_shared.py::_control_for`'s;
 * `comma-floats` has had no emitter since 2026-10-02 -- plan D3):
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

    /* ---------- one field state (form-schema.md § 1.1, plan § 5w K7) ----
     *
     * A field HOLDS a value only where the surface is drawn from the
     * calculation's template and edits it -- the transport tab's shared
     * panel: the template's value, named by its source.  Everywhere else --
     * a new calculation's form, a rung's tab of overrides -- it holds what
     * the person gives it, and nothing until they do.
     *
     * A BLANK FIELD IS NOT CHOSEN, and its hint is what then applies: on a
     * surface of overrides the template's value, with whose it is; an
     * optional item's own blank (its `null_label`, "(auto)"); else the
     * kind's default.  Never the first choice of a list, never a zero in a
     * triple, never an unticked box -- a value the person did not choose is
     * never presented as if they had.  (A select showed its first choice
     * and a triple 0 0 0 until 2026-09-30, and both were sent: the M11
     * review's T-F25.  The Build form drew every default as a value, so a
     * template could not tell the person's 300 Ry from nobody's.) */
    function fieldState(f, ctx) {
        const has = (v) => v !== undefined && v !== null;
        // A field the rung fixes shows the rung's answer and no hint: the
        // bias, answered point by point, shows none rather than a number.
        if (f.locked) return { drawn: f.locked.value, hint: null, hintKind: "" };
        const drawn = !ctx.overrides && has(f.value) ? f.value : null;
        if (ctx.overrides && has(f.value)) {
            return { drawn: drawn, hint: f.value, hintKind: "template" };
        }
        if (f.optional) return { drawn: drawn, hint: null, hintKind: "optional" };
        if (has(f.default)) {
            return { drawn: drawn, hint: f.default, hintKind: "default" };
        }
        return { drawn: drawn, hint: null, hintKind: "" };
    }

    /** A value as a person reads it -- a mesh as "4, 4, 1", a box as on/off. */
    function show(v, f) {
        const unit = f.unit ? " " + f.unit : "";
        if (Array.isArray(v)) return v.join(", ") + unit;
        if (v === true) return "on";
        if (v === false) return "off";
        return String(v) + unit;
    }

    /** The words a source is said in -- the schema's, never a copy. */
    function said(ctx, source) { return ctx.words[source || "unrecorded"] || ""; }

    /** What a blank field says: not chosen, and what then applies. */
    function blankWords(f, st, ctx) {
        const then = st.hintKind === "template"
                ? show(st.hint, f) + " " + said(ctx, f.source)
            : st.hintKind === "optional" ? (f.null_label || "")
            : st.hintKind === "default" ? "recommended " + show(st.hint, f)
            : "";
        return [said(ctx, "default"), then].filter(Boolean).join(" \u00b7 ");
    }

    /** The hint inside a box: what applies when it stays blank. */
    function placeholderOf(f, st) {
        if (Array.isArray(st.hint)) return st.hint.join(", ");
        if (st.hint !== null) return String(st.hint);
        return f.optional ? (f.null_label || "") : "";
    }

    function makeNumber(f, isInt, st) {
        // type=number with step=any handles both ints and floats.
        // step=1 for ints so browser spinners go in integer steps.
        const inp = el("input", {
            id:   f.id,
            type: "number",
            step: isInt ? "1" : (f.step || "any"),
        });
        if (f.min !== undefined) inp.min = f.min;
        if (f.max !== undefined) inp.max = f.max;
        inp.value = st.drawn !== null ? String(st.drawn) : "";
        inp.placeholder = placeholderOf(f, st);
        return inp;
    }

    function makeSelect(f, st, ctx) {
        const sel = el("select", { id: f.id });
        // THE BLANK OPTION, first and always: not chosen.  An optional
        // item's blank is its own answer and says so (`null_label`); any
        // other says what then applies.  Without it a select nobody touched
        // showed -- and sent -- its first choice (T-F25).
        sel.appendChild(el("option", { value: "" },
            (f.optional || f.null_option) ? (f.null_label || "(default)")
                                           : "(" + blankWords(f, st, ctx) + ")"));
        let picked = false;
        for (const c of f.choices) {
            const opt = el("option", { value: String(c) }, String(c));
            if (st.drawn !== null && c === st.drawn) {
                opt.selected = true;
                picked = true;
            }
            sel.appendChild(opt);
        }
        // A VALUE THIS KIND DOES NOT OFFER is shown as itself, marked: a
        // select with no option for it shows another one instead, and the
        // form would say a value nobody holds (`engines/template.md` § 6.3a).
        if (st.drawn !== null && !picked) {
            const opt = el("option", { value: String(st.drawn) },
                           String(st.drawn) + " (not offered here)");
            opt.selected = true;
            sel.appendChild(opt);
        }
        if (st.drawn === null) sel.value = "";
        return sel;
    }

    function makeTriSelect(f, st) {
        // Optional[bool] tri-state: auto/true/false, where "auto" is the
        // item's own blank (None).
        const sel = el("select", { id: f.id });
        const now = st.drawn === true ? "true"
                  : st.drawn === false ? "false" : "auto";
        for (const c of f.choices) {       // ["auto", "true", "false"]
            const opt = el("option", { value: c }, c);
            if (c === now) opt.selected = true;
            sel.appendChild(opt);
        }
        return sel;
    }

    function makeCheckbox(f, st) {
        // An unticked box SAYS "off", so a box nobody answered is drawn
        // INDETERMINATE -- the checkbox's own blank -- and the first click
        // answers it.
        const box = el("input", { id: f.id, type: "checkbox" });
        if (st.drawn === null) box.indeterminate = true;
        else box.checked = Boolean(st.drawn);
        return box;
    }

    function makeText(f, st) {
        const attrs = {
            id: f.id, type: "text",
            value: st.drawn === null ? ""
                 : Array.isArray(st.drawn) ? st.drawn.join(", ")
                 : String(st.drawn),
            autocomplete: "off",
        };
        if (f.pattern) attrs.pattern = f.pattern;
        const inp = el("input", attrs);
        inp.placeholder = placeholderOf(f, st);
        return inp;
    }

    /* (The stage-table field kind -- makeStageTable, its presets, the
     * section wrapper and the collect/setValues arms -- retired at the
     * U6 close, 2026-08-22.  Its Python producer
     * ``_stagespec_to_field_schemas`` died when stages.md § 1.1a made a
     * PySCF ladder N decks, so no schema could carry the kind; the
     * renderer, reached by nothing, stayed until the user's cleanup ask.
     * The live stage table is Task setup's own, hand-rolled in
     * task-setup/viewer.js over task.json.) */


    // The two triple kinds, so the places that special-case a triple ask one
    // question instead of listing both.
    const TRIPLE_KINDS = ["int-triple", "float-triple"];
    function isTriple(kind) { return TRIPLE_KINDS.indexOf(kind) !== -1; }

    function makeTriple(f, isInt, st) {
        // Three labelled number inputs sharing one id prefix.  Each
        // cell carries its own sub-label so kgrid (Tuple[int,int,int])
        // reads as "kx 1  ky 1  kz 1" instead of three anonymous boxes.
        // Sub-ids: f.id + "-" + label, e.g. "p-kgrid-x" / "p-kgrid-y" / "p-kgrid-z";
        // collectForm reassembles into [int, int, int].
        // ``isInt`` splits the step exactly as makeNumber does for the
        // scalars.  A float triple stepping by 1 makes the browser call 0.5
        // invalid before any JS runs, and parseInt then reads it back as 0 --
        // which is the Gamma-centred grid the user was moving off.
        //
        // THE FIELD'S OWN ID is on the wrapper -- one id per field, as every
        // other kind has -- so a finding about the mesh lands beside it;
        // with only the cells' ids it fell back to the card (the K3 review).
        // A blank cell is blank, its hint the component that then applies:
        // `[0, 0, 0]` stood in for a mesh nobody gave until 2026-09-30, and
        // was sent (T-F25).
        const wrap = el("span", { class: "schema-int-triple", id: f.id });
        const drawn = Array.isArray(st.drawn) ? st.drawn : null;
        const hint = Array.isArray(st.hint) ? st.hint : null;
        f.labels.forEach((lab, i) => {
            const cell = el("span", { class: "schema-int-triple-cell" });
            cell.appendChild(el("span", {
                class: "schema-int-triple-label",
            }, lab));
            const cellInput = el("input", {
                id: `${f.id}-${lab}`, type: "number",
                step: isInt ? "1" : "any",
                value: drawn && drawn[i] != null ? drawn[i] : "",
            });
            if (hint && hint[i] != null) cellInput.placeholder = String(hint[i]);
            // Bounds apply PER COMPONENT -- a triple's ``range`` bounds each
            // axis, not their sum.  Missing until 2026-08-15: makeNumber
            // honoured f.min/f.max and this did not, so kgrid accepted 0 and
            // -4 (a Monkhorst-Pack count is a COUNT) and the displacement
            // accepted anything at all, while both declared no range to
            // honour either.  Same two lines as the scalar path, so the two
            // controls cannot drift on what a bound means.
            if (f.min !== undefined) cellInput.min = f.min;
            if (f.max !== undefined) cellInput.max = f.max;
            // A COMPONENT THIS KIND FIXES (`kmesh.fixed`, engines/siesta.md
            // § 6.1) is drawn locked at its value, its reason as the title:
            // a transport calculation's third k component, which no rung
            // reads.  The value still travels -- collectForm reads it like
            // any other -- so the template states it; renderField writes
            // the reason beside the control.
            const held = f.fixed && f.fixed[lab];
            if (held) {
                cellInput.value = String(held.value);
                cellInput.disabled = true;
                cellInput.title = held.why;
            }
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

    /* The caption of one field, from what it holds now: the template's
     * source while it holds the value it was drawn with, *you set this*
     * once it holds another, *not chosen* and what then applies when it is
     * blank -- and the refusal itself, beside the field, when what it holds
     * will not read as its type. */
    function sayState(f, labelEl, st, ctx) {
        const cap = labelEl.querySelector(":scope > .schema-source");
        if (!cap) return;
        let now;
        try {
            now = collectField(f, labelEl);
        } catch (e) {
            labelEl.classList.add("is-invalid");
            cap.classList.remove("is-blank");
            cap.textContent = e.reason || e.message;
            return;
        }
        labelEl.classList.remove("is-invalid");
        cap.classList.toggle("is-blank", now === null);
        cap.textContent = now === null ? blankWords(f, st, ctx)
            : (st.drawn !== null && same(now, st.drawn))
                ? said(ctx, f.source) : said(ctx, "person");
    }

    function renderField(f, ctx) {
        const st = fieldState(f, ctx);
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
        // DRAWN REQUIRED (`required`, form-schema.md § 1.1): the value may
        // still be blank, and the Send refuses until it is answered.
        if (f.required) {
            labelEl.classList.add("is-required");
        }
        let input;
        switch (f.kind) {
            case "checkbox":   input = makeCheckbox(f, st);  break;
            case "int":        input = makeNumber(f, true, st);  break;
            case "number":     input = makeNumber(f, false, st); break;
            case "text":       input = makeText(f, st);      break;
            case "select":     input = makeSelect(f, st, ctx);   break;
            case "tri-select": input = makeTriSelect(f, st); break;
            case "int-triple":   input = makeTriple(f, true, st);  break;
            case "float-triple": input = makeTriple(f, false, st); break;
            case "comma-floats":
                // Variable-length List[float] field (Transport's
                // bias_voltages_v).  Render as a plain text input with
                // a placeholder hinting the comma-separated format;
                // the server-side coercer (``coerce_to_field_type``'s
                // ``Sequence[float]`` branch in _shared.py) parses the
                // string back into a list before the dataclass sees it.
                input = makeText(f, st);
                if (!input.placeholder) {
                    input.setAttribute("placeholder", "0.0, 0.5, 1.0");
                }
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
                input = makeText(f, st);
        }
        // The caption is a SPAN, not a bare text node (2026-09-15).  A
        // text node inside a flex/grid <label> becomes an ANONYMOUS item
        // that no selector can reach, and three defects followed from
        // that one fact: the `.is-advanced` bullet had to be a ::before
        // on the label, which in a column layout is an item of its own
        // and drew the "•" on a line by itself; the engine-key badge
        // stretched to the label's full width and read as a second input
        // box; and a checkbox row could only be `flex-direction: row`,
        // so checkbox + caption + badge + help shared one line and the
        // caption wrapped to three.  With the caption addressable,
        // `form-schema.css` places all three.
        const caption = el("span", {class: "schema-field-text"},
                           labelText(f));
        if (f.kind === "checkbox") {
            labelEl.appendChild(input);
            labelEl.appendChild(caption);
        } else {
            labelEl.appendChild(caption);
            labelEl.appendChild(input);
        }
        if (f.locked) {
            // A FIELD THE RUNG FIXES (`locked`, the catalogue's `role`) is
            // SHOWN and never a control: drawn at the rung's answer, with
            // why (`engines/template.md` § 6.6 obligation 3).  Never
            // collected -- the answer is the rung's, and every door refuses
            // another.
            labelEl.classList.add("is-locked");
            for (const c of labelEl.querySelectorAll("input, select")) {
                c.disabled = true;
            }
        } else {
            // WHOSE THE VALUE IS, beside the field and kept current as it
            // is edited (form-schema.md § 1.1): the template's source, *you
            // set this*, or *not chosen* with what then applies.
            labelEl.appendChild(el("span", { class: "schema-source" }));
            const say = () => sayState(f, labelEl, st, ctx);
            labelEl.addEventListener("input", say);
            labelEl.addEventListener("change", say);
            say();
        }
        const badge = engineKeyBadge(f);
        if (badge) labelEl.appendChild(badge);
        // ...AND RED WHILE IT IS EMPTY -- a typed value, a pick or a restore
        // (`setValues` fires `input`) each clears it.
        if (f.required && input && "value" in input) {
            const mark = () => labelEl.classList.toggle(
                "is-empty", !String(input.value || "").trim());
            input.addEventListener("input", mark);
            mark();
        }
        if (f.locked) {
            labelEl.appendChild(el("span", { class: "lock-reason" },
                                   "fixed: " + f.locked.why));
        }
        // WHY A COMPONENT IS LOCKED, beside the control -- the same
        // `.lock-reason` hint a locked field carries (form-schema.css).
        if (f.fixed) {
            for (const lab of Object.keys(f.fixed)) {
                labelEl.appendChild(el("span", { class: "lock-reason" },
                    "\u21b3 " + lab + " is fixed at " + f.fixed[lab].value
                    + ": " + f.fixed[lab].why));
            }
        }
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
    // The roles come from each item's catalogue ``group``, emitted as
    // ``workflow_group`` by ``_shared.catalogue_to_form_schema``.  Fields whose
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
            subtitle: "WHAT you're computing \u2014 the physical character of "
                    + "the system: charge, spin, metallic vs organic, "
                    + "smearing, and the functional.  Set once per run; "
                    + "doesn't change between stages.",
        },
        "stage": {
            title:    "Convergence targets",
            subtitle: "What counts as converged \u2014 the knobs a staged "
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

    function renderForm(container, schema, opts) {
        if (!container || !schema || !Array.isArray(schema.sections)) {
            throw new Error("form-schema.renderForm: bad container/schema");
        }
        // FOLDABLE CARDS are the CALLER's choice (form-schema.md § 1.3):
        // `opts.foldable` draws each workflow-group card as a <details>
        // whose header is its <summary>, and `opts.folded(role, fields)`
        // says which start closed.  The Build tab passes nothing and gets
        // the open <section> cards it always had; the transport tab folds
        // per rung (transport.md § 3.8.2a).
        opts = opts || {};
        const foldable = !!opts.foldable;
        // WHAT THIS SURFACE'S FIELDS HOLD (form-schema.md § 1.1): the
        // template's values, which it edits -- the default -- or
        // `opts.holds = "overrides"`, a rung's own over the template, whose
        // value is then what a blank field runs.  The words a source is said
        // in are the schema's, the template's one vocabulary.
        const ctx = {
            overrides: opts.holds === "overrides",
            words: schema.source_words || {},
        };
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
        // A section's DESCRIPTION, and how many fields the section has
        // in total -- the two facts PASS 2 needs to decide whether a
        // card may show that description.  See the note where it does.
        const sectionDesc  = new Map();
        const sectionTotal = new Map();

        for (const sect of schema.sections) {
            sectionDesc.set(sect.name, sect.description);
            sectionTotal.set(sect.name, sect.fields.length);
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
            const fieldsInCard = [];
            for (const fs_ of sectMap.values()) fieldsInCard.push(...fs_);
            const folded = foldable && typeof opts.folded === "function"
                && !!opts.folded(role, fieldsInCard);
            const card = el(foldable ? "details" : "section",
                { class: "workflow-group workflow-group--" + role });
            if (foldable) card.open = !folded;
            const header = el(foldable ? "summary" : "header",
                              { class: "workflow-group-header" });
            header.appendChild(el("h3",
                { class: "workflow-group-title" }, meta.title));
            if (foldable) {
                // A folded card still says how much it holds.
                header.appendChild(el("span",
                    { class: "workflow-group-count" },
                    fieldsInCard.length + (fieldsInCard.length === 1
                                           ? " setting" : " settings")));
            }
            card.appendChild(header);
            // A <details> lays its content out in its own slot, so the
            // card's grid cannot reach the fieldsets through it: a
            // foldable card puts everything but the summary in ONE body
            // element, and the body carries the grid (form-schema.css).
            const body = foldable
                ? el("div", { class: "workflow-group-body" }) : card;
            if (foldable) card.appendChild(body);
            body.appendChild(el(
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
                // THE SECTION'S OWN EXPLANATION, which reached the
                // screen on the bare path only until 2026-09-15.  Every
                // field on the transport tab is workflow-group tagged,
                // so every section rendered inside a card -- and all
                // ten paragraphs of `_form_section_descriptions` were
                // written, tested for presence, and displayed NOWHERE.
                //
                // Shown only when this card holds the WHOLE section.  A
                // section split across cards is a SUBSET here, and a
                // paragraph about the whole section is then partly
                // false: "Runtime ... memory budget, CPU thread count,
                // log verbosity" over a card holding only verbosity.
                // Repeating it in each card would say it twice and be
                // wrong twice, so a split section keeps its bare legend
                // and the fix is to stop splitting it.
                const desc = sectionDesc.get(sectName);
                if (desc && fields.length === sectionTotal.get(sectName)) {
                    fs.appendChild(el(
                        "p",
                        { class: "schema-section-desc" },
                        desc,
                    ));
                }
                for (const f of fields) {
                    fs.appendChild(renderField(f, ctx));
                }
                body.appendChild(fs);
            }
            // Per-card issues panel — appended at the bottom of the
            // card so validator findings tagged with this workflow-
            // group land WITH the fields they concern.  Per
            // docs/web/ui-contract.md Rule 2.  Hidden
            // until ``renderIssues`` populates it; tagged with the
            // role so the JS render path can find it via
            // ``[data-workflow-group="<role>"]``.
            body.appendChild(el(
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
                fs.appendChild(renderField(f, ctx));
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

    /* What ONE field holds, read as its type -- `null` when it is blank, not
     * chosen (form-schema.md § 1.1).  A value that will not read is refused,
     * naming the field: a count with a fraction is never rounded (`parseInt`
     * read 4.5 as 4, the K3 review), and a box holding text the browser
     * cannot read is not "blank" (it reports an empty value, and
     * `validity.badInput` is the only witness). */
    function collectField(f, container) {
        const refuse = (why) => {
            const e = new Error((f.label || f.name) + ": " + why);
            e.field = f.name;
            e.reason = why;
            return e;
        };
        if (isTriple(f.kind)) return readTriple(f, container, refuse);
        const elx = container.querySelector("#" + cssEsc(f.id));
        // Schema/DOM mismatch: the schema lists a field whose id has no
        // element.  Warn once; the field holds nothing.
        if (!elx) {
            _warnStale(f.name, "has id '" + f.id + "' but no DOM element");
            return null;
        }
        switch (f.kind) {
            case "checkbox":
                // INDETERMINATE is the box's blank: nobody answered it.
                return elx.indeterminate ? null : !!elx.checked;
            case "int":
            case "number":
                return readNumber(elx, f.kind === "int", refuse);
            case "select": {
                const v = elx.value;
                if (v === "") return null;
                // AN ENUM'S MEMBER KEEPS ITS TYPE (`engines/template.md` § 5):
                // an <option> carries String(member), so it is read back as
                // the member itself -- `unpaired_electrons`' 2 is a number
                // and "free" a word -- and what the form holds is the value
                // the template and the card are about.  (The server's own
                // coercion maps a text "2" to the member too, for a caller
                // that sends one.)
                const m = (f.choices || []).find((c) => String(c) === v);
                return m !== undefined ? m : v;
            }
            case "tri-select": {
                const v = elx.value;
                if (v === "auto" || v === "") return null;
                return v === "true";
            }
            default: {
                // text, and comma-floats -- sent as typed; the server's
                // coercer (_shared.py) reads the list and names the field
                // if it cannot.
                const v = String(elx.value).trim();
                return v === "" ? null : v;
            }
        }
    }

    function readNumber(inp, isInt, refuse) {
        const v = String(inp.value).trim();
        if (v === "") {
            if (inp.validity && inp.validity.badInput) {
                throw refuse("not a number");
            }
            return null;
        }
        const n = Number(v);
        if (!Number.isFinite(n)) throw refuse(v + " is not a number");
        if (isInt && !Number.isInteger(n)) {
            throw refuse(v + " is not a whole number \u2014 a count is "
                         + "never rounded");
        }
        return n;
    }

    /* A triple is ONE value: blank when every component it leaves open is
     * blank, refused when only some are -- a mesh is all three counts or
     * none.  A component the kind fixes reads as the kind's value. */
    function readTriple(f, container, refuse) {
        const isInt = f.kind === "int-triple";
        const labs = f.labels || ["x", "y", "z"];
        const out = [];
        let open = 0, blank = 0;
        for (const lab of labs) {
            const held = f.fixed && f.fixed[lab];
            if (held) {
                out.push(held.value);
                continue;
            }
            open++;
            const sub = container.querySelector("#" + cssEsc(f.id + "-" + lab));
            if (!sub) {
                _warnStale(f.name, "has missing triple sub-input(s)");
                return null;
            }
            const n = readNumber(sub, isInt,
                                 (why) => refuse(lab + " " + why));
            if (n === null) blank++;
            out.push(n);
        }
        if (blank === open) return null;
        if (blank) throw refuse("give every component, or none");
        return out;
    }

    /* `names` (optional) reads those fields alone -- the chemistry card
     * asks for the electronic state, and a half-typed mesh elsewhere on the
     * form is not its refusal to make. */
    function collectForm(container, schema, names) {
        if (!container || !schema || !Array.isArray(schema.sections)) {
            throw new Error("form-schema.collectForm: bad container/schema");
        }
        const out = {};
        let refused = null;
        for (const sect of schema.sections) {
            for (const f of sect.fields) {
                if (f.locked) continue;      // the rung's, never collected
                if (names && names.indexOf(f.name) === -1) continue;
                try {
                    out[f.name] = collectField(f, container);
                } catch (e) {
                    if (!refused) refused = e;
                }
            }
        }
        // The first refusal, naming its field; each field's own caption
        // already says why beside it.
        if (refused) throw refused;
        return out;
    }

    /* What the form holds, FOR SAVING: field by field, a field that will
     * not read keeping the value it was last saved with (``kept``, the
     * previous save, or nothing).  One half-typed field must not cost the
     * rest of the form its save -- a whole-form read that threw wiped a
     * tab's saved form, and stopped every later edit being saved (the K7
     * review).  The Send reads through `collectForm`, which refuses. */
    function heldValues(container, schema, kept) {
        if (!container || !schema || !Array.isArray(schema.sections)) {
            throw new Error("form-schema.heldValues: bad container/schema");
        }
        const out = {};
        for (const sect of schema.sections) {
            for (const f of sect.fields) {
                if (f.locked) continue;      // the rung's, never collected
                try {
                    out[f.name] = collectField(f, container);
                } catch (_) {
                    if (kept && typeof kept === "object" && f.name in kept) {
                        out[f.name] = kept[f.name];
                    }
                }
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
     * Used by the Recommended panel's "Reset ticked" (Optimization tab)
     * and by a tab restoring its saved form.  (The Auto-detect button that
     * also filled forms through here retired on 2026-09-28: a blank charge
     * or spin is now the instruction "work it out", and the chemistry card
     * shows the answer -- `lib/chemistry.js`.)
     */
    function setValues(container, schema, values) {
        if (!container || !schema || !Array.isArray(schema.sections)) {
            throw new Error("form-schema.setValues: bad container/schema");
        }
        if (!values || typeof values !== "object"
            || Array.isArray(values)) return;
        for (const sect of schema.sections) {
            for (const f of sect.fields) {
                if (!(f.name in values) || f.locked) continue;
                const v = values[f.name];
                const blank = v === null || v === undefined;
                // A triple is written through its sub-ids
                // ``<f.id>-<label>``: the element with ``f.id`` is the
                // <span> wrapping the three (where a finding about the
                // field lands), and it has no ``.value`` -- so the standard
                // ``#f.id`` write below would skip the field.
                if (isTriple(f.kind)) {
                    // `null` blanks it -- not chosen; otherwise three values.
                    if (!blank && (!Array.isArray(v) || v.length !== 3)) {
                        continue;
                    }
                    const labs = (Array.isArray(f.labels)
                        && f.labels.length === 3)
                        ? f.labels
                        : ["x", "y", "z"];
                    for (let i = 0; i < 3; i++) {
                        const sub = container.querySelector(
                            "#" + cssEsc(f.id + "-" + labs[i]));
                        if (!sub) continue;
                        // A fixed component keeps the kind's value: the
                        // form collects what will be written, and another
                        // value is refused on every door.
                        if (f.fixed && f.fixed[labs[i]]) continue;
                        sub.value = blank ? "" : String(v[i]);
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
                    // `null` is the box's blank: indeterminate.
                    elx.indeterminate = blank;
                    elx.checked = !blank && Boolean(v);
                } else if (f.kind === "tri-select") {
                    elx.value = v === true ? "true"
                              : v === false ? "false"
                              : blank ? "auto" : String(v);
                } else {
                    // Numbers, selects, text -- `.value` is the writer, and
                    // "" the blank every one of them has.
                    elx.value = blank ? ""
                              : Array.isArray(v) ? v.join(", ") : String(v);
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
     * on the user's behalf.  So is a BLANK field: not chosen, so the
     * recommendation is already what applies (form-schema.md § 1.1).
     */
    function diffFromDefaults(container, schema) {
        const current = collectForm(container, schema);
        const out = [];
        for (const sec of (schema.sections || [])) {
            for (const f of (sec.fields || [])) {
                if (f.locked) continue;
                if (f.default === undefined || f.default === null) continue;
                const now = current[f.name];
                if (now === null || now === undefined) continue;
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
        heldValues:  heldValues,
        fetchSchema: fetchSchema,
        setValues:   setValues,
        diffFromDefaults: diffFromDefaults,
    };
})(typeof window !== "undefined" ? window : this);
