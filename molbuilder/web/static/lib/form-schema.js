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

    function makeIntTriple(f) {
        // Three labelled number inputs sharing one id prefix.  Each
        // cell carries its own sub-label so kgrid (Tuple[int,int,int])
        // reads as "kx 1  ky 1  kz 1" instead of three anonymous boxes.
        // Sub-ids: f.id + "-" + label, e.g. "p-k-x" / "p-k-y" / "p-k-z";
        // collectForm reassembles into [int, int, int].
        const wrap = el("span", { class: "schema-int-triple" });
        const defaults = Array.isArray(f.default) ? f.default : [0, 0, 0];
        f.labels.forEach((lab, i) => {
            const cell = el("span", { class: "schema-int-triple-cell" });
            cell.appendChild(el("span", {
                class: "schema-int-triple-label",
            }, lab));
            cell.appendChild(el("input", {
                id: `${f.id}-${lab}`, type: "number", step: "1",
                value: defaults[i] != null ? defaults[i] : "",
            }));
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

    function makeHelpDetails(help) {
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
            case "int-triple": input = makeIntTriple(f); break;
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
        if (helpIsLong(f.help)) {
            labelEl.appendChild(makeHelpDetails(f.help));
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
        "profile": {
            title:    "Run profile",
            subtitle: "WHAT you're computing — identity (label, "
                    + "pseudopotentials, charge) and physical character "
                    + "(metallic vs organic, open-shell, smearing).  "
                    + "Set once per run; doesn't change between stages.",
        },
        "stage": {
            title:    "Stage convergence target",
            subtitle: "These TIGHTEN as you go 1 → 2 → 3.  Switching "
                    + "the stage selector mutates ONLY this card.",
        },
        "budget": {
            title:    "Resource budget",
            subtitle: "Caps on patience — does NOT change what counts "
                    + "as converged.  Scale with system size.",
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
    const WORKFLOW_GROUP_ORDER = ["profile", "stage", "budget"];

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
        if (!elx && f.kind !== "int-triple") {
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
            case "int-triple": {
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
                    const n = v === "" ? null : parseInt(v, 10);
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
        // ``opts.structurePath`` (optional) is forwarded as a
        // ``?structure_path=…`` query so the server can pre-fill
        // structure-dependent field defaults (e.g. the /spectra
        // schema overrides ``frozen_indices.default`` from the
        // .molstruct.json sidecar's ``frozen_atoms`` -- design.md
        // "Sidecar-driven boundary conditions").  Callers that
        // don't pass it (or pass an empty value) get the static
        // schema, same as before.
        const structurePath = (opts && opts.structurePath) || "";
        let url = "/api/build/schema/" + encodeURIComponent(engine);
        if (structurePath) {
            url += "?structure_path=" + encodeURIComponent(structurePath);
        }
        const r = await fetch(url);
        const body = await r.json();
        if (!r.ok || !body.ok) {
            throw new Error(
                "form-schema.fetchSchema: server returned "
                + r.status + " — " + (body.error || "")
            );
        }
        // ``body.notice`` is an optional server-side hint (e.g.
        // "sidecar atom count differs from structure -- re-export
        // from /modify"); the caller can choose to surface it
        // inline.  Exposed as a non-enumerable property so existing
        // callers that just receive ``schema`` aren't broken.
        if (body.notice) {
            Object.defineProperty(body.schema, "_notice", {
                value: body.notice, enumerable: false,
            });
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
     * one call.  See docs/protocols/scientific-validation.md § 5.1.
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
                if (f.kind === "int-triple") {
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

    root.molbuilder = root.molbuilder || {};
    root.molbuilder.formSchema = {
        renderForm:  renderForm,
        collectForm: collectForm,
        fetchSchema: fetchSchema,
        setValues:   setValues,
    };
})(typeof window !== "undefined" ? window : this);
