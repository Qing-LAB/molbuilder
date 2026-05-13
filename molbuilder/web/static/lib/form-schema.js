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

    function renderField(f) {
        // Build a single <label> wrapping the input.  Checkbox lays
        // out as "[x] Label" -- the checkbox comes BEFORE the label
        // text; everything else lays out as "Label: <input>".
        const labelEl = el("label", {
            class: "schema-field schema-field-" + f.kind,
            title: f.help || "",
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
        return labelEl;
    }

    /* ---------- public API ---------- */

    function renderForm(container, schema) {
        if (!container || !schema || !Array.isArray(schema.sections)) {
            throw new Error("form-schema.renderForm: bad container/schema");
        }
        // Fresh render -> schema and DOM are presumed to match, so
        // clear the stale-warning cache.  Any actual mismatch on
        // the next collectForm will re-warn.
        _staleWarnings.clear();
        container.innerHTML = "";
        for (const sect of schema.sections) {
            const fs = el("fieldset", { class: "schema-section" });
            fs.appendChild(el("legend", null, sect.name));
            // Optional per-section description.  Appears as a small
            // paragraph directly below the legend so the user reads
            // "what is this group of knobs for?" before digging into
            // the individual fields.  Schemas without a description
            // for a section simply skip this paragraph.
            if (sect.description) {
                fs.appendChild(el(
                    "p",
                    { class: "schema-section-desc" },
                    sect.description
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

    async function fetchSchema(engine) {
        const r = await fetch("/api/build/schema/" + encodeURIComponent(engine));
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

    root.molbuilder = root.molbuilder || {};
    root.molbuilder.formSchema = {
        renderForm:  renderForm,
        collectForm: collectForm,
        fetchSchema: fetchSchema,
    };
})(typeof window !== "undefined" ? window : this);
