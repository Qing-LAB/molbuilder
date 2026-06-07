/* Transport-calculation tab core.
 *
 * Fetches the form schema from /api/transport/schema and renders
 * it into #transport-form-container via the shared form-schema
 * helper.  Generate is intentionally disabled — engine backends
 * (TranSIESTA, PySCF-NEGF) land in a follow-up phase; until then
 * the form is "configure now, generate later" UX so users can
 * prototype parameter combinations against the dataclass.
 *
 * Subscribes to projects.onCommit (dblclick = commit) so a
 * sidebar pick updates the visible structure-file context.  The
 * commit doesn't trigger a script render today — Generate stays
 * disabled — but the wire-up follows the universal interaction
 * model so when engines land the path is already correct.
 *
 * Persists collected form values to sessionStorage under
 * ``molbuilder.transport_form`` so refreshes don't wipe a
 * half-typed configuration.
 *
 * Design ref: docs/tabs/architecture.md § 8 (Transport tab —
 * Phase D form skeleton).
 */
(function (root) {
    "use strict";

    var SCHEMA_URL = "/api/transport/schema";
    var FORM_KEY   = "molbuilder.transport_form";

    function _setStatus(msg) {
        var el = document.getElementById("transport-status");
        if (el) el.textContent = msg || "";
    }

    function _$(id) { return document.getElementById(id); }

    /**
     * Fetch the schema + render the form.  On error, surface a
     * developer-readable message via the status line so the page
     * doesn't fail silently.
     */
    function _fetchAndRender(formContainer, formSchema) {
        return root.fetch(SCHEMA_URL)
            .then(function (r) {
                return r.json().then(function (body) {
                    if (!r.ok || !body.ok) {
                        throw new Error(body.error
                            || "schema fetch failed");
                    }
                    return body.schema;
                });
            })
            .then(function (schema) {
                while (formContainer.firstChild) {
                    formContainer.removeChild(formContainer.firstChild);
                }
                formSchema.renderForm(formContainer, schema);
                _restoreFormValues(formContainer, schema, formSchema);
                _wirePersistence(formContainer, schema, formSchema);
                _setStatus("Form loaded ("
                    + schema.sections.reduce(function (n, s) {
                        return n + (s.fields ? s.fields.length : 0);
                    }, 0)
                    + " fields).");
            })
            .catch(function (e) {
                _renderErrorParagraph(
                    formContainer,
                    "Could not load the transport form schema: "
                    + (e && e.message ? e.message : String(e))
                );
                _setStatus("schema error");
            });
    }

    /**
     * Render a single ``<p class="error" role="alert">`` with
     * ``textContent`` so any message (including unsanitised
     * server error strings) renders as literal text instead of
     * HTML.  Pinned by tests/test_xss_audit.py — any
     * ``.innerHTML = "..."`` in a hot path is a XSS sink waiting
     * for a network response with HTML-looking error text.
     */
    function _renderErrorParagraph(container, message) {
        while (container.firstChild) {
            container.removeChild(container.firstChild);
        }
        var p = document.createElement("p");
        p.className = "error";
        p.setAttribute("role", "alert");
        p.textContent = message;
        container.appendChild(p);
    }

    function _restoreFormValues(container, schema, formSchema) {
        var raw;
        try { raw = root.sessionStorage.getItem(FORM_KEY); }
        catch (_) { return; }
        if (!raw) return;
        var saved;
        try { saved = JSON.parse(raw); } catch (_) { return; }
        if (!saved || typeof saved !== "object") return;
        // Walk the form inputs and reapply each saved value.  We
        // don't use ``collectForm`` here because the form is fresh
        // and untouched; setting .value + .checked directly is the
        // cheapest restore path.
        for (var name in saved) {
            if (!Object.prototype.hasOwnProperty.call(saved, name)) continue;
            var els = container.querySelectorAll(
                '[name="' + CSS.escape(name) + '"]');
            for (var i = 0; i < els.length; i++) {
                var el = els[i];
                if (el.type === "checkbox") {
                    el.checked = !!saved[name];
                } else {
                    el.value = saved[name] != null ? saved[name] : "";
                }
            }
        }
    }

    function _wirePersistence(container, schema, formSchema) {
        var debounceHandle = null;
        function persist() {
            try {
                var values = formSchema.collectForm(container, schema);
                root.sessionStorage.setItem(
                    FORM_KEY, JSON.stringify(values));
            } catch (_) {
                // Best-effort — quota / collectForm validation
                // failure shouldn't break the form's interactive
                // state.  The form still functions; only the
                // refresh-survives behavior degrades.
            }
        }
        container.addEventListener("input", function () {
            if (debounceHandle) clearTimeout(debounceHandle);
            debounceHandle = setTimeout(persist, 250);
        });
        container.addEventListener("change", persist);
    }

    /**
     * Subscribe to the universal commit channel so a sidebar
     * dblclick on a structure file updates the visible "current
     * structure" context.  Single-click stays preview only.
     *
     * Today the only effect is a status-line nudge naming the
     * file; once the engine backends land, this is where the
     * Geometry section's structure_xyz_path + molstruct_json_path
     * fields will be auto-populated.
     */
    function _wireCommitChannel() {
        var runtime = root.molbuilder && root.molbuilder.runtime;
        if (!runtime || typeof runtime.whenReady !== "function") return;
        runtime.whenReady("projects").then(function (proj) {
            if (!proj) return;
            var subscribe = (typeof proj.onCommit === "function")
                ? proj.onCommit.bind(proj)
                : proj.onChange.bind(proj);
            subscribe(function (sel) {
                var f = (sel && sel.file) ? String(sel.file) : "";
                if (!f) return;
                var lc = f.toLowerCase();
                if (!lc.endsWith(".xyz") && !lc.endsWith(".pdb")) return;
                var name = f.split("/").pop();
                _setStatus("Picked structure: " + name
                    + " (Generate stays disabled until engines land).");
            });
        });
    }

    function _init() {
        var formContainer = _$("transport-form-container");
        if (!formContainer) return;
        var formSchema = root.molbuilder
                      && root.molbuilder.formSchema;
        if (!formSchema
            || typeof formSchema.renderForm !== "function") {
            _renderErrorParagraph(
                formContainer,
                "form-schema.js did not load — check the script "
                + "order in transport_calculation.html."
            );
            return;
        }
        _fetchAndRender(formContainer, formSchema);
        _wireCommitChannel();
    }

    if (root.document) {
        if (root.document.readyState === "loading") {
            root.document.addEventListener("DOMContentLoaded", _init);
        } else {
            _init();
        }
    }
})(typeof window !== "undefined" ? window : globalThis);
