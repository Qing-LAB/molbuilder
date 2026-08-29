/* Transport-calculation tab core — the COMPOSITE's describe surface
 * (plans/transport-design.md § 4.1, P7b).
 *
 * ONE driver: the junction citation.  Picking the relaxed junction's
 * finished attempt (the shared tree-picker; only run-N directories are
 * choosable) sets everything downstream — the viewer loads the CITED
 * calculation's own labeled structure, the chemistry analysis runs on
 * it, and Describe writes the FINISHED task.json into the selected
 * folder (no hand-over -- user ruling 2026-08-29: nothing is awaiting,
 * so the tab selects and decides; the shared door in
 * lib/task-handover.js still supplies the destination guards and the
 * file-layer write).  There is no sidebar commit channel here: a
 * second way to fill the viewer would be a second source for the
 * composite's one fact (molview.md § 9.3a, one level up).
 *
 * The form (schema-driven from /api/transport/schema) persists to
 * sessionStorage; the cited junction persists in the tab's own
 * workspace note, so a reload restores the whole describe state.
 */
import { mount } from "/static/lib/molview/index.js";

/* WHO THIS TAB'S SAVED WORK BELONGS TO (workspace.md § 4) — the one string used
 * both as the viewer's `owner` and as the tag on any workspace call, so the two
 * cannot drift into naming different slots. */
const WORKSPACE_TAG = "transport";
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
                // Cached for the Send handler's changed-fields
                // diff -- one fetch per page life.
                _cachedSchema = schema;
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
        // "status error" -- the shared severity vocabulary
        // (page-shell.css); a bare "error" class has no rule anywhere.
        p.className = "status error";
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

    /* THE TAB'S TWO FACTS (P7b review, 2026-08-29): the citation, and
     * the cited calculation's source file the viewer shows.  Both are
     * the SERVER's answers (/api/transport/describe_attempt spells the
     * citation and names the source), adopted whole -- the tab derives
     * neither. */
    var _junction = "";          // the citation string, "" until cited
    var _junctionSource = "";    // sidebar-space path of the cited source

    /* The composite's send gate: a citation is the ONE thing the
     * describe cannot go without (transport-design.md 4.1). */
    function _refreshSendButton() {
        var btn = _$("transport-send-btn");
        if (!btn) return;
        btn.disabled = !_junction;
    }

    function _refreshAnalyzeButton() {
        var btn = _$("auto-detect-btn");
        if (!btn) return;
        btn.disabled = !_junctionSource;
    }

    // The mounted MolView handle (null until the first structure is committed
    // or a saved session is restored at init).
    var _mvHandle = null;

    // The tab's own context, kept under its own tag beside the viewer's
    // (workspace.md § 4 -- the modify:panel pattern): which FILE is committed.
    // The viewer persists the structure + labels; the file is a fact about an
    // operation the TAB performed, so the tab remembers it (molview.md § 6.7).
    var PANEL_TAG = WORKSPACE_TAG + ":panel";

    function _panelIdentity(ws) {
        return { workspace_id: ws.workspaceId(PANEL_TAG), state_index: 0 };
    }

    function _writePanelNote() {
        var ws = root.molbuilder && root.molbuilder.workspace;
        if (!ws || typeof ws.persist !== "function") return;
        /* v2 (P7b review): the tab's fact is the CITATION -- the viewer
         * persists the structure itself; the note is what re-drives the
         * readout, the meta line and the send gate on a reload. */
        ws.persist(PANEL_TAG, { v: 2, junction: _junction || "",
                                source: _junctionSource || "" },
                   _panelIdentity(ws));
    }

    /**
     * Mount the viewer if it is not already up; resolves to the handle or
     * null.  Split out of _showInMolview (2026-08-19) so the init-restore
     * can mount WITHOUT a file -- a reload has no commit to ride on, and a
     * restore that waits for one can never run.
     */
    function _ensureViewer() {
        var ws   = root.molbuilder && root.molbuilder.workspace;
        var host = _$("transport-molview-host");
        if (!ws || !host || typeof mount !== "function") {
            return Promise.resolve(null);
        }
        if (_mvHandle && _mvHandle.ok) return Promise.resolve(_mvHandle);
        /* READ-ONLY (molview.md § 9.4): this viewer shows the CITED
         * calculation's structure, and labels are assigned where the
         * junction is built -- never here (the card's own prose says so).
         * The first install into the empty viewer runs in any mode, and a
         * new citation swaps the structure through the load door's own
         * `enforce` (projects/parser.js) -- a deliberate swap is the
         * host's business, not an edit (§ 11.2a).
         *
         * Both mount paths run at projects-ready moments -- the pick
         * handler by construction, the init-restore because it awaits
         * whenReady("projects") -- so the files door is real when read. */
        var _proj = root.molbuilder && root.molbuilder.projects;
        return mount(host, ws, { mode: "readonly",
                                 owner: WORKSPACE_TAG,
                                 files: _proj && _proj.molviewFiles })
            .then(function (h) {
                _mvHandle = (h && h.ok) ? h : null;
                /* THE TAB'S DEFAULT REPRESENTATION IS BALL-AND-STICK
                 * (user, 2026-08-29).  Stick draws BONDS AND NOTHING
                 * ELSE, so a junction whose atoms the library does not
                 * perceive as bonded -- or one viewed end-on down its
                 * transport axis, which is EVERY junction here --
                 * renders an empty-looking window (the demo fixture's
                 * own recorded failure, lib/molview/demo.js).  Spheres
                 * draw regardless.  Set at mount, BEFORE the view
                 * context restores, so a preference the user actually
                 * chose still wins (ui-context applies saved.view after
                 * the first structure). */
                if (_mvHandle && _mvHandle.data && _mvHandle.data.view
                        && typeof _mvHandle.data.view.set === "function") {
                    _mvHandle.data.view.set("style", "ball-and-stick");
                }
                return _mvHandle;
            });
    }

    /**
     * Coming back to a session: the tab's own note carries the citation,
     * and the structure is RE-OPENED from the cited file -- the pattern
     * molview.md § 12.3 gives a display tab ("a read-only tab keeps its
     * structure by RELOADING it; the tab owns that, not the viewer").
     * There is no draft branch: on a read-only viewer `load(0)` is a
     * documented no-op (§ 11.2a), and the Results inspector's own 2026-08-03
     * bug record shows what a draft-restore on the wrong mode looks like --
     * "Loaded." over an empty viewer, no request, no error.  The note is
     * read FIRST and unconditionally, so the citation, the meta line and
     * the send gate come back even if the structure file has moved.
     */
    function _restoreSession() {
        var ws = root.molbuilder && root.molbuilder.workspace;
        var runtime = root.molbuilder && root.molbuilder.runtime;
        if (!ws || !runtime
                || typeof runtime.whenReady !== "function") return;
        // Projects first: the viewer's files door rides the namespace, and a
        // restore that mounts before the sidebar module ran would capture an
        // undefined door for the life of the viewer.
        runtime.whenReady("projects").then(function () {
            return Promise.resolve(ws.readState(_panelIdentity(ws)));
        }).then(function (note) {
            if (!note || note.v !== 2 || !note.junction) return;
            _junction = note.junction;
            _junctionSource = note.source || "";
            var out = _$("transport-junction-readout");
            if (out) out.textContent = _junction;
            _refreshSendButton();
            _refreshAnalyzeButton();
            _refreshJunctionMeta();
            if (_junctionSource) _showInMolview(_junctionSource);
            _setStatus("Restored your last session: " + _junction + ".");
        }).catch(function (e) {
            if (root.console) {
                root.console.error("[transport] session restore failed", e);
            }
        });
    }

    /**
     * Show the CITED structure so the user can eyeball what they cited —
     * atoms, region labels, the cell — before describing.  Read-only view of
     * the citation (molview.md § 12.3); every call is a fresh open of the
     * cited file through the one load door (``projects.parser.openMolecule``
     * reads the .xyz + .molstruct.json server-side and installs the model in
     * ONE write; its `enforce` makes a new citation a deliberate swap).
     * Best-effort: if the molview/projects stack failed to load, the tab
     * still describes (the viewer is an aid, not a gate).
     */
    function _showInMolview(path) {
        var proj = root.molbuilder && root.molbuilder.projects;
        if (!proj || !proj.parser
                || typeof proj.parser.openMolecule !== "function") {
            return;
        }
        /* THE VIEWER FIRST, THEN THE FILE. A viewer mounts before it has a
         * structure (molview.md § 8), and the load door needs somewhere to put
         * what it reads. This ran the other way round, which only worked while
         * the load door could find a viewer by name in a global. */
        _ensureViewer().then(function (viewer) {
            if (!viewer) return;
            return proj.parser.openMolecule(viewer, path).then(function (r) {
                if (r && r.ok === false && root.console) {
                    root.console.error("[transport] load failed", r.error);
                }
            });
        }).catch(function (e) {
            if (root.console) {
                root.console.error("[transport] MolView load/mount failed", e);
            }
        });
    }

    // ---------- Auto-detect chemistry handler (Card 2) ---------- //

    var _autoDetectSeq = 0;
    // J3 2026-06-14: shared AbortController so a spam-click or a
    // commit-on-file-change racing a prior request kills the older
    // server-side parse instead of letting both complete.  Mirrors
    // viewer.js's _autoDetectAbort pattern.
    var _autoDetectAbort = null;

    function _analyzeStructure(path) {
        if (!path) return;
        var mySeq = ++_autoDetectSeq;
        if (_autoDetectAbort) _autoDetectAbort.abort();
        _autoDetectAbort = new (root.AbortController)();
        var mySignal = _autoDetectAbort.signal;
        var btn = _$("auto-detect-btn");
        if (btn) btn.disabled = true;
        _autoDetectSetStatus("Analyzing chemistry…", null);
        root.fetch("/api/structure/analyze", {
            method:  "POST",
            headers: { "Content-Type": "application/json" },
            body:    JSON.stringify({ structure_path: path }),
            signal:  mySignal,
        })
            .then(function (r) {
                return r.json().then(function (body) {
                    return { status: r.status, body: body };
                });
            })
            .then(function (resp) {
                if (mySeq !== _autoDetectSeq) return;
                var b = resp.body;
                if (!b || !b.ok) {
                    _autoDetectSetStatus(
                        b && b.error ? b.error
                            : "Analyze failed (HTTP " + resp.status + ").",
                        "error");
                    _refreshAnalyzeButton();
                    return;
                }
                _renderAutoDetectPanel(b);
                // Inject the same workflow-group detection chip the
                // SIESTA + PySCF tabs show — single source of truth
                // (lib/detection-chip.js) per web-ui-coherence.md
                // Rule 1.  Au junctions are the canonical use case;
                // pre-2026-06-13 Transport silently rendered no chip.
                var chipApi = (root.molbuilder
                               && root.molbuilder.detectionChip);
                if (chipApi && chipApi.render) chipApi.render(b);
                _autoDetectSetStatus(
                    "Chemistry analyzed — review the rationale "
                    + "panel before sending.",
                    null);
                _refreshAnalyzeButton();
            })
            .catch(function (e) {
                // AbortError = J3 supersede.  Silent; the new
                // request owns the UI.
                if (e && e.name === "AbortError") return;
                if (mySeq !== _autoDetectSeq) return;
                _autoDetectSetStatus(
                    "Network error: "
                    + (e && e.message ? e.message : String(e)),
                    "error");
                _refreshAnalyzeButton();
            });
    }

    function _autoDetectSetStatus(msg, kind) {
        var el = _$("auto-detect-status");
        if (!el) return;
        el.textContent = msg || "";
        el.className = "status"
            + (kind === "error" ? " error" : "")
            + (kind === "ok"    ? " ok"    : "");
    }

    function _renderAutoDetectPanel(resp) {
        var panel = _$("auto-detect-panel");
        if (!panel) return;
        panel.hidden = false;
        panel.open = true;
        var ratEl  = _$("auto-detect-rationale");
        var warnEl = _$("auto-detect-warnings");
        var metEl  = _$("auto-detect-metals");
        if (ratEl) {
            var sug = (resp.suggested || {}).pyscf
                || (resp.suggested || {}).siesta
                || {};
            ratEl.textContent = sug.rationale || "";
        }
        if (warnEl) {
            warnEl.textContent = "";
            var ws = resp.warnings || [];
            for (var i = 0; i < ws.length; i++) {
                var li = document.createElement("li");
                li.textContent = ws[i];
                warnEl.appendChild(li);
            }
            warnEl.hidden = ws.length === 0;
        }
        if (metEl) {
            metEl.textContent = "";
            var hs = resp.metal_hints || [];
            for (var j = 0; j < hs.length; j++) {
                var h = hs[j];
                var dt = document.createElement("dt");
                dt.textContent = h.element;
                metEl.appendChild(dt);
                var cs = h.common_spins || [];
                for (var k = 0; k < cs.length; k++) {
                    var c = cs[k];
                    var dd = document.createElement("dd");
                    dd.textContent =
                        "spin=" + c.spin + " — " + c.label;
                    metEl.appendChild(dd);
                }
            }
            metEl.hidden = hs.length === 0;
        }
    }

    function _wireAutoDetectButton() {
        var btn = _$("auto-detect-btn");
        if (!btn) return;
        btn.addEventListener("click", function () {
            if (!_junctionSource) return;
            _analyzeStructure(_junctionSource);
        });
    }

    /* =================================================================
     *  The COMPOSITE (transport-design.md § 4.1): cite the junction,
     *  state the bias, Describe -- the tab writes the finished task.json
     *  itself (no hand-over; user ruling 2026-08-29).  `_junction` and
     *  `_junctionSource` are declared once, with the tab's two facts
     *  above.
     * ================================================================= */

    /** One fetch answers everything about an attempt: the summary for
     *  the meta line, the CITATION (the server spells it -- the tree
     *  grammar has one home), and the cited calculation's source file.
     *  ``rel`` is tree-relative (proj.relativeToProjects). */
    function _describeAttempt(rel) {
        return root.fetch("/api/transport/describe_attempt?path="
                          + encodeURIComponent(rel.replace(/^\/+/, "")))
            .then(function (r) { return r.json(); })
            .then(function (b) {
                if (!b || b.ok === false) {
                    throw new Error((b && b.error) || "unreadable");
                }
                return b;
            });
    }

    /** Adopt a described attempt as THE citation: readout, meta, send
     *  gate, the workspace note -- and the viewer + chemistry analysis
     *  follow it, because the citation is the tab's one driver
     *  (user, 2026-08-29: the viewer responds to the active
     *  calculation). */
    function _adoptCitation(described, pickedPath, rel) {
        _junction = described.citation || "";
        if (!_junction) {
            // The pick failed, so the complaint belongs in card 1 with
            // the picker -- not three cards down beside Describe.
            _setStatus("That folder is not an attempt inside a "
                + "calculation (no task.json above it).");
            return;
        }
        var out = _$("transport-junction-readout");
        if (out) out.textContent = _junction;
        var meta = _$("transport-junction-meta");
        if (meta) {
            meta.hidden = false;
            meta.textContent = described.summary || "";
        }
        /* The cited source, in the sidebar's own path space: the picker
         * hands back pickedPath whose tail is `rel`, so the root prefix
         * is pickedPath minus rel -- no second spelling of the root. */
        _junctionSource = "";
        if (described.source && pickedPath && rel
                && pickedPath.length > rel.length) {
            var prefix = pickedPath.slice(0, pickedPath.length - rel.length);
            _junctionSource = prefix + described.source;
        }
        _writePanelNote();
        _refreshSendButton();
        _refreshAnalyzeButton();
        /* The server said whether the attempt CONCLUDED (strict
         * composition, transport-design.md ruling Q2, is a PREP gate --
         * describing ahead of conclusion is legal, so the tab warns
         * loudly rather than blocking). */
        var late = described.concluded
            ? ""
            : "  NOT CONCLUDED yet: you can describe now, but prep "
              + "will refuse this citation until the relaxation "
              + "finishes.";
        if (_junctionSource) {
            _showInMolview(_junctionSource);
            _analyzeStructure(_junctionSource);
            _setStatus("Cited " + _junction.split("/").pop()
                + " — the viewer shows the cited junction." + late);
        } else {
            _setStatus("Cited " + _junction + " — its source structure "
                + "was not found beside the calculation, so the viewer "
                + "keeps its last content." + late);
        }
    }

    /** Re-fetch the meta line for a RESTORED citation (the note keeps
     *  the strings; the summary is re-read so a junction that has
     *  changed state since says so). */
    function _refreshJunctionMeta() {
        if (!_junction) return;
        var rel = _junction.replace("@", "/");
        var meta = _$("transport-junction-meta");
        _describeAttempt(rel).then(function (b) {
            if (meta) { meta.hidden = false;
                        meta.textContent = b.summary || ""; }
        }).catch(function (e) {
            if (meta) { meta.hidden = false;
                        meta.textContent = "Could not re-read the cited "
                            + "attempt: "
                            + (e && e.message ? e.message : String(e)); }
        });
    }

    function _wireJunctionPicker() {
        var btn = _$("transport-junction-btn");
        if (!btn) return;
        btn.addEventListener("click", function () {
            var proj = root.molbuilder && root.molbuilder.projects;
            function toRel(path) {
                return (proj && proj.relativeToProjects)
                    ? String(proj.relativeToProjects(path) || "") : path;
            }
            /* THE one pop-out picker (lib/tree-picker.js): only run-N
             * attempt directories can be the answer, and the meta line
             * is the attempt's own .fdf -- the deck that actually ran
             * is the truth about a result (user, 2026-08-28). */
            import("../tree-picker.js").then(function (mod) {
                return mod.pickPath({
                    title: "Cite the relaxed junction",
                    hint: "Walk to the junction relaxation's finished "
                        + "attempt (a run-N folder).  \u25b8 expands.",
                    mode: "dir",
                    pickable: function (entry) {
                        return /^run-\d+$/.test(entry.name || "");
                    },
                    describe: function (path) {
                        return _describeAttempt(toRel(path))
                            .then(function (b) { return b.summary || ""; });
                    },
                    confirmLabel: "Cite this attempt",
                });
            }).then(function (picked) {
                if (!picked) return;
                var rel = toRel(picked);
                return _describeAttempt(rel).then(function (b) {
                    _adoptCitation(b, picked, rel);
                });
            }).catch(function (e) {
                _setStatus("Picker failed: "
                    + (e && e.message ? e.message : String(e)));
            });
        });
    }

    function _setSendStatus(msg) {
        var el = _$("transport-send-status");
        if (el) el.textContent = msg || "";
    }

    /** The transport-only knobs: fields whose value differs from the
     *  schema default.  The server refuses a sealed one BY NAME (the
     *  electronic contract is the citation's to say), so an untouched
     *  form sends nothing and a touched contract field gets a clear
     *  answer instead of a silent drop. */
    function _changedFields(formContainer) {
        var fs = root.molbuilder && root.molbuilder.formSchema;
        var schema = _cachedSchema;
        if (!fs || !schema || typeof fs.collectForm !== "function") {
            return {};
        }
        var values;
        try { values = fs.collectForm(formContainer, schema); }
        catch (e) { return null; }        // invalid form: the caller says so
        var defaults = {};
        (schema.sections || []).forEach(function (s) {
            (s.fields || []).forEach(function (f) {
                defaults[f.name] = f.default;
            });
        });
        var out = {};
        Object.keys(values || {}).forEach(function (k) {
            var v = values[k];
            var d = defaults[k];
            if (JSON.stringify(v) !== JSON.stringify(d)) out[k] = v;
        });
        return out;
    }

    function _wireSendButton(formContainer) {
        var btn = _$("transport-send-btn");
        if (!btn) return;
        btn.addEventListener("click", function () {
            var mb = root.molbuilder || {};
            if (!mb.taskHandover) {
                _setSendStatus("lib/task-handover.js is not loaded.");
                return;
            }
            var bias = String((_$("transport-bias") || {}).value || "0.0")
                .split(",").map(function (s) { return s.trim(); })
                .filter(Boolean).map(Number);
            if (bias.some(isNaN)) {
                _setSendStatus("Bias must be comma-separated volts, "
                    + "e.g. 0.0,0.2");
                return;
            }
            var overrides = _changedFields(formContainer);
            if (overrides === null) {
                _setSendStatus("Form has invalid values — fix them "
                    + "and retry.");
                return;
            }
            mb.taskHandover.send({
                projects: mb.projects,
                say: function (kind, msg) {
                    // Severity reaches the eye: the shared setter
                    // applies .status.error/.ok/.warn (page-shell).
                    var st = root.molbuilder && root.molbuilder.status;
                    if (st && typeof st.set === "function") {
                        st.set("transport-send-status", msg, kind);
                    } else { _setSendStatus(msg); }
                },
                engine: "siesta",
                calculation: "transport",
                junction: _junction,
                bias: bias,
                overrides: overrides,
            });
        });
    }

    // Cache the schema so the Generate handler doesn't have to
    // re-fetch on every click; populated by _fetchAndRender on
    // first load.
    var _cachedSchema = null;

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
        _restoreSession();
        _wireJunctionPicker();
        _wireSendButton(formContainer);
        _wireAutoDetectButton();
        _refreshSendButton();
        _refreshAnalyzeButton();
    }

    if (root.document) {
        if (root.document.readyState === "loading") {
            root.document.addEventListener("DOMContentLoaded", _init);
        } else {
            _init();
        }
    }
})(typeof window !== "undefined" ? window : globalThis);
