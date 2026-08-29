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
        /* The contract lane rides the query (4.1b): "cited" hides the
         * electronic-contract fields (the deck's to say); "open"
         * offers them (a labeled pair has no deck). */
        var url = SCHEMA_URL
            + (_junctionContract === "open" ? "?contract=open" : "");
        return root.fetch(url)
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

    /* THE TAB'S FACTS (4.1b, 2026-08-29): the citation (a directory
     * whose FILES satisfy the condition), the composed labeled
     * structure the server answers with (the viewer + the chemistry
     * analysis run on it -- no file path is assumed), and which
     * contract lane the form serves ("cited" = the deck's, contract
     * fields hidden; "open" = the description's own, offered).  All
     * three are /api/transport/describe_attempt's answers, adopted
     * whole -- the tab derives none of them. */
    var _junction = "";           // the citation path, "" until cited
    var _junctionStructure = null;    // the composed structure envelope
    var _junctionContract = "cited";  // which schema lane the form shows

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
        btn.disabled = !_junctionStructure;
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
        /* v3 (4.1b): the tab's ONE fact is the CITATION.  A reload
         * re-describes it through the same seam a pick uses, so the
         * structure, the meta line and the contract lane always come
         * back fresh from the server, never from a stale copy. */
        ws.persist(PANEL_TAG, { v: 3, junction: _junction || "" },
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
            if (!note || note.v !== 3 || !note.junction) return;
            // The SAME flow a pick takes: re-describe, then adopt --
            // one adoption path, so a restored citation and a fresh
            // one cannot drift apart.
            return _describeAttempt(note.junction).then(function (b) {
                _adoptCitation(b, "Restored your last session: ");
            });
        }).catch(function (e) {
            if (root.console) {
                root.console.error("[transport] session restore failed", e);
            }
        });
    }

    /**
     * Show the CITED structure so the user can eyeball what they cited —
     * atoms, region labels, the cell — before describing.  Read-only view
     * of the citation (molview.md § 12.3), installed from the SERVER'S
     * OWN composition (describe_attempt answers the structure envelope,
     * the same shape /api/build/load's {structure} branch takes) — a
     * form-A citation has no .xyz on disk, so no file path is assumed
     * (4.1b).  `enforce` makes a new citation a deliberate swap
     * (molview.md § 11.2a).  Best-effort: if the molview stack failed to
     * load, the tab still describes (the viewer is an aid, not a gate).
     */
    function _showStructure(wire) {
        if (!wire) return;
        _ensureViewer().then(function (viewer) {
            if (!viewer || !viewer.data
                    || typeof viewer.data.installMolecule !== "function") {
                return;
            }
            return viewer.data.installMolecule({
                structure: wire,
                source: { kind: "citation", file: _junction || null,
                          generator_input: null },
                enforce: true,
            });
        }).catch(function (e) {
            if (root.console) {
                root.console.error("[transport] MolView load/mount failed", e);
            }
        });
    }

    // ---------- Auto-detect chemistry (Card 2) ---------- //
    //
    // The SHARED surface (lib/auto-detect.js): analyze() owns the
    // supersede protocol, renderPanel() owns the card + the detection
    // chips.  This tab carried its own copy of all three until
    // 2026-08-29 -- the recorded hold-out, retired with the redesign
    // round it was deferred to.

    function _analyzeStructure(path) {
        var ad = root.molbuilder && root.molbuilder.autoDetect;
        var st = root.molbuilder && root.molbuilder.status;
        if (!ad || typeof ad.analyze !== "function") return;
        var btn = _$("auto-detect-btn");
        if (btn) btn.disabled = true;
        if (st) st.set("auto-detect-status", "Analyzing chemistry…");
        ad.analyze(path).then(function (res) {
            if (res && res.superseded) return;
            if (!res || !res.ok) {
                if (st) st.set("auto-detect-status",
                    (res && res.error) || "Analyze failed.", "error");
                _refreshAnalyzeButton();
                return;
            }
            ad.renderPanel(res.body);
            if (st) st.set("auto-detect-status",
                "Chemistry analyzed — review the rationale panel "
                + "before sending.");
            _refreshAnalyzeButton();
        });
    }

    function _wireAutoDetectButton() {
        var btn = _$("auto-detect-btn");
        if (!btn) return;
        btn.addEventListener("click", function () {
            if (!_junctionStructure) return;
            _analyzeStructure({ structure: _junctionStructure });
        });
    }

    /* =================================================================
     *  The COMPOSITE (transport-design.md § 4.1): cite the junction,
     *  state the bias, Describe -- the tab writes the finished task.json
     *  itself (no hand-over; user ruling 2026-08-29).  The tab's facts
     *  are declared once, above.
     * ================================================================= */

    /** One fetch answers everything about a picked directory: the
     *  4.1b classification (form, contract lane), the meta line, the
     *  composed structure for the viewer, and the citation spelling
     *  (the server's).  ``rel`` is tree-relative
     *  (proj.relativeToProjects). */
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
    function _adoptCitation(described, statusPrefix) {
        if (!described || !described.form || !described.citation) {
            // Not citable: the server's summary IS the condition,
            // naming the missing file.  Card 1, with the picker.
            _setStatus((described && described.summary)
                || "That directory is not citable.");
            return;
        }
        _junction = described.citation;
        _junctionStructure = described.structure || null;
        var out = _$("transport-junction-readout");
        if (out) out.textContent = _junction;
        var meta = _$("transport-junction-meta");
        if (meta) {
            meta.hidden = false;
            meta.textContent = described.summary || "";
        }
        _writePanelNote();
        _refreshSendButton();
        _refreshAnalyzeButton();
        /* The contract lane follows the FORM (4.1b): a relaxation's
         * deck owns the electronic contract (fields hidden); a plain
         * labeled pair has no deck, so the fields are the
         * description's own and the form offers them. */
        var lane = described.contract === "open" ? "open" : "cited";
        if (lane !== _junctionContract) {
            _junctionContract = lane;
            var fc = _$("transport-form-container");
            var fs = root.molbuilder && root.molbuilder.formSchema;
            if (fc && fs) _fetchAndRender(fc, fs);
        }
        /* Honest state, not a gate (strict composition refuses at
         * PREP; describing ahead is legal).  The summary already says
         * CONCLUDED / not / no-record; add the road note only when
         * prep would refuse today. */
        var late = (described.form === "relaxation"
                    && described.concluded === false
                    && /NOT CONCLUDED/.test(described.summary || ""))
            ? "  You can describe now, but prep will refuse this "
              + "citation until the relaxation finishes."
            : "";
        if (_junctionStructure) {
            _showStructure(_junctionStructure);
            _analyzeStructure({ structure: _junctionStructure });
            _setStatus((statusPrefix || "Cited ")
                + _junction + " — the viewer shows the cited junction."
                + late);
        } else {
            _setStatus((statusPrefix || "Cited ") + _junction
                + " — it classifies but does not compose (the meta "
                + "line says why), so the viewer keeps its last "
                + "content." + late);
        }
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
            /* THE one pop-out picker (lib/tree-picker.js): ANY
             * directory can be chosen -- what makes it citable is its
             * FILES (4.1b, user ruling 2026-08-29: a finished
             * relaxation's .fdf+.XV together, or a labeled
             * .xyz+.molstruct.json pair), and the describe seam
             * classifies each selection so the meta line answers
             * before you confirm. */
            import("../tree-picker.js").then(function (mod) {
                return mod.pickPath({
                    title: "Cite the relaxed junction",
                    hint: "Pick the DIRECTORY holding the relaxed "
                        + "junction: a finished relaxation (.fdf + .XV "
                        + "together) or a labeled structure (.xyz + "
                        + ".molstruct.json).  \u25b8 expands.",
                    mode: "dir",
                    describe: function (path) {
                        return _describeAttempt(toRel(path))
                            .then(function (b) { return b.summary || ""; });
                    },
                    confirmLabel: "Cite this directory",
                });
            }).then(function (picked) {
                if (!picked) return;
                var rel = toRel(picked);
                return _describeAttempt(rel).then(function (b) {
                    _adoptCitation(b);
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
    /** The transport-only knobs whose value differs from the schema
     *  default -- through the SHARED differ (formSchema.diffFromDefaults,
     *  which owns the typed comparison incl. the 300-vs-"300" trap).
     *  An untouched form sends nothing; an invalid one answers null so
     *  the caller says so instead of silently dropping fields. */
    function _changedFields(formContainer) {
        var fs = root.molbuilder && root.molbuilder.formSchema;
        if (!fs || !_cachedSchema
                || typeof fs.diffFromDefaults !== "function") {
            return {};
        }
        try {
            var out = {};
            fs.diffFromDefaults(formContainer, _cachedSchema)
                .forEach(function (d) { out[d.name] = d.current; });
            return out;
        } catch (e) { return null; }      // invalid form: the caller says so
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
