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
 * Design ref: docs/web/tabs.md (Transport tab —
 * Phase D form skeleton).
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
                // Phase B.3 step 2: cache the schema for the
                // Generate handler so it doesn't have to re-fetch
                // on every click.
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
            // Mark the form dirty so the sidebar-commit handler can
            // surface the unsaved-modifications modal before
            // overwriting parameter edits.  Cleared after Generate.
            _formDirty = true;
            if (debounceHandle) clearTimeout(debounceHandle);
            debounceHandle = setTimeout(persist, 250);
        });
        container.addEventListener("change", function () {
            _formDirty = true;
            persist();
        });
    }

    // 2026-06-09 audit fix: form-dirty tracking parallel to Spectra's
    // pattern.  Without this, a sidebar dblclick on a different
    // structure would silently discard the user's parameter edits;
    // the BOMB-3 comment in ``_wireCommitChannel`` explicitly named
    // this gap.  Cleared on successful Generate (the form values
    // landed in the .fdf — discarding the structure swap is OK).
    var _formDirty = false;

    // Current structure file (committed via sidebar dblclick).
    // Drives the Generate button's enable state and the
    // ``structure_path`` field on the /api/transport/render POST.
    var _currentStructureFile = "";
    // Viewer-is-truth (2026-07): the committed structure's labels (frozen atoms +
    // electrode regions) live in the mounted MolView model (loaded from the sidecar
    // by projects.parser.openMolecule); the Generate POST sources them from molview.data at send
    // time, so there is no separate label cache here.  The labels ride INSIDE the
    // structure envelope `exportFile()` produces (web-api.md § 1); the server
    // rebuilds the Structure from that one object (`_shared.struct_from_body`)
    // and the model validates the indices itself (`Structure._validate_regions`).

    /* The composite's send gate: a citation is the ONE thing the
     * describe cannot go without (transport-design.md 4.1) -- the
     * structure viewer above is an inspection aid, not an input. */
    function _refreshSendButton() {
        var btn = _$("transport-send-btn");
        if (!btn) return;
        btn.disabled = !_junction;
    }

    function _refreshAutoDetectButton() {
        var btn = _$("auto-detect-btn");
        if (!btn) return;
        btn.disabled = !_currentStructureFile;
    }

    function _setCurrentStructureReadout(path) {
        var el = _$("transport-current-structure");
        if (!el) return;
        if (!path) {
            el.textContent = "No structure committed yet.";
            return;
        }
        var name = path.split("/").pop();
        el.textContent = "Committed: " + name;
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

    function _writePanelNote(file) {
        var ws = root.molbuilder && root.molbuilder.workspace;
        if (!ws || typeof ws.persist !== "function") return;
        ws.persist(PANEL_TAG, { v: 1, structureFile: file || "" },
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
        /* EDITABLE, and said the way MolView says it: editable is the ABSENCE
         * of the mode flag.  Designating an electrode is a label write, which
         * is exactly what this tab exists for.
         *
         * The files door comes off the namespace (this file is a classic
         * script and cannot import it); both mount paths run at
         * projects-ready moments -- the commit channel by construction, the
         * init-restore because it awaits whenReady("projects"). */
        var _proj = root.molbuilder && root.molbuilder.projects;
        return mount(host, ws, { owner: WORKSPACE_TAG,
                                 files: _proj && _proj.molviewFiles })
            .then(function (h) {
                _mvHandle = (h && h.ok) ? h : null;
                return _mvHandle;
            });
    }

    /**
     * Coming back to a session (molview.md § 11.2a): mount and `load(0)`.
     * A non-null answer means the draft was adopted -- the structure, the
     * user's electrode labels, the position on the sequence -- so the tab
     * shows it exactly as a commit would have.  Until 2026-08-19 this tab
     * wrote its draft on every label edit and NEVER read it back: the
     * labels were saved and lost anyway (write-only persistence).
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
            return _ensureViewer();
        }).then(function (viewer) {
            if (!viewer || !viewer.data
                    || typeof viewer.data.load !== "function") return null;
            return viewer.data.load(0);
        }).then(function (at) {
            if (at === null || at === undefined) return;
            // The viewer is back; now the tab's own note, under its own tag.
            return Promise.resolve(ws.readState(_panelIdentity(ws)))
                .then(function (note) {
                    var f = note && note.v === 1 ? (note.structureFile || "") : "";
                    if (!f) return;
                    _currentStructureFile = f;
                    _setCurrentStructureReadout(f);
                    _refreshAutoDetectButton();
                    _setStatus("Restored your last session: "
                        + f.split("/").pop() + ".");
                });
        }).catch(function (e) {
            if (root.console) {
                root.console.error("[transport] session restore failed", e);
            }
        });
    }

    /**
     * Display the committed structure in the concealed MolView component so the
     * user can CHECK it before generating — atom/region labels, electrode regions,
     * the unit cell, and alignment via the view toggles.  This is the SAME module
     * Modify mounts, in full "modify" mode (interactive selection + cell panel +
     * view menu); the geometry op-tabs are Modify-only and not part of the mount.
     *
     * Load the project file through the format-aware sidebar door
     * (``projects.parser.openMolecule`` — reads the .xyz+.molstruct.json via the
     * projects file package + installs the model in ONE write) then mount the fused
     * card into the empty host on FIRST commit; later commits just reload the model
     * and the mounted render reacts.  Best-effort: if the molview/projects stack
     * failed to load, the tab still works as a form generator (the viewer is an aid).
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

    /**
     * Subscribe to the universal commit channel so a sidebar
     * dblclick on a structure file updates the visible "current
     * structure" context.  Single-click stays preview only.
     *
     * Phase B.3 step 2 (2026-06-10): committing a structure also
     * enables the Generate button so the user can render the
     * transiesta device .fdf.
     */
    function _wireCommitChannel() {
        var runtime = root.molbuilder && root.molbuilder.runtime;
        if (!runtime || typeof runtime.whenReady !== "function") return;
        runtime.whenReady("projects").then(function (proj) {
            if (!proj) return;
            // BOMB-3 fix (2026-06-07): require ``onCommit``; do NOT
            // fall back to ``onChange``.  ``onChange`` fires on EVERY
            // sidebar click + fires-on-subscribe — using it as a
            // fallback would clobber the status line on every preview
            // click and re-build the geometry on every browse-click.
            //
            // 2026-06-09 audit fix: Transport now has form-dirty
            // tracking (``_formDirty`` above) + the warning-modal
            // gate (parallel to Spectra's _commitStructureForSpectra)
            // so a sidebar swap no longer silently discards parameter
            // edits.  The original BOMB-3 comment named this gap.
            if (typeof proj.onCommit !== "function") return;
            async function _commit(sel) {
                var f = (sel && sel.file) ? String(sel.file) : "";
                if (!f) return;
                var lc = f.toLowerCase();
                if (!lc.endsWith(".xyz") && !lc.endsWith(".pdb")) return;
                // Form-dirty gate: ask the user before discarding
                // unsaved parameter edits.  No-op when the form
                // hasn't been touched (fresh mount, just after
                // Generate, ...).
                var modal = root.molbuilder
                         && root.molbuilder.warningModal;
                if (_formDirty && modal
                        && typeof modal.confirmDiscardUnsaved === "function") {
                    try {
                        var proceed = await modal.confirmDiscardUnsaved();
                        if (!proceed) return;
                    } catch (_) { /* modal unavailable — proceed */ }
                }
                _formDirty = false;
                _currentStructureFile = f;
                _writePanelNote(f);
                // Show the structure in the MolView inspection card (viewer +
                // selection/cell panel + view toggles).  Fire-and-forget: runs in
                // parallel with the auto-analyze below.  MolView reads the sidecar
                // labels (frozen atoms + electrode regions) as part of the load, so
                // the Generate POST sources them from molview.data — no separate
                // sidecarLabels.fetch, no path-pointer indirection.
                _showInMolview(f);
                _setCurrentStructureReadout(f);
                _refreshAutoDetectButton();
                var name = f.split("/").pop();
                _setStatus("Structure: " + name + " — inspect it here; "
                    + "the composite cites a finished attempt below.");
                // Phase 3-style auto-analyze: surface chemistry
                // conclusions on commit so users see the open-
                // shell-metal warn BEFORE clicking Generate.
                // Forms are NOT mutated (TransportConfig has no
                // charge/spin/method fields); analysis is
                // informational only.
                _autoAnalyzeOnCommit(f);
            }
            proj.onCommit(_commit);
            // 2026-06-09 audit fix: mount-time getCurrentFile
            // pickup.  Parallel to Spectra's pattern — if the user
            // navigated here from another tab where a structure
            // was already picked, surface it as a committed
            // structure on this tab too (otherwise the readout sits
            // at "No structure committed yet." even though the
            // sidebar shows the file as the current pick).
            if (typeof proj.getCurrentFile === "function") {
                var initFile = proj.getCurrentFile();
                if (initFile) {
                    var initDir = (typeof proj.getCurrentDir === "function")
                        ? proj.getCurrentDir() : "";
                    _commit({ dir: initDir, file: initFile });
                }
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

    function _autoAnalyzeOnCommit(path) {
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
                    _refreshAutoDetectButton();
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
                    + "panel before generating.",
                    null);
                _refreshAutoDetectButton();
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
                _refreshAutoDetectButton();
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
            if (!_currentStructureFile) return;
            _autoAnalyzeOnCommit(_currentStructureFile);
        });
    }

    // ---------- Generate handler (Phase B.3 step 2) ---------- //

    /* =================================================================
     *  The COMPOSITE (P7b, transport-design.md § 4.1): cite the
     *  junction, state the bias, hand over -- the tab describes and
     *  Task setup saves; nothing renders here.
     * ================================================================= */

    var _junction = "";        // the citation string, "" until chosen

    /** tree-relative path -> the citation spelling
     *  (project/topic/calc @ stage/run-N -- the calculation is depth 3,
     *  the same rule the send door's own depth guard measures). */
    function _citationFromRel(rel) {
        var seg = String(rel || "").split("/").filter(Boolean);
        if (seg.length < 4) return "";
        return seg.slice(0, 3).join("/") + "@" + seg.slice(3).join("/");
    }

    function _wireJunctionPicker() {
        var btn = _$("transport-junction-btn");
        if (!btn) return;
        btn.addEventListener("click", function () {
            var proj = root.molbuilder && root.molbuilder.projects;
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
                        var rel = (proj && proj.relativeToProjects)
                            ? proj.relativeToProjects(path) : path;
                        return root.fetch(
                            "/api/transport/describe_attempt?path="
                            + encodeURIComponent(rel))
                            .then(function (r) { return r.json(); })
                            .then(function (b) {
                                if (!b || b.ok === false) {
                                    throw new Error(
                                        (b && b.error) || "unreadable");
                                }
                                return b.summary || "";
                            });
                    },
                    confirmLabel: "Cite this attempt",
                });
            }).then(function (picked) {
                if (!picked) return;
                var proj = root.molbuilder && root.molbuilder.projects;
                var rel = (proj && proj.relativeToProjects)
                    ? proj.relativeToProjects(picked) : picked;
                var cite = _citationFromRel(rel);
                if (!cite) {
                    _setSendStatus("That folder is not an attempt inside "
                        + "a calculation (project/topic/calc/.../run-N).");
                    return;
                }
                _junction = cite;
                var out = _$("transport-junction-readout");
                if (out) out.textContent = cite;
                var meta = _$("transport-junction-meta");
                if (meta) {
                    meta.hidden = false;
                    meta.textContent = "Reading the attempt\u2019s own "
                        + ".fdf\u2026";
                    root.fetch("/api/transport/describe_attempt?path="
                               + encodeURIComponent(rel.replace(/^\/+/, "")))
                        .then(function (r) { return r.json(); })
                        .then(function (b) {
                            meta.textContent = (b && b.summary)
                                ? b.summary
                                : ((b && b.error) || "");
                        })
                        .catch(function () { meta.textContent = ""; });
                }
                _refreshSendButton();
            }).catch(function (e) {
                _setSendStatus("Picker failed: "
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
                say: function (kind, msg) { _setSendStatus(msg); },
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
        _wireCommitChannel();
        _restoreSession();
        _wireJunctionPicker();
        _wireSendButton(formContainer);
        _wireAutoDetectButton();
        _refreshSendButton();
        _refreshAutoDetectButton();
        _setCurrentStructureReadout("");
    }

    if (root.document) {
        if (root.document.readyState === "loading") {
            root.document.addEventListener("DOMContentLoaded", _init);
        } else {
            _init();
        }
    }
})(typeof window !== "undefined" ? window : globalThis);
