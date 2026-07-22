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
import { mount } from "/static/lib/molview/index.js";
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
    // time, so there is no separate label cache here.  See _shared.apply_labels_to_struct.

    function _refreshGenerateButton() {
        var btn = _$("transport-generate-btn");
        if (!btn) return;
        btn.disabled = !_currentStructureFile;
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

    // The mounted MolView handle (null until the first structure is committed).
    var _mvHandle = null;

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
        var mv   = root.molbuilder && root.molbuilder.molview;
        var ws   = root.molbuilder && root.molbuilder.workspace;
        var proj = root.molbuilder && root.molbuilder.projects;
        var host = _$("transport-molview-host");
        if (!mv || !ws || !host
                || typeof mount !== "function"
                || !proj || !proj.parser
                || typeof proj.parser.openMolecule !== "function") {
            return;
        }
        proj.parser.openMolecule(path).then(function (r) {
            if (r && r.ok === false) {
                if (root.console) {
                    root.console.error("[transport] load failed", r.error);
                }
                return;
            }
            if (_mvHandle && _mvHandle.ok) return;   // already mounted; the reload redrew it
            return mount(host, ws, { mode: "modify", owner: "transport" })
                // Cache ONLY a live handle (mount contract: failure -> {ok:false}),
                // so a failed mount doesn't permanently block a later remount.
                .then(function (h) { _mvHandle = (h && h.ok) ? h : null; });
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
                // Show the structure in the MolView inspection card (viewer +
                // selection/cell panel + view toggles).  Fire-and-forget: runs in
                // parallel with the auto-analyze below.  MolView reads the sidecar
                // labels (frozen atoms + electrode regions) as part of the load, so
                // the Generate POST sources them from molview.data — no separate
                // sidecarLabels.fetch, no path-pointer indirection.
                _showInMolview(f);
                _setCurrentStructureReadout(f);
                _refreshGenerateButton();
                _refreshAutoDetectButton();
                var name = f.split("/").pop();
                _setStatus("Structure: " + name
                    + " — Generate enabled.");
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

    /**
     * Render the issues list returned by /api/transport/render.
     * Severity-colored bullets matching the SIESTA + spectra
     * panels.  textContent on every node so unsanitised server
     * strings don't leak as HTML (XSS guard).
     */
    function _renderIssues(issues) {
        var panel = _$("transport-issues");
        var list  = _$("transport-issues-list");
        if (!panel || !list) return;
        var arr = (issues || []).filter(Boolean);

        // 2026-06-14 (F4b cross-tab consistency): TransportConfig
        // carries 21 ``workflow_group``-tagged fields.  Per
        // web-ui-coherence.md Rule 2 those issues route to the
        // per-card ``.card-issues[data-workflow-group="<role>"]``
        // UL inside the workflow-group card the form-schema
        // renderer drew.  Untagged issues land in the residual
        // panel.  Same fan-out shape as SIESTA/PySCF + spectra.
        var form = _$("transport-form-container");
        var cardPanels = form
            ? form.querySelectorAll(".card-issues[data-workflow-group]")
            : [];
        for (var c = 0; c < cardPanels.length; c++) {
            cardPanels[c].innerHTML = "";
            cardPanels[c].hidden    = true;
        }

        while (list.firstChild) list.removeChild(list.firstChild);
        if (!arr.length) {
            panel.hidden = true;
            return;
        }

        // Split by workflow_group → card buckets / residual.
        var buckets = {};
        var residual = [];
        for (var k = 0; k < arr.length; k++) {
            var g = arr[k].workflow_group;
            if (g && form) {
                if (!buckets[g]) buckets[g] = [];
                buckets[g].push(arr[k]);
            } else {
                residual.push(arr[k]);
            }
        }

        // Per-card fan-out.
        for (var c2 = 0; c2 < cardPanels.length; c2++) {
            var cp = cardPanels[c2];
            var role = cp.getAttribute("data-workflow-group");
            var bucket = buckets[role];
            if (!bucket || !bucket.length) continue;
            for (var b = 0; b < bucket.length; b++) {
                _appendCardIssue(cp, bucket[b]);
            }
            cp.hidden = false;
        }

        // Residual UL.
        if (!residual.length) {
            panel.hidden = true;
            return;
        }
        for (var i = 0; i < residual.length; i++) {
            var issue = residual[i];
            var li = document.createElement("li");
            li.className = "issue-" + (issue.severity || "info");
            var tag = document.createElement("strong");
            tag.textContent = (issue.severity || "info").toUpperCase();
            li.appendChild(tag);
            li.appendChild(document.createTextNode(" — "));
            var msg = document.createElement("span");
            msg.textContent = issue.message || "";
            li.appendChild(msg);
            if (issue.where) {
                var where = document.createElement("code");
                where.style.marginLeft = "0.6em";
                where.style.fontSize   = "0.85em";
                where.textContent = "[" + issue.where + "]";
                li.appendChild(where);
            }
            list.appendChild(li);
        }
        panel.hidden = false;
    }

    // Append one Issue to a per-card .card-issues UL.  Mirrors the
    // SIESTA/PySCF _appendIssueItem shape (data-severity attr +
    // .issue-msg + .issue-where) so card styling is consistent
    // across engines.  textContent on every node guards XSS.
    function _appendCardIssue(ul, issue) {
        var li = document.createElement("li");
        li.className = "issue-item";
        li.setAttribute("data-severity", issue.severity || "info");
        var msg = document.createElement("span");
        msg.className = "issue-msg";
        msg.textContent = issue.message || "";
        li.appendChild(msg);
        if (issue.where) {
            var tag = document.createElement("span");
            tag.className = "issue-where";
            tag.textContent = issue.where;
            li.appendChild(tag);
        }
        ul.appendChild(li);
    }

    /**
     * Show the generated script in a <pre> with copy + download
     * affordances.  Single-file preview today (transiesta is
     * one .fdf per click); multi-file output (electrode wizard,
     * bias-scan driver) will need a different layout.
     */
    function _showScriptPreview(filename, scriptText) {
        var details = _$("transport-script-preview");
        var name    = _$("transport-script-filename");
        var body    = _$("transport-script-body");
        var dl      = _$("transport-download-link");
        if (!details || !name || !body || !dl) return;
        name.textContent = filename;
        body.textContent = scriptText;
        var blob = new Blob([scriptText], { type: "text/plain" });
        var url  = URL.createObjectURL(blob);
        // Revoke any previous URL we created for the download link.
        if (dl.dataset.objectUrl) {
            try { URL.revokeObjectURL(dl.dataset.objectUrl); }
            catch (_) {}
        }
        dl.href = url;
        dl.download = filename;
        dl.dataset.objectUrl = url;
        details.hidden = false;
        details.open = true;
    }

    function _wireCopyButton() {
        var btn = _$("transport-copy-btn");
        if (!btn) return;
        btn.addEventListener("click", function () {
            var body = _$("transport-script-body");
            if (!body) return;
            var text = body.textContent;
            if (!text) return;
            if (root.navigator && root.navigator.clipboard
                && root.navigator.clipboard.writeText) {
                root.navigator.clipboard.writeText(text)
                    .then(function () { _setStatus("Copied to clipboard."); })
                    .catch(function () {
                        _setStatus("Copy failed — select the text manually.");
                    });
            } else {
                _setStatus("Clipboard API unavailable; "
                           + "select the text manually.");
            }
        });
    }

    function _wireGenerateButton(formContainer) {
        var btn = _$("transport-generate-btn");
        if (!btn) return;
        btn.addEventListener("click", function () {
            if (!_currentStructureFile) return;
            var fs = root.molbuilder && root.molbuilder.formSchema;
            if (!fs || typeof fs.collectForm !== "function") return;
            var schema = root.__transport_schema_for_test
                || _cachedSchema;
            if (!schema) return;
            var params;
            try { params = fs.collectForm(formContainer, schema); }
            catch (e) {
                _setStatus("Form has invalid values — fix them and retry.");
                return;
            }
            btn.disabled = true;
            _setStatus("Generating…");
            // Viewer-is-truth: source the frozen/region labels from the mounted
            // MolView model — the exact structure the user just inspected — so what
            // they SEE is what generates.  ``structure_path`` still ships for the
            // GEOMETRY (the server reads + parses it).  When MolView isn't loaded
            // (best-effort mount failed), OMIT the label keys so the server falls
            // back to the disk sidecar at structure_path (apply_labels_to_struct
            // keys on presence, not truthiness).
            var _genBody = { params: params, structure_path: _currentStructureFile };
            var _mvData = root.molbuilder && root.molbuilder.molview
                       && root.molbuilder.molview.data;
            if (_mvData
                    && typeof _mvData.getFrozen === "function"
                    && typeof _mvData.getRegions === "function"
                    && typeof _mvData.isEmpty === "function"
                    && !_mvData.isEmpty()) {
                _genBody.frozen_atoms = _mvData.getFrozen() || [];
                _genBody.regions      = _mvData.getRegions() || {};
            }
            root.fetch("/api/transport/render", {
                method:  "POST",
                headers: { "Content-Type": "application/json" },
                body:    JSON.stringify(_genBody),
            })
                .then(function (r) {
                    return r.json().then(function (body) {
                        return { status: r.status, body: body };
                    });
                })
                .then(function (resp) {
                    var b = resp.body;
                    _renderIssues(b && b.issues);
                    if (!b || !b.ok) {
                        _setStatus(b && b.error
                            ? b.error
                            : "Generate failed (HTTP " + resp.status + ").");
                        // Re-enable if the structure is still loaded;
                        // the user can fix params + retry.
                        _refreshGenerateButton();
                        return;
                    }
                    _showScriptPreview(b.filename, b.script);
                    _setStatus("Generated " + b.filename + ".");
                    _refreshGenerateButton();
                    // Form-dirty cleared: the parameter edits made
                    // it into the rendered .fdf, so a subsequent
                    // sidebar swap no longer needs to gate on the
                    // unsaved-modifications modal.
                    _formDirty = false;
                })
                .catch(function (e) {
                    _setStatus("Network error: "
                        + (e && e.message ? e.message : String(e)));
                    _refreshGenerateButton();
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
        _wireGenerateButton(formContainer);
        _wireCopyButton();
        _wireAutoDetectButton();
        _refreshGenerateButton();
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
