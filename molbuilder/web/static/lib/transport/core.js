/* Transport-calculation tab core — the COMPOSITE's describe surface
 * (archive/2026-09-01-transport-design.md § 4.1, P7b).
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
    /**
     * ONE fetch for BOTH surfaces (engines/transport.md 3.8.2): the
     * catalogue narrowed to the kind, split by the markers on the
     * server.  `surface` is "rung" -- the per-rung form, whose values
     * are override bags -- or "shared" -- the panel that edits the
     * template, whose values are the citation's answers, so it carries
     * the junction and is rendered again whenever the junction changes.
     * Resolves to the answer body; renders an error paragraph and
     * resolves to null on failure so the page never fails silently.
     */
    function _fetchSurface(surface, host, formSchema, rung, renderOpts) {
        var url = SCHEMA_URL + "?surface=" + surface
            + (surface === "shared" && _junction
               ? "&junction=" + encodeURIComponent(_junction) : "")
            + (rung ? "&rung=" + encodeURIComponent(rung) : "");
        return root.fetch(url)
            .then(function (r) {
                return r.json().then(function (body) {
                    if (!r.ok || !body.ok) {
                        throw new Error(body.error || "schema fetch failed");
                    }
                    return body;
                });
            })
            .then(function (body) {
                while (host.firstChild) host.removeChild(host.firstChild);
                formSchema.renderForm(host, body.schema, renderOpts || {});
                return body;
            })
            .catch(function (e) {
                _renderErrorParagraph(host,
                    "Could not load the " + surface + " form: "
                    + (e && e.message ? e.message : String(e)));
                return null;
            });
    }

    function _el(tag, attrs) {
        var n = root.document.createElement(tag);
        Object.keys(attrs || {}).forEach(function (k) {
            if (attrs[k] !== null && attrs[k] !== undefined) {
                n.setAttribute(k, attrs[k]);
            }
        });
        for (var i = 2; i < arguments.length; i++) {
            var kid = arguments[i];
            if (kid === null || kid === undefined) continue;
            n.appendChild(typeof kid === "string"
                          ? root.document.createTextNode(kid) : kid);
        }
        return n;
    }

    /* A card starts FOLDED when none of its fields is this rung's OWN
     * (its `stages` names the rung); the SCF, output and runtime cards
     * are one click away, the rung's own physics is what the eye lands
     * on (transport.md 3.8.2a). */
    function _foldedUnlessOwned(role, fields) {
        return !fields.some(function (f) {
            return f.stages && f.stages.length;
        });
    }

    /* THE PER-RUNG FORM IS A TAB PER RUNG (transport.md 3.8.2a): the
     * server answers the rung list -- name, ladder index, one-line note
     * -- and one schema per rung; a tab holds that rung's own items and
     * the ones any rung may set, each written into THAT rung's bag.  Its
     * values are kept across a reload (session storage, per rung), its
     * CHANGED fields are the bags the description carries. */
    function _fetchAndRender(formContainer, formSchema) {
        return root.fetch(SCHEMA_URL + "?surface=rung")
            .then(function (r) { return r.json(); })
            .then(function (body) {
                if (!body || !body.ok) {
                    throw new Error((body && body.error) || "schema fetch failed");
                }
                var rungs = body.rungs || [];
                // A REBUILD KEEPS THE TAB the person is on (the form is
                // rebuilt when the junction changes): read the active rung
                // off the strip being replaced, fall back to the first.
                var was = formContainer.querySelector(".tab-btn.active");
                var keep = (was && was.dataset.tab) || (rungs[0] && rungs[0].name);
                while (formContainer.firstChild) {
                    formContainer.removeChild(formContainer.firstChild);
                }
                var strip = _el("div", { "class": "tabs", role: "tablist",
                                         "aria-label": "Rung",
                                         id: "transport-rung-tabs" });
                var panels = [];
                rungs.forEach(function (r) {
                    var pid = "transport-rung-panel-" + r.name;
                    var word = r.name.replace("_", " ");
                    strip.appendChild(_el("button", {
                        type: "button", "class": "tab-btn", role: "tab",
                        "data-tab": r.name, "aria-controls": pid,
                        "aria-selected": "false" }, r.index + " \u00b7 " + word));
                    var host = _el("div", { "class": "param-grid" });
                    var panel = _el("div", { "class": "tab-panel", id: pid,
                                             role: "tabpanel", hidden: "" },
                        _el("p", { "class": "hint transport-rung-note" },
                            _el("strong", null, r.index + ". " + word + " \u2014 "),
                            r.note),
                        host);
                    panels.push({ rung: r.name, panel: panel, host: host });
                });
                formContainer.appendChild(strip);
                panels.forEach(function (p) { formContainer.appendChild(p.panel); });
                var ts = root.molbuilder && root.molbuilder.tabStrip;
                if (ts && rungs.length) {
                    var api = ts.mount(strip, {});
                    if (!api.select(keep)) api.select(rungs[0].name);
                }
                return Promise.all(panels.map(function (p) {
                    return _fetchSurface("rung", p.host, formSchema, p.rung,
                                         { foldable: true,
                                           folded: _foldedUnlessOwned })
                        .then(function (b) {
                            if (!b) return 0;
                            _rungSchemas[p.rung] = b.schema;
                            _rungHosts[p.rung] = p.host;
                            _restoreFormValues(p.host, b.schema, formSchema, p.rung);
                            _wirePersistence(p.host, b.schema, formSchema, p.rung);
                            return b.schema.sections.reduce(function (n, s) {
                                return n + (s.fields ? s.fields.length : 0);
                            }, 0);
                        });
                })).then(function (counts) {
                    _setStatus("Form loaded ("
                        + counts.reduce(function (a, b) { return a + b; }, 0)
                        + " settings over " + rungs.length + " rungs).");
                });
            })
            .catch(function (e) {
                _renderErrorParagraph(formContainer,
                    "Could not load the rung forms: "
                    + (e && e.message ? e.message : String(e)));
            });
    }

    function _formKey(rung) { return rung ? FORM_KEY + ":" + rung : FORM_KEY; }

    function _restoreFormValues(container, schema, formSchema, rung) {
        var raw;
        try { raw = root.sessionStorage.getItem(_formKey(rung)); }
        catch (_) { return; }
        if (!raw) return;
        var saved;
        try { saved = JSON.parse(raw); } catch (_) { return; }
        if (!saved || typeof saved !== "object") return;
        // BY THE SCHEMA, NOT BY A `name` ATTRIBUTE -- and that is the whole
        // of this function's history.  It queried `[name="<field>"]`, and
        // `form-schema.js` sets `id` on every control it builds and `name`
        // on none of them (makeNumber, makeSelect, makeTriSelect,
        // makeCheckbox, makeText, makeTriple -- all `id` only).  The two
        // are not even the same string: the id is `t-transmission-emin-ev`
        // where the saved key is `transmission_emin_ev`.
        //
        // So the restore matched ZERO elements on every load since it was
        // written, and the module docstring's promise that "a reload
        // restores the whole describe state" was false: the citation came
        // back (it rides the workspace note) so the page LOOKED restored,
        // while every parameter silently sat at its default.
        //
        // `setValues` is the renderer's own door and knows how each field
        // kind is built, which is what stops this drifting again.
        try {
            formSchema.setValues(container, schema, saved);
        } catch (_) {
            // A saved bag from an older schema can name a field this form
            // no longer has.  Best-effort, exactly like the write side.
        }
    }

    function _wirePersistence(container, schema, formSchema, rung) {
        var debounceHandle = null;
        function persist() {
            try {
                var values = formSchema.collectForm(container, schema);
                root.sessionStorage.setItem(
                    _formKey(rung), JSON.stringify(values));
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
                enforce: true,
            });
        }).catch(function (e) {
            if (root.console) {
                root.console.error("[transport] MolView load/mount failed", e);
            }
        });
    }

    // ---------- The chemistry card (step 3) ---------- //
    //
    // THE SHARED SURFACE (`lib/chemistry.js`): the junction's charge and
    // spin, resolved by the one electronic-state class for exactly what the
    // shared panel says -- the transport kind, so the charge is 0 by rule --
    // about the composed junction, the structure the description sends.
    // Attached once at start-up (`_init`); asked when the shared panel has
    // been rendered for a citation -- so never with the previous panel's
    // values -- and again on every edit to the panel's spin fields.  A
    // citation that does not compose leaves nothing to answer, and the card
    // is hidden.  It changes no setting.  (A "Re-analyze chemistry" button
    // stood here until 2026-09-28; the card follows the panel now, so there
    // is nothing to press.)
    var _chemistry = null;

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
    /* A WARNING WITH ITS ACTION ATTACHED, never a block (user ruling,
     * 2026-08-29).  Labels the reverse of the usual convention are a
     * valid junction that biases the other end; only the author knows
     * which end they meant, so the citation goes through and the
     * choice sits here.
     *
     * What is WRONG with it is said once, by the server, in the meta
     * line directly above this panel (sort.py::inverted_note, with the
     * measured z centroids).  This panel adds only what that sentence
     * cannot: the button, and what pressing it costs.
     *
     * The server answers `fix` as a WORD; nothing here matches prose. */
    var SWAP_LABEL = "Swap L-electrode and R-electrode";

    function _offerFix(described) {
        var host = _$("transport-junction-fix");
        if (!host) return;
        if (!described || described.fix !== "swap_electrodes") {
            host.hidden = true;
            host.textContent = "";
            return;
        }
        host.hidden = false;
        host.textContent = "";
        var say = root.document.createElement("span");
        say.className = "hint";
        say.textContent = "A rename, nothing else: the two labels trade "
            + "names in the cited folder.  No coordinate, keyword or "
            + "result is touched, and the relaxation stays valid.";
        var btn = root.document.createElement("button");
        btn.type = "button";
        btn.id = "transport-swap-electrodes-btn";
        btn.className = "full-btn";
        btn.textContent = SWAP_LABEL;
        btn.addEventListener("click", function () {
            btn.disabled = true;
            btn.textContent = "Swapping…";
            _swapElectrodes(described.citation, btn);
        });
        host.appendChild(say);
        host.appendChild(btn);
    }

    function _swapElectrodes(citation, btn) {
        function _failed(why) {
            btn.disabled = false;
            btn.textContent = SWAP_LABEL;
            _setStatus(why);
        }
        root.fetch("/api/transport/swap_electrodes", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ path: citation })
        }).then(function (r) { return r.json(); }).then(function (out) {
            if (!out || !out.ok) {
                _failed((out && out.error) || "The swap failed.");
                return;
            }
            /* Re-describe through the SAME door the picker uses: the
             * citation is unchanged, its labels are not, and every
             * answer (viewer, lane, meta line) must come from the
             * server's fresh reading -- never from patching what the
             * page already had.
             *
             * RETURNED, so the outer .catch covers it.  Left dangling
             * this chain had no handler of its own: a failed re-read
             * left the button disabled reading "Swapping…" with
             * nothing said, and only a reload got out of it. */
            return root.fetch("/api/transport/describe_attempt?path="
                + encodeURIComponent(citation))
                .then(function (r) { return r.json(); })
                .then(function (d) {
                    _adoptCitation(d, out.message + "  Re-cited: ");
                });
        }).catch(function (e) {
            _failed("The swap failed: " + e);
        });
    }

    function _adoptCitation(described, statusPrefix) {
        if (!described || !described.form || !described.citation) {
            // Not citable: the server's summary IS the condition,
            // naming the missing file.  Card 1, with the picker.
            // Clear any offer first -- it belongs to the PREVIOUS
            // citation, and a button left standing here would act on
            // that one.
            _offerFix(null);
            _setStatus((described && described.summary)
                || "That directory is not citable.");
            return;
        }
        _junction = described.citation;
        _junctionStructure = described.structure || null;
        // The shared panel follows the citation: its values are what the
        // cited directory answers (3.8.1).
        _fetchAndRenderShared(root.molbuilder && root.molbuilder.formSchema);
        var out = _$("transport-junction-readout");
        if (out) out.textContent = _junction;
        var meta = _$("transport-junction-meta");
        if (meta) {
            meta.hidden = false;
            meta.textContent = described.summary || "";
        }
        _offerFix(described);
        _writePanelNote();
        _refreshSendButton();
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
    /* {rung: {item: value}} -- each rung's CHANGED fields, its own bag;
     * a rung with nothing changed sends no bag.  `null` when a panel holds
     * an invalid value: the caller says so. */
    function _changedByRung() {
        var fs = root.molbuilder && root.molbuilder.formSchema;
        if (!fs || typeof fs.diffFromDefaults !== "function") return {};
        var bags = {};
        try {
            Object.keys(_rungHosts).forEach(function (rung) {
                var bag = {};
                fs.diffFromDefaults(_rungHosts[rung], _rungSchemas[rung])
                    .forEach(function (d) { bag[d.name] = d.current; });
                if (Object.keys(bag).length) bags[rung] = bag;
            });
            return bags;
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
            var bags = _changedByRung();
            if (bags === null) {
                _setSendStatus("A rung's form has invalid values — fix them "
                    + "and retry.");
                return;
            }
            var shared = _sharedValues();
            if (shared === null) {
                _setSendStatus("The shared panel has invalid values — fix "
                    + "them and retry.");
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
                // THE DESCRIPTION CHECK'S FINDINGS, through the one
                // renderer, in card 5's panel -- every send redraws it, so
                // a clean one clears what the last one said.
                showFindings: function (issues) {
                    var vf = (root.molbuilder || {}).validationFindings;
                    var panel = _$("transport-send-findings");
                    if (vf && panel) vf.render(issues, { panel: panel });
                },
                engine: "siesta",
                calculation: "transport",
                junction: _junction,
                bias: bias,
                stages: bags,
                shared: shared,
            });
        });
    }

    // Each rung's schema and panel host, populated by _fetchAndRender, so
    // the Describe handler diffs every rung's panel without re-fetching.
    var _rungSchemas = {};
    var _rungHosts = {};
    var _sharedSchema = null;      // the shared panel's, by _fetchAndRenderShared

    /* THE SHARED PANEL (engines/transport.md 3.8.2): the items the
     * catalogue marks `shared` for transport, which bind every rung and
     * edit the TEMPLATE.  Its values are the citation's answers -- a
     * deck's, a record's, or none -- so it is rendered again whenever
     * the junction changes, and the line under its heading names the
     * source (3.8.1).  A `citation` row the citation does not answer
     * renders blank: not chosen (3.8.3). */
    function _fetchAndRenderShared(formSchema) {
        var host = _$("transport-shared-container");
        if (!host || !formSchema) return Promise.resolve();
        return _fetchSurface("shared", host, formSchema).then(function (body) {
            if (body) _sharedSchema = body.schema;
            // The panel is rendered for the junction now (or could not be):
            // the chemistry card answers for what it says, about the
            // junction the tab holds -- or hides, with none.
            if (_chemistry) _chemistry.refresh();
            if (!body) return;
            var line = _$("transport-shared-source");
            if (!line) return;
            var src = body.source || {kind: "none", name: ""};
            line.textContent = !_junction
                ? "No junction cited yet: these are the catalogue's starting "
                  + "values."
                : src.kind === "deck"
                ? "Values from the run you cited (" + src.name + ").  Change "
                  + "any of them; a change applies to all five rungs at once."
                : src.kind === "record"
                ? "Values recorded with the structure you cited"
                  + (src.name ? " (" + src.name + ")" : "") + ".  Change any "
                  + "of them; a change applies to all five rungs at once."
                : "The citation carries no deck and no record, so it answers "
                  + "none of these.  A blank field is not chosen: the "
                  + "template records no value for it and prep falls back to "
                  + "the catalogue's default until you choose.";
        });
    }

    /* What the shared panel says now -- every field, not only the changed
     * ones: the panel edits the template, and the template answers all
     * five rungs. */
    function _sharedValues() {
        var fs = root.molbuilder && root.molbuilder.formSchema;
        var host = _$("transport-shared-container");
        if (!fs || !host || !_sharedSchema
                || typeof fs.collectForm !== "function") {
            return {};
        }
        try { return fs.collectForm(host, _sharedSchema); }
        catch (e) { return null; }
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
        var chem = root.molbuilder && root.molbuilder.chemistry;
        if (chem && typeof chem.attach === "function") {
            _chemistry = chem.attach({
                kind: "transport",
                forms: function () {
                    var host = _$("transport-shared-container");
                    return (host && _sharedSchema)
                        ? { siesta: { host: host, schema: _sharedSchema } }
                        : {};
                },
                structure: function () { return _junctionStructure; },
            });
        }
        _fetchAndRender(formContainer, formSchema);
        _fetchAndRenderShared(formSchema);
        _restoreSession();
        _wireJunctionPicker();
        _wireSendButton(formContainer);
        _refreshSendButton();
    }

    if (root.document) {
        if (root.document.readyState === "loading") {
            root.document.addEventListener("DOMContentLoaded", _init);
        } else {
            _init();
        }
    }
})(typeof window !== "undefined" ? window : globalThis);
