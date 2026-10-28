/* Transport-calculation tab core — the COMPOSITE's describe surface
 * (engines/transport.md § 3.8).
 *
 * ONE driver: the junction citation.  Picking the relaxed junction -- a
 * run's folder, or a structure pair's .xyz (the shared tree-picker; what
 * it holds makes it citable, engines/transport.md § 3.1) -- sets
 * everything downstream — the viewer loads the CITED
 * calculation's own labeled structure, the chemistry analysis runs on
 * it, and Describe has the FINISHED description written into the
 * selected folder by its server door (no hand-over -- user ruling
 * 2026-08-29: nothing is awaiting, so the tab selects and decides; the
 * shared sender in lib/task-handover.js names the folder, and the door
 * refuses a described one -- transport.md § 3.8.2a).  There is no sidebar commit channel here: a
 * second way to fill the viewer would be a second source for the
 * composite's one fact (molview.md § 9.3a, one level up).
 *
 * What the forms hold (schema-driven from /api/transport/schema) -- the
 * shared panel and each rung's tab -- and the cited junction persist
 * through the workspace's one door, each under the tab's own tag
 * (`transport:form`, `transport:panel`; workspace.md § 4), so a reload
 * restores them; the bias list, its treatment and the electrode rename
 * start over (`web/tabs.md`).
 */
import { mount } from "/static/lib/molview/index.js";
import { fitRow } from "/static/lib/transport/fit.js";

/* WHO THIS TAB'S SAVED WORK BELONGS TO (workspace.md § 4) — the one string used
 * both as the viewer's `owner` and as the tag on any workspace call, so the two
 * cannot drift into naming different slots. */
const WORKSPACE_TAG = "transport";
(function (root) {
    "use strict";

    var SCHEMA_URL = "/api/transport/schema";

    /* THE ONE STATUS WRITER (`lib/status.js`): its severities are the ones
     * `page-shell.css` declares, so a failure reads as one on every tab. */
    function _setStatus(msg, kind) {
        root.molbuilder.status.set("transport-status", msg || "", kind || null);
    }

    /* A panel that could not be drawn says why, in its place -- a status
     * line of its own, through the same writer. */
    function _renderErrorParagraph(host, msg) {
        while (host.firstChild) host.removeChild(host.firstChild);
        var p = root.document.createElement("p");
        host.appendChild(p);
        root.molbuilder.status.set(p, msg, "error");
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
     * template.  BOTH are drawn from the template this describe will write
     * (plan § 5w K7), so both carry the junction and are rendered again
     * whenever it changes; a rung's tab also carries what the shared panel
     * holds, since a rung's value can follow a shared one (the
     * transmission grid starts at the SCF's).
     * Resolves to the answer body; renders an error paragraph and
     * resolves to null on failure so the page never fails silently.
     */
    /* ``isCurrent``, given, is asked before anything is drawn: an answer
     * that lands after a newer ask went out draws nothing (``null``). */
    function _fetchSurface(surface, host, formSchema, rung, renderOpts, isCurrent) {
        var url = SCHEMA_URL + "?surface=" + surface
            + "&engine=" + encodeURIComponent(_engine())
            + (_junction ? "&junction=" + encodeURIComponent(_junction) : "")
            + (_reuses ? "&reuses=" + encodeURIComponent(_reuses) : "")
            + (rung ? "&rung=" + encodeURIComponent(rung) : "")
            + (rung && _junction
               ? "&shared=" + encodeURIComponent(JSON.stringify(_sharedHeld()))
               : "");
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
                if (isCurrent && !isCurrent()) return null;
                while (host.firstChild) host.removeChild(host.firstChild);
                formSchema.renderForm(host, body.schema, renderOpts || {});
                return body;
            })
            .catch(function (e) {
                if (isCurrent && !isCurrent()) return null;
                _renderErrorParagraph(host,
                    "Could not load the " + surface + " form: "
                    + (e && e.message ? e.message : String(e)));
                return null;
            });
    }

    /* THE ENGINE THE CALCULATION IS DESCRIBED ON is the engine strip's
     * active tab (engines/transport.md 3.8.8) -- drawn by the page from
     * the engines molbuilder knows; an engine that runs no transport is a
     * disabled tab, so it is never this. */
    function _engine() {
        var b = root.document.querySelector(
            "#transport-engine-tabs .tab-btn[aria-selected=\"true\"]");
        return (b && b.dataset.tab) || "";
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
    function _foldedUnlessOwned(card, fields) {
        return !fields.some(function (f) {
            return f.stages && f.stages.length;
        });
    }

    /* A RUNG'S NAME AS EVERY SURFACE HERE SAYS IT -- its tab, its panel's
     * note and the rail's stage list: the ladder index and the name, "1 ·
     * seed".  One spelling, kept apart from the cards' "1." steps, so the
     * page's two numberings never read as one. */
    function _rungLabel(r) { return r.index + " \u00b7 " + r.name; }

    /* THE RAIL'S STAGE LIST is the rung list the tab strip is built from
     * -- name, ladder index, one-line note (transport/stages.py
     * RUNG_NOTES) -- so the two say the same thing from one answer. */
    function _renderStageFlow(rungs) {
        var host = root.document.getElementById("transport-stage-flow");
        if (!host) return;
        while (host.firstChild) host.removeChild(host.firstChild);
        rungs.forEach(function (r) {
            host.appendChild(_el("li", null,
                _el("strong", null, _rungLabel(r)),
                " \u2014 " + r.note));
        });
    }

    /* THE PER-RUNG FORM IS A TAB PER RUNG (transport.md 3.8.2a): the
     * server answers the rung list -- name, ladder index, one-line note
     * -- and one schema per rung; a tab holds that rung's own items and
     * the ones any rung may set, each written into THAT rung's bag.  Its
     * values are kept across a reload (the `transport:form` note, per
     * rung), and what it HOLDS is the bag the description carries -- the
     * template's values are what a blank field runs (form-schema.md § 1.1). */
    function _fetchAndRender(formContainer, formSchema) {
        // A SEQUENCE, like the other in-flight guards here: the tabs are
        // drawn again at each citation and each change on the shared panel,
        // and a draw superseded while its answers were in flight touches
        // nothing -- a late one would point `_rungHosts` at a tab already
        // gone from the page, and the Send would read that.
        var seq = ++_rungsSeq;
        return root.fetch(SCHEMA_URL + "?surface=rung&engine="
                          + encodeURIComponent(_engine()))
            .then(function (r) { return r.json(); })
            .then(function (body) {
                if (!body || !body.ok) {
                    throw new Error((body && body.error) || "schema fetch failed");
                }
                if (seq !== _rungsSeq) return false;
                var rungs = body.rungs || [];
                _renderStageFlow(rungs);
                // A REBUILD KEEPS THE TAB the person is on (the form is
                // rebuilt when the junction changes): read the active rung
                // off the strip being replaced, fall back to the first.
                var was = formContainer.querySelector(
                    '.tab-btn[aria-selected="true"]');
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
                    strip.appendChild(_el("button", {
                        type: "button", "class": "tab-btn", role: "tab",
                        "data-tab": r.name, "aria-controls": pid,
                        "aria-selected": "false" }, _rungLabel(r)));
                    var host = _el("div", { "class": "param-grid" });
                    var panel = _el("div", { "class": "tab-panel", id: pid,
                                             role: "tabpanel", hidden: "" },
                        _el("p", { "class": "hint" },
                            _el("strong", null, _rungLabel(r) + " \u2014 "),
                            r.note),
                        // THE WINDOW FITTED TO THE FINISHED GRID, beside its
                        // fields, for a calculation that reuses one
                        // (engines/transport.md \u00a7 2a.16).
                        r.name === "transmission" && _reuses
                            ? fitRow({ reuses: _reuses,
                                       name: _extend ? _extend.name : _reuses,
                                       use: function (v) { _useWindow(v); } })
                            : null,
                        // AN EXTENSION'S RUNG: what a change here re-runs,
                        // or whose values it holds -- and, on the
                        // transmission's, what a wider window re-runs
                        // (\u00a7 2a.16; `Extension.says`, drawn as it is).
                        _extend ? _el("p", { "class": "hint" },
                            _extend.says[r.name]) : null,
                        _extend && r.name === "transmission"
                            ? _el("p", { "class": "hint" },
                                  _extend.says.window) : null,
                        host);
                    panels.push({ rung: r.name, panel: panel, host: host });
                });
                formContainer.appendChild(strip);
                panels.forEach(function (p) { formContainer.appendChild(p.panel); });
                // AN EXTENSION OFFERS ITS PER-VOLTAGE STAGES ALONE: the
                // fixed ones' tabs are drawn but not shown, every field
                // locked and never collected -- the describe door writes
                // the finished calculation's own values for them (§ 2a.16,
                // *On the page*).
                if (_extend) {
                    var offered = rungs.filter(function (r) {
                        return _extend.fixed.indexOf(r.name) === -1;
                    });
                    _extend.fixed.forEach(function (name) {
                        var b = strip.querySelector(
                            '.tab-btn[data-tab="' + name + '"]');
                        if (b) b.hidden = true;
                    });
                    if (_extend.fixed.indexOf(keep) !== -1 && offered[0]) {
                        keep = offered[0].name;
                    }
                }
                var ts = root.molbuilder && root.molbuilder.tabStrip;
                if (ts && rungs.length) {
                    var api = ts.mount(strip, {});
                    _rungTabs = api;
                    if (!api.select(keep)) api.select(rungs[0].name);
                }
                return Promise.all(panels.map(function (p) {
                    // A tab HOLDS THE RUNG'S OWN VALUES: the template's are
                    // what a blank field runs, shown as its hint
                    // (form-schema.md § 1.1).
                    return _fetchSurface("rung", p.host, formSchema, p.rung,
                                         { foldable: true,
                                           folded: _foldedUnlessOwned,
                                           holds: "overrides" })
                        .then(function (b) {
                            if (!b || seq !== _rungsSeq) return false;
                            _rungSchemas[p.rung] = b.schema;
                            _rungHeld[p.rung] = b.held || {};
                            _rungHosts[p.rung] = p.host;
                            _restoreFormValues(p.host, b.schema, formSchema,
                                               p.rung, function () {
                                                   return seq === _rungsSeq;
                                               }, b.held);
                            _wirePersistence(p.host, b.schema, formSchema, p.rung);
                            return true;
                        });
                })).then(function (drew) {
                    // THE NEWEST DRAW SETTLES DESCRIBE: drawn when every
                    // rung's tab drew; a draw superseded since says
                    // nothing -- the newer one settles it.
                    if (seq !== _rungsSeq) return false;
                    _panelsDrawn = drew.every(Boolean);
                    _echoBias();
                    _refreshSendButton();
                    _schedulePlan();
                    return _panelsDrawn;
                });
            })
            .catch(function (e) {
                _renderErrorParagraph(formContainer,
                    "Could not load the rung forms: "
                    + (e && e.message ? e.message : String(e)));
                if (seq === _rungsSeq) {
                    _panelsDrawn = false;
                    _refreshSendButton();
                }
                return false;
            });
    }

    /* WHAT THE FORMS HOLD -- the shared panel's values and each rung's
     * tab -- kept through the workspace's one door under the tab's own tag
     * (`workspace.md` §§ 4-5), so a reload brings back what was typed:
     * one note, `{v, junction, shared, rungs: {rung: values}}`, the draft
     * slot of `transport:form`.  It belongs to the JUNCTION it was typed
     * for: citing another -- another calculation -- starts an empty note,
     * never the last one's values. */
    var FORM_TAG = WORKSPACE_TAG + ":form";
    var _formNote = null;          // the note as last read or written
    var _formNoteRead = null;      // its one read, a promise
    var _restoring = false;        // a restore is writing the fields back

    function _formIdentity(ws) {
        return { workspace_id: ws.workspaceId(FORM_TAG), state_index: 0 };
    }

    /** The stored note, read once per page -- every restore and every save
     *  waits for it, so a save never writes over values not yet read. */
    function _readFormNote() {
        if (_formNoteRead) return _formNoteRead;
        var ws = root.molbuilder && root.molbuilder.workspace;
        _formNoteRead = (ws && typeof ws.readState === "function")
            ? Promise.resolve(ws.readState(_formIdentity(ws)))
                .catch(function () { return null; })
            : Promise.resolve(null);
        _formNoteRead = _formNoteRead.then(function (note) {
            _formNote = (note && note.v === 1) ? note : null;
            return _formNote;
        });
        return _formNoteRead;
    }

    /** One surface's values in the note -- ``"shared"`` or a rung's name. */
    function _heldIn(note, surface) {
        if (!note) return null;
        return surface === "shared" ? note.shared
                                    : (note.rungs || {})[surface];
    }

    /* A surface just drawn gets back what it held, for the junction it was
     * typed for.  BY THE SCHEMA, NOT BY A `name` ATTRIBUTE: `setValues` is
     * the renderer's own door and knows how each field kind is built. */
    /* ``isCurrent``, given, is asked when the note is in hand: a surface
     * drawn again since (another citation, a change on the shared panel)
     * gets nothing -- its newer draw restores for itself. */
    /* ``held``, given, is what the surface starts from when the note
     * holds nothing of it -- a rung's own values in the calculation this
     * one reuses (engines/transport.md § 2a.16). */
    function _restoreFormValues(container, schema, formSchema, surface,
                                isCurrent, held) {
        return _readFormNote().then(function () {
            // THE NOTE AS IT IS NOW, not as it was read at load: the saves
            // since (`_saveSurface`) write `_formNote`, and a redraw after an
            // edit must put the edit back.
            var note = _formNote;
            if (isCurrent && !isCurrent()) return;
            var mine = note && note.junction === _noteKey();
            var saved = mine ? _heldIn(note, surface) : null;
            var typed = !!(saved && typeof saved === "object");
            if (!typed) saved = held || null;
            if (!saved || typeof saved !== "object") return;
            // TYPED FOR THE FINISHED CALCULATION BEFORE, on a stage it
            // offers, and not its own value: said (`_paintRestored`).  A
            // field left blank is no value typed: blank against a value it
            // never set is the same (`_sameValue`).
            var put = saved;
            if (_extend && surface !== "shared") {
                var own = held || {};
                var names = typed ? Object.keys(saved).filter(function (k) {
                    return !_blank(saved[k]) && !_sameValue(saved[k], own[k]);
                }) : [];
                // A BLANK IS NO VALUE TYPED (engines/transport.md § 2a.16,
                // *On the page*): the stage's own values, the values typed
                // over them -- never a blank put over its own.
                if (typed) {
                    put = Object.assign({}, own);
                    Object.keys(saved).forEach(function (k) {
                        if (!_blank(saved[k])) put[k] = saved[k];
                    });
                }
                if (names.length) {
                    _restored[surface] = {
                        container: container, schema: schema, held: own,
                        values: saved, names: names,
                        label: function (n) {
                            var found = null;
                            (schema.sections || []).forEach(function (sec) {
                                (sec.fields || []).forEach(function (f) {
                                    if (f.name === n) found = f.label || n;
                                });
                            });
                            return found || n;
                        },
                    };
                } else {
                    delete _restored[surface];
                }
                _paintRestored();
            }
            // WRITING BACK IS NOT AN EDIT: the fields' input and change
            // events would each save the note it came from (`_saveSurface`).
            _restoring = true;
            try {
                formSchema.setValues(container, schema, put);
            } catch (_) {
                // A note from an older schema can name a field this form
                // no longer has; the rest is put back.
            } finally {
                _restoring = false;
            }
        });
    }

    /* Two values of one field the same -- a blank (`null`, `undefined`,
     * "") the same as a blank, whatever spells it (`form-schema.md`
     * § 1.1: a blank is not chosen). */
    function _blank(v) { return v === null || v === undefined || v === ""; }

    function _sameValue(a, b) {
        if (_blank(a) || _blank(b)) return _blank(a) && _blank(b);
        return String(a) === String(b);
    }

    /* A value as the page says it: a list's entries space-separated, a
     * blank a dash. */
    function _said(value) {
        if (_blank(value)) return "—";
        return Array.isArray(value) ? value.join(" ") : String(value);
    }

    /* What one surface holds, into the note -- field by field
     * (`formSchema.heldValues`): a field that will not read keeps the value
     * it was last saved with, so one half-typed field does not stop every
     * later edit on the surface being kept. */
    function _saveSurface(container, schema, formSchema, surface) {
        var ws = root.molbuilder && root.molbuilder.workspace;
        if (!ws || typeof ws.persist !== "function") return;
        // A RESTORE IS NOT AN EDIT, and values typed with NO JUNCTION cited
        // belong to no calculation -- nor are they kept: a reload draws the
        // forms before it re-cites, and a save then would write over the
        // note the citation is about to bring back.
        if (_restoring || !_junction) return;
        _readFormNote().then(function () {
            var junction = _noteKey();
            var note = (_formNote && _formNote.junction === junction)
                ? _formNote : { v: 1, junction: junction, shared: {}, rungs: {} };
            var values;
            try {
                values = formSchema.heldValues(container, schema,
                                               _heldIn(note, surface));
            } catch (_) { return; }
            if (surface === "shared") note.shared = values;
            else (note.rungs = note.rungs || {})[surface] = values;
            _formNote = note;
            ws.persist(FORM_TAG, note, _formIdentity(ws));
        });
    }

    /* Saved on EVERY edit, not after a pause: the surfaces are drawn again
     * at each citation and each change on the shared panel, and the restore
     * after a redraw is what puts the person's values back -- a save that
     * waited could miss a value typed just before one. */
    function _wirePersistence(container, schema, formSchema, surface) {
        function persist() {
            _saveSurface(container, schema, formSchema, surface);
            _schedulePlan();
        }
        container.addEventListener("input", persist);
        container.addEventListener("change", persist);
    }

    /* THE TAB'S FACTS: the citation (a run's folder or a pair's .xyz
     * that satisfies § 3.1's condition), the composed labeled
     * structure the server answers with (the viewer + the chemistry
     * analysis run on it -- no file path is assumed).  Both are
     * /api/transport/describe_attempt's answers, adopted whole -- the tab
     * derives neither. */
    var _junction = "";           // the citation path, "" until cited
    var _junctionStructure = null;    // the composed structure envelope
    /* THE TRANSPORT CALCULATION THIS ONE REUSES -- its folder, "" for none
     * (engines/transport.md \u00a7 2a.16): picked in the junction's place,
     * its junction is the citation and its description fills the cards. */
    var _reuses = "";
    /* WHAT AN EXTENSION OF IT MAY CHANGE, AND WHAT IS FIXED -- the describe
     * door's `extend` for the calculation reused (`fixed`, `offered`, and
     * `says`, each control's sentence), null for a new calculation. */
    var _extend = null;
    /* Each rung's own values in the calculation reused, as its tab drew
     * them -- what a new calculation from its junction starts from. */
    var _rungHeld = {};
    /* THE CALCULATION REUSED, AS ITS FOLDER ANSWERED -- its own junction,
     * which *Use <name>'s junction* returns to (engines/transport.md
     * \u00a7 2a.16, *A frame set of the same junction*); null for none. */
    var _reusedAnswer = null;

    /* WHAT THE FORMS' NOTE AND THE PANEL'S NOTE ARE KEPT FOR -- the
     * calculation reused, or the junction cited: what a reload describes
     * again. */
    function _noteKey() { return _reuses || _junction || ""; }

    /* The composite's send gate: a citation is the ONE thing the
     * describe cannot go without (transport.md § 3.1) -- and its panels
     * drawn for it, since Describe sends what they hold: before the shared
     * panel and the rungs' tabs answer for a new citation, they still hold
     * the last one's values. */
    var _panelsDrawn = false;
    function _refreshSendButton() {
        var btn = _$("transport-send-btn");
        if (!btn) return;
        btn.disabled = !_junction || !_panelsDrawn;
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

    function _writePanelNote(keep) {
        var ws = root.molbuilder && root.molbuilder.workspace;
        if (!ws || typeof ws.persist !== "function") return;
        /* The tab's ONE fact is the CITATION.  A reload
         * re-describes it through the same seam a pick uses, so the
         * structure, the meta line and both surfaces always come back
         * fresh from the server, never from a stale copy. */
        // ``keep``: a frame set the page could not ask about, kept for the
        // next reload to ask again.
        ws.persist(PANEL_TAG, { v: 3, junction: _noteKey(),
                               frame_set: keep !== undefined ? keep
                                          : (_citesFrameSet() ? _junction
                                                              : null) },
                   _panelIdentity(ws));
    }

    /**
     * Mount the viewer if it is not already up; resolves to the handle or
     * null.  Its own function so the init-restore
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
     * documented no-op (§ 11.2a).  The note is
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
            // The rename is not kept: a reload starts it over (tabs.md).
            var mine = ++_citeSeq;
            return _describeAttempt(note.junction, false)
                    .then(function (b) {
                if (mine !== _citeSeq) return;     // a citation made since
                _adoptCitation(b, "Restored your last session: ");
                // ...AND THE FRAME SET CITED IN ITS JUNCTION'S PLACE, asked
                // again against it (engines/transport.md \u00a7 2a.16).
                if (!note.frame_set || !_reuses) return;
                var reuses = _reuses;
                function lost(why) {
                    _setStatus("Your last session's frame set, "
                        + note.frame_set + ", no longer cites (" + why
                        + ") \u2014 " + reuses + "'s own junction is held.",
                        "warn");
                }
                return _describeAttempt(note.frame_set, false, reuses)
                        .then(function (f) {
                    if (mine !== _citeSeq || reuses !== _reuses) return;
                    if (f && f.citation) {
                        _showJunction(f, "Restored your last session: ");
                    } else {
                        lost((f && f.summary) || "it is not a frame set "
                             + "of that junction");
                    }
                }, function (e) {
                    if (mine !== _citeSeq || reuses !== _reuses) return;
                    // THE QUESTION FAILED, not the set: kept, and asked
                    // again at the next reload.
                    _writePanelNote(note.frame_set);
                    _setStatus("Your last session's frame set, "
                        + note.frame_set + ", could not be asked about ("
                        + (e && e.message ? e.message : String(e))
                        + ") \u2014 " + reuses + "'s own junction is held "
                        + "for now; a reload asks again.", "warn");
                });
            });
        }).catch(function (e) {
            /* SAID ON THE PAGE, through the one status writer (F8): the
             * citation of the last session is gone or no longer
             * describes; the person cites again. */
            _setStatus("Your last session's citation could not be read "
                       + "(" + (e && e.message ? e.message : e)
                       + ") -- cite the junction again.", "warn");
        });
    }

    /**
     * Show the CITED structure so the user can eyeball what they cited —
     * atoms, region labels, the cell — before describing.  Read-only view
     * of the citation (molview.md § 12.3), installed from the SERVER'S
     * OWN composition (describe_attempt answers the structure envelope,
     * the same shape /api/build/load's {structure} branch takes).
     * `enforce` makes a new citation a deliberate swap
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
    // is hidden.  It changes no setting.
    var _chemistry = null;

    /* =================================================================
     *  The COMPOSITE (transport.md § 3.8): cite the junction,
     *  state the bias, Describe -- the describe door writes the finished
     *  description into the selected folder (no hand-over; user ruling
     *  2026-08-29; transport.md § 3.8.2a).  The tab's facts are declared
     *  once, above.
     * ================================================================= */

    /** One fetch answers everything about a picked path: whether
     *  it is citable, the summary line, what its composition measured
     *  (findings), the composed structure for the viewer, `fix` and the
     *  citation spelling (the server's).  ``rel`` is tree-relative
     *  (proj.relativeToProjects). */
    /* THE RENAME IS THIS CALCULATION'S OWN (transport.md § 4; user,
     * 2026-10-04): the choice lives here until Describe writes it into the
     * description (`swap_electrodes: true`), and every describe of the
     * citation is read WITH it, so the viewer, the findings and the meta
     * line show the junction as this calculation will compose it.  The cited
     * junction is never written.  The choice is about ONE citation: a pick
     * of a junction starts without it, and the value held is the one the
     * adopted answer echoes (`_adoptCitation`). */
    var _swapElectrodesChoice = false;
    /* THE NEWEST CITATION ASKED FOR -- a restore, a pick or a swap: an
     * answer for an earlier one that lands late is dropped, never adopted
     * over the person's latest choice. */
    var _citeSeq = 0;
    /* AN EXTENSION HAS ITS OWN LAYOUT (engines/transport.md § 2a.16,
     * *On the page*), five parts in order, each card headed as its part
     * (`data-extending`) and the prose a new calculation reads not shown
     * (`data-new-only`): 1. what you are doing (`_paintMode`) and the way
     * out on a row of its own; 2. the junction -- the finished one as it
     * ran, or a frame set of it; 3. what you can change (the voltages, the
     * treatment, the window, the device's and the transmission's tabs);
     * 4. the values fixed by the finished calculation, folded and view
     * only; 5. the plan, then Describe.  The shared panel, the chemistry
     * card and the fixed stages' tabs are the finished calculation's, and
     * are not drawn as controls. */
    var RUNG_WORDS = { seed: "The seed", electrode_L: "The left lead",
                       electrode_R: "The right lead" };
    function _paintExtend() {
        var host = _$("transport-extend");
        var bias = _$("transport-bias-reruns");
        var out = _$("transport-extend-out");
        var fixedCard = _$("transport-fixed-card");
        var fixedHost = _$("transport-extend-fixed-host");
        ["shared-card", "chemistry-card"].forEach(function (id) {
            var card = _$(id);
            if (card) card.hidden = !!_extend;
        });
        root.document.querySelectorAll("[data-extending]").forEach(
            function (h) {
                if (h.dataset.newTitle === undefined) {
                    h.dataset.newTitle = h.textContent;
                }
                h.textContent = _extend ? h.dataset.extending
                                        : h.dataset.newTitle;
            });
        root.document.querySelectorAll("[data-new-only]").forEach(
            function (el) { el.hidden = !!_extend; });
        root.document.querySelectorAll("[data-extension-only]").forEach(
            function (el) { el.hidden = !_extend; });
        [bias, out, fixedHost].forEach(function (el) {
            if (el) el.textContent = "";
        });
        if (bias) bias.hidden = !_extend;
        if (out) out.hidden = !_extend;
        if (fixedCard) fixedCard.hidden = !_extend;
        // VALUES TYPED FOR IT BEFORE, which the forms' note brought back
        // -- said above the stages' tabs, with the way back to its own.
        _paintRestored();
        if (!host) return;
        host.textContent = "";
        host.hidden = !_extend;
        if (!_extend) return;
        var name = _extend.name;
        // THE JUNCTION IT DESCRIBES WITH: the finished one as it ran, its
        // frames counted -- or a frame set of it (engines/transport.md
        // § 2a.16, *A frame set of the same junction*).
        var own = _reusedAnswer ? _reusedAnswer.citation : _junction;
        var set = _citesFrameSet();
        var ran = ((_reusedAnswer || {}).frame_set || {}).n_frames || 1;
        // ...AND THE ONE CITED IN ITS PLACE: a frame set, or one geometry.
        var cited = ((_shownAnswer || {}).frame_set || {}).n_frames || 1;
        var choice = _el("div", { "class": "choice-list",
                                  id: "transport-extend-junction-choice" },
            _el("strong", null, "2. The junction"));
        function radio(value, id, words) {
            var input = _el("input", { type: "radio", id: id, value: value,
                                       name: "transport-extend-junction" });
            input.checked = (value === "set") === set;
            choice.appendChild(_el("label", null, input, " " + words));
            return input;
        }
        var ownRadio = radio("own", "transport-extend-own-btn",
            name + "'s junction, as " + name + " ran it — "
            + (ran === 1 ? "one frame" : ran + " frames"));
        var setRadio = radio("set", "transport-extend-set",
            "a frame set of it, its first frame " + name + "'s junction");
        var pick = _el("button", { type: "button", "class": "btn",
                                   id: "transport-extend-frameset-btn" },
                       set ? "Choose another frame set…"
                           : "Choose a frame set…");
        pick.addEventListener("click", function () { _citeFrameSet(name); });
        choice.appendChild(pick);
        ownRadio.addEventListener("change", function () {
            // THE LATEST CHOICE STANDS: a frame set still being described
            // is superseded (`_citeSeq`).
            ++_citeSeq;
            if (_reusedAnswer && _citesFrameSet()) {
                _showJunction(_reusedAnswer,
                              "Back to " + name + "'s junction: ");
            }
        });
        setRadio.addEventListener("change", function () {
            if (_lastFrameSet && _lastFrameSet.reuses === _reuses) {
                ++_citeSeq;
                _showJunction(_lastFrameSet.answer,
                              "Cited a frame set of " + name
                              + "'s junction: ");
            } else {
                _citeFrameSet(name);
            }
        });
        choice.appendChild(_el("p", { "class": "hint",
                                      id: "transport-extend-junction" },
            set && cited === 1
                ? "Junction: " + _junction + " — one geometry, the "
                  + "junction " + name + " ran at its first frame.  The "
                  + "plan below says which of " + name
                  + "'s results are reused."
            : set ? "Junction: " + _junction + " — a frame set whose "
                  + "first frame is the junction " + name + " ran.  The "
                  + "device and the transmission run once per frame; the "
                  + "plan below says which of " + name
                  + "'s results are reused."
                : "Junction: " + own + " — the one " + name
                  + " ran, " + (ran === 1 ? "one frame" : ran + " frames")
                  + ".  Without a frame set, the extension describes with "
                  + "it."));
        host.appendChild(choice);
        // PART 4, FIXED BY IT: folded, view only -- drawn when opened, from
        // the panels' locked fields (`_paintFixed`).
        var title = _$("transport-fixed-title");
        if (title) title.textContent = "4. Fixed by " + name;
        var fixed = _el("details", { id: "transport-extend-fixed" },
            _el("summary", null, "The shared values, and the seed's and "
                + "both leads', as " + name + " ran them (view only) — "
                + "changing one is a new calculation: the way out above"),
            _el("div", { "class": "hint",
                         id: "transport-extend-fixed-list" }));
        fixed.addEventListener("toggle", function () {
            if (fixed.open) _paintFixed(name);
        });
        if (fixedHost) fixedHost.appendChild(fixed);
        // PART 1'S WAY OUT, on a row of its own and styled apart from what
        // stays, naming what it starts from.
        var leave = _el("button", { type: "button", "class": "btn",
                                    id: "transport-extend-leave-btn" },
                        "Start a new calculation from "
                        + (set ? _junction : name + "'s junction")
                        + " with these settings");
        leave.addEventListener("click", function () {
            _startFromJunction(leave, name);
        });
        if (out) out.appendChild(leave);
        if (bias) {
            // THE BIAS LIST'S AND THE TREATMENT'S SENTENCES, as served.
            bias.appendChild(_el("span", null, _extend.says.bias));
            bias.appendChild(_el("br"));
            bias.appendChild(_el("span", null, _extend.says.treatment));
        }
    }

    /* The last frame set cited while extending, kept so choosing "a frame
     * set" again brings it back without the picker. */
    var _lastFrameSet = null;
    /* The describe door's answer for the junction shown -- the finished
     * one, or what is cited in its place. */
    var _shownAnswer = null;

    /* WHAT THE FINISHED CALCULATION FIXES, view only: each locked field of
     * the shared panel and of the fixed stages' tabs, as it ran. */
    function _paintFixed(name) {
        var list = _$("transport-extend-fixed-list");
        if (!list || !_extend) return;
        list.textContent = "";
        var parts = [["Shared by every stage", _sharedSchema]].concat(
            _extend.fixed.map(function (r) {
                return [RUNG_WORDS[r] || r, _rungSchemas[r]];
            }));
        parts.forEach(function (part) {
            var items = [];
            ((part[1] || {}).sections || []).forEach(function (sec) {
                (sec.fields || []).forEach(function (f) {
                    if (f.locked) {
                        var v = _said(f.locked.value);
                        items.push((f.label || f.name) + ": " + v
                                   + (f.unit && v !== "\u2014"
                                      ? " " + f.unit : ""));
                    }
                });
            });
            if (items.length) {
                list.appendChild(_el("p", null,
                    _el("strong", null, part[0] + " — "),
                    items.join(" · ")));
            }
        });
        if (!list.firstChild) {
            list.appendChild(_el("p", null, "Still drawing " + name
                                 + "'s values…"));
        }
    }

    /* VALUES TYPED FOR THE FINISHED CALCULATION BEFORE, brought back by the
     * forms' note onto a stage it offers, which differ from its own: said
     * where they land, with *Use <name>'s own values* -- a value restored
     * unseen re-runs what it changes (engines/transport.md § 2a.16,
     * *On the page*). */
    var _restored = {};
    function _paintRestored() {
        var host = _$("transport-extend-restored");
        if (!host) return;
        host.textContent = "";
        var surfaces = Object.keys(_restored);
        host.hidden = !(_extend && surfaces.length);
        if (host.hidden) return;
        var said = [];
        surfaces.forEach(function (surface) {
            var r = _restored[surface];
            r.names.forEach(function (n) {
                said.push((RUNG_WORDS[surface] || surface) + " " + r.label(n)
                          + " " + _said(r.values[n]));
            });
        });
        host.appendChild(document.createTextNode(
            "Restored values you typed for " + _extend.name + " before: "
            + said.join(", ") + ".  "));
        var back = _el("button", { type: "button", "class": "btn",
                                   id: "transport-extend-own-values-btn" },
                       "Use " + _extend.name + "'s own values");
        back.addEventListener("click", function () {
            var fs = (root.molbuilder || {}).formSchema;
            Object.keys(_restored).forEach(function (surface) {
                var r = _restored[surface];
                var values = {};
                // ITS OWN VALUE, or BLANK -- `null`, the renderer's blank
                // for every field kind -- where it set none.
                r.names.forEach(function (n) {
                    values[n] = Object.prototype.hasOwnProperty.call(
                        r.held, n) ? r.held[n] : null;
                });
                if (fs) {
                    fs.setValues(r.container, r.schema, values);
                    _saveSurface(r.container, r.schema, fs, surface);
                }
            });
            _restored = {};
            _paintRestored();
            _schedulePlan();
        });
        host.appendChild(back);
    }

    /* WHAT YOU ARE DOING, in one line (engines/transport.md § 2a.16,
     * *On the page*): EXTENDING a finished calculation, or a NEW
     * CALCULATION from the junction cited. */
    function _paintMode() {
        var line = _$("transport-mode");
        if (!line) return;
        line.textContent = "";
        line.hidden = !_junction;
        if (!_junction) return;
        if (_extend) {
            var name = _extend.name;
            var answer = _reusedAnswer || {};
            var ran = (answer.frame_set || {}).n_frames || 1;
            var volts = (answer.bias || [0]).map(function (v) {
                return v === 0 ? "0.0" : String(v);
            }).join(", ");
            line.appendChild(_el("strong", null, "EXTENDING " + name));
            line.appendChild(document.createTextNode(
                " — it describes " + volts + " V, "
                + (ran === 1 ? "one frame" : ran + " frames")
                + (answer.window ? ", " + answer.window : "") + ".  Its "
                + "finished results are copied wherever nothing changed "
                + "— the plan below names each; only what you change "
                + "below is computed, and " + name + " is never "
                + "modified."));
        } else {
            line.appendChild(_el("strong", null, "NEW CALCULATION"));
            line.appendChild(document.createTextNode(
                " from " + _junction + "."));
        }
    }

    /* THE WAY OUT: a new calculation from this junction, which reuses
     * nothing, its cards starting from the finished calculation's settings
     * -- written as the forms' note for the junction, which its surfaces
     * restore when they draw (`_restoreFormValues`). */
    function _startFromJunction(btn, name) {
        var formSchema = (root.molbuilder || {}).formSchema;
        var ws = root.molbuilder && root.molbuilder.workspace;
        btn.disabled = true;
        var shared = {};
        // ONLY WHAT A PERSON CHOSE travels (the lock's `chosen`): a value
        // cited from the junction is cited again, and a default stays
        // not chosen.
        ((_sharedSchema || {}).sections || []).forEach(function (sec) {
            sec.fields.forEach(function (f) {
                if (f.locked && f.locked.chosen) {
                    shared[f.name] = f.locked.value;
                } else if (f.locked && f.locked.blank) {
                    // A VALUE NOBODY CHOSE -- a default, or a cited value
                    // the person emptied -- stays blank, not cited anew.
                    shared[f.name] = null;
                }
            });
        });
        var rungs = {};
        Object.keys(_rungSchemas).forEach(function (r) {
            rungs[r] = _extend.fixed.indexOf(r) !== -1
                ? (_rungHeld[r] || {})
                : formSchema.heldValues(_rungHosts[r], _rungSchemas[r], null);
        });
        var citation = _junction;
        // THE EXTENSION STAYS until the new calculation's answer is
        // adopted (`_adoptCitation` leaves it): a request that fails leaves
        // the page as it was, the forms' note put back.
        var before = _formNote;
        function putBack() {
            _formNote = before;
            if (before && ws && typeof ws.persist === "function") {
                ws.persist(FORM_TAG, before, _formIdentity(ws));
            }
        }
        _formNote = { v: 1, junction: citation, shared: shared,
                      rungs: rungs };
        if (ws && typeof ws.persist === "function") {
            ws.persist(FORM_TAG, _formNote, _formIdentity(ws));
        }
        var mine = ++_citeSeq;
        _describeAttempt(citation, _swapElectrodesChoice)
            .then(function (d) {
                if (mine !== _citeSeq) return;
                if (!d || !d.citation) {
                    putBack();
                    btn.disabled = false;
                    _setStatus((d && d.summary)
                        || "That junction does not cite.", "warn");
                    return;
                }
                _adoptCitation(d, "A new calculation from this junction, "
                    + "starting from " + name + "'s settings \u2014 it "
                    + "reuses nothing.  Cited ");
            })
            .catch(function (e) {
                putBack();
                btn.disabled = false;
                _setStatus("The junction did not cite: " + e, "error");
            });
    }

    function _describeAttempt(rel, swap, reuses) {
        return root.fetch("/api/transport/describe_attempt?path="
                          + encodeURIComponent(rel.replace(/^\/+/, ""))
                          + (swap ? "&swap_electrodes=1" : "")
                          + (reuses ? "&reuses=" + encodeURIComponent(reuses)
                                    : ""))
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

    var WITHDRAW_LABEL = "Withdraw the rename";
    function _offerFix(described) {
        var host = _$("transport-junction-fix");
        if (!host) return;
        /* Shown while the composed junction is inverted (the offer), and
         * while this calculation holds the rename (so it can be withdrawn
         * and so it is never a silent state); hidden otherwise. */
        var holds = !!(described && described.swap_electrodes);
        // A CALCULATION REUSED keeps its rename: its own, and not offered
        // (engines/transport.md \u00a7 2a.16).
        if (!described || described.reuses || _reuses
                || (described.fix !== "swap_electrodes" && !holds)) {
            host.hidden = true;
            host.textContent = "";
            return;
        }
        host.hidden = false;
        host.textContent = "";
        var say = root.document.createElement("span");
        say.className = "hint";
        say.textContent = (holds
            ? "This calculation composes the junction with L-electrode and "
              + "R-electrode traded (swap_electrodes: true); the cited "
              + "junction is unchanged.  "
            : "")
            + "A rename in this calculation's own copy of the junction: "
            + "the two labels trade names when it is composed, and the "
            + "description records it.  The cited junction is never written; "
            + "another calculation citing it makes its own choice.";
        var btn = root.document.createElement("button");
        btn.type = "button";
        btn.id = "transport-swap-electrodes-btn";
        btn.className = "full-btn";
        btn.textContent = holds ? WITHDRAW_LABEL : SWAP_LABEL;
        btn.addEventListener("click", function () {
            btn.disabled = true;
            btn.textContent = _swapElectrodesChoice ? "Withdrawing…" : "Swapping…";
            _swapElectrodes(described.citation, btn);
        });
        host.appendChild(say);
        host.appendChild(btn);
    }

    function _swapElectrodes(citation, btn) {
        function _failed(why) {
            btn.disabled = false;
            btn.textContent = _swapElectrodesChoice ? WITHDRAW_LABEL : SWAP_LABEL;
            _setStatus(why, "error");
        }
        /* The choice is made HERE and read back through the SAME door the
         * picker uses: the citation is unchanged, this calculation's
         * reading of its labels is not, and every answer (viewer, findings,
         * meta line) comes from the server's composition with the swap --
         * never from patching what the page already had. */
        var want = !_swapElectrodesChoice;
        var mine = ++_citeSeq;
        _describeAttempt(citation, want)
            .then(function (d) {
                if (mine !== _citeSeq) return;     // a citation made since
                _adoptCitation(d, (want
                    ? "L-electrode and R-electrode trade names in this "
                      + "calculation (swap_electrodes: true).  Re-cited: "
                    : "The rename is withdrawn.  Re-cited: "));
            })
            .catch(function (e) {
                _failed("The swap failed: " + e);
            });
    }

    var _adoptSeq = 0;             // the newest citation adopted
    function _adoptCitation(described, statusPrefix) {
        if (!described || !described.citation) {
            // Not citable: the server's summary IS the refusal, naming
            // what is missing and the whole condition -- said in the status
            // line, and nothing else changes: the citation held stays the
            // citation, with its findings and its offer.
            _setStatus((described && described.summary)
                || "That is not citable.", "warn");
            return;
        }
        _junction = described.citation;
        _junctionStructure = described.structure || null;
        _swapElectrodesChoice = !!described.swap_electrodes;
        _reuses = described.reuses || "";
        _extend = _reuses ? (described.extend || null) : null;
        _reusedAnswer = _reuses ? described : null;
        _rungHeld = {};
        _restored = {};
        _lastFrameSet = null;
        if (_reuses) {
            // ITS VOLTAGES AND THEIR TREATMENT, from its description
            // (engines/transport.md \u00a7 2a.16) -- the person's to change.
            var list = _$("transport-bias");
            if (list) {
                list.value = (described.bias || [0]).map(function (v) {
                    return v === 0 ? "0.0" : String(v);
                }).join(", ");
            }
            root.document.querySelectorAll(
                'input[name="transport-bias-treatment"]').forEach(function (b) {
                    b.checked = String(described.low_bias_approximation)
                        === b.value;
                });
            _syncTreatment();
        }
        // BOTH SURFACES FOLLOW THE CITATION: the shared panel holds what the
        // citation answers (3.8.1), and each rung's tab shows what
        // that rung runs -- drawn from the template the panel describes, so
        // after it.  Describe waits for both (`_refreshSendButton`); a newer
        // citation's draw supersedes this one's.
        // The last citation's schemas go with it: nothing reads a panel
        // through a schema it was not drawn from.
        var drawn = ++_adoptSeq;
        // ...AND EVERY RUNG DRAW STILL IN FLIGHT is the last citation's:
        // superseded, it settles nothing (`_fetchAndRender`).
        _rungsSeq++;
        _panelsDrawn = false;
        _refreshSendButton();
        _sharedSchema = null;
        _rungSchemas = {};
        _rungHosts = {};
        // THE RUNGS' NEWEST DRAW SETTLES DESCRIBE; a shared panel that did
        // not draw leaves it waiting, and says why in its own place.
        _fetchAndRenderShared(root.molbuilder && root.molbuilder.formSchema)
            .then(function (ok) {
                if (ok && drawn === _adoptSeq) _renderRungs();
            }, function (e) {
                if (drawn !== _adoptSeq) return;
                _setStatus("The shared panel did not draw: "
                    + (e && e.message ? e.message : String(e)), "error");
            });
        _showJunction(described, statusPrefix);
    }

    /* THE JUNCTION THE TAB DESCRIBES WITH, shown -- its readout, the meta
     * line, its frames, what its composition measured, the rename's offer,
     * the extension's junction line, the viewer: one path for a citation
     * adopted and for a frame set cited in an extension's junction's place
     * (engines/transport.md \u00a7 2a.16). */
    function _showJunction(described, statusPrefix) {
        var vf = (root.molbuilder || {}).validationFindings;
        var fpanel = _$("transport-junction-findings");
        _junction = described.citation;
        _junctionStructure = described.structure || null;
        _shownAnswer = described;
        // A FRAME SET CITED IN AN EXTENSION'S JUNCTION'S PLACE, kept --
        // picked or brought back by a reload -- so choosing a frame set
        // again brings it back without the picker.
        if (_citesFrameSet()) {
            _lastFrameSet = { reuses: _reuses, answer: described };
        }
        // DESCRIBE'S CARD SPEAKS OF WHAT THE PAGE HOLDS: what it said of
        // another junction -- its message, its findings, its plan -- goes.
        _setSendStatus("");
        var sentFindings = _$("transport-send-findings");
        if (vf && sentFindings) vf.render([], { panel: sentFindings });
        var plan = _$("transport-send-manifest");
        if (plan) {
            plan.hidden = true;
            plan.textContent = "";
        }
        var out = _$("transport-junction-readout");
        if (out) out.textContent = _junction;
        var meta = _$("transport-junction-meta");
        if (meta) {
            meta.hidden = false;
            meta.textContent = described.summary || "";
        }
        /* WHAT ITS FRAMES ARE (transport.md § 3.8.1), by the one card. */
        var fsHost = _$("transport-junction-frameset");
        import("../frameset/card.js").then(function (mod) {
            mod.renderFrameSet(fsHost, described.frame_set || null);
        });
        /* WHAT THE COMPOSITION MEASURED -- each lead's measurements, the
         * orientation warning, a refusal -- through the one renderer
         * (validation.md 4.1 R2), under the one-line summary. */
        if (vf && fpanel) vf.render(described.findings || [], { panel: fpanel });
        _offerFix(described);
        _paintExtend();
        _paintMode();
        _writePanelNote();
        _refreshSendButton();
        _schedulePlan();
        /* Honest state, not a gate (strict composition refuses at
         * PREP; describing ahead is legal).  The summary already says
         * CONCLUDED / not / no-record; add the road note only when
         * prep would refuse today. */
        var late = described.concluded === false
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

    /* Whether the junction held is a frame set cited in the place of the
     * junction of the calculation reused. */
    function _citesFrameSet() {
        return !!(_reuses && _reusedAnswer
                  && _junction !== _reusedAnswer.citation);
    }

    /* A FRAME SET OF THE FINISHED JUNCTION, cited in its place
     * (engines/transport.md \u00a7 2a.16, *A frame set of the same
     * junction*): the one pop-out picker, each citation it selects -- a
     * frame set's .xyz, or any other whose first frame is that junction --
     * described against that junction; one whose first frame is another
     * junction answers its refusal, and the junction held stays held. */
    function _citeFrameSet(name) {
        var proj = root.molbuilder && root.molbuilder.projects;
        function toRel(path) {
            return (proj && proj.relativeToProjects)
                ? String(proj.relativeToProjects(path) || "") : path;
        }
        var reuses = _reuses;
        import("../tree-picker.js").then(function (mod) {
            return mod.pickPath({
                title: "Cite a frame set of " + name + "'s junction",
                hint: "Pick a frame set's .xyz -- a vibration mode's "
                    + "frames -- whose first frame is the junction " + name
                    + " ran.  \u25b8 expands.",
                mode: "any",
                describe: function (path) {
                    return _describeAttempt(toRel(path), false, reuses)
                        .then(function (b) { return b.summary || ""; });
                },
                confirmLabel: "Cite this",
            });
        }).then(function (picked) {
            if (!picked) {
                _paintExtend();             // the choice as the page holds it
                return;
            }
            var mine = ++_citeSeq;
            return _describeAttempt(toRel(picked), false, reuses)
                    .then(function (b) {
                if (mine !== _citeSeq || reuses !== _reuses) return;
                if (!b || !b.citation) {
                    _paintExtend();
                    _setStatus((b && b.summary)
                        || "That is not a frame set of " + name
                           + "'s junction.", "warn");
                    return;
                }
                _showJunction(b, "Cited a frame set of " + name
                                 + "'s junction: ");
            });
        }).catch(function (e) {
            _setStatus("Picker failed: "
                + (e && e.message ? e.message : String(e)), "error");
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
            /* THE one pop-out picker (lib/tree-picker.js), taking either
             * kind of transport.md \u00a7 3.1 -- a finished relaxation run's
             * folder (a run of ours, launched through jobset), or a
             * structure pair's .xyz -- and the describe seam classifies each
             * selection so the meta line answers before you confirm. */
            import("../tree-picker.js").then(function (mod) {
                return mod.pickPath({
                    title: "Cite the relaxed junction",
                    hint: "Pick the finished relaxation RUN of the "
                        + "junction (<calculation>/<NN>_<stage>/run-<n>, "
                        + "launched through jobset), or a structure pair's "
                        + ".xyz -- a frame set's included -- or a transport "
                        + "calculation's folder, to reuse it.  \u25b8 expands.",
                    mode: "any",
                    describe: function (path) {
                        return _describeAttempt(toRel(path), false)
                            .then(function (b) { return b.summary || ""; });
                    },
                    confirmLabel: "Cite this",
                });
            }).then(function (picked) {
                if (!picked) return;
                var rel = toRel(picked);
                var dropped = _swapElectrodesChoice;
                var extending = !!_reuses;
                var mine = ++_citeSeq;
                return _describeAttempt(rel, false).then(function (b) {
                    if (mine !== _citeSeq) return; // a citation made since
                    _adoptCitation(b, dropped && b && b.citation
                        ? "The electrode rename stayed with the last "
                          + "citation.  Cited " : null);
                    // LEAVING AN EXTENSION for an unrelated citation: its
                    // voltages were the finished calculation's -- started
                    // over, as a reload does (§ 2a.16, *On the page*).
                    if (extending && b && b.citation && !_reuses) {
                        _resetBias();
                    }
                });
            }).catch(function (e) {
                _setStatus("Picker failed: "
                    + (e && e.message ? e.message : String(e)), "error");
            });
        });
    }

    function _setSendStatus(msg, kind) {
        root.molbuilder.status.set("transport-send-status", msg || "",
                                   kind || null);
    }

    /* {rung: {item: value}} -- what each rung's tab HOLDS, its own bag
     * (form-schema.md § 1.1: a tab holds the rung's values, the template's
     * shown as what a blank field runs).  A blank is not chosen and not
     * sent; a rung holding nothing sends no bag.  Every value the person
     * gave is sent -- one equal to the catalogue's default, or set on an
     * item that has none.
     * Throws, naming the field, on a value that will not read. */
    function _bagsByRung() {
        var fs = root.molbuilder && root.molbuilder.formSchema;
        if (!fs) return {};
        var bags = {};
        Object.keys(_rungHosts).forEach(function (rung) {
            var held = fs.collectForm(_rungHosts[rung], _rungSchemas[rung]);
            var bag = {};
            Object.keys(held).forEach(function (k) {
                if (held[k] !== null) bag[k] = held[k];
            });
            if (Object.keys(bag).length) bags[rung] = bag;
        });
        return bags;
    }

    function _wireSendButton(formContainer) {
        var btn = _$("transport-send-btn");
        if (!btn) return;
        btn.addEventListener("click", function () {
            var mb = root.molbuilder || {};
            if (!mb.taskHandover) {
                _setSendStatus("lib/task-handover.js is not loaded.", "error");
                return;
            }
            var body;
            try {
                body = _describeBody();
            } catch (e) {
                if (e && e.bias) _refuseBias(e.message);
                else _setSendStatus(e && e.message ? e.message : String(e),
                                    "error");
                return;
            }
            // DESCRIBE'S ANSWER STANDS in the plan's place.
            _cancelPlan();
            var manifestBox = _$("transport-send-manifest");
            if (manifestBox) {
                manifestBox.hidden = true;
                manifestBox.textContent = "";
            }
            mb.taskHandover.send({
                projects: mb.projects,
                say: function (kind, msg) { _setSendStatus(msg, kind); },
                // WHAT A CALCULATION THAT REUSES A FINISHED ONE WILL COPY,
                // AND WHAT RUNS -- Describe's answer, the manifest's words
                // (engines/transport.md \u00a7 2a.16).
                showManifest: function (lines) {
                    if (!manifestBox) return;
                    manifestBox.hidden = !(lines && lines.length);
                    manifestBox.textContent = lines ? lines.join("\n") : "";
                },
                // THE DESCRIPTION CHECK'S FINDINGS, through the one
                // renderer, in card 5's panel -- every send redraws it, so
                // a clean one clears what the last one said.
                showFindings: function (issues) {
                    var vf = (root.molbuilder || {}).validationFindings;
                    var panel = _$("transport-send-findings");
                    if (!vf || !panel) return;
                    // BESIDE THE CONTROL IT NAMES (validation.md 4.1 R2):
                    // a rung's field by the schema its tab rendered, the
                    // bias list and its treatment by their rows -- and
                    // that rung's tab brought up first, so a refusal is
                    // read where it is fixed.
                    var first = (issues || []).filter(function (i) {
                        return i && i.severity === "error" && i.stage
                            && _rungHosts[i.stage];
                    })[0];
                    if (first && _rungTabs) _rungTabs.select(first.stage);
                    vf.render(issues, {
                        panel: panel,
                        formScope: _$("parameters-card"),
                        fieldIds: _fieldIds(),
                        reveal: true,
                    });
                },
                engine: body.engine,
                calculation: body.calculation,
                junction: body.junction,
                // THE CALCULATION IT REUSES, written as `reuses`
                // (engines/transport.md \u00a7 2a.16).
                reuses: body.reuses,
                swap_electrodes: body.swap_electrodes,
                bias: body.bias,
                low_bias_approximation: body.low_bias_approximation,
                stages: body.stages,
                shared: body.shared,
            });
        });
    }

    /* WHAT DESCRIBE SENDS, and what the plan is asked with
     * (engines/transport.md \u00a7 2a.16, *On the page*): the citation, the
     * calculation reused, the rename, the voltages and -- for several --
     * their treatment, each rung's own values and the shared panel's.  A
     * list that does not read throws with `bias` set; a value that will
     * not read as its type, naming its field. */
    function _describeBody() {
        var bias = _biasList();
        if (bias.some(isNaN)) {
            var e = new Error("Bias must be comma-separated volts, "
                              + "e.g. 0.0,0.2");
            e.bias = true;
            throw e;
        }
        // THE LOW-BIAS APPROXIMATION, for several voltages -- the person's
        // choice, true or false; the description's check refuses a list
        // without it.
        var picked = root.document.querySelector(
            'input[name="transport-bias-treatment"]:checked');
        return {
            engine: _engine(),
            calculation: "transport",
            junction: _junction,
            reuses: _reuses,
            swap_electrodes: _swapElectrodesChoice,
            bias: bias,
            low_bias_approximation: bias.length > 1 && picked
                ? picked.value === "true" : null,
            stages: _bagsByRung(),
            shared: _sharedValues(),
        };
    }

    /* THE PLAN, BEFORE ANYTHING IS WRITTEN (engines/transport.md
     * \u00a7 2a.16, *On the page*): an extension's -- what is copied from the
     * finished calculation and what is computed -- asked of the describe
     * door with `plan: true` whenever anything that changes it changes, a
     * moment after the last change; a newer answer supersedes an older
     * one (`_planSeq`).  A new calculation copies nothing: no plan. */
    var _planSeq = 0;
    var _planTimer = null;
    /* NO PLAN ON ITS WAY: one asked is superseded, one waiting dropped, its
     * line gone -- at every change, and when Describe answers instead. */
    function _cancelPlan() {
        if (_planTimer) {
            root.clearTimeout(_planTimer);
            _planTimer = null;
        }
        ++_planSeq;
        var status = _$("transport-plan-status");
        if (status) {
            status.hidden = true;
            status.textContent = "";
        }
    }

    function _schedulePlan() {
        _cancelPlan();
        if (!_reuses) return;
        var status = _$("transport-plan-status");
        if (status) {
            status.hidden = false;
            status.textContent = "Working out the plan\u2026";
        }
        _planTimer = root.setTimeout(_askPlan, 600);
    }

    function _askPlan() {
        _planTimer = null;
        var status = _$("transport-plan-status");
        var box = _$("transport-send-manifest");
        if (!_reuses || !_panelsDrawn || !status || !box) return;
        var body;
        try {
            body = _describeBody();
        } catch (e) {
            status.hidden = false;
            status.textContent = "The plan waits for the values above to "
                + "read: " + (e && e.message ? e.message : String(e));
            return;
        }
        body.plan = true;
        var mine = ++_planSeq;
        var name = _extend ? _extend.name : _reuses;
        status.hidden = false;
        status.textContent = "Working out the plan\u2026";
        root.fetch("/api/transport/describe", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(body),
        }).then(function (r) { return r.json(); })
            .then(function (a) {
                if (mine !== _planSeq) return;
                if (!a || !a.ok) {
                    box.hidden = true;
                    box.textContent = "";
                    status.textContent = "The plan could not be worked "
                        + "out: " + ((a && a.error) || "no answer");
                    return;
                }
                var lines = a.manifest || [];
                box.hidden = !lines.length;
                box.textContent = lines.join("\n");
                status.textContent = "The plan \u2014 what is copied from "
                    + name + ", and what is computed.  Nothing is copied "
                    + "until prep; Describe writes the description.";
            })
            .catch(function (e) {
                if (mine !== _planSeq) return;
                status.textContent = "The plan could not be worked out: "
                    + (e && e.message ? e.message : String(e));
            });
    }

    /* THE VOLTAGES STARTED OVER -- 0.0, no treatment -- as a reload starts
     * them: an extension left for an unrelated citation (\u00a7 2a.16, *On
     * the page*). */
    function _resetBias() {
        var list = _$("transport-bias");
        if (list) list.value = "0.0";
        root.document.querySelectorAll(
            'input[name="transport-bias-treatment"]').forEach(function (b) {
                b.checked = false;
            });
        _syncTreatment();
    }

    // Each rung's schema and panel host, populated by _fetchAndRender, so
    // the Describe handler reads every rung's panel without re-fetching.
    var _rungSchemas = {};
    var _rungHosts = {};
    var _rungTabs = null;          // the rung strip's handle (select(rung))
    var _rungsSeq = 0;             // the newest draw of the rung tabs
    var _sharedSeq = 0;            // the newest draw of the shared panel

    /* A REFUSAL OF THE BIAS LIST BY THIS PAGE lands where the codec's does:
     * beside the list, through the one renderer (validation.md 4.1 R2),
     * and the status line says it too, as a Send's refusal does. */
    function _refuseBias(message) {
        var vf = (root.molbuilder || {}).validationFindings;
        var panel = _$("transport-send-findings");
        if (vf && panel) {
            vf.render([{ severity: "error", message: message,
                         where: "task.bias" }],
                      { panel: panel, formScope: _$("parameters-card"),
                        fieldIds: _fieldIds(), reveal: true });
        }
        _setSendStatus(message, "error");
    }

    /* {stage -> {field name -> the DOM id its tab gave it}} from the schemas
     * this page rendered -- the id rule is the schema's, never derived here
     * -- plus the description's own two rows (`task.bias`, its treatment),
     * whose ids name the row itself.  The map the findings renderer places
     * by. */
    function _fieldIds() {
        var out = {
            "task.bias": "transport-bias-row",
            "task.bias.low_bias_approximation": "transport-bias-treatment",
        };
        Object.keys(_rungSchemas).forEach(function (rung) {
            var m = {};
            ((_rungSchemas[rung] || {}).sections || []).forEach(function (s) {
                (s.fields || []).forEach(function (f) { m[f.name] = f.id; });
            });
            out[rung] = m;
        });
        return out;
    }
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
        if (!host || !formSchema) return Promise.resolve(false);
        // A RUNG'S VALUE CAN FOLLOW A SHARED ONE -- the transmission grid
        // starts at the SCF's -- so a change here draws the rungs' tabs
        // again from what the panel now holds.  One listener for the host,
        // which outlives every render of its contents.
        if (!host.dataset.rungsFollow) {
            host.dataset.rungsFollow = "1";
            var t = null;
            host.addEventListener("change", function () {
                // A RESTORE IS NOT AN EDIT: the panel's draw waits for its
                // restore before the rungs are drawn from it.
                if (_restoring) return;
                if (t) clearTimeout(t);
                t = setTimeout(function () {
                    // A VALUE THAT WILL NOT READ is said beside its field,
                    // and the rungs keep the template the panel last
                    // described: drawn without it, a hint that followed the
                    // person's value would fall back to the cited one while
                    // the panel still showed theirs.
                    try { _sharedValues(); } catch (_) { return; }
                    _renderRungs();
                }, 300);
            });
            // ...AND WHAT IT HOLDS IS KEPT, as a rung's tab is
            // (`_wirePersistence`): one listener for the host, saving
            // through the schema the panel is drawn from at that moment.
            var keep = function () {
                if (_sharedSchema) {
                    _saveSurface(host, _sharedSchema, formSchema, "shared");
                }
            };
            host.addEventListener("input", keep);
            host.addEventListener("change", keep);
        }
        // A SEQUENCE, as the rung tabs': the panel's host outlives each
        // citation, so an answer for an earlier citation that lands late
        // must not draw that citation's panel under this one.
        var seq = ++_sharedSeq;
        return _fetchSurface("shared", host, formSchema, null, null,
                             function () { return seq === _sharedSeq; })
                .then(function (body) {
            if (seq !== _sharedSeq) return false;
            var restored = Promise.resolve();
            if (body) {
                _sharedSchema = body.schema;
                // What the panel held, for this junction, comes back --
                // before the rungs are drawn from it, so a rung value that
                // follows a shared one follows the person's.
                restored = _restoreFormValues(
                    host, body.schema, formSchema, "shared",
                    function () { return seq === _sharedSeq; });
            }
            return restored.then(function () {
                _paintShared(body);
                return !!body;          // drawn, or the form did not load
            });
        });
    }

    function _paintShared(body) {
        // The panel is rendered for the junction now (or could not be):
        // the chemistry card answers for what it says, about the
        // junction the tab holds -- or hides, with none.
        if (_chemistry) _chemistry.refresh();
        if (!body) return;
        var line = _$("transport-shared-source");
        if (!line) return;
        var src = body.source || {kind: "none", name: ""};
        line.textContent = !_junction
            ? "No junction cited yet: nothing is chosen, and each blank "
              + "field shows the value that would apply."
            : src.kind === "reuses"
            ? (_extend ? _extend.says.shared : "")
            : src.kind === "deck"
            ? "Values from the run you cited (" + src.name + ").  Change "
              + "any of them; a change applies to every rung that reads it."
            : src.kind === "pair"
            ? "Values from the record of the optimization your structure "
              + "pair carries (" + src.name + ").  Change any of them; a "
              + "change applies to every rung that reads it."
            : "The junction you cited states none of these settings in "
              + "a form this build reads, so it answers none of them.  "
              + "A blank field is not chosen: the template records no "
              + "value for it and prep falls back to the catalogue's "
              + "default until you choose.";
    }

    /* What the shared panel HOLDS now -- every field, a blank as null: the
     * panel edits the template, and is drawn holding the citation's
     * answers, so the server tells the citation's value from the person's
     * by the citation itself, and a field sent blank was emptied
     * (`_panel_template`).  Throws, naming the field, on a value that will
     * not read. */
    function _sharedValues() {
        var fs = root.molbuilder && root.molbuilder.formSchema;
        var host = _$("transport-shared-container");
        if (!fs || !host || !_sharedSchema
                || typeof fs.collectForm !== "function") {
            return {};
        }
        return fs.collectForm(host, _sharedSchema);
    }

    /* ...for a rung's tab to be drawn from.  The panel's own change listener
     * draws the tabs only once it reads, and a citation draws the panel
     * afresh first; {} answers a panel not drawn yet. */
    function _sharedHeld() {
        try { return _sharedValues(); }
        catch (_) { return {}; }
    }

    /* Each rung's tab, drawn again from the template the panel describes. */
    function _renderRungs() {
        var fc = _$("transport-form-container");
        var fs = root.molbuilder && root.molbuilder.formSchema;
        return (fc && fs) ? _fetchAndRender(fc, fs) : Promise.resolve(false);
    }

    /* THE FITTED WINDOW, USED: written into the transmission's own fields,
     * as a value typed there is, and kept with them (`_saveSurface`). */
    function _useWindow(values) {
        var fs = root.molbuilder && root.molbuilder.formSchema;
        var host = _rungHosts.transmission;
        var schema = _rungSchemas.transmission;
        if (!fs || !host || !schema) return;
        fs.setValues(host, schema, values);
        _saveSurface(host, schema, fs, "transmission");
        _schedulePlan();
        _setStatus("The transmission window is the fitted one -- "
                   + values.transmission_emin_ev.toFixed(5) + "\u2026"
                   + values.transmission_emax_ev.toFixed(5) + " eV at "
                   + values.transmission_n_points + " points.");
    }

    /** The bias list as typed: volts, in order. */
    function _biasList() {
        return String((_$("transport-bias") || {}).value || "0.0")
            .split(",").map(function (s) { return s.trim(); })
            .filter(Boolean).map(Number);
    }

    /** The treatment is asked when the list holds several voltages
     *  (`engines/transport.md` § 2a.10). */
    function _syncTreatment() {
        var box = _$("transport-bias-treatment");
        if (box) box.hidden = _biasList().length <= 1;
        _echoBias();
        _schedulePlan();
    }

    /* THE BIAS ON EACH RUNG'S TAB IS THE LIST'S POINTS: the device and the
     * transmission are answered point by point (`template.PER_POINT`), so
     * their locked box shows the list as typed, kept current with it. */
    function _echoBias() {
        var fs = root.molbuilder && root.molbuilder.formSchema;
        if (!fs || typeof fs.showPoints !== "function") return;
        var pts = _biasList();
        fs.showPoints(_$("transport-form-container"), "bias_voltage_v",
            pts.some(isNaN) ? "the bias list, above"
                : pts.join(", ") + " V \u2014 "
                  + (pts.length > 1 ? "one run per point" : "one run"));
    }

    /* THE BUILDER: 0 V first, then start..stop by step, written into the
     * list, which stays editable. */
    function _wireBiasBuilder() {
        var list = _$("transport-bias");
        var fill = _$("transport-bias-fill");
        if (!list || !fill) return;
        list.addEventListener("input", _syncTreatment);
        root.document.querySelectorAll(
            'input[name="transport-bias-treatment"]').forEach(function (b) {
                b.addEventListener("change", _schedulePlan);
            });
        fill.addEventListener("click", function () {
            var a = Number(_$("transport-bias-start").value);
            var b = Number(_$("transport-bias-stop").value);
            var d = Number(_$("transport-bias-step").value);
            if (![a, b, d].every(isFinite) || d === 0
                || (b - a) / d < 0) {
                _refuseBias("The builder needs a start, a stop and a "
                    + "step that walks from one to the other.");
                return;
            }
            var out = [0];
            var n = Math.floor((b - a) / d + 1e-9);
            for (var i = 0; i <= n; i++) {
                var v = Math.round((a + i * d) * 1e6) / 1e6;
                if (out.indexOf(v) === -1) out.push(v);
            }
            list.value = out.map(function (v) {
                return v === 0 ? "0.0" : String(v);
            }).join(", ");
            _syncTreatment();
        });
        _syncTreatment();
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
                    var out = {};
                    if (host && _sharedSchema) {
                        out[_engine()] = [{ host: host,
                                            schema: _sharedSchema }];
                    }
                    return out;
                },
                structure: function () { return _junctionStructure; },
            });
        }
        _fetchAndRender(formContainer, formSchema);
        _fetchAndRenderShared(formSchema);
        _restoreSession();
        _wireJunctionPicker();
        _wireSendButton(formContainer);
        _wireBiasBuilder();
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
