/* /results tab front-end controller (registry-driven dispatch).
 *
 * Mounts whatever the file picker announces, via
 * ``window.molbuilder.inspectors`` (see lib/inspectors/registry.js
 * for the contract).  Nothing in the sidebar reaches this panel;
 * the picker's list is the only source (results.md § 2.1).
 *
 * The dispatch is intentionally tiny: pick + mount + dispose.  All
 * file-type-specific logic lives in the inspector modules under
 * ``static/lib/inspectors/``.  Adding a new file type is one new
 * inspector module + a ``<script>`` tag in results.html; no edit
 * to this file.
 *
 * Mount context: we build it ONCE here and pass it explicitly to
 * registry.mount().  A future /results-style page (e.g. a "compare
 * runs" tab) can build its own context with cached readFile or a
 * custom showError without patching registry.js.
 */
(function () {
    "use strict";

    /* A TRANSPORT POINT AS THE UI NAMES IT -- `Frame 2 of 3 · 0.2 V` -- the
     * one wording the report's table and legend use too
     * (`lib/results/point-words.js`; results.md § 2.5).  A module, loaded
     * once; the ladder is drawn once it is. */
    let _pointWords = null;
    const _wordsReady = import("/static/lib/results/point-words.js")
        .then((m) => { _pointWords = m.pointWords; });

    const $ = (id) => document.getElementById(id);

    const els = {
        host:           null,
        fallback:       null,
        fileReadout:    null,
        kindReadout:    null,
        loadingOverlay: null,   // lock-down cover shown during an async load
    };

    // The currently-mounted inspector's handle, or null when the
    // fallback is showing.  Disposing happens BEFORE the next mount
    // so handlers / timers / observers from the previous inspector
    // can't leak into the new one.
    let currentHandle = null;

    // Inspectors that load ASYNC (fetch + parse + 3D mount) and dispatch
    // ``molbuilder:inspector:ready`` when their first render is on screen.  ONLY
    // these get the loading lock-down: while one loads, the PREVIOUS scene must not
    // stay interactive (the user would mistake it for the new result).  Sync text
    // inspectors (source / markdown) render instantly, so they need no cover.
    const ASYNC_VIEWER_INSPECTORS = { trajectory: 1, structure: 1, spectra: 1 };
    // Safety net: never leave the cover up forever if a ready event never lands.
    let loadingTimer = null;
    const LOADING_TIMEOUT_MS = 15000;

    // The mount context built once at init.  Captures the host
    // element + the standard /api/files/read wrapper.  Inspectors
    // read showError + readFile off this object.
    let mountContext = null;
    //: What is on screen right now -- the comparison the re-announce guard
    //: in `_onSelectionChange` makes.  `currentHandle` alone cannot answer
    //: it: it says something is mounted, not which file it is showing.
    let mountedFile = "";
    let mountedName = "";
    //: The point the report was last opened at by an announcement -- a
    //: re-announce of the same file at the same point (a tab re-entry's
    //: rescan) leaves the point you picked since in place.
    let announcedPoint = "";

    //: The scope last announced, so a caller that does not know it cannot
    //: erase it.  See `_renderStatus`.
    let _scope = null;

    /* THE ROOT'S OWN ANNOUNCEMENT, kept while a run picked off its ladder
     * is on screen (results.md § 2.4): `rootDir` names the calculation the
     * ladder belongs to, `rootSel` is what *back to the calculation*
     * re-announces, `pickedRun` says a run is showing in its place. */
    let rootDir = "";
    let rootSel = null;
    let pickedRun = false;
    let pickedFile = "";        // the file the picked run opened
    let pickSeq = 0;            // the newest pick; a later answer wins

    /* WHAT THE LADDER SAYS under its table -- a pick that failed, or how to
     * reach a transmission point when the report is not on screen.  Held
     * here, not on the element: every announcement redraws the ladder, and
     * the click that asks for the line announces too.  Any announcement
     * clears it; what sets it does so after its own. */
    let ladderSaid = ["", null];      // [the words, their severity]
    let ladderSaidEl = null;
    function _sayOnLadder(text, kind) {
        ladderSaid = [text || "", kind || null];
        if (ladderSaidEl) {
            // THE ONE STATUS WRITER (`lib/status.js`), with its severity.
            window.molbuilder.status.set(ladderSaidEl, ladderSaid[0],
                                         ladderSaid[1]);
            ladderSaidEl.hidden = !ladderSaid[0];
        }
    }

    /* PICK A RUN IN PLACE: ask the run door for that run's folder -- the
     * same answer the picker builds a folder's menu from -- and announce
     * its file with the root's ladder kept, so the Run panel reads that
     * run's record and the ladder stays above it; the sidebar is not
     * moved. */
    function _pickRun(file) {
        const folder = _dirOf(file);
        const mine = ++pickSeq;
        const name = file.split("/").pop();
        fetch("/api/results/dir?path=" + encodeURIComponent(folder),
              { cache: "no-store" })
            .then((r) => r.json())
            .then((body) => {
                if (mine !== pickSeq) return;
                if (!body || body.ok !== true) {
                    _sayOnLadder("Could not open " + name + ": "
                        + ((body && body.error) || "the folder did not answer")
                        + ".", "error");
                    return;
                }
                const hit = (body.files || []).find((f) => f.name === name);
                pickedRun = true;
                pickedFile = file;
                document.dispatchEvent(new CustomEvent(
                    window.molbuilder.constants.EVENT_FILE_SELECTED,
                    { detail: {
                        file: file,
                        meta: hit ? { role: hit.role, label: hit.label,
                                      stage: hit.stage, parser: hit.parser,
                                      engine: body.engine } : null,
                        place: body.place,
                        dir: rootDir, diverged: false,
                        ladder: rootSel ? rootSel.ladder : null,
                        record: body.record, files: body.files,
                        record_dir: folder,
                        force: true,
                        picked_run: true,
                    } }));
            })
            .catch((e) => {
                if (mine !== pickSeq) return;
                _sayOnLadder("Could not open " + name + ": "
                    + (e && e.message ? e.message : e) + ".", "error");
            });
    }

    function _backToRoot() {
        if (!rootSel) return;
        /* FORCED ONLY WHEN A RUN IS SHOWING in the root's place: the root's
         * own file is then a new mount.  With the report already on screen
         * the same-file guard keeps it -- the curves added and the tab
         * chosen stay (results.md § 2.5) -- and a selection lands on it. */
        const wasPicked = pickedRun;
        pickedRun = false;
        pickedFile = "";
        document.dispatchEvent(new CustomEvent(
            window.molbuilder.constants.EVENT_FILE_SELECTED,
            { detail: Object.assign({}, rootSel, { force: wasPicked }) }));
    }

    function _dirOf(path) {
        const p = String(path || "");
        const cut = Math.max(p.lastIndexOf("/"), p.lastIndexOf("\\"));
        return cut > 0 ? p.slice(0, cut) : "";
    }

    function _basename(path) {
        const u = (window.molbuilder || {}).path;
        return u ? u.basename(path) : (path || "");
    }

    /**
     * The header answers WHERE rather than WHAT.
     *
     * The dropdown owns the filename.  This owns the folder, which is the
     * thing that can now differ from where the sidebar is pointing -- and it
     * is the reason the panel is allowed to stop following it.  When the two
     * have parted it says so, and names the gesture that closes the gap.
     */
    function _renderStatus(file, inspectorName, scope) {
        /* THE LAST SCOPE STICKS.  Callers that know nothing about the folder
         * -- `_showFallback`, and `init`'s first paint -- pass none, and the
         * readout keeps the bound folder rather than "No folder selected". */
        if (scope) _scope = scope;
        const sc = _scope || { dir: "", diverged: false };
        if (els.fileReadout) {
            const dir = sc.dir || "";
            let text;
            if (!dir) {
                text = file ? _basename(file) : "No folder selected";
            } else if (!file) {
                text = dir + " / (nothing selected)";
            } else if (_dirOf(file) === dir) {
                text = dir + " / " + _basename(file);
            } else if (sc.point) {
                /* THE FOLDER'S OWN ANSWER, outside it by design: a
                 * transmission run's result is its calculation's report at
                 * its point (results.md § 2). */
                text = dir + "  \u2014 its result is its calculation's "
                     + "report, " + file;
            } else if (pickedRun && dir && file.startsWith(dir + "/")) {
                /* A RUN PICKED OFF THE ROOT'S LADDER (results.md § 2.4):
                 * its folder under the calculation, by design -- not the
                 * divergence the readout below warns about. */
                text = dir + " / " + file.slice(dir.length + 1);
            } else {
                /* THE FILE IS NOT IN THE BOUND FOLDER, so do not write it as
                 * if it were.  `<bound>/<basename>` is a path that does not
                 * exist, and printing it would put
                 * a plausible name over another run's numbers.  It happens
                 * for one round-trip after every Reload, and permanently
                 * when that Reload's scan fails. */
                text = dir + "  \u2014 showing " + file
                     + ", which is not in this folder";
            }
            if (sc.diverged) {
                text += "  \u2014 the sidebar has moved on; Reload to follow";
            }
            els.fileReadout.textContent = text;
            els.fileReadout.classList.toggle(
                "is-diverged",
                !!sc.diverged || (!!file && !!dir && _dirOf(file) !== dir
                                  && !sc.point
                                  && !(pickedRun && file.startsWith(dir + "/"))));
            // the whole line, warning included -- the ellipsis may hide it
            els.fileReadout.title = text;
        }
        if (els.kindReadout) {
            // The kind readout is rendered as an accent pill by
            // results/style.css; no parens needed, the visual
            // treatment is the separator.  Empty -> the :empty CSS
            // rule hides the pill entirely (no stray chip when
            // nothing is mounted).
            els.kindReadout.textContent = inspectorName || "";
        }
    }

    //: WHAT A DIRECTORY IS -> what the empty state should say about it.
    //: `project-layout.md` § 1.4a's vocabulary, and the only copy of it in
    //: the browser: the server answers `place.role` in the same response the
    //: menu is built from, so the page states that answer.  `null` is not
    //: "neither" -- it is *unknown*, and says so.
    const PLACE_NOTE = {
        container:
            "This is a container, not a run \u2014 its results are in the "
            + "run directories below it.",
        run:
            "This is a run directory, but nothing in it is this "
            + "calculation's result yet.",
    };
    const PLACE_NOTE_UNKNOWN =
        "This directory is not marked as part of a calculation, so only "
        + "what is in it can be shown \u2014 nothing about what it belongs to.";

    function _renderPlaceNote(place) {
        const el = els.fallback
            && els.fallback.querySelector(".results-place-note");
        if (!el) return;
        const role = place && place.role;
        el.textContent = role ? (PLACE_NOTE[role] || "") : PLACE_NOTE_UNKNOWN;
        el.hidden = !el.textContent;
    }

    /* THE LADDER of a calculation root (results.md § 2.4; plan.md § 5c.3
     * c-d).  `/api/results/dir` answers a root with `ladder`, the ladder
     * door's own reading, and the empty-state card draws it: one row per
     * rung in ladder order, its state and detail, the rung to resume from
     * named in the title.  The state chip is the one chip
     * (`lib/state-chip.js`) -- one vocabulary of words and tones
     * (`running-a-job.md` § 4.2). */
    /* THE NEXT STEP under the ladder (results.md \u00a7 2.4; job-system.md,
     * *The task*, *The next step*): each group's words, the next one's lines
     * as the terminal prints them -- a hint, read only, saying when it was
     * read, since the panel reads the ladder again only when the tab is
     * shown again or the folder reloaded. */
    function _renderSteps(host, steps) {
        if (!steps.length) return;
        const box = document.createElement("div");
        box.className = "results-ladder-steps";
        const head = document.createElement("p");
        head.className = "results-ladder-steps-head";
        head.textContent = "Each group, and what is next \u2014 read at "
            + new Date().toLocaleTimeString([], { hour: "2-digit",
                                                  minute: "2-digit" })
            + "; returning to this tab, or Reload from current project dir, "
            + "reads it again.";
        box.appendChild(head);
        const list = document.createElement("ul");
        list.className = "results-ladder-steps-list";
        steps.forEach((g) => {
            const li = document.createElement("li");
            if (g.next) li.className = "is-next";
            li.textContent = "group " + g.group + " (" + g.stages.join(", ")
                + ") \u2014 " + g.words + (g.next ? " \u2014 next" : "");
            if (g.next && g.lines.length) {
                const pre = document.createElement("pre");
                pre.className = "results-ladder-steps-line";
                pre.textContent = g.lines.join("\n");
                li.appendChild(pre);
            }
            list.appendChild(li);
        });
        box.appendChild(list);
        host.appendChild(box);
    }

    function _renderLadder(ladder) {
        const host = $("results-ladder");
        if (!host) return;
        host.textContent = "";
        const stages = ladder && Array.isArray(ladder.stages) ? ladder.stages : [];
        if (!stages.length) { host.hidden = true; return; }
        const chip = window.molbuilder.stateChip;
        const title = document.createElement("h3");
        title.className = "results-ladder-title";
        title.textContent = "This calculation's ladder \u2014 " + stages.length
            + " rung" + (stages.length === 1 ? "" : "s")
            + (ladder.complete ? ", every rung finished" : "");
        host.appendChild(title);
        const table = document.createElement("table");
        table.className = "results-ladder-table";
        // A TRANSMISSION ROW'S "open" SHOWS THE REPORT AT ITS POINT -- when
        // the report is what the panel shows; when another file is, this
        // says what to do.
        const said = document.createElement("p");
        said.className = "status";
        ladderSaidEl = said;
        _sayOnLadder(ladderSaid[0], ladderSaid[1]);
        // THE COLUMNS, named -- the state and whether the run converged are
        // two facts (`runstatus.StageStatus.converged`), side by side.
        const head = document.createElement("tr");
        ["#", "stage", "state", "converged", "detail", "result", "files"].forEach((h) => {
            const th = document.createElement("th");
            th.textContent = h;
            head.appendChild(th);
        });
        table.appendChild(head);
        const td = (content) => {
            const c = document.createElement("td");
            if (typeof content === "string") c.textContent = content;
            else c.appendChild(content);
            return c;
        };
        stages.forEach((s, i) => {
            const tr = document.createElement("tr");
            if (s.name === ladder.first_incomplete) tr.className = "is-next";
            tr.appendChild(td(String(s.seq != null ? s.seq : i + 1)));
            tr.appendChild(td(String(s.name)));
            tr.appendChild(td(chip(s.state)));
            tr.appendChild(td(s.converged || "\u2014"));
            // A QUEUED RUNG SAYS ITS STANDING as its scheduler states it
            // here, or where to ask (results.md § 2.4) -- the server's
            // words, beside the detail.
            tr.appendChild(td((s.detail || "")
                + (s.standing && s.standing.say ? " \u00b7 " + s.standing.say : "")
                + (s.tsgf ? " \u00b7 " + s.tsgf : "")));
            tr.appendChild(td(""));
            tr.appendChild(td(""));
            table.appendChild(tr);
            /* EVERY RUN OF THE RUNG, picked in place (results.md \u00a7 2.4):
             * each attempt and, for a swept rung, each attempt's points,
             * with its own state -- the run door's reading -- and the
             * file that run's folder opens.  Picking one shows that
             * result here while the sidebar stays on the calculation. */
            (Array.isArray(s.runs) ? s.runs : []).forEach((r) => {
                const row = document.createElement("tr");
                row.className = "is-run";
                row.appendChild(td(""));
                const words = r.bias_v != null
                    ? _pointWords(r.frame, r.of, r.bias_v) : "";
                const name = r.run + (words ? " \u00b7 " + words : "");
                row.appendChild(td(name));
                row.appendChild(td(chip(r.state)));
                row.appendChild(td(r.converged || "\u2014"));
                row.appendChild(td(r.detail || ""));
                const result = td("");
                if (r.opens && rootDir) {
                    const path = rootDir + "/" + r.opens;
                    /* THE ROOT'S OWN FILE -- a transmission run, whose result
                     * is the report at its point (results.md § 0.1) -- is
                     * shown by re-announcing the calculation with that bias
                     * selected; any other run is picked in place. */
                    const atRoot = r.opens.indexOf("/") === -1;
                    const open = document.createElement("button");
                    open.type = "button";
                    open.className = "results-ladder-open";
                    open.textContent = "open";
                    open.title = atRoot
                        ? (words ? "the report at " + words : "the report")
                        : r.opens + " — shown here; the sidebar "
                          + "stays on the calculation.";
                    // N identical "open" buttons name their run, and a picked
                    // run's says it is showing (ui-contract.md § 4.1); the
                    // report's rows carry no pressed state -- its selected
                    // point is the report's own, shown in its table.
                    open.setAttribute("aria-label", "open " + s.name + " " + name);
                    if (!atRoot) {
                        open.setAttribute("aria-pressed",
                                          pickedRun && pickedFile === path ? "true" : "false");
                    }
                    open.addEventListener("click", () => {
                        if (!atRoot) { _pickRun(path); return; }
                        _backToRoot();
                        if (r.bias_v == null) return;
                        if (currentHandle
                                && typeof currentHandle.selectPoint === "function") {
                            currentHandle.selectPoint({ frame: r.frame,
                                                        bias: Number(r.bias_v) });
                        } else {
                            // THE REPORT IS NOT ON SCREEN -- another file
                            // of the calculation is -- so say how to get there.
                            _sayOnLadder("To see this point, open the report "
                                + "first: choose task.json in the Result "
                                + "file menu above, then click \u201copen\u201d "
                                + "on this row again.", "warn");
                        }
                    });
                    result.appendChild(open);
                }
                row.appendChild(result);
                /* WHAT THE RUN READ AND WHAT WAS CHECKED BEFORE IT WAS
                 * WRITTEN (results.md § 2.4): the run's deck and the
                 * `.validation.txt` prep wrote beside it -- each named by
                 * the server, opened in the sidebar's file viewer, as the
                 * Run panel's View opens a deck (§ 3a). */
                const files = td("");
                const proj = window.molbuilder.projects;
                [["input", r.deck], ["check report", r.validation]]
                    .forEach(([word, rel]) => {
                        if (!rel || !rootDir || !proj
                                || typeof proj.showPreview !== "function") return;
                        const b = document.createElement("button");
                        b.type = "button";
                        b.className = "results-ladder-open";
                        b.textContent = word;
                        b.title = rel + " — opens in the sidebar's file viewer.";
                        b.setAttribute("aria-label", word + " of " + s.name + " " + name);
                        b.addEventListener("click", () => proj.showPreview(rootDir + "/" + rel));
                        files.appendChild(b);
                    });
                row.appendChild(files);
                table.appendChild(row);
            });
        });
        host.appendChild(table);
        _renderSteps(host, Array.isArray(ladder.steps) ? ladder.steps : []);
        host.appendChild(said);
        if (pickedRun) {
            const back = document.createElement("button");
            back.type = "button";
            back.className = "results-ladder-back";
            back.textContent = "\u2190 back to the calculation";
            back.addEventListener("click", _backToRoot);
            host.appendChild(back);
        }
        const note = document.createElement("p");
        note.className = "inspector-card-note";
        // EACH RUNG'S RUN WRITES ITS OWN RESULT in its directory (a
        // vibration's spectrum in its `freq` attempt, engines/vibration.md
        // 5.5); only a calculation that gathers its rungs -- transport's I-V
        // record -- has one here (web/results.md 2.4).
        note.textContent = "Each rung runs in its own directory below this one; "
            + "every run of it is listed under the rung, and its \u201copen\u201d "
            + "shows that run's result here.  A calculation that gathers its "
            + "rungs into one result -- a transport calculation's report -- "
            + "has it below, opened through the calculation's description.";
        host.appendChild(note);
        host.hidden = false;
    }

    function _showFallback(file, place) {
        if (currentHandle) {
            try { currentHandle.dispose(); } catch (_) { /* swallow */ }
            currentHandle = null;
        }
        // Restore the fallback section.  The host owns its contents
        // exclusively while an inspector is mounted, so we re-insert
        // the fallback markup here.  Cheap (small static block).
        els.host.innerHTML = "";
        if (els.fallback) {
            els.host.appendChild(els.fallback);
            _renderPlaceNote(place);
        }
        _renderStatus(file, null);
    }

    // ---- Loading lock-down ------------------------------------------------ //
    // While an async inspector loads, cover the host with an opaque "Loading /
    // parsing…" overlay that BLOCKS interaction (pointer-events).  This kills the
    // confusion the user hit: the previous scene staying live + interactive during
    // the load, so a click/scrub reads as the new result until the view suddenly
    // swaps.  Removed on ``molbuilder:inspector:ready`` (first render on screen) or
    // the safety timer.  The overlay is a SIBLING of the host inside a relative
    // wrapper, so it survives the host's innerHTML churn during the async mount.
    function _showLoading(file) {
        if (!els.loadingOverlay) return;
        const base = (file || "").split("/").pop() || file || "";
        const txt = els.loadingOverlay.querySelector(".results-loading-text");
        if (txt) txt.textContent = base ? ("Loading " + base + " — parsing…") : "Loading…";
        els.loadingOverlay.hidden = false;
        if (loadingTimer) clearTimeout(loadingTimer);
        loadingTimer = setTimeout(_hideLoading, LOADING_TIMEOUT_MS);
    }
    function _hideLoading() {
        if (loadingTimer) { clearTimeout(loadingTimer); loadingTimer = null; }
        if (els.loadingOverlay) els.loadingOverlay.hidden = true;
    }

    /** Put down whatever is on screen, and forget it. */
    function _clearMounted() {
        if (currentHandle) {
            try { currentHandle.dispose(); } catch (_) { /* swallow */ }
            currentHandle = null;
        }
        mountedFile = "";
        mountedName = "";
    }

    function _onSelectionChange(sel) {
        const file  = sel && sel.file ? sel.file : "";
        const meta  = (sel && sel.meta) || null;
        const place = (sel && sel.place) || null;
        const ladder = (sel && sel.ladder) || null;
        const scope = { dir: (sel && sel.dir) || "",
                        diverged: !!(sel && sel.diverged),
                        point: (sel && sel.point) || null };
        _sayOnLadder("");
        /* THE PICKER'S ANNOUNCEMENT OF A ROOT is remembered whole, so a run
         * picked off the ladder can be left again; a picked run's own
         * announcement never replaces it. */
        if (!(sel && sel.picked_run)) {
            ++pickSeq;              // a pick still in flight is overtaken
            pickedRun = false;
            pickedFile = "";
            if (ladder) { rootSel = sel; rootDir = scope.dir; }
            else { rootSel = null; rootDir = ""; }
        }
        // THE LADDER, ABOVE THE PANEL, on every answer: a root's product is
        // shown with it, and anything that is not a root hides it -- drawn
        // once its rows' wording has loaded, in the order announced.
        _wordsReady.then(() => _renderLadder(ladder), (e) => {
            // THE WORDING DID NOT LOAD: said where the ladder stands, never
            // a ladder silently missing -- and nothing where there is none.
            const host = $("results-ladder");
            if (!host || !(ladder && Array.isArray(ladder.stages)
                           && ladder.stages.length)) return;
            host.hidden = false;
            host.textContent = "The ladder could not be drawn: "
                + "lib/results/point-words.js did not load (" + e + ").";
        });
        const reg  = (window.molbuilder || {}).inspectors;
        if (!reg) {
            _hideLoading();
            _clearMounted();
            _showFallback(file, place);
            _renderStatus("", "", scope);
            return;
        }
        /* THE SAME FILE IS NOT A NEW MOUNT.  The picker announces its choice
         * on every scan now (that is what makes the list the one source), and
         * a tab-re-entry scan usually re-announces the file already on screen.
         * Remounting it would throw away what the viewer is holding -- the
         * frame you had scrubbed to, the mode you had selected -- to redraw
         * the same thing.  Only the header is refreshed, because the FOLDER
         * may have started diverging while you were away. */
        if (file && file === mountedFile && currentHandle
                 && !(sel && sel.force)) {
            _renderStatus(file, mountedName, scope);
            _selectPoint(scope.point, true);
            // ...and what is on screen IS the answer: say so, or the picker's
            // "Parsing <file>..." line waits out its fallback timer over a
            // view that never went away.
            try {
                document.dispatchEvent(new CustomEvent(
                    window.molbuilder.constants.EVENT_INSPECTOR_READY,
                    { detail: { inspector: mountedName, unchanged: true } }));
            } catch (_) { /* older headless runners */ }
            return;
        }
        // Dispatch on THE SERVER'S ANSWER when the picker sent one
        // (`{role, parser, engine}` from `/api/results/dir`), not on
        // the filename.  `null` outside the Results flow, where each
        // inspector falls back to its own suffix test.
        const inspector = reg.pick(file, meta);
        if (!inspector) {
            _hideLoading();
            _clearMounted();
            _showFallback(file, place);
            _renderStatus("", "", scope);
            return;
        }
        // Lock the view down BEFORE disposing the old inspector, so there is no
        // window where the stale scene is live + interactive.  Only for async
        // inspectors (the ready event lifts the cover); sync ones render instantly.
        if (ASYNC_VIEWER_INSPECTORS[inspector.name]) _showLoading(file);
        else _hideLoading();
        // Dispose the previous inspector BEFORE handing the host
        // to the next one -- listeners / timers / 3Dmol viewers
        // leak otherwise.
        if (currentHandle) {
            try { currentHandle.dispose(); } catch (_) { /* swallow */ }
            currentHandle = null;
        }
        currentHandle = reg.mount(els.host, file, mountContext, meta);
        mountedFile = file;
        mountedName = inspector.displayName;
        _renderStatus(file, inspector.displayName, scope);
        _selectPoint(scope.point, false);
    }

    /* THE POINT A TRANSMISSION RUN'S FOLDER OPENS ITS REPORT AT (results.md
     * § 2) -- `{frame, of, bias_v}`, the point's wire form -- handed to the
     * report as its ladder row's *open* hands it (§ 2.4).  `again`: the
     * report already on screen, where only another point is a new pick. */
    function _selectPoint(point, again) {
        const key = point ? JSON.stringify(point) : "";
        if (again && key === announcedPoint) return;
        announcedPoint = key;
        if (!point || point.bias_v == null || !currentHandle
                || typeof currentHandle.selectPoint !== "function") return;
        currentHandle.selectPoint({ frame: point.frame,
                                    bias: Number(point.bias_v) });
    }

    function init() {
        els.host        = $("inspector-host");
        els.fallback    = $("results-fallback");
        els.fileReadout = $("results-current-file");
        els.kindReadout = $("results-current-kind");
        if (!els.host) return;   // template invariant broken; bail.

        // Build the loading lock-down overlay: wrap the host in a relative box and
        // add the overlay as a SIBLING, so it survives the host's innerHTML churn
        // during an async mount (the inspector replaces host.innerHTML; a child
        // overlay would be wiped).  Built with createElement, like every node here.
        if (els.host.parentNode && !els.loadingOverlay) {
            const wrap = document.createElement("div");
            wrap.className = "results-inspector-wrap";
            els.host.parentNode.insertBefore(wrap, els.host);
            wrap.appendChild(els.host);
            const overlay = document.createElement("div");
            overlay.className = "results-loading-overlay";
            overlay.setAttribute("role", "status");
            overlay.setAttribute("aria-live", "polite");
            overlay.hidden = true;
            const box = document.createElement("div");
            box.className = "results-loading-box";
            const spin = document.createElement("span");
            spin.className = "spinner";
            spin.setAttribute("aria-hidden", "true");
            const txt = document.createElement("span");
            txt.className = "results-loading-text";
            txt.textContent = "Loading…";
            box.appendChild(spin);
            box.appendChild(txt);
            overlay.appendChild(box);
            wrap.appendChild(overlay);
            els.loadingOverlay = overlay;
        }
        // Lift the lock-down when the async inspector signals its first render is on
        // screen (frame bar populated / plots drawn / viewer non-blank).
        document.addEventListener(
            window.molbuilder.constants.EVENT_INSPECTOR_READY, _hideLoading);

        // Validate the registry is populated before the dispatch
        // wires up.  An empty registry means the inspector module
        // <script> tags failed to load (or failed to self-register
        // -- e.g., a parse error in one inspector silently breaks
        // the chain).  Surface loud + show the fallback so the
        // user gets a clear "nothing's wired up" view instead of
        // a blank panel.
        const reg = (window.molbuilder || {}).inspectors;
        if (!reg || typeof reg.list !== "function" || reg.list().length === 0) {
            console.error(
                "[/results] inspector registry empty at init; "
                + "no inspector modules registered.  Check that "
                + "static/lib/inspectors/*.js script tags loaded "
                + "(network / parse error)."
            );
            // Continue anyway -- the dispatch will route every
            // file to the fallback, which is the right
            // degradation.
        }

        // Build the mount context ONCE (the host is the only
        // closure variable it captures; future inspectors that
        // need it just call ctx.showError / ctx.readFile).
        mountContext = reg && reg.createDefaultContext
            ? reg.createDefaultContext(els.host)
            : null;

        // Detach the fallback from the DOM so the inspector can
        // take exclusive ownership of the host.  _showFallback
        // re-inserts it when needed.
        if (els.fallback && els.fallback.parentNode === els.host) {
            els.host.removeChild(els.fallback);
        }

        const proj = (window.molbuilder || {}).projects;
        if (!proj) {
            // Sidebar didn't initialise; the fallback view is the
            // graceful degradation.
            _showFallback("");
            return;
        }

        // /results listens for an
        // explicit ``molbuilder:results:fileSelected`` event that
        // the dropdown picker dispatches when the user picks a
        // file from #results-file-picker-select (or when the
        // picker auto-picks on first rescan).  Sidebar single-
        // clicks do not steer the inspector.
        document.addEventListener(
            window.molbuilder.constants.EVENT_FILE_SELECTED,
            /* THE DETAIL IS PASSED THROUGH, NOT RE-LISTED.  A
             * hand-kept field list between two halves of one contract is a
             * second shape of the same record; `_onSelectionChange` already
             * guards every field it reads. */
            (evt) => _onSelectionChange((evt && evt.detail) || {})
        );

        /* The header, kept true between selections.  Divergence starts when
         * the sidebar walks off, which is not a selection -- so the panel
         * would otherwise go on claiming it was in step until you next
         * picked something.  Display only: it re-renders the readout around
         * whatever is already mounted and touches nothing else. */
        document.addEventListener(
            window.molbuilder.constants.EVENT_SCOPE_CHANGED,
            (evt) => {
                const d = (evt && evt.detail) || {};
                // the point stays with its folder: a Reload re-binding
                // elsewhere leaves the report outside the new one
                const same = !!_scope && _scope.dir === (d.dir || "");
                _renderStatus(mountedFile, mountedName,
                              { dir: d.dir || "", diverged: !!d.diverged,
                                point: same ? _scope.point : null });
            }
        );

        // Tab-level result-file picker.  Owns its own
        // directory rescan + dropdown population.  Dispatches
        // ``molbuilder:results:fileSelected`` when the user picks
        // (or the auto-pick fires on a fresh dir); the event
        // listener above mounts the right inspector.
        const picker = (window.molbuilder || {}).resultsFilePicker;
        if (picker && typeof picker.mount === "function") {
            picker.mount(document);
        }

        /* A DOUBLE-CLICK SHOWS THE FILE -- and still does not touch this
         * panel.  The interaction model (2026-06-07) says a commit runs the
         * active tab's "use this file" action: Molbuilder loads the structure
         * onto the canvas, spectra loads the file.
         *
         * Its action is the sidebar's own viewer, NOT a mount.  The panel is
         * built from the list and the list is re-read only when you ask
         * (user, 2026-09-19), so a sidebar gesture never mounts anything.
         * Showing the file answers "what is in this one?" without disturbing
         * what you are reading.
         *
         * `showPreview` reads the global pick, and a double-click's FIRST
         * click already set it, so there is nothing to pass. */
        if (proj && typeof proj.onCommit === "function"
                 && typeof proj.showPreview === "function") {
            // The PAYLOAD, not the global pick -- every other commit
            // subscriber uses `sel`, and the pick can be empty while the
            // payload is right (see showPreview's own note).
            proj.onCommit((sel) => proj.showPreview(sel && sel.file));
        }

        /* NOTHING IS MOUNTED FROM OUTSIDE THE LIST (2026-09-19): the picker
         * announces its choice on every scan, including the "we kept what you
         * had" arm, so the wait is one round-trip and what lands is the
         * server's answer rather than a filename guess.
         */
        _showFallback("");
    }

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", init);
    } else {
        init();
    }
})();
