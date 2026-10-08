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

    //: The scope last announced, so a caller that does not know it cannot
    //: erase it.  See `_renderStatus`.
    let _scope = null;

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
                !!sc.diverged || (!!file && !!dir && _dirOf(file) !== dir));
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
     * named in the title.  The state chip is the bench summary's
     * (`inspectors.stateChip`) -- one vocabulary of words and tones
     * (`running-a-job.md` § 4.2); a third copy is what § 5c.3 forbids. */
    function _renderLadder(ladder) {
        const host = els.fallback
            && els.fallback.querySelector(".results-ladder");
        if (!host) return;
        host.textContent = "";
        const stages = ladder && Array.isArray(ladder.stages) ? ladder.stages : [];
        if (!stages.length) { host.hidden = true; return; }
        const reg = (window.molbuilder || {}).inspectors || {};
        const chip = (state) => {
            if (typeof reg.stateChip === "function") return reg.stateChip(state);
            const s = document.createElement("span");
            s.textContent = String(state);
            return s;
        };
        const title = document.createElement("h3");
        title.className = "results-ladder-title";
        title.textContent = "This calculation's ladder \u2014 " + stages.length
            + " rung" + (stages.length === 1 ? "" : "s")
            + (ladder.complete ? ", every rung finished"
               : (ladder.first_incomplete
                  ? ", next to run: " + ladder.first_incomplete.replace("_", " ")
                  : ""));
        host.appendChild(title);
        const table = document.createElement("table");
        table.className = "results-ladder-table";
        stages.forEach((s, i) => {
            const tr = document.createElement("tr");
            if (s.name === ladder.first_incomplete) tr.className = "is-next";
            const td = (content) => {
                const c = document.createElement("td");
                if (typeof content === "string") c.textContent = content;
                else c.appendChild(content);
                return c;
            };
            tr.appendChild(td(String(s.seq != null ? s.seq : i + 1)));
            tr.appendChild(td(String(s.name).replace("_", " ")));
            tr.appendChild(td(chip(s.state)));
            tr.appendChild(td(s.detail || ""));
            table.appendChild(tr);
        });
        host.appendChild(table);
        const note = document.createElement("p");
        note.className = "inspector-card-note";
        // EACH RUNG'S RUN WRITES ITS OWN RESULT in its directory (a
        // vibration's spectrum in its `freq` attempt, engines/vibration.md
        // 5.5); only a calculation that gathers its rungs -- transport's I-V
        // record -- has one here (web/results.md 2.4).
        note.textContent = "Each rung runs in its own directory below this one; "
            + "open a rung in the sidebar to read its files and its result.  "
            + "A calculation that gathers its rungs into one result -- a "
            + "transport calculation's I\u2013V record -- has it here, and "
            + "this panel opens it.";
        host.appendChild(note);
        host.hidden = false;
    }

    function _showFallback(file, place, ladder) {
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
            _renderLadder(ladder || null);
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
                        diverged: !!(sel && sel.diverged) };
        const reg  = (window.molbuilder || {}).inspectors;
        if (!reg) {
            _hideLoading();
            _clearMounted();
            _showFallback(file, place, ladder);
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
            _showFallback(file, place, ladder);
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
                _renderStatus(mountedFile, mountedName,
                              { dir: d.dir || "", diverged: !!d.diverged });
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
