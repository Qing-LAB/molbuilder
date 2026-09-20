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
     * The header, which since 2026-09-19 answers WHERE rather than WHAT.
     *
     * It was the filename alone, and hidden by CSS as "redundant -- the
     * selected option carries the same filename".  Both halves were true and
     * together they were the 2026-08-04 defect: the panel rendered a run from
     * another folder and the option said only `siesta.out`, so every number
     * was plausible and every number was wrong, "with nothing on screen
     * saying so".
     *
     * The dropdown owns the filename.  This owns the folder, which is the
     * thing that can now differ from where the sidebar is pointing -- and it
     * is the reason the panel is allowed to stop following it.  When the two
     * have parted it says so, and names the gesture that closes the gap.
     */
    function _renderStatus(file, inspectorName, scope) {
        /* THE LAST SCOPE STICKS.  Callers that know nothing about the folder
         * -- `_showFallback`, and `init`'s first paint -- used to pass none,
         * and this reset the readout to "No folder selected" over a panel
         * that was bound.  On a page load into a folder whose scan then
         * failed, that was the final state: no folder named anywhere, and an
         * error in the menu. */
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
                 * exist, and printing it is exactly the 2026-08-04 failure --
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
            els.fileReadout.title = file || dir || "";
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
    //: menu is built from, so the page states that answer instead of the one
    //: generic sentence it used to show for every case.  `null` is not
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
        const scope = { dir: (sel && sel.dir) || "",
                        diverged: !!(sel && sel.diverged) };
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
        // overlay would be wiped).  Static markup -> innerHTML is a fixed literal.
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

        // SIDEBAR-DRIVEN dispatch RETIRED 2026-06-09 (task #301).
        // Pre-301: every sidebar pick fired _onSelectionChange on
        // /results, dispose-then-mount-ing an inspector — single-
        // clicking around the sidebar was hijacking the Results
        // tab mid-read.  Post-301, /results listens for an
        // explicit ``molbuilder:results:fileSelected`` event that
        // the dropdown picker dispatches when the user picks a
        // file from #results-file-picker-select (or when the
        // picker auto-picks on first rescan).  Sidebar single-
        // clicks no longer steer the inspector; the user gets
        // back full control of what's mounted.
        document.addEventListener(
            window.molbuilder.constants.EVENT_FILE_SELECTED,
            /* THE DETAIL IS PASSED THROUGH, NOT RE-LISTED.  This copied
             * `file` / `meta` / `place` into a fresh object by hand, so the
             * picker gained `dir` and `diverged` on 2026-09-19 and the header
             * went on showing a bare filename -- the listener was silently
             * dropping the two fields the folder readout is made of.  A
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

        // Tab-level result-file picker (2026-06-01).  Owns its own
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
         * active tab's "use this file" action, and every other tab has one:
         * Molbuilder loads the structure onto the canvas, spectra loads the
         * file.  /results had none, so a double-click here did nothing at
         * all -- the only tab that ignored the gesture.
         *
         * Its action is the sidebar's own viewer, NOT a mount.  The panel is
         * built from the list and the list is re-read only when you ask
         * (user, 2026-09-19), so letting a sidebar gesture mount something
         * would put back exactly the coupling the Reload button replaced.
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

        /* NOTHING IS MOUNTED FROM OUTSIDE THE LIST (2026-09-19).
         *
         * A direct `_onSelectionChange({file: projects.getCurrentFile()})`
         * stood here, to show a remembered file without waiting for the scan.
         * It was a second source for what the panel displays, and it beat the
         * scan to the host every time -- so on the commonest page load the
         * viewer was chosen by FILENAME, with no `meta` and no `place`, which
         * is the guessing the server door replaced.  It could also mount a
         * file this directory does not offer, or one no parser can read.
         *
         * The picker announces its choice on every scan now, including the
         * "we kept what you had" arm that used to stay silent -- which is
         * exactly the case this bootstrap existed to cover.  So the wait is
         * one round-trip, and what lands is the answer rather than a guess.
         */
        _showFallback("");
    }

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", init);
    } else {
        init();
    }
})();
