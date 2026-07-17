/* /results tab front-end controller (registry-driven dispatch).
 *
 * Subscribes to the projects-sidebar selection state and routes
 * the selected file to the matching inspector via
 * ``window.molbuilder.inspectors`` (see lib/inspectors/registry.js
 * for the contract).
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

    function _basename(path) {
        const u = (window.molbuilder || {}).path;
        return u ? u.basename(path) : (path || "");
    }

    function _renderStatus(file, inspectorName) {
        if (els.fileReadout) {
            // Show the bare filename so the eye lands on the data, not
            // the verb.  Full path goes in the title attribute --
            // hover-reveal handles "where exactly is this file" without
            // burning header real estate.
            els.fileReadout.textContent = file
                ? _basename(file)
                : "No file selected";
            els.fileReadout.title = file || "";
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

    function _showFallback(file) {
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

    function _onSelectionChange(sel) {
        const file = sel && sel.file ? sel.file : "";
        const reg  = (window.molbuilder || {}).inspectors;
        if (!reg) {
            _hideLoading();
            _showFallback(file);
            return;
        }
        const inspector = reg.pick(file);
        if (!inspector) {
            _hideLoading();
            _showFallback(file);
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
        currentHandle = reg.mount(els.host, file, mountContext);
        _renderStatus(file, inspector.displayName);
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
            (evt) => _onSelectionChange({
                file: (evt && evt.detail && evt.detail.file) || "",
            })
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

        // Initial bootstrap: with the onChange subscription
        // retired, the inspector needs a manual first render so
        // returning users with a sessionStorage-saved current
        // file see their content without first having to click
        // through the dropdown.  The picker's _forceRescan in
        // its mount call ALSO emits a fileSelected event when it
        // auto-picks; that path covers the "fresh dir with no
        // sessionStorage pick" case.  This direct call covers the
        // complementary "sessionStorage has a file already".
        const initialFile = (typeof proj.getCurrentFile === "function")
            ? proj.getCurrentFile() : "";
        if (initialFile) {
            _onSelectionChange({ file: initialFile });
        } else {
            _showFallback("");
        }
    }

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", init);
    } else {
        init();
    }
})();
