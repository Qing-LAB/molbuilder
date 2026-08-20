/* /results tab-level file picker.
 *
 * Promoted from the in-trajectory ``lib/trajectory/result-list.js``
 * (2026-06-01).  Pre-refactor: the dropdown lived inside the
 * trajectory inspector and only listed SIESTA .out / .molwatch.log
 * files in the current directory.  Post-refactor: the dropdown sits
 * at the results-tab header (#results-file-picker-bar) and lists
 * every file in the current directory that any ``isResult: true``
 * inspector would mount -- trajectory + spectra + structure today,
 * any future result inspector tomorrow.
 *
 * Mount target: ``#results-file-picker-bar`` in
 * ``templates/results.html``.  The bar is pre-shown during the scan
 * itself (with the dropdown empty and the meta line showing
 * "Scanning for output files…") so the user sees an immediate
 * acknowledgement; it stays hidden only when the directory has
 * zero result files (the inspector then renders its fallback).
 *
 * Picking a different entry from the <select> mirrors the choice
 * to the sidebar via ``projects.setShared(dir, newFile)`` (so the
 * sidebar's "current file" highlight matches) AND dispatches a
 * ``molbuilder:results:fileSelected`` custom event on ``document``.
 * results/viewer.js listens for that event and disposes + remounts
 * the matching inspector.  Pre-task-#301 (2026-06-09) the dispatch
 * was implicit via the sidebar's ``onChange`` subscriber; that
 * subscription was retired so a stray sidebar single-click on
 * /results no longer hijacks the inspector mid-read.  The single
 * source of truth for "which file is mounted" is the dropdown's
 * selectedOption + the most-recent fileSelected event.
 *
 * One scan, one choice, one announcement (results.md § 2.2): a scan
 * keeps the file already showing if this directory still offers it and
 * otherwise takes the newest, then labels the menu with that one value
 * AND announces it.  Labelling without announcing -- or the reverse --
 * is how the menu came to name one file while displaying another.
 *
 * Status line: the meta row doubles as a status surface.  Shows
 * "Scanning for output files…" during the directory listing fetch,
 * then "Parsing <basename>…" once a file selection lands, then the
 * steady-state "X of N · 5m ago" once the inspector dispatches
 * ``molbuilder:inspector:ready`` on ``document`` (or after the
 * ``PARSE_TIMEOUT_MS`` fallback fires).
 *
 * Visible API on ``window.molbuilder.resultsFilePicker``:
 *
 *   ``mount(root)``    -- one-shot init: attaches the <select>
 *                         handler + Refresh-button click handler,
 *                         subscribes to ``molbuilder:inspector:ready``
 *                         + pageshow / visibilitychange (for tab
 *                         re-entry auto-refresh), and triggers the
 *                         first scan.  Returns a disposer that
 *                         aborts any in-flight fetch and detaches
 *                         subscriptions.  Subscribes to the sidebar's
 *                         onChange for DIRECTORY changes only -- a file
 *                         pick there still changes nothing here, which
 *                         is what #301 protected (results.md § 2.1).
 *   ``parseDir(file)``            -- pure helper (exported for testing).
 *   ``formatRelativeTime(epoch)`` -- pure helper (testing).
 *   ``filterToResultFiles(entries, dir, pickResult)`` -- pure helper.
 *   ``absorbSatellites(entries, pickResult)`` -- pure helper that drops
 *                         the files a run's master subsumes, so one run is
 *                         one entry (results.md § 2.3).
 *   ``groupResultFiles(entries, pickResult)`` -- pure helper that
 *                         buckets pre-filtered entries by
 *                         inspector ``resultCategory(file)`` into
 *                         ``[{label, entries}]`` for ``<optgroup>``
 *                         rendering (testing).
 *
 * Idempotency note: the picker reacts to onChange events; calling
 * mount() twice on the same root would double-subscribe.  Tests
 * MUST dispose the first handle before remounting.
 */
(function (root) {
    "use strict";

    // -------- pure helpers (exported for testing) --------------- //

    /**
     * Split a file path into ``{dir, name}`` parts WITHOUT importing
     * the global path utility -- the picker needs to be loadable in
     * test contexts without a full ``molbuilder.path`` mount.
     */
    function parseDir(file) {
        if (!file) return { dir: "", name: "" };
        const ix = Math.max(
            file.lastIndexOf("/"),
            file.lastIndexOf("\\")
        );
        if (ix < 0) return { dir: "", name: file };
        return {
            dir:  file.slice(0, ix),
            name: file.slice(ix + 1),
        };
    }

    /**
     * Format an epoch_seconds value as a relative-time string for
     * the dropdown's ``(2m ago)`` / ``(3h ago)`` suffix.  Returns ""
     * when the epoch is missing.
     */
    function formatRelativeTime(epoch_s) {
        if (epoch_s == null || !Number.isFinite(epoch_s)) return "";
        const now = Date.now() / 1000;
        const delta = Math.max(0, now - epoch_s);
        if (delta < 60)    return Math.floor(delta) + "s ago";
        if (delta < 3600)  return Math.floor(delta / 60) + "m ago";
        if (delta < 86400) return Math.floor(delta / 3600) + "h ago";
        const days = Math.floor(delta / 86400);
        if (days < 30)     return days + "d ago";
        return new Date(epoch_s * 1000).toLocaleDateString();
    }

    /**
     * Given the ``/api/files/list`` response entries (each
     * ``{name, kind, size, mtime}``) and the inspector registry's
     * ``pickResult`` function, return the subset of file entries that
     * any result inspector would mount.  Sorted by mtime descending
     * (newest first); secondary sort by name for deterministic
     * tie-break on filesystems with low mtime precision.
     *
     * Pure (no DOM, no fetch).  Exported for unit tests.
     */
    function filterToResultFiles(entries, dirPath, pickResult) {
        if (!Array.isArray(entries) || typeof pickResult !== "function") {
            return [];
        }
        // Use a forward-slash separator regardless of OS.  The backend
        // returns native path separators in ``/api/files/list``'s
        // ``path`` field but file matching is purely on the basename
        // suffix so this doesn't affect correctness.
        const sep = dirPath && dirPath.indexOf("\\") >= 0 ? "\\" : "/";
        const out = [];
        for (const entry of entries) {
            if (!entry || entry.kind !== "file") continue;
            const fullPath = dirPath ? (dirPath + sep + entry.name) : entry.name;
            if (!pickResult(fullPath)) continue;
            out.push({
                name:  entry.name,
                path:  fullPath,
                mtime: entry.mtime,
                size:  entry.size,
            });
        }
        // Newest first; tie-break by name so the order is deterministic
        // across runs even when the filesystem reports mtime with
        // 1-second resolution.
        out.sort((a, b) => {
            const da = (a.mtime != null) ? a.mtime : -Infinity;
            const db = (b.mtime != null) ? b.mtime : -Infinity;
            if (db !== da) return db - da;
            return a.name.localeCompare(b.name);
        });
        return out;
    }

    /**
     * Drop the entries that another listed entry subsumes, so a run shows up
     * as ONE result rather than as its pile of working files
     * (results.md § 2.3).
     *
     * A presenter declares this with an optional ``absorbs(master, other)``;
     * the engine naming lives there, with the code that knows it.  Absorption
     * only applies between files that are BOTH in this listing, so a run whose
     * master was deleted or never written still lists its parts and nothing
     * becomes unreachable.
     *
     * An absorbed file cannot absorb anything itself -- that keeps a mutual
     * claim (a presenter bug) from removing both sides and losing the run
     * entirely.  Order is the caller's newest-first list, so which side of a
     * mutual claim survives is deterministic.
     *
     * Pure (no DOM, no fetch).  Exported for unit tests.
     */
    function absorbSatellites(entries, pickResult) {
        if (!Array.isArray(entries) || typeof pickResult !== "function") {
            return Array.isArray(entries) ? entries : [];
        }
        const absorbed = new Set();
        for (const master of entries) {
            if (absorbed.has(master.path)) continue;
            let insp;
            try { insp = pickResult(master.path); } catch (e) { insp = null; }
            if (!insp || typeof insp.absorbs !== "function") continue;
            for (const other of entries) {
                if (other === master || absorbed.has(other.path)) continue;
                let claims = false;
                try {
                    claims = !!insp.absorbs(master.path, other.path);
                } catch (e) {
                    console.warn(
                        "[results-file-picker] absorbs() threw for "
                        + insp.name + ":", e);
                }
                if (claims) absorbed.add(other.path);
            }
        }
        return absorbed.size
            ? entries.filter(e => !absorbed.has(e.path))
            : entries;
    }

    /**
     * Build the <option> label text.  The filename is the primary
     * read; the relative timestamp lives as a parenthesised tail.
     * Keep the prefix short so long base names don't push the
     * timestamp off-screen.
     */
    function labelForResult(entry) {
        const rel = formatRelativeTime(entry.mtime);
        return rel ? (entry.name + " (" + rel + ")") : entry.name;
    }

    /**
     * Bucket the filtered result entries by inspector
     * ``resultCategory(path)`` and return a sorted ``[{label,
     * entries}]`` array suitable for rendering as a sequence of
     * ``<optgroup>`` blocks.
     *
     *   * Inside each group, entries stay in the input order
     *     (callers feed pre-sorted-by-mtime input from
     *     ``filterToResultFiles``, so newest is first within
     *     each group).
     *   * Groups are sorted by their NEWEST entry's mtime
     *     descending, so the group containing the most recent
     *     result floats to the top of the dropdown -- usually
     *     what the user just ran.
     *   * If an inspector doesn't expose ``resultCategory``, we
     *     fall back to its ``displayName`` (a sensible default
     *     for single-purpose inspectors).
     *
     * Pure (no DOM).  Exported for unit tests.
     */
    function groupResultFiles(entries, pickResult) {
        if (!Array.isArray(entries) || typeof pickResult !== "function") {
            return [];
        }
        // Map insertion order matches first-occurrence order in
        // ``entries``; sort step below drops back to mtime order so
        // input order doesn't leak through.
        const buckets = new Map();
        for (const entry of entries) {
            const inspector = pickResult(entry.path);
            if (!inspector) continue;  // already filtered, but defensive.
            let label;
            try {
                label = (typeof inspector.resultCategory === "function")
                    ? inspector.resultCategory(entry.path)
                    : inspector.displayName;
            } catch (e) {
                console.warn(
                    "[results-file-picker] resultCategory() threw for "
                    + inspector.name + ":", e);
                label = inspector.displayName;
            }
            if (typeof label !== "string" || !label) {
                label = inspector.displayName || "Other";
            }
            if (!buckets.has(label)) buckets.set(label, []);
            buckets.get(label).push(entry);
        }
        // Group sort key = newest entry's mtime in that group.
        // Negative-infinity fallback for groups whose entries all
        // had null mtime so they land at the bottom deterministically.
        const groups = [];
        for (const [label, list] of buckets) {
            let newest = -Infinity;
            for (const e of list) {
                if (e.mtime != null && e.mtime > newest) newest = e.mtime;
            }
            groups.push({ label, entries: list, _newest: newest });
        }
        groups.sort((a, b) => {
            if (b._newest !== a._newest) return b._newest - a._newest;
            return a.label.localeCompare(b.label);
        });
        // Strip the private sort key before returning.
        return groups.map(g => ({ label: g.label, entries: g.entries }));
    }

    /**
     * Populate the <select> with one ``<optgroup>`` per inspector
     * category, each wrapping one ``<option>`` per result.  Mark
     * the current selection.  Defensive clear via firstChild loop
     * (not ``innerHTML = ""``) to avoid XSS reflection if a
     * filename ever contains tag chars.
     *
     * ``groups`` is the output of ``groupResultFiles``; if it's
     * empty (no results) the dropdown is cleared but stays in the
     * DOM.  Callers gate visibility by toggling ``barEl.hidden``.
     */
    /**
     * Fill the dropdown with a single non-selectable placeholder
     * option whose text explains the current empty state to the
     * user.  Called from the four no-results / error paths so the
     * picker bar stays visible (Refresh button reachable) instead
     * of disappearing entirely.
     */
    function _populatePlaceholder(selectEl, placeholderText) {
        while (selectEl.firstChild) {
            selectEl.removeChild(selectEl.firstChild);
        }
        const opt = document.createElement("option");
        opt.value = "";
        opt.textContent = placeholderText;
        opt.disabled = true;
        opt.selected = true;
        selectEl.appendChild(opt);
    }

    function _populate(selectEl, groups, currentPath) {
        while (selectEl.firstChild) {
            selectEl.removeChild(selectEl.firstChild);
        }
        for (const group of groups) {
            const og = document.createElement("optgroup");
            og.label = group.label;
            for (const r of group.entries) {
                const opt = document.createElement("option");
                opt.value = r.path;
                opt.textContent = labelForResult(r);
                opt.title = r.name;
                if (r.path === currentPath) {
                    opt.selected = true;
                }
                og.appendChild(opt);
            }
            selectEl.appendChild(og);
        }
    }

    /**
     * Render the "N of M, last updated X ago" meta line under the
     * dropdown.  Pure derivation from the results list + current
     * file path.
     */
    function _renderMeta(metaEl, results, currentPath) {
        if (!metaEl) return;
        const ix = results.findIndex(r => r.path === currentPath);
        if (ix < 0) {
            metaEl.textContent = results.length + " file" +
                (results.length === 1 ? "" : "s");
            return;
        }
        // The list is sorted newest-first; the user-facing "X of N"
        // counts from oldest to newest so a freshly-created result
        // shows as the highest index.
        const chronIx = results.length - ix;
        const cur = results[ix];
        const rel = formatRelativeTime(cur.mtime);
        const parts = [chronIx + " of " + results.length];
        if (rel) parts.push(rel);
        metaEl.textContent = parts.join(" · ");
    }

    // -------- main mount ---------------------------------------- //

    /**
     * One-shot init.  Wires the picker to the projects sidebar so it
     * reacts to directory changes + file selection changes.  Call
     * once at /results page load.
     *
     * ``rootEl`` is the document or any ancestor of
     * #results-file-picker-bar; defaults to ``document`` so the
     * standard /results page just calls ``mount()`` with no args.
     */
    function mount(rootEl) {
        rootEl = rootEl || document;
        const barEl  = rootEl.querySelector("#results-file-picker-bar");
        const selEl  = rootEl.querySelector("#results-file-picker-select");
        const metaEl = rootEl.querySelector("#results-file-picker-meta");
        if (!barEl || !selEl) {
            // Template invariant broken; bail silently.  An empty
            // dispose is safer than throwing -- /results still works
            // without the picker.
            return { dispose() { /* nothing */ } };
        }

        const proj      = (root.molbuilder || {}).projects;
        const inspReg   = (root.molbuilder || {}).inspectors;
        // Shared string-constant namespace.  Centralised in
        // lib/constants.js so the picker's event + storage key
        // strings can't drift from the dispatcher's.
        const C         = (root.molbuilder || {}).constants;
        if (!proj || typeof proj.onChange !== "function"
            || !inspReg || typeof inspReg.pickResult !== "function") {
            // The picker depends on both the sidebar AND the inspector
            // registry.  Either missing means /results has a deeper
            // load-order problem; the fallback is to hide the picker
            // entirely and let the rest of the page work.
            console.warn(
                "[results-file-picker] projects sidebar or inspector "
                + "registry unavailable; picker disabled"
            );
            barEl.hidden = true;
            return { dispose() { /* nothing */ } };
        }

        // -- per-scan state.  ``aborter`` is replaced on each scan
        //    so an in-flight stale request is aborted when a new
        //    directory selection lands. -------------------------- //
        let aborter         = null;
        let lastScannedDir  = null;
        let cachedResults   = [];  // last successful scan -- flat, newest first
        let cachedGroups    = [];  // same data bucketed via groupResultFiles;
                                   // ``_populate`` consumes this so we don't
                                   // re-group on every same-dir selection swap.
        let disposed        = false;
        // Transient parse-status timeout.  Set when a file selection
        // kicks off an inspector mount; cleared by the next selection
        // change, a fresh scan, the inspector's "ready" event (see
        // ``molbuilder:inspector:ready`` below), or a fallback timer
        // that reverts the meta line to the steady-state "N of M · X
        // ago" readout.
        let parseTimer      = null;
        // The file the parse status is currently FOR -- the inspector
        // ready handler clears the status only if it still matches
        // (a stale ready event from a previous file's mount can't
        // pull the status off a new file's parse).
        let parsingFor      = null;
        // Fallback "if no inspector ever signals ready, drop the
        // label after this long".  Longer than the real render
        // typically takes so the "Parsing…" label hands off
        // naturally; tuned for SIESTA .out files in the 5-10 MB
        // range which are the user's primary case.
        const PARSE_TIMEOUT_MS = 8000;

        function _abortInFlight() {
            if (aborter) {
                try { aborter.abort(); } catch (_) { /* ignore */ }
                aborter = null;
            }
        }

        // ---- transient-status helpers ------------------------- //
        // The picker's meta line doubles as a status surface: it
        // shows "Scanning for output files…" while a directory
        // listing is in flight, "Parsing <basename>…" while the
        // newly-mounted inspector loads its first frame, and the
        // steady-state "N of M · X ago" readout otherwise.
        //
        // Mutates ``metaEl`` directly so the picker's existing
        // _renderMeta path stays a pure derivation -- this keeps
        // unit tests of the pure helpers unchanged.

        function _showTransientStatus(message) {
            if (!metaEl) return;
            metaEl.textContent = message;
            // Class lets CSS distinguish transient status from the
            // steady-state meta (e.g. italicise + softer colour).
            metaEl.classList.add("is-busy");
        }

        function _showIdleMeta(file) {
            if (!metaEl) return;
            metaEl.classList.remove("is-busy");
            _renderMeta(metaEl, cachedResults, file);
        }

        function _clearParseTimer() {
            if (parseTimer !== null) {
                clearTimeout(parseTimer);
                parseTimer = null;
            }
            parsingFor = null;
        }

        function _startParseStatus(file) {
            _clearParseTimer();
            if (!file) return;
            const basename = parseDir(file).name || file;
            parsingFor = file;
            _showTransientStatus("Parsing " + basename + "…");
            parseTimer = setTimeout(() => {
                parseTimer = null;
                if (disposed) return;
                _showIdleMeta(file);
                parsingFor = null;
            }, PARSE_TIMEOUT_MS);
        }

        // The trajectory + spectra inspectors dispatch
        // ``molbuilder:inspector:ready`` on ``document`` once their
        // first render is on screen (frame counter populated, plots
        // drawn, viewer non-blank).  When that lands, drop the
        // "Parsing…" label early -- the user no longer needs the
        // acknowledgement.
        //
        // Guarded by ``parsingFor`` so a stale ready event from a
        // prior load can't clear a fresh parse status (rapid file-
        // switching case: A starts loading, user switches to B,
        // A's deferred ready fires; we want it to be a no-op).
        function _onInspectorReady(_evt) {
            if (disposed) return;
            if (parsingFor === null) return;
            const file = parsingFor;
            _clearParseTimer();
            parsingFor = null;
            _showIdleMeta(file);
        }

        /**
         * Scan ``dir`` via /api/files/list, filter via
         * registry.pickResult, populate the dropdown.  Async; safe
         * to call concurrently (older calls are aborted).
         */
        function _scan(dir, currentFile) {
            _abortInFlight();
            _clearParseTimer();
            if (!dir) {
                // No directory selected (sidebar cleared).  Show
                // the picker bar in its placeholder state so the
                // user can still see the Refresh button -- they
                // may have just opened the page before navigating
                // the sidebar.  Empty dropdown + idle status.
                _populatePlaceholder(selEl,
                    "(navigate the Projects sidebar to a directory)");
                cachedResults = [];
                cachedGroups  = [];
                _showIdleMeta(null);
                return;
            }
            // Pre-show the picker bar with an empty dropdown + a
            // "Scanning…" status so the user sees that something is
            // happening DURING the fetch -- not just after it
            // resolves.  If the dir has no result files, the fetch's
            // resolver hides the bar again.
            while (selEl.firstChild) selEl.removeChild(selEl.firstChild);
            barEl.hidden = false;
            _showTransientStatus("Scanning for output files…");

            aborter = new AbortController();
            const signal = aborter.signal;
            // List through the sidebar file layer (projects.listDir -> /api/files/list)
            // rather than a hand-rolled fetch -- the ONE file-access path.  ``apiList``
            // already sends ``cache:no-store``, which is load-bearing: it fixes the "click
            // Results, see stale dropdown" bug (an identical /api/files/list URL would
            // otherwise serve the cached prior scan, hiding newly-generated result files
            // until a sidebar out+back).  ``signal`` aborts a superseded scan.
            const _proj = (window.molbuilder || {}).projects;
            (_proj && typeof _proj.listDir === "function"
                ? _proj.listDir(dir, { signal })
                : Promise.resolve({ ok: false,
                    error: "projects.listDir unavailable" }))
                .then(body => {
                    if (disposed || signal.aborted) return;
                    if (!body || body.ok !== true) {
                        // Fetch failed (file listing API errored).
                        // Keep the bar visible so Refresh is still
                        // clickable -- a transient error shouldn't
                        // strand the user; one Refresh click retries.
                        if (metaEl) metaEl.classList.remove("is-busy");
                        _populatePlaceholder(selEl,
                            "(directory listing failed — click Refresh)");
                        cachedResults = [];
                        cachedGroups  = [];
                        _showIdleMeta(null);
                        return;
                    }
                    // Filter to result-class files, then let a run's master
                    // absorb its working files so the menu lists the RUN, not
                    // its parts (results.md § 2.3).
                    const results = absorbSatellites(
                        filterToResultFiles(
                            body.entries || [],
                            body.path     || dir,
                            inspReg.pickResult
                        ),
                        inspReg.pickResult
                    );
                    cachedResults = results;
                    if (results.length === 0) {
                        // Empty result set.  Bar stays VISIBLE with
                        // an empty-state placeholder so the user
                        // can hit Refresh when their first job
                        // emits its first output file -- they
                        // shouldn't have to re-click the Results
                        // tab to recover the button.
                        if (metaEl) metaEl.classList.remove("is-busy");
                        cachedGroups = [];
                        _populatePlaceholder(selEl,
                            "(no result files yet — click Refresh)");
                        _showIdleMeta(null);
                        /* AND SAY SO, instead of tidying our own bar and
                         * leaving.  This used to just return: the menu showed
                         * "no result files yet" while the PREVIOUS folder's
                         * run stayed mounted below it -- plots, run badge,
                         * convergence targets and 3-D structure, all from a
                         * directory the user had left.  Announcing the empty
                         * selection is what makes the dispatcher dispose the
                         * old inspector and restore the empty state
                         * (results/viewer.js::_showFallback).  results.md
                         * § 2.2: a scan that changes what is current always
                         * announces it. */
                        _emitFileSelected("");
                        return;
                    }
                    cachedGroups = groupResultFiles(
                        results, inspReg.pickResult);
                    barEl.hidden = false;
                    /* ONE CHOICE, then label AND announce it (results.md
                     * § 2.2).  Keep what we were already showing if this
                     * directory still offers it; otherwise take the newest.
                     *
                     * WHY THIS IS ONE VALUE.  It used to be two.  `_populate`
                     * labelled the menu from ``cachedGroups`` -- grouped, ties
                     * broken by CATEGORY LABEL -- while the auto-pick mounted
                     * ``results[0]`` from the flat list, ties broken by FILE
                     * NAME.  Both orderings are correct; having two is the
                     * defect.  They agree until several results share an
                     * mtime, which is precisely what a job produces when it
                     * finishes and flushes its outputs in the same second: a
                     * pySCF run with four files stamped 10:31:08 labelled the
                     * menu `…molwatch.log` and displayed `…_optimized.xyz`
                     * (2026-08-04).  The chosen path now feeds both, so they
                     * cannot disagree. */
                    const keepCurrent =
                        currentFile && results.some(r => r.path === currentFile);
                    const chosen = keepCurrent ? currentFile : results[0].path;
                    _populate(selEl, cachedGroups, chosen);
                    if (keepCurrent) {
                        // Already mounted; just acknowledge the re-entry.
                        _startParseStatus(chosen);
                    } else {
                        _adoptSelection(chosen);
                    }
                })
                .catch(err => {
                    if (err && err.name === "AbortError") return;
                    console.warn(
                        "[results-file-picker] scan failed; "
                        + "showing placeholder",
                        err
                    );
                    if (metaEl) metaEl.classList.remove("is-busy");
                    _populatePlaceholder(selEl,
                        "(scan failed — click Refresh to retry)");
                    cachedResults = [];
                    cachedGroups  = [];
                    _showIdleMeta(null);
                });
        }

        /**
         * Make ``path`` the current file: mirror it to the sidebar via
         * ``projects.setShared`` and announce it with
         * ``molbuilder:results:fileSelected`` so the /results dispatcher
         * mounts the matching inspector.  No-op if the sidebar is locked
         * (setShared returns ok:false; we log + don't retry).
         *
         * The CALLER labels the menu with the same path first -- announcing
         * and labelling are two uses of one chosen value (results.md § 2.2),
         * not two derivations.  This used to rely on setShared's onChange
         * coming back round to relabel the menu, which stopped happening when
         * that subscription was retired (#301) and left the label behind.
         */
        function _adoptSelection(path) {
            if (!path) return;
            const parts = parseDir(path);
            const r = proj.setShared(parts.dir, path);
            if (r && r.ok === false) {
                console.warn(
                    "[results-file-picker] selection refused:",
                    r.error
                );
                return;
            }
            _emitFileSelected(path);
        }

        /**
         * Re-scope the menu to ``dir`` -- the ONE path that changes which
         * directory the picker is listing.
         *
         * ``preferredFile`` is kept if this directory still offers it, so a
         * Refresh does not jump you to a different result; pass "" to take the
         * newest, which is what a folder change does (results.md § 2.1).
         *
         * This replaced ``_onSelectionChange``, which branched on "same dir or
         * not".  Its same-dir half had been unreachable since #301 retired the
         * subscription that fed it: the only remaining caller, _forceRescan,
         * set ``lastScannedDir = null`` immediately before calling, so the
         * dir-changed branch was always taken -- as the comment there admitted
         * ("we deliberately bypass the same-dir branch").  ~30 lines of
         * file-swap handling, and the ``lastSelectedFile`` that branch was the
         * only reader of, were dead weight held up by a function shape that no
         * longer had two cases.
         */
        function _rescanDir(dir, preferredFile) {
            lastScannedDir = dir;
            _scan(dir, preferredFile || "");
        }

        // Sidebar onChange subscription RETIRED 2026-06-09 (task
        // #301).  Pre-task-301 the picker re-scanned on every
        // sidebar pick — single-clicking around the sidebar would
        // hijack the Results inspector mid-read.  Post-301, the
        // picker treats its <select> as the single source of truth
        // for the active result file; the sidebar's role on
        // /results is "set the project directory" only.  Re-scans
        // happen on pageshow / visibilitychange (tab re-entry,
        // bfcache restore) and on the explicit Refresh button.
        /* ...but the DIRECTORY half of it has to come back.
         *
         * THE CONTRACT IS results.md § 2.1, and it is the whole of the rule:
         * the sidebar sets the SCOPE (which folder), the dropdown decides
         * WHAT IS SHOWN within it.  So a folder change re-scopes this menu and
         * auto-picks the newest; a file click -- single OR double -- changes
         * nothing here, which is exactly what #301 protected and stays true.
         *
         * Why the directory half is not optional: a menu that dictates the
         * display has to be listing the folder the user is actually in.  This
         * scoped itself once, at mount, and never again -- so moving the
         * sidebar to another run folder left it enumerating the PREVIOUS
         * folder's files, and the four plots, the run badge, the convergence
         * card and the 3-D structure all went on rendering a different run
         * with nothing on screen saying so.  Seen 2026-08-04: a LIVE
         * BDT-Au111 job displayed as a finished BDT run from another
         * directory, every number plausible and every number wrong.
         *
         * The first fire is skipped deliberately.  ``onChange`` fires once
         * immediately on subscribe (projects/state.js), and the initial scan
         * is already owned by the pageshow/visibilitychange path below --
         * scanning here too would list the same directory twice on every
         * page load. */
        let sawInitialSelectionFire = false;
        const unsubscribeSelection = proj.onChange(function (sel) {
            if (!sawInitialSelectionFire) {
                sawInitialSelectionFire = true;
                return;
            }
            const dir = (sel && sel.dir) ? sel.dir : "";
            if (!dir || dir === lastScannedDir) return;
            // "" -> take the newest; a folder change auto-picks
            // (results.md § 2.1).
            _rescanDir(dir, "");
        });
        document.addEventListener(C.EVENT_INSPECTOR_READY,
                                  _onInspectorReady);

        // -- dropdown change handler ----------------------------- //
        //
        // Two side-effects per pick:
        //
        //   1. setShared(dir, file) mirrors the sidebar's current
        //      pointer so the sidebar UI highlights the active
        //      file (cosmetic; the sidebar no longer steers the
        //      inspector on /results — task #301).
        //
        //   2. dispatch ``molbuilder:results:fileSelected`` so the
        //      /results dispatcher mounts the matching inspector.
        //      A custom event (vs. a method call) keeps the picker
        //      decoupled from the dispatcher's module identity.
        function _emitFileSelected(file) {
            try {
                document.dispatchEvent(new CustomEvent(
                    C.EVENT_FILE_SELECTED,
                    { detail: { file: file || "" } }));
            } catch (_) {
                // CustomEvent should always be available in supported
                // browsers; the try/catch is belt + braces for older
                // headless test runners.
            }
            _startParseStatus(file);
        }
        function _onSelectChange() {
            const newPath = selEl.value;
            if (!newPath) return;
            const parts = parseDir(newPath);
            const r = proj.setShared(parts.dir, newPath);
            if (r && r.ok === false) {
                console.warn(
                    "[results-file-picker] setShared refused:",
                    r.error
                );
                _revertSelectTo(/*last-known good*/ null);
                return;
            }
            _emitFileSelected(newPath);
        }

        function _revertSelectTo(path) {
            const opts = selEl.options;
            for (let i = 0; i < opts.length; i += 1) {
                if (opts[i].value === path) {
                    selEl.selectedIndex = i;
                    return;
                }
            }
        }

        selEl.addEventListener("change", _onSelectChange);

        // -- pageshow / visibilitychange: force-rescan on tab re-entry -- //
        //
        // The picker rescans when the directory CHANGES.  That misses two
        // real-world re-entry scenarios where the directory is the same:
        //
        //   1. bfcache restore.  Browsers (Chromium + Firefox by
        //      default) cache the whole page when the user navigates
        //      away.  Hitting the Results tab via back/forward, or
        //      via the in-app tab link on some routes, restores the
        //      cached DOM + JS state -- no DCL, no module re-init,
        //      no fresh onChange fire.  ``cachedResults`` from the
        //      previous visit stay stale, so a new .out file
        //      generated while the user was on /modify never appears.
        //
        //   2. Same-dir refresh after an external change.  The user
        //      generates a new result file from another tab.  Returning to
        //      /results, the directory is unchanged, so nothing would
        //      re-scope the menu.  Result: a stale dropdown.
        //
        // Hooking ``pageshow`` covers both (the event fires on every
        // page show -- initial load AND bfcache restore -- with
        // ``event.persisted`` distinguishing them).  We also re-trigger
        // on ``visibilitychange``->visible so a backgrounded tab
        // refreshes when the user re-focuses it; same defense.
        //
        // Force-rescan policy: read the current sessionStorage state and
        // re-scope unconditionally -- the whole point is that an unchanged
        // dir should still get a fresh listing.

        function _forceRescan() {
            if (disposed) return;
            if (!proj || typeof proj.onChange !== "function") return;
            // ONE reader owns the per-tab keying (projects.md § 2) --
            // projects.getCurrentDir()/getCurrentFile().  When that reader
            // is absent (the sidebar module never mounted) there is nothing
            // compliant to ask: a raw shared-key read here was the fork the
            // contract forbids, and it answered with another tab's place.
            const cur = (typeof proj.getCurrentDir === "function")
                ? proj.getCurrentDir()
                : "";
            const curFile = (typeof proj.getCurrentFile === "function")
                ? proj.getCurrentFile()
                : "";
            if (!cur) return;
            // Unconditional: an unchanged dir must still get a fresh listing,
            // which is the whole point of a force-rescan.
            _rescanDir(cur, curFile || "");
        }

        function _onPageShow(_evt) {
            // ``event.persisted`` is true for bfcache restore, false
            // for a fresh navigation.  We force-rescan in BOTH cases
            // -- the fresh-navigation case is already handled by the
            // initial onChange fire, so the second invocation is a
            // cheap no-op for empty cachedResults; the bfcache case
            // is the load-bearing one.
            _forceRescan();
        }

        function _onVisibilityChange(_evt) {
            if (root.document
                && root.document.visibilityState === "visible") {
                _forceRescan();
            }
        }

        if (root.addEventListener) {
            root.addEventListener("pageshow", _onPageShow);
        }
        if (root.document && root.document.addEventListener) {
            root.document.addEventListener(
                "visibilitychange", _onVisibilityChange);
        }

        // Initial bootstrap: with the sidebar onChange subscription
        // retired (task #301), the picker no longer gets a "current
        // selection" callback on mount.  Trigger one rescan
        // explicitly so the dropdown populates on first load.
        // pageshow also fires once on fresh navigation but only
        // AFTER mount returns; this call covers the early window so
        // the dropdown is visible by the time the user looks.
        _forceRescan();

        // -- Refresh button: explicit user-driven rescan -------- //
        //
        // Click-stacking guard: a double-click would otherwise fire
        // two _forceRescan calls, each spawning its own fetch +
        // resolver — the later resolver wins, but the wasted scan
        // can leave the meta line flickering through two transient
        // statuses.  Disable the button while a scan is in flight;
        // re-enable when ``cachedResults`` lands or the picker
        // hides itself.  ``aborter`` is the picker's per-scan
        // AbortController; we tie the button state to its
        // presence.  The CSS rule on .result-list-refresh:disabled
        // shifts cursor to ``wait`` so the disabled state reads as
        // "doing it" rather than "not allowed".
        const refreshBtn = rootEl.querySelector("#results-file-picker-refresh");
        function _setRefreshBusy(busy) {
            if (refreshBtn) refreshBtn.disabled = !!busy;
        }
        function _onRefreshClick() {
            if (disposed) return;
            if (refreshBtn && refreshBtn.disabled) return;
            _setRefreshBusy(true);
            // Brief visual ack so the click feels responsive even
            // when the listing is already up-to-date.
            _showTransientStatus("Refreshing…");
            _forceRescan();
            // Tell any currently-mounted inspector to re-fetch its
            // underlying data NOW instead of waiting for its next
            // polling tick (which the trajectory inspector sets at
            // 60 s -- way too slow for a deliberate user refresh).
            // Fired separately from EVENT_FILE_SELECTED because
            // the file path didn't change -- re-emitting that would
            // remount + lose camera/playback state.  Inspectors
            // that don't poll (e.g. spectra, structure) can ignore
            // the event entirely.
            try {
                document.dispatchEvent(new CustomEvent(
                    C.EVENT_REFRESH_REQUESTED));
            } catch (_) {
                // CustomEvent unavailable -- belt + braces for older
                // headless test runners; the picker's rescan still
                // gives the user the dir listing update.
            }
        }
        if (refreshBtn) {
            refreshBtn.addEventListener("click", _onRefreshClick);
        }
        // Hook the button's busy state to the picker's scan
        // lifecycle without re-plumbing _scan.  Watch the meta
        // line's ``is-busy`` class — true during scan + parse —
        // and clear the busy flag when it goes back to idle.
        // MutationObserver is the cleanest signal that doesn't
        // require touching _scan's internals (and survives the
        // edge case where the scan resolves to an empty dir, which
        // hides the bar entirely).
        const _metaObserver =
            (metaEl && typeof MutationObserver === "function")
                ? new MutationObserver(() => {
                    if (!refreshBtn) return;
                    if (!metaEl.classList.contains("is-busy")) {
                        _setRefreshBusy(false);
                    }
                })
                : null;
        if (_metaObserver) {
            _metaObserver.observe(metaEl, {
                attributes: true,
                attributeFilter: ["class"],
            });
        }

        // -- lock-state subscriber (disable while Save in flight) //
        let unsubscribeLock = null;
        if (typeof proj.onLockChange === "function") {
            unsubscribeLock = proj.onLockChange((st) => {
                selEl.disabled = !!(st && st.locked);
                selEl.title = st && st.locked
                    ? "Sidebar is locked while a save is in progress."
                    : "";
            });
        }

        // -- disposer ------------------------------------------- //
        return {
            dispose() {
                disposed = true;
                _abortInFlight();
                _clearParseTimer();
                try {
                    document.removeEventListener(
                        C.EVENT_INSPECTOR_READY, _onInspectorReady);
                } catch (_) { /* ignore */ }
                try { selEl.removeEventListener("change", _onSelectChange); }
                catch (_) { /* ignore */ }
                try { if (refreshBtn) {
                    refreshBtn.removeEventListener("click", _onRefreshClick);
                } }
                catch (_) { /* ignore */ }
                try { if (_metaObserver) _metaObserver.disconnect(); }
                catch (_) { /* ignore */ }
                try { if (unsubscribeSelection) unsubscribeSelection(); }
                catch (_) { /* ignore */ }
                try { if (unsubscribeLock) unsubscribeLock(); }
                catch (_) { /* ignore */ }
                try {
                    if (root.removeEventListener) {
                        root.removeEventListener("pageshow", _onPageShow);
                    }
                } catch (_) { /* ignore */ }
                try {
                    if (root.document
                        && root.document.removeEventListener) {
                        root.document.removeEventListener(
                            "visibilitychange", _onVisibilityChange);
                    }
                } catch (_) { /* ignore */ }
            },
        };
    }

    // -------- export -------------------------------------------- //

    root.molbuilder = root.molbuilder || {};
    root.molbuilder.resultsFilePicker = {
        mount:               mount,
        parseDir:            parseDir,
        formatRelativeTime:  formatRelativeTime,
        filterToResultFiles: filterToResultFiles,
        groupResultFiles:    groupResultFiles,
        absorbSatellites:    absorbSatellites,
        _labelForResult:     labelForResult,
    };
})(typeof window !== "undefined" ? window : this);
