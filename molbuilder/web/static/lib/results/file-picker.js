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
 * Picking a different entry from the <select> fires
 * ``projects.setShared(dir, newFile)`` -- the sidebar's onChange
 * subscriber then triggers viewer.js to dispose + remount the
 * matching inspector for the new file.  Single source of truth for
 * current-file state stays on the projects sidebar (design § 7).
 *
 * Auto-pick: when the directory contains result files AND no file is
 * currently selected, the picker auto-selects the most recent one.
 * Auto-pick is bypassed when the current selection is already a
 * result file in the same dir.
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
 *                         handler, subscribes to projects.onChange
 *                         + ``molbuilder:inspector:ready``, and
 *                         triggers the first scan.  Returns a
 *                         disposer that aborts any in-flight fetch
 *                         and detaches subscriptions.
 *   ``parseDir(file)``            -- pure helper (exported for testing).
 *   ``formatRelativeTime(epoch)`` -- pure helper (testing).
 *   ``filterToResultFiles(entries, dir, pickResult)`` -- pure helper.
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

    const LIST_ENDPOINT = "/api/files/list";

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
                barEl.hidden = true;
                cachedResults = [];
                cachedGroups  = [];
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
            const url = LIST_ENDPOINT + "?path=" + encodeURIComponent(dir);
            // ``cache: "no-store"`` is the load-bearing bit that fixes
            // the "user clicks Results, sees stale dropdown" bug.
            // Without it the browser HTTP cache returns the previous
            // /api/files/list response (since the URL is identical to
            // the prior scan's), so newly-generated result files don't
            // appear until the user navigates the sidebar out + back
            // in (which forces a different URL).  The directory
            // listing is cheap + always live; no caching needed.
            fetch(url, {
                credentials: "same-origin",
                signal:      signal,
                cache:       "no-store",
            })
                .then(resp => resp.ok ? resp.json() : null)
                .then(body => {
                    if (disposed || signal.aborted) return;
                    if (!body || body.ok !== true) {
                        // Bar-hide path: drop the transient busy
                        // status so the Refresh button's
                        // MutationObserver re-enables it.  Without
                        // this the button stays disabled until
                        // remount.
                        if (metaEl) metaEl.classList.remove("is-busy");
                        barEl.hidden = true;
                        cachedResults = [];
                        cachedGroups  = [];
                        return;
                    }
                    const results = filterToResultFiles(
                        body.entries || [],
                        body.path     || dir,
                        inspReg.pickResult
                    );
                    cachedResults = results;
                    if (results.length === 0) {
                        // Empty result set: no point showing the
                        // bar.  Same is-busy clear as above.
                        if (metaEl) metaEl.classList.remove("is-busy");
                        cachedGroups = [];
                        barEl.hidden = true;
                        return;
                    }
                    cachedGroups = groupResultFiles(
                        results, inspReg.pickResult);
                    _populate(selEl, cachedGroups, currentFile);
                    barEl.hidden = false;
                    // Auto-pick: if no file is currently selected (or
                    // the selection is outside this directory's result
                    // set), promote the newest result to the active
                    // selection so the inspector mounts without an
                    // extra user click.  Closes the user's primary
                    // ask (2026-06-01).
                    const selectionIsAResult =
                        currentFile && results.some(r => r.path === currentFile);
                    if (!selectionIsAResult) {
                        // Auto-pick will fire setShared -> onChange
                        // -> _onSelectionChange, which will start the
                        // parse-status timer for the auto-picked file.
                        _autoSelect(results[0]);
                    } else if (currentFile) {
                        // The current selection IS in the new dir's
                        // result set; show "Parsing…" briefly so the
                        // user gets the same acknowledgement when
                        // they re-enter a known dir.
                        _startParseStatus(currentFile);
                    } else {
                        _showIdleMeta(currentFile);
                    }
                })
                .catch(err => {
                    if (err && err.name === "AbortError") return;
                    console.warn(
                        "[results-file-picker] scan failed; hiding bar",
                        err
                    );
                    if (metaEl) metaEl.classList.remove("is-busy");
                    barEl.hidden = true;
                    cachedResults = [];
                    cachedGroups  = [];
                });
        }

        /**
         * Auto-select ``entry`` as the current file.  Mirrors the
         * pick to the sidebar via ``projects.setShared`` AND
         * dispatches ``molbuilder:results:fileSelected`` so the
         * /results dispatcher mounts the matching inspector
         * (task #301 — the onChange subscription that used to do
         * that work was retired).  No-op if the sidebar is locked
         * (setShared returns ok:false; we log + don't retry).
         */
        function _autoSelect(entry) {
            if (!entry || !entry.path) return;
            const parts = parseDir(entry.path);
            const r = proj.setShared(parts.dir, entry.path);
            if (r && r.ok === false) {
                console.warn(
                    "[results-file-picker] auto-select refused:",
                    r.error
                );
                return;
            }
            _emitFileSelected(entry.path);
        }

        // -- selection-change subscriber ------------------------- //
        // onChange fires immediately with the current selection (per
        // projects/state.js contract), so the initial scan happens
        // here without an extra call. -------------------------- //
        let lastSelectedFile = null;
        function _onSelectionChange(sel) {
            const dir  = sel && sel.dir  ? sel.dir  : "";
            const file = sel && sel.file ? sel.file : "";
            if (dir !== lastScannedDir) {
                // Directory changed: rescan from scratch.  _scan
                // clears any in-flight parse timer.
                lastScannedDir = dir;
                lastSelectedFile = file;
                _scan(dir, file);
                return;
            }
            // Same dir.
            if (cachedGroups.length > 0) {
                _populate(selEl, cachedGroups, file);
            }
            if (file && file !== lastSelectedFile) {
                // File swap within the same dir -- show "Parsing…"
                // status as acknowledgement that the inspector is
                // mounting.  Cleared by the timer (or by a follow-on
                // selection change).
                lastSelectedFile = file;
                if (cachedResults.length > 0
                    && cachedResults.some(r => r.path === file)) {
                    _startParseStatus(file);
                } else if (cachedResults.length > 0) {
                    // File is in the dir but not a result type (e.g.
                    // the user clicked a config file in the sidebar);
                    // fall back to the steady-state readout.
                    _showIdleMeta(file);
                }
            } else if (cachedResults.length > 0
                && parseTimer === null) {
                // No file change AND no parse in flight -- ensure
                // the meta is in steady state (in case a previous
                // transient was left on screen by some edge path).
                _showIdleMeta(file);
            }
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
        const unsubscribeSelection = null;
        document.addEventListener("molbuilder:inspector:ready",
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
                    "molbuilder:results:fileSelected",
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
        // The picker normally rescans only when the directory CHANGES
        // (gated on ``lastScannedDir`` in _onSelectionChange).  That
        // misses two real-world re-entry scenarios:
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
        //      generates a new result file in another tab (or another
        //      process on the same projects dir).  Returning to
        //      /results, sessionStorage's dir is unchanged, so the
        //      "if (dir !== lastScannedDir)" guard suppresses the
        //      rescan.  Result: the user sees a stale dropdown.
        //
        // Hooking ``pageshow`` covers both (the event fires on every
        // page show -- initial load AND bfcache restore -- with
        // ``event.persisted`` distinguishing them).  We also re-trigger
        // on ``visibilitychange``->visible so a backgrounded tab
        // refreshes when the user re-focuses it; same defense.
        //
        // Force-rescan policy: reset ``lastScannedDir`` to null + read
        // the current sessionStorage state + call _scan directly.
        // We deliberately bypass _onSelectionChange's "same-dir"
        // branch (which would no-op) -- the whole point is that an
        // unchanged dir should still get a fresh listing.

        function _forceRescan() {
            if (disposed) return;
            if (!proj || typeof proj.onChange !== "function") return;
            const cur = (typeof proj.getCurrentDir === "function")
                ? proj.getCurrentDir()
                : (root.sessionStorage
                    ? root.sessionStorage.getItem("molbuilder.current_dir")
                    : "");
            const curFile = (typeof proj.getCurrentFile === "function")
                ? proj.getCurrentFile()
                : (root.sessionStorage
                    ? root.sessionStorage.getItem("molbuilder.current_file")
                    : "");
            if (!cur) return;
            // Force the dir-change branch to re-fire even when the
            // sessionStorage dir matches lastScannedDir.
            lastScannedDir = null;
            lastSelectedFile = null;
            _onSelectionChange({ dir: cur, file: curFile || "" });
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
                        "molbuilder:inspector:ready", _onInspectorReady);
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
        _labelForResult:     labelForResult,
    };
})(typeof window !== "undefined" ? window : this);
