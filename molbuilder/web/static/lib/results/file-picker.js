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
     * Given ``/api/results/dir``'s ``files`` (each ``{name, role, label,
     * stage, parser, opens, size, mtime}``), return **the ones this menu
     * lists** — both halves of that: a parser can read it AND a presenter
     * claims it as a result.
     *
     * ``parser`` is the REGISTRY's verdict, decided on the server by
     * ``detect()``; ``null`` means nothing can read the file and it is not
     * offered.  This took ``/api/files/list``'s content-blind entries and
     * seven filename predicates until 2026-09-18 (`plans/plan.md` N9).
     *
     * Still sorted newest-first, but that is now only the MENU's order --
     * WHICH file opens is ``openable``, the door's own pick, applied in
     * ``_scan``.  The two were the same value here, which is how a spectrum
     * run came to open a 1,373-byte progress stub: its spectrum shared an
     * mtime with it and lost the name tie-break.
     *
     * Pure (no DOM, no fetch).  Exported for unit tests.
     */
    function filterToResultFiles(entries, dirPath, engine, pickResult) {
        if (!Array.isArray(entries)) return [];
        // Use a forward-slash separator regardless of OS.
        const sep = dirPath && dirPath.indexOf("\\") >= 0 ? "\\" : "/";
        const out = [];
        for (const entry of entries) {
            if (!entry) continue;
            // THE SERVER SAYS WHETHER A FILE CAN BE OPENED, and this is the
            // whole of the change: `parser` is the REGISTRY's answer
            // (`/api/results/dir`), where this used to run seven filename
            // predicates of its own.  Measured 2026-09-18 over 110 real run
            // directories, that guess offered 13 files no parser can read --
            // Slurm logs, a 0-byte `.out`, a `job-set.json` whose own route
            // then answers 400 -- and hid 155 that a parser handles.
            if (!entry.parser) continue;
            const fullPath = dirPath ? (dirPath + sep + entry.name) : entry.name;
            const rec = {
                name:   entry.name,
                path:   fullPath,
                mtime:  entry.mtime,
                size:   entry.size,
                role:   entry.role || null,
                // WHICH RUN, and which rung -- the server's reading of the
                // name (`runfiles.parse`).  What `absorbs` compares.
                label:  entry.label || null,
                stage:  entry.stage || null,
                parser: entry.parser,
                engine: engine || null,
            };
            // AND A PRESENTER MUST CLAIM IT, which is the SECOND half of
            // "a file this menu lists" and used to be asked somewhere else.
            // `parser != null` is the REGISTRY's verdict (can anything read
            // it); `pickResult` is the PICKER's (is it a result, or a
            // catch-all the dropdown must not flood with).  Two questions,
            // and this list answered only the first while `_populate`
            // answered both -- so the meta line counted files the menu did
            // not show.  Measured 2026-09-19 on
            // `BDT-only-init-siesta`: "7 of 7" printed under a two-option
            // dropdown.  One list, one count.
            if (typeof pickResult === "function") {
                let claimed = null;
                try { claimed = pickResult(fullPath, rec); }
                catch (e) {
                    console.warn("[results-file-picker] pickResult() threw "
                                 + "for " + entry.name + ":", e);
                }
                if (!claimed) continue;
            }
            out.push(rec);
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
     * A presenter declares this with an optional ``absorbs(master, other,
     * masterMeta, otherMeta)`` -- the metas being the server's reading of
     * each file (`{role, label, stage, ...}` from `/api/results/dir`), which
     * is what lets a presenter ask *same run?* instead of doing arithmetic on
     * the two names.  The engine naming lives there, with the code that
     * knows it.  Absorption
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
            try { insp = pickResult(master.path, master); }
            catch (e) { insp = null; }
            if (!insp || typeof insp.absorbs !== "function") continue;
            for (const other of entries) {
                if (other === master || absorbed.has(other.path)) continue;
                let claims = false;
                try {
                    claims = !!insp.absorbs(master.path, other.path,
                                            master, other);
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
     * ``resultCategory(path, entry)`` and return a sorted ``[{label,
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
            const inspector = pickResult(entry.path, entry);
            if (!inspector) continue;  // already filtered, but defensive.
            let label;
            try {
                /* THE SAME ANSWER THE PICK WAS MADE ON, one line up.
                 * `entry` carries the server's `{role, parser, engine}`
                 * (`filterToResultFiles`), `pickResult` is given it, and
                 * `resultCategory` was not -- so every heading fell back to
                 * the engine-less spelling and a SIESTA directory's runs sat
                 * under "Optimization" (measured 2026-09-19 on
                 * `BDT-only-init-siesta`, whose `engine` the server answered
                 * `siesta`).  Two calls, one rule: both take `meta`
                 * (`web/presenters.md` § 3). */
                label = (typeof inspector.resultCategory === "function")
                    ? inspector.resultCategory(entry.path, entry)
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

    /* A selected, disabled first row for "the door picked nothing".
     * Without it the browser displays the first option anyway, so the menu
     * would name a file that is not mounted -- the label/display split this
     * picker already had to fix once. */
    function _prependUnchosen(selectEl, text) {
        const opt = document.createElement("option");
        opt.value = "";
        opt.textContent = text;
        opt.disabled = true;
        opt.selected = true;
        selectEl.insertBefore(opt, selectEl.firstChild);
        selectEl.selectedIndex = 0;
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
        //: THE DIRECTORY THIS PANEL IS BOUND TO -- owned here, changed by
        //: ONE thing (Refresh, via `_alignToSidebar`).  It was a scan memo
        //: called `lastScannedDir` until 2026-09-19, when the sidebar stopped
        //: dragging the panel around: with the panel able to sit on a folder
        //: the sidebar has left, "which folder is this" is state somebody has
        //: to own, and a memo cannot be asked.
        let boundDir        = null;
        //: Set by the Reload click, consumed by the next announcement.
        //: `results.md` 4 states the contract in four words -- *"Reload =
        //: open the same file again"* -- and the same-file no-op added on
        //: 2026-09-19 quietly broke it for every viewer that neither polls
        //: nor listens for a refresh: structure, source, markdown.  For
        //: those three the remount WAS the re-read, so Reload stopped
        //: reaching the disk and showed you the geometry you already had.
        let forceNextAnnounce = false;
        //: What the SERVER said this directory is -- `{role, calculation}`
        //: or null when it does not say (project-layout.md § 1.4a).  Kept
        //: from the last scan so the empty state can render the answer
        //: instead of a generic sentence.
        let lastPlace       = null;
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
            /* FORGET THE LAST DIRECTORY'S ANSWER HERE, once, rather than in
             * each of the four ways this scan can end.  Only the success
             * path can restate it, so a failed listing, an aborted scan or a
             * cleared sidebar cannot leave the previous directory's `place`
             * riding on the next announcement. */
            lastPlace = null;
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
            // ASK THE DOOR.  `/api/results/dir` is the HTTP surface over
            // `parse.dirs` -- one call answering what is here, what reads
            // each file, and which one to open.  This listed through the
            // content-blind file browser and decided all three itself until
            // 2026-09-18 (`plans/plan.md` N9: the door had no consumer).
            fetch("/api/results/dir?path=" + encodeURIComponent(dir),
                  { signal: signal, cache: "no-store" })
                .then(r => r.json())
                .then(body => {
                    if (disposed || signal.aborted) return;
                    if (!body || body.ok !== true) {
                        // Fetch failed (file listing API errored).
                        // Keep the bar visible so Refresh is still
                        // clickable -- a transient error shouldn't
                        // strand the user; one Refresh click retries.
                        if (metaEl) metaEl.classList.remove("is-busy");
                        _populatePlaceholder(selEl,
                            "(directory listing failed — click Reload)");
                        cachedResults = [];
                        cachedGroups  = [];
                        _showIdleMeta(null);
                        return;
                    }
                    // Filter to result-class files, then let a run's master
                    // absorb its working files so the menu lists the RUN, not
                    // its parts (results.md § 2.3).
                    const results = absorbSatellites(
                        filterToResultFiles(body.files || [],
                                            body.run_dir || dir,
                                            body.engine,
                                            inspReg.pickResult),
                        inspReg.pickResult
                    );
                    lastPlace = body.place || null;
                    cachedResults = results;
                    cachedGroups = results.length
                        ? groupResultFiles(results, inspReg.pickResult)
                        : [];
                    barEl.hidden = false;

                    /* ---- ONE DECISION ------------------------------- //
                     * What should be mounted for this directory -- a path,
                     * or nothing.  Three answers, one variable, because the
                     * announcement below must not depend on which of them
                     * happened.
                     *
                     * KEEP what is already showing if this directory still
                     * offers it; otherwise take THE DOOR'S PICK -- `openable`
                     * is `parse.dirs.openable_in`'s answer, the file this
                     * CALCULATION produced, the same file during the run and
                     * after it.  Taking `results[0]` instead differed from
                     * the door on 18 of 96 real run directories.
                     *
                     * AND NOTHING IS AN ANSWER.  When the door offers no
                     * pick, nothing here is this directory's product, so a
                     * fallback to `results[0]` shows something that is not
                     * the result and looks like one: measured 2026-09-19,
                     * the transmission rung -- holding `.TBT.nc` and both
                     * transmission curves -- opened `…util.csv`, the CPU
                     * utilisation samples; and a hierarchical calculation
                     * root opened its own INPUT structure's sidecar as raw
                     * JSON.  The menu still lists every readable file; only
                     * the guess is gone.
                     *
                     * ONE VALUE, because it used to be two: `_populate`
                     * labelled the menu from the GROUPED list (ties by
                     * category label) while the auto-pick took `results[0]`
                     * from the flat one (ties by file name).  Both orderings
                     * are right; having two is the defect -- four files
                     * stamped 10:31:08 labelled the menu `…molwatch.log` and
                     * displayed `…_optimized.xyz` (2026-08-04).  The chosen
                     * path now feeds both. */
                    const keepCurrent =
                        currentFile && results.some(r => r.path === currentFile);
                    const byDoor = body.openable
                        ? results.find(r => r.name === body.openable)
                        : null;
                    const chosen = keepCurrent ? currentFile
                                 : (byDoor ? byDoor.path : null);

                    /* ---- ONE EXIT ----------------------------------- //
                     * `results.md` § 2.2: a scan that changes what is
                     * current ALWAYS announces it.  That rule was stated in
                     * a comment and kept by hand in each branch, so adding a
                     * branch broke it -- the "nothing openable" case landed
                     * 2026-09-19 with a bare `return`, and a stage container
                     * went on showing its run-0 spectrum, tabs and all, from
                     * a directory the user had left.  The announcement is
                     * structural now: every path out of this scan passes
                     * through it, so the next branch cannot forget. */
                    if (metaEl) metaEl.classList.remove("is-busy");
                    if (chosen === null) {
                        if (results.length === 0) {
                            _populatePlaceholder(selEl,
                                "(no result files yet — click Reload)");
                        } else {
                            _populate(selEl, cachedGroups, null);
                            _prependUnchosen(selEl,
                                "— nothing here is this directory's result; "
                                + "pick a file —");
                        }
                        _showIdleMeta(null);
                        _emitFileSelected("");
                        return;
                    }
                    _populate(selEl, cachedGroups, chosen);
                    /* ONE EXIT, ALWAYS ANNOUNCED.  The `keepCurrent` arm used
                     * to stop at `_startParseStatus` on the grounds that the
                     * file was "already mounted" -- true on a re-entry, false
                     * on the FIRST scan of a page load, which is the arm that
                     * runs whenever the remembered file is still in the
                     * listing.  The panel was then showing an inspector the
                     * picker had never announced, chosen by filename because
                     * no `meta` ever reached it.  Since 2026-09-19 the list is
                     * the only thing that decides what is shown, so it says so
                     * every time; `viewer.js` no-ops when the file has not
                     * actually changed, which is what makes a tab-re-entry
                     * rescan cheap instead of a remount. */
                    _adoptSelection(chosen);
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
                        "(scan failed — click Reload to retry)");
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
        /**
         * Mirror this pick into the sidebar's pointer -- ONLY when the
         * sidebar is already listing our folder.  Returns false when the
         * caller should stop (a refusal), true otherwise.
         *
         * MIRRORING IS A COURTESY: it highlights the row you picked, and
         * that is meaningful only while the sidebar is showing this folder.
         * Once the panel stopped following the sidebar (2026-09-19) the same
         * call became a shove in the other direction -- bound to A, browsing
         * B, you pick in the menu and the sidebar snaps back to A.  Worse
         * than losing your place: `_divergedFromSidebar()` then answers
         * false, so the header drops its warning while the sidebar visibly
         * still lists B, and Reload can no longer reach B at all.  The one
         * honest signal on the panel is switched off by a pick inside it.
         *
         * THIS IS ONE FUNCTION because the guard was written on the
         * automatic path alone and the MANUAL one -- the case the comment
         * itself described, "you pick in the menu" -- was left open until
         * 2026-09-19.  Two call sites, one rule, no second chance to guard
         * only half of it.
         */
        function _mirrorToSidebar(dir, path) {
            if (_divergedFromSidebar()) return true;
            const r = proj.setShared(dir, path);
            if (r && r.ok === false) {
                console.warn(
                    "[results-file-picker] selection refused:", r.error);
                return false;
            }
            return true;
        }

        function _adoptSelection(path) {
            if (!path) return;
            const parts = parseDir(path);
            if (!_mirrorToSidebar(parts.dir, path)) return;
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
            /* BIND FIRST, ANNOUNCE FIRST -- and the header is written to
             * survive it.  `boundDir` has to move before the fetch, because
             * the fetch's own resolver keys off it; so for one round-trip
             * (seconds on a cold NFS mount) the panel is bound to a folder
             * it has not read, still showing the previous folder's file.
             *
             * That window is not hypothetical: if the scan then FAILS -- the
             * folder was deleted, the route answers 404 -- there is no
             * announcement to correct it and the state is permanent until
             * the next Reload.  The header used to print `<new folder> /
             * <old file>`, a path that does not exist, in the plain colour.
             * That is the 2026-08-04 defect wearing its own mitigation.
             *
             * The fix is in `_renderStatus`, not here: it compares the
             * mounted file's own directory against the bound one and says
             * "showing <full path>, which is not in this folder" when they
             * disagree.  Binding early is then safe because the header
             * cannot imply the file came from the new folder. */
            boundDir = dir;
            _announceScope();
            _scan(dir, preferredFile || "");
        }

        /* THE SIDEBAR DOES NOT MOVE THIS PANEL.  Browsing is browsing.
         *
         * This has been decided three times and the record matters, because
         * both answers are defensible and each fixed the other's bug:
         *
         *   2026-06-09 (#301) -- the subscription was RETIRED: single-clicking
         *     around the sidebar hijacked the inspector mid-read.
         *   2026-08-04 -- the DIRECTORY half came back, because a panel that
         *     scoped itself once at mount went on rendering a previous folder:
         *     a live BDT-Au111 job displayed as a finished BDT run from another
         *     directory, "every number plausible and every number wrong".
         *   2026-09-19 (user) -- retired again, WITH the thing that was
         *     missing both times: the header now names the folder these
         *     results came from and says when the sidebar has left it.
         *
         * Read the 2026-08-04 note again and the actual fault is in its last
         * clause -- "with nothing on screen saying so".  Following the sidebar
         * was one way to make the panel honest; naming the folder is the
         * other, and it is the one that also lets you scroll around without
         * losing your place.  Taking the second does not make the first wrong;
         * it makes it unnecessary.  If the readout ever goes away, this
         * subscription has to come back.
         *
         * Re-scans now happen on exactly three things: the Refresh button
         * (which re-points the panel at the sidebar), tab re-entry (which
         * re-reads the folder already bound), and your own pick in the menu.
         */
        function _announceScope() {
            try {
                document.dispatchEvent(new CustomEvent(
                    C.EVENT_SCOPE_CHANGED,
                    { detail: { dir: boundDir,
                                diverged: _divergedFromSidebar() } }));
            } catch (_) { /* older headless runners */ }
        }

        /* The ONE thing the sidebar still reaches: the header's wording.
         * No scan, no re-scope, no mount -- `_announceScope` dispatches a
         * display-only event.  Keeping this subscription is what lets the
         * readout stay true while you browse, which is the condition the
         * note above attaches to not following. */
        const unsubscribeSelection = proj.onChange(function () {
            _announceScope();
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
        /** The server's answer for one path -- `{role, parser, engine}` --
         * so the viewer dispatches on it instead of re-reading the name. */
        function _metaFor(file) {
            const hit = (cachedResults || []).find(r => r.path === file);
            return hit ? { role: hit.role, parser: hit.parser,
                           engine: hit.engine } : null;
        }

        function _emitFileSelected(file) {
            const forced = forceNextAnnounce;
            forceNextAnnounce = false;
            try {
                document.dispatchEvent(new CustomEvent(
                    C.EVENT_FILE_SELECTED,
                    /* `place` RIDES ALONG so the empty state can say what
                     * this directory IS rather than guessing.  The server
                     * already answered it in the same response the menu was
                     * built from (`/api/results/dir`, project-layout.md
                     * § 1.4a); dropping it here is why a stage container and
                     * a folder nobody described got the same sentence --
                     * "no result files yet", which is true of neither. */
                    /* `dir` and `diverged` ride along for the same reason
                     * `place` does -- the header has to NAME the folder these
                     * results come from, and say when the sidebar has moved on
                     * without them.  That readout is not decoration: it is the
                     * mitigation for 2026-08-04, where the panel went on
                     * rendering another run "with nothing on screen saying
                     * so".  Unbinding the panel from the sidebar is only safe
                     * because this is said out loud. */
                    { detail: { file: file || "", meta: _metaFor(file),
                                place: lastPlace, dir: boundDir,
                                diverged: _divergedFromSidebar(),
                                force: forced } }));
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
            if (!_mirrorToSidebar(parts.dir, newPath)) {
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

        /** Is the sidebar somewhere else than this panel? */
        function _divergedFromSidebar() {
            if (!boundDir || !proj
                || typeof proj.getCurrentDir !== "function") return false;
            const cur = proj.getCurrentDir();
            return !!cur && cur !== boundDir;
        }

        /**
         * REFRESH -- the one gesture that re-points this panel.
         *
         * Reads where the sidebar is now, binds to it, and scans.  This is
         * the whole of the sidebar's authority over the Results tab since
         * 2026-09-19 (`results.md` § 2.1): browsing does nothing, Refresh
         * adopts.  `preferredFile` keeps your pick when the new listing
         * still offers it.
         */
        function _alignToSidebar() {
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
            if (!cur) {
                // Nothing to adopt -- the sidebar has not resolved a folder
                // yet.  Say so and RELEASE THE BUTTON: the busy class is set
                // by the click handler and cleared only inside `_scan`, so
                // bailing here used to disable Reload for the rest of the
                // page load -- on the one screen whose empty state tells you
                // to press it.
                _populatePlaceholder(selEl,
                    "(no project directory yet — pick one in the sidebar)");
                _showIdleMeta(null);
                return;
            }
            // THE MENU'S CHOICE, NOT THE SIDEBAR'S FILE.  This read
            // `proj.getCurrentFile()`, so a sidebar SINGLE-CLICK -- which the
            // contract says does nothing here -- silently decided what the
            // next Reload would mount.  #301's hijack, deferred behind one
            // button press.  `_rescanBound` already used the menu; both
            // rescan paths now agree where "the file we want" comes from.
            _rescanDir(cur, _currentChoice());
        }

        /**
         * TAB RE-ENTRY -- re-read the folder we are ALREADY on.
         *
         * The other half of what `_forceRescan` used to be.  Coming back to
         * the browser tab should show files written while you were away; it
         * must not quietly re-point the panel at wherever the sidebar
         * drifted, because nobody asked it to.  Before the split these were
         * one function, so focus-return was a silent Refresh.
         *
         * Before the first bind there is nothing to re-read, so the very
         * first one adopts -- that is the initial bind, not a re-point.
         */
        function _rescanBound() {
            if (disposed) return;
            if (boundDir === null) { _alignToSidebar(); return; }
            _rescanDir(boundDir, _currentChoice());
        }

        /** What the menu is showing right now, so a rescan can keep it. */
        function _currentChoice() {
            return (selEl && selEl.value) ? selEl.value : "";
        }

        function _onPageShow(_evt) {
            // ``event.persisted`` is true for bfcache restore, false
            // for a fresh navigation.  We force-rescan in BOTH cases
            // -- the fresh-navigation case is already handled by the
            // initial onChange fire, so the second invocation is a
            // cheap no-op for empty cachedResults; the bfcache case
            // is the load-bearing one.
            //
            // RE-READ, NOT RE-POINT (2026-09-19): coming back to the tab
            // shows what was written while you were away; it does not adopt
            // wherever the sidebar has since gone.
            _rescanBound();
        }

        function _onVisibilityChange(_evt) {
            if (root.document
                && root.document.visibilityState === "visible") {
                _rescanBound();
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
        //
        // This one DOES read the sidebar -- it is the initial bind, and
        // `projects` already keeps a per-page folder slot (`projects.md`
        // § 2, "the Results tab keeps its run folder"), so what it adopts is
        // where this tab was last pointed, not some other tab's place.
        _alignToSidebar();

        // -- Refresh button: explicit user-driven rescan -------- //
        //
        // Click-stacking guard: a double-click would otherwise fire
        // two _alignToSidebar calls, each spawning its own fetch +
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
            _showTransientStatus("Reloading…");
            forceNextAnnounce = true;
            _alignToSidebar();
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
