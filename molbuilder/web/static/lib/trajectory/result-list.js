/* Result-list dropdown -- in-inspector navigator for SIESTA / PySCF
 * output files in the same directory.
 *
 * Mount target: the ``#result-list-bar`` block in
 * ``_trajectory_inspector.html``.  Render is conditional: when
 * ``/api/files/result-list?path=<file>`` returns ``<= 1`` entry the
 * bar stays hidden (nothing to navigate to).
 *
 * Picking a different entry from the <select> fires
 * ``projects.setShared(dir, newFile)`` -- the sidebar's onChange
 * subscriber then re-mounts the trajectory inspector with the new
 * file, just like a sidebar click would.  Single source of truth
 * for current-file state stays on the projects sidebar.
 *
 * Visible API on ``window.molbuilder.trajectoryResultList``:
 *
 *   ``mount(host, file)``  -- fetch + populate; returns a disposer.
 *   ``parseDir(file)``     -- pure helper (exported for testing).
 *   ``formatRelativeTime(epoch_seconds)`` -- pure helper (testing).
 */
(function (root) {
    "use strict";

    const ENDPOINT = "/api/files/result-list";

    /**
     * Split a file path into ``{dir, name}`` parts WITHOUT importing
     * the global path utility -- this module needs to be loadable
     * in test contexts without a full ``molbuilder.path`` mount.
     * Handles both POSIX and Windows separators defensively.
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
     * Format an epoch_seconds value as a relative-time string
     * suitable for the dropdown's "(2m ago)" / "(3h ago)" suffix.
     * Returns "" when the epoch is missing.
     */
    function formatRelativeTime(epoch_s) {
        if (epoch_s == null || !Number.isFinite(epoch_s)) return "";
        const now = Date.now() / 1000;
        const delta = Math.max(0, now - epoch_s);
        if (delta < 60)       return Math.floor(delta) + "s ago";
        if (delta < 3600)     return Math.floor(delta / 60) + "m ago";
        if (delta < 86400)    return Math.floor(delta / 3600) + "h ago";
        const days = Math.floor(delta / 86400);
        if (days < 30)        return days + "d ago";
        return new Date(epoch_s * 1000).toLocaleDateString();
    }

    /**
     * Build the <option> label text.  The filename is the primary
     * read; the relative timestamp + (optional) run-index live as
     * a parenthesised tail.  Keep the prefix short so long base
     * names don't push the timestamp off-screen.
     */
    function labelForResult(entry) {
        const rel = formatRelativeTime(entry.mtime);
        const tag = entry.run_index !== null
            ? "run " + entry.run_index
            : "single";
        const tail = [tag, rel].filter(Boolean).join(", ");
        return entry.name + " (" + tail + ")";
    }

    /**
     * Populate the <select> with one <option> per result; mark the
     * current one as selected.  Used both at mount time and when
     * the inspector swaps files (the dropdown re-syncs its
     * selected-option to match the new active file).
     */
    function _populate(selectEl, results, currentPath) {
        // Clear without using innerHTML (avoids any chance of an
        // XSS reflection if a filename ever contains tag chars --
        // unlikely on a filesystem but cheap to be safe).
        while (selectEl.firstChild) {
            selectEl.removeChild(selectEl.firstChild);
        }
        for (const r of results) {
            const opt = document.createElement("option");
            opt.value = r.path;
            opt.textContent = labelForResult(r);
            opt.title = r.name;   // hover-reveal full name when ellipsised
            if (r.path === currentPath) {
                opt.selected = true;
            }
            selectEl.appendChild(opt);
        }
    }

    /**
     * Render the "N of M, last updated X ago" meta line under the
     * dropdown.  Pure derivation from the results list + current
     * file path.
     */
    function _renderMeta(metaEl, results, currentPath) {
        const ix = results.findIndex(r => r.path === currentPath);
        if (ix < 0) {
            // Current file isn't in the result list (rare: e.g. user
            // picked a .spectra.json file that the inspector didn't
            // include).  Show total only.
            metaEl.textContent = results.length + " file" +
                (results.length === 1 ? "" : "s");
            return;
        }
        // The list is sorted newest-first (server-side); convert
        // back to chronological-index for the user-facing
        // "X of N" so a freshly-created run0 shows as "1 of 3".
        const chronIx = results.length - ix;
        const cur = results[ix];
        const rel = formatRelativeTime(cur.mtime);
        const parts = [chronIx + " of " + results.length];
        if (rel) parts.push(rel);
        metaEl.textContent = parts.join(" · ");
    }

    /**
     * Mount the result-list bar against ``host``.  The host is the
     * trajectory inspector's already-injected DOM (see
     * ``lib/inspectors/trajectory.js`` for the partial mount flow).
     * The bar's elements live inside the partial; we grab them by
     * id and wire the fetch + change handler.
     *
     * Returns a disposer that aborts any in-flight fetch and
     * detaches the change listener.  Idempotent.
     */
    function mount(host, file) {
        // Scope lookups to ``host`` only -- no document.getElementById
        // fallback.  Each inspector mount clears the host and re-
        // injects the partial; falling back to document.* would risk
        // matching a stale node from a previous mount cycle on a
        // different host (e.g. /watch + /results sharing the page).
        const barEl   = host.querySelector("#result-list-bar");
        const selEl   = host.querySelector("#result-list-select");
        const metaEl  = host.querySelector("#result-list-meta");
        if (!barEl || !selEl) {
            // Partial DOM missing -- this inspector instance won't
            // have the bar (e.g. caller mounted on a non-trajectory
            // host).  Silent no-op.
            return { dispose() { /* nothing to clean */ } };
        }

        let aborted = false;
        const abortCtl = new AbortController();

        function _onChange() {
            const newPath = selEl.value;
            if (!newPath || newPath === file) return;
            const proj = (root.molbuilder || {}).projects;
            if (!proj || typeof proj.setShared !== "function") {
                console.warn(
                    "[result-list] projects.setShared unavailable; "
                    + "cannot navigate"
                );
                _revertSelectTo(file);
                return;
            }
            // Derive dir from the new path (it lives in the same
            // directory as ``file`` so we could reuse parseDir(file).dir,
            // but using newPath is robust if the user ever moves a
            // file into a sibling dir mid-session).
            const parts = parseDir(newPath);
            // setShared returns {ok:true} on success, {ok:false,
            // error:"sidebar is locked: ..."} when the lock guard
            // (#177) rejects.  Before this commit we ignored the
            // return -- a locked rejection left the dropdown showing
            // the new (unapplied) option and the inspector stale.
            const r = proj.setShared(parts.dir, newPath);
            if (r && r.ok === false) {
                console.warn(
                    "[result-list] setShared refused:", r.error
                );
                _revertSelectTo(file);
            }
        }

        /** Revert the <select>'s selected option back to ``path``.
         *  Used when setShared rejects (locked / unavailable) so the
         *  UI doesn't pretend the change took effect.  Silent no-op
         *  if no option matches (the option list was rebuilt mid-
         *  click; the inspector re-mount will sync DOM to state). */
        function _revertSelectTo(path) {
            const opts = selEl.options;
            for (let i = 0; i < opts.length; i += 1) {
                if (opts[i].value === path) {
                    selEl.selectedIndex = i;
                    return;
                }
            }
        }

        // Lock-state subscriber: visually disable the dropdown while
        // a Save pipeline is in flight (the sidebar's CSS lock only
        // covers the sidebar itself; in-inspector navigators stay
        // clickable by default).  Defense-in-depth: even if the
        // user does manage to click, _onChange's revert+warn path
        // handles the rejection.
        let lockUnsubscribe = null;
        if ((root.molbuilder || {}).projects
            && typeof root.molbuilder.projects.onLockChange === "function") {
            lockUnsubscribe = root.molbuilder.projects.onLockChange(
                (st) => {
                    selEl.disabled = !!(st && st.locked);
                    selEl.title = st && st.locked
                        ? "Sidebar is locked while a save is in progress."
                        : "";
                }
            );
        }

        selEl.addEventListener("change", _onChange);

        fetch(ENDPOINT + "?path=" + encodeURIComponent(file), {
            credentials: "same-origin",
            signal:      abortCtl.signal,
        })
            .then(resp => resp.ok ? resp.json() : null)
            .then(body => {
                if (aborted) return;
                if (!body || body.ok !== true) {
                    barEl.hidden = true;
                    return;
                }
                const results = Array.isArray(body.results)
                    ? body.results : [];
                if (results.length <= 1) {
                    // Nothing to navigate; keep hidden.  This is
                    // the common case for a fresh project with one
                    // stage's first run.
                    barEl.hidden = true;
                    return;
                }
                _populate(selEl, results, file);
                if (metaEl) _renderMeta(metaEl, results, file);
                barEl.hidden = false;
            })
            .catch(err => {
                if (err && err.name === "AbortError") return;
                console.warn(
                    "[result-list] fetch failed; hiding bar",
                    err
                );
                barEl.hidden = true;
            });

        return {
            dispose() {
                aborted = true;
                try { abortCtl.abort(); } catch (_) { /* ignore */ }
                try { selEl.removeEventListener("change", _onChange); }
                catch (_) { /* ignore */ }
                // Lock-change subscription is module-level state on
                // the projects module; failing to detach leaks one
                // entry per inspector mount cycle.
                if (lockUnsubscribe) {
                    try { lockUnsubscribe(); } catch (_) { /* ignore */ }
                    lockUnsubscribe = null;
                }
            },
        };
    }

    root.molbuilder = root.molbuilder || {};
    root.molbuilder.trajectoryResultList = {
        mount:               mount,
        parseDir:            parseDir,
        formatRelativeTime:  formatRelativeTime,
        // Test-only export.
        _labelForResult:     labelForResult,
    };
})(typeof window !== "undefined" ? window : this);
