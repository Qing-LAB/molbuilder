/* Source-file inspector: paginated read-only listing for .fdf / .py /
 * .json / .log / .txt / .md files the user wants to peek at without
 * leaving the page.
 *
 * History:
 *   * 2026-05 -- v1 (single-shot): called ``ctx.readFile(file,
 *     {maxBytes: 16 MB})`` and rendered the whole text into a <pre>.
 *     Worked for tiny configs; slow + memory-heavy for MB-scale logs;
 *     hard-failed at 16 MB.
 *   * 2026-06-02 (task #119) -- v2 (paginated): fetches a 256 KB
 *     window via ``/api/files/read_range``, appends chunks as the
 *     user scrolls near the bottom, supports an explicit "Jump to
 *     end" shortcut for "show me the tail of this 100 MB log".
 *     Works on arbitrarily-large files; DOM grows only with what
 *     the user has actually scrolled past.
 */
(function (root) {
    "use strict";

    // Window size per range fetch.  The server's default is also
    // 256 KB; we mirror it explicitly so a future server-side bump
    // doesn't silently change the client's behaviour.
    const PAGE_BYTES = 256 * 1024;

    // Threshold (in pixels) for auto-loading the next page as the
    // user scrolls.  When ``scrollTop + clientHeight >= scrollHeight
    // - NEXT_PAGE_TRIGGER_PX``, the inspector fetches the next
    // chunk.  Picked to give a smooth "infinite scroll" feel without
    // double-firing on a single scroll event.
    const NEXT_PAGE_TRIGGER_PX = 200;

    const inspector = {
        name:        "source",
        displayName: "Source file",
        // Source matches generic text types (configs, scripts, READMEs)
        // and is the /results catch-all viewer.  It is NOT a "result"
        // inspector: the tab-level file-picker dropdown explicitly
        // skips files whose match resolves here, otherwise the picker
        // would flood with .fdf / .py / .json / .txt / .md files that
        // are inputs/configs, not results the user wants to inspect.
        isResult:    false,
        // Source inspector intentionally does NOT claim ``.out`` --
        // the trajectory inspector takes those (SIESTA's redirected
        // stdout parses to a frame trajectory; a raw text view of a
        // multi-MB SIESTA .out goes through the source inspector
        // ONLY via an explicit /results sidebar pick on the .out
        // file, which the trajectory inspector already eats first).
        match:       (file) => {
            const lower = file.toLowerCase();
            return lower.endsWith(".fdf")
                || lower.endsWith(".py")
                || lower.endsWith(".log")
                || lower.endsWith(".json")
                || lower.endsWith(".txt")
                || lower.endsWith(".md");
        },

        mount(host, file, ctx) {
            host.innerHTML = "";

            // ---- DOM scaffold ---------------------------------- //
            const card = document.createElement("section");
            card.className = "inspector-card source-card";

            const header = document.createElement("header");
            header.className = "inspector-card-header";
            const title = document.createElement("h2");
            title.className = "inspector-card-title";
            title.textContent = "Source — " + _basename(file);
            header.appendChild(title);

            // Status / progress strip + a Jump-to-end button.
            const controls = document.createElement("div");
            controls.className = "source-controls";
            const status = document.createElement("span");
            status.className = "source-status";
            status.textContent = "Loading…";
            controls.appendChild(status);
            const jumpEnd = document.createElement("button");
            jumpEnd.type = "button";
            jumpEnd.className = "source-jump-end";
            jumpEnd.textContent = "Jump to end";
            jumpEnd.disabled = true;
            controls.appendChild(jumpEnd);
            header.appendChild(controls);
            card.appendChild(header);

            const scroller = document.createElement("div");
            scroller.className = "source-scroller";
            const pre = document.createElement("pre");
            pre.className = "source-body";
            scroller.appendChild(pre);
            card.appendChild(scroller);

            host.appendChild(card);

            // ---- State ----------------------------------------- //
            //
            // ``mode`` tracks WHICH end of the file the user is
            // currently paging through.  Starts in "head" (forward
            // pagination from byte 0).  "Jump to end" switches to
            // "tail" mode where successive loads PREPEND earlier
            // bytes (so the user sees the last N bytes and can
            // scroll BACKWARDS through the file).
            //
            // Mixing the two within a single mount is intentionally
            // not supported -- switching modes mid-session would
            // require gap detection + a "..." separator UI that's
            // a v3 problem.

            const state = {
                mode:       "head",   // "head" | "tail"
                offset:     0,        // start byte of the first chunk
                length:     0,        // total bytes loaded so far
                fileSize:   null,
                disposed:   false,
                loading:    false,
                eof:        false,
                error:      null,
            };

            // ---- Range-fetch helper --------------------------- //
            //
            // ``ctx.readFile`` is a thin wrapper over /api/files/read
            // that only supports the all-or-nothing read.  For
            // range reads we go direct to the new endpoint.  The
            // ctx surface still owns the error envelope so a
            // future "swap fetch for cached fetch" still works
            // here -- we just don't have a registry-level helper
            // for range reads yet.

            function _fetchRange(offset, maxBytes) {
                const url = "/api/files/read_range?path="
                          + encodeURIComponent(file)
                          + "&offset="    + encodeURIComponent(offset)
                          + "&max_bytes=" + encodeURIComponent(maxBytes);
                return fetch(url, { credentials: "same-origin" })
                    .then((r) => r.json())
                    .catch((e) => ({
                        ok: false,
                        error: "network error: "
                             + (e && e.message ? e.message : String(e)),
                    }));
            }

            function _renderStatus() {
                if (state.error) {
                    status.textContent = "Error: " + state.error;
                    status.classList.add("inspector-inline-error");
                    return;
                }
                status.classList.remove("inspector-inline-error");
                if (state.fileSize === null) {
                    status.textContent = "Loading…";
                    return;
                }
                const kbLoaded = Math.round(state.length / 1024);
                const kbTotal  = Math.round(state.fileSize / 1024);
                const verb = state.mode === "tail"
                    ? "tail (latest)"
                    : "head";
                const pct = state.fileSize === 0
                    ? 100
                    : Math.round(100 * state.length / state.fileSize);
                status.textContent =
                    "Showing " + verb + ": " + kbLoaded + " KB of "
                    + kbTotal + " KB (" + pct + "%)";
            }

            function _loadInitial() {
                state.loading = true;
                _renderStatus();
                _fetchRange(0, PAGE_BYTES).then((r) => {
                    if (state.disposed) return;
                    state.loading = false;
                    if (!r.ok) {
                        state.error = r.error || "unknown read error";
                        _renderStatus();
                        return;
                    }
                    state.offset   = r.offset;
                    state.length   = r.length;
                    state.fileSize = r.file_size;
                    state.eof      = r.eof;
                    pre.textContent = r.text || "";
                    jumpEnd.disabled = state.eof;
                    _renderStatus();
                });
            }

            function _loadNextPage() {
                if (state.loading || state.eof) return;
                if (state.mode !== "head") return;
                state.loading = true;
                _renderStatus();
                const nextOffset = state.offset + state.length;
                _fetchRange(nextOffset, PAGE_BYTES).then((r) => {
                    if (state.disposed) return;
                    state.loading = false;
                    if (!r.ok) {
                        state.error = r.error || "read failed";
                        _renderStatus();
                        return;
                    }
                    // Append to the existing text.  Using textContent +
                    // string concatenation is correct for the
                    // happy path; for files in the 10s of MB this
                    // keeps node count at 1 (one large text node)
                    // rather than ballooning DOM with one node
                    // per page.
                    pre.textContent = pre.textContent + (r.text || "");
                    state.length += r.length;
                    state.eof    = r.eof;
                    jumpEnd.disabled = state.eof;
                    _renderStatus();
                });
            }

            function _loadTail() {
                if (state.loading) return;
                state.loading = true;
                state.mode    = "tail";
                state.eof     = true;
                _renderStatus();
                // Negative offset = last N bytes.  Server clamps if
                // file is smaller than the page size.
                _fetchRange(-PAGE_BYTES, PAGE_BYTES).then((r) => {
                    if (state.disposed) return;
                    state.loading = false;
                    if (!r.ok) {
                        state.error = r.error || "tail read failed";
                        _renderStatus();
                        return;
                    }
                    state.offset   = r.offset;
                    state.length   = r.length;
                    state.fileSize = r.file_size;
                    pre.textContent = r.text || "";
                    jumpEnd.disabled = true;   // already at the end
                    _renderStatus();
                    // Scroll to the bottom so the user lands on the
                    // most-recent line, not the start of the chunk.
                    scroller.scrollTop = scroller.scrollHeight;
                });
            }

            // ---- Scroll-driven auto-load ----------------------- //
            function _onScroll() {
                if (state.disposed) return;
                if (state.eof || state.loading) return;
                if (state.mode !== "head") return;
                const remaining = scroller.scrollHeight
                                - scroller.scrollTop
                                - scroller.clientHeight;
                if (remaining < NEXT_PAGE_TRIGGER_PX) {
                    _loadNextPage();
                }
            }
            scroller.addEventListener("scroll", _onScroll);
            jumpEnd.addEventListener("click", _loadTail);

            _loadInitial();

            return {
                dispose() {
                    state.disposed = true;
                    try { scroller.removeEventListener("scroll", _onScroll); }
                    catch (_) { /* ignore */ }
                    try { jumpEnd.removeEventListener("click", _loadTail); }
                    catch (_) { /* ignore */ }
                    host.innerHTML = "";
                },
            };
        },
    };

    const _basename = (window.molbuilder
                      && window.molbuilder.path
                      && window.molbuilder.path.basename)
                   || ((p) => p || "");

    root.molbuilder = root.molbuilder || {};
    root.molbuilder.inspectors = root.molbuilder.inspectors || {};
    root.molbuilder.inspectors.sourceInspector = inspector;
    if (root.molbuilder.inspectors.register) {
        root.molbuilder.inspectors.register(inspector);
    }
})(typeof window !== "undefined" ? window : this);
