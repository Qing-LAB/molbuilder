/* projects/preview.js -- file preview modal with view + edit + save.
 *
 * Task #302 (2026-06-09): the modal previously rendered file
 * contents read-only with a disabled Save button.  The save endpoint
 * (/api/files/write with expected_mtime) had been shipped for
 * months; only the UI wiring was missing.  Edit + Save are now
 * functional with the safe-overwrite contract (mtime-based
 * conflict detection).
 *
 * Task #310 (2026-06-09): the read-only viewer was a plain ``<pre>``
 * that accumulated every loaded chunk in DOM — a 500 MB log
 * scrolled-through-fully would put 500 MB of text into the page
 * and kill the browser.  The editable mode was a ``<textarea>``
 * with no find / no line numbers / no jump-to-line.  Both have
 * been replaced with a single CodeMirror 5 instance: virtual
 * scrolling caps DOM memory at the visible window regardless of
 * file size, the search addon gives Ctrl-F over the whole loaded
 * doc, the jump-to-line addon gives Alt-G "go to line", and edit
 * mode is the same editor with ``readOnly`` toggled off.  The
 * vendored bundle is lazy-loaded on first open so other tabs pay
 * no JS cost.
 *
 * Large-file handling:
 *   * Read requests with ``max_bytes`` = server cap (16 MB) so
 *     mid-sized result files load without a "file is N bytes;
 *     exceeds max_bytes" 413.
 *   * Files past EDIT_MAX_BYTES (32 MB) load in paginated chunks
 *     via /api/files/read_range and Edit is disabled with a clear
 *     "use external editor" hint.
 *   * Non-UTF-8 files (400 from /api/files/read) similarly
 *     disable Edit — the v1 contract is text-only.
 *
 * Concurrent-write safety: on read, the response's ``mtime`` is
 * captured.  Save sends ``expected_mtime`` back; the server
 * returns 409 when disk-mtime has changed in the meantime.  On
 * 409 the user gets a "file changed on disk; reload to see the
 * new content" prompt instead of silently clobbering someone
 * else's edits.
 *
 * Spec: docs/protocols/selection.md § Preview modal.
 */

import { projects } from "./state.js";

// Modal DOM handles, populated by initPreview().
let elModal, elTitle, elMeta, elCmView, elError, elStatus;
let elEditBtn, elSaveBtn;

// CodeMirror instance, created lazily on first showPreview() via
// _ensureCmMounted().  Reused across opens — much cheaper than
// teardown + remount, and the modal's lifetime is the page's.
let _cm = null;

// Per-session state for the currently-open file.
let _state = _emptyState();

// CodeMirror loader — lazy + cached.  Returns the global
// ``window.CodeMirror`` once the vendored bundle (core + addons +
// dialog + search + jump-to-line) is fully loaded.  Concurrent
// callers receive the same promise.
let _cmLoaderPromise = null;

// Edit budget — files this size or smaller load wholesale via
// /api/files/read into the editor and are editable.  Files
// LARGER than this are loaded in paginated chunks via
// /api/files/read_range and Edit is disabled with a "use external
// editor" hint — even with CodeMirror's virtual scroll, a 100 MB
// save would block the UI for seconds on JSON-encode + transfer,
// and the dominant workflow above this cap is read-only log
// inspection rather than hand-editing.
//
// Server's hard ceiling is _MAX_READ_BYTES = 16 MB on
// /api/files/read.  We raise the in-modal edit cap to 32 MB so
// the post-#302 cap is more generous, but bulk-read still has
// to hit the server's ceiling; we chunk above that.
const EDIT_MAX_BYTES = 32 * 1024 * 1024;

// Bulk-read uses the server's hard ceiling so single-shot
// requests carry as much as the API permits.
const BULK_READ_MAX_BYTES = 16 * 1024 * 1024;

// Per-chunk window when streaming in paginated mode.  256 KB is
// the source viewer's default; mirror it so the source viewer
// and the modal feel identical when reading the same file.
const PAGE_BYTES = 256 * 1024;

// Vendored bundle root.  Files committed under
// molbuilder/web/static/vendor/codemirror/ — LICENSE included.
const CM_VENDOR_BASE = "/static/vendor/codemirror/";

function _emptyState() {
    return {
        path:          null,
        originalText:  "",
        mtime:         null,
        size:          null,
        editable:      false,
        editing:       false,
        readError:     null,
        // Paginated-view bookkeeping for files larger than
        // ``EDIT_MAX_BYTES``.  ``mode`` distinguishes bulk vs.
        // range; ``loadedBytes`` tracks how much of the file is
        // currently in the editor; ``loadingMore`` blocks
        // overlapping fetches when the user scrolls fast.
        mode:          "bulk",
        loadedBytes:   0,
        loadingMore:   false,
        eof:           true,
    };
}

function _onKeydown(ev) {
    if (ev.key === "Escape") tryCloseModal();
}

export function openPreviewModal() {
    if (!elModal) return;
    elModal.hidden = false;
    document.addEventListener("keydown", _onKeydown);
}

export function closePreviewModal() {
    if (!elModal) return;
    elModal.hidden = true;
    document.removeEventListener("keydown", _onKeydown);
    _state = _emptyState();
    _renderUiFromState();
}

/**
 * Close the modal, prompting first if the editor has unsaved
 * changes.  Wired to Esc / × / Close.
 */
function tryCloseModal() {
    if (_state.editing && _isDirty()) {
        const ok = window.confirm(
            "You have unsaved edits in this file.  Close anyway?\n\n"
            + "Click Cancel to stay and Save first."
        );
        if (!ok) return;
    }
    closePreviewModal();
}

function _isDirty() {
    if (!_cm) return false;
    return _cm.getValue() !== _state.originalText;
}

function _setStatus(message, kind /* "ok" | "dirty" | null */) {
    if (!elStatus) return;
    elStatus.textContent = message || "";
    elStatus.className = "ps-preview-status"
        + (kind === "ok"    ? " is-ok"    : "")
        + (kind === "dirty" ? " is-dirty" : "");
}

/* ---------- CodeMirror loading + mount ---------- */

function _injectStylesheet(href) {
    return new Promise((resolve, reject) => {
        const link = document.createElement("link");
        link.rel  = "stylesheet";
        link.href = href;
        link.onload  = () => resolve();
        link.onerror = () => reject(
            new Error("Could not load stylesheet: " + href));
        document.head.appendChild(link);
    });
}

function _injectScript(src) {
    return new Promise((resolve, reject) => {
        const s = document.createElement("script");
        s.src = src;
        s.async = false;  // preserve evaluation order vs. previous calls
        s.onload  = () => resolve();
        s.onerror = () => reject(
            new Error("Could not load script: " + src));
        document.head.appendChild(s);
    });
}

/**
 * Lazy-load the vendored CodeMirror 5 bundle + addons.  Cached
 * promise so concurrent showPreview() calls don't trigger
 * duplicate fetches.  Returns ``window.CodeMirror`` once
 * everything is parsed.
 *
 * Load order matters:
 *   1. CSS (parallel)
 *   2. CM core (must finish before addons execute — addons
 *      register themselves on ``CodeMirror.commands`` /
 *      ``CodeMirror.defineOption`` at parse time)
 *   3. dialog addon (search + jumpToLine both need its
 *      ``cm.openDialog`` method)
 *   4. searchcursor addon (search needs it)
 *   5. search + jumpToLine (both depend on the above)
 */
async function _loadCodeMirror() {
    if (window.CodeMirror) return window.CodeMirror;
    if (_cmLoaderPromise)  return _cmLoaderPromise;
    _cmLoaderPromise = (async () => {
        await Promise.all([
            _injectStylesheet(CM_VENDOR_BASE + "codemirror.min.css"),
            _injectStylesheet(CM_VENDOR_BASE + "dialog.min.css"),
            _injectScript(CM_VENDOR_BASE + "codemirror.min.js"),
        ]);
        await _injectScript(CM_VENDOR_BASE + "dialog.min.js");
        await _injectScript(CM_VENDOR_BASE + "searchcursor.min.js");
        await Promise.all([
            _injectScript(CM_VENDOR_BASE + "search.min.js"),
            _injectScript(CM_VENDOR_BASE + "jump-to-line.min.js"),
        ]);
        if (!window.CodeMirror) {
            throw new Error(
                "CodeMirror bundle loaded but the global is missing — "
                + "vendored bundle may be corrupt"
            );
        }
        return window.CodeMirror;
    })();
    return _cmLoaderPromise;
}

/**
 * Create the editor instance on first need.  Mounted into
 * ``#ps-preview-cmview``.  Read-only by default; ``startEdit``
 * flips ``readOnly`` off.
 *
 * Test hook: ``elModal.__molbuilder_test_cm`` exposes the instance
 * so Playwright E2E tests can drive setValue / execCommand
 * without scraping CM-internal DOM.
 */
async function _ensureCmMounted() {
    if (_cm) return _cm;
    const CM = await _loadCodeMirror();
    elCmView.textContent = "";  // drop any placeholder text
    _cm = CM(elCmView, {
        value:        "",
        readOnly:     true,
        lineNumbers:  true,
        lineWrapping: false,
        // ``viewportMargin`` defaults to 10; bump slightly so
        // scrolling is smoother but virtual-render still caps
        // memory.
        viewportMargin: 50,
        autofocus:    false,
    });
    _cm.on("change", _onCmChange);
    _cm.on("scroll", _onCmScroll);
    elModal.__molbuilder_test_cm = _cm;
    return _cm;
}

function _onCmChange() {
    // Programmatic setValue() also triggers change; only treat as
    // dirty when the user is editing.
    if (!_state.editing) return;
    if (_isDirty()) {
        elSaveBtn.disabled = false;
        _setStatus("Unsaved changes", "dirty");
    } else {
        elSaveBtn.disabled = true;
        _setStatus("", null);
    }
}

function _onCmScroll() {
    if (_state.mode !== "paginated"
        || _state.eof || _state.loadingMore) {
        return;
    }
    const info = _cm.getScrollInfo();
    // Trigger 200 px before the bottom — same threshold as the
    // source viewer (lib/inspectors/source.js).
    const near = info.top + info.clientHeight >= info.height - 200;
    if (near && _state.path) {
        _fetchNextRangeChunk(_state.path);
    }
}

/**
 * Re-render the UI from ``_state``.  Single re-paint point so the
 * View ↔ Edit ↔ Saved transitions all go through the same
 * function — no scattered ``el.hidden = ...`` calls.
 */
function _renderUiFromState() {
    if (!elModal) return;
    if (!_cm) return;
    if (_state.editing) {
        _cm.setOption("readOnly", false);
        elCmView.classList.add("is-editing");
        // Disable Edit when already editing; enable Save when
        // the editor has unsaved changes.
        elEditBtn.disabled = true;
        elSaveBtn.disabled = !_isDirty();
    } else {
        _cm.setOption("readOnly", true);
        elCmView.classList.remove("is-editing");
        // Edit is disabled in view mode when the file isn't
        // editable (read error, binary, oversized).
        elEditBtn.disabled = !_state.editable;
        elSaveBtn.disabled = true;
    }
}

/**
 * Switch the modal from view to edit mode.  Flips CodeMirror's
 * ``readOnly`` off, focuses the editor, places the caret at the
 * end so the user can append immediately.
 */
function startEdit() {
    if (!_state.editable) return;
    _state.editing = true;
    _renderUiFromState();
    _setStatus("", null);
    elError.textContent = "";
    _cm.focus();
    // Caret at end of doc.
    const lastLine = _cm.lastLine();
    const lastCh   = _cm.getLine(lastLine).length;
    _cm.setCursor({ line: lastLine, ch: lastCh });
}

/**
 * Persist the editor's current content via /api/files/write.
 * Sends ``expected_mtime`` for safe-overwrite detection.  On a
 * 409 mtime mismatch the user is informed that the file changed
 * on disk; they can reload to see the new bytes (re-open the
 * preview).
 */
async function saveEdit() {
    if (!_state.editing || !_state.path) return;
    if (!_isDirty()) {
        _setStatus("Nothing to save.", null);
        return;
    }
    const body = {
        path:           _state.path,
        text:           _cm.getValue(),
        expected_mtime: _state.mtime,
    };
    elError.textContent = "";
    _setStatus("Saving…", null);
    elSaveBtn.disabled = true;
    let resp;
    try {
        const r = await fetch("/api/files/write", {
            method:  "POST",
            headers: { "Content-Type": "application/json" },
            body:    JSON.stringify(body),
        });
        resp = await r.json();
        if (!r.ok || !resp.ok) {
            // 409 = mtime mismatch (someone else edited).  Other
            // errors = path rejected, parent missing, etc.  The
            // server's message is user-facing.
            elError.textContent = resp && resp.error
                ? resp.error
                : `Save failed (HTTP ${r.status}).`;
            _setStatus("", null);
            elSaveBtn.disabled = !_isDirty();
            return;
        }
    } catch (e) {
        elError.textContent = "Network error: "
            + (e && e.message ? e.message : String(e));
        _setStatus("", null);
        elSaveBtn.disabled = !_isDirty();
        return;
    }
    // Success: lock in the new mtime + originalText.  Keep the
    // editor in edit mode so the user can keep typing; dirty
    // tracking now reads clean because originalText matches.
    _state.originalText = _cm.getValue();
    _state.mtime        = resp.mtime;
    _state.size         = resp.size;
    _renderUiFromState();
    const now = new Date();
    _setStatus(
        "Saved at " + now.toLocaleTimeString(),
        "ok"
    );
}

/**
 * Open the preview modal showing the contents of the currently-
 * selected file (sessionStorage.molbuilder.current_file).
 *
 * Two read paths:
 *   * Size ≤ EDIT_MAX_BYTES → bulk /api/files/read into the
 *     editor; full edit + save round-trip available.
 *   * Size  > EDIT_MAX_BYTES → paginated /api/files/read_range
 *     into the editor (read-only); Edit is disabled with a clear
 *     "use external editor" hint.  Scroll-driven append fetches
 *     the next chunk; CodeMirror's virtual scroll keeps DOM
 *     memory bounded regardless of how much of the file is loaded.
 */
export async function showPreview() {
    if (!elModal) return;
    const path = projects.getCurrentFile();
    if (!path) return;
    _state = _emptyState();
    _state.path = path;
    elTitle.textContent = path.split("/").pop();
    elMeta.textContent  = path;
    elError.textContent = "";
    _setStatus("Loading…", null);
    openPreviewModal();

    // Mount CodeMirror on first open — costs ~200 KB of vendored
    // JS that pages without an open preview never pay.
    try {
        await _ensureCmMounted();
    } catch (e) {
        elError.textContent = "Could not load viewer: "
            + (e && e.message ? e.message : String(e));
        return;
    }
    _cm.setValue("");
    // Container size may have changed since the last open; CM
    // measures on mount but needs a refresh if the modal was
    // hidden during create.
    _cm.refresh();
    _renderUiFromState();

    // Step 1: cheap stat to find the file size before deciding
    // bulk vs. paginated.  ``/api/files/stat`` returns the size
    // without reading any bytes; for files past the bulk cap the
    // stat call is the only way to avoid wasting a /read request
    // that'll 413.
    let size = null;
    let mtime = null;
    try {
        const r = await fetch(
            "/api/files/stat?path=" + encodeURIComponent(path)
        );
        const body = await r.json();
        if (!r.ok || !body.ok) {
            elError.textContent = (body && body.error)
                || `Could not stat file (HTTP ${r.status}).`;
            _state.readError = "stat";
            _renderUiFromState();
            return;
        }
        size  = body.size;
        mtime = body.mtime;
    } catch (e) {
        elError.textContent = "Network error: "
            + (e && e.message ? e.message : String(e));
        _state.readError = "network";
        _renderUiFromState();
        return;
    }
    _state.size  = size;
    _state.mtime = mtime;

    if (size <= EDIT_MAX_BYTES) {
        await _loadBulk(path);
    } else {
        await _loadPaginated(path, size);
    }
}

/**
 * Bulk-read path: ``/api/files/read`` with the server's max
 * budget.  Editable round-trip available afterwards.
 */
async function _loadBulk(path) {
    _state.mode = "bulk";
    let body = null;
    let httpStatus = 0;
    try {
        const r = await fetch(
            "/api/files/read?path=" + encodeURIComponent(path)
            + "&max_bytes=" + BULK_READ_MAX_BYTES
        );
        httpStatus = r.status;
        body = await r.json();
    } catch (e) {
        elError.textContent = "Network error reading file: "
            + (e && e.message ? e.message : String(e));
        _state.readError = "network";
        _renderUiFromState();
        return;
    }
    if (!body || !body.ok) {
        const reason = (body && body.error) || `HTTP ${httpStatus}`;
        elError.textContent = reason;
        if (/not valid UTF-8/.test(reason)) {
            _setStatus("Binary content; this modal edits text only.",
                       null);
            _state.readError = "binary";
        } else {
            _state.readError = "other";
        }
        _renderUiFromState();
        return;
    }
    // Success.  Stash mtime + size for the edit-save contract;
    // refresh mtime because the file may have changed between
    // stat and read (the safe-overwrite check uses the read
    // mtime).
    _state.originalText = body.text;
    _state.mtime        = body.mtime;
    _state.size         = body.size;
    _state.editable     = true;
    _state.loadedBytes  = body.size;
    _state.eof          = true;
    _cm.setValue(body.text);
    _setStatus("", null);
    _renderUiFromState();
}

/**
 * Paginated path for files past ``EDIT_MAX_BYTES``: fetch the
 * first ``PAGE_BYTES`` window via /api/files/read_range, append
 * more chunks as the user scrolls near the bottom of the editor.
 * Edit is disabled (above-cap files are read-only).  Tips for
 * find / jump-to-line surface in the status line because the
 * default keybindings (Ctrl-F / Alt-G) are not discoverable.
 */
async function _loadPaginated(path, totalSize) {
    _state.mode        = "paginated";
    _state.editable    = false;
    _state.eof         = false;
    _state.loadedBytes = 0;
    _cm.setValue("");
    const sizeMB    = (totalSize / (1024 * 1024)).toFixed(1);
    const capMB     = (EDIT_MAX_BYTES / (1024 * 1024)) | 0;
    _setStatus(
        `Large file (${sizeMB} MB > ${capMB} MB edit cap) — `
        + "viewing only.  Use an external editor to modify.  "
        + "Tip: Ctrl-F to find, Alt-G to jump to a line.",
        null
    );
    await _fetchNextRangeChunk(path);
}

async function _fetchNextRangeChunk(path) {
    if (_state.loadingMore || _state.eof) return;
    _state.loadingMore = true;
    let body = null;
    try {
        const r = await fetch(
            "/api/files/read_range?path=" + encodeURIComponent(path)
            + "&offset="    + _state.loadedBytes
            + "&max_bytes=" + PAGE_BYTES
        );
        body = await r.json();
        if (!r.ok || !body.ok) {
            elError.textContent = (body && body.error)
                || `Range read failed (HTTP ${r.status}).`;
            _state.readError = "range";
            _renderUiFromState();
            return;
        }
    } catch (e) {
        elError.textContent = "Network error: "
            + (e && e.message ? e.message : String(e));
        _state.readError = "network";
        _renderUiFromState();
        return;
    } finally {
        _state.loadingMore = false;
    }
    // Append the chunk to the end of the editor doc.  CM5 accepts
    // ``{line: Infinity}`` and clamps to the last position.
    _cm.replaceRange(body.text, { line: Infinity, ch: 0 });
    _state.loadedBytes += body.length;
    _state.eof          = !!body.eof;
    _state.mtime        = body.mtime;
    _renderUiFromState();
}

export function initPreview() {
    elModal = document.getElementById("ps-preview-modal");
    if (!elModal) return;
    elTitle   = document.getElementById("ps-preview-title");
    elMeta    = document.getElementById("ps-preview-meta");
    elCmView  = document.getElementById("ps-preview-cmview");
    elError   = document.getElementById("ps-preview-error");
    elStatus  = document.getElementById("ps-preview-status");
    elEditBtn = document.getElementById("ps-preview-edit-btn");
    elSaveBtn = document.getElementById("ps-preview-save-btn");

    // Close handlers: backdrop click, header × button, footer Close.
    elModal.querySelectorAll(
        ".ps-preview-close, .ps-preview-close-footer, .ps-preview-backdrop"
    ).forEach((n) => n.addEventListener("click", tryCloseModal));

    if (elEditBtn) elEditBtn.addEventListener("click", startEdit);
    if (elSaveBtn) elSaveBtn.addEventListener("click", saveEdit);
}
