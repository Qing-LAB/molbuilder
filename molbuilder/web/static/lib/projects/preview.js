/* projects/preview.js -- file preview modal with view + edit + save.
 *
 * Task #302 (2026-06-09): the modal previously rendered file
 * contents read-only with a disabled Save button.  The save endpoint
 * (/api/files/write with expected_mtime) had been shipped for
 * months; only the UI wiring was missing.  Edit + Save are now
 * functional with the safe-overwrite contract (mtime-based
 * conflict detection).
 *
 * Large-file handling:
 *   * Read requests with ``max_bytes`` = server cap (16 MB) so
 *     mid-sized result files load without a "file is N bytes;
 *     exceeds max_bytes" 413.
 *   * Files past the 16 MB server cap surface a clear error +
 *     hint to use an external editor; the modal stays in view
 *     mode with the Edit button disabled.
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
let elModal, elTitle, elMeta, elBody, elEdit, elError, elStatus;
let elEditBtn, elSaveBtn;

// Per-session state for the currently-open file.
let _state = _emptyState();

// Edit budget — files this size or smaller load wholesale via
// /api/files/read into the <textarea> and are editable.  Files
// LARGER than this are loaded in paginated chunks via
// /api/files/read_range (matching the source viewer on /results)
// and Edit is disabled with a "use external editor" hint —
// textareas degrade past ~30-50 MB and a 100 MB save would block
// the UI for seconds on encode + transfer.
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
        // currently in the <pre>; ``loadingMore`` blocks
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
    if (elBody) elBody.removeEventListener("scroll", _onBodyScroll);
    _state = _emptyState();
    _renderUiFromState();
}

/**
 * Close the modal, prompting first if the textarea has unsaved
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
    if (!elEdit) return false;
    return elEdit.value !== _state.originalText;
}

function _setStatus(message, kind /* "ok" | "dirty" | null */) {
    if (!elStatus) return;
    elStatus.textContent = message || "";
    elStatus.className = "ps-preview-status"
        + (kind === "ok"    ? " is-ok"    : "")
        + (kind === "dirty" ? " is-dirty" : "");
}

/**
 * Re-render the UI from ``_state``.  Single re-paint point so the
 * View ↔ Edit ↔ Saved transitions all go through the same
 * function — no scattered ``el.hidden = ...`` calls.
 */
function _renderUiFromState() {
    if (!elModal) return;
    if (_state.editing) {
        elBody.hidden = true;
        elEdit.hidden = false;
        // Disable Edit when already editing; enable Save when
        // the textarea has unsaved changes.
        elEditBtn.disabled = true;
        elSaveBtn.disabled = !_isDirty();
    } else {
        elBody.hidden = false;
        elEdit.hidden = true;
        // Edit is disabled in view mode when the file isn't
        // editable (read error, binary, oversized).
        elEditBtn.disabled = !_state.editable;
        elSaveBtn.disabled = true;
    }
}

function _onEditInput() {
    // ``input`` fires on every keystroke / paste / undo.  Update
    // the Save-enable + status line; no re-render of the
    // textarea itself (the browser owns that).
    if (_state.editing) {
        elSaveBtn.disabled = !_isDirty();
        if (_isDirty()) {
            _setStatus("Unsaved changes", "dirty");
        } else {
            _setStatus("", null);
        }
    }
}

/**
 * Switch the modal from view to edit mode.  Populates the
 * textarea from the current originalText (so the user starts
 * from what they just read, not whatever the textarea contained
 * before), focuses it, and updates button states.
 */
function startEdit() {
    if (!_state.editable) return;
    elEdit.value = _state.originalText;
    _state.editing = true;
    _renderUiFromState();
    _setStatus("", null);
    elError.textContent = "";
    // Focus + place caret at the end; small but the user expects
    // their typing to land somewhere immediately.
    elEdit.focus();
    try {
        elEdit.setSelectionRange(
            elEdit.value.length, elEdit.value.length);
    } catch (_) { /* setSelectionRange unsupported on some legacy */ }
}

/**
 * Persist the textarea's current content via /api/files/write.
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
        text:           elEdit.value,
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
    // Success: lock in the new mtime + originalText, drop back
    // to view mode (the user explicitly stays in Edit if they
    // want to continue typing — but Save behaviour matches "save
    // + keep editing" by default).  Keep editing flag on so the
    // textarea stays focused; just update originalText so the
    // dirty check now reads clean.
    _state.originalText = elEdit.value;
    _state.mtime        = resp.mtime;
    _state.size         = resp.size;
    // Reflect the new content in the read-only body too, so the
    // user who toggles back to View sees their saved bytes.
    elBody.textContent  = _state.originalText;
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
 *     <textarea>; full edit + save round-trip available.
 *   * Size  > EDIT_MAX_BYTES → paginated /api/files/read_range
 *     into the <pre> only; Edit is disabled with a clear "use
 *     external editor" hint.  Scroll-driven append fetches the
 *     next chunk; explicit "Load more" not provided yet (rare;
 *     the source viewer's UX is the model).
 */
export async function showPreview() {
    if (!elModal) return;
    const path = projects.getCurrentFile();
    if (!path) return;
    _state = _emptyState();
    _state.path = path;
    elTitle.textContent = path.split("/").pop();
    elMeta.textContent  = path;
    elBody.textContent  = "Loading…";
    elEdit.value        = "";
    elError.textContent = "";
    _setStatus("", null);
    _renderUiFromState();
    openPreviewModal();

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
            elBody.textContent  = "";
            elError.textContent = (body && body.error)
                || `Could not stat file (HTTP ${r.status}).`;
            _state.readError = "stat";
            _renderUiFromState();
            return;
        }
        size  = body.size;
        mtime = body.mtime;
    } catch (e) {
        elBody.textContent  = "";
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
        elBody.textContent  = "";
        elError.textContent = "Network error reading file: "
            + (e && e.message ? e.message : String(e));
        _state.readError = "network";
        _renderUiFromState();
        return;
    }
    if (!body || !body.ok) {
        elBody.textContent = "";
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
    elBody.textContent  = body.text;
    _renderUiFromState();
}

/**
 * Paginated path for files past ``EDIT_MAX_BYTES``: fetch the
 * first ``PAGE_BYTES`` window via /api/files/read_range, append
 * more chunks as the user scrolls near the bottom of the <pre>.
 * Edit is disabled because textareas degrade past ~30 MB and a
 * full-content POST is hostile UX on a 100 MB file.
 */
async function _loadPaginated(path, totalSize) {
    _state.mode        = "paginated";
    _state.editable    = false;
    _state.eof         = false;  // override the _emptyState default
    _state.loadedBytes = 0;
    elBody.textContent = "";     // clear the "Loading…" placeholder
    const sizeMB    = (totalSize / (1024 * 1024)).toFixed(1);
    const capMB     = (EDIT_MAX_BYTES / (1024 * 1024)) | 0;
    _setStatus(
        `Large file (${sizeMB} MB > ${capMB} MB edit cap) — `
        + "viewing only.  Use an external editor to modify.",
        null
    );
    await _fetchNextRangeChunk(path);
    // Wire the scroll listener for "near-bottom → load more".
    // Only attached in paginated mode; bulk mode has the whole
    // file already and doesn't need this.
    elBody.addEventListener("scroll", _onBodyScroll);
}

function _onBodyScroll() {
    if (_state.mode !== "paginated" || _state.eof
        || _state.loadingMore) {
        return;
    }
    // Trigger 200 px before the bottom — same threshold the
    // source viewer uses (lib/inspectors/source.js).
    const near = elBody.scrollTop + elBody.clientHeight
        >= elBody.scrollHeight - 200;
    if (near && _state.path) {
        _fetchNextRangeChunk(_state.path);
    }
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
    // Append the chunk text to whatever's already rendered.
    elBody.textContent += body.text;
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
    elBody    = document.getElementById("ps-preview-body");
    elEdit    = document.getElementById("ps-preview-edit");
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
    if (elEdit)    elEdit.addEventListener("input", _onEditInput);
}
