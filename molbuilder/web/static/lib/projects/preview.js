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

// Server-side hard ceiling per /api/files/read; matches
// _MAX_READ_BYTES in molbuilder/web/blueprints/files.py.  Requesting
// less would cap us below the server's actual limit; requesting
// more rejects with 400.  Future tweaks to the server's ceiling
// should mirror here.
const READ_MAX_BYTES = 16 * 1024 * 1024;

function _emptyState() {
    return {
        path:         null,
        originalText: "",
        mtime:        null,
        size:         null,
        editable:     false,
        editing:      false,
        readError:    null,
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
 * Reads with the server's max budget so multi-MB result files
 * (SIESTA .out, large .json) don't trip a 413 by default.
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

    // Fetch with the server's max_bytes upfront so we don't trip
    // the default 1 MB cap on mid-sized files (typical SIESTA
    // .out is 2-8 MB).
    let body = null;
    let httpStatus = 0;
    try {
        const r = await fetch(
            "/api/files/read?path=" + encodeURIComponent(path)
            + "&max_bytes=" + READ_MAX_BYTES
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
        // Editability hint when we know the failure mode:
        //   * 413 / "exceeds max_bytes" → file too big to edit
        //     in-browser; suggest external editor.
        //   * 400 / "not valid UTF-8"   → binary content; v1 is
        //     text-only.
        if (httpStatus === 413 || /exceeds max_bytes/.test(reason)) {
            _setStatus(
                "File exceeds the in-browser edit budget "
                + `(${(READ_MAX_BYTES / (1024*1024)) | 0} MB).  Use an `
                + "external editor to modify this file.",
                null
            );
            _state.readError = "too_large";
        } else if (/not valid UTF-8/.test(reason)) {
            _setStatus("Binary content; this modal edits text only.",
                       null);
            _state.readError = "binary";
        } else {
            _state.readError = "other";
        }
        _renderUiFromState();
        return;
    }
    // Success.  Stash mtime + size for the edit-save contract.
    _state.originalText = body.text;
    _state.mtime        = body.mtime;
    _state.size         = body.size;
    _state.editable     = true;
    elBody.textContent  = body.text;
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
