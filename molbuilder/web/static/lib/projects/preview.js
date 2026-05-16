/* projects/preview.js -- file-preview modal (view: functional; save: stub).
 *
 * Self-contained modal widget.  Triggered by the per-entry "view"
 * button in list.js (which calls :func:`showPreview` directly).
 *
 * View is fully functional via /api/files/read (already shipped).
 * Save is UI-only: the button is visible-but-disabled with a
 * "coming soon" tooltip; the backend (/api/files/write) is shipped
 * but no UX path wires Save through it yet.
 *
 * Spec: docs/protocols/selection.md § Preview modal.
 */

import { projects } from "./state.js";

let elModal, elTitle, elMeta, elBody, elError;

function _onKeydown(ev) {
  if (ev.key === "Escape") closePreviewModal();
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
}

/**
 * Open the preview modal showing the contents of the currently-
 * selected file (sessionStorage.molbuilder.current_file).
 *
 * Reads via projects.readCurrentFile() (which wraps /api/files/read).
 * 413 (too large) / 400 (non-UTF-8) / 404 errors surface in the
 * modal's error slot.
 */
export async function showPreview() {
  if (!elModal) return;
  const path = projects.getCurrentFile();
  if (!path) return;
  elTitle.textContent = path.split("/").pop();
  elMeta.textContent  = path;
  elBody.textContent  = "Loading...";
  elError.textContent = "";
  openPreviewModal();
  const payload = await projects.readCurrentFile();
  if (!payload) {
    elBody.textContent = "";
    // readCurrentFile returns null on any failure (network or
    // non-ok body); re-fetch raw to surface the backend message.
    try {
      const r = await fetch(
        "/api/files/read?path=" + encodeURIComponent(path)
      );
      const j = await r.json();
      elError.textContent = j.error || "Failed to read file.";
    } catch (e) {
      elError.textContent = "Network error reading file: " + e.message;
    }
    return;
  }
  elBody.textContent = payload.text;
}

export function initPreview() {
  elModal = document.getElementById("ps-preview-modal");
  if (!elModal) return;
  elTitle = document.getElementById("ps-preview-title");
  elMeta  = document.getElementById("ps-preview-meta");
  elBody  = document.getElementById("ps-preview-body");
  elError = document.getElementById("ps-preview-error");
  // Close handlers: backdrop click, header × button, footer Close.
  elModal.querySelectorAll(
    ".ps-preview-close, .ps-preview-close-footer, .ps-preview-backdrop"
  ).forEach((n) => n.addEventListener("click", closePreviewModal));
}
