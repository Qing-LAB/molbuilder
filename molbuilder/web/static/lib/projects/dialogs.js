/* projects/dialogs.js -- modal dialogs for sidebar mutations.
 *
 * 2026-06-12: created when the foldable <details> create-form sections
 * + the per-entry "view"/"×" inline buttons were replaced with a single
 * + dropdown + a kebab menu (see web/projects.md § 4).
 *
 * Each export opens a <dialog>, lets the user type / pick something,
 * and resolves with the chosen value (or null on cancel / ESC).  No
 * HTTP here — the caller dispatches to projects.{mkdir, move, copy,
 * rename, upload, createProject} after the dialog resolves.
 *
 * Single-instance: opening dialog A while dialog A is already open
 * returns the existing pending promise; opening dialog A while B is
 * open closes B and starts A.  No modal stacking.
 *
 * Each dialog runs the warning-modal pattern (`lib/warning-modal.js`):
 *   * ESC + Cancel + clicking off the modal all resolve to null
 *   * primary input gets initial focus
 *   * destructive actions (overwrite) default-focus on Cancel
 */

import { apiList } from "./api.js";
import { getProjectsRoot } from "./state.js";

// ─── Single-instance bookkeeping ─────────────────────────────────── //

let _active = null;

function _settle(value) {
  if (!_active) return;
  const { dialog, resolve } = _active;
  _active = null;
  try { dialog.close(); } catch (_) {}
  try {
    if (dialog.parentNode) dialog.parentNode.removeChild(dialog);
  } catch (_) {}
  resolve(value);
}

function _open(dialog) {
  // If a prior dialog is open, kill it (resolves to null so its
  // caller gets the "cancelled" signal).  No stacking — one modal
  // at a time keeps the user's focus + the ESC contract sane.
  if (_active) _settle(null);
  document.body.appendChild(dialog);
  const resolve = { fn: null };
  const promise = new Promise((res) => { resolve.fn = res; });
  _active = { dialog, resolve: resolve.fn };
  // Identity-guard the close/cancel listeners: ``dialog.close()`` fires ``close`` on a
  // QUEUED task, so when one dialog settles and the caller SYNCHRONOUSLY opens the next
  // (e.g. Copy: chooseDestinationDir -> chooseName), the FIRST dialog's late ``close``
  // must not resolve the SECOND (now-active) one to null and tear it down.  Only settle
  // when the event's own dialog is still the active one.
  dialog.addEventListener("cancel", () => {
    if (_active && _active.dialog === dialog) _settle(null);
  });
  dialog.addEventListener("close", () => {
    if (_active && _active.dialog === dialog) _settle(null);
  });
  try {
    if (typeof dialog.showModal === "function") dialog.showModal();
    else if (typeof dialog.show === "function") dialog.show();
  } catch (_) {}
  return promise;
}

// ─── Filename / dirname validation ───────────────────────────────── //

// Same rules as the rename endpoint: basename-only, no separators,
// not . / .. .  (The rule came from `modify/structure/save-dialog.js`,
// retired 2026-09-02 when the Save panel moved onto this module's own
// `chooseSavePath`; this is now its only home.)
function _validateName(raw) {
  const s = (raw || "").trim();
  if (!s) return { ok: false, reason: "" };
  if (s.indexOf("/") !== -1 || s.indexOf("\\") !== -1) {
    return {
      ok: false,
      reason: "Name cannot contain '/' or '\\\\'.",
    };
  }
  if (s === "." || s === "..") {
    return { ok: false, reason: "Reserved name." };
  }
  return { ok: true, value: s };
}

// ─── DOM scaffolding ─────────────────────────────────────────────── //

function _mkDialog(cls) {
  const dlg = document.createElement("dialog");
  dlg.className = "molbuilder-projects-dialog " + cls;
  return dlg;
}

function _mkHeader(text) {
  const h = document.createElement("h2");
  h.textContent = text;
  return h;
}

function _mkParagraph(text, cls) {
  const p = document.createElement("p");
  if (cls) p.className = cls;
  p.textContent = text;
  return p;
}

function _mkErrorSlot() {
  const p = document.createElement("p");
  p.className = "molbuilder-projects-dialog-error";
  p.setAttribute("data-role", "error");
  p.hidden = true;
  return p;
}

function _mkLabeledInput(labelText, opts) {
  opts = opts || {};
  const wrap = document.createElement("label");
  wrap.className = "molbuilder-projects-dialog-field";

  const span = document.createElement("span");
  span.textContent = labelText;
  wrap.appendChild(span);

  const input = document.createElement("input");
  input.type = opts.type || "text";
  input.value = opts.value || "";
  input.setAttribute("autocomplete", "off");
  input.setAttribute("spellcheck", "false");
  if (opts.placeholder) input.placeholder = opts.placeholder;
  if (opts.dataRole)    input.setAttribute("data-role", opts.dataRole);
  wrap.appendChild(input);

  return { wrap, input };
}

function _mkActions(buttons) {
  const row = document.createElement("div");
  row.className = "molbuilder-projects-dialog-actions";
  buttons.forEach((b) => {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.textContent = b.label;
    btn.setAttribute("data-action", b.action);
    if (b.cls) btn.className = b.cls;
    btn.addEventListener("click", b.onClick);
    row.appendChild(btn);
  });
  return row;
}

// ─── chooseFolderName / choose project ───────────────────────────── //

/**
 * Generic "name + Cancel/OK" dialog.  Used by:
 *   * new project (intent="project")
 *   * new folder  (intent="folder", contextDir provided)
 *   * rename      (intent="rename", initial = current basename)
 *
 * Resolves to a non-empty string (validated basename) on OK or
 * null on cancel / ESC.
 *
 * @param {object} opts
 * @param {string} opts.title    — modal heading
 * @param {string} opts.label    — input label
 * @param {string} [opts.initial] — initial value (selected on focus)
 * @param {string} [opts.hint]   — small hint paragraph below input
 * @param {string} [opts.confirmLabel] — primary button text (default "OK")
 * @param {string} [opts.placeholder]
 */
export function chooseName(opts) {
  opts = opts || {};
  const dialog = _mkDialog("molbuilder-projects-name-dialog");
  dialog.appendChild(_mkHeader(opts.title || "Name"));
  if (opts.hint) {
    dialog.appendChild(_mkParagraph(opts.hint,
      "molbuilder-projects-dialog-hint"));
  }
  const { wrap, input } = _mkLabeledInput(opts.label || "Name", {
    value:       opts.initial || "",
    placeholder: opts.placeholder || "",
    dataRole:    "name",
  });
  dialog.appendChild(wrap);

  const err = _mkErrorSlot();
  dialog.appendChild(err);

  let confirmBtn;
  function _validate() {
    const v = _validateName(input.value);
    err.textContent = v.ok ? "" : v.reason;
    err.hidden = v.ok;
    if (confirmBtn) confirmBtn.disabled = !v.ok;
    return v;
  }

  const actions = _mkActions([
    {
      label:   "Cancel",
      action:  "cancel",
      onClick: () => _settle(null),
    },
    {
      label:   opts.confirmLabel || "OK",
      action:  "confirm",
      cls:     "is-primary",
      onClick: () => {
        const v = _validate();
        if (v.ok) _settle(v.value);
      },
    },
  ]);
  confirmBtn = actions.querySelector('[data-action="confirm"]');
  dialog.appendChild(actions);

  input.addEventListener("input", _validate);
  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
      e.preventDefault();
      const v = _validate();
      if (v.ok) _settle(v.value);
    }
  });

  const p = _open(dialog);
  try {
    input.focus();
    if (typeof input.select === "function") input.select();
  } catch (_) {}
  _validate();
  return p;
}

// ─── Upload dialog ───────────────────────────────────────────────── //

/**
 * Pick a file from disk + confirm upload to the current directory.
 * Resolves to the selected File on confirm or null on cancel.
 *
 * @param {object} opts
 * @param {string} opts.contextDir  — destination dir basename (for the hint)
 */
export function chooseUploadFile(opts) {
  opts = opts || {};
  const dialog = _mkDialog("molbuilder-projects-upload-dialog");
  dialog.appendChild(_mkHeader("Upload file"));
  if (opts.contextDir) {
    dialog.appendChild(_mkParagraph(
      `Uploaded into ${opts.contextDir}.`,
      "molbuilder-projects-dialog-hint",
    ));
  }

  const field = document.createElement("label");
  field.className = "molbuilder-projects-dialog-field";
  const span = document.createElement("span");
  span.textContent = "Pick a file";
  field.appendChild(span);
  const input = document.createElement("input");
  input.type = "file";
  input.setAttribute("data-role", "file");
  field.appendChild(input);
  dialog.appendChild(field);

  const err = _mkErrorSlot();
  dialog.appendChild(err);

  let confirmBtn;
  function _validate() {
    const ok = input.files && input.files.length > 0;
    err.textContent = ok ? "" : "";
    err.hidden = ok || !input.files;
    if (confirmBtn) confirmBtn.disabled = !ok;
  }

  const actions = _mkActions([
    {
      label:   "Cancel",
      action:  "cancel",
      onClick: () => _settle(null),
    },
    {
      label:   "Upload",
      action:  "confirm",
      cls:     "is-primary",
      onClick: () => {
        if (input.files && input.files.length > 0) {
          _settle(input.files[0]);
        }
      },
    },
  ]);
  confirmBtn = actions.querySelector('[data-action="confirm"]');
  dialog.appendChild(actions);

  input.addEventListener("change", _validate);

  const p = _open(dialog);
  try { input.focus(); } catch (_) {}
  _validate();
  return p;
}

// ─── Destination-dir picker (tree) ───────────────────────────────── //

/**
 * Show a tree of the projects directory; user picks a folder to
 * be the destination of a move/copy.  Resolves to the absolute
 * picked path, or null on cancel.
 *
 * @param {object} opts
 * @param {string} opts.title   — modal heading (e.g. "Move to…")
 * @param {string} opts.srcPath — source path (highlighted as the
 *                                "current location" if its parent
 *                                is in the tree; cannot be picked
 *                                as the destination)
 */
export async function chooseDestinationDir(opts) {
  // Delegates to the ONE pop-out picker (lib/tree-picker.js, promoted
  // from the implementation that lived here 2026-06-12..2026-08-28) --
  // this wrapper keeps the sidebar's question named for what it asks.
  opts = opts || {};
  const { pickPath } = await import("../tree-picker.js");
  return pickPath({
    title: opts.title || "Choose destination",
    hint: "Click a folder to select.  Double-click to open it.",
    mode: "dir",
    confirmLabel: opts.confirmLabel || "Choose",
  });
}

// ─── Confirm overwrite (for move + copy when destination matches) ── //

/**
 * Confirm a destructive action with the user.  Used for sidecar-
 * overwrite warnings + future bulk-action confirmations.  Resolves
 * to true on confirm, false on cancel.
 */
export function confirmDestructive(opts) {
  opts = opts || {};
  const dialog = _mkDialog("molbuilder-projects-confirm-dialog");
  dialog.appendChild(_mkHeader(opts.title || "Confirm"));
  if (opts.body) {
    dialog.appendChild(_mkParagraph(opts.body));
  }
  const actions = _mkActions([
    {
      label:   "Cancel",
      action:  "cancel",
      onClick: () => _settle(false),
    },
    {
      label:   opts.confirmLabel || "Proceed",
      action:  "confirm",
      cls:     "is-destructive",
      onClick: () => _settle(true),
    },
  ]);
  dialog.appendChild(actions);
  const p = _open(dialog);
  // Default focus on Cancel (the safe action) for destructive flows.
  try {
    const cancel = dialog.querySelector('[data-action="cancel"]');
    if (cancel) cancel.focus();
  } catch (_) {}
  return p;
}

/**
 * The one "save it WHERE, as WHAT" question — the move-to dialog composed
 * with the name dialog, from the project root, on any tab
 * (docs/web/projects.md § 5).
 *
 * A UNIFIED surface on purpose (user, 2026-08-19): every future flow that
 * deposits or consolidates artifacts under the project root — the MolView
 * exports today, a transport calculation assembling several results
 * tomorrow — asks this one question through this one door, so "where may
 * files go" never grows a second implementation.
 *
 * @param opts {title?, nameTitle?, initial?, hint?}
 * @returns "<dir>/<name>" (no extension appended), or null on cancel.
 * @throws when the project root is not available on this page yet — said,
 *         never a silent nothing.
 */
export async function chooseSavePath(opts) {
  opts = opts || {};
  if (!getProjectsRoot()) {
    throw new Error("the project tree is not available on this page yet");
  }
  const dir = await chooseDestinationDir({
    title: opts.title || "Save where?",
  });
  if (!dir) return null;
  const name = await chooseName({
    title: opts.nameTitle || "Save as",
    label: "Name",
    initial: opts.initial || "",
    hint: opts.hint || "",
  });
  if (!name) return null;
  return String(dir).replace(/\/+$/, "") + "/" + name;
}

// ─── Test seam ─────────────────────────────────────────────────────── //

export function _resetDialogsForTests() {
  if (_active) _settle(null);
}

export function _isAnyDialogOpen() {
  return _active !== null;
}
