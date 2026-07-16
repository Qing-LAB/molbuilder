/* projects/dialogs.js -- modal dialogs for sidebar mutations.
 *
 * 2026-06-12: created when the foldable <details> create-form sections
 * + the per-entry "view"/"×" inline buttons were replaced with a single
 * + dropdown + a kebab menu (see projects-sidebar.md § Mutation UX).
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
 * Each dialog runs the warning-modal pattern (save-dialog.js's idiom):
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
// not . / .. .  Mirrors save-dialog.js's _validate.
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
  opts = opts || {};
  const root = getProjectsRoot();
  if (!root) return Promise.resolve(null);

  const dialog = _mkDialog("molbuilder-projects-dest-dialog");
  dialog.appendChild(_mkHeader(opts.title || "Choose destination"));
  dialog.appendChild(_mkParagraph(
    "Click a folder to select.  Double-click to open it.",
    "molbuilder-projects-dialog-hint",
  ));

  const tree = document.createElement("div");
  tree.className = "molbuilder-projects-dest-tree";
  tree.setAttribute("role", "tree");
  dialog.appendChild(tree);

  const err = _mkErrorSlot();
  dialog.appendChild(err);

  let chosenPath = null;
  let confirmBtn;
  function _setChosen(path) {
    chosenPath = path;
    if (confirmBtn) confirmBtn.disabled = !path;
    // Mark the .is-selected node + clear others.
    tree.querySelectorAll(".is-selected").forEach(
      (n) => n.classList.remove("is-selected"));
    if (path) {
      const el = tree.querySelector(
        `[data-path="${path.replace(/"/g, '\\"')}"]`);
      if (el) el.classList.add("is-selected");
    }
  }

  // Lazy tree: each folder lists its children on first expand.
  async function _expand(node) {
    if (node.dataset.loaded === "1") return;
    node.dataset.loaded = "1";
    const path = node.dataset.path;
    const r = await apiList(path);
    if (!r || !r.ok) {
      const errLi = document.createElement("li");
      errLi.className = "molbuilder-projects-dest-error";
      errLi.textContent = r && r.error
        ? `Listing failed: ${r.error}`
        : "Listing failed.";
      const sub = node.querySelector("ul");
      if (sub) sub.appendChild(errLi);
      return;
    }
    const sub = node.querySelector("ul");
    // Filter to directories only — moving to a file doesn't make sense.
    const dirs = (r.entries || []).filter((e) => e.kind === "directory");
    if (dirs.length === 0) {
      const empty = document.createElement("li");
      empty.className = "molbuilder-projects-dest-empty";
      empty.textContent = "(no subdirectories)";
      sub.appendChild(empty);
      return;
    }
    dirs.forEach((d) => sub.appendChild(_buildNode(d.name,
      path.replace(/\/$/, "") + "/" + d.name)));
  }

  function _buildNode(name, path) {
    const li = document.createElement("li");
    li.className = "molbuilder-projects-dest-node";
    li.dataset.path = path;
    li.setAttribute("role", "treeitem");

    const row = document.createElement("div");
    row.className = "molbuilder-projects-dest-row";

    const tw = document.createElement("button");
    tw.type = "button";
    tw.className = "molbuilder-projects-dest-twisty";
    tw.textContent = "▸";
    tw.setAttribute("aria-label", "Expand");
    row.appendChild(tw);

    const label = document.createElement("span");
    label.className = "molbuilder-projects-dest-label";
    label.textContent = name;
    row.appendChild(label);

    li.appendChild(row);

    const sub = document.createElement("ul");
    sub.className = "molbuilder-projects-dest-children";
    sub.hidden = true;
    li.appendChild(sub);

    let expanded = false;
    async function toggle() {
      expanded = !expanded;
      sub.hidden = !expanded;
      tw.textContent = expanded ? "▾" : "▸";
      if (expanded) await _expand(li);
    }
    tw.addEventListener("click", (ev) => {
      ev.stopPropagation();
      toggle();
    });
    row.addEventListener("click", () => _setChosen(path));
    row.addEventListener("dblclick", () => {
      if (!expanded) toggle();
    });
    return li;
  }

  // Root node: projects/ itself.  Always open + expanded on mount
  // so the user sees the project list immediately.
  const rootUl = document.createElement("ul");
  rootUl.className = "molbuilder-projects-dest-root";
  tree.appendChild(rootUl);
  const rootNode = _buildNode("projects", root);
  rootUl.appendChild(rootNode);
  // Auto-expand the root.
  await _expand(rootNode);
  rootNode.querySelector(".molbuilder-projects-dest-children").hidden = false;
  rootNode.querySelector(".molbuilder-projects-dest-twisty").textContent = "▾";

  const actions = _mkActions([
    {
      label:   "Cancel",
      action:  "cancel",
      onClick: () => _settle(null),
    },
    {
      label:   opts.confirmLabel || "Choose",
      action:  "confirm",
      cls:     "is-primary",
      onClick: () => { if (chosenPath) _settle(chosenPath); },
    },
  ]);
  confirmBtn = actions.querySelector('[data-action="confirm"]');
  confirmBtn.disabled = true;
  dialog.appendChild(actions);

  return _open(dialog);
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

// ─── Test seam ─────────────────────────────────────────────────────── //

export function _resetDialogsForTests() {
  if (_active) _settle(null);
}

export function _isAnyDialogOpen() {
  return _active !== null;
}
