/* lib/tree-picker.js — the ONE pop-out path picker.
 *
 * Promoted 2026-08-28 from `projects/dialogs.js::chooseDestinationDir`
 * (user: reuse the picker window, don't reinvent — and make it a module
 * every surface can integrate).  One lazy-expanding tree over the
 * projects root, three consumers by design:
 *
 *   * the sidebar's Move/Copy destination question (dirs only) — the
 *     original caller, now delegating here;
 *   * any form field that names a file in the tree;
 *   * the Transport tab's SLOT selection (plans/transport-design.md
 *     § 4.1): pick a concluded attempt, with the metadata line read
 *     from the attempt's own `.fdf` — the deck that actually ran is
 *     the truth about a result (user, 2026-08-28), never a file this
 *     module invents.
 *
 * The metadata seam is `describe(path, entry)`: the caller supplies
 * what a selection MEANS (an fdf summary, a file size, nothing) and
 * this module only displays it.  Listing goes through the same fenced
 * `projects` API the sidebar uses; nothing here touches HTTP directly.
 *
 * Dialog chrome: the shared `molbuilder-projects-dialog` class
 * (lib/dialog.css, the ONE modal sheet — which also owns every `tp-*`
 * rule this file writes).  Single-instance like every dialog: opening
 * while open settles the previous one to null first.
 */

import { apiList } from "./projects/api.js";
import { getProjectsRoot } from "./projects/state.js";

let _active = null;

function _settle(value) {
  if (!_active) return;
  const { dialog, resolve } = _active;
  _active = null;
  try { dialog.close(); } catch (_) { /* already closed */ }
  dialog.remove();
  resolve(value);
}

function _el(tag, cls, text) {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (text !== undefined) node.textContent = text;
  return node;
}

/**
 * Open the picker and resolve with the chosen path (or null).
 *
 * @param {object} opts
 *   title         dialog heading
 *   hint          one explanatory line under the heading
 *   mode          "dir" (default): only directories listed + selectable;
 *                 "file": files listed and selectable, directories
 *                 expandable but not selectable;
 *                 "any": both listed, both selectable
 *   filter        optional (entry, path) => bool over listed entries —
 *                 e.g. only `run-*` directories, only `.fdf` files
 *   describe      optional async (path, entry) => string; shown in the
 *                 meta line when a selection lands (the fdf seam)
 *   confirmLabel  the primary button's label (default "Choose")
 * @returns {Promise<string|null>}
 */
export async function pickPath(opts) {
  opts = opts || {};
  const mode = opts.mode || "dir";
  const root = getProjectsRoot();
  if (!root) return null;
  if (_active) _settle(null);

  const dialog = _el("dialog",
    "molbuilder-projects-dialog molbuilder-tree-picker-dialog");
  dialog.appendChild(_el("h2", null, opts.title || "Choose"));
  dialog.appendChild(_el("p", "molbuilder-projects-dialog-hint",
    opts.hint || (mode === "dir"
      ? "Click a folder to select it.  ▸ expands."
      : "Click an entry to select it.  ▸ expands a folder.")));

  const tree = _el("div", "tp-tree");
  tree.setAttribute("role", "tree");
  dialog.appendChild(tree);

  // The meta line: what the current selection MEANS, when the caller
  // can say (the fdf-metadata seam for slot picking).
  const meta = _el("p", "tp-meta");
  meta.hidden = true;
  dialog.appendChild(meta);

  const err = _el("p", "molbuilder-projects-dialog-error");
  err.setAttribute("data-role", "error");
  err.hidden = true;
  dialog.appendChild(err);

  let chosenPath = null;
  let confirmBtn = null;
  let describeSeq = 0;

  async function _setChosen(path, entry) {
    chosenPath = path;
    if (confirmBtn) confirmBtn.disabled = !path;
    tree.querySelectorAll(".is-selected").forEach(
      (n) => n.classList.remove("is-selected"));
    if (path) {
      const node = tree.querySelector(
        `[data-path="${path.replace(/"/g, '\\"')}"]`);
      if (node) node.classList.add("is-selected");
    }
    if (!opts.describe || !path) { meta.hidden = true; return; }
    const seq = ++describeSeq;
    meta.hidden = false;
    meta.textContent = "Reading…";
    try {
      const text = await opts.describe(path, entry);
      if (seq !== describeSeq) return;      // a newer selection landed
      meta.textContent = text || "";
      meta.hidden = !text;
    } catch (e) {
      if (seq !== describeSeq) return;
      meta.textContent = "Could not read: "
        + (e && e.message ? e.message : String(e));
    }
  }

  const selectable = (kind) =>
    mode === "any" || (mode === "dir" ? kind === "directory"
                                      : kind === "file");

  function _buildNode(name, path, kind) {
    const li = _el("li", "tp-node");
    li.dataset.path = path;
    li.dataset.kind = kind;
    li.setAttribute("role", "treeitem");

    const row = _el("div", "tp-row");
    if (!selectable(kind)) row.classList.add("tp-row--inert");

    const isDir = kind === "directory";
    const tw = _el("button", "tp-twisty", isDir ? "▸" : "");
    tw.type = "button";
    tw.setAttribute("aria-label", "Expand");
    if (!isDir) tw.classList.add("tp-twisty--leaf");
    row.appendChild(tw);
    row.appendChild(_el("span", "tp-icon", isDir ? "📁" : "📄"));
    row.appendChild(_el("span", "tp-label", name));
    li.appendChild(row);

    const sub = _el("ul", "tp-children");
    sub.hidden = true;
    li.appendChild(sub);

    let expanded = false;
    async function toggle() {
      if (!isDir) return;
      expanded = !expanded;
      sub.hidden = !expanded;
      tw.textContent = expanded ? "▾" : "▸";
      li.classList.toggle("is-open", expanded);
      if (expanded) await _expand(li);
    }
    tw.addEventListener("click", (ev) => { ev.stopPropagation(); toggle(); });
    row.addEventListener("click", () => {
      if (selectable(kind)) _setChosen(path, { name, kind });
    });
    row.addEventListener("dblclick", () => { if (!expanded) toggle(); });
    return li;
  }

  async function _expand(node) {
    if (node.dataset.loaded === "1") return;
    node.dataset.loaded = "1";
    const path = node.dataset.path;
    const sub = node.querySelector("ul");
    const r = await apiList(path);
    if (!r || !r.ok) {
      sub.appendChild(_el("li", "tp-error",
        r && r.error ? `Listing failed: ${r.error}` : "Listing failed."));
      return;
    }
    let entries = (r.entries || []);
    if (mode === "dir") {
      entries = entries.filter((e) => e.kind === "directory");
    }
    if (opts.filter) {
      entries = entries.filter((e) => opts.filter(
        e, path.replace(/\/$/, "") + "/" + e.name));
    }
    // Folders first, each half in name order — the hierarchy reads
    // top-down instead of interleaving files into it.
    entries.sort((a, b) => (a.kind === b.kind)
      ? a.name.localeCompare(b.name)
      : (a.kind === "directory" ? -1 : 1));
    if (entries.length === 0) {
      sub.appendChild(_el("li", "tp-empty", "(empty)"));
      return;
    }
    for (const e of entries) {
      sub.appendChild(_buildNode(
        e.name, path.replace(/\/$/, "") + "/" + e.name, e.kind));
    }
  }

  const rootUl = _el("ul", "tp-root");
  tree.appendChild(rootUl);
  const rootNode = _buildNode("projects", root, "directory");
  rootUl.appendChild(rootNode);
  await _expand(rootNode);
  rootNode.querySelector(".tp-children").hidden = false;
  rootNode.querySelector(".tp-twisty").textContent = "▾";
  rootNode.classList.add("is-open");

  const actions = _el("div", "molbuilder-projects-dialog-actions");
  const cancel = _el("button", null, "Cancel");
  cancel.type = "button";
  cancel.setAttribute("data-action", "cancel");
  cancel.addEventListener("click", () => _settle(null));
  actions.appendChild(cancel);
  confirmBtn = _el("button", "is-primary", opts.confirmLabel || "Choose");
  confirmBtn.type = "button";
  confirmBtn.setAttribute("data-action", "confirm");
  confirmBtn.disabled = true;
  confirmBtn.addEventListener("click",
    () => { if (chosenPath) _settle(chosenPath); });
  actions.appendChild(confirmBtn);
  dialog.appendChild(actions);

  dialog.addEventListener("cancel", (ev) => {   // ESC
    ev.preventDefault();
    _settle(null);
  });
  dialog.addEventListener("click", (ev) => {    // click off the panel
    if (ev.target === dialog) _settle(null);
  });

  document.body.appendChild(dialog);
  return new Promise((resolve) => {
    _active = { dialog, resolve };
    try { dialog.showModal(); } catch (_) { dialog.setAttribute("open", ""); }
  });
}
