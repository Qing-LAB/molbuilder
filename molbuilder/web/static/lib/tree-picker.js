/* lib/tree-picker.js — the ONE pop-out path picker.
 *
 * One lazy-expanding tree over the projects root, used by:
 *
 *   * the sidebar's Move/Copy destination question (dirs only);
 *   * the Modify tab's slab panel, picking the relaxed file a lattice is
 *     measured from;
 *   * the Transport tab's junction citation: any directory, with the
 *     meta line from the caller's `describe`.
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
 *   pickable      optional (entry, path) => bool over SELECTABLE
 *                 entries: rows failing it still list (and expand),
 *                 but cannot be chosen — the slab panel's lattice
 *                 picker walks the whole tree yet lets only `.xyz` /
 *                 `.XV` files be the answer
 *   describe      optional async (path, entry) => string; shown in the
 *                 meta line when a selection lands
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
  // can say (`describe`).
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
    tree.querySelectorAll(".is-selected").forEach((n) => {
      n.classList.remove("is-selected");
      const r = n.querySelector(":scope > .tp-row");
      if (r) r.setAttribute("aria-selected", "false");
    });
    if (path) {
      const node = tree.querySelector(
        `[data-path="${path.replace(/"/g, '\\"')}"]`);
      if (node) {
        node.classList.add("is-selected");
        const r = node.querySelector(":scope > .tp-row");
        if (r) r.setAttribute("aria-selected", "true");
      }
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

  const selectable = (kind, path, name) => {
    const byMode = (mode === "any"
                    || (mode === "dir" ? kind === "directory"
                                       : kind === "file"));
    if (!byMode) return false;
    if (typeof opts.pickable === "function") {
      return !!opts.pickable({ name, kind }, path);
    }
    return true;
  };

  /* THE KEYBOARD PATH (ui-contract.md § 4.1): a row is the tab stop of its
   * treeitem; Enter picks it (or opens a folder that cannot be picked), the
   * arrows move between the rows on screen, open a folder and close it. */
  function _rowsOnScreen() {
    return Array.from(tree.querySelectorAll(".tp-row"))
      .filter((r) => r.tabIndex === 0 && r.offsetParent !== null);
  }
  function _rowKey(ev, n) {
    const rows = _rowsOnScreen();
    const i = rows.indexOf(n.row);
    if (ev.key === "Enter" || ev.key === " ") {
      ev.preventDefault();
      if (n.canPick) _setChosen(n.path, { name: n.name, kind: n.kind });
      else if (n.isDir) n.toggle();
    } else if (ev.key === "ArrowRight") {
      ev.preventDefault();
      if (n.isDir && !n.expanded()) n.toggle();
      else if (rows[i + 1]) rows[i + 1].focus();
    } else if (ev.key === "ArrowLeft") {
      ev.preventDefault();
      if (n.isDir && n.expanded()) { n.toggle(); return; }
      const parent = n.li.parentElement && n.li.parentElement.closest(".tp-node");
      const prow = parent && parent.querySelector(":scope > .tp-row");
      if (prow) prow.focus();
    } else if (ev.key === "ArrowDown") {
      ev.preventDefault();
      if (rows[i + 1]) rows[i + 1].focus();
    } else if (ev.key === "ArrowUp") {
      ev.preventDefault();
      if (rows[i - 1]) rows[i - 1].focus();
    }
  }

  function _buildNode(name, path, kind) {
    const li = _el("li", "tp-node");
    li.dataset.path = path;
    li.dataset.kind = kind;
    // THE ROW IS THE TREEITEM (ui-contract.md § 4.1): the element that
    // takes focus carries the role, its name (the label's text) and both
    // states; the list item is structure only.
    li.setAttribute("role", "none");

    const row = _el("div", "tp-row");
    row.setAttribute("role", "treeitem");
    row.setAttribute("aria-selected", "false");
    const canPick = selectable(kind, path, name);
    if (!canPick) row.classList.add("tp-row--inert");

    const isDir = kind === "directory";
    if (isDir) row.setAttribute("aria-expanded", "false");
    const tw = _el("button", "tp-twisty", isDir ? "▸" : "");
    tw.type = "button";
    tw.setAttribute("aria-label", "Expand");
    tw.tabIndex = -1;            // the row is the tab stop; ArrowRight opens
    if (!isDir) tw.classList.add("tp-twisty--leaf");
    row.appendChild(tw);
    row.appendChild(_el("span", "tp-icon", isDir ? "📁" : "📄"));
    row.appendChild(_el("span", "tp-label", name));
    li.appendChild(row);

    const sub = _el("ul", "tp-children");
    sub.setAttribute("role", "group");
    sub.hidden = true;
    li.appendChild(sub);

    let expanded = false;
    async function toggle() {
      if (!isDir) return;
      expanded = !expanded;
      sub.hidden = !expanded;
      tw.textContent = expanded ? "▾" : "▸";
      tw.setAttribute("aria-label", expanded ? "Collapse" : "Expand");
      row.setAttribute("aria-expanded", expanded ? "true" : "false");
      li.classList.toggle("is-open", expanded);
      if (expanded) await _expand(li);
    }
    tw.addEventListener("click", (ev) => { ev.stopPropagation(); toggle(); });
    row.addEventListener("click", () => {
      if (canPick) _setChosen(path, { name, kind });
    });
    row.addEventListener("dblclick", () => { if (!expanded) toggle(); });
    row.tabIndex = (canPick || isDir) ? 0 : -1;
    row.addEventListener("keydown", (ev) => _rowKey(ev, {
      li, row, isDir, canPick, path, name, kind, toggle,
      expanded: () => expanded,
    }));
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
