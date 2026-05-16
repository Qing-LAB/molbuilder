/* projects/list.js -- breadcrumb + directory listing + per-entry buttons.
 *
 * Owns the DOM under #ps-breadcrumb + #ps-list.  Two click paths:
 *
 *   * Click an entry (directory) -> openDir(fullPath); breadcrumb
 *     redraws + list re-renders.
 *   * Click a per-entry "view" or "×" button -> direct call into
 *     preview.showPreview() / confirmAndDelete().  stopPropagation
 *     so the row click doesn't ALSO fire.
 *
 * Registers openDir as state.setRefreshHandler(handler) at init time
 * so state.refresh() / state.saveToWorkspace() can ask for a re-list
 * without importing list.js (avoids a circular dep).
 *
 * Spec: docs/protocols/selection.md § Sidebar interaction rules.
 */

import { apiList, apiDelete } from "./api.js";
import {
  setShared, getProjectsRoot, setRefreshHandler,
} from "./state.js";
import { showPreview } from "./preview.js";

let elCrumb, elList;

// Names that may NOT be deleted via the sidebar -- ever -- because
// doing so would orphan the projects layout.  Mirrors
// CANONICAL_TOPICS (kept in JS to avoid an extra round-trip; the
// backend will also refuse via its depth rules).
const _UNDELETABLE_AT_DEPTH_1 = new Set([
  "structure", "pseudopotential",
  "optimization", "frequency", "spectrum",
  "transport", "single-point", "scan",
  "user",
]);

function _isDeletableEntry(entry, currentPath) {
  const projectsRoot = getProjectsRoot();
  if (!projectsRoot) return false;
  // Depth 0 (at projects/ root): every entry is a project; user
  // goes to the shell to delete those.
  if (currentPath === projectsRoot
      || currentPath === projectsRoot.replace(/\/$/, "")) {
    return false;
  }
  // Depth 1: canonical-topic dirs are off-limits (would orphan the
  // project layout).  Detect by relative depth from projects root.
  const rel = currentPath.slice(projectsRoot.length).replace(/^\/+/, "");
  const depth = rel.split("/").filter(Boolean).length;
  if (depth === 1 && entry.kind === "directory"
      && _UNDELETABLE_AT_DEPTH_1.has(entry.name)) {
    return false;
  }
  return true;
}

function _humanSize(n) {
  if (n < 1024) return n + " B";
  if (n < 1024 * 1024) return (n / 1024).toFixed(0) + " K";
  if (n < 1024 * 1024 * 1024) return (n / 1024 / 1024).toFixed(1) + " M";
  return (n / 1024 / 1024 / 1024).toFixed(1) + " G";
}

function _markSelected(li) {
  elList.querySelectorAll(".ps-entry.is-selected")
        .forEach((n) => n.classList.remove("is-selected"));
  if (li) li.classList.add("is-selected");
}

function _renderBreadcrumb(currentPath) {
  const projectsRoot = getProjectsRoot();
  elCrumb.innerHTML = "";
  if (!projectsRoot) return;
  const hops = [{label: "projects", path: projectsRoot}];
  if (currentPath && currentPath !== projectsRoot) {
    const rel = currentPath.slice(projectsRoot.length).replace(/^\/+/, "");
    const parts = rel.split("/").filter(Boolean);
    let accum = projectsRoot;
    for (const part of parts) {
      accum = accum.replace(/\/$/, "") + "/" + part;
      hops.push({label: part, path: accum});
    }
  }
  hops.forEach((hop, idx) => {
    if (idx > 0) {
      const sep = document.createElement("span");
      sep.className = "ps-crumb-sep";
      sep.textContent = "/";
      elCrumb.appendChild(sep);
    }
    const crumb = document.createElement("span");
    crumb.className = "ps-crumb"
      + (idx === hops.length - 1 ? " is-current" : "");
    crumb.textContent = hop.label;
    crumb.title = hop.path;
    if (idx < hops.length - 1) {
      crumb.addEventListener("click", () => openDir(hop.path));
    }
    elCrumb.appendChild(crumb);
  });
}

function _renderSelectionStatus(filename, fullPath) {
  // The "Selected: <name>" status line lives in the actions section.
  const sel = document.querySelector("#ps-actions .ps-selection");
  if (!sel) return;
  if (filename) {
    sel.innerHTML = "Selected: <strong></strong>";
    sel.querySelector("strong").textContent = filename;
    sel.title = fullPath;
  } else {
    sel.textContent = "No file selected.";
    sel.title = "";
  }
}

async function _confirmAndDelete(fullPath, entry) {
  const what = entry.kind === "directory" ? "directory" : "file";
  if (!window.confirm(
    "Delete " + what + " '" + entry.name + "'?\n\n"
    + "This cannot be undone."
  )) return;
  const j = await apiDelete(fullPath, entry.kind === "directory");
  if (!j.ok) {
    window.alert(j.error || "Delete failed.");
    return;
  }
  const dir = sessionStorage.getItem("molbuilder.current_dir")
              || getProjectsRoot();
  if (dir) await openDir(dir);
}

function _renderList(entries, currentPath) {
  elList.innerHTML = "";
  elList.classList.toggle("is-empty", entries.length === 0);
  if (entries.length === 0) return;
  const selectedFile = sessionStorage.getItem("molbuilder.current_file") || "";
  for (const e of entries) {
    const fullPath = currentPath.replace(/\/$/, "") + "/" + e.name;
    const li = document.createElement("li");
    li.className = "ps-entry";
    li.dataset.path = fullPath;
    li.dataset.kind = e.kind;
    if (fullPath === selectedFile) li.classList.add("is-selected");

    const icon = document.createElement("span");
    icon.className = "ps-entry-icon";
    icon.textContent = (
      e.kind === "directory" ? "▸" :
      e.kind === "symlink"   ? "→" :
                               "·"
    );
    li.appendChild(icon);

    const name = document.createElement("span");
    name.className = "ps-entry-name";
    name.textContent = e.name;
    name.title = fullPath;
    li.appendChild(name);

    if (e.kind === "file" && e.size !== null) {
      const meta = document.createElement("span");
      meta.className = "ps-entry-meta";
      meta.textContent = _humanSize(e.size);
      li.appendChild(meta);
    }

    // Per-entry hover buttons: preview (file-only) + delete (eligibility-gated).
    // Order: preview-first / delete-last so destructive lives on the far right.
    if (e.kind === "file") {
      const view = document.createElement("button");
      view.type = "button";
      view.className = "ps-entry-action ps-entry-preview";
      view.textContent = "view";
      view.title = "Preview " + e.name;
      view.addEventListener("click", (ev) => {
        ev.stopPropagation();
        _markSelected(li);
        setShared(currentPath, fullPath);
        _renderSelectionStatus(e.name, fullPath);
        showPreview();
      });
      li.appendChild(view);
    }

    if (_isDeletableEntry(e, currentPath)) {
      const del = document.createElement("button");
      del.type = "button";
      del.className = "ps-entry-action ps-entry-delete";
      del.textContent = "×";
      del.title = "Delete " + e.name;
      del.addEventListener("click", (ev) => {
        ev.stopPropagation();
        _confirmAndDelete(fullPath, e);
      });
      li.appendChild(del);
    }

    li.addEventListener("click", () => {
      if (e.kind === "directory") {
        openDir(fullPath);
      } else {
        _markSelected(li);
        setShared(currentPath, fullPath);
        _renderSelectionStatus(e.name, fullPath);
      }
    });
    elList.appendChild(li);
  }
}

/**
 * Navigate the sidebar into the given absolute directory.  Fetches
 * the listing, redraws breadcrumb + entry list, and updates state.
 * Tolerant of API failures (shows an inline error in the list area).
 *
 * Public so forms.js can call it after a successful mkdir / create-
 * project / etc., and so state.js can call it via the refreshHandler
 * registration below.
 */
export async function openDir(absPath) {
  const resp = await apiList(absPath);
  if (!resp.ok) {
    _renderBreadcrumb(absPath);
    elList.innerHTML = "";
    elList.classList.add("is-empty");
    const li = document.createElement("li");
    li.style.cssText = "padding: 0.7rem; color: #e07a7a;";
    li.textContent = resp.error || "Failed to list directory.";
    elList.appendChild(li);
    setShared(absPath, "");
    _renderSelectionStatus("", "");
    return;
  }
  _renderBreadcrumb(resp.path);
  _renderList(resp.entries, resp.path);
  setShared(resp.path, "");
  _renderSelectionStatus("", "");
}

/**
 * Re-mark the current file selection + show its status, called after
 * the initial directory listing so a cross-tab persistent selection
 * survives a page navigation.
 */
export function restoreSelection() {
  const projectsRoot = getProjectsRoot();
  const file = sessionStorage.getItem("molbuilder.current_file") || "";
  if (!file || !projectsRoot || !file.startsWith(projectsRoot)) return;
  const li = elList.querySelector(
    `.ps-entry[data-path="${_cssEscape(file)}"]`
  );
  if (li) _markSelected(li);
  _renderSelectionStatus(file.split("/").pop(), file);
}

function _cssEscape(s) {
  if (typeof CSS !== "undefined" && CSS.escape) return CSS.escape(s);
  return s.replace(/["\\]/g, "\\$&");
}

export function initList() {
  elCrumb = document.getElementById("ps-breadcrumb");
  elList  = document.getElementById("ps-list");
  // Register ourselves as the refresh handler so state.refresh() +
  // state.writeFile()'s post-save re-list can call into us without
  // a circular import.
  setRefreshHandler(openDir);
}
