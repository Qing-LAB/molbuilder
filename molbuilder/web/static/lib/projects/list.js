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
  setShared, publishCommit, getProjectsRoot, setRefreshHandler,
  projects as _projectsApi,
} from "./state.js";
import { showPreview } from "./preview.js";

let elCrumb, elList, elSidebar, elLockBanner, elLockMsg, elLockCancel;

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

/**
 * Render-sidebar subscriber: ONE function that derives the
 * selection-dependent DOM state from the current ``{dir, file}``
 * payload.  Wired as a ``projects.onChange`` subscriber so every
 * ``setShared`` / ``navigateTo`` call automatically syncs the
 * sidebar -- no inline DOM-mutation calls from click handlers.
 *
 * Scope (M2 / #166, 2026-05-31):
 *   * Mark the matching entry as ``.is-selected`` (or clear when
 *     ``file`` is blank).
 *   * Render the "Selected: <basename>" status line.
 *
 * NOT in scope:
 *   * Breadcrumb -- driven by directory state, not selection.
 *     openDir handles it when the dir changes; if a programmatic
 *     ``setShared(newDir, newFile)`` from another module crosses
 *     directories, the breadcrumb will be stale until openDir is
 *     called.  In practice the only cross-dir navigator is openDir
 *     itself.
 *   * Entry list -- driven by directory listing, owned by openDir.
 *
 * Subscriber fires once on subscribe with the initial state, so
 * the sidebar paints correctly on first load without a manual call.
 */
function renderSidebar(payload) {
  if (!elList) return;   // initList hasn't run yet
  const file = payload && payload.file ? payload.file : "";
  // Mark the entry in the list, if it exists.  The list may not
  // contain the file (if the file is in a different dir than the
  // currently-listed one); _markSelectedByPath silently clears
  // any prior selection in that case.
  if (file) {
    const li = elList.querySelector(
      `.ps-entry[data-path="${_cssEscape(file)}"]`
    );
    _markSelected(li);   // null li -> just clear prior selection
  } else {
    _markSelected(null);
  }
  // Status line: derived from the file basename.
  const name = file ? file.slice(file.lastIndexOf("/") + 1) : "";
  _renderSelectionStatus(name, file);
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
  // Phase 6e fifth-review LANDMINE-A: filter user-initiated
  // Cancel before treating as failure.  No signal is wired in
  // today's caller, so this is defensive — but the public
  // ``projects.deleteEntry`` contract advertises ``opts.signal``,
  // so the next caller that adopts a Cancel widget would hit
  // ``window.alert("aborted")`` without this guard.
  if (j && j.aborted) return;
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
        // 2026-05-31 #166: setShared fires the renderSidebar
        // onChange subscriber synchronously, which marks the
        // entry + renders the selection-status line.  No inline
        // DOM mutation here.
        setShared(currentPath, fullPath);
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
        // setShared fires the renderSidebar onChange subscriber
        // synchronously, which marks the entry + renders the
        // selection-status line.  No inline DOM mutation here.
        // Per the sidebar interaction model (memory/
        // project_sidebar_interaction_model.md): single-click is
        // PREVIEW only — sets the global pick.  Commit happens
        // on dblclick (handler below) which fires
        // ``publishCommit`` for tabs that subscribe via
        // ``projects.onCommit``.
        setShared(currentPath, fullPath);
      }
    });
    // Double-click on a file = commit.  Tabs that subscribe via
    // ``projects.onCommit`` run their "use this file" action
    // (the Molbuilder tab Load, future form-tab structure
    // rebuilds, etc.).  Directories don't fire commit — a
    // dblclick on a folder is just a fast navigation; openDir
    // already ran from the first click.
    li.addEventListener("dblclick", (ev) => {
      if (e.kind !== "file") return;
      ev.preventDefault();
      // Selecting text on dblclick is the browser default; clear
      // it so the user doesn't see a highlighted filename after
      // committing.
      if (window.getSelection) {
        try { window.getSelection().removeAllRanges(); } catch (_) {}
      }
      // setShared already ran from the first click; publishCommit
      // is the discrete commit event subscribers gate on.
      publishCommit(currentPath, fullPath);
    });
    elList.appendChild(li);
  }
}

/**
 * Navigate the sidebar into the given absolute directory.  Fetches
 * the listing, redraws breadcrumb + entry list, and updates state.
 * Tolerant of API failures (shows an inline error in the list area).
 *
 * Returns the envelope shape from docs/protocols/projects-sidebar.md
 * § C7:
 *   ``{ok: true, path, entries}`` on success
 *   ``{ok: false, error}`` on failure
 * The function still runs its DOM side effects (breadcrumb +
 * entry list + cursor update) regardless of caller's interest in
 * the return value.  Existing internal callers (sidebar click
 * handlers + refreshHandler) ignore the return; the
 * ``projects.navigateTo`` public API (in state.js) returns it
 * verbatim per the design contract.
 *
 * Public so forms.js can call it after a successful mkdir / create-
 * project / etc., and so state.js can call it via the refreshHandler
 * registration below.
 */
export async function openDir(absPath) {
  // Read the current file selection BEFORE we clobber sessionStorage.
  // ``state.writeFile`` (lib/projects/state.js) calls refreshHandler(cur)
  // -> openDir() after every successful write -- which means every
  // "Generate script + Save", every "Modify + Save", and every project-
  // tree mutation re-runs this function with absPath = parent_dir.
  // The pre-2026-05-18 behaviour was to always ``setShared(resp.path, "")``
  // at the bottom -- which wiped the user's selection out of sessionStorage
  // on every save, silently breaking every load-from-selection downstream.
  //
  // Preservation rule: if the previously-selected file STILL appears in
  // the new listing (same parent dir, file still exists), keep it
  // selected.  Otherwise blank.  Comparison is by exact full-path match,
  // not by basename, so file rename / replace correctly drops the stale
  // selection.
  const prevFile = sessionStorage.getItem("molbuilder.current_file") || "";

  const resp = await apiList(absPath);
  if (!resp.ok) {
    _renderBreadcrumb(absPath);
    elList.innerHTML = "";
    elList.classList.add("is-empty");
    const li = document.createElement("li");
    li.style.cssText = "padding: 0.7rem; color: #e07a7a;";
    li.textContent = resp.error || "Failed to list directory.";
    elList.appendChild(li);
    // Listing failed -- we don't know the new state of resp.path's
    // children, so we can't decide whether to keep prevFile.  Blank
    // it (same as the pre-fix behaviour for the error path).
    // 2026-05-31 #166: renderSidebar subscriber wired in initList()
    // picks up setShared synchronously and clears the selection
    // status; no inline call needed here.
    setShared(absPath, "");
    return { ok: false, error: resp.error || "Failed to list directory." };
  }
  _renderBreadcrumb(resp.path);
  _renderList(resp.entries, resp.path);
  // Re-apply the persisted filter after each render — entries are
  // rebuilt on every openDir, so a previously-active filter would
  // otherwise show ALL entries on directory change.
  _applyFilter();

  // Determine whether prevFile survives the refresh.
  let keptFile = "";
  if (prevFile) {
    const dirSlash = resp.path.replace(/\/$/, "") + "/";
    if (prevFile.startsWith(dirSlash)) {
      // prevFile claims to live inside the directory we just listed.
      // Verify it's still in the entry list (and is a file, not a
      // directory) -- the file may have been deleted or renamed.
      const expectedName = prevFile.slice(dirSlash.length);
      const stillPresent = resp.entries.some(
        (e) => e.kind === "file" && e.name === expectedName
      );
      if (stillPresent) keptFile = prevFile;
    }
    // else: prevFile is inside a different directory -- the user
    // navigated away.  Drop it.
  }
  // 2026-05-31 #166: the renderSidebar onChange subscriber
  // (registered in initList) handles entry marking + status line
  // in response to this setShared.  The subscriber's DOM lookup
  // for the kept-file's <li> runs AFTER _renderList has populated
  // elList, so the element is reachable.
  setShared(resp.path, keptFile);
  return { ok: true, path: resp.path, entries: resp.entries };
}

/**
 * Re-mark the current file selection + show its status, called after
 * the initial directory listing so a cross-tab persistent selection
 * survives a page navigation.
 */
export function restoreSelection() {
  const projectsRoot = getProjectsRoot();
  const file = sessionStorage.getItem("molbuilder.current_file") || "";
  const dir  = sessionStorage.getItem("molbuilder.current_dir")  || "";
  if (!file || !projectsRoot || !file.startsWith(projectsRoot)) return;
  // 2026-05-31 #166: render via the shared subscriber so the
  // entry-marker + status-line code lives in exactly one place.
  // The initial onChange fire ran before the list was populated;
  // this re-fires renderSidebar now that the list has rows.
  renderSidebar({ dir: dir, file: file });
}

function _cssEscape(s) {
  if (typeof CSS !== "undefined" && CSS.escape) return CSS.escape(s);
  return s.replace(/["\\]/g, "\\$&");
}

/** Apply / clear the sidebar lock visual.  Called on every lock change. */
function _applyLockVisual(payload) {
  // Re-resolve elSidebar each call so this works even if initLockUI
  // ran before the partial was rendered (it shouldn't, but defence
  // in depth -- the 2026-05-28 sighting was a "lock fires but UI
  // doesn't paint" desync caused by the wiring being gated on an
  // unrelated init success).
  const sb  = elSidebar    || document.getElementById("projects-sidebar");
  const ban = elLockBanner || document.getElementById("ps-lock-banner");
  const msg = elLockMsg    || document.getElementById("ps-lock-message");
  if (!sb) return;
  sb.classList.toggle("is-locked", !!payload.locked);
  if (ban) {
    ban.hidden = !payload.locked;
    if (msg) msg.textContent = payload.reason || "Working…";
  }
}

/**
 * Wire the sidebar LOCK UI (banner subscription + Cancel-click).
 *
 * Split from ``initList`` (2026-05-28) so lock-UI wiring no longer
 * depends on ``apiRoots()`` succeeding -- the original design gated
 * it on initList running, which only ran if project-root resolution
 * succeeded.  When ``apiRoots`` returned empty (no projects/ dir,
 * slow network, misconfigured roots), ``lock()`` would set state
 * but no subscriber existed to paint it, AND the Cancel button was
 * unwired.  Decoupling the lock-UI from the file-listing wiring
 * means the lock works on every page that includes the sidebar
 * partial, regardless of file-listing state.
 *
 * Cancel uses delegated dispatch on document (not addEventListener
 * on the button) so the wiring is robust even if the partial
 * hasn't rendered yet at the time this function runs.
 */
export function initLockUI() {
  elSidebar    = document.getElementById("projects-sidebar");
  elLockBanner = document.getElementById("ps-lock-banner");
  elLockMsg    = document.getElementById("ps-lock-message");
  // Delegated click: any future click on #ps-lock-cancel hits this
  // handler, regardless of whether the button existed at wire time
  // or was re-created later.
  document.addEventListener("click", (ev) => {
    const t = ev.target;
    if (t && t.id === "ps-lock-cancel") {
      _projectsApi.cancelLockedOperation();
    }
  });
  _projectsApi.onLockChange(_applyLockVisual);
}

// Filter state (B.5.5).  Free-text filter that hides non-matching
// FILES from the rendered entry list — folders always stay visible
// so the user can still navigate.  Case-insensitive substring; a
// leading "." is a shortcut for extension match (".xyz" matches
// files whose name ends in ".xyz" only, not "water.xyz.bak").
const FILTER_KEY = "molbuilder.projects_sidebar_filter";
let _filterQuery = "";

function _filterMatches(name, kind) {
  if (kind === "directory") return true;   // folders always visible
  if (!_filterQuery) return true;
  const lc = name.toLowerCase();
  if (_filterQuery.startsWith(".")) {
    return lc.endsWith(_filterQuery);
  }
  return lc.includes(_filterQuery);
}

/**
 * Apply the current filter to the rendered entries.  Called after
 * every render + every filter-input keystroke.  Hides via the
 * ``is-hidden`` class so the entry DOM (click handlers, dataset
 * attrs) stays intact for when the user clears the filter.
 *
 * Also flips ``is-filtered-empty`` on the list root so the CSS
 * "(no match)" placeholder surfaces when every entry is hidden.
 */
function _applyFilter() {
  if (!elList) return;
  let visibleCount = 0;
  for (const li of elList.children) {
    const kind = li.dataset.kind || "";
    const name = (li.querySelector(".ps-entry-name") || {}).textContent || "";
    const show = _filterMatches(name, kind);
    li.classList.toggle("is-hidden", !show);
    if (show) visibleCount++;
  }
  // Don't flip is-filtered-empty when the underlying list is
  // empty (no entries to filter) — the existing is-empty
  // "(empty)" affordance fires there.
  const underlyingEmpty = elList.children.length === 0;
  elList.classList.toggle(
    "is-filtered-empty",
    !underlyingEmpty && visibleCount === 0 && Boolean(_filterQuery)
  );
  const clearBtn = document.getElementById("ps-filter-clear");
  if (clearBtn) clearBtn.hidden = !_filterQuery;
}

/**
 * Wire the filter <input> + clear button.  Idempotent — calling
 * twice is a no-op on the second call (the listener-already-
 * attached guard prevents stacking).
 */
function _initFilter() {
  const input = document.getElementById("ps-filter-input");
  if (!input || input.dataset.psFilterWired === "1") return;
  input.dataset.psFilterWired = "1";
  try { _filterQuery = (sessionStorage.getItem(FILTER_KEY) || "").toLowerCase(); }
  catch (_) { _filterQuery = ""; }
  if (_filterQuery) input.value = _filterQuery;
  input.addEventListener("input", () => {
    _filterQuery = input.value.trim().toLowerCase();
    try {
      if (_filterQuery) sessionStorage.setItem(FILTER_KEY, _filterQuery);
      else              sessionStorage.removeItem(FILTER_KEY);
    } catch (_) { /* swallow quota / private-mode throws */ }
    _applyFilter();
  });
  const clearBtn = document.getElementById("ps-filter-clear");
  if (clearBtn) {
    clearBtn.addEventListener("click", () => {
      input.value = "";
      _filterQuery = "";
      try { sessionStorage.removeItem(FILTER_KEY); } catch (_) {}
      _applyFilter();
      input.focus();
    });
  }
}

export function initList() {
  elCrumb = document.getElementById("ps-breadcrumb");
  elList  = document.getElementById("ps-list");
  // Lock-UI wiring lives in initLockUI() and runs unconditionally in
  // projects-sidebar.js's init() -- KEEP IT THAT WAY.  Pulling those
  // calls back into initList would re-couple lock to file-listing.
  setRefreshHandler(openDir);
  _initFilter();
  // renderSidebar subscriber.  Single point of selection-state-to-DOM
  // sync.  Fires once on subscribe with the current state (per the
  // onChange contract in state.js), so the initial paint happens
  // here without a manual call.  Every ``setShared`` / ``navigateTo``
  // thereafter -- whether from a sidebar click, the /results file-
  // picker dropdown, or any future programmatic navigator -- auto-
  // syncs the entry-marker + status line.
  _projectsApi.onChange(renderSidebar);
}
