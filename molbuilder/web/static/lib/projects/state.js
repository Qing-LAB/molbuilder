/* projects/state.js -- shared selection state + the Inquire API.
 *
 * Owns:
 *   * sessionStorage slots (current_dir / current_file)
 *   * the public ``window.molbuilder.projects.*`` API
 *   * the writeFile primitive + the saveToWorkspace convenience
 *   * the subscriber pattern (onChange) for selection mutations
 *   * the refresh handler registration (list.js plugs in here)
 *
 * Does NOT own:
 *   * any DOM manipulation
 *   * directory listing (that's list.js -- which subscribes via
 *     setRefreshHandler + responds when state asks for a re-list)
 *
 * Spec: docs/protocols/selection.md.
 */

import { apiRead, apiWrite } from "./api.js";

export const SS_DIR  = "molbuilder.current_dir";
export const SS_FILE = "molbuilder.current_file";

// Module-private state.
let projectsRoot = null;
const selectionSubscribers = new Set();
// list.js registers itself here at init time; state's refresh() and
// saveToWorkspace() invoke this to ask the list module to re-list
// the current directory + re-render.  Single handler (not a set)
// because there's exactly one list view per page.
let refreshHandler = null;

// ----- internal: set + notify ------------------------------------ //

function publishSelectionChange() {
  const payload = {
    dir:  sessionStorage.getItem(SS_DIR)  || "",
    file: sessionStorage.getItem(SS_FILE) || "",
  };
  selectionSubscribers.forEach((cb) => {
    try { cb(payload); } catch (_) { /* one bad subscriber shouldn't break the loop */ }
  });
}

// ----- exposed to other modules ---------------------------------- //

export function setShared(dir, file) {
  sessionStorage.setItem(SS_DIR,  dir  || "");
  sessionStorage.setItem(SS_FILE, file || "");
  publishSelectionChange();
}

export function setProjectsRoot(root) { projectsRoot = root; }
export function getProjectsRoot()     { return projectsRoot; }

export function setRefreshHandler(handler) { refreshHandler = handler; }

// ----- Public Inquire API (mounted on window.molbuilder.projects) //

export function relativeToProjects(path) {
  if (!path || !projectsRoot) return path || "";
  if (!path.startsWith(projectsRoot)) return path;
  return path.slice(projectsRoot.length).replace(/^\/+/, "") || "/";
}

/** True when ``dir`` is the projects/ root itself (or unset).
 *  Used by callers that need to gate "no operation at depth 0"
 *  behaviour: ``+ New subdir`` visibility, ``+ Upload`` visibility,
 *  saveToWorkspace's silent-skip, etc.  Centralised here so every
 *  caller agrees on what "at the root" means (handles trailing-
 *  slash edge cases). */
export function atProjectsRoot(dir) {
  if (!dir) return true;
  if (!projectsRoot) return true;
  return dir === projectsRoot
      || dir === projectsRoot.replace(/\/$/, "");
}

async function readCurrentFile() {
  const path = sessionStorage.getItem(SS_FILE) || "";
  if (!path) return null;
  const j = await apiRead(path);
  if (!j.ok) return null;
  return {path: j.path, text: j.text};
}

async function refresh() {
  const dir = sessionStorage.getItem(SS_DIR) || projectsRoot;
  if (!dir) return;
  if (!refreshHandler) {
    // initList() registers itself as the refresh handler.  If we
    // got here without one, the sidebar's init order is broken --
    // log loud-but-non-fatal so a developer notices.
    console.warn(
      "molbuilder.projects.refresh(): no refresh handler registered.  "
      + "list.js should call setRefreshHandler(openDir) at init time."
    );
    return;
  }
  await refreshHandler(dir);
}

/**
 * Low-level primitive: write text to a specific path.
 *
 * The path is sent to /api/files/write as-is; the backend validates
 * (inside an allowed root, depth >= 1, parent exists, etc.).
 *
 * Returns:
 *   * ``{ok: true, path, relPath, size, mtime}`` on success.
 *     The sidebar's current directory is auto-refreshed if the
 *     written file landed inside it (so the user sees it appear).
 *   * ``{ok: false, error}`` on any backend failure (409 conflict,
 *     400 bad path, 403 perm denied, etc.).  Callers display
 *     ``error`` verbatim to the user; backend messages already
 *     say what to do.
 *
 * Use this when you have the *exact* path to write -- the future
 * edit-and-save flow (Preview modal Save), derive-job (Phase 2),
 * pseudo-prep, etc.  For the "write into current_dir/<filename>"
 * pattern, use :func:`saveToWorkspace` instead.
 */
async function writeFile(path, text, opts) {
  const w = await apiWrite(path, text, opts);
  if (!w.ok) return {ok: false, error: w.error};
  // Refresh the sidebar listing if we wrote into its current dir.
  // Harmless if not (no-op).  We don't await refresh's own failures.
  try {
    const cur = sessionStorage.getItem(SS_DIR);
    if (cur && path.startsWith(cur.replace(/\/$/, "") + "/")
        && refreshHandler) {
      await refreshHandler(cur);
    }
  } catch (_) { /* refresh failure shouldn't fail the write */ }
  return {
    ok:       true,
    path:     w.path,
    relPath:  relativeToProjects(w.path),
    size:     w.size,
    mtime:    w.mtime,
  };
}

/**
 * Convenience: write into ``<current_dir>/<filename>`` via writeFile.
 *
 * Returns:
 *   * ``null`` silently when current_dir is unset or at the projects/
 *     root.  Callers fall back to local Download / Copy without
 *     showing an error.
 *   * Otherwise the result of writeFile (``{ok:true,...}`` or
 *     ``{ok:false, error}``).
 */
async function saveToWorkspace(text, filename, opts) {
  const dir = sessionStorage.getItem(SS_DIR) || "";
  if (atProjectsRoot(dir)) return null;
  const path = dir.replace(/\/$/, "") + "/" + filename;
  return await writeFile(path, text, opts);
}

export const projects = {
  getCurrentDir:       () => sessionStorage.getItem(SS_DIR)  || "",
  getCurrentFile:      () => sessionStorage.getItem(SS_FILE) || "",
  // The resolved projects/ root path (Capabilities.file_picker_roots[0]).
  // Used by the Build form's psml_lib live-resolution caption so the
  // user sees that ``pseudopotential`` resolves to
  // ``<root>/pseudopotential`` -- the projects/ anchor isn't visible
  // in the input itself.  Returns "" until the sidebar's bootstrap
  // has resolved /api/files/roots.
  getProjectsRoot:     () => getProjectsRoot() || "",
  onChange: (cb) => {
    selectionSubscribers.add(cb);
    // Fire once immediately so subscribers can initialise from the
    // current state without a separate getCurrent* call.
    try {
      cb({
        dir:  sessionStorage.getItem(SS_DIR)  || "",
        file: sessionStorage.getItem(SS_FILE) || "",
      });
    } catch (_) { /* swallow */ }
    return () => selectionSubscribers.delete(cb);
  },
  readCurrentFile,
  relativeToProjects,
  refresh,
  writeFile,
  saveToWorkspace,
};
