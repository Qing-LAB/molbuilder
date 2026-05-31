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

// Sidebar lock state.  Set by long-running operations (Save .fdf,
// Save spectra .py, multi-step pseudo install + wrapper write) that
// must not race against the user re-navigating the sidebar mid-flight
// -- e.g. clicking another directory between "write .fdf" and
// "install pseudos", which would silently retarget the pseudo copy
// to the new directory.
//
// Recovery design (per the 2026-05-27 review):
//   Layer A -- try/finally:  every callsite wraps lock() in try {}
//              finally { unlock() } so a thrown promise still releases.
//   Layer B -- per-fetch timeout:  every network call in the locked
//              window has an AbortController + setTimeout(abort, T)
//              so a hung server can't hold the lock indefinitely.
//   Layer C -- Cancel button:  if A and B both fail (genuine JS bug
//              or backend deadlock), the lock banner renders a
//              user-visible Cancel that runs the registered abort
//              callbacks + forces unlock.  No silent stuck state.
const lockSubscribers = new Set();
let lockState = null;        // { reason: string, cancelers: Function[] }
                              // or null when unlocked.

function _publishLockChange() {
  const payload = { locked: lockState !== null,
                    reason: lockState ? lockState.reason : "" };
  lockSubscribers.forEach((cb) => {
    try { cb(payload); } catch (_) { /* one bad subscriber can't break the loop */ }
  });
}

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
  // Defense-in-depth lock guard (2026-05-30, #177).  The UI side
  // already blocks sidebar clicks via CSS pointer-events:none while
  // the lock is held; this rejects programmatic mutations too so
  // an in-inspector navigator (e.g. /results result-list dropdown)
  // can't slip a directory change past an active Save pipeline.
  // Returns {ok, error?} so callers can branch -- the previous
  // void-return contract is preserved for the success path
  // (most callers don't check the return value).  See
  // docs/protocols/projects-sidebar.md § 8.5.
  if (lockState !== null) {
    return {
      ok:    false,
      error: "sidebar is locked: " + (lockState.reason || "operation in progress"),
    };
  }
  sessionStorage.setItem(SS_DIR,  dir  || "");
  sessionStorage.setItem(SS_FILE, file || "");
  publishSelectionChange();
  return { ok: true };
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

/**
 * Acquire the sidebar lock for a multi-step operation.
 *
 * While locked the sidebar's list / breadcrumb / create-form clicks
 * are visually + functionally disabled.  Subscribers (see ``onLockChange``)
 * receive the lock transition synchronously so they can render UI
 * state immediately.
 *
 * ``cancelers`` is an array of zero-arg callables to invoke if the
 * user clicks the lock banner's Cancel button.  Typically these abort
 * an in-flight AbortController -- pass ``[() => controller.abort()]``
 * so a hung backend can be aborted from the UI.
 *
 * Returns the *original lock token* — callers don't need to pass it
 * back; ``unlock()`` is global.  Re-entry is rejected (throws) on
 * purpose: nested locks would tangle the cancel-button semantics.
 * If you need to layer two operations, compose them in one lock or
 * unlock between them.
 */
function lock(reason, cancelers) {
  if (lockState !== null) {
    throw new Error(
      "molbuilder.projects.lock(): already locked -- "
      + "previous reason: " + lockState.reason + ", new: " + reason
    );
  }
  lockState = {
    reason:    String(reason || "Working…"),
    cancelers: Array.isArray(cancelers) ? cancelers.slice() : [],
  };
  _publishLockChange();
  return lockState;
}

/** Release the sidebar lock.  Idempotent (no-op when already unlocked).
 *  Always call from a ``finally`` so an exception in the locked
 *  operation can't leave the sidebar stuck. */
function unlock() {
  if (lockState === null) return;
  lockState = null;
  _publishLockChange();
}

/** Run the registered cancelers (does NOT itself unlock -- the operation
 *  promise's own finally is responsible for that, AFTER its abort path
 *  has unwound).  Called by the lock banner's Cancel button. */
function cancelLockedOperation() {
  if (lockState === null) return;
  const fns = lockState.cancelers.slice();
  for (const fn of fns) {
    try { fn(); } catch (_) { /* one bad canceler can't break the rest */ }
  }
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
  // True when ``current_dir`` is unset or equals the projects/ root.
  // Consumers (Build + Spectra Save buttons) use this to gate "are
  // we in a state where saveToWorkspace will succeed?" -- raw ``!!dir``
  // truthy-checks would enable Save at the projects root, then
  // saveToWorkspace would silently return null and the click would
  // produce a confusing "no current_dir" error.
  atRoot:              () => atProjectsRoot(
                          sessionStorage.getItem(SS_DIR) || ""),
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
  // ---- Programmatic navigation (2026-05-30) --------------------- //
  // Promoted to the public surface so in-inspector navigators (the
  // /results result-list dropdown) can set the active file without
  // going through a sidebar click handler.  Updates BOTH dir and
  // file in sessionStorage + fires onChange subscribers; the
  // /results viewer.js subscriber then dispatches to the matching
  // inspector for the new file.  Part of sidebar gap M4 (#175).
  setShared,
  // ---- Sidebar lock API (2026-05-27) ---------------------------- //
  // Long-running pipelines (Save .fdf, Save .py, install pseudos +
  // wrapper) call lock() before step 1 and unlock() in finally so
  // the user can't re-navigate the sidebar to a different directory
  // mid-pipeline and have step 2 land in the wrong place.
  //
  // See state.js's lock-state docstring for the 3-layer recovery
  // design (try/finally + per-fetch timeout + Cancel button).
  lock,
  unlock,
  isLocked: () => lockState !== null,
  getLockReason: () => lockState ? lockState.reason : "",
  onLockChange: (cb) => {
    lockSubscribers.add(cb);
    // Fire once so the subscriber can render its current state.
    try {
      cb({ locked: lockState !== null,
           reason: lockState ? lockState.reason : "" });
    } catch (_) { /* swallow */ }
    return () => lockSubscribers.delete(cb);
  },
  cancelLockedOperation,
};
