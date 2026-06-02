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

import {
  apiCreateProject,
  apiDelete,
  apiMkdir,
  apiRead,
  apiRename,
  apiUpload,
  apiWrite,
} from "./api.js";

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

// ----- shared subscriber helpers --------------------------------- //
//
// Two contracts driven by docs/protocols/projects-sidebar.md (per
// the 2026-05-31 design decisions A1/A1b/A2):
//
//   * Registration is idempotent intent.  Re-registering the same
//     callback (by reference) is a programming error and THROWS.
//     Catches forgotten-unsubscribe / double-init bugs at the call
//     site instead of letting them silently no-op.
//   * Publish snapshots subscribers BEFORE iterating.  Subscribers
//     added during a publish loop fire only on subsequent events --
//     they already received the current state via fire-once-
//     immediately on their own subscribe call.  This guarantees
//     every subscriber fires exactly once per state change.
//
// Errors thrown by individual subscriber callbacks are caught and
// swallowed (per-subscriber error isolation) so one bad subscriber
// can't poison the publish loop for the rest.

function _registerSubscriber(set, cb, label) {
  if (set.has(cb)) {
    throw new Error(
      label + ": this callback is already registered.  Call the "
      + "returned unsub() before re-subscribing."
    );
  }
  set.add(cb);
}

function _publishToSet(set, payload) {
  // Snapshot first so new subscribers added inside a callback are
  // NOT visited (they got the current state via fire-once-immediately
  // on their own subscribe call).
  const snapshot = Array.from(set);
  for (const cb of snapshot) {
    // Skip subscribers that were removed during the loop (e.g. a
    // subscriber that calls its own unsub() inside its callback).
    if (!set.has(cb)) continue;
    try { cb(payload); } catch (_) { /* per-subscriber isolation */ }
  }
}

// ----- internal: publish loops ----------------------------------- //

function _publishLockChange() {
  _publishToSet(lockSubscribers, {
    locked: lockState !== null,
    reason: lockState ? lockState.reason : "",
  });
}

function publishSelectionChange(payload) {
  // Caller may supply the payload explicitly when sessionStorage
  // can't be relied on (e.g. setShared in private-mode where setItem
  // threw).  When omitted, read live from sessionStorage.
  if (!payload) {
    payload = {
      dir:  sessionStorage.getItem(SS_DIR)  || "",
      file: sessionStorage.getItem(SS_FILE) || "",
    };
  }
  _publishToSet(selectionSubscribers, payload);
}

// ----- exposed to other modules ---------------------------------- //

export function setShared(dir, file) {
  // Defense-in-depth lock guard (2026-05-30, #177).  The UI side
  // already blocks sidebar clicks via CSS pointer-events:none while
  // the lock is held; this rejects programmatic mutations too so
  // a tab-level navigator (e.g. the /results file-picker dropdown
  // in ``lib/results/file-picker.js``) can't slip a directory
  // change past an active Save pipeline.
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
  // sessionStorage may throw on quota / private-mode SecurityError /
  // explicit storage-denied.  Per design § 11.4 + Principle 6 we MUST
  // NOT propagate the throw -- publish the new state regardless so
  // subscribers still update; the cursor just won't survive a reload.
  const payload = { dir: dir || "", file: file || "" };
  try {
    sessionStorage.setItem(SS_DIR,  payload.dir);
    sessionStorage.setItem(SS_FILE, payload.file);
  } catch (e) {
    // Logged loud-but-non-fatal so a developer notices in DevTools.
    // No user-facing error -- the page keeps working.
    console.warn(
      "molbuilder.projects.setShared: sessionStorage write failed; "
      + "cursor won't survive reload",
      e && e.message ? e.message : e
    );
  }
  publishSelectionChange(payload);
  return { ok: true };
}

// Projects-root subscribers (sidebar gap M8 / design § C2, 2026-05-31).
// Tabs that depend on the projects-root being resolved (e.g. Build's
// psml_lib live-resolution caption) need a one-shot notification so
// they don't have to poll ``getProjectsRoot()``.  Single-fire-ish:
// the publish happens AT MOST ONCE per page lifetime (the sidebar
// only resolves the root once at init).  Subscribers registered
// AFTER resolution receive the fire-once-immediately call from
// onProjectsRootResolved itself with the already-resolved root.
const rootSubscribers = new Set();
let rootResolved = false;

function _publishRootResolved() {
  _publishToSet(rootSubscribers, { root: projectsRoot || "" });
}

export function setProjectsRoot(root) {
  projectsRoot = root;
  // Only publish on the FIRST resolution.  Subsequent setProjectsRoot
  // calls (theoretical; sidebar only calls once) shouldn't fan out as
  // re-resolutions -- tabs would over-react.
  if (!rootResolved && projectsRoot) {
    rootResolved = true;
    _publishRootResolved();
  }
}
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

async function readCurrentFile(opts) {
  // Per design § C3 ReadResult:
  //   * null  — no file is currently selected (cursor's file slot empty)
  //   * {ok:true, path, text}   — read succeeded
  //   * {ok:false, error}       — read failed
  // The null case is INTENTIONALLY distinct from a failure so callers
  // can render "no file selected" vs "read failed" differently.
  const path = sessionStorage.getItem(SS_FILE) || "";
  if (!path) return null;
  const j = await apiRead(path, opts);
  if (!j.ok) return { ok: false, error: j.error || "read failed" };
  return { ok: true, path: j.path, text: j.text };
}

async function refresh(opts) {
  // Per design § C6: returns {ok:true} | {ok:false, error}.  The
  // earlier void-return contract violated Principle 6 (every async
  // public method returns an envelope or null).
  const dir = sessionStorage.getItem(SS_DIR) || projectsRoot;
  if (!dir) {
    return { ok: false, error: "no current directory to refresh" };
  }
  if (!refreshHandler) {
    // initList() registers itself as the refresh handler.  If we
    // got here without one, the sidebar's init order is broken --
    // log loud-but-non-fatal so a developer notices.
    console.warn(
      "molbuilder.projects.refresh(): no refresh handler registered.  "
      + "list.js should call setRefreshHandler(openDir) at init time."
    );
    return { ok: false, error: "no refresh handler registered" };
  }
  try {
    await refreshHandler(dir);
    return { ok: true };
  } catch (e) {
    return {
      ok:    false,
      error: "refresh failed: " + (e && e.message ? e.message : String(e)),
    };
  }
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
  if (!w.ok) {
    // Preserve `actual_mtime` on a 409 edit-conflict so tabs can
    // distinguish edit-conflict from other failures per design § 6.2.
    // Also preserve `aborted` if a Cancel cancelled the write.
    const err = { ok: false, error: w.error };
    if (w.actual_mtime != null) err.actual_mtime = w.actual_mtime;
    if (w.aborted) err.aborted = true;
    return err;
  }
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

// ---- Public mutator wrappers (sidebar gap M4, #175, 2026-05-31) //
//
// Thin pass-throughs over api.js that ALSO trigger a sidebar
// listing refresh on success.  These promote previously-internal
// operations onto the public ``window.molbuilder.projects.*``
// surface so tab-level code (the /results file-picker dropdown at
// ``lib/results/file-picker.js``, future programmatic file
// managers) can call them without reaching into ``projects/api.js``
// directly.
//
// NONE of these take the lock guard (#177).  The Save pipeline
// holds the lock while calling these as its own steps; guarding
// them would deadlock the very flow the lock was added to protect.
// User-initiated calls (CSS pointer-events:none in the sidebar)
// can't reach them while the user-visible lock UI is up.
//
// Each wrapper:
//   1. Calls the underlying api function.
//   2. On success, fires ``refreshHandler(<parent>)`` so the
//      sidebar listing picks up the mutation.  Refresh failure is
//      swallowed (not a user error; the mutation already landed).
//   3. Returns the api response verbatim so the caller sees the
//      same {ok, ...} envelope shape.

/** Read a file's contents (returns ``{ok, text, mtime, ...}`` or
 *  ``{ok:false, error}``).  Companion to ``readCurrentFile()``
 *  which is the no-argument form keyed on the sidebar's current
 *  selection.  ``opts.signal`` honoured (per docs/protocols/
 *  projects-sidebar.md § C3). */
async function readFile(path, opts) {
  return await apiRead(path, opts);
}

/** Create a new project directory at the projects root.  On
 *  success refreshes the projects listing so the new directory
 *  appears in the sidebar without a manual reload.
 *  ``opts.signal`` honoured. */
async function createProject(name, opts) {
  const r = await apiCreateProject(name, opts);
  if (r && r.ok && refreshHandler) {
    const root = projectsRoot;
    if (root) {
      try { await refreshHandler(root); }
      catch (_) { /* refresh failure must not fail the create */ }
    }
  }
  return r;
}

/** Make a subdirectory under ``parent``.  Refreshes ``parent``
 *  on success.  ``opts.signal`` honoured. */
async function mkdir(parent, name, opts) {
  const r = await apiMkdir(parent, name, opts);
  if (r && r.ok && refreshHandler) {
    try { await refreshHandler(parent); }
    catch (_) { /* refresh failure must not fail the mkdir */ }
  }
  return r;
}

/** Delete a file or directory.  ``recursive`` (default false)
 *  must be explicit for directory deletes.  Refreshes the
 *  containing directory on success.  ``opts.signal`` honoured. */
async function deleteEntry(path, recursive, opts) {
  const r = await apiDelete(path, recursive, opts);
  if (r && r.ok && refreshHandler) {
    // Derive the parent dir from the deleted path.  If path has no
    // separator (a top-level entry), refresh the projects root.
    const parent = path.indexOf("/") >= 0
      ? path.replace(/\/[^/]+$/, "")
      : projectsRoot;
    if (parent) {
      try { await refreshHandler(parent); }
      catch (_) { /* refresh failure must not fail the delete */ }
    }
  }
  return r;
}

/** Rename a file or directory in place.  ``new_name`` is the new
 *  basename (no path separators; no ``..``).  Refreshes the parent
 *  directory on success so the sidebar shows the new name without
 *  a manual reload.  ``opts.signal`` honoured.
 *
 *  Per design § C5.  Atomic-no-overwrite: backend returns 409 if
 *  ``new_name`` already exists in the parent. */
async function rename(path, newName, opts) {
  const r = await apiRename(path, newName, opts);
  if (r && r.ok && refreshHandler) {
    const parent = path.indexOf("/") >= 0
      ? path.replace(/\/[^/]+$/, "")
      : projectsRoot;
    if (parent) {
      try { await refreshHandler(parent); }
      catch (_) { /* refresh failure must not fail the rename */ }
    }
  }
  return r;
}

/** Upload a file into ``targetDir``.  ``opts.signal`` honoured.
 *  Refreshes ``targetDir`` on success.  Adds ``relPath`` to the
 *  success envelope per design § C6 (UploadOk = WriteOk shape;
 *  backend's /api/files/upload only returns {ok, path, size, mtime}
 *  so we compute relPath from the projects root here). */
async function upload(targetDir, file, opts) {
  const r = await apiUpload(targetDir, file, opts);
  if (r && r.ok) {
    if (r.path && !r.relPath) {
      r.relPath = relativeToProjects(r.path);
    }
    if (refreshHandler) {
      try { await refreshHandler(targetDir); }
      catch (_) { /* refresh failure must not fail the upload */ }
    }
  }
  return r;
}

/** Programmatic navigation -- the public-API form of openDir.
 *
 *  Per docs/protocols/projects-sidebar.md § C7: takes an absolute
 *  path + optional opts, lists the directory, updates the cursor
 *  + sidebar DOM, and returns:
 *     ``{ok: true, path, entries}``   on success
 *     ``{ok: false, error}``          on failure
 *
 *  The actual implementation lives in list.js as ``openDir``;
 *  navigateTo is wired by ``setNavigateToImpl`` at sidebar init
 *  time.  Until that wiring fires (pre-init or in a tab that
 *  doesn't load the sidebar), navigateTo returns the documented
 *  error envelope rather than throwing.
 *
 *  Lock guard (§ 8.5 defense-in-depth): navigateTo early-returns
 *  ``{ok: false, error: "sidebar is locked: <reason>"}`` when a
 *  lock is held -- the impl (openDir) is NEVER called in that case.
 *  openDir itself is intentionally NOT lock-guarded because it
 *  doubles as the refresh handler called mid-Save-pipeline by
 *  writeFile; the lock guard lives at the public-surface layer
 *  here so external callers respect the contract without
 *  deadlocking the internal refresh path.
 */
let _navigateToImpl = null;
async function navigateTo(absPath, opts) {
  // Defense-in-depth lock guard per § 8.5.  openDir (the underlying
  // impl) is intentionally NOT lock-guarded because it doubles as
  // the refresh handler called mid-Save-pipeline by writeFile.  The
  // public-surface wrapper enforces the design contract here:
  // external callers go through navigateTo and get the locked
  // rejection; internal callers go through openDir directly and
  // bypass the guard intentionally.
  if (lockState !== null) {
    return {
      ok:    false,
      error: "sidebar is locked: " + (lockState.reason || "operation in progress"),
    };
  }
  if (typeof _navigateToImpl !== "function") {
    return {
      ok:    false,
      error: "navigateTo unavailable: sidebar not initialised",
    };
  }
  return _navigateToImpl(absPath, opts);
}
/** Wire navigateTo's impl from list.js (the openDir export).
 *  Called once at sidebar init.  Idempotent. */
export function setNavigateToImpl(fn) { _navigateToImpl = fn; }

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
    _registerSubscriber(selectionSubscribers, cb, "onChange");
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
  // ---- Public mutator + navigation surface (#175, 2026-05-31) -- //
  // Promoted from internal api.js consumers so external callers
  // (the /results tab-level file-picker dropdown at
  // ``lib/results/file-picker.js``, future programmatic file
  // managers) can use them WITHOUT reaching into ./api.js
  // directly.  Each method auto-fires a sidebar listing refresh on
  // success so the tree stays in sync.
  readFile,
  createProject,
  mkdir,
  deleteEntry,
  rename,
  upload,
  setShared,
  navigateTo,
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
    _registerSubscriber(lockSubscribers, cb, "onLockChange");
    // Fire once so the subscriber can render its current state.
    try {
      cb({ locked: lockState !== null,
           reason: lockState ? lockState.reason : "" });
    } catch (_) { /* swallow */ }
    return () => lockSubscribers.delete(cb);
  },
  // Projects-root resolution subscriber (design § C2, 2026-05-31).
  // Fires AT MOST ONCE per page lifetime when the sidebar's init
  // resolves the root from apiRoots().  Subscribers that register
  // BEFORE resolution receive the call when resolution lands; those
  // that register AFTER resolution receive an immediate call with
  // the resolved root (the fire-once-immediately contract).
  onProjectsRootResolved: (cb) => {
    _registerSubscriber(rootSubscribers, cb, "onProjectsRootResolved");
    if (rootResolved) {
      try { cb({ root: projectsRoot || "" }); }
      catch (_) { /* per-subscriber isolation */ }
    }
    return () => rootSubscribers.delete(cb);
  },
  cancelLockedOperation,
};
