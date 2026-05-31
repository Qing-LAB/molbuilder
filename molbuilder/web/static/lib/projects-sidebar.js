/* projects-sidebar.js -- entry point.
 *
 * The Projects sidebar's behaviour is split across small modules
 * under projects/ for clarity and independent testability:
 *
 *   api.js       -- HTTP wrappers (no DOM, no state)
 *   state.js     -- sessionStorage + the public Inquire API
 *                   (window.molbuilder.projects.*)
 *   list.js      -- breadcrumb + entry list + per-entry buttons +
 *                   openDir
 *   forms.js     -- + New project / + New subdir / + Upload file
 *   preview.js   -- file-preview modal
 *
 * This entry file imports each module, mounts the public API on
 * window, and runs the bootstrap (resolve projects/ root via
 * /api/files/roots, then list whichever dir the user was last in).
 *
 * Loaded via `<script type="module">` -- ES modules are supported
 * natively in every modern browser; no bundler.
 */

import { apiRoots } from "./projects/api.js";
import {
  projects, setProjectsRoot, SS_DIR,
} from "./projects/state.js";
import {
  initList, initLockUI, openDir, restoreSelection,
} from "./projects/list.js";
import { initForms } from "./projects/forms.js";
import { initPreview } from "./projects/preview.js";

window.molbuilder = window.molbuilder || {};
window.molbuilder.projects = projects;
// Module-init contract (design.md "Module init contract"): also
// register with the runtime so consumers can ``whenReady("projects")``
// instead of polling for ``window.molbuilder.projects`` (which is
// undefined when classic-script consumers run before this
// type=module script's deferred initialisation).
if (window.molbuilder.runtime
    && typeof window.molbuilder.runtime.register === "function") {
    window.molbuilder.runtime.register("projects", projects);
}

async function init() {
  const sidebar = document.getElementById("projects-sidebar");
  if (!sidebar) return;                  // page didn't include the partial

  // Wire the lock UI FIRST -- before any await that could throw or
  // bail.  The lock UI needs to work regardless of project-root
  // resolution; see initLockUI() docstring in list.js for the
  // 2026-05-28 background.  If we put this after the apiRoots()
  // await, a slow / failed roots call leaves the lock UI unwired
  // and lock() becomes a silent no-op visually.
  initLockUI();

  // NOTE: `class="has-projects-sidebar"` is set on <body> in each
  // template that includes the sidebar partial -- NOT here.  Adding
  // it via JS races with the initial paint: any layout-sensitive
  // widget that init'd before the type=module script ran (Plotly
  // plots in Watch / Spectra; 3Dmol viewer; CSS-grid auto-fit
  // dependent layouts) would have measured the WIDER pre-sidebar
  // geometry and look broken until the next browser resize.
  // Resolve projects/ root from the backend's single-root contract.
  // 2026-05-30: apiRoots now returns the uniform envelope
  // ``{ok, roots, error?}`` -- failure cases (network drop, server
  // misconfig) surface here instead of throwing.
  const rootsResp = await apiRoots();
  const roots = rootsResp.roots || [];
  if (!rootsResp.ok || roots.length === 0) {
    const list = document.getElementById("ps-list");
    if (list) {
      list.classList.add("is-empty");
      const reason = rootsResp.ok
        ? "No file-picker roots configured."
        : ("File-picker roots unavailable: "
           + (rootsResp.error || "unknown error"));
      list.innerHTML = "<li style='padding:0.7rem;color:#e07a7a;'>"
                     + reason + "</li>";
    }
    return;
  }
  setProjectsRoot(roots[0].path);

  // Wire each module's DOM handlers.  Order matters only in so
  // much as list.js must register its refresh handler BEFORE any
  // state-mutating call could trigger one (in practice all of
  // these are synchronous and the trigger comes later via openDir).
  initList();
  initForms();
  initPreview();

  // Navigate to the previously-visited dir if it's still inside
  // projects/, else start at the root.
  const lastDir = sessionStorage.getItem(SS_DIR) || "";
  const start = (lastDir && lastDir.startsWith(roots[0].path))
              ? lastDir : roots[0].path;
  await openDir(start);

  // Re-mark a persisted file selection (cross-tab navigation).
  restoreSelection();
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", init);
} else {
  init();
}
