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
import { initList, openDir, restoreSelection } from "./projects/list.js";
import { initForms } from "./projects/forms.js";
import { initPreview } from "./projects/preview.js";

window.molbuilder = window.molbuilder || {};
window.molbuilder.projects = projects;

async function init() {
  const sidebar = document.getElementById("projects-sidebar");
  if (!sidebar) return;                  // page didn't include the partial

  document.body.classList.add("has-projects-sidebar");
  // Resolve projects/ root from the backend's single-root contract.
  const roots = await apiRoots();
  if (roots.length === 0) {
    const list = document.getElementById("ps-list");
    if (list) {
      list.classList.add("is-empty");
      list.innerHTML = "<li style='padding:0.7rem;color:#e07a7a;'>"
                     + "No file-picker roots configured.</li>";
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
