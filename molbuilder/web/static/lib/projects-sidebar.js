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
  projects, setProjectsRoot, SS_DIR, setNavigateToImpl,
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

/**
 * Narrow-viewport drawer toggle (task #182, 2026-06-02).
 *
 * Wires the hamburger button + backdrop in
 * ``templates/_projects_sidebar.html`` to a ``has-mobile-sidebar-open``
 * class on ``<body>``.  The CSS in ``lib/projects-sidebar.css``
 * keys off that class to slide the sidebar in.  At wider
 * viewports the toggle button + backdrop are display:none, so
 * this wiring is inert above 640 px.
 *
 * Behaviour:
 *   * Button click toggles the class + flips aria-expanded on the
 *     button + aria-hidden on the backdrop.
 *   * Backdrop click closes the drawer (modal-overlay convention).
 *   * Escape key closes the drawer (matches standard modal
 *     dismissal pattern).
 *   * Viewport resize past the 640 px breakpoint auto-closes the
 *     drawer so users don't end up with a stale "open" state when
 *     they rotate from portrait to landscape.
 *
 * No-ops when the optional toggle/backdrop elements are missing,
 * so older partial revisions (and any future template that doesn't
 * include the drawer scaffolding) just retain desktop behaviour.
 */
function initMobileDrawer() {
  const toggle   = document.getElementById("ps-mobile-toggle");
  const backdrop = document.getElementById("ps-mobile-backdrop");
  if (!toggle || !backdrop) return;

  const CLASS = "has-mobile-sidebar-open";
  const MOBILE_BREAKPOINT = 640;

  function setOpen(open) {
    document.body.classList.toggle(CLASS, open);
    toggle.setAttribute("aria-expanded", String(open));
    backdrop.setAttribute("aria-hidden", String(!open));
  }
  function toggleOpen() {
    setOpen(!document.body.classList.contains(CLASS));
  }
  toggle.addEventListener("click", toggleOpen);
  backdrop.addEventListener("click", () => setOpen(false));
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && document.body.classList.contains(CLASS)) {
      setOpen(false);
    }
  });
  // Resize past the breakpoint -> drawer becomes desktop sidebar; the
  // ``has-mobile-sidebar-open`` class is harmless at wide widths
  // (display:none on the affected elements), but clearing it keeps
  // ARIA + class state honest for screen readers.
  window.addEventListener("resize", () => {
    if (window.innerWidth > MOBILE_BREAKPOINT) setOpen(false);
  });
}


async function init() {
  const sidebar = document.getElementById("projects-sidebar");
  if (!sidebar) return;                  // page didn't include the partial

  // Wire the narrow-viewport drawer toggle.  Independent of
  // project-root resolution; runs first so the toggle is responsive
  // even if /api/files/roots is slow / fails (the user would see an
  // error in the sidebar list, but can still close the drawer to
  // reach the rest of the page).
  initMobileDrawer();

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
  // 2026-05-31 design § C7: wire navigateTo's public impl from
  // list.js's openDir so projects.navigateTo(absPath, opts) returns
  // the documented {ok, path, entries} envelope (or {ok:false,
  // error} on failure).
  // ORDER: setNavigateToImpl BEFORE setProjectsRoot because the
  // latter synchronously fires onProjectsRootResolved subscribers.
  // A pre-init subscriber that immediately calls projects.navigateTo
  // from inside its onRootResolved callback would otherwise hit the
  // "unavailable: sidebar not initialised" fallback.  Same reasoning
  // applies to initList() below (its onChange subscribers run synchronously
  // on register).
  setNavigateToImpl(openDir);
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
