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
 *   mutation-bar.js -- New project / New folder / Upload action bar
 *   preview.js   -- file-preview modal
 *   parser.js    -- format-aware structure doors (openMolecule /
 *                   saveMolecule)
 *   checkpoint.js -- run-history panel (checkpoint-domain consumer)
 *
 * This entry file imports each module, mounts the public API on
 * window, and runs the bootstrap (resolve projects/ root via
 * /api/files/roots, then list whichever dir the user was last in).
 *
 * Loaded via `<script type="module">` -- ES modules are supported
 * natively in every modern browser; no bundler.
 */

import { apiRoots } from "./api.js";
import {
  projects, setProjectsRoot, SS_DIR, setNavigateToImpl, readSelectionSlot,
} from "./state.js";
import {
  initList, initLockUI, openDir, restoreSelection,
} from "./list.js";
import { initForms } from "./mutation-bar.js";
import { initPreview } from "./preview.js";
import { parser } from "./parser.js";
import { molviewFiles } from "./molview-doors.js";
import { chooseSavePath } from "./dialogs.js";
import { initCheckpointPanel,
         status as ckStatus, init as ckInit, saveState as ckSaveState }
    from "./checkpoint.js";

window.molbuilder = window.molbuilder || {};
// The format-aware sub-namespace: projects.parser.openMolecule / saveMolecule (the
// structure-file DOORS).  Attached alongside the format-blind readFile/writeFile so the
// whole file-handling surface lives under ONE `projects` namespace
// (docs/model/structure.md).
projects.parser = parser;
// The MolView files door (`docs/web/projects.md` § 5): the ONE implementation
// of the `files` option every viewer mount hands in -- exports become files
// here, never inside the viewer (molview.md § 11.4).
projects.molviewFiles = molviewFiles;
// The unified "save it WHERE, as WHAT" question (projects.md § 5): one door
// for every flow that deposits artifacts under the project root -- MolView's
// exports today, multi-result consolidation (e.g. transport inputs) tomorrow.
projects.chooseSavePath = chooseSavePath;
// The checkpoint sub-namespace (`docs/web/projects.md` § 5), attached beside
// `parser` for the same reason: the whole surface a tab may use lives under ONE
// `projects` namespace.  Restore and tag are deliberately absent -- they are
// decisions taken at the panel, looking at the history.
projects.checkpoint = {
    status:    ckStatus,
    init:      ckInit,
    saveState: ckSaveState,
};
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
 * class on ``<body>``.  The CSS in ``lib/projects/projects-sidebar.css``
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
/**
 * Desktop collapse toggle (B.5.4).  Hide/show the sidebar on
 * viewports >= 640 px (the mobile drawer handles narrow ones).
 * Wires:
 *   - #ps-collapse-toggle (edge tab on the sidebar's right edge) → click to hide
 *   - #ps-collapsed-handle (edge tab at the screen's left edge) → click to show
 *     (symmetric twins: both ``.ps-edge-tab`` at the same mid-height)
 *
 * State persists per origin in sessionStorage under
 * ``molbuilder.projects_sidebar_collapsed`` so a refresh keeps the
 * preference within a tab.  Initial state is read on mount BEFORE
 * any layout-sensitive widget paints so 3Dmol / Plotly measure the
 * correct geometry.
 *
 * No-ops gracefully when the toggle elements are missing (older
 * partial revisions, future templates that don't include the
 * collapse scaffolding).
 */
const COLLAPSED_KEY = "molbuilder.projects_sidebar_collapsed";
function initDesktopCollapse() {
  const toggle = document.getElementById("ps-collapse-toggle");
  const handle = document.getElementById("ps-collapsed-handle");
  if (!toggle || !handle) return;

  const CLASS = "is-projects-sidebar-collapsed";

  function setCollapsed(collapsed) {
    document.body.classList.toggle(CLASS, collapsed);
    toggle.setAttribute("aria-expanded", String(!collapsed));
    handle.setAttribute("aria-expanded", String(!collapsed));
    try {
      if (collapsed) sessionStorage.setItem(COLLAPSED_KEY, "1");
      else           sessionStorage.removeItem(COLLAPSED_KEY);
    } catch (_) {
      // sessionStorage may throw on quota / private-mode; the
      // collapse state still applies to the current page —
      // it just won't survive a refresh.
    }
  }
  toggle.addEventListener("click",
    () => setCollapsed(true));
  handle.addEventListener("click",
    () => setCollapsed(false));
}

/**
 * Read the persisted collapsed-state and apply it BEFORE the
 * sidebar JS init runs.  Called synchronously during module init
 * so the body class is set before any layout-sensitive widget
 * (Plotly, 3Dmol, CSS-grid auto-fit) takes its first measurement.
 */
function _restoreCollapsedState() {
  let collapsed = false;
  try { collapsed = sessionStorage.getItem(COLLAPSED_KEY) === "1"; }
  catch (_) { /* swallow */ }
  if (collapsed) {
    document.body.classList.add("is-projects-sidebar-collapsed");
    const toggle = document.getElementById("ps-collapse-toggle");
    const handle = document.getElementById("ps-collapsed-handle");
    if (toggle) toggle.setAttribute("aria-expanded", "false");
    if (handle) handle.setAttribute("aria-expanded", "false");
  }
}

/**
 * Sidebar width resize (2026-06-12).
 *
 * Drag the .ps-resize-handle to change --ps-w live.  Width persists
 * to localStorage so it survives reloads + new tabs from the same
 * origin (matches the typical browser-state contract for layout
 * preferences -- the user picks a width once, then it stays).
 *
 * Bounds:
 *   min 14rem  (entries stay readable; below this the
 *               two-column breadcrumb breaks)
 *   max 40rem  (sidebar doesn't dominate a 1080p viewport)
 *
 * Restoration runs in ``_restoreWidth`` BEFORE the layout-sensitive
 * widgets (Plotly, 3Dmol) paint -- same reasoning as the
 * collapsed-state restore.  Without that, those widgets measure the
 * default 18rem geometry and then mis-layout when the saved width
 * gets applied later.
 */
const WIDTH_KEY = "molbuilder.projects_sidebar_width";
const WIDTH_MIN_PX = 14 * 16;   // 14rem at the 16px default font
const WIDTH_MAX_PX = 40 * 16;   // 40rem

function _clampWidth(px) {
  if (!Number.isFinite(px)) return null;
  return Math.max(WIDTH_MIN_PX, Math.min(WIDTH_MAX_PX, Math.round(px)));
}

function _setWidth(px) {
  const clamped = _clampWidth(px);
  if (clamped === null) return;
  document.documentElement.style.setProperty(
    "--ps-w", clamped + "px");
}

function _restoreWidth() {
  let saved = null;
  try { saved = parseFloat(localStorage.getItem(WIDTH_KEY)); }
  catch (_) { /* localStorage may throw in private mode */ }
  if (Number.isFinite(saved)) _setWidth(saved);
}

function initWidthResize() {
  const handle = document.getElementById("ps-resize-handle");
  if (!handle) return;

  let dragging = false;
  let startX = 0;
  let startW = 0;

  function onDown(e) {
    // Only handle primary button (mouse / touch primary).
    if (e.button !== undefined && e.button !== 0) return;
    dragging = true;
    startX = e.clientX;
    // Read current width from the CSS var (resolved px) rather than
    // re-deriving from the sidebar element so a mid-drag click on
    // the handle without movement doesn't snap to a different value.
    const computed = getComputedStyle(document.documentElement)
      .getPropertyValue("--ps-w").trim();
    if (computed.endsWith("px")) {
      startW = parseFloat(computed);
    } else if (computed.endsWith("rem")) {
      startW = parseFloat(computed) * 16;
    } else {
      const sidebar = document.getElementById("projects-sidebar");
      startW = sidebar ? sidebar.getBoundingClientRect().width : 18 * 16;
    }
    document.body.classList.add("is-projects-sidebar-resizing");
    e.preventDefault();
  }
  function onMove(e) {
    if (!dragging) return;
    const delta = e.clientX - startX;
    _setWidth(startW + delta);
  }
  function onUp() {
    if (!dragging) return;
    dragging = false;
    document.body.classList.remove("is-projects-sidebar-resizing");
    // Persist the new width.  Read from the CSS var so the saved
    // value matches what's actually applied (post-clamp).
    const computed = getComputedStyle(document.documentElement)
      .getPropertyValue("--ps-w").trim();
    try { localStorage.setItem(WIDTH_KEY, computed); }
    catch (_) { /* swallow */ }
  }

  handle.addEventListener("mousedown", onDown);
  window.addEventListener("mousemove", onMove);
  window.addEventListener("mouseup",   onUp);
  // Double-click resets to the default 18rem.  Quick out for users
  // who dragged too far + can't find the handle's intended size.
  handle.addEventListener("dblclick", () => {
    try { localStorage.removeItem(WIDTH_KEY); }
    catch (_) { /* swallow */ }
    document.documentElement.style.removeProperty("--ps-w");
  });
}

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

  // Restore the desktop-collapse state from sessionStorage AND the
  // persisted width from localStorage BEFORE any further wiring so
  // layout-sensitive widgets (Plotly, 3Dmol, CSS-grid) measure the
  // correct geometry on their first paint.  The body class controls
  // padding-left + sidebar visibility; --ps-w controls width.
  _restoreCollapsedState();
  _restoreWidth();

  // Wire the narrow-viewport drawer toggle + the desktop collapse
  // toggle.  Independent of project-root resolution; runs first so
  // the toggles are responsive even if /api/files/roots is slow /
  // fails (the user would see an error in the sidebar list, but
  // can still close the drawer / sidebar to reach the rest of the
  // page).
  initMobileDrawer();
  initDesktopCollapse();
  initWidthResize();

  // Wire the lock UI FIRST -- before any await that could throw or
  // bail.  The lock UI needs to work regardless of project-root
  // resolution; see initLockUI() docstring in list.js for the
  // 2026-05-28 background.  If we put this after the apiRoots()
  // await, a slow / failed roots call leaves the lock UI unwired
  // and lock() becomes a silent no-op visually.
  initLockUI();

  // NOTE: `data-sidebars="projects"` is set on <body> in each template
  // (server-side) -- NOT here.  The app-shell layout (lib/page-shell.css) is
  // pure CSS off that attribute, so the FIRST paint already has the correct
  // sidebar-beside-content geometry; adding it via JS would race the initial
  // paint and make layout-sensitive widgets (Plotly, 3Dmol) measure a wrong
  // geometry until the next resize.
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
  // Run-history panel (PR-B Phase 2/3).  Idempotent; no-ops on pages
  // that don't include the projects sidebar template.
  initCheckpointPanel();

  // Navigate to the previously-visited dir if it's still inside
  // projects/, else start at the root.
  const lastDir = readSelectionSlot(SS_DIR);
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
