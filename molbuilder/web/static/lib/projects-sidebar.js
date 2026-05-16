/* Persistent Projects sidebar -- "inquire model" v1.
 *
 * The sidebar's job:
 *   1. Browse the projects/ tree (single root from /api/files/roots).
 *   2. Track + publish two pieces of state:
 *        molbuilder.current_dir   -- the user's working directory
 *        molbuilder.current_file  -- the optional file selection
 *   3. Offer file-MANIPULATION buttons that operate on the projects
 *      tree itself (v1 ships: "+ New subdir").
 *
 * The sidebar's NON-job:
 *   * It does not know about tabs, file extensions, or how files
 *     get loaded into anything.  No "Open in <Tab>" buttons.
 *   * Each tab pulls from the public API below when its own UI
 *     triggers a load / generate.
 *
 * Public API (the "Inquire" surface; spec'd in docs/protocols/selection.md):
 *
 *   window.molbuilder.projects.getCurrentDir()    -- always inside projects/
 *   window.molbuilder.projects.getCurrentFile()   -- "" if nothing selected
 *   window.molbuilder.projects.onChange(cb)       -- subscribe to selection changes
 *                                                    cb is called with ({dir, file})
 *                                                    returns an unsubscribe function
 *   window.molbuilder.projects.readCurrentFile()  -- async; fetch text via /api/files/read
 *                                                    returns {path, text} or null
 *   window.molbuilder.projects.relativeToProjects(path)
 *                                                 -- strip projects/ prefix for display
 *   window.molbuilder.projects.refresh()          -- re-list the current dir; call this
 *                                                    after a tab creates a new file
 *
 * Layout: fixed-position aside on the left; lib/projects-sidebar.js
 * measures the page <header> + .app-tabs nav heights on init + resize
 * and sets the sidebar's top: dynamically.  body { padding-left:
 * var(--ps-w) } shifts the main content right.
 */

(function () {
  "use strict";

  const SS_DIR  = "molbuilder.current_dir";
  const SS_FILE = "molbuilder.current_file";

  // ----- DOM refs (resolved after DOMContentLoaded) ------------- //
  let elCrumb, elList, elActions, elSidebar;
  let elMkdirForm, elMkdirInput, elMkdirError, elMkdirContext;
  let elNewProjForm, elNewProjInput, elNewProjError, elNewProjSubdirs;

  // The root path of `projects/`, resolved from /api/files/roots
  // at startup.
  let projectsRoot = null;

  // ----- Selection change subscribers --------------------------- //
  const subscribers = new Set();

  function publishChange() {
    const payload = {
      dir:  sessionStorage.getItem(SS_DIR)  || "",
      file: sessionStorage.getItem(SS_FILE) || "",
    };
    subscribers.forEach((cb) => {
      try { cb(payload); } catch (e) { /* a bad subscriber shouldn't kill the loop */ }
    });
    // Cross-window listeners get the standard 'storage' event for free.
  }

  function setShared(dir, file) {
    sessionStorage.setItem(SS_DIR,  dir  || "");
    sessionStorage.setItem(SS_FILE, file || "");
    publishChange();
    // Keep the "New subdir" form's context label in sync with where
    // the user is now (e.g. "spectrum/water_v2" instead of stale "/").
    updateMkdirContext();
  }

  // ----- API helpers (private; the public API at the bottom calls them) //
  async function apiRoots() {
    const r = await fetch("/api/files/roots");
    return (await r.json()).roots || [];
  }

  async function apiList(path) {
    const r = await fetch(
      "/api/files/list?path=" + encodeURIComponent(path)
    );
    return await r.json();
  }

  async function apiMkdir(parent, name) {
    const r = await fetch("/api/files/mkdir", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({parent: parent, name: name}),
    });
    return await r.json();
  }

  async function apiCreateProject(name) {
    const r = await fetch("/api/projects/create", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({name: name}),
    });
    return await r.json();
  }

  async function apiRead(path) {
    const r = await fetch("/api/files/read?path=" + encodeURIComponent(path));
    return await r.json();
  }

  // ----- Dynamic sidebar top measurement ------------------------ //

  function measureSidebarTop() {
    const headerEl = document.querySelector("body > header");
    const navEl    = document.querySelector("body > nav.app-tabs");
    let topPx = 0;
    if (headerEl) topPx += headerEl.offsetHeight;
    if (navEl)    topPx += navEl.offsetHeight;
    elSidebar.style.top = topPx + "px";
  }

  // ----- Rendering ---------------------------------------------- //

  function renderBreadcrumb(currentPath) {
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

  function renderList(entries, currentPath) {
    elList.innerHTML = "";
    elList.classList.toggle("is-empty", entries.length === 0);
    if (entries.length === 0) return;
    const selectedFile = sessionStorage.getItem(SS_FILE) || "";
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
        meta.textContent = humanSize(e.size);
        li.appendChild(meta);
      }
      li.addEventListener("click", () => {
        if (e.kind === "directory") {
          openDir(fullPath);
        } else {
          markSelected(li);
          setShared(currentPath, fullPath);
          renderSelectionStatus(e.name, fullPath);
        }
      });
      elList.appendChild(li);
    }
  }

  function markSelected(li) {
    elList.querySelectorAll(".ps-entry.is-selected")
          .forEach((n) => n.classList.remove("is-selected"));
    if (li) li.classList.add("is-selected");
  }

  function humanSize(n) {
    if (n < 1024) return n + " B";
    if (n < 1024 * 1024) return (n / 1024).toFixed(0) + " K";
    if (n < 1024 * 1024 * 1024) return (n / 1024 / 1024).toFixed(1) + " M";
    return (n / 1024 / 1024 / 1024).toFixed(1) + " G";
  }

  // ----- Selection status (just a display, no tab actions) ------ //

  function renderSelectionStatus(filename, fullPath) {
    const sel = elActions.querySelector(".ps-selection");
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

  // ----- File-manipulation handlers ----------------------------- //

  // Update the "(in <current-dir>)" hint next to the New-subdir field
  // every time the sidebar's current_dir changes, AND toggle the
  // section's visibility based on depth:
  //
  //   depth 0 (projects/ root)   -> hide the section.  Keep the root
  //                                 clean: only `+ New project` is
  //                                 meaningful there.
  //   depth 1+                   -> show the section.  At depth 1 the
  //                                 backend validator requires the
  //                                 name be a CANONICAL_TOPIC (incl.
  //                                 'user'); at depth 2+ any valid
  //                                 name is accepted.
  function updateMkdirContext() {
    if (!elMkdirContext) return;
    const dir = sessionStorage.getItem(SS_DIR) || projectsRoot || "";

    // Toggle the <details> section.  Container = the <details> element
    // that wraps the form; we walk up the DOM once at init time and
    // cache it for speed.
    const section = elMkdirForm ? elMkdirForm.closest("details") : null;
    if (section) {
      const atRoot = !projectsRoot
        || dir === projectsRoot
        || dir === projectsRoot.replace(/\/$/, "")
        || !dir;
      section.hidden = atRoot;
      // Force-close when hidden so re-show starts collapsed (avoids a
      // stale half-open animation when navigating in/out of root).
      if (atRoot) section.open = false;
    }

    elMkdirContext.textContent = dir
      ? (window.molbuilder.projects.relativeToProjects(dir) || "projects/")
      : "current directory";
    elMkdirContext.title = dir || "";
  }

  function resetMkdirForm() {
    elMkdirInput.value = "";
    elMkdirError.textContent = "";
  }

  async function submitMkdir(ev) {
    ev.preventDefault();
    elMkdirError.textContent = "";
    const name = elMkdirInput.value.trim();
    if (!name) {
      elMkdirError.textContent = "Name cannot be empty.";
      return;
    }
    const parent = sessionStorage.getItem(SS_DIR) || projectsRoot;
    const j = await apiMkdir(parent, name);
    if (!j.ok) {
      // Surface backend message (409 for conflict, 400 for bad name,
      // 403 for permission, etc.) directly so the user sees what's
      // wrong without translation.
      elMkdirError.textContent = j.error || "mkdir failed.";
      return;
    }
    resetMkdirForm();
    // Navigate into the newly-created dir so the user sees they're in it.
    await openDir(j.path);
  }

  function resetNewProjectForm() {
    elNewProjInput.value = "";
    elNewProjError.textContent = "";
  }

  async function submitNewProject(ev) {
    ev.preventDefault();
    elNewProjError.textContent = "";
    const name = elNewProjInput.value.trim();
    if (!name) {
      elNewProjError.textContent = "Name cannot be empty.";
      return;
    }
    const j = await apiCreateProject(name);
    if (!j.ok) {
      // 409 (name conflict) lands here with a helpful message;
      // 400 (invalid name) similarly.  Display verbatim.
      elNewProjError.textContent = j.error || "create failed.";
      return;
    }
    resetNewProjectForm();
    // Navigate into the new project so the user sees the skeleton.
    await openDir(j.path);
  }

  // ----- Navigation --------------------------------------------- //

  async function openDir(absPath) {
    const resp = await apiList(absPath);
    if (!resp.ok) {
      renderBreadcrumb(absPath);
      elList.innerHTML = "";
      elList.classList.add("is-empty");
      const li = document.createElement("li");
      li.style.cssText = "padding: 0.7rem; color: #e07a7a;";
      li.textContent = resp.error || "Failed to list directory.";
      elList.appendChild(li);
      setShared(absPath, "");
      renderSelectionStatus("", "");
      return;
    }
    renderBreadcrumb(resp.path);
    renderList(resp.entries, resp.path);
    setShared(resp.path, "");
    renderSelectionStatus("", "");
  }

  // ----- Init --------------------------------------------------- //

  async function init() {
    elSidebar = document.getElementById("projects-sidebar");
    if (!elSidebar) return;          // page doesn't include the partial

    elCrumb        = document.getElementById("ps-breadcrumb");
    elList         = document.getElementById("ps-list");
    elActions      = document.getElementById("ps-actions");
    // "+ New subdir" form (any depth).
    elMkdirForm    = document.getElementById("ps-mkdir-form");
    elMkdirInput   = document.getElementById("ps-mkdir-input");
    elMkdirError   = document.getElementById("ps-mkdir-error");
    elMkdirContext = document.querySelector(".ps-mkdir-context");
    // "+ New project" form (bootstraps the full skeleton).
    elNewProjForm    = document.getElementById("ps-newproject-form");
    elNewProjInput   = document.getElementById("ps-newproject-input");
    elNewProjError   = document.getElementById("ps-newproject-error");
    elNewProjSubdirs = document.getElementById("ps-newproject-subdirs");

    document.body.classList.add("has-projects-sidebar");
    measureSidebarTop();
    window.addEventListener("resize", measureSidebarTop);

    // Form wiring.  Each form lives inside a <details> so users see
    // the input only when they expand the section.
    if (elMkdirForm) elMkdirForm.addEventListener("submit", submitMkdir);
    const mkdirCancel = document.getElementById("ps-mkdir-cancel");
    if (mkdirCancel) mkdirCancel.addEventListener("click", resetMkdirForm);

    if (elNewProjForm) elNewProjForm.addEventListener("submit", submitNewProject);
    const newProjCancel = document.getElementById("ps-newproject-cancel");
    if (newProjCancel) newProjCancel.addEventListener("click", resetNewProjectForm);
    // Populate the "subdirs that will be created" note from the
    // canonical list (kept in JS to avoid an extra API roundtrip;
    // the backend still validates -- this is just a display hint).
    if (elNewProjSubdirs) {
      const CANONICAL_TOPICS_DISPLAY = [
        "structure", "pseudopotential",
        "optimization", "frequency", "spectrum",
        "transport", "single-point", "scan",
      ];
      elNewProjSubdirs.innerHTML = CANONICAL_TOPICS_DISPLAY
        .map((t) => `<code>${t}/</code>`).join(", ");
    }

    const roots = await apiRoots();
    if (roots.length === 0) {
      elList.classList.add("is-empty");
      elList.innerHTML = "<li style='padding:0.7rem;color:#e07a7a;'>"
                       + "No file-picker roots configured.</li>";
      return;
    }
    projectsRoot = roots[0].path;

    const lastDir = sessionStorage.getItem(SS_DIR) || "";
    const start = (lastDir && lastDir.startsWith(projectsRoot))
                ? lastDir : projectsRoot;
    await openDir(start);

    // If a file was already selected (cross-tab persistence), highlight it.
    const file = sessionStorage.getItem(SS_FILE) || "";
    if (file && file.startsWith(projectsRoot)) {
      const li = elList.querySelector(
        `.ps-entry[data-path="${cssEscape(file)}"]`
      );
      if (li) markSelected(li);
      renderSelectionStatus(file.split("/").pop(), file);
    }
  }

  function cssEscape(s) {
    if (typeof CSS !== "undefined" && CSS.escape) return CSS.escape(s);
    return s.replace(/["\\]/g, "\\$&");
  }

  // ----- Public API (the "Inquire" surface) --------------------- //

  window.molbuilder = window.molbuilder || {};
  window.molbuilder.projects = {
    getCurrentDir:  () => sessionStorage.getItem(SS_DIR)  || "",
    getCurrentFile: () => sessionStorage.getItem(SS_FILE) || "",
    onChange: (cb) => {
      subscribers.add(cb);
      // Fire once immediately so subscribers can initialise from the
      // current state without a separate getCurrent* call.
      try {
        cb({
          dir:  sessionStorage.getItem(SS_DIR)  || "",
          file: sessionStorage.getItem(SS_FILE) || "",
        });
      } catch (e) { /* swallow */ }
      return () => subscribers.delete(cb);
    },
    readCurrentFile: async () => {
      const path = sessionStorage.getItem(SS_FILE) || "";
      if (!path) return null;
      const j = await apiRead(path);
      if (!j.ok) return null;
      return {path: j.path, text: j.text};
    },
    relativeToProjects: (path) => {
      if (!path || !projectsRoot) return path || "";
      if (!path.startsWith(projectsRoot)) return path;
      return path.slice(projectsRoot.length).replace(/^\/+/, "") || "/";
    },
    refresh: async () => {
      const dir = sessionStorage.getItem(SS_DIR) || projectsRoot;
      if (!dir) return;
      await openDir(dir);
    },
  };

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
