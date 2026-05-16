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
  let elMkdirBtn, elMkdirForm, elMkdirInput, elMkdirError;

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

  // ----- "New subdirectory" file-manipulation handler ----------- //

  function showMkdirForm() {
    elMkdirForm.hidden = false;
    elMkdirError.textContent = "";
    elMkdirInput.value = "";
    elMkdirInput.focus();
  }

  function hideMkdirForm() {
    elMkdirForm.hidden = true;
    elMkdirError.textContent = "";
  }

  async function submitMkdir(ev) {
    ev.preventDefault();
    const name = elMkdirInput.value.trim();
    if (!name) {
      elMkdirError.textContent = "Name cannot be empty.";
      return;
    }
    const parent = sessionStorage.getItem(SS_DIR) || projectsRoot;
    const j = await apiMkdir(parent, name);
    if (!j.ok) {
      elMkdirError.textContent = j.error || "mkdir failed.";
      return;
    }
    hideMkdirForm();
    // Navigate into the newly-created dir so the user sees they're in it.
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

    elCrumb       = document.getElementById("ps-breadcrumb");
    elList        = document.getElementById("ps-list");
    elActions     = document.getElementById("ps-actions");
    elMkdirBtn    = document.getElementById("ps-mkdir-btn");
    elMkdirForm   = document.getElementById("ps-mkdir-form");
    elMkdirInput  = document.getElementById("ps-mkdir-input");
    elMkdirError  = document.getElementById("ps-mkdir-error");

    document.body.classList.add("has-projects-sidebar");
    measureSidebarTop();
    window.addEventListener("resize", measureSidebarTop);

    if (elMkdirBtn)  elMkdirBtn.addEventListener("click", showMkdirForm);
    if (elMkdirForm) elMkdirForm.addEventListener("submit", submitMkdir);
    const cancelBtn = document.getElementById("ps-mkdir-cancel");
    if (cancelBtn) cancelBtn.addEventListener("click", hideMkdirForm);

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
