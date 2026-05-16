/* Persistent Projects sidebar.
 *
 * Layout: single-column directory listing with a clickable breadcrumb
 * at the top.  Click a directory entry → drill into it.  Click a
 * crumb → jump back to that level.  Click a file → select it (writes
 * shared sessionStorage state for subscriber tabs to react).
 *
 * Single root: `projects/` (from Capabilities.file_picker_roots,
 * which now returns just that single entry).  No CWD, no
 * user-configurable roots in v1.
 *
 * Shared state (sessionStorage; cross-tab via 'storage' event,
 * same-tab via 'molbuilder.selection' CustomEvent):
 *   molbuilder.current_dir   = absolute path of the displayed directory
 *   molbuilder.current_file  = absolute path of the selected file ("" if none)
 *
 * Subscriber tabs read this state via lib/projects-selection.js.
 *
 * The sidebar is hydrated on DOMContentLoaded; the calling template
 * just needs to include _projects_sidebar.html + this script.
 */

(function () {
  "use strict";

  const SS_DIR   = "molbuilder.current_dir";
  const SS_FILE  = "molbuilder.current_file";
  const SS_COLLAPSED = "molbuilder.sidebar_collapsed";

  // ----- DOM refs (resolved after DOMContentLoaded) ------------- //
  let elCrumb, elList, elSelection, elToggle, elSidebar;

  // The root path of `projects/`, resolved from /api/files/roots
  // at startup.  All paths displayed in the sidebar are inside this.
  let projectsRoot = null;

  // ----- API helpers -------------------------------------------- //
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

  // ----- Shared-state writers ----------------------------------- //
  function setShared(dir, file) {
    sessionStorage.setItem(SS_DIR,  dir  || "");
    sessionStorage.setItem(SS_FILE, file || "");
    window.dispatchEvent(new CustomEvent("molbuilder.selection", {
      detail: {dir: dir || "", file: file || ""},
    }));
  }

  // ----- Rendering ---------------------------------------------- //

  function renderBreadcrumb(currentPath) {
    elCrumb.innerHTML = "";
    if (!projectsRoot) return;

    // Build the sequence of crumb hops from projectsRoot to currentPath.
    // Each crumb shows a single path segment; the first is always
    // "projects".
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

    // Track the currently-selected file so re-renders preserve highlight.
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
          updateSelectionLabel();
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

  function updateSelectionLabel() {
    const file = sessionStorage.getItem(SS_FILE) || "";
    const dir  = sessionStorage.getItem(SS_DIR)  || "";
    let display = "(none)";
    if (file && projectsRoot) {
      display = file.startsWith(projectsRoot)
        ? file.slice(projectsRoot.length).replace(/^\/+/, "") || file
        : file;
    } else if (dir && projectsRoot) {
      display = dir.startsWith(projectsRoot)
        ? dir.slice(projectsRoot.length).replace(/^\/+/, "") || "/"
        : dir;
    }
    elSelection.textContent = display;
    elSelection.title = file || dir || "";
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
      updateSelectionLabel();
      return;
    }
    renderBreadcrumb(resp.path);
    renderList(resp.entries, resp.path);
    // Update the current dir; clear file selection because we just
    // navigated (selection of a file is an explicit click below).
    setShared(resp.path, "");
    updateSelectionLabel();
  }

  // ----- Collapse toggle ---------------------------------------- //

  function applyCollapsed(collapsed) {
    document.body.classList.toggle("sidebar-collapsed", !!collapsed);
    elToggle.setAttribute("aria-expanded", String(!collapsed));
    sessionStorage.setItem(SS_COLLAPSED, collapsed ? "1" : "0");
  }

  // ----- Init --------------------------------------------------- //

  async function init() {
    elSidebar   = document.getElementById("projects-sidebar");
    if (!elSidebar) return;        // page doesn't include the partial

    elCrumb     = document.getElementById("ps-breadcrumb");
    elList      = document.getElementById("ps-list");
    elSelection = document.getElementById("ps-selection");
    elToggle    = document.getElementById("ps-toggle");

    document.body.classList.add("has-projects-sidebar");
    // Restore prior collapse state.
    if (sessionStorage.getItem(SS_COLLAPSED) === "1") {
      applyCollapsed(true);
    }
    elToggle.addEventListener("click", () => {
      const isNowCollapsed = !document.body.classList.contains("sidebar-collapsed");
      applyCollapsed(isNowCollapsed);
    });

    // Resolve projects/ from the API (single root).
    const roots = await apiRoots();
    if (roots.length === 0) {
      elList.classList.add("is-empty");
      elList.innerHTML = "<li style='padding:0.7rem;color:#e07a7a;'>"
                       + "No file-picker roots configured.</li>";
      return;
    }
    projectsRoot = roots[0].path;

    // Navigate to the previously-displayed dir if it's still inside
    // projects/, else start at the root.
    const lastDir = sessionStorage.getItem(SS_DIR) || "";
    const start = (lastDir && lastDir.startsWith(projectsRoot))
                ? lastDir : projectsRoot;
    await openDir(start);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
