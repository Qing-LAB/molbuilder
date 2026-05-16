/* Persistent Projects sidebar.
 *
 * Layout: single-column directory listing with a clickable breadcrumb
 * at the top.  Click a directory → drill in.  Click a crumb → jump
 * back to that level.  Click a file → select it (writes shared
 * sessionStorage state; sidebar then renders contextual "Open in
 * <Tab>" buttons in the actions area).
 *
 * v2 (this revision):
 *   * Drop the collapse toggle -- always open.
 *   * JS-measure header + nav heights on load + resize so the
 *     sidebar's `top:` matches the actual rendered offset (the
 *     hardcoded `top: 3rem` of v1 didn't account for the page
 *     header above the app-tabs nav).
 *   * Replace the per-tab "Use this file" banner pattern with
 *     contextual "Open in <Tab>" buttons in the sidebar itself.
 *     Buttons navigate via `location.href = "/<tab>"`; the target
 *     tab's auto-load hook (registered via window.molbuilderTabAutoLoad
 *     by each tab's own JS) reads sessionStorage.molbuilder.current_file
 *     on init and fires the tab's Load flow if the file type matches.
 *
 * Single root: `projects/` (Capabilities.file_picker_roots() returns
 * just that single entry).  No CWD, no user-configurable additions.
 *
 * Shared state (sessionStorage):
 *   molbuilder.current_dir   = absolute path of the displayed directory
 *   molbuilder.current_file  = absolute path of the selected file ("" if none)
 */

(function () {
  "use strict";

  const SS_DIR   = "molbuilder.current_dir";
  const SS_FILE  = "molbuilder.current_file";

  // Map of (extension -> [{tab_path, label}, ...]).  The sidebar
  // renders one "Open in <Tab>" button per entry whose extension
  // matches the selected file's name.  Order matters because
  // ".spectra.json" is checked before ".json" via length-sorted keys.
  const OPEN_TARGETS = {
    ".xyz":          [{tab: "/modify",  label: "Open in Modify"},
                      {tab: "/",        label: "Open in Build"}],
    ".pdb":          [{tab: "/modify",  label: "Open in Modify"}],
    ".molwatch.log": [{tab: "/watch",   label: "Open in Watch"}],
    ".spectra.json": [{tab: "/spectra", label: "Open in Spectra"}],
    ".log":          [{tab: "/watch",   label: "Open in Watch"}],
    ".out":          [{tab: "/watch",   label: "Open in Watch"}],
  };
  const EXT_KEYS = Object.keys(OPEN_TARGETS).sort(
    (a, b) => b.length - a.length    // longest first: ".spectra.json" > ".json"
  );

  function pickTargets(name) {
    const lower = name.toLowerCase();
    for (const ext of EXT_KEYS) {
      if (lower.endsWith(ext)) return OPEN_TARGETS[ext];
    }
    return [];
  }

  // ----- DOM refs (resolved after DOMContentLoaded) ------------- //
  let elCrumb, elList, elActions, elActionsHint, elSidebar;

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
    // Tabs that mount auto-loaders subscribe to this CustomEvent
    // (same-window).  Cross-window listeners can also use the
    // standard 'storage' event.
    window.dispatchEvent(new CustomEvent("molbuilder.selection", {
      detail: {dir: dir || "", file: file || ""},
    }));
  }

  // ----- Dynamic sidebar top measurement ------------------------ //

  function measureSidebarTop() {
    // Top of sidebar = bottom of the last non-sidebar element above
    // the main content (page <header> + .app-tabs nav).  Measuring
    // both elements handles a wrapped multi-line tagline correctly.
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
          renderActions(e.name, fullPath);
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

  // ----- Action buttons (the "what now?" panel) ----------------- //

  function renderActions(filename, fullPath) {
    elActions.innerHTML = "";

    const sel = document.createElement("div");
    sel.className = "ps-action-selected";
    sel.innerHTML = "Selected: <strong></strong>";
    sel.querySelector("strong").textContent = filename;
    sel.title = fullPath;
    elActions.appendChild(sel);

    const targets = pickTargets(filename);
    if (targets.length === 0) {
      const hint = document.createElement("p");
      hint.className = "ps-actions-hint";
      hint.textContent = "No quick-open target for this file type.";
      elActions.appendChild(hint);
      return;
    }

    const currentPath = window.location.pathname;
    for (const t of targets) {
      const btn = document.createElement("button");
      btn.className = "ps-action-btn"
        + (t.tab === currentPath ? " is-current-tab" : "");
      btn.type = "button";
      btn.textContent = t.tab === currentPath
        ? "Load here (" + t.label.replace(/^Open in /, "") + ")"
        : t.label;
      btn.addEventListener("click", () => triggerOpenIn(t.tab));
      elActions.appendChild(btn);
    }
  }

  function triggerOpenIn(tabPath) {
    // The target tab's auto-load hook (registered as
    // window.molbuilderTabAutoLoad) reads sessionStorage.current_file
    // on init.  If we're already on that tab, fire the auto-load
    // directly without navigating.
    const currentPath = window.location.pathname;
    if (tabPath === currentPath && window.molbuilderTabAutoLoad) {
      window.molbuilderTabAutoLoad();
      return;
    }
    window.location.href = tabPath;
  }

  function clearActions() {
    elActions.innerHTML = "";
    const hint = document.createElement("p");
    hint.className = "ps-actions-hint";
    hint.textContent = "Click a file above to see open actions.";
    elActions.appendChild(hint);
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
      clearActions();
      return;
    }
    renderBreadcrumb(resp.path);
    renderList(resp.entries, resp.path);
    setShared(resp.path, "");
    clearActions();
  }

  // ----- Init --------------------------------------------------- //

  async function init() {
    elSidebar = document.getElementById("projects-sidebar");
    if (!elSidebar) return;          // page doesn't include the partial

    elCrumb       = document.getElementById("ps-breadcrumb");
    elList        = document.getElementById("ps-list");
    elActions     = document.getElementById("ps-actions");
    elActionsHint = document.getElementById("ps-actions-hint");

    document.body.classList.add("has-projects-sidebar");
    measureSidebarTop();
    window.addEventListener("resize", measureSidebarTop);

    // Resolve projects/ from the API (single root).
    const roots = await apiRoots();
    if (roots.length === 0) {
      elList.classList.add("is-empty");
      elList.innerHTML = "<li style='padding:0.7rem;color:#e07a7a;'>"
                       + "No file-picker roots configured.</li>";
      return;
    }
    projectsRoot = roots[0].path;

    // Restore prior dir from sessionStorage if it's still inside projects/.
    const lastDir = sessionStorage.getItem(SS_DIR) || "";
    const start = (lastDir && lastDir.startsWith(projectsRoot))
                ? lastDir : projectsRoot;
    await openDir(start);

    // If a file was already selected (e.g., user navigated here from
    // another tab via "Open in X"), keep its highlight + show actions.
    const file = sessionStorage.getItem(SS_FILE) || "";
    if (file && file.startsWith(projectsRoot)) {
      // Re-mark highlight inside the rendered list.
      const li = elList.querySelector(
        `.ps-entry[data-path="${cssEscape(file)}"]`
      );
      if (li) markSelected(li);
      renderActions(file.split("/").pop(), file);
    }
  }

  // Minimal CSS.escape polyfill (just enough for paths -- escape
  // backslash + quote characters that could break the attribute
  // selector).  Modern browsers have CSS.escape; this is a fallback.
  function cssEscape(s) {
    if (typeof CSS !== "undefined" && CSS.escape) return CSS.escape(s);
    return s.replace(/["\\]/g, "\\$&");
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
