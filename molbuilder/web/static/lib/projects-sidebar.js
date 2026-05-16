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
  let elUploadForm, elUploadInput, elUploadError, elUploadContext;
  let elPreviewModal, elPreviewTitle, elPreviewMeta;
  let elPreviewBody, elPreviewError;

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
    // Keep all sidebar UI bits in sync with the new selection:
    //   * mkdir form's "(in <dir>)" hint + depth-0 hide rule
    //   * upload form's "(lands in <dir>)" hint + same hide rule
    // (Per-entry preview/delete buttons live on the list rows
    //  themselves; no global state to sync for those.)
    updateMkdirContext();
    updateUploadContext();
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

  async function apiUpload(targetDir, file) {
    const fd = new FormData();
    fd.append("target_dir", targetDir);
    fd.append("file", file);
    const r = await fetch("/api/files/upload", {method: "POST", body: fd});
    // 501 still returns valid JSON; the inline error UX renders it.
    try { return await r.json(); }
    catch (_) {
      return {ok: false, error: "upload server returned non-JSON (status "
                                 + r.status + ")"};
    }
  }

  async function apiDelete(path, recursive) {
    const r = await fetch("/api/files/delete", {
      method: "DELETE",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({path: path, recursive: !!recursive}),
    });
    try { return await r.json(); }
    catch (_) {
      return {ok: false, error: "delete server returned non-JSON (status "
                                 + r.status + ")"};
    }
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

  // Names that may NOT be deleted via the sidebar -- ever -- because
  // doing so would orphan the projects layout.  Mirrors
  // CANONICAL_TOPICS (kept in JS to avoid an extra round-trip; the
  // backend will also refuse via _validate_subdir_name's depth rules).
  const _UNDELETABLE_AT_DEPTH_1 = new Set([
    "structure", "pseudopotential",
    "optimization", "frequency", "spectrum",
    "transport", "single-point", "scan",
    "user",
  ]);

  function _isDeletableEntry(entry, currentPath) {
    // At depth 0 (currentPath == projectsRoot): every entry is a
    // project; user goes to the shell to delete those.
    if (!projectsRoot) return false;
    if (currentPath === projectsRoot
        || currentPath === projectsRoot.replace(/\/$/, "")) {
      return false;
    }
    // At depth 1 (one level inside a project): canonical-topic dirs
    // are off-limits.  Detect "depth 1" by checking the parent path
    // is exactly projects/<one-segment>.
    const rel = currentPath.slice(projectsRoot.length).replace(/^\/+/, "");
    const depth = rel.split("/").filter(Boolean).length;
    if (depth === 1 && entry.kind === "directory"
        && _UNDELETABLE_AT_DEPTH_1.has(entry.name)) {
      return false;
    }
    return true;
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
      // Per-entry action buttons on hover (preview + delete).  Same
      // visual + behavioural pattern: tiny text/icon button revealed
      // by the entry's :hover; click stops propagation so it doesn't
      // also select / navigate; backend may or may not be implemented
      // (delete is stubbed today).  Order: preview first, delete last
      // so destructive action is consistently on the far right.

      if (e.kind === "file") {
        const view = document.createElement("button");
        view.type = "button";
        view.className = "ps-entry-action ps-entry-preview";
        view.textContent = "view";
        view.title = "Preview " + e.name;
        view.addEventListener("click", (ev) => {
          ev.stopPropagation();
          // Mark + publish selection so the preview reads the right
          // file (in case the user clicked view on a non-selected
          // entry); then open the modal.
          markSelected(li);
          setShared(currentPath, fullPath);
          renderSelectionStatus(e.name, fullPath);
          showPreview();
        });
        li.appendChild(view);
      }

      if (_isDeletableEntry(e, currentPath)) {
        const del = document.createElement("button");
        del.type = "button";
        del.className = "ps-entry-action ps-entry-delete";
        del.textContent = "×";   // ×
        del.title = "Delete " + e.name;
        del.addEventListener("click", (ev) => {
          ev.stopPropagation();
          confirmAndDelete(fullPath, e);
        });
        li.appendChild(del);
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

  async function confirmAndDelete(fullPath, entry) {
    const what = entry.kind === "directory" ? "directory" : "file";
    // Native confirm() is enough for v1: the stub backend returns
    // 501 anyway so the user never gets to the destructive step.
    // When the real backend lands we replace this with a modal.
    if (!window.confirm(
      "Delete " + what + " '" + entry.name + "'?\n\n"
      + "This cannot be undone."
    )) return;
    const j = await apiDelete(fullPath, entry.kind === "directory");
    if (!j.ok) {
      // Today: 501 stub message.  Tomorrow: real backend's 403/409/etc.
      window.alert(j.error || "Delete failed.");
      return;
    }
    // Future: refresh the listing after a successful delete.
    const dir = sessionStorage.getItem(SS_DIR) || projectsRoot;
    if (dir) await openDir(dir);
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

  // ----- Upload handler (stub) ---------------------------------- //
  // POSTs multipart to /api/files/upload which currently 501s; the
  // 501 message lands in the inline error slot exactly like the
  // future real implementation's 400/409 would.  No special-case
  // handling needed when the real backend lands.

  function resetUploadForm() {
    if (elUploadInput) elUploadInput.value = "";
    if (elUploadError) elUploadError.textContent = "";
  }

  function updateUploadContext() {
    if (!elUploadContext) return;
    const dir = sessionStorage.getItem(SS_DIR) || projectsRoot || "";
    elUploadContext.textContent = dir
      ? (window.molbuilder.projects.relativeToProjects(dir) || "projects/")
      : "current directory";
    elUploadContext.title = dir || "";
    // Same depth-0 hiding as + New subdir: no upload at projects/ root.
    const section = elUploadForm ? elUploadForm.closest("details") : null;
    if (section) {
      const atRoot = !projectsRoot
        || dir === projectsRoot
        || dir === projectsRoot.replace(/\/$/, "")
        || !dir;
      section.hidden = atRoot;
      if (atRoot) section.open = false;
    }
  }

  async function submitUpload(ev) {
    ev.preventDefault();
    elUploadError.textContent = "";
    if (!elUploadInput.files || elUploadInput.files.length === 0) {
      elUploadError.textContent = "Pick a file to upload first.";
      return;
    }
    const file = elUploadInput.files[0];
    const target = sessionStorage.getItem(SS_DIR) || projectsRoot;
    const j = await apiUpload(target, file);
    if (!j.ok) {
      // Today this is the 501 message from the stub; tomorrow it's
      // the real backend's 409 / 400 / 403 -- same code path either
      // way, no special-case branch.
      elUploadError.textContent = j.error || "upload failed.";
      return;
    }
    resetUploadForm();
    // After a real upload lands the file in current_dir, refresh the
    // listing so the user sees it.  Today this branch is unreached
    // (501), but the future-implementation hook is ready.
    await openDir(target);
  }

  // ----- File-preview modal (view: functional; save: stub) ------ //

  function openPreviewModal() {
    if (!elPreviewModal) return;
    elPreviewModal.hidden = false;
    document.addEventListener("keydown", _previewKeydown);
  }

  function closePreviewModal() {
    if (!elPreviewModal) return;
    elPreviewModal.hidden = true;
    document.removeEventListener("keydown", _previewKeydown);
  }

  function _previewKeydown(ev) {
    if (ev.key === "Escape") closePreviewModal();
  }

  async function showPreview() {
    const path = sessionStorage.getItem(SS_FILE) || "";
    if (!path) return;
    elPreviewTitle.textContent = path.split("/").pop();
    elPreviewMeta.textContent  = path;
    elPreviewBody.textContent  = "Loading...";
    elPreviewError.textContent = "";
    openPreviewModal();
    const payload = await window.molbuilder.projects.readCurrentFile();
    if (!payload) {
      elPreviewBody.textContent = "";
      // The /api/files/read endpoint surfaces 413 (too large) + 400
      // (non-UTF-8) + 404 with specific messages; use stat to fetch
      // the actual message instead of a generic fallback.
      try {
        const r = await fetch(
          "/api/files/read?path=" + encodeURIComponent(path)
        );
        const j = await r.json();
        elPreviewError.textContent = j.error || "Failed to read file.";
      } catch (e) {
        elPreviewError.textContent = "Network error reading file: " + e.message;
      }
      return;
    }
    elPreviewBody.textContent = payload.text;
  }

  // (refreshPreviewButton retired in v5.1 -- the Preview affordance is
  //  now per-entry on hover in the list, not a separate bottom-bar
  //  button.  setShared() no longer calls it.)

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

    // Upload form (stub backend; UX is wired).
    elUploadForm    = document.getElementById("ps-upload-form");
    elUploadInput   = document.getElementById("ps-upload-input");
    elUploadError   = document.getElementById("ps-upload-error");
    elUploadContext = document.querySelector(".ps-upload-context");
    if (elUploadForm) elUploadForm.addEventListener("submit", submitUpload);
    const uploadCancel = document.getElementById("ps-upload-cancel");
    if (uploadCancel) uploadCancel.addEventListener("click", resetUploadForm);

    // File-preview modal (view: functional; save: stub).  The Preview
    // trigger is per-entry (renderList attaches a "view" button to
    // each file row on hover), not a separate sidebar-level button.
    elPreviewModal = document.getElementById("ps-preview-modal");
    elPreviewTitle = document.getElementById("ps-preview-title");
    elPreviewMeta  = document.getElementById("ps-preview-meta");
    elPreviewBody  = document.getElementById("ps-preview-body");
    elPreviewError = document.getElementById("ps-preview-error");
    if (elPreviewModal) {
      elPreviewModal.querySelectorAll(
        ".ps-preview-close, .ps-preview-close-footer, .ps-preview-backdrop"
      ).forEach((n) => n.addEventListener("click", closePreviewModal));
    }
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
