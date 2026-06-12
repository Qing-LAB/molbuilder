/* projects/forms.js -- mutation menu wiring.
 *
 * 2026-06-12: replaced the foldable <details> create-form sections
 * with a single "+" button + dropdown menu.  Each menu item opens a
 * modal dialog from projects/dialogs.js.  Backend calls go through
 * projects.* (state.js), which dispatches to projects/api.js + fires
 * a directory refresh on success.
 *
 * Depth-aware item disable (driven by projects.onChange):
 *   * New project    -- always enabled
 *   * New folder     -- disabled at projects/ root (no useful parent)
 *   * Upload file    -- disabled at projects/ root
 *
 * Spec: docs/protocols/projects-sidebar.md § Mutation UX.
 *
 * The file is still named ``forms.js`` to keep its import path
 * (``./forms.js``) stable for ``projects-sidebar.js``; the inline-
 * form era ended in this commit.
 */

import {
  chooseName, chooseUploadFile,
} from "./dialogs.js";
import {
  projects, getProjectsRoot, SS_DIR, atProjectsRoot,
} from "./state.js";
import { openDir } from "./list.js";

let elBtn, elMenu;

function _closeMenu() {
  if (!elMenu) return;
  elMenu.hidden = true;
  if (elBtn) elBtn.setAttribute("aria-expanded", "false");
}

function _openMenu() {
  if (!elMenu) return;
  elMenu.hidden = false;
  if (elBtn) elBtn.setAttribute("aria-expanded", "true");
  _updateItemEnablement();
  // Focus the first enabled item so keyboard nav works.
  const firstEnabled = elMenu.querySelector(
    ".ps-create-menu-item:not([disabled])");
  if (firstEnabled) {
    try { firstEnabled.focus(); } catch (_) {}
  }
}

function _toggleMenu() {
  if (!elMenu) return;
  if (elMenu.hidden) _openMenu(); else _closeMenu();
}

function _updateItemEnablement() {
  if (!elMenu) return;
  const dir = sessionStorage.getItem(SS_DIR) || getProjectsRoot() || "";
  const root = atProjectsRoot(dir);
  // "New project" always enabled (it lands at root regardless).
  // "New folder" + "Upload" require a project context.
  elMenu.querySelectorAll(".ps-create-menu-item").forEach((item) => {
    const action = item.dataset.action;
    const needsDir = action === "new-folder" || action === "upload";
    item.disabled = needsDir && root;
    item.title = item.disabled
      ? "Pick a project folder in the sidebar first."
      : "";
  });
}

async function _doNewProject() {
  const name = await chooseName({
    title:        "New project",
    label:        "Project name",
    hint:         "Creates projects/<name>/ with the canonical "
                  + "subdirs (structure, optimization, spectrum, …).",
    placeholder:  "letters / digits / _ / -",
    confirmLabel: "Create",
  });
  if (!name) return;
  const r = await projects.createProject(name);
  if (r && r.ok && r.path) {
    await openDir(r.path);
  } else {
    window.alert((r && r.error) || "Project creation failed.");
  }
}

async function _doNewFolder() {
  const currentDir = sessionStorage.getItem(SS_DIR) || getProjectsRoot();
  if (!currentDir || atProjectsRoot(currentDir)) {
    window.alert(
      "Cannot create a folder at the projects root.  "
      + "Pick a project in the sidebar first.",
    );
    return;
  }
  const ctx = projects.relativeToProjects(currentDir) || "projects/";
  const name = await chooseName({
    title:        "New folder",
    label:        "Folder name",
    hint:         `Created inside ${ctx}.`,
    placeholder:  "letters / digits / _ / -",
    confirmLabel: "Create",
  });
  if (!name) return;
  const r = await projects.mkdir(currentDir, name);
  if (r && r.aborted) return;
  if (r && r.ok && r.path) {
    await openDir(r.path);
  } else {
    window.alert((r && r.error) || "Folder creation failed.");
  }
}

async function _doUpload() {
  const currentDir = sessionStorage.getItem(SS_DIR) || getProjectsRoot();
  if (!currentDir || atProjectsRoot(currentDir)) {
    window.alert(
      "Cannot upload to the projects root.  Pick a project in the "
      + "sidebar first.",
    );
    return;
  }
  const ctx = projects.relativeToProjects(currentDir) || "projects/";
  const file = await chooseUploadFile({ contextDir: ctx });
  if (!file) return;
  const r = await projects.upload(currentDir, file);
  if (r && r.aborted) return;
  if (!r || !r.ok) {
    window.alert((r && r.error) || "Upload failed.");
  }
}

export function initForms() {
  elBtn  = document.getElementById("ps-create-btn");
  elMenu = document.getElementById("ps-create-menu");
  if (!elBtn || !elMenu) return;

  elBtn.addEventListener("click", (ev) => {
    ev.stopPropagation();
    _toggleMenu();
  });

  // Close on outside click + ESC.
  document.addEventListener("click", (ev) => {
    if (elMenu.hidden) return;
    if (elBtn.contains(ev.target) || elMenu.contains(ev.target)) return;
    _closeMenu();
  });
  document.addEventListener("keydown", (ev) => {
    if (!elMenu.hidden && ev.key === "Escape") {
      _closeMenu();
      try { elBtn.focus(); } catch (_) {}
    }
  });

  // Item dispatch.  Close the menu BEFORE awaiting the dialog so the
  // user's focus moves cleanly from menu to dialog.
  elMenu.querySelectorAll(".ps-create-menu-item").forEach((item) => {
    item.addEventListener("click", async () => {
      const action = item.dataset.action;
      _closeMenu();
      if (action === "new-project") await _doNewProject();
      else if (action === "new-folder") await _doNewFolder();
      else if (action === "upload")     await _doUpload();
    });
  });

  // Re-evaluate enablement on selection change (the user navigates
  // away from root, items become enabled; navigate back, they re-
  // disable).  Also refresh once the projects root resolves so the
  // initial paint shows the right state.
  projects.onChange(_updateItemEnablement);
  _updateItemEnablement();
}
