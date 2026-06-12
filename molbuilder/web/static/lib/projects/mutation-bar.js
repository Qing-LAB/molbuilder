/* projects/mutation-bar.js -- header-bar wiring.
 *
 * Three SEPARATE buttons in the sidebar header (New project /
 * New folder / Upload).  No dropdown — each click opens its
 * modal dialog directly (from projects/dialogs.js).  Backend
 * calls go through projects.* (state.js), which dispatches to
 * projects/api.js + fires a directory refresh on success.
 *
 * Depth-aware enable/disable (driven by projects.onChange):
 *   * New project    -- always enabled
 *   * New folder     -- disabled at projects/ root (no useful parent)
 *   * Upload file    -- disabled at projects/ root
 *
 * Spec: docs/protocols/projects-sidebar.md § Mutation UX.
 *
 * 2026-06-12: renamed from ``forms.js`` after the v2 buttons-not-
 * inline-forms refactor (commit c929a28).  The historical name no
 * longer reflected the content; the file's only role now is the
 * mutation bar wiring.  Pre-1.0 repo policy [[no_backward_compat]]
 * — no module-name redirect kept around.
 */

import {
  chooseName, chooseUploadFile,
} from "./dialogs.js";
import {
  projects, getProjectsRoot, SS_DIR, atProjectsRoot,
} from "./state.js";
import { openDir } from "./list.js";

let elProjBtn, elFolderBtn, elUploadBtn;

function _updateButtonEnablement() {
  const dir = sessionStorage.getItem(SS_DIR) || getProjectsRoot() || "";
  const root = atProjectsRoot(dir);
  // "New project" always enabled (it lands at projects/ root).
  // "New folder" + "Upload" require a project context.
  if (elFolderBtn) {
    elFolderBtn.disabled = root;
    elFolderBtn.title = root
      ? "Pick a project folder in the sidebar first."
      : "Create a new folder inside the current directory";
  }
  if (elUploadBtn) {
    elUploadBtn.disabled = root;
    elUploadBtn.title = root
      ? "Pick a project folder in the sidebar first."
      : "Upload a file into the current directory";
  }
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
  elProjBtn   = document.getElementById("ps-create-project-btn");
  elFolderBtn = document.getElementById("ps-create-folder-btn");
  elUploadBtn = document.getElementById("ps-create-upload-btn");
  if (!elProjBtn && !elFolderBtn && !elUploadBtn) return;

  // Each button is a direct entry point — no dropdown, no extra
  // click between intent and dialog.
  if (elProjBtn) {
    elProjBtn.addEventListener("click", () => { _doNewProject(); });
  }
  if (elFolderBtn) {
    elFolderBtn.addEventListener("click", () => { _doNewFolder(); });
  }
  if (elUploadBtn) {
    elUploadBtn.addEventListener("click", () => { _doUpload(); });
  }

  // Re-evaluate enablement on selection change (the user navigates
  // away from root → folder/upload become enabled; navigate back →
  // they disable).  Also refresh once the projects root resolves so
  // the initial paint shows the right state.
  projects.onChange(_updateButtonEnablement);
  _updateButtonEnablement();
}
