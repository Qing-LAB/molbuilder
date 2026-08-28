/* projects/mutation-bar.js -- header-bar wiring.
 *
 * Four SEPARATE buttons in the sidebar header (New project /
 * New folder / Upload / Download).  No dropdown — each click acts
 * directly (a modal dialog from projects/dialogs.js, or for
 * Download a plain navigation the server answers as an
 * attachment).  Backend calls go through projects.* (state.js),
 * which dispatches to projects/api.js + fires a directory refresh
 * on success.
 *
 * Depth-aware enable/disable (driven by projects.onChange):
 *   * New project    -- always enabled
 *   * New folder     -- disabled at projects/ root (no useful parent)
 *   * Upload file    -- disabled at projects/ root
 *   * Download .zip  -- disabled at projects/ root (zipping the whole
 *                       tree is never the intent; the endpoint refuses
 *                       it too)
 *
 * Spec: docs/web/projects.md § Mutation UX.
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
  projects, getProjectsRoot, SS_DIR, atProjectsRoot, readSelectionSlot,
} from "./state.js";
import { openDir } from "./list.js";

let elProjBtn, elFolderBtn, elUploadBtn, elZipBtn;

function _updateButtonEnablement() {
  const dir = readSelectionSlot(SS_DIR) || getProjectsRoot() || "";
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
  if (elZipBtn) {
    elZipBtn.disabled = root;
    elZipBtn.title = root
      ? "Pick a folder in the sidebar first."
      : "Download the current directory as a .zip";
  }
}

function _doDownloadZip() {
  const dir = readSelectionSlot(SS_DIR) || "";
  if (!dir || atProjectsRoot(dir)) return;
  // A plain navigation: the server answers with
  // Content-Disposition: attachment, so the page stays put and the
  // browser saves "<folder>.zip".  The use this exists for (user,
  // 2026-08-28): carry a calculation folder to a cluster without
  // ssh -- download it here, drop it there through the cluster's
  // own web portal.
  window.location.assign(
    "/api/files/download_zip?path=" + encodeURIComponent(dir));
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
  const currentDir = readSelectionSlot(SS_DIR) || getProjectsRoot();
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
  const currentDir = readSelectionSlot(SS_DIR) || getProjectsRoot();
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
  elZipBtn    = document.getElementById("ps-download-zip-btn");
  if (!elProjBtn && !elFolderBtn && !elUploadBtn && !elZipBtn) return;

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
  if (elZipBtn) {
    elZipBtn.addEventListener("click", () => { _doDownloadZip(); });
  }

  // Re-evaluate enablement on selection change (the user navigates
  // away from root → folder/upload become enabled; navigate back →
  // they disable).  Also refresh once the projects root resolves so
  // the initial paint shows the right state.
  projects.onChange(_updateButtonEnablement);
  _updateButtonEnablement();
}
