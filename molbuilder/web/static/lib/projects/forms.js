/* projects/forms.js -- + New project / + New subdir / + Upload file forms.
 *
 * Three foldable <details> sections at the top of the sidebar.  Each
 * is self-contained: HTML in _projects_sidebar.html, behaviour here.
 * Backend calls go through projects/api.js; post-success navigation
 * uses list.openDir.
 *
 * Depth-aware visibility (driven by selection changes via
 * projects.onChange):
 *   * + New project    -- always visible
 *   * + New subdir     -- hidden at projects/ root
 *   * + Upload file    -- hidden at projects/ root (stub backend)
 *
 * Spec: docs/protocols/selection.md § Sidebar interaction rules.
 */

import {
  apiMkdir, apiCreateProject, apiUpload,
} from "./api.js";
import {
  projects, getProjectsRoot, SS_DIR, atProjectsRoot,
} from "./state.js";
import { openDir } from "./list.js";

// Mkdir form (single-dir creation at current location).
let elMkdirForm, elMkdirInput, elMkdirError, elMkdirContext;
// Upload form (stub backend; UX wired).
let elUploadForm, elUploadInput, elUploadError, elUploadContext;
// New-project form (always visible; bootstraps full skeleton).
let elNewProjForm, elNewProjInput, elNewProjError, elNewProjSubdirs;

// ----- "(in <current-dir>)" hints + depth-aware visibility ------- //
//
// Two forms (+ New subdir, + Upload file) share the depth-0 hide
// rule -- they're meaningless at the projects/ root because no
// useful filename / dirname can land there.  The atProjectsRoot
// predicate is centralised in state.js so every caller agrees on
// what "at the root" means (handles trailing-slash edge cases).

function _setSectionHiddenIfRoot(formEl, hidden) {
  const section = formEl ? formEl.closest("details") : null;
  if (!section) return;
  section.hidden = hidden;
  if (hidden) section.open = false;
}

function _updateMkdirContext() {
  if (!elMkdirContext) return;
  const dir = sessionStorage.getItem(SS_DIR) || getProjectsRoot() || "";
  elMkdirContext.textContent = dir
    ? (projects.relativeToProjects(dir) || "projects/")
    : "current directory";
  elMkdirContext.title = dir || "";
  _setSectionHiddenIfRoot(elMkdirForm, atProjectsRoot(dir));
}

function _updateUploadContext() {
  if (!elUploadContext) return;
  const dir = sessionStorage.getItem(SS_DIR) || getProjectsRoot() || "";
  elUploadContext.textContent = dir
    ? (projects.relativeToProjects(dir) || "projects/")
    : "current directory";
  elUploadContext.title = dir || "";
  _setSectionHiddenIfRoot(elUploadForm, atProjectsRoot(dir));
}

// ----- Mkdir form ------------------------------------------------ //

function _resetMkdirForm() {
  elMkdirInput.value = "";
  elMkdirError.textContent = "";
}

async function _submitMkdir(ev) {
  ev.preventDefault();
  elMkdirError.textContent = "";
  const name = elMkdirInput.value.trim();
  if (!name) {
    elMkdirError.textContent = "Name cannot be empty.";
    return;
  }
  const parent = sessionStorage.getItem(SS_DIR) || getProjectsRoot();
  const j = await apiMkdir(parent, name);
  if (!j.ok) {
    // Backend's 409 / 400 / 403 message verbatim -- already says
    // what to do (rename, pick a different parent, etc.).
    elMkdirError.textContent = j.error || "mkdir failed.";
    return;
  }
  _resetMkdirForm();
  await openDir(j.path);
}

// ----- New-project form ----------------------------------------- //

function _resetNewProjectForm() {
  elNewProjInput.value = "";
  elNewProjError.textContent = "";
}

async function _submitNewProject(ev) {
  ev.preventDefault();
  elNewProjError.textContent = "";
  const name = elNewProjInput.value.trim();
  if (!name) {
    elNewProjError.textContent = "Name cannot be empty.";
    return;
  }
  const j = await apiCreateProject(name);
  if (!j.ok) {
    elNewProjError.textContent = j.error || "create failed.";
    return;
  }
  _resetNewProjectForm();
  await openDir(j.path);
}

// ----- Upload form (stub backend; UX wired) --------------------- //

function _resetUploadForm() {
  if (elUploadInput) elUploadInput.value = "";
  if (elUploadError) elUploadError.textContent = "";
}

async function _submitUpload(ev) {
  ev.preventDefault();
  elUploadError.textContent = "";
  if (!elUploadInput.files || elUploadInput.files.length === 0) {
    elUploadError.textContent = "Pick a file to upload first.";
    return;
  }
  const file = elUploadInput.files[0];
  const target = sessionStorage.getItem(SS_DIR) || getProjectsRoot();
  const j = await apiUpload(target, file);
  if (!j.ok) {
    // Today the 501 message lands here.  Tomorrow the real backend's
    // 409 / 400 / 403 land via the same path -- no caller changes.
    elUploadError.textContent = j.error || "upload failed.";
    return;
  }
  _resetUploadForm();
  await openDir(target);
}

// ----- Init ------------------------------------------------------ //

export function initForms() {
  // Mkdir.
  elMkdirForm    = document.getElementById("ps-mkdir-form");
  elMkdirInput   = document.getElementById("ps-mkdir-input");
  elMkdirError   = document.getElementById("ps-mkdir-error");
  elMkdirContext = document.querySelector(".ps-mkdir-context");
  if (elMkdirForm) elMkdirForm.addEventListener("submit", _submitMkdir);
  const mkdirCancel = document.getElementById("ps-mkdir-cancel");
  if (mkdirCancel) mkdirCancel.addEventListener("click", _resetMkdirForm);

  // New project.
  elNewProjForm    = document.getElementById("ps-newproject-form");
  elNewProjInput   = document.getElementById("ps-newproject-input");
  elNewProjError   = document.getElementById("ps-newproject-error");
  elNewProjSubdirs = document.getElementById("ps-newproject-subdirs");
  if (elNewProjForm) elNewProjForm.addEventListener("submit", _submitNewProject);
  const newProjCancel = document.getElementById("ps-newproject-cancel");
  if (newProjCancel) newProjCancel.addEventListener("click", _resetNewProjectForm);
  // Subdir list hint (hard-coded; matches CANONICAL_TOPICS).
  if (elNewProjSubdirs) {
    const CANONICAL_TOPICS_DISPLAY = [
      "structure", "pseudopotential",
      "optimization", "frequency", "spectrum",
      "transport", "single-point", "scan", "user",
    ];
    elNewProjSubdirs.innerHTML = CANONICAL_TOPICS_DISPLAY
      .map((t) => `<code>${t}/</code>`).join(", ");
  }

  // Upload.
  elUploadForm    = document.getElementById("ps-upload-form");
  elUploadInput   = document.getElementById("ps-upload-input");
  elUploadError   = document.getElementById("ps-upload-error");
  elUploadContext = document.querySelector(".ps-upload-context");
  if (elUploadForm) elUploadForm.addEventListener("submit", _submitUpload);
  const uploadCancel = document.getElementById("ps-upload-cancel");
  if (uploadCancel) uploadCancel.addEventListener("click", _resetUploadForm);

  // Update context labels + depth-aware visibility on every
  // selection change.
  projects.onChange(() => {
    _updateMkdirContext();
    _updateUploadContext();
  });
}
