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

/* THE DOWNLOAD BUTTON'S WHOLE STATE, in one place.
 *
 * `zipBusy` is what the build is doing; `zipReport` is what the last
 * one produced.  Neither is written onto the element directly -- they
 * are INPUTS to `_updateButtonEnablement`, which is the only function
 * that touches the button's label, tooltip or disabled flag.
 *
 * Two bugs came from having a second writer.  `projects.onChange`
 * fires the enablement pass on every sidebar navigation, so browsing
 * during a build handed the button back and a second click started a
 * second archive; and the busy reset set `disabled = false`
 * unconditionally, so a person who had navigated to the projects root
 * meanwhile was left with a lit button that does nothing.  With one
 * writer the state cannot contradict itself. */
let zipBusy = false;
let zipBusyLabel = "";
let zipReport = "";

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
  if (!elZipBtn) return;
  const label = elZipBtn.querySelector(".ps-create-action-label");
  if (label) label.textContent = zipBusy ? zipBusyLabel : "Download";
  elZipBtn.classList.toggle("is-busy", zipBusy);
  // Busy wins over context: mid-build the button is unclickable
  // wherever you have browsed to.  Idle, it is the context that
  // decides -- so a build that finishes while you sit at the projects
  // root leaves it correctly disabled, not lit and inert.
  elZipBtn.disabled = zipBusy || root;
  elZipBtn.title = zipBusy
    ? zipBusyLabel + " — one archive at a time."
    : (root
       ? "Pick a folder in the sidebar first."
       : "Download " + _dirName(dir) + " as a .zip — the folder as it "
         + "stands now: no checkpoint history (.git, .binsnapshots) and "
         + "no workspace store."
         + (zipReport ? "  Last archive: " + zipReport : ""));
}

/* Compressing a results directory takes minutes, and the button is
 * the only place a person is looking.  So it SAYS what is happening
 * and refuses a second click until the browser's save has started
 * (user, 2026-08-29): "Zipping…" while the server builds, then the
 * navigation, then back to "Download".
 *
 * The archive is built by /api/files/zip_prepare and fetched by
 * token, which is what makes the two states knowable -- a plain
 * navigation reports neither its start nor its end. */
function _setZipBusy(busy, label) {
  zipBusy = !!busy;
  zipBusyLabel = label || "";
  _updateButtonEnablement();
}

/* WHAT THE ARCHIVE TURNED OUT TO BE.  The server counts the files, the
 * bytes and anything it had to leave behind, and answers all three --
 * saying none of it was the original complaint ("nothing happens until
 * several minutes later") only half addressed.  `skipped` matters
 * most: those are symlinks pointing out of the projects tree, dropped
 * on purpose, and silently dropping a file is not an option. */
function _describeArchive(out) {
  const mb = (Number(out.bytes) || 0) / (1024 * 1024);
  const size = mb >= 1 ? mb.toFixed(1) + " MB"
    : Math.max(1, Math.round((Number(out.bytes) || 0) / 1024)) + " KB";
  let text = (Number(out.files) || 0) + " files, " + size;
  if (Number(out.skipped) > 0) {
    text += " — " + out.skipped + " link(s) pointing outside the "
      + "projects tree were left out";
  }
  return text;
}

/* The last path segment -- what the button is about to zip.  The
 * control used to name no target at all, which is exactly how a person
 * ends up downloading a whole project by accident (user, 2026-08-29). */
function _dirName(dir) {
  const parts = String(dir || "").split("/").filter(Boolean);
  return parts.length ? parts[parts.length - 1] : "this folder";
}

async function _doDownloadZip() {
  const dir = readSelectionSlot(SS_DIR) || "";
  if (!dir || atProjectsRoot(dir)) return;
  _setZipBusy(true, "Zipping…");
  let out = null;
  try {
    const r = await fetch("/api/files/zip_prepare", {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify({ path: dir }),
    });
    out = await r.json();
  } catch (e) {
    out = { ok: false, error: String(e) };
  }
  if (!out || !out.ok) {
    _setZipBusy(false, "");
    window.alert((out && out.error) || "Could not build the archive.");
    return;
  }
  zipReport = _dirName(dir) + ".zip — " + _describeArchive(out);
  if (Number(out.skipped) > 0) {
    // Not a tooltip matter: files the person expected to be in the
    // archive are not in it, and they have to know before they carry
    // it to another machine.
    window.alert(
      "Archive ready: " + zipReport + ".\n\nThose links point outside "
      + "the projects tree, so their targets could not be included.");
  }
  // The bytes exist: this navigation starts the save immediately.
  _setZipBusy(true, "Saving…");
  window.location.assign(
    "/api/files/download_zip?token=" + encodeURIComponent(out.token));
  window.setTimeout(() => { _setZipBusy(false, ""); }, 1500);
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
