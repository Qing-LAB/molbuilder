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
 */

import {
  chooseName, chooseUploadFile,
} from "./dialogs.js";
import {
  projects, getProjectsRoot, SS_DIR, atProjectsRoot, readSelectionSlot, tell,
} from "./state.js";
import { openDir } from "./list.js";

let elProjBtn, elFolderBtn, elUploadBtn, elZipBtn;

/* THE DOWNLOAD BUTTON'S WHOLE STATE, in one place.
 *
 * `zipBusy` is what the download is doing; `zipReport` is what the last
 * one held.  Neither is written onto the element directly -- they
 * are INPUTS to `_updateButtonEnablement`, which is the only function
 * that touches the button's label, tooltip or disabled flag.
 *
 * `projects.onChange` fires the enablement pass on every sidebar
 * navigation, so a second writer would hand the button back mid-build
 * or light it at the projects root.  With one writer the state cannot
 * contradict itself. */
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
         + "stands now: no checkpoint history (.git, .binsnapshots), no "
         + "workspace store and no engine scratch.  Its size is shown "
         + "first."
         + (zipReport ? "  Last archive: " + zipReport : ""));
}

/* THE DOWNLOAD ASKS FIRST, WITH ITS SIZE (`projects.md` § 4; user,
 * 2026-10-20: "is there a way to add a UI warning for this operation").
 * /api/files/zip_prepare reads names and sizes only, so the button says
 * "Reading sizes…" for seconds, not minutes; the person then sees what the
 * archive holds, what it leaves out and the line that copies the folder
 * instead, and only "Download" sends a byte -- /api/files/download_zip
 * streams the archive as it reads the files, its length exact, so the
 * browser shows progress and a cut download reads as failed. */
function _setZipBusy(busy, label) {
  zipBusy = !!busy;
  zipBusyLabel = label || "";
  _updateButtonEnablement();
}

/* WHAT THE ARCHIVE WOULD BE: its files and exact size, in the server's
 * one spelling of a size (`units.size_words`); `skipped` -- links
 * pointing out of the projects tree, left behind on purpose, never in
 * silence; and what it leaves out, each with how much. */
function _describeArchive(out) {
  let text = (Number(out.files) || 0) + " files, " + out.size_words;
  if (Number(out.skipped) > 0) {
    text += " — " + out.skipped + " link(s) pointing outside the "
      + "projects tree left out";
  }
  return text;
}

function _leftOutWords(out) {
  const parts = (out.left_out || []).map((l) =>
    l.what + " (" + l.files + " files, " + l.size_words + ")");
  return parts.length ? "Left out: " + parts.join("; ") + "." : "";
}

/* The last path segment -- what the button is about to download.  Naming
 * the target keeps a person from downloading a whole project by accident
 * (user, 2026-08-29). */
function _dirName(dir) {
  const parts = String(dir || "").split("/").filter(Boolean);
  return parts.length ? parts[parts.length - 1] : "this folder";
}

async function _doDownloadZip() {
  const dir = readSelectionSlot(SS_DIR) || "";
  if (!dir || atProjectsRoot(dir)) return;
  _setZipBusy(true, "Reading sizes…");
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
  _setZipBusy(false, "");
  if (!out || !out.ok) {
    tell("error", (out && out.error) || "Could not read the folder.");
    return;
  }
  const report = out.name + " — " + _describeArchive(out);
  const body = [report];
  const left = _leftOutWords(out);
  if (left) body.push(left);
  body.push("To copy it without a zip, from the machine it goes to: "
            + out.rsync);
  const go = await window.molbuilder.warningModal.confirm({
    title:        "Download " + out.name + "?",
    body:         body,
    confirmLabel: "Download",
  });
  if (!go) return;
  zipReport = report;
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
    tell("error", (r && r.error) || "Project creation failed.");
  }
}

async function _doNewFolder() {
  const currentDir = readSelectionSlot(SS_DIR) || getProjectsRoot();
  if (!currentDir || atProjectsRoot(currentDir)) {
    tell("warn", "Cannot create a folder at the projects root.  "
        + "Pick a project in the sidebar first.");
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
    tell("error", (r && r.error) || "Folder creation failed.");
  }
}

async function _doUpload() {
  const currentDir = readSelectionSlot(SS_DIR) || getProjectsRoot();
  if (!currentDir || atProjectsRoot(currentDir)) {
    tell("warn", "Cannot upload to the projects root.  Pick a project in "
        + "the sidebar first.");
    return;
  }
  const ctx = projects.relativeToProjects(currentDir) || "projects/";
  const file = await chooseUploadFile({ contextDir: ctx });
  if (!file) return;
  const r = await projects.upload(currentDir, file);
  if (r && r.aborted) return;
  if (!r || !r.ok) {
    tell("error", (r && r.error) || "Upload failed.");
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
