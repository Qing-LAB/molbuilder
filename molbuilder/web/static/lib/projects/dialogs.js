/* projects/dialogs.js -- the projects sidebar's questions
 * (see web/projects.md § 4).
 *
 * Each export asks one question and resolves with the answer (or null on
 * cancel / Esc).  No HTTP here -- the caller dispatches to
 * projects.{mkdir, move, copy, rename, upload, createProject} after the
 * dialog resolves.
 *
 * Built on the ONE modal component (`lib/dialog.js`, `lib/dialog.css`):
 * one dialog at a time, its look and its keys said once there.  A yes/no
 * question is the one confirm, `warningModal.confirm`.
 */

import { askFields } from "../dialog.js";
import { getProjectsRoot } from "./state.js";

// ─── Filename / dirname validation ───────────────────────────────── //

/** The rename endpoint's rules -- a basename: no separators, not . / ..
 *  -- as `askFields`' refusal: ``null`` when it can stand. */
function _nameRefusal(raw) {
  const s = (raw || "").trim();
  if (!s) return "";
  if (s.indexOf("/") !== -1 || s.indexOf("\\") !== -1) {
    return "Name cannot contain '/' or '\\'.";
  }
  if (s === "." || s === "..") return "Reserved name.";
  return null;
}

// ─── chooseName ──────────────────────────────────────────────────── //

/**
 * A name for a new project, a new folder, a rename, a copy -- resolves to
 * the validated basename, or null.
 *
 * @param {object} opts
 * @param {string} opts.title    — the dialog's heading
 * @param {string} opts.label    — the field's label
 * @param {string} [opts.initial] — the value offered (selected)
 * @param {string} [opts.hint]   — why, in smaller prose
 * @param {string} [opts.confirmLabel] — the answer's button (default "OK")
 * @param {string} [opts.placeholder]
 */
export async function chooseName(opts) {
  opts = opts || {};
  const got = await askFields({
    title: opts.title || "Name",
    hint: opts.hint,
    fields: [{ name: "name", label: opts.label || "Name",
               value: opts.initial || "", placeholder: opts.placeholder }],
    validate: (v) => _nameRefusal(v.name),
    confirmLabel: opts.confirmLabel || "OK",
  });
  return got ? got.name.trim() : null;
}

// ─── chooseFrame ─────────────────────────────────────────────────── //

/**
 * Which frame of a file holding several (`web/tabs.md` § 2): counted from
 * 1, as the frame bar counts (`model/structure.md` § 2.2e).  Resolves to the
 * frame NUMBER (1 … count), or null.
 *
 * @param {object} opts
 * @param {string} opts.title    — the dialog's heading (the file's name)
 * @param {number} opts.count    — how many frames the file holds
 * @param {number} [opts.initial] — the number offered (default 1)
 */
export async function chooseFrame(opts) {
  opts = opts || {};
  const count = Number(opts.count) || 0;
  const got = await askFields({
    title: opts.title || "Which frame?",
    hint: "This file holds " + count
      + " frames -- which one do you want to load?",
    fields: [{ name: "frame", label: "Frame (1 … " + count + ")",
               type: "number", value: String(opts.initial || 1),
               attrs: { min: 1, max: count, step: 1 } }],
    validate: (v) => {
      const n = Number(String(v.frame).trim());
      return Number.isInteger(n) && n >= 1 && n <= count ? null
        : "A frame is a whole number from 1 to " + count + ".";
    },
    confirmLabel: "Load",
  });
  return got ? Number(String(got.frame).trim()) : null;
}

// ─── chooseUploadFile ────────────────────────────────────────────── //

/**
 * A file from disk to upload into the current directory -- resolves to the
 * File, or null.
 *
 * @param {object} opts
 * @param {string} opts.contextDir — the destination's name, for the hint
 */
export async function chooseUploadFile(opts) {
  opts = opts || {};
  const got = await askFields({
    title: "Upload file",
    hint: opts.contextDir ? "Uploaded into " + opts.contextDir + "." : null,
    fields: [{ name: "file", label: "Pick a file", type: "file" }],
    validate: (v) => (v.file ? null : ""),
    confirmLabel: "Upload",
  });
  return got ? got.file : null;
}

// ─── confirmByTyping ─────────────────────────────────────────────── //

/**
 * The costliest questions -- deleting a project, a directory with
 * everything in it -- are answered by TYPING the thing's name, so a stray
 * Enter cannot answer them.  Resolves true when the name was typed and
 * confirmed, false otherwise.
 *
 * @param {object} opts
 * @param {string} opts.title
 * @param {string} opts.body          — what will be lost
 * @param {string} opts.phrase        — the name to type
 * @param {string} [opts.confirmLabel] — default "Delete"
 */
export async function confirmByTyping(opts) {
  opts = opts || {};
  const phrase = String(opts.phrase || "");
  const got = await askFields({
    title: opts.title || "Confirm",
    hint: opts.body,
    fields: [{ name: "typed", label: "Type \u201c" + phrase
                                   + "\u201d to confirm" }],
    validate: (v) => (v.typed.trim() === phrase ? null : ""),
    confirmLabel: opts.confirmLabel || "Delete",
    danger: true,
  });
  return got !== null;
}

// ─── Destination-dir picker (tree) ───────────────────────────────── //

/**
 * A folder of the projects tree, the destination of a move or a copy --
 * resolves to its absolute path, or null.
 *
 * @param {object} opts
 * @param {string} opts.title   — the dialog's heading (e.g. "Move to…")
 */
export async function chooseDestinationDir(opts) {
  // Delegates to the ONE pop-out picker (lib/tree-picker.js) --
  // this wrapper keeps the sidebar's question named for what it asks.
  opts = opts || {};
  const { pickPath } = await import("../tree-picker.js");
  return pickPath({
    title: opts.title || "Choose destination",
    hint: "Click a folder to select.  Double-click to open it.",
    mode: "dir",
    confirmLabel: opts.confirmLabel || "Choose",
  });
}

// ─── confirmDestructive ──────────────────────────────────────────── //

/**
 * A destructive yes/no -- an overwrite, a delete.  The ONE confirm
 * (`warningModal.confirm`): its costly choice marked `.danger`, Cancel first
 * and focused.  Resolves true on confirm, false otherwise.
 */
export function confirmDestructive(opts) {
  opts = opts || {};
  return window.molbuilder.warningModal.confirm({
    title: opts.title || "Confirm",
    body: opts.body || "",
    confirmLabel: opts.confirmLabel || "Proceed",
  });
}

/**
 * The one "save it WHERE, as WHAT" question — the move-to dialog composed
 * with the name dialog, from the project root, on any tab
 * (docs/web/projects.md § 5).
 *
 * A UNIFIED surface on purpose (user, 2026-08-19): every future flow that
 * deposits or consolidates artifacts under the project root — the MolView
 * exports today, a transport calculation assembling several results
 * tomorrow — asks this one question through this one door, so "where may
 * files go" never grows a second implementation.
 *
 * THE EXTENSION IS ADDED, never asked (`opts.ext`, e.g. ".xyz"): the name
 * is asked without it, the dialog says so, and it is added once -- a name
 * typed with it, in any case, is taken as typed: `x.xyz` is `x.xyz`, never
 * `x.xyz.xyz` (projects.md § 5).
 *
 * @param opts {title?, nameTitle?, initial?, hint?, ext?}
 * @returns "<dir>/<name>" -- `ext` added where given -- or null on cancel.
 * @throws when the project root is not available on this page yet — said,
 *         never a silent nothing.
 */
export async function chooseSavePath(opts) {
  opts = opts || {};
  if (!getProjectsRoot()) {
    throw new Error("the project tree is not available on this page yet");
  }
  const dir = await chooseDestinationDir({
    title: opts.title || "Save where?",
  });
  if (!dir) return null;
  const ext = opts.ext || "";
  const name = await chooseName({
    title: opts.nameTitle || "Save as",
    label: "Name",
    initial: opts.initial || "",
    hint: [opts.hint || "", ext ? "The extension is added for you." : ""]
      .filter(Boolean).join(" "),
  });
  if (!name) return null;
  const typed = name.toLowerCase().endsWith(ext.toLowerCase())
    ? name : name + ext;
  return String(dir).replace(/\/+$/, "") + "/" + typed;
}
