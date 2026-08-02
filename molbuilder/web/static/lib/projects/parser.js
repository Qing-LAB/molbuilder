/* projects/parser.js -- the FORMAT-AWARE sub-namespace of the projects package.
 *
 * `projects.parser` owns the format-AWARE structure-file doors -- `openMolecule` /
 * `saveMolecule` -- that know the `.xyz` <-> `.molstruct.json` pairing.  FILE-ONLY
 * (structure-authority.md § 3.3): BOTH doors go through the SERVER so Python owns the
 * file access, the pairing, AND the sidecar schema:
 *
 *   * openMolecule(path) -> model.installMolecule({path}) -> POST /api/build/load
 *     (the server reads the .xyz + paired .molstruct.json via StructureCodec.read), and
 *   * saveMolecule(path) -> POST /api/structure/save with the model's structure
 *     structure (the server writes the pair via
 *     StructureCodec.write, stamping schema_version + a real structure_hash).
 *
 * The browser therefore moves NO structure bytes and NEVER authors the sidecar schema
 * (a browser-written sidecar had no schema_version, so the file-only load door rejected
 * it -- a save->reload breaker).  This module orchestrates the model <-> server
 * round-trip; it interprets nothing.
 *
 * Contract + map: docs/model/structure.md.
 */

/* THE VIEWER IS PASSED IN, by whoever mounted it.
 *
 * These doors move a file between disk and a viewer, so they need one — and the
 * only way to a viewer is the handle `mount` gave you (molview.md § 5.6). This
 * used to look one up by name on `window`, which stopped working the day MolView
 * was rebuilt to publish nothing, and every load on every tab failed with
 * "molview.data.installMolecule unavailable".
 *
 * A page can have more than one viewer, so "the viewer" was never a question this
 * file could answer anyway. The caller knows which one it means. */
function _modelOf(viewer) {
  return (viewer && viewer.ok && viewer.data) || null;
}

// The save door refuses an existing target (no overwrite) with a 409 whose envelope is
// {ok:false, needsOverwrite:true, error:"file already exists: ..."}.  Detect it so the tab
// shows its overwrite dialog (the needsOverwrite contract, §2) instead of a raw error banner.
function _isExistsEnvelope(env) {
  if (!env) return false;
  if (env.needsOverwrite || env.exists) return true;
  const e = String(env.error || "").toLowerCase();
  return e.indexOf("exist") !== -1 || e.indexOf("overwrite") !== -1;
}

/**
 * openMolecule(viewer, path, { confirmDiscard? }) -- THE load door (contract §2).
 *
 * FILE-ONLY load (structure-authority.md § 3.3): hand the PATH to the model install
 * primitive (`molview.data.installMolecule({path})`), which POSTs it to `/api/build/load`
 * where the SERVER reads the `.xyz` AND its paired `.molstruct.json` via
 * `StructureCodec.read` and installs the whole model in ONE write.  The browser reads no
 * bytes and derives no sidecar path -- Python owns the file access + the pairing.  A
 * missing sidecar is NOT an error (a plain geometry loads label-less); format (.xyz vs
 * .pdb) is dispatched server-side from the extension.  `confirmDiscard` is the injected
 * dirty-canvas gate (UI policy; this layer stays DOM-free).  Returns `{ok:true, payload}`
 * | `{ok:false, cancelled|error}`.
 */
export async function openMolecule(viewer, path, opts) {
  opts = opts || {};
  if (typeof path !== "string" || !path) {
    return { ok: false, error: "projects.parser.openMolecule(viewer, path): non-empty string required" };
  }
  const model = _modelOf(viewer);
  if (!model) {
    return { ok: false, error: "projects.parser.openMolecule: no viewer was given to load into" };
  }
  // Dirty-canvas gate (injected UI policy): a fresh open discards unsaved edits.
  // `uncommitted` is the viewer's own answer to "is there work here that is not
  // on the sequence yet" (molview.md § 11.2) -- a value it holds, not a question.
  if (typeof opts.confirmDiscard === "function" && model.uncommitted) {
    const proceed = await opts.confirmDiscard();
    if (!proceed) return { ok: false, cancelled: true };
  }
  // FILE-ONLY load (structure-authority.md): hand the PATH to the model; the SERVER
  // reads the .xyz + paired .molstruct.json through StructureCodec.read -- Python owns
  // the file access AND the .xyz<->.molstruct pairing.  The browser no longer reads the
  // bytes or derives the sidecar path (that was the raw-text/second-file-stack seam the
  // consolidation exists to abolish).  The cell is DEDUCED from the file data (the .xyz's
  // own lattice + the sidecar) and is never overridden at load time.
  // Contract (§2): openMolecule NEVER throws -- it returns {ok:false, error} on any
  // failure, like its guard cases above.  The file-only load surfaces a missing/
  // unreadable file OR a parse/sidecar error as a rejected installMolecule (the
  // server returns 404/400 -> _loadText throws); catch it here so a caller doing
  // `if (!r.ok)` (transport/core, inspectors/structure, selection-bootstrap) can't
  // crash on a bad path.  (Before the file-only move the browser's readFile caught
  // the missing-file case; that guard now lives here.)
  let payload;
  try {
    payload = await model.installMolecule({
      path:   path,
      source: { kind: "file", file: path, generator_input: null },
    });
  } catch (e) {
    return { ok: false, error: (e && e.message) || ("Could not load " + path) };
  }
  return { ok: true, payload: payload };
}

/**
 * saveMolecule(viewer, path, { overwrite?, range? }) -- THE save door (contract §2).
 *
 * FILE-ONLY save (structure-authority.md § 3.3): read the SETTLED model
 * (`molview.data.exportFile` -> the structure) and hand it to the SERVER
 * (`POST /api/structure/save`), which reconstructs the Structure and writes the
 * `.xyz` + paired `.molstruct.json` via `StructureCodec.write` -- Python owns the pairing,
 * the write order/atomicity, AND the sidecar schema (schema_version + a real hash).  The
 * browser no longer writes the sidecar itself (a browser-authored sidecar had no
 * schema_version, so the file-only load door rejected the pair on the next open).  Then
 * On an "exists" envelope
 * (no overwrite) -> `{needsOverwrite:true}` so the tab confirms + retries with
 * `{overwrite:true}` (the dialog is UI policy).  Returns `{ok:true, path}` |
 * `{ok:false, needsOverwrite|error}`.  Never throws.
 */
export async function saveMolecule(viewer, path, opts) {
  opts = opts || {};
  if (typeof path !== "string" || !path) {
    return { ok: false, error: "projects.parser.saveMolecule(viewer, path): non-empty string required" };
  }
  const model = _modelOf(viewer);
  if (!model) {
    return { ok: false, error: "projects.parser.saveMolecule: no viewer was given to save from" };
  }
  /* THE RANGE GOES STRAIGHT THROUGH (molview.md § 11.3). `saveMolecule` does
   * not decide which frames leave -- the caller does, the model resolves it
   * against what exists, and this hands the answer on. Omitted, the model falls
   * back to the displayed frame, which is what this door has always saved.
   *
   * That is the whole benefit of the range living on `exportFile`: saving a
   * trajectory into the project and downloading one are the SAME read, so the
   * two destinations cannot come to disagree about what "the trajectory" was. */
  const file = model.exportFile(opts.range);  // {name, structure, frames?}; null = empty OR desync
  if (!file) {
    // Nothing loaded reads as nothing, which is a different answer from a
    // structure with no atoms (molview.md § 9.3).
    return { ok: false, error: (model.getStructure() === null)
      ? "save: there is nothing loaded to save"
      : "save: the structure is inconsistent (atom-count desync); reload before saving." };
  }
  // Hand the STRUCTURE to the server save door; the SERVER writes the pair through
  // StructureCodec.write -- the one paired-file writer, which owns the pairing rule,
  // the write order and the schema.  The browser assembles no bytes at all (the blob
  // it used to send was the last place it did).  Its "exists" gate (409 ->
  // needsOverwrite) drives the tab's overwrite dialog.
  let env;
  try {
    const resp = await window.fetch("/api/structure/save", {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      // `frames` rides BESIDE the envelope, present only for a range -- so a
      // one-frame save is byte-for-byte the request it always was.
      body:    JSON.stringify(Object.assign(
                 { path: path, structure: file.structure,
                   overwrite: !!opts.overwrite },
                 file.frames ? { frames: file.frames } : {})),
    });
    env = await resp.json().catch(function () {
      return { ok: false, error: "save: malformed server response" };
    });
  } catch (err) {
    return { ok: false, error: "Save failed: " + (err && err.message ? err.message : String(err)) };
  }
  if (env && env.ok === false) {
    if (_isExistsEnvelope(env)) return { ok: false, needsOverwrite: true };
    return { ok: false, error: (env.error || "Save failed.") };
  }
  /* NOTHING IS MARKED ON THE VIEWER. Where a structure was saved TO is a fact
   * about a file operation the caller performed, so the caller keeps it
   * (molview.md § 6.7), and the unsaved-work badge is raised and cleared inside
   * the viewer's own gate rather than set from out here (§ 11.2). */
  return { ok: true, path: path };
}

// The format-aware sub-namespace, mounted at window.molbuilder.projects.parser by the
// sidebar entry point (projects-sidebar.js).
export const parser = { openMolecule, saveMolecule };
