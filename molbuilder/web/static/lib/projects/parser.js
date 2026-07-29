/* projects/parser.js -- the FORMAT-AWARE sub-namespace of the projects package.
 *
 * `projects.parser` owns the format-AWARE structure-file doors -- `openMolecule` /
 * `saveMolecule` -- that know the `.xyz` <-> `.molstruct.json` pairing.  FILE-ONLY
 * (structure-authority.md § 3.3): BOTH doors go through the SERVER so Python owns the
 * file access, the pairing, AND the sidecar schema:
 *
 *   * openMolecule(path) -> model.installMolecule({path}) -> POST /api/build/load
 *     (the server reads the .xyz + paired .molstruct.json via StructureCodec.read), and
 *   * saveMolecule(path) -> POST /api/structure/save with the model's {xyz, sidecar}
 *     blob (the server reconstructs the Structure + writes the pair via
 *     StructureCodec.write, stamping schema_version + a real structure_hash).
 *
 * The browser therefore moves NO structure bytes and NEVER authors the sidecar schema
 * (a browser-written sidecar had no schema_version, so the file-only load door rejected
 * it -- a save->reload breaker).  This module orchestrates the model <-> server
 * round-trip; it interprets nothing.
 *
 * Contract + map: docs/model/structure.md.
 */

// The molview MODEL primitives these doors call.  molview is a classic script mounted on
// window; this is an ES module -- look the model up at CALL time (never a static import,
// which would also invert the package dependency).
function _model() {
  return (typeof window !== "undefined"
    && window.molbuilder && window.molbuilder.molview
    && window.molbuilder.molview.data) || null;
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
 * openMolecule(path, { confirmDiscard? }) -- THE load door (contract §2).
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
export async function openMolecule(path, opts) {
  opts = opts || {};
  if (typeof path !== "string" || !path) {
    return { ok: false, error: "projects.parser.openMolecule(path): non-empty string required" };
  }
  const model = _model();
  if (!model || typeof model.installMolecule !== "function") {
    return { ok: false, error: "projects.parser.openMolecule: molview.data.installMolecule unavailable" };
  }
  // Dirty-canvas gate (injected UI policy): a fresh open discards unsaved edits.
  if (typeof opts.confirmDiscard === "function"
      && typeof model.isDirty === "function" && model.isDirty()) {
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
 * saveMolecule(path, { overwrite? }) -- THE save door (contract §2).
 *
 * FILE-ONLY save (structure-authority.md § 3.3): serialise the SETTLED model
 * (`molview.data.exportFile` -> `{xyz, sidecar}`) and hand that blob to the SERVER
 * (`POST /api/structure/save`), which reconstructs the Structure and writes the
 * `.xyz` + paired `.molstruct.json` via `StructureCodec.write` -- Python owns the pairing,
 * the write order/atomicity, AND the sidecar schema (schema_version + a real hash).  The
 * browser no longer writes the sidecar itself (a browser-authored sidecar had no
 * schema_version, so the file-only load door rejected the pair on the next open).  Then
 * `markSaved` (clears dirty + re-anchors the store sourceFile).  On an "exists" envelope
 * (no overwrite) -> `{needsOverwrite:true}` so the tab confirms + retries with
 * `{overwrite:true}` (the dialog is UI policy).  Returns `{ok:true, path}` |
 * `{ok:false, needsOverwrite|error}`.  Never throws.
 */
export async function saveMolecule(path, opts) {
  opts = opts || {};
  if (typeof path !== "string" || !path) {
    return { ok: false, error: "projects.parser.saveMolecule(path): non-empty string required" };
  }
  const model = _model();
  if (!model || typeof model.exportFile !== "function") {
    return { ok: false, error: "projects.parser.saveMolecule: molview.data.exportFile unavailable" };
  }
  const blob = model.exportFile();   // model -> {xyz, sidecar}; null = empty OR desync
  if (!blob) {
    return { ok: false, error: (model.isEmpty && model.isEmpty())
      ? "save: workspace has no data"
      : "save: workspace state is inconsistent (atom-count desync); reload before saving." };
  }
  // Hand the whole {xyz, sidecar} blob to the server save door; the SERVER writes the pair
  // through StructureCodec.write (schema owned server-side).  Its "exists" gate (409 ->
  // needsOverwrite) drives the tab's overwrite dialog.
  let env;
  try {
    const resp = await window.fetch("/api/structure/save", {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify({ path: path, blob: blob, overwrite: !!opts.overwrite }),
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
  if (typeof model.markSaved === "function") model.markSaved(path);
  return { ok: true, path: path };
}

// The format-aware sub-namespace, mounted at window.molbuilder.projects.parser by the
// sidebar entry point (projects-sidebar.js).
export const parser = { openMolecule, saveMolecule };
