/* projects/parser.js -- the FORMAT-AWARE sub-namespace of the projects package.
 *
 * `projects.readFile` / `projects.writeFile` move format-BLIND bytes (they serve every
 * tab, know nothing about contents).  `projects.parser` owns the format-AWARE
 * structure-file doors -- `openMolecule` / `saveMolecule` -- that know the
 * `.xyz` <-> `.molstruct.json` pairing and drive the parse seam.  They:
 *
 *   * move bytes through readFile / writeFile (THIS package -- format-blind), and
 *   * install into / serialise from the molview MODEL
 *     (window.molbuilder.molview.data -- installMolecule / exportFile / markSaved).
 *
 * The layering this keeps clean: the byte layer stays format-blind, the model owns no
 * file endpoint, and the sidecar SCHEMA lives ONLY server-side (sidecars/molstruct.py,
 * reached through /api/build/load).  This module orchestrates; it interprets nothing.
 *
 * Contract + map: docs/protocols/structure-load-save-contract.md.
 */
import { readFile, writeFile } from "./state.js";

// The molview MODEL primitives these doors call.  molview is a classic script mounted on
// window; this is an ES module -- look the model up at CALL time (never a static import,
// which would also invert the package dependency).
function _model() {
  return (typeof window !== "undefined"
    && window.molbuilder && window.molbuilder.molview
    && window.molbuilder.molview.data) || null;
}

// Canonical sidecar path -- MIRROR of sidecars/molstruct.py:sidecar_path_for: strip the
// LAST extension, append ".molstruct.json" ("relaxed.xyz" -> "relaxed.molstruct.json";
// "job.spectra.xyz" -> "job.spectra.molstruct.json"; "bridge" -> "bridge.molstruct.json").
function _sidecarPathFor(path) {
  return String(path).replace(/\.[^./]+$/, "") + ".molstruct.json";
}

// /api/files/write refuses an existing target (no overwrite) with a 409 whose envelope
// is {ok:false, error:"file already exists: ..."}.  Detect it so the tab shows its
// overwrite dialog (the needsOverwrite contract, §2) instead of a raw error banner.
function _isExistsEnvelope(env) {
  if (!env) return false;
  if (env.needsOverwrite || env.exists) return true;
  const e = String(env.error || "").toLowerCase();
  return e.indexOf("exist") !== -1 || e.indexOf("overwrite") !== -1;
}

/**
 * openMolecule(path, { confirmDiscard? }) -- THE load door (contract §2).
 *
 * Read the `.xyz` AND its `.molstruct.json` through the projects file package
 * (`readFile` -- bytes only, no parse), then hand their CONTENT to the model install
 * primitive (`molview.data.installMolecule`), which parses via `/api/build/load` (the
 * server applies the sidecar) and installs the whole model in ONE write.  A missing
 * sidecar is NOT an error -- a plain geometry loads label-less.  Format (.xyz vs .pdb)
 * is sniffed server-side from the filename; we hand RAW bytes, not a canonicalised copy.
 * `confirmDiscard` is the injected dirty-canvas gate (UI policy; this layer stays
 * DOM-free).  Returns `{ok:true, payload}` | `{ok:false, cancelled|error}`.
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
  // Geometry bytes (format-blind read).
  const xyzEnv = await readFile(path);
  if (!xyzEnv || xyzEnv.ok === false || typeof xyzEnv.text !== "string") {
    return { ok: false, error: (xyzEnv && xyzEnv.error) || ("Could not read " + path) };
  }
  // Paired .molstruct.json -- best-effort (a 404 / missing sidecar is fine, not an error).
  let sidecarText = null;
  try {
    const scEnv = await readFile(_sidecarPathFor(path));
    if (scEnv && scEnv.ok !== false && typeof scEnv.text === "string") {
      sidecarText = scEnv.text;
    }
  } catch (_) { /* no sidecar -> label-less load */ }
  // Install: the model parses the xyz + (server-applied) sidecar in ONE write.  The cell
  // is DEDUCED from the actual data -- the .xyz's own lattice (extended-xyz) and the
  // .molstruct.json sidecar -- and is NEVER overridden at load time.  A caller that needs
  // a different cell edits it explicitly on the Cell page after loading; the loader has
  // no cell parameter (hiding that logic in the load call was the wrong design).
  const payload = await model.installMolecule({
    text:     xyzEnv.text,
    filename: path,
    sidecar:  sidecarText,
    source:   { kind: "file", file: path, generator_input: null },
  });
  return { ok: true, payload: payload };
}

/**
 * saveMolecule(path, { overwrite? }) -- THE save door (contract §2).
 *
 * Serialise the SETTLED model (`molview.data.exportFile` -> `{xyz, sidecar}`) and write
 * BOTH files through the projects file package (`writeFile`), then `markSaved` (clears
 * dirty + re-anchors the store sourceFile).  Geometry is written FIRST (the primary
 * artifact); a mid-failure leaves the dirty bit set for retry.  On an "exists" envelope
 * (no overwrite) -> `{needsOverwrite:true}` so the tab confirms + retries with
 * `{overwrite:true}` (the dialog is UI policy).  Returns `{ok:true, path}` |
 * `{ok:false, needsOverwrite|error}`.
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
  const xyzText     = (typeof blob.xyz === "string") ? blob.xyz : "";
  const sidecarJson = JSON.stringify(blob.sidecar || {}, null, 2);
  const sidecarPath = _sidecarPathFor(path);
  const writeOpts   = opts.overwrite ? { overwrite: true } : undefined;
  // Geometry first (the primary artifact); its "exists" gate drives the dialog.
  let env;
  try {
    env = await writeFile(path, xyzText, writeOpts);
  } catch (err) {
    return { ok: false, error: "Save failed: " + (err && err.message ? err.message : String(err)) };
  }
  if (env && env.ok === false) {
    if (_isExistsEnvelope(env)) return { ok: false, needsOverwrite: true };
    return { ok: false, error: (env.error || "Save failed.") };
  }
  // Sidecar second: labels/regions/frozen + full periodicity.  Always overwrite -- a
  // dependent member of the pair we just (re)wrote the geometry for; a stale one must
  // never survive.
  let scEnv;
  try {
    scEnv = await writeFile(sidecarPath, sidecarJson, { overwrite: true });
  } catch (err) {
    return { ok: false, error: "saved the geometry but the sidecar write threw: "
      + (err && err.message ? err.message : String(err)) };
  }
  if (scEnv && scEnv.ok === false) {
    return { ok: false, error: "saved the geometry but the sidecar write failed: "
      + (scEnv.error || "unknown") + " -- labels/cell may be stale on disk." };
  }
  if (typeof model.markSaved === "function") model.markSaved(path);
  return { ok: true, path: path };
}

// The format-aware sub-namespace, mounted at window.molbuilder.projects.parser by the
// sidebar entry point (projects-sidebar.js).
export const parser = { openMolecule, saveMolecule };
