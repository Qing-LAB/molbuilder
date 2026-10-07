/* projects/parser.js -- the FORMAT-AWARE sub-namespace of the projects package.
 *
 * `projects.parser` owns the format-AWARE structure-file OPEN door --
 * `openMolecule` -- which knows the `.xyz` <-> `.molstruct.json` pairing.
 * FILE-ONLY (model/structure.md § 2.4): it goes through the SERVER, so
 * Python owns the file access, the pairing, AND the sidecar schema:
 *
 *   * openMolecule(path) -> model.installMolecule({path}) -> POST /api/build/load
 *     (the server reads the .xyz + paired .molstruct.json via StructureCodec.read).
 *
 * Saving a structure is `projects.molviewFiles.save("project", ...)`, which asks
 * WHERE and owns the overwrite flow (`tabs.md` § 6).
 *
 * The browser therefore moves NO structure bytes and NEVER authors the sidecar
 * schema.  This module orchestrates the model <-> server
 * round-trip; it interprets nothing.
 *
 * Contract + map: docs/model/structure.md.
 */

/* THE VIEWER IS PASSED IN, by whoever mounted it.
 *
 * These doors move a file between disk and a viewer, so they need one — and the
 * only way to a viewer is the handle `mount` gave you (molview.md § 5.6):
 * MolView publishes nothing.
 *
 * A page can have more than one viewer, so "the viewer" is not a question this
 * file can answer. The caller knows which one it means. */
function _modelOf(viewer) {
  return (viewer && viewer.ok && viewer.data) || null;
}


/**
 * openMolecule(viewer, path, { confirmDiscard? }) -- THE load door (contract §2).
 *
 * FILE-ONLY load (model/structure.md § 2.4): hand the PATH to the model install
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
  // FILE-ONLY load (model/structure.md § 2.4): hand the PATH to the model; the SERVER
  // reads the .xyz + paired .molstruct.json through StructureCodec.read -- Python owns
  // the file access AND the .xyz<->.molstruct pairing.  The browser does not read the
  // bytes or derive the sidecar path.  The cell is DEDUCED from the file data (the .xyz's
  // own lattice + the sidecar) and is never overridden at load time.
  // Contract (§2): openMolecule NEVER throws -- it returns {ok:false, error} on any
  // failure, like its guard cases above.  The file-only load surfaces a missing/
  // unreadable file OR a parse/sidecar error as a rejected installMolecule (the
  // server returns 404/400); catch it here so a caller doing
  // `if (!r.ok)` can't crash on a bad path.
  let payload;
  try {
    payload = await model.installMolecule({
      path:   path,
      /* THIS IS THE USER SAYING "LOAD THIS FILE", so it replaces whatever is
       * there.  Without it a read-only viewer that already holds a structure
       * answers null and does nothing.
       *
       * Enforcing is right here and nowhere else: swapping the structure
       * outright is not an EDIT of the one on screen, which is what read-only
       * exists to prevent, and this door is only ever reached by an explicit
       * gesture -- the Load button, or a double-click in the sidebar.  Work
       * that would be lost is guarded above, by the confirm. */
      enforce: true,
    });
  } catch (e) {
    return { ok: false, error: (e && e.message) || ("Could not load " + path) };
  }
  return { ok: true, payload: payload };
}

// The format-aware sub-namespace, mounted at window.molbuilder.projects.parser by the
// sidebar entry point (projects-sidebar.js).
export const parser = { openMolecule };
