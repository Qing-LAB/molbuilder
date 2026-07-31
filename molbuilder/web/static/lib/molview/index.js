/* MolView — the single ES-module entry, and the whole of what is importable.
 *
 * Contract: docs/web/molview.md § 4, § 9.1.
 * Owns:     the two names a consumer may write — `mount` and `formula`.
 * Called by: a page, once:
 *
 *     import { mount, formula } from "/static/lib/molview/index.js";
 *     const viewer = await mount(hostEl, workspace, { owner, mode });
 *
 * NEVER (§ 4, § 9.1):
 *   - export anything else. "Every other file in the module is internal — a
 *     consumer that imports any of them directly has broken the module, not
 *     found a shortcut."
 *   - name the sealed layer. § 15: no consumer names its file and neither does
 *     the document.
 *   - write or read `window.molbuilder`, in either direction. Nothing MolView
 *     needs comes from a global; the workspace arrives as an argument to mount.
 *
 * `formula` is here beside `mount` because a caller can want it with no viewer
 * at all — a list of elements is enough (§ 4).
 */
"use strict";

export { mount } from "./mount.js";
export { formula } from "./_atom.js";
