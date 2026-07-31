/* MolView — the model's helpers: load a structure in, write it out, and the
 * geometry edits.
 *
 * Contract: docs/web/molview.md § 7.3 (one central file, and helpers it hands
 *           work to) and § 11.1 (geometry edits go to the server).
 * Owns:     three jobs, each doing only its own:
 *              load     — put a loaded structure into the model
 *              write out — turn the structure into text, for export and for
 *                          saved states
 *              edits    — the geometry operations
 * Called by: model.js, and nothing else. Not reachable from outside the model.
 *
 * THE RULE THAT MAKES THESE HELPERS (§ 7.3): "When the central file builds a
 * helper, it hands over exactly the functions that helper is allowed to call."
 * What each is handed:
 *   load     — where to put it; how to announce a change; a way to record the
 *              first state
 *   write out — read-only access to the atoms, cell, selection and history
 *              position
 *   edits    — read the atoms; apply the structure the server sends back
 *
 * NEVER (§ 7.3):
 *   - reach out on its own. A helper calls what it was handed and nothing else.
 *     That is what keeps each one small, testable with stand-in functions, and
 *     replaceable without disturbing anything else.
 *   - hold the cell. It is a field of the structure (§ 6.2), so it lives with
 *     the structure and is edited through the one cell door (§ 9.3). "A helper
 *     holding 'the cell' beside the structure that already has one would be a
 *     second home for it, which is the thing § 5.2 exists to prevent."
 *
 * EMPTY BY DESIGN — plan step A. Step D writes the body.
 */
"use strict";
