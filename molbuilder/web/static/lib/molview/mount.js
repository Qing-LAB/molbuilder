/* MolView — assembling a viewer, and the handle that comes back. § 7 level 2.
 *
 * Contract: docs/web/molview.md § 8 (making and tearing down a viewer) and
 *           § 9.2 (the handle — for a tab that wants a viewer).
 * Owns:     assembling the card, the panel, the controls and MolView's own menu;
 *           the playback timer; the handle itself.
 * Called by: a tab, through index.js — and a tab holds a handle and reaches its
 *           viewer only through it.
 *
 * A HANDLE *IS* A VIEWER: one owner, one structure, one of everything (§ 7).
 *
 * What the handle carries (§ 9.2): lifecycle, playback, and ONE route to the
 * model, written `viewer.data`. There is no other way to it, and no other
 * viewer's model is reachable from it.
 *
 * NEVER (§ 7 level 2, § 9.2):
 *   - hold structure data of its own;
 *   - answer a question the model already answers. "Adding a read to the handle
 *     that the model already answers is the specific move this rule forbids" —
 *     a mirrored read is a second surface over the same fact, and one of the two
 *     is the one somebody forgets to update;
 *   - expose the drawing library's object, or the DOM inside the card. (§ 9.2
 *     names the library here; this file does not, because § 5.3 says everything
 *     above the sealed layer can be read end to end without learning which
 *     library draws the molecule — a "never" that names it teaches it.);
 *   - accept a finished appearance. There is no "set the arrows", "set the
 *     atom-number labels", "show a busy state", "add a toggle": arrows, labels
 *     and the highlight are WORKED OUT FROM THE DATA by the renderEngine, never
 *     given to it;
 *   - own playback's effect on truth. The timer lives here, but it moves the
 *     frame through the same write everyone else uses (§ 6.4).
 *
 * EMPTY BY DESIGN — plan step A. Step G writes the body.
 */
"use strict";
