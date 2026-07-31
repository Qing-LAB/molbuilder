/* MolView — the sealed layer: the only code in the module that names 3Dmol.
 *
 * Contract: docs/web/molview.md § 9.9, § 7 level 7.
 * Owns:     the DRAWING COPY — the movie, the camera, the styles, the picking,
 *           the highlight spheres. It draws the frame it is handed. (The camera
 *           is here because a window must have a point of view; § 9.6 is why
 *           nothing above it keeps one.)
 * Called by: level 6 — the drawing commands — and nothing else, ever.
 *
 * It answers EXACTLY TWO questions, and both are about itself (§ 9.9):
 *   - is there a movie loaded at all?
 *   - how many frames does it have?
 * Both are asked only by the layer immediately above, both exist so that layer
 * can find out whether its own last instruction landed (§ 10.10), and neither
 * answer ever reaches a user.
 *
 * NEVER (§ 7 level 7):
 *   - keep its own frame number, or be a source of truth about coordinates.
 *   - offer any way to read coordinates out, to ask which frame is showing, or
 *     to ask where the camera is pointing. "Everything else, it refuses."
 *   - be named by any consumer, or by the contract (§ 15).
 *
 * The line is not a loophole: "did what I told you to do land?" is a check;
 * "what is the structure?" is a question about the truth, and the truth is not
 * here. Everything this layer holds is either DERIVED (the drawing copy, worked
 * out from the master copy every redraw) or GIVEN to it.
 *
 * THE ONE FILE NOT WRITTEN FROM THE DOCUMENT — plan step B. It is carried across
 * from the frozen tree and carved down to the job above, because it is hard-won
 * knowledge about a library that punishes guessing: that restyling one atom
 * rebuilds the whole model's geometry, that shapes need re-placing every frame,
 * the batching behind § 10.7's measurements. The contract records the
 * conclusions, not the calls.
 *
 * Carved OUT of it and up into the layers above (plan § 3 B):
 *   - the card scaffold and info line       -> mount.js
 *   - the knob bar, the frame strip         -> ui.js
 *   - the animation interval                -> ui.js + mount's one timer
 *   - the export menu, snapshot, GIF encoder -> ui.js (§ 11.4)
 * Deleted outright: the `molbuilder.projects` reach and the `/api/files/*`
 * calls — a file route at the bottom of the stack (§ 6.7), and task #39.
 *
 * EMPTY BY DESIGN — plan step A. Step B carries and carves.
 */
"use strict";
