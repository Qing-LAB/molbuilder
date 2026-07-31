/* MolView — the two stores: what is picked out, and how it is drawn. § 7 level 4.
 *
 * Contract: docs/web/molview.md § 9.5 (`selection`) and § 9.6 (`view`).
 * Owns:     `selection` — what is picked, and the switches beside it.
 *           `view`      — style, radius, background, projection.
 * Called by: assembled by the model and reached ONLY through it (§ 9.3), so a
 *           change asked for through a store meets the same rules as one asked
 *           for anywhere else (§ 9.4).
 * Shape:    change-and-subscribe. "They exist so state has a home that knows
 *           nothing about drawing."
 *
 * NEVER (§ 7 level 4):
 *   - draw anything;
 *   - hold the displayed frame — that is not a switch (§ 6.4). It belongs to the
 *     model, with the range it is checked against;
 *   - be kept by anything outside the viewer once it has been reached.
 *
 * AND NEVER (§ 9.6): hold the camera. Not here, not in the model, not in the
 * handle — the camera is held nowhere above the drawing. `view` is the four
 * drawing settings and no more; the camera lives in the sealed layer because a
 * window must have a point of view, and § 9.9 explains why that is not a third
 * kind of question.
 *
 * EMPTY BY DESIGN — plan step A. Step E writes the body.
 */
"use strict";
