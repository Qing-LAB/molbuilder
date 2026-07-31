/* MolView — what an atom IS to everything above it: how it is numbered, and what
 * it can be filtered by.
 *
 * Contract: docs/web/molview.md § 11.5 (one atom-numbering translation, in one
 *           place) and § 9.5 / § 6.2 (the channels a filter row can match).
 * Owns:     the 0-based-in-code / 1-based-on-screen translation, and the
 *           enumeration of what kinds of thing an atom can be filtered by.
 * Called by: the model, the stores, the render engine and the UI — every level
 *           above reads it.
 *
 * NEVER:
 *   - let a bare `+1` or `-1` of its own be written anywhere else in the module
 *     (§ 11.5). This file is the single home; that is its entire reason to exist.
 *   - read anything back. It is the bottom of the import graph: no DOM, no store,
 *     no network, no drawing. Every layer above imports it and it imports none of
 *     them — which is why it is its own leaf rather than folded into the model.
 *
 * EMPTY BY DESIGN — plan step A. Step C writes the body, tests first.
 */
"use strict";
