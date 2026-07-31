/* MolView — session history: the sequence of states, the position on it, and the
 * write machine.
 *
 * Contract: docs/web/molview.md § 11.2 (session history, and what the workspace
 *           owns), § 7.3 (a helper, handed exactly what it may call).
 * Owns:     the sequence and the position on it; point 0; the write machine
 *           (SETTLED / CHANGING / WRITING); the badge that says where you are.
 * Called by: model.js, and nothing else.
 *
 * What it is handed (§ 7.3): "record the current state" + "put a state back";
 * where the bytes go. Nothing more.
 *
 * NEVER (§ 11.2):
 *   - know or care what is in it. This is the clearest case of the § 7.3 rule:
 *     the helper has to save and restore the structure, but it does not need to
 *     know the file format.
 *
 * WHY THIS IS ITS OWN FILE and not part of model-jobs.js: § 11.2's claim that
 * "the mechanism does not know or care what is in it" only holds while it is not
 * sitting beside the serialiser it is handed (plan § 2).
 *
 * Retract spends unsaved work first (§ 11.2) — the rule that keeps undo from
 * silently discarding something the user has not saved.
 *
 * EMPTY BY DESIGN — plan step A. Step E writes the body.
 */
"use strict";
