/* MolView — the renderEngine, and the drawing commands beneath it. Two levels,
 * one file, because § 9.7 splits them itself: "a MATHS HALF that works out what
 * to draw with no drawing library anywhere near it, and an I/O HALF that is the
 * only code allowed to issue drawing commands." That split is why the
 * interesting part can be exercised with no browser at all (§ 13.2).
 *
 * Contract: docs/web/molview.md § 9.7, § 9.8, § 10 whole.
 *
 * ── The maths half — § 7 level 5 ──────────────────────────────────────────────
 * Owns:     nothing. It is HANDED the master copy, the selection and the
 *           switches, works out what each frame looks like, and passes the
 *           result down. Chooses how much work a change costs (§ 10.5: frame
 *           swap / overlay refresh / append / rebuild) by WHAT CHANGED, never by
 *           atom count. Holds the rebuild window (§ 10.9), where nothing that
 *           arrives is dropped. Checks its own work (§ 10.10).
 * Called by: the model, and only the model.
 * Surface:  COMMANDS ONLY. "Here is the data", "here is the cell", "add these
 *           frames", "the forces changed", "show this frame", "draw", "throw it
 *           away". Every one an instruction; none a question, "because the
 *           renderEngine is told what to draw and is never consulted about what
 *           the data is."
 * NEVER (§ 7 level 5):
 *   - keep its own copy of the displayed frame;
 *   - answer a question about what the data is;
 *   - run a change notification of its own.
 *
 * ── The I/O half — § 7 level 6 ────────────────────────────────────────────────
 * Owns:     exactly one fact — the multi-frame format the library expects.
 * Called by: the maths half, and nothing else.
 * Surface:  small, decision-free operations. Load frames, swap to a frame,
 *           append frames, apply the overlays, set this frame's arrows, set the
 *           cell geometry, show or hide the "Updating view…" cover, batch a
 *           group of changes so the screen updates once — and produce a picture
 *           of what is currently drawn, since only the bottom can do that
 *           (§ 11.4). Plus the two self-check questions of § 9.8, asked only by
 *           the maths half, only about the drawing itself.
 * NEVER (§ 7 level 6):
 *   - decide how much work a change needs — that is the maths half's call;
 *   - hold state;
 *   - answer anything upward.
 *
 * Data goes down. Nothing comes back up (§ 7.1). The per-frame result carries
 * CONTENT and never styling (§ 6.5) — a `color` or a `radius` on per-frame data
 * is the specific defect this rule exists to catch.
 *
 * EMPTY BY DESIGN — plan step A. Step C writes the per-frame maths of § 10.3
 * and § 6.5; step F writes the cost decision, the rebuild window and the
 * self-checks.
 */
"use strict";
