/* MolView — every control MolView itself draws.
 *
 * Contract: docs/web/molview.md § 1.1 (what it looks like in use), § 9.5 (the
 *           panel and the switches), § 11.4 (who owns the Export menu), § 11.6
 *           (measurement is its own layer, not part of drawing), § 6.4 (the
 *           frame bar).
 * Owns:     the selection panel and the click-to-select wiring; the frame bar;
 *           the switches; the View menu; the Export menu, snapshot and GIF;
 *           the measurement readout; the corner badge.
 * Called by: mount.js, which assembles them. Each control is a CALLER OF THE
 *           MODEL — the same doors a tab would use, with the same rules and the
 *           same read-only gate (§ 9.4) in front of them.
 *
 * NEVER:
 *   - talk to the 3D window directly (§ 7.3). "The panel reads what is selected;
 *     it never talks to the 3D window directly." Click-to-select arrives as data
 *     from below; everything the panel does goes back out through the model.
 *   - hold truth of its own. A control that remembers the displayed frame, the
 *     count, or what is selected has given that fact a second home (§ 5.2).
 *   - hand a finished appearance downward (§ 9.2). A control asks for a change in
 *     the DATA or a SWITCH; what that looks like is worked out below it.
 *   - reach past the model to a store it was not given (§ 7 level 4).
 *
 * The distance/angle maths belongs to measurement, which is its own layer
 * (§ 11.6) and not part of drawing.
 *
 * EMPTY BY DESIGN — plan step A. Step G writes the body; § 1.1 end to end lands
 * there, the first browser check of the programme.
 */
"use strict";
