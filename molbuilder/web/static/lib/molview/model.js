/* MolView — the model: the one place the structure lives. § 7 level 3, and the
 * central file of § 7.3.
 *
 * Contract: docs/web/molview.md § 9.3 (the data API), § 9.4 (read-only),
 *           § 6.3 (two copies), § 6.4 (the displayed frame and its range).
 * Owns:     THE MASTER COPY — what a save, an export, a measurement and a server
 *           request all read — plus the selection, the displayed frame and its
 *           range. This is where the rules are enforced and where read-only is
 *           applied, so nothing may go around it.
 * Called by: the handle, and every level inside the same viewer. One model per
 *           owner.
 *
 * Every read of data returns a COPY, so changing what you were given can never
 * change what the viewer holds (§ 9.3).
 *
 * The ordering rule (§ 6.4), which is the whole reason the master copy lives
 * here and not below: the master copy is updated FULLY, then the range is
 * recomputed FROM IT, then the frame index is checked against that range, and
 * only then is one notification sent.
 *
 * Read-only is ONE QUESTION asked of every truth-changing door (§ 9.4) — a
 * no-op that does not throw, not a list of disabled buttons.
 *
 * NEVER (§ 7 level 3):
 *   - touch the drawing library;
 *   - exist as one shared instance behind several viewers.
 *
 * EMPTY BY DESIGN — plan step A. Step D writes the body.
 */
"use strict";
