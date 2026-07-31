/* MolView — the in-repo demo page.
 *
 * Contract: docs/web/molview.md § 13.4 (what makes this testable at all).
 * Owns:     a page that mounts a viewer over a multi-frame structure, using the
 *           module exactly as any other consumer does.
 * Called by: nobody. It is a consumer, not a layer.
 *
 * NEVER:
 *   - import anything but index.js. The demo is worth having precisely because
 *     it is held to § 4's single import like every other consumer; a demo with a
 *     private door proves nothing.
 *   - be what a lower-level change is judged by. § 13.2's third level is a
 *     FINISHED-MODULE check: a page-level test fails for anything on the page,
 *     so using one to judge a data-structure or an API change says nothing about
 *     the change and throws away work that was correct (plan § 1).
 *
 * EMPTY BY DESIGN — plan step A. Step G writes the body, after there is a module
 * to demonstrate.
 */
"use strict";
