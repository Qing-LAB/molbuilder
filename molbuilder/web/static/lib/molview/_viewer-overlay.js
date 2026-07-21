/* MolView viewer-overlay FRAMEWORK — concealed corner overlays over the viewer square.
 *
 * MODULE: molview (lib/molview/).  A concealed ESM submodule imported by the pieces that draw
 *   overlays (mount.js's persistence indicator today; the measurement chip is the migration
 *   target).  No global shim -- internal consumers import `createViewerOverlay` directly.
 *
 * ROLE: ONE way to put a small pill/badge in a corner of the viewer square, so EVERY overlay the
 *   module draws is positioned + styled identically (from the design tokens, via the shared
 *   `.molview-overlay` class in fused-layout.css) with NO per-consumer px tweaking and consistent
 *   results across tabs/embeds.  The element anchors to `.viewer-wrap` (position: relative), so it
 *   tracks the viewer in every fold state -- NOT the card (whose width persists when the panel folds,
 *   which would strand a card-anchored overlay in the empty gap).
 *
 * USED BY: lib/molview/mount.js (persistence indicator).  createViewerOverlay(anchorEl, opts)
 *   returns a small handle; the caller drives show/hide off a data signal and dispose()s on teardown.
 *
 * opts: { corner: "top-right"|"top-left"|"bottom-right"|"bottom-left" (default "top-right"),
 *         kind?: "warn"|"info" (semantic tint), text?, title?, role?, live? (aria-live) }
 */
"use strict";

const CORNERS = { "top-right": 1, "top-left": 1, "bottom-right": 1, "bottom-left": 1 };

export function createViewerOverlay(anchorEl, opts) {
    opts = opts || {};
    // DOM-only (browser).  In a document-less context (node mount-test stub) return an inert handle
    // so callers need no guard of their own.
    if (typeof document === "undefined" || !anchorEl || typeof anchorEl.appendChild !== "function") {
        return { el: null, show: function () {}, hide: function () {},
                 toggle: function () {}, setText: function () {}, dispose: function () {} };
    }
    const corner = CORNERS[opts.corner] ? opts.corner : "top-right";
    const el = document.createElement("div");
    el.className = "molview-overlay molview-overlay--" + corner
                 + (opts.kind ? " molview-overlay--" + opts.kind : "");
    el.hidden = true;
    if (opts.role)  el.setAttribute("role", opts.role);
    if (opts.live)  el.setAttribute("aria-live", opts.live);
    if (opts.title) el.title = opts.title;
    if (opts.text != null) el.textContent = opts.text;
    anchorEl.appendChild(el);

    return {
        el: el,
        show:    function () { el.hidden = false; },
        hide:    function () { el.hidden = true; },
        toggle:  function (on) { el.hidden = !on; },
        setText: function (t) { el.textContent = t; },
        dispose: function () { try { el.remove(); } catch (_) { /* already detached */ } },
    };
}
