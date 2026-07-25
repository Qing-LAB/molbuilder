/* molbuilder shared 3Dmol viewer factory.
 *
 * Single entry point for tab code that instantiates a 3Dmol GLViewer.  Routing every caller
 * through one factory makes a future change to the default style spec land in ONE place.
 *
 * Used via ``molbuilder.viewer.create(target, opts?)`` / `import { viewer } from "./mol-viewer.js"`.
 *   target -- the DOM element OR element-id string $3Dmol.createViewer accepts.
 *   opts   -- optional overrides (backgroundColor, defaultcolors).
 * Returns a 3Dmol GLViewer instance; the factory does NOT cache or track viewers.
 *
 * A native ES module: it exports the shared `viewer` object (with `.create`); mol-viewer-embed.js
 * IMPORTS that same object and adds `.embed` to it.  It ALSO publishes the transitional
 * ``window.molbuilder.viewer`` global (the SAME object) for not-yet-migrated classic readers.
 */
"use strict";

const root = (typeof window !== "undefined") ? window : globalThis;

export function create(target, opts) {
    const $3Dmol = root.$3Dmol;
    if (!$3Dmol) {
        throw new Error("mol-viewer.create: 3Dmol-min.js must be loaded first");
    }
    const defaults = {
        backgroundColor: "white",
        defaultcolors:   $3Dmol.elementColors.Jmol,
    };
    const merged = Object.assign({}, defaults, opts || {});
    return $3Dmol.createViewer(target, merged);
}

// The shared viewer surface.  mol-viewer-embed.js imports THIS object and adds `.embed` to it, so
// `import { viewer }` from either file resolves to the one object (create + embed).
export const viewer = { create };

// ── Transitional global (§3.2 shim): the SAME object as the export. ──
root.molbuilder = root.molbuilder || {};
root.molbuilder.viewer = viewer;
