/* Shared 3Dmol style-spec builder for the Build and Watch viewers.
 *
 * Both pages use 3Dmol.js and offer a representation picker (stick / ball-and-stick / sphere /
 * line) plus an atom-radius scale.  This module is the single source of truth for the
 * per-representation sizing numbers so both viewers stay in lock-step.
 *
 * Invocation: molbuilder.style.spec({ rep, scale, colorscheme }) -> a 3Dmol style object.
 *   rep         : "stick" | "ballstick" | "sphere" | "line"
 *   scale       : number, atom-radius scale factor (default 1.0)
 *   colorscheme : string | null (null/undefined drops the key so 3Dmol uses its defaultcolors)
 *
 * A native ES module (the MolView embed graph); it ALSO publishes the transitional
 * Consumers `import { spec }` from it (the MolView embed graph); no window.molbuilder.style global.
 */
"use strict";


export function spec(opts) {
    opts = opts || {};
    const rep   = opts.rep   || "stick";
    const scale = opts.scale || 1.0;
    const cs    = opts.colorscheme;
    // Drop the colorscheme key when null/undefined so 3Dmol uses its viewer-level defaultcolors.
    const colorOpt = cs ? { colorscheme: cs } : {};

    switch (rep) {
        case "sphere":
            // True CPK: full vdW radius per element.
            return { sphere: { scale: 1.0 * scale, ...colorOpt } };
        case "line":
            return { line: { linewidth: 1 + 2 * scale, ...colorOpt } };
        case "ballstick":
            // Balls scale with vdW radius, sticks are a fixed thickness.
            return {
                stick:  { radius: 0.12 * scale, ...colorOpt },
                sphere: { scale:  0.32 * scale, ...colorOpt },
            };
        case "stick":
        default:
            // Plain licorice has no per-element size; tack on tiny spheres so Au reads apart from H.
            return {
                stick:  { radius: 0.16 * scale, ...colorOpt },
                sphere: { scale:  0.18 * scale, ...colorOpt },
            };
    }
}

