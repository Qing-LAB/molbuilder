/* Shared 3Dmol style-spec builder for the Build and Watch viewers.
 *
 * Both pages use 3Dmol.js to render molecular structures and offer
 * the user a representation picker (stick / ball-and-stick / sphere /
 * line) plus an atom-radius scale slider.  The Watch page additionally
 * exposes a color-scheme select; the Build page doesn't.  This module
 * is the single source of truth for the per-representation sizing
 * numbers (radii, scales, line widths) so both viewers stay in
 * lock-step when the spec evolves.
 *
 * Invocation: molbuilder.style.spec({ rep, scale, colorscheme }) ->
 * 3Dmol style object, ready to pass to viewer.setStyle().
 *
 *   rep         : "stick" | "ballstick" | "sphere" | "line"
 *   scale       : number, atom-radius scale factor (default 1.0)
 *   colorscheme : string | null, e.g. "Jmol" / "rasmol" / "default";
 *                 null/undefined drops the colorscheme key entirely
 *                 so 3Dmol falls back to the viewer's defaultcolors
 *                 (Jmol on Build, controlled on Watch).
 *
 * Sizing rationale: stick + ballstick include a small sphere overlay
 * so heavy/light atoms (Au vs H, S vs C) stay visually distinguishable
 * even in licorice modes; pure sphere uses the vdW-radius scale; line
 * mode uses the slider as a linewidth multiplier (1 + 2*scale, so the
 * default 1.0 gives ~3 px width).
 */
(function (root) {
    "use strict";

    function spec(opts) {
        opts = opts || {};
        const rep   = opts.rep   || "stick";
        const scale = opts.scale || 1.0;
        const cs    = opts.colorscheme;
        // Drop the colorscheme key when null/undefined so 3Dmol uses
        // its viewer-level defaultcolors instead.  Spread merges in
        // an empty object when there's no colorscheme.
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
                // Plain licorice has no per-element size; tack on tiny
                // spheres so the user can still tell Au from H.
                return {
                    stick:  { radius: 0.16 * scale, ...colorOpt },
                    sphere: { scale:  0.18 * scale, ...colorOpt },
                };
        }
    }

    // Expose under the molbuilder namespace.  Idempotent so it's safe
    // to load this script before either viewer.js or both.
    root.molbuilder = root.molbuilder || {};
    root.molbuilder.style = root.molbuilder.style || {};
    root.molbuilder.style.spec = spec;
})(typeof window !== "undefined" ? window : this);
