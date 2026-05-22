/* molbuilder shared 3Dmol viewer factory.
 *
 * Single entry point for the four tabs that instantiate a 3Dmol
 * GLViewer.  Three of them (Build, Modify, Watch) want identical
 * defaults (white background, Jmol element colors); the fourth
 * (Spectra mode-animation) wants a dark background.  Routing all
 * four through one factory makes a future change to the default
 * style spec land in ONE place.
 *
 * Used via ``window.molbuilder.viewer.create(target, opts?)``.
 *
 * Arguments:
 *   target -- the DOM element OR the element id string 3Dmol's
 *             ``$3Dmol.createViewer`` accepts.
 *   opts   -- optional object that overrides the defaults.  Common
 *             keys: ``backgroundColor`` (CSS colour string),
 *             ``defaultcolors`` (3Dmol palette object).
 *
 * Returns whatever ``$3Dmol.createViewer`` returns -- a 3Dmol
 * GLViewer instance.  The factory does NOT cache or track viewers;
 * callers retain ownership.
 */
(function (root) {
    "use strict";

    function create(target, opts) {
        const $3Dmol = root.$3Dmol;
        if (!$3Dmol) {
            throw new Error(
                "mol-viewer.create: 3Dmol-min.js must be loaded first"
            );
        }
        const defaults = {
            backgroundColor: "white",
            defaultcolors:   $3Dmol.elementColors.Jmol,
        };
        const merged = Object.assign({}, defaults, opts || {});
        return $3Dmol.createViewer(target, merged);
    }

    root.molbuilder = root.molbuilder || {};
    root.molbuilder.viewer = root.molbuilder.viewer || {};
    root.molbuilder.viewer.create = create;
})(typeof window !== "undefined" ? window : this);
