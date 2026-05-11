/* molbuilder shared selection-halo helper.
 *
 * Both the Modify and Watch tabs draw a high-contrast wireframe
 * sphere around picked atoms.  Same colour, same radius, same
 * rendering shape -- so the helper lives once here and both
 * viewers import it.
 *
 * Public API (attached to window.molbuilder.pick):
 *
 *     addHalo(viewer, {x, y, z})
 *         -- creates a wireframe sphere shape at the given centre;
 *            returns the shape object (caller stores so it can
 *            remove later).
 *
 *     clearHalos(viewer, shapes)
 *         -- removes every shape in the array from the viewer and
 *            empties the array.
 *
 * The colour is bright orange (#fb923c) chosen for contrast against
 * the default white viewer background AND against the dark colour
 * schemes used by 3Dmol's CPK / Jmol palettes.  Fixed radius 0.45 Å
 * so H and a heavy metal feel equally selectable.
 */

(function () {
    "use strict";
    const HALO_COLOR  = "#fb923c";
    const HALO_RADIUS = 0.45;
    const HALO_LW     = 3.0;

    function addHalo(viewer, pos) {
        return viewer.addSphere({
            center:    { x: pos.x, y: pos.y, z: pos.z },
            radius:    HALO_RADIUS,
            color:     HALO_COLOR,
            wireframe: true,
            linewidth: HALO_LW,
        });
    }

    function clearHalos(viewer, shapes) {
        for (const s of shapes) viewer.removeShape(s);
        shapes.length = 0;
    }

    window.molbuilder = window.molbuilder || {};
    window.molbuilder.pick = {
        HALO_COLOR:  HALO_COLOR,
        HALO_RADIUS: HALO_RADIUS,
        addHalo:     addHalo,
        clearHalos:  clearHalos,
    };
})();
