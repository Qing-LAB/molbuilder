/* mol-axes.js — orientation-reference axis triad for 3Dmol viewers.
 *
 * Why this exists.  Every tab that mounts a 3Dmol viewer wants the
 * same affordance: a small RGB triad the user can toggle on as an
 * orientation reference while rotating the camera.  Pre-this-module
 * the triad was inline in modify/viewer.js with a fixed 1.5 Å
 * length anchored at the world origin -- useful as a "compass"
 * for a non-periodic molecule, but the wrong shape when the
 * structure has a periodic cell (the triad should follow the a/b/c
 * lattice vectors, not stay at the world axes).
 *
 * This module unifies both cases under one API + adds the cell
 * mode that #118 asked for.  Two modes:
 *
 *   * **Cartesian** (no cell) — fixed-length unit-X / unit-Y / unit-Z
 *     arrows at the origin, labels "x" / "y" / "z".  Compass mode.
 *     Use this when the structure is non-periodic (XYZ load, peptide
 *     build, modify-tab pre-cell-definition state).
 *   * **Cell** (cell vectors supplied) — arrows along the lattice
 *     vectors a, b, c, scaled to the cell vector lengths, labels
 *     "a" / "b" / "c".  Use this when the structure has a periodic
 *     cell (parsed SIESTA .out, PDB CRYST1, future sidecar metadata).
 *     Pairs naturally with the cell-wireframe overlay that the
 *     trajectory inspector already draws.
 *
 * Public API.  ``export const axes`` (the MolView embed graph):
 *
 *   draw(viewer, opts) -> { clear: () => void }
 *
 * A native ES module; the embed `import`s `axes` directly (``import { axes } from
 * "./mol-axes.js"``).  It is a pure, stateless helper (a MolView-internal Axes-toggle detail),
 * so there is NO window.molbuilder.axes global -- nothing external ever reaches in to draw axes.
 *
 * Pure logic lives in ``_buildAxisSpecs(opts)`` (no 3Dmol dependency)
 * so it can be unit-tested under Node without a browser.  See
 * tests/test_mol_axes_js.py.
 */
"use strict";

// (IIFE unwrapped -> native ES module; the former body stays indented, harmless.)

    // Fixed length for the Cartesian-mode triad.  At 1.5 Å the triad
    // is longer than a typical bond (~1.4 Å) and short enough to stay
    // out of the way of the structure even in dense junctions.
    const DEFAULT_CARTESIAN_LEN_ANG = 1.5;

    // Default per-axis colors.  Same hex codes as the legacy
    // modify/viewer.js triad to preserve visual continuity.
    // Cartesian (x/y/z) and cell (a/b/c) share the same triplet:
    // red / green / blue maps to the first / second / third axis
    // in each mode.
    const DEFAULT_COLORS = {
        x: "0xff5555", y: "0x55cc55", z: "0x5588ff",
        a: "0xff5555", b: "0x55cc55", c: "0x5588ff",
    };

    // Label is placed 15 % past the arrow tip so it sits clear of the
    // arrowhead without flying off into space at long cell vectors.
    const LABEL_PAST_TIP_FRACTION = 1.15;

    /**
     * Compute the three arrow specs (start / end / label / color)
     * for a given options object.  Pure: no 3Dmol calls, no DOM, no
     * mutation of opts.  Returns an array of three plain objects.
     *
     * Exported (on the `axes` object: ``axes._buildAxisSpecs``) for
     * unit testing; production callers should use ``draw(viewer, opts)``.
     */
    function _buildAxisSpecs(opts) {
        opts = opts || {};
        const origin = (opts.origin && opts.origin.length === 3)
            ? [opts.origin[0], opts.origin[1], opts.origin[2]]
            : [0, 0, 0];
        const colors = Object.assign({}, DEFAULT_COLORS, opts.colors || {});

        let vectors;
        let labels;
        if (_looksLikeCell(opts.cell)) {
            // Cell mode: vectors are the supplied lattice rows.
            vectors = opts.cell.map((v) => [v[0], v[1], v[2]]);
            labels  = opts.labels || ["a", "b", "c"];
        } else {
            // Cartesian mode: unit X/Y/Z scaled by ``length``.
            const L = (typeof opts.length === "number" && opts.length > 0)
                ? opts.length
                : DEFAULT_CARTESIAN_LEN_ANG;
            vectors = [[L, 0, 0], [0, L, 0], [0, 0, L]];
            labels  = opts.labels || ["x", "y", "z"];
        }

        const specs = [];
        for (let i = 0; i < 3; i++) {
            const v = vectors[i];
            const lbl = labels[i];
            // Color lookup: explicit per-label override wins, then
            // per-mode default ("a"/"b"/"c" or "x"/"y"/"z").  An
            // unrecognised label falls back to grey so a typo is
            // visible rather than silently invisible.
            const color = colors[lbl]
                || colors[lbl && lbl.toLowerCase && lbl.toLowerCase()]
                || "0x888888";
            const start = { x: origin[0], y: origin[1], z: origin[2] };
            const end = {
                x: origin[0] + v[0],
                y: origin[1] + v[1],
                z: origin[2] + v[2],
            };
            const labelPos = {
                x: origin[0] + v[0] * LABEL_PAST_TIP_FRACTION,
                y: origin[1] + v[1] * LABEL_PAST_TIP_FRACTION,
                z: origin[2] + v[2] * LABEL_PAST_TIP_FRACTION,
            };
            specs.push({ start, end, label: lbl, color, labelPos });
        }
        return specs;
    }

    /**
     * Cheap "is this a 3x3 lattice" check.  Returns true iff cell is
     * an array of 3 length-3 numeric arrays.  Null / undefined /
     * malformed shape returns false (Cartesian fallback fires).
     */
    function _looksLikeCell(cell) {
        if (!Array.isArray(cell) || cell.length !== 3) return false;
        for (let i = 0; i < 3; i++) {
            const row = cell[i];
            if (!Array.isArray(row) || row.length !== 3) return false;
            for (let j = 0; j < 3; j++) {
                if (typeof row[j] !== "number"
                    || !Number.isFinite(row[j])) {
                    return false;
                }
            }
        }
        return true;
    }

    /**
     * Draw an axis triad in a 3Dmol viewer.  Returns a handle with
     * ``.clear()`` that removes every shape + label the call created.
     * Safe to call ``.clear()`` multiple times.
     *
     * **Two modes** (selected by whether ``opts.cell`` is present):
     *
     *   * Cartesian — unit X/Y/Z arrows at the origin, labels "x"/
     *     "y"/"z", length = ``opts.length`` (default 1.5 Å).
     *   * Cell      — arrows along the supplied lattice vectors a/b/c,
     *     labels "a"/"b"/"c", length = the cell vector lengths
     *     themselves.
     *
     * Options:
     *   * ``cell``     (number[3][3] | null) — 3 lattice row vectors
     *                                          in Å.  Triggers cell
     *                                          mode when valid.
     *   * ``origin``   (number[3])           — start point; default
     *                                          [0,0,0].
     *   * ``length``   (number)              — Cartesian-mode length
     *                                          (Å); default 1.5.
     *   * ``labels``   (string[3])           — override per-axis labels.
     *   * ``colors``   ({label: hex})        — override per-axis color.
     *   * ``radius``   (number)              — arrow shaft radius;
     *                                          default 0.05 Å (matches
     *                                          legacy modify triad).
     *   * ``render``   (bool)                — call viewer.render()
     *                                          after the shapes land;
     *                                          default true.
     *
     * Returns: { clear() }
     */
    function draw(viewer, opts) {
        opts = opts || {};
        if (!viewer || typeof viewer.addArrow !== "function") {
            // Defensive: return a no-op handle so callers don't have
            // to guard every call site.  Errors that would have come
            // from the missing viewer are documented at the API level
            // rather than producing runtime exceptions.
            return { clear: function () { /* no shapes to clear */ } };
        }
        const specs = _buildAxisSpecs(opts);
        const radius = (typeof opts.radius === "number" && opts.radius > 0)
            ? opts.radius
            : 0.05;

        const shapes = [];
        const labelHandles = [];
        for (let i = 0; i < specs.length; i++) {
            const s = specs[i];
            const arrow = viewer.addArrow({
                start:       s.start,
                end:         s.end,
                radius:      radius,
                radiusRatio: 2.5,
                mid:         0.85,
                color:       s.color,
            });
            shapes.push(arrow);
            const lbl = viewer.addLabel(s.label, {
                position:          s.labelPos,
                fontColor:         s.color,
                backgroundOpacity: 0.0,
                fontSize:          12,
                inFront:           true,
            });
            labelHandles.push(lbl);
        }
        if (opts.render !== false && typeof viewer.render === "function") {
            viewer.render();
        }

        let cleared = false;
        return {
            clear: function () {
                if (cleared) return;
                cleared = true;
                for (const s of shapes) {
                    try { viewer.removeShape(s); }
                    catch (_) { /* viewer torn down; ignore */ }
                }
                for (const l of labelHandles) {
                    try { viewer.removeLabel(l); }
                    catch (_) { /* viewer torn down; ignore */ }
                }
            },
        };
    }

    export const axes = {
        draw:             draw,
        _buildAxisSpecs:  _buildAxisSpecs,  // exported for unit tests
        _looksLikeCell:   _looksLikeCell,   // exported for unit tests
    };
