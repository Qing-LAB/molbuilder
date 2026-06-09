/* Selection-driven measurements: xyz / distance / angle.
 *
 * Pure helpers — no DOM, no fetch, no store coupling.  Consumers
 * (Selection panel on Molbuilder, trajectory inspector + structure
 * inspector on Results) pass in the current selection indices, the
 * atoms-metadata array (for labels), and a positions array
 * ``[[x,y,z], …]``.  The module returns ``null`` for selections
 * that don't map to a measurement (0 or 4+ atoms), or a typed
 * envelope describing the result + a pre-formatted human display
 * string.
 *
 * Why this is its own module: both the Modify selection panel and
 * the trajectory inspector need the same math + the same display
 * conventions (1-based indexing, 3-decimal coords, 4-decimal
 * distances, 1-decimal angles, atom-name vs element fallback for
 * labels).  Drift between the two — different decimal counts, one
 * using 0-based and the other 1-based — was the original sin that
 * prompted this extraction.
 *
 * Result envelope (one of):
 *
 *   null  — selection is empty or 4+ atoms (caller hides the row).
 *
 *   { kind: "xyz",
 *     indices: [i],
 *     labels:  [labelOf(i)],
 *     xyz:     [x, y, z],
 *     display: "Au #3 — (0.000, 0.000, 0.000) Å" }
 *
 *   { kind: "distance",
 *     indices: [i, j],
 *     labels:  [labelOf(i), labelOf(j)],
 *     valueAng: 0.957,
 *     display:  "|H #5 – O #0| = 0.957 Å" }
 *
 *   { kind: "angle",
 *     indices: [i, j, k],
 *     labels:  [labelOf(i), labelOf(j), labelOf(k)],
 *     valueDeg: 104.5,
 *     display:  "∠H #5 – O #0 – H #6 = 104.5°" }
 *
 * Indices in the envelope preserve the input order; the second
 * atom in an angle is the vertex (the 1-2 and 2-3 bonds meet at
 * it).  Atom-name (e.g. "OE1") is preferred over element if the
 * atoms-metadata row carries one — chemists read residue names
 * faster than raw indices.
 */
(function (root) {
    "use strict";

    function _vec3sub(a, b) {
        return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
    }

    function _vec3norm(v) {
        return Math.sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
    }

    function _vec3dot(a, b) {
        return a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
    }

    function distance(p, q) {
        return _vec3norm(_vec3sub(p, q));
    }

    /**
     * Bond angle in degrees, with ``b`` at the vertex.  Returns
     * the angle formed by the two bonds a-b and b-c (i.e. the
     * angle between vectors (a-b) and (c-b)).  Clamps the cosine
     * to [-1, 1] before acos to avoid NaN on near-collinear
     * triples where floating-point error pushes the dot product
     * slightly outside the valid range.
     */
    function angleDeg(a, b, c) {
        const u = _vec3sub(a, b);
        const v = _vec3sub(c, b);
        const nu = _vec3norm(u);
        const nv = _vec3norm(v);
        if (nu === 0 || nv === 0) return NaN;
        let cos = _vec3dot(u, v) / (nu * nv);
        if (cos >  1) cos =  1;
        if (cos < -1) cos = -1;
        return Math.acos(cos) * 180 / Math.PI;
    }

    /**
     * Human label for an atom.  Prefers an explicit atom_name
     * (e.g. ``OE1``, ``Au-L1``) since residue-aware names tell
     * the chemist what they're looking at; falls back to
     * ``${element} #${index+1}`` when no atom_name is present.
     * Always 1-based to match the indices the user sees in the
     * atom list and the viewer overlays.
     */
    function labelOf(idx, atomsMeta) {
        const a = atomsMeta && atomsMeta[idx];
        if (a) {
            const name = (a.atom_name || a.atomName || "").toString().trim();
            if (name) return name + " #" + (idx + 1);
            const el = (a.element || "").toString().trim();
            if (el) return el + " #" + (idx + 1);
        }
        return "#" + (idx + 1);
    }

    function _fmtCoord(v) {
        return Number(v).toFixed(3);
    }

    function _fmtXyz(p) {
        return "("
            + _fmtCoord(p[0]) + ", "
            + _fmtCoord(p[1]) + ", "
            + _fmtCoord(p[2])
            + ") Å";
    }

    /**
     * The single entry point.  Returns ``null`` when ``selection``
     * is empty or has 4+ atoms; otherwise the typed envelope
     * documented at the top of this file.  Caller is responsible
     * for hiding the readout when this returns null.
     *
     * Bails (returns null) if any selected index is out of range
     * for ``positions`` — caller should treat that as "no
     * measurement yet" (positions array hasn't caught up with a
     * recent atom-count-changing op).
     *
     * ``pickOrder`` (optional) is the user's click sequence.
     * When supplied with three atoms, the second click is taken
     * as the angle vertex (the chemist's convention: pick the
     * two ends + the middle atom in the middle of the sequence).
     * Falls back to the geometric heuristic (vertex = atom with
     * smallest sum-of-distances to the other two) when pickOrder
     * is missing or doesn't match the selection set.
     */
    function compute(selection, atomsMeta, positions, pickOrder) {
        if (!Array.isArray(selection)) return null;
        const n = selection.length;
        if (n === 0 || n > 3) return null;
        if (!Array.isArray(positions)) return null;

        for (let i = 0; i < n; i++) {
            const idx = selection[i];
            if (typeof idx !== "number"
                || !positions[idx]
                || positions[idx].length < 3) {
                return null;
            }
        }

        const labels = selection.map((i) => labelOf(i, atomsMeta));

        if (n === 1) {
            const idx = selection[0];
            const p = positions[idx];
            return {
                kind:    "xyz",
                indices: [idx],
                labels:  [labels[0]],
                xyz:     [p[0], p[1], p[2]],
                display: labels[0] + " — " + _fmtXyz(p),
            };
        }

        if (n === 2) {
            // Use pick order when available so the readout reflects
            // the user's "from → to" intent — even though distance
            // is symmetric, the label A-B reads naturally as "the
            // bond I just drew."
            const order = _pickOrderMatches(pickOrder, selection)
                ? pickOrder : selection;
            const [iA, iB] = order;
            const d = distance(positions[iA], positions[iB]);
            const labA = labelOf(iA, atomsMeta);
            const labB = labelOf(iB, atomsMeta);
            return {
                kind:     "distance",
                indices:  [iA, iB],
                labels:   [labA, labB],
                valueAng: d,
                display:  "|" + labA + " – " + labB + "| = "
                          + d.toFixed(4) + " Å",
            };
        }

        // n === 3: angle.  Pick-order semantics win when the
        // user clicked the atoms in sequence (chemist's
        // convention: middle click = vertex), so the readout
        // reflects 1-2 + 2-3 bonds meeting at atom 2.  When
        // pickOrder is missing or stale (filter eval, session
        // restore, batch setSelection from a non-click source),
        // fall back to the geometric heuristic — vertex = atom
        // with smallest sum-of-distances to the other two, which
        // for a typical bonded triple still gives the centre
        // atom.
        let triplet;
        if (_pickOrderMatches(pickOrder, selection)) {
            triplet = pickOrder.slice();
        } else {
            triplet = _geometricVertexOrder(selection, positions);
        }
        const [iA, iV, iB] = triplet;
        const deg = angleDeg(
            positions[iA], positions[iV], positions[iB]);
        const labA = labelOf(iA, atomsMeta);
        const labV = labelOf(iV, atomsMeta);
        const labB = labelOf(iB, atomsMeta);
        return {
            kind:        "angle",
            indices:     [iA, iV, iB],
            labels:      [labA, labV, labB],
            vertexIndex: iV,
            valueDeg:    deg,
            display:     "∠" + labA + " – " + labV + " – " + labB
                         + " = " + deg.toFixed(1) + "°",
        };
    }

    // True iff ``pickOrder`` is the same SET as ``selection`` —
    // i.e. the click trail is in sync with the sorted selection.
    // A mismatch (length, missing atoms, repeats) usually means
    // the source wasn't a click (filter eval, batch setSelection,
    // corrupt pickOrder) and the angle vertex must be chosen
    // geometrically.  Repeat detection is what saves the
    // angle math from feeding two identical vectors to angleDeg
    // and emitting "NaN°" — the helper is the last gate before
    // pickOrder[1] becomes the vertex.
    function _pickOrderMatches(pickOrder, selection) {
        if (!Array.isArray(pickOrder)) return false;
        if (pickOrder.length !== selection.length) return false;
        const set = new Set(selection);
        const seen = new Set();
        for (let i = 0; i < pickOrder.length; i++) {
            const idx = pickOrder[i];
            if (!set.has(idx) || seen.has(idx)) return false;
            seen.add(idx);
        }
        return true;
    }

    // Reorder a 3-atom selection so the geometric centre atom
    // (smallest sum-of-distances to the other two) is in the
    // middle.  Used when pickOrder is unavailable.
    function _geometricVertexOrder(selection, positions) {
        const [i0, i1, i2] = selection;
        const p0 = positions[i0], p1 = positions[i1], p2 = positions[i2];
        const s0 = distance(p0, p1) + distance(p0, p2);
        const s1 = distance(p1, p0) + distance(p1, p2);
        const s2 = distance(p2, p0) + distance(p2, p1);
        let vertexLocal = 0;
        if (s1 < s0) vertexLocal = 1;
        if (s2 < (vertexLocal === 0 ? s0 : s1)) vertexLocal = 2;
        const others = [0, 1, 2].filter((k) => k !== vertexLocal);
        return [
            selection[others[0]],
            selection[vertexLocal],
            selection[others[1]],
        ];
    }

    const api = {
        compute:   compute,
        distance:  distance,
        angleDeg:  angleDeg,
        labelOf:   labelOf,
    };

    if (typeof module !== "undefined" && module.exports) {
        module.exports = api;
    }
    if (root) {
        root.molbuilder = root.molbuilder || {};
        root.molbuilder.selection = root.molbuilder.selection || {};
        root.molbuilder.selection.measurements = api;
    }
})(typeof window !== "undefined" ? window : null);
