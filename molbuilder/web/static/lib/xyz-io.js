/* molbuilder shared XYZ I/O helpers.
 *
 * Single source of truth for the minimal XYZ-format parse/format
 * pair that several tabs need (currently Spectra; Build + Modify
 * will pick this up if they ever need to round-trip an XYZ string
 * client-side instead of through /api/build/load).
 *
 * Used via ``window.molbuilder.xyz.parse(...)`` and ``.toText(...)``,
 * mirroring the lib/mol-format.js namespace convention.
 *
 * Format reminder (xyz v1):
 *   line 1 : <atom-count>
 *   line 2 : <title> (may be empty)
 *   lines  : <element> <x> <y> <z>      -- Å, 3 floats
 */
(function (root) {
    "use strict";

    /**
     * Parse an XYZ text block.  Tolerates extra whitespace and blank
     * trailing lines.  Returns ``{elements: string[], positions:
     * number[][]}`` (Å).  Throws Error on malformed input.
     */
    function parse(text) {
        const lines = String(text || "").split(/\r?\n/);
        if (lines.length < 3) throw new Error("xyz too short");
        const n = parseInt(lines[0].trim(), 10);
        if (!Number.isFinite(n) || n < 1) throw new Error("bad atom count");
        const elements = [];
        const positions = [];
        for (let i = 0; i < n; i++) {
            const parts = (lines[i + 2] || "").trim().split(/\s+/);
            if (parts.length < 4) throw new Error("bad atom line " + (i + 2));
            elements.push(parts[0]);
            positions.push([
                parseFloat(parts[1]),
                parseFloat(parts[2]),
                parseFloat(parts[3]),
            ]);
        }
        return { elements: elements, positions: positions };
    }

    /**
     * Format an ``{elements, positions}`` pair as an XYZ block.
     * Title line is an empty placeholder -- 3Dmol ingests this
     * shape fine.  Positions are written with 8-decimal precision
     * to round-trip the Å values without visible loss.
     */
    function toText(elements, positions) {
        const lines = [String(elements.length), ""];
        for (let i = 0; i < elements.length; i++) {
            const p = positions[i];
            lines.push(
                elements[i]
                + " " + p[0].toFixed(8)
                + " " + p[1].toFixed(8)
                + " " + p[2].toFixed(8)
            );
        }
        return lines.join("\n");
    }

    root.molbuilder = root.molbuilder || {};
    root.molbuilder.xyz = root.molbuilder.xyz || {};
    root.molbuilder.xyz.parse  = parse;
    root.molbuilder.xyz.toText = toText;
})(typeof window !== "undefined" ? window : this);
