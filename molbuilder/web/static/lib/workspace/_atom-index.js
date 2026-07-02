/* Atom index-base conversion — the L1 interpretation of the
 * 0-based-internal / 1-based-user-facing rule (data-vocabulary.md § 3.1).
 *
 * ──────────────────────────── L1 CONTRACT ────────────────────────────
 * WHAT L1 IS.  The LOW-LEVEL PRESENTATION API for atom indices.  It exists only
 *   to serve the UI (the backend never converts index base) and it OWNS the
 *   primitive + the convention: internal 0-based ⇄ display 1-based.
 * PRIMITIVES, NOT FINISHED PRESENTATION.  toDisplay(i) returns the display
 *   NUMBER (i+1) — a primitive value.  It does NOT format it ("#5"), pick a
 *   widget, or render.  Higher presentation layers (L2 store / L3 panel +
 *   viewer) COMPOSE this primitive into formatted, rendered UI.
 * HIGHER LAYERS BUILD ON THIS.  L2/L3 MUST use L1's primitive + convention and
 *   never re-derive them.  A layer that cannot import L1 at runtime (the
 *   standalone viewer embed) MUST provably CONFORM — bound by a test.
 * GUARANTEES (bound by tests/test_atom_index_js.py):
 *   toDisplay(i) === i + 1 ;  fromDisplay is its inverse ;
 *   shiftExpression shifts every index bound in an expression by `delta`.
 *   The viewer's inline "+1" labels conform to toDisplay (drift-guarded).
 * ──────────────────────────────────────────────────────────────────────
 *
 * PURE: no DOM/store/HTTP.  Browser global (window.molbuilder.atomIndexModel)
 * AND node export, so the contract is unit-tested under Node without a browser.
 */
(function (root) {
    "use strict";

    function toDisplay(i)   { return i + 1; }   // internal -> user-facing
    function fromDisplay(i) { return i - 1; }   // user-facing -> internal

    /**
     * Shift every integer / range bound in a by-index expression by ``delta``.
     * "1-4, 6, 10-11"  --shiftExpression(-1)-->  "0-3, 5, 9-10".
     * Used to translate the user's 1-based filter input into the 0-based
     * expression the server ``by_index_range`` rule expects.  Preserves token
     * order; tolerates whitespace; leaves unrecognised tokens untouched (the
     * server validates).
     */
    function shiftExpression(expr, delta) {
        if (typeof expr !== "string") return expr;
        return expr.split(",").map(function (tok) {
            var t = tok.trim();
            if (t === "") return t;
            var m = t.match(/^(\d+)\s*-\s*(\d+)$/);
            if (m) {
                return (parseInt(m[1], 10) + delta) + "-"
                     + (parseInt(m[2], 10) + delta);
            }
            if (/^\d+$/.test(t)) return String(parseInt(t, 10) + delta);
            return t;
        }).join(", ");
    }

    var api = {
        toDisplay: toDisplay,
        fromDisplay: fromDisplay,
        shiftExpression: shiftExpression,
    };
    root.molbuilder = root.molbuilder || {};
    root.molbuilder.atomIndexModel = api;
    if (typeof module !== "undefined" && module.exports) module.exports = api;

})(typeof window !== "undefined" ? window
   : (typeof globalThis !== "undefined" ? globalThis : this));
