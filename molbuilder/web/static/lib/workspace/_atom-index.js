/* Atom index-base conversion — the single implementation of the
 * 0-based-internal / 1-based-user-facing rule (data-vocabulary.md § 3.1).
 *
 * display = internal + 1 ; internal = display - 1.  Convert ONLY at the
 * user-facing edge: never let a 1-based value into internal state, never show a
 * 0-based value to a user.
 *
 * PURE: no DOM/store/HTTP.  Browser global (window.molbuilder.atomIndexModel)
 * AND node export, so the logic is unit-tested under Node without a browser.
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
