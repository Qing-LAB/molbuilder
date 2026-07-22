/* molbuilder shared formatting helpers used by multiple viewers.
 *
 * Single source of truth for numeric / chemical-formula presentation so the Build and Modify
 * viewers can't drift.  A native ES module (the MolView embed graph); it ALSO publishes the
 * transitional ``window.molbuilder.fmt`` global for any not-yet-migrated classic reader.
 */
"use strict";

const root = (typeof window !== "undefined") ? window : globalThis;

/**
 * Format a list of element symbols as a Hill-ordered chemical formula string, with subscripts
 * written inline (``H2O``, ``C6H12O6``).
 *
 * Convention: C and H first (common organic-chemistry layout), then N / O / P / S in that order,
 * then the rest alphabetic.  The ordering matches what the Build tab's info panel showed before
 * this helper was extracted; do not change without updating tests/test_*.py and
 * tests/test_modify_e2e.py that read the formula readout.
 */
export function formula(elements) {
    if (!elements || !elements.length) return "—";
    const counts = {};
    elements.forEach((e) => (counts[e] = (counts[e] || 0) + 1));
    const order = ["C", "H", "N", "O", "P", "S"];
    const parts = [];
    order.forEach((e) => {
        if (counts[e]) {
            parts.push(counts[e] > 1 ? `${e}${counts[e]}` : e);
            delete counts[e];
        }
    });
    Object.keys(counts).sort().forEach((e) => {
        parts.push(counts[e] > 1 ? `${e}${counts[e]}` : e);
    });
    return parts.join("");
}

// ── Transitional global (§3.2 shim): mirrors mol-style.js; classic consumers read molbuilder.fmt. ──
root.molbuilder = root.molbuilder || {};
root.molbuilder.fmt = root.molbuilder.fmt || {};
root.molbuilder.fmt.formula = formula;
