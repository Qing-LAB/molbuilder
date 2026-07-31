/* MolView — the pure facts about atoms: how one is numbered, what one can be
 * filtered by, and how a set of them reads as a formula.
 *
 * Contract: docs/web/molview.md § 11.5 (one atom-numbering translation, in one
 *           place) and § 9.5 / § 6.2 (the channels a filter row can match).
 * Owns:     the 0-based-in-code / 1-based-on-screen translation, the enumeration
 *           of what kinds of thing an atom can be filtered by, and the Hill
 *           formula.
 * Called by: the model, the stores, the render engine and the UI — every level
 *           above reads it.
 *
 * NEVER:
 *   - let a bare `+1` or `-1` of its own be written anywhere else in the module
 *     (§ 11.5). This file is the single home; that is its entire reason to exist.
 *   - read anything back. It is the bottom of the import graph: no DOM, no store,
 *     no network, no drawing. Every layer above imports it and it imports none of
 *     them — which is why it is its own leaf rather than folded into the model.
 */
"use strict";


/* ══ Numbering ═══════════════════════════════════════════════════════════════
 *
 * § 11.5: atom numbers are 0-based in code and 1-based on screen, and "MolView
 * never writes a bare +1 of its own anywhere". Every surface that shows or
 * accepts an atom number comes through here — the measurement readout, the
 * atom-list column, the atom-number labels in the 3D window, the filter panel —
 * which is why they cannot drift apart, and why the first atom reads as #1
 * everywhere even though the code sees 0.
 *
 * This is the browser end of a rule that spans the application: the same
 * translation exists on the server side, and the number a user reads must equal
 * the atom number in the generated input file. Its single home is
 * model/overview.md § 2; MolView defers to it rather than restating it.
 */

/** The number a user reads, from the number the code holds. */
export function toDisplay(index) { return index + 1; }

/** The number the code holds, from the number a user typed. */
export function fromDisplay(number) { return number - 1; }

/**
 * Shift every whole number and range bound in a by-index expression.
 *
 *     "1-4, 6, 10-11"  --shiftExpression(-1)-->  "0-3, 5, 9-10"
 *
 * § 9.5: "by atom index" is the one filter row that crosses the numbering
 * boundary, because an atom's index is not something it HAS — it is where it
 * sits. The user types 1-based, matching what is on screen; the rule sent is
 * 0-based; and the shift happens exactly once, here, at the point the row
 * becomes a rule. Every other row compares names to names and never touches a
 * number.
 *
 * Token order is preserved and whitespace is tolerated. A token that is not a
 * number or a range is passed through untouched — the server validates the rule,
 * and silently dropping what it does not recognise would hide a typo rather than
 * report it.
 */
export function shiftExpression(expression, delta) {
    if (typeof expression !== "string") return expression;
    return expression.split(",").map(function (token) {
        const t = token.trim();
        if (t === "") return t;
        const range = t.match(/^(\d+)\s*-\s*(\d+)$/);
        if (range) {
            return (parseInt(range[1], 10) + delta) + "-"
                 + (parseInt(range[2], 10) + delta);
        }
        if (/^\d+$/.test(t)) return String(parseInt(t, 10) + delta);
        return t;
    }).join(", ");
}


/* ══ Channels — what a filter row can match ══════════════════════════════════
 *
 * § 6.2: the facts an atom carries are its ELEMENT, the LABELS it carries, and
 * its RESIDUE name if the source had one. § 9.5 enumerates the same three from
 * the atom when it builds the filter's rows.
 *
 * "They are the same list, which is why filtering needs no case per property"
 * (§ 6.2) — and the rule that keeps it that way is that the enumerated list and
 * the carried list MOVE TOGETHER. Neither can grow without the other.
 *
 * TWO CHANNELS THE OLD CODE HAD AND THIS DOES NOT, both because the contract
 * removed the special case rather than because they were forgotten:
 *
 *   `frozen` — the old atom carried an `isFrozen` field, so frozen needed a
 *              channel of its own. § 6.6 settles it the other way: `frozen
 *              atoms` is an ORDINARY LABEL, stored, filtered and displayed
 *              exactly like any other, and "no code here acts on the name". It
 *              therefore arrives as a TAG through `labels` with no case of its
 *              own — which is the whole point of § 6.6, and is what makes
 *              "filtering for frozen atoms needs no case of its own" (§ 9.5)
 *              literally true rather than aspirational.
 *
 *   `values` — per-atom scalars (charge, spin). Not among § 6.2's carried facts,
 *              so by the move-together rule they cannot be among the enumerated
 *              ones either. They are task #24's programme, which extends BOTH
 *              lists in one step; adding the channel here first would break the
 *              rule in the direction it exists to prevent.
 *
 * The "by atom index" row of § 9.5 is deliberately not a channel: an index is
 * where an atom sits, not something it carries, which is why it is the one rule
 * matched against positions and the one that crosses the numbering boundary.
 */

/**
 * A by-index expression as the user typed it (1-based), turned into the 0-based
 * one the server's rule expects.
 *
 * The named direction lives here rather than at the call site so that no caller
 * writes the shift itself — a caller passing a delta would be a caller that
 * knows which way it goes, and that knowledge is exactly what § 11.5 keeps in
 * one place.
 */
export function expressionToCode(expression) {
    return shiftExpression(expression, -1);
}

export const KIND = {
    CATEGORY: "category",   // one value per atom (element, residue) — matches on equality
    TAG:      "tag",        // a named membership set (labels)       — matches on membership
};

/**
 * The channels present on ONE atom, as `{name: {kind, value?}}`.
 *
 * `element` is the atom's element symbol; `facts` is its entry from the
 * structure's `annotations` — `{labels, residue}` (§ 6.2). A CATEGORY channel
 * carries the value it matches on; a TAG channel is membership, so it carries
 * none.
 */
export function atomChannels(element, facts) {
    const out = {};
    if (element != null && element !== "") {
        out.element = { kind: KIND.CATEGORY, value: element };
    }
    const f = facts || {};
    if (f.residue != null && f.residue !== "") {
        out.residue = { kind: KIND.CATEGORY, value: f.residue };
    }
    const labels = Array.isArray(f.labels) ? f.labels : [];
    for (const label of labels) {
        if (label) out[label] = { kind: KIND.TAG };
    }
    return out;
}

/**
 * Every filterable channel present across a whole structure, as
 * `[{name, kind}]` in a stable order: element, then residue, then the labels
 * sorted by name.
 *
 * § 9.5: "which rows are worth offering is read from the structure, not
 * hard-coded" — this is that reading. It decides WHAT TO OFFER; the four rules
 * decide HOW TO MATCH. Keeping those apart is why a new label needs no panel
 * change.
 */
export function channelKinds(elements, annotations) {
    const els = elements || [];
    const facts = annotations || [];
    let hasElement = false;
    let hasResidue = false;
    const tags = new Set();

    const count = Math.max(els.length, facts.length);
    for (let i = 0; i < count; i++) {
        const channels = atomChannels(els[i], facts[i]);
        for (const name of Object.keys(channels)) {
            if (name === "element") hasElement = true;
            else if (name === "residue") hasResidue = true;
            else tags.add(name);
        }
    }

    const out = [];
    if (hasElement) out.push({ name: "element", kind: KIND.CATEGORY });
    if (hasResidue) out.push({ name: "residue", kind: KIND.CATEGORY });
    for (const name of Array.from(tags).sort()) {
        out.push({ name: name, kind: KIND.TAG });
    }
    return out;
}


/* ══ The formula ═════════════════════════════════════════════════════════════
 *
 * § 4 makes this the module's second export, beside `mount`, because a caller
 * can want it with no viewer at all — a list of elements is enough.
 *
 * It lives here rather than in a file of its own because it has exactly this
 * file's shape: pure, dependency-free, and read by several layers above without
 * reading any of them back.
 */

/**
 * A list of element symbols as a Hill-ordered formula — `H2O`, `C6H12O6`.
 *
 * C and H first, then N, O, P, S in that order, then everything else
 * alphabetically. That is the common organic-chemistry layout, and it is what a
 * user reads in the info line; changing the order changes what they see.
 */
export function formula(elements) {
    if (!elements || !elements.length) return "—";
    const counts = {};
    for (const element of elements) {
        counts[element] = (counts[element] || 0) + 1;
    }
    const parts = [];
    const write = (e) => parts.push(counts[e] > 1 ? e + counts[e] : e);
    for (const e of ["C", "H", "N", "O", "P", "S"]) {
        if (counts[e]) { write(e); delete counts[e]; }
    }
    for (const e of Object.keys(counts).sort()) write(e);
    return parts.join("");
}
