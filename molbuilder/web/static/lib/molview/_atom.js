/* MolView — what an atom IS to everything above: how it is numbered, and what it can be
 * filtered by.  Two pure, dependency-free leaves; nothing here touches the DOM, a store,
 * the network or the drawing.
 *
 * Contract: docs/web/molview.md § 11.5 (ONE atom-numbering translation, in one place —
 * "MolView never writes a bare +1 of its own anywhere") and § 9.5 / § 6.2 (the channels a
 * filter row can match: element, labels, residue — the same list the structure carries,
 * which is why filtering needs no case per property).
 *
 * Both halves are the bottom of the import graph: every layer above reads them and none of
 * them reads anything back.  That is why they are one file — same layer, same subject, and
 * § 11.5's "one place" is satisfied by one place whatever it is called.
 *
 * Assembled 2026-07-30 from _atom-index.js + _atom-channels.js.  Bodies unchanged.
 */
"use strict";


/* ── Numbering: 0-based in code, 1-based on screen (§ 11.5) — was _atom-index.js  */

export function toDisplay(i)   { return i + 1; }   // internal -> user-facing
export function fromDisplay(i) { return i - 1; }   // user-facing -> internal

/**
 * Shift every integer / range bound in a by-index expression by ``delta``.
 * "1-4, 6, 10-11"  --shiftExpression(-1)-->  "0-3, 5, 9-10".
 * Used to translate the user's 1-based filter input into the 0-based
 * expression the server ``by_index_range`` rule expects.  Preserves token
 * order; tolerates whitespace; leaves unrecognised tokens untouched (the
 * server validates).
 */
export function shiftExpression(expr, delta) {
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

export const atomIndexModel = { toDisplay, fromDisplay, shiftExpression };

// ── Transitional global (removed once every consumer imports this module) ──
if (typeof window !== "undefined") {
    window.molbuilder = window.molbuilder || {};
    window.molbuilder.atomIndexModel = atomIndexModel;
}

/* ── Channels: what a filter row can match (§ 9.5) — was _atom-channels.js ───── */

// Channel kinds mirror Python § 2 + the two reserved categories the UI
// always offers:
//   category — one value per atom (element, residue); filter = equals
//   tag      — named membership set (regions);         filter = in-set
//   flag     — boolean subset (frozen);                filter = is-set
//   value    — per-atom scalar (charge, spin, …);      filter = predicate
export const KIND = { CATEGORY: "category", TAG: "tag", FLAG: "flag", VALUE: "value" };
export const FROZEN_CHANNEL = "frozen";

/**
 * The channels on ONE atom -> ``{name: {kind, value?}}``.  Accepts either
 * the normalized store shape (``labels``/``isFrozen``/``residueName``) or
 * the raw server wire shape (``regions``/``is_frozen``/``residue_name``);
 * ``values`` is an optional ``{name: scalar}`` map for value channels.
 */
export function atomChannels(atom) {
    var out = {};
    if (atom == null) return out;
    if (atom.element != null) {
        out.element = { kind: KIND.CATEGORY, value: atom.element };
    }
    var res = atom.residueName != null ? atom.residueName : atom.residue_name;
    if (res != null && res !== "") {
        out.residue = { kind: KIND.CATEGORY, value: res };
    }
    var labels = Array.isArray(atom.labels) ? atom.labels
               : (Array.isArray(atom.regions) ? atom.regions : []);
    for (var i = 0; i < labels.length; i++) {
        if (labels[i]) out[labels[i]] = { kind: KIND.TAG };
    }
    var frozen = atom.isFrozen !== undefined ? atom.isFrozen : atom.is_frozen;
    if (frozen) out[FROZEN_CHANNEL] = { kind: KIND.FLAG };
    var vals = atom.values;
    if (vals && typeof vals === "object") {
        var names = Object.keys(vals);
        for (var j = 0; j < names.length; j++) {
            out[names[j]] = { kind: KIND.VALUE, value: vals[names[j]] };
        }
    }
    return out;
}

/**
 * Every filterable channel present across ``atoms`` -> ``[{name, kind}]``,
 * in a STABLE order: element, residue, then tags (sorted), then frozen,
 * then value channels (sorted).  This is what the filter UI enumerates
 * (replacing the regions-vs-frozen special-case).
 */
export function channelKinds(atoms) {
    var tags = {}, values = {};
    var hasElement = false, hasResidue = false, hasFrozen = false;
    atoms = atoms || [];
    for (var i = 0; i < atoms.length; i++) {
        var ch = atomChannels(atoms[i]);
        var names = Object.keys(ch);
        for (var j = 0; j < names.length; j++) {
            var name = names[j], kind = ch[name].kind;
            if (name === "element") hasElement = true;
            else if (name === "residue") hasResidue = true;
            else if (name === FROZEN_CHANNEL) hasFrozen = true;
            else if (kind === KIND.VALUE) values[name] = true;
            else tags[name] = true;
        }
    }
    var out = [];
    if (hasElement) out.push({ name: "element", kind: KIND.CATEGORY });
    if (hasResidue) out.push({ name: "residue", kind: KIND.CATEGORY });
    Object.keys(tags).sort().forEach(function (n) {
        out.push({ name: n, kind: KIND.TAG });
    });
    if (hasFrozen) out.push({ name: FROZEN_CHANNEL, kind: KIND.FLAG });
    Object.keys(values).sort().forEach(function (n) {
        out.push({ name: n, kind: KIND.VALUE });
    });
    return out;
}

export const atomChannelModel = { atomChannels, channelKinds, KIND, FROZEN_CHANNEL };

// ── Transitional global (removed once every consumer imports this module) ──
if (typeof window !== "undefined") {
    window.molbuilder = window.molbuilder || {};
    window.molbuilder.atomChannelModel = atomChannelModel;
}
