/* Per-atom channel model — the pure L1 layer of the unified selection
 * (atom-annotations.md § 5).  The JS mirror of Python
 * ``Structure.channels()``: given one normalized atom, produce its unified
 * channels; aggregate the filterable channels across a set of atoms.
 *
 * ──────────────────────────── L1 CONTRACT ────────────────────────────
 * WHAT L1 IS.  The LOW-LEVEL PRESENTATION API for atom metadata.  It OWNS the
 *   channel taxonomy (category/tag/flag/value), the mapping "raw atom fields →
 *   its channels", and the stable enumeration ORDER.  (Mostly presentation-
 *   support; the enumeration also feeds the filter, which becomes a server
 *   query — so it borders on data-model, but the taxonomy + order are its
 *   presentation contract.)
 * PRIMITIVES, NOT FINISHED PRESENTATION.  atomChannels/channelKinds return the
 *   channel MODEL (values + kinds + order).  They do NOT render dropdowns,
 *   columns, colours, or labels — higher presentation layers (L2 store /
 *   L3 panel + viewer-adapter) COMPOSE the model into UI.
 * HIGHER LAYERS BUILD ON THIS.  L2/L3 MUST use this taxonomy + order and never
 *   re-derive them (e.g. no re-implementing the regions-vs-frozen split).  The
 *   panel's FROZEN_TAG_LABEL is drift-guarded against FROZEN_CHANNEL here.
 * GUARANTEES (bound by tests/test_atom_channels_js.py):
 *   kinds are category/tag/flag/value; channelKinds order is element, residue,
 *   sorted tags, frozen, sorted values; atomChannels is null-safe + accepts
 *   both wire and normalized atom shapes.
 * ──────────────────────────────────────────────────────────────────────
 *
 * PURE by contract: no DOM, no store, no HTTP, no molbuilder deps.  A native ES
 * module (private submodule of the MolView module, frontend-module-architecture.md
 * §4) that ALSO publishes the transitional browser global
 * (``window.molbuilder.atomChannelModel``, §3.2) so still-classic consumers
 * (selection store / panel) keep reading it until they convert.
 */
"use strict";

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
