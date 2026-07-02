/* Per-atom channel model — the pure L1 layer of the unified selection
 * (atom-annotations.md § 5).  The JS mirror of Python
 * ``Structure.channels()``: given one normalized atom, produce its unified
 * channels; aggregate the filterable channels across a set of atoms.
 *
 * PURE by contract: no DOM, no store, no HTTP, no molbuilder deps.  The store
 * (L2), panel + viewer-adapter (L3) build on it; it builds on nothing.  This is
 * what keeps "filter by any channel" from being bolted onto the store.
 *
 * Loadable both as a browser global (``window.molbuilder.atomChannelModel``)
 * and as a Node module (``module.exports``) so the pure logic is unit-tested
 * under Node without a browser.
 */
(function (root) {
    "use strict";

    // Channel kinds mirror Python § 2 + the two reserved categories the UI
    // always offers:
    //   category — one value per atom (element, residue); filter = equals
    //   tag      — named membership set (regions);         filter = in-set
    //   flag     — boolean subset (frozen);                filter = is-set
    //   value    — per-atom scalar (charge, spin, …);      filter = predicate
    var KIND = { CATEGORY: "category", TAG: "tag", FLAG: "flag", VALUE: "value" };
    var FROZEN_CHANNEL = "frozen";

    /**
     * The channels on ONE atom -> ``{name: {kind, value?}}``.  Accepts either
     * the normalized store shape (``labels``/``isFrozen``/``residueName``) or
     * the raw server wire shape (``regions``/``is_frozen``/``residue_name``);
     * ``values`` is an optional ``{name: scalar}`` map for value channels.
     */
    function atomChannels(atom) {
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
    function channelKinds(atoms) {
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

    var api = {
        atomChannels: atomChannels,
        channelKinds: channelKinds,
        KIND: KIND,
        FROZEN_CHANNEL: FROZEN_CHANNEL,
    };

    // Browser global.
    root.molbuilder = root.molbuilder || {};
    root.molbuilder.atomChannelModel = api;
    // Node (tests).
    if (typeof module !== "undefined" && module.exports) module.exports = api;

})(typeof window !== "undefined" ? window
   : (typeof globalThis !== "undefined" ? globalThis : this));
