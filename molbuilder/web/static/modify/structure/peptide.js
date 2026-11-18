/* Peptide-builder generator panel — Molbuilder-tab Source.
 *
 * The "Generate peptide" panel: the one panel skeleton (`panel-deps.js`,
 * `panel`) with its words; the sequence goes as typed to
 * `/api/build/molecule` (kind "peptide"), whose builder reads one-letter
 * codes and bracketed modified residues ("AR[SEP]C") by the builders' one
 * grammar and refuses what it cannot, in its words (`engines/builders.md`
 * § 1).
 *
 * Design ref: docs/web/tabs.md § 2 (Creating a structure — the in-gate).
 */

(function (root, mod) {
    "use strict";

    root.molbuilder.panelDeps.panel(root, mod, {
        kind: "peptide", label: "the peptide", global: "structurePeptide",
        /** A peptide from the sequence typed, recorded as its sequence. */
        generate: function (slot) {
            return function (sequence) {
                return slot.generateText(sequence, "peptide",
                    "Enter a peptide sequence first (one-letter codes).",
                    "Building the peptide\u2026", "sequence");
            };
        },
    });
})(typeof window !== "undefined" ? window : globalThis,
   typeof module !== "undefined" ? module : null);
