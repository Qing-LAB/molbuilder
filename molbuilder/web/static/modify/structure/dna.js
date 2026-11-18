/* DNA-builder generator panel — Molbuilder-tab Source.
 *
 * The DNA panel: the one nucleic-acid panel (`panel-deps.js`,
 * `generateNucleic`) in the one panel skeleton (`panel`), as DNA -- its
 * words, and its own field, clash relief.  ``build_dna`` builds a strand or
 * a duplex (``ds,ATGC``, or two comma-separated strands).  Nothing of the
 * builder's rules is checked here: the text goes as typed and the builder's
 * refusal is the one shown (`engines/builders.md` § 1).
 *
 * Design ref: docs/web/tabs.md § 2 (Creating a structure — the in-gate).
 */

(function (root, mod) {
    "use strict";

    root.molbuilder.panelDeps.panel(root, mod, {
        kind: "dna", label: "DNA", global: "structureDna",
        /** DNA from what was typed -- a strand, ``ds,<strand>``, or two
         *  comma-separated strands -- in the form and on the backend asked
         *  (``opts.form``, ``opts.backend``), with clash relief when
         *  ``opts.relaxClashes``. */
        generate: function (slot) {
            return function (sequence, opts) {
                opts = opts || {};
                return slot.generateNucleic(
                    "dna", sequence, opts,
                    "Enter a DNA sequence first (ACGT).",
                    "Building the DNA\u2026",
                    { relax_clashes: !!opts.relaxClashes });
            };
        },
    });
})(typeof window !== "undefined" ? window : globalThis,
   typeof module !== "undefined" ? module : null);
