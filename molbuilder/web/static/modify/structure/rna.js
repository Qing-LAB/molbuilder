/* RNA-builder generator panel — Molbuilder-tab Source.
 *
 * The RNA panel: the one nucleic-acid panel (`panel-deps.js`,
 * `generateNucleic`) in the one panel skeleton (`panel`), as RNA -- its
 * words.  ``build_rna`` builds a single strand.  Nothing of the builder's
 * rules is checked here: the text goes as typed and the builder's refusal
 * is the one shown (`engines/builders.md` § 1).
 *
 * Design ref: docs/web/tabs.md § 2 (Creating a structure — the in-gate).
 */

(function (root, mod) {
    "use strict";

    root.molbuilder.panelDeps.panel(root, mod, {
        kind: "rna", label: "RNA", global: "structureRna",
        /** RNA from what was typed, in the form and on the backend asked
         *  (``opts.form``, ``opts.backend``). */
        generate: function (slot) {
            return function (sequence, opts) {
                return slot.generateNucleic(
                    "rna", sequence, opts,
                    "Enter an RNA sequence first (ACGU).",
                    "Building the RNA\u2026");
            };
        },
    });
})(typeof window !== "undefined" ? window : globalThis,
   typeof module !== "undefined" ? module : null);
