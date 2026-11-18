/* SMILES generator panel — Molbuilder-tab Source.
 *
 * The "Generate from SMILES" panel: the one panel skeleton
 * (`panel-deps.js`, `panel`) with its words; the text goes as typed to
 * `/api/build/molecule` (kind "smiles"), whose answer -- the structure, the
 * engine that built it, what the build says of it -- reaches the page's
 * load gate and the panel's status and findings.  RDKit builds it on the
 * server, OpenBabel where RDKit cannot (`engines/builders.md` § 6).
 *
 * Design ref: docs/web/tabs.md § 2 (Creating a structure — the in-gate).
 */

(function (root, mod) {
    "use strict";

    root.molbuilder.panelDeps.panel(root, mod, {
        kind: "smiles", label: "the molecule", global: "structureSmiles",
        /** A structure from the SMILES typed. */
        generate: function (slot) {
            return function (smiles) {
                return slot.generateText(smiles, "smiles",
                    "Enter a SMILES string first.",
                    "Generating 3-D structure from SMILES\u2026");
            };
        },
    });
})(typeof window !== "undefined" ? window : globalThis,
   typeof module !== "undefined" ? module : null);
