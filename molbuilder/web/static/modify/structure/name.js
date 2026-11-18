/* Generate-from-name panel — Molbuilder-tab Source.
 *
 * The "Generate from name" panel: the one panel skeleton (`panel-deps.js`,
 * `panel`) with its words; the name goes as typed to `/api/build/molecule`
 * (kind "name"), which looks it up in PubChem and builds its SMILES
 * (`engines/builders.md` § 6).
 *
 * Design ref: docs/web/tabs.md § 2 (Creating a structure — the in-gate).
 */

(function (root, mod) {
    "use strict";

    root.molbuilder.panelDeps.panel(root, mod, {
        kind: "name", label: "the molecule", global: "structureName",
        /** A structure from the name typed (IUPAC, common or trade). */
        generate: function (slot) {
            return function (name) {
                return slot.generateText(name, "name", "Enter a name first.",
                    "Generating the structure from its name\u2026");
            };
        },
    });
})(typeof window !== "undefined" ? window : globalThis,
   typeof module !== "undefined" ? module : null);
