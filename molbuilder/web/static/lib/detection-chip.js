/**
 * detection-chip.js — each form's one-line chemistry summary.
 *
 * The chip on a form's Profile card says what that FORM's calculation will
 * carry -- its own charge and spin, as the electronic-state class resolved
 * them for exactly what the form says (/api/structure/analyze's
 * ``state.<engine>``, docs/science/chemistry-correctness.md § 2a).  One form,
 * one chip, from that form's own answer: a page with a SIESTA and a PySCF form
 * shows two chips that may differ, because the two forms may say different
 * things.  The Budget card's chip is a size hint.
 *
 * (Until 2026-09-28 the chip read one analyzer verdict for the whole page,
 * judged at charge 0, and never read the form's charge.)
 *
 * Exports (on window.molbuilder.detectionChip):
 *   buildText(resp, engine) → { profile, budget } — pure text helper
 *   render(resp, hosts) → number of headers patched (null resp: removed)
 */
(function () {
    "use strict";

    var root = (typeof globalThis !== "undefined") ? globalThis
            : (typeof window !== "undefined") ? window : this;

    function _spin(st) {
        var t = st.spin_treatment.value, c = st.unpaired_electrons.value;
        if (t === "restricted") return "closed shell";
        return t + (c === "free" ? ", moment free" : ", 2S = " + c);
    }

    function buildText(resp, engine) {
        var n_atoms = (resp && typeof resp.n_atoms === "number")
            ? resp.n_atoms : null;
        var metals = (resp && Array.isArray(resp.metals)) ? resp.metals : [];
        var st = resp && resp.state && resp.state[engine];

        // --- Profile line: this form's own state ------------------------- //
        var parts = [];
        if (n_atoms != null) parts.push(n_atoms + " atoms");
        if (metals.length) parts.push(metals.join(", "));
        if (st) {
            parts.push(_spin(st));
            var q = st.net_charge.value;
            if (q) parts.push("charge " + (q > 0 ? "+" : "") + q);
        }
        var sysLine = parts.join(" · ");

        // --- Budget line ------------------------------------------------ //
        // Size-aware hint.  Au-BDT-Au-class systems (≥ 150 metallic atoms)
        // get an explicit "bump the caps" nudge; pure organics under 100
        // atoms a "defaults are fine" so nobody second-guesses them.
        var budgetLine = (n_atoms != null) ? (n_atoms + " atoms") : "";
        if (n_atoms != null) {
            if (n_atoms >= 150 && metals.length > 0) {
                budgetLine += " · bump relax_steps + max_scf_iter "
                            + "for large metallic systems";
            } else if (n_atoms >= 100) {
                budgetLine += " · consider higher caps for large systems";
            } else if (n_atoms >= 50 && metals.length > 0) {
                budgetLine += " · metallic system — watch SCF "
                            + "convergence; bump caps if needed";
            } else {
                budgetLine += " · defaults are fine";
            }
        }
        return { profile: sysLine, budget: budgetLine };
    }

    /**
     * Inject (or refresh) the chip in every Profile and Budget card header
     * inside each engine's form host (``hosts``: {engine: element}).
     * Idempotent — re-running replaces the chip text in place; a null
     * ``resp`` removes the chips.  Returns the number of headers patched.
     */
    function render(resp, hosts) {
        var n = 0;
        var sel = ".workflow-group--profile .workflow-group-header, "
                + ".workflow-group--budget .workflow-group-header";
        Object.keys(hosts || {}).forEach(function (engine) {
            var host = hosts[engine];
            if (!host || !host.querySelectorAll) return;
            var chips = buildText(resp, engine);
            var headers = host.querySelectorAll(sel);
            for (var i = 0; i < headers.length; i++) {
                var header = headers[i];
                var card = header.closest(".workflow-group");
                if (!card) continue;
                var role = card.classList.contains("workflow-group--profile")
                    ? "profile" : "budget";
                var text = chips[role];
                var chip = header.querySelector(".workflow-detection-chip");
                if (!text) {
                    // No answer (no structure, or the server could not
                    // give one): the chip goes rather than keep the last.
                    if (chip) chip.remove();
                    continue;
                }
                if (!chip) {
                    chip = root.document.createElement("span");
                    chip.className = "workflow-detection-chip";
                    header.appendChild(chip);
                }
                chip.textContent = text;
                n++;
            }
        });
        return n;
    }

    var api = { buildText: buildText, render: render };
    if (typeof module !== "undefined" && module.exports) {
        module.exports = api;
    }
    root.molbuilder = root.molbuilder || {};
    root.molbuilder.detectionChip = api;
})();
