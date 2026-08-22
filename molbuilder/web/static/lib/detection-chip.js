/**
 * detection-chip.js — shared chip text + injection helper.
 *
 * Every engine tab that renders workflow-group cards (Profile /
 * Stage / Budget) and also asks the analyzer about the loaded
 * structure surfaces the same one-line chemistry summary on the
 * Profile + Budget cards.  The text is built from the
 * /api/structure/analyze response; the chip is a plain <span> with
 * class .workflow-detection-chip whose colour is auto-derived from
 * the host card's --workflow-group-accent token (see
 * lib/form-schema.css).
 *
 * Before this module existed (2026-06-13) the helpers lived inside
 * viewer.js as private closures, which meant the Transport tab —
 * which has the highest-value chip use case (Au junctions) —
 * silently rendered no chip even after analyzing.  Per web-ui-
 * coherence.md Rule 1 every UI surface that talks about open-vs-
 * closed shell reads ChemistryAnalysis.suggested_treatment from one
 * function; this module is the single chip implementation that
 * enforces that for the chip surface.
 *
 * Exports (on window.molbuilder.detectionChip):
 *   buildText(resp) → { profile, budget } — pure text helper
 *   render(resp, opts) → number of headers patched
 *
 * resp shape (from /api/structure/analyze):
 *   { n_atoms, metals: [...], suggested: { pyscf: {...}, siesta: {...} } }
 */
(function () {
    "use strict";

    var root = (typeof globalThis !== "undefined") ? globalThis
            : (typeof window !== "undefined") ? window : this;

    function buildText(resp) {
        var n_atoms = (resp && typeof resp.n_atoms === "number")
            ? resp.n_atoms : null;
        var metals = (resp && Array.isArray(resp.metals))
            ? resp.metals : [];
        // The analyzer's own verdict rides the response top-level
        // (`suggested_treatment` -- the ONE function's answer, sent
        // since the U6 close; the per-engine suggested blocks never
        // carried it, so the old read order always fell through to the
        // spin heuristic).  The heuristic stays as the last fallback
        // for older responses.
        var sug = (resp && resp.suggested) || {};
        var sugP = sug.pyscf || sug.siesta || {};
        var treatment = (resp && resp.suggested_treatment)
            || ((typeof sugP.spin === "number" && sugP.spin > 0)
                ? "open" : "closed");
        var spinNum = (typeof sugP.spin === "number") ? sugP.spin : null;

        // --- Profile line --------------------------------------- //
        // Format priority (highest first):
        //   open-d metal      → "<N> atoms · <Metals> · open-shell (2S=<n>)"
        //   noble-metal cluster→ "<N> atoms · <Metal> cluster · closed-shell singlet"
        //   pure organic      → "<N> atoms · closed-shell singlet"
        var sysLine = (n_atoms != null) ? (n_atoms + " atoms") : "";
        if (metals.length > 0) {
            var list = metals.join(", ");
            if (treatment === "open") {
                var spinHint = (spinNum != null)
                    ? " (2S=" + spinNum + ")" : "";
                sysLine = sysLine
                    ? sysLine + " · " + list + " · open-shell" + spinHint
                    : list + " · open-shell" + spinHint;
            } else {
                sysLine = sysLine
                    ? sysLine + " · " + list
                          + " cluster · closed-shell singlet"
                    : list + " · closed-shell singlet";
            }
        } else {
            var closed = (treatment === "closed");
            var tag = closed
                ? "closed-shell" + (spinNum === 0 ? " singlet" : "")
                : "open-shell"
                      + (spinNum != null ? " (2S=" + spinNum + ")" : "");
            sysLine = sysLine ? sysLine + " · " + tag : tag;
        }

        // --- Budget line ---------------------------------------- //
        // Size-aware hint.  Au-BDT-Au-class systems (≥ 150 metallic
        // atoms) get an explicit "bump the caps" nudge because the
        // cluster-context closed-shell argument doesn't help
        // convergence speed.  Pure organics under 100 atoms get a
        // "defaults are fine" nudge so the user doesn't second-guess.
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
     * Inject (or refresh) the chip inside every workflow-group
     * profile + budget header under ``opts.root`` (default: document).
     * Idempotent — re-running replaces the chip text in place.
     * Returns the number of headers patched (for tests).
     */
    function render(resp, opts) {
        opts = opts || {};
        var host = opts.root || (root.document ? root.document : null);
        if (!host || !host.querySelectorAll) return 0;
        var chips = buildText(resp);
        var sel = ".workflow-group--profile .workflow-group-header, "
                + ".workflow-group--budget .workflow-group-header";
        var headers = host.querySelectorAll(sel);
        var n = 0;
        for (var i = 0; i < headers.length; i++) {
            var header = headers[i];
            var card = header.closest(".workflow-group");
            if (!card) continue;
            var role = card.classList.contains("workflow-group--profile")
                ? "profile" : "budget";
            var text = chips[role];
            if (!text) continue;
            var chip = header.querySelector(".workflow-detection-chip");
            if (!chip) {
                chip = root.document.createElement("span");
                chip.className = "workflow-detection-chip";
                header.appendChild(chip);
            }
            chip.textContent = text;
            n++;
        }
        return n;
    }

    var api = { buildText: buildText, render: render };
    if (typeof module !== "undefined" && module.exports) {
        module.exports = api;
    }
    root.molbuilder = root.molbuilder || {};
    root.molbuilder.detectionChip = api;
})();
