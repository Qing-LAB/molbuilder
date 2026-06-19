/* Region-label definitions for transport-tab region labels.
 *
 * Provides a small inline reference: what does each label mean
 * scientifically?  What atoms belong in it?  What's "common
 * practice" per the published transport literature?
 *
 * Mounts into ``#selection-target-info-panel`` (the popover next
 * to the Target: dropdown in _selection_panel.html).  Toggled by
 * clicking the ⓘ button.
 *
 * The reference table covers:
 *   - L-electrode / R-electrode — canonical 2-terminal defaults
 *   - bridge — the molecule under test + non-periodic contact atoms
 *   - interface — sub-label for chemically-bonded contact atoms
 *   - <name>-electrode — the convention for arbitrary electrode names
 *
 * Citations: Brandbyge 2002 PRB 65 165401 (the canonical
 * TranSIESTA paper), Stokbro 2003 Comp Mat Sci 27 151 (Au-BDT-Au
 * geometry), Reed 2006 JACS 128 14328 (asymmetric STM junctions),
 * Solomon 2008 (interface analysis convention).
 *
 * Pinned by tests/test_region_label_definitions_js.py (L2 popover
 * mount + content tests).
 */
(function () {
    "use strict";

    /**
     * Canonical region-label definitions.  Each entry:
     *   key   — the label string the modify tab assigns
     *           (matches struct.regions keys)
     *   short — one-line summary shown in the popover header row
     *   atoms — what atoms belong in this region
     *   practice — common-practice notes from the literature
     *   citation — short cite tag (full ref in
     *              docs/protocols/region-labels.md)
     *
     * The popover renders ALL canonical entries plus a synthetic
     * entry for any non-canonical *-electrode label present in
     * the current structure.
     */
    var CANONICAL_DEFINITIONS = [
        {
            key: "L-electrode",
            short: "Left semi-infinite lead (periodic bulk).",
            atoms: (
                "The slice of LEAD METAL that SIESTA will replicate as " +
                "a semi-infinite bulk lead.  3-4 atomic layers, deep " +
                "enough that the electronic structure matches bulk at " +
                "the boundary."
            ),
            practice: (
                "Use ONLY the bulk portion (the part SIESTA will " +
                "replicate) — surface terminations / dangling-H caps " +
                "go in ``bridge``.  The lattice along the transport " +
                "direction (z) MUST match the bulk lead's periodicity " +
                "so the .TSHS self-energy matches.  Au(111): typically " +
                "3 layers × N atoms.  Pt(111): 4 layers."
            ),
            citation: "Brandbyge 2002 PRB §III"
        },
        {
            key: "R-electrode",
            short: "Right semi-infinite lead (periodic bulk).",
            atoms: (
                "Mirror of L-electrode on the opposite side of the " +
                "device."
            ),
            practice: (
                "Same metal + crystallographic orientation as L is the " +
                "canonical 2-terminal case.  Asymmetric leads (different " +
                "metals, STM tip vs. flat surface) are an advanced case " +
                "and currently warned at preflight."
            ),
            citation: "Brandbyge 2002 PRB §III"
        },
        {
            key: "bridge",
            short: "Scattering region — the molecule under test + non-periodic contacts.",
            atoms: (
                "The molecule(s) being measured PLUS any lead-side atoms " +
                "that break periodicity (surface terminations, tip atoms, " +
                "chemisorbed contact atoms)."
            ),
            practice: (
                "For Au-BDT-Au: the entire benzene-1,4-dithiol molecule " +
                "INCLUDING the two S anchors (they're chemically bonded to " +
                "the leads but still 'molecule side' for the transport " +
                "partition).  Surface layers BELOW the molecule and ABOVE " +
                "the periodic bulk go here too."
            ),
            citation: "Brandbyge 2002 PRB §III; Stokbro 2003 CMS"
        },
        {
            key: "interface",
            short: "Optional: contact atoms chemically bonded to the leads.",
            atoms: (
                "A SUB-label that flags atoms still chemically inside " +
                "``bridge`` but participating in the metal-molecule bond.  " +
                "Useful for projected-DOS and charge-transfer analysis; " +
                "does NOT change the TranSIESTA partition."
            ),
            practice: (
                "The 2 S anchors in Au-BDT-Au.  The N atoms in Au-PDA-Au " +
                "(pyridine derivatives).  Optional — don't add unless " +
                "you'll use it downstream."
            ),
            citation: "Reed 2006 JACS; Solomon 2008"
        }
    ];

    /**
     * Synthetic definition for any non-canonical *-electrode label
     * present in the current structure.  Helps the user understand
     * the convention without having to look it up elsewhere.
     */
    var CUSTOM_ELECTRODE_TEMPLATE = {
        short: "Additional electrode region (per the *-electrode convention).",
        atoms: (
            "Any region whose label ends with ``-electrode`` is treated " +
            "as a TranSIESTA lead.  The modern syntax emitter creates a " +
            "%block TS.Elec.<name> block per electrode and binds it to a " +
            "chemical potential."
        ),
        practice: (
            "Use for STM-tip leads, 3-terminal devices (gate-electrode), " +
            "or asymmetric junctions where ``L``/``R`` doesn't capture " +
            "the topology.  The label stem (before ``-electrode``) is " +
            "the SIESTA block name; pick something short and ASCII."
        ),
        citation: "molbuilder convention; see docs/protocols/region-labels.md"
    };

    /**
     * Bias-direction guidance shown at the bottom of the popover
     * (per the user request 2026-06-18: pair the label definitions
     * with the bias-direction reminder).
     */
    var BIAS_NOTE = (
        "Bias direction: ``V_left - V_right`` in volts.  POSITIVE bias " +
        "raises μ_L above μ_R; electrons flow from the higher chemical " +
        "potential to the lower (L → R for positive V), conventional " +
        "current flows R → L.  Pick the L electrode to be the more " +
        "negative reservoir in your forward-bias measurement."
    );

    /**
     * Render the popover body.  Called with the set of label
     * strings actually present in the current structure (passed
     * in so we can highlight which are in use vs. defaults).
     *
     * @param {HTMLElement} panel - the #selection-target-info-panel element
     * @param {Set<string>}  presentLabels - region labels present in struct
     */
    function renderPopover(panel, presentLabels) {
        if (!panel) return;
        if (!presentLabels) presentLabels = new Set();
        var rows = CANONICAL_DEFINITIONS.map(function (def) {
            return rowFor(def, presentLabels.has(def.key));
        });
        // Synthetic rows for any non-canonical *-electrode label
        // present in the current structure.
        var canonicalKeys = new Set(
            CANONICAL_DEFINITIONS.map(function (d) { return d.key; })
        );
        var custom = [];
        presentLabels.forEach(function (lbl) {
            if (canonicalKeys.has(lbl)) return;
            if (!isElectrodeLabel(lbl)) return;
            var def = Object.assign({}, CUSTOM_ELECTRODE_TEMPLATE, {
                key: lbl,
                short: (
                    "Custom electrode region ``" + lbl + "`` " +
                    "(matched by *-electrode convention)."
                )
            });
            custom.push(rowFor(def, true));
        });
        var html =
            "<h4 class=\"region-defs-title\">Region label reference</h4>" +
            rows.join("") +
            custom.join("") +
            "<p class=\"region-defs-bias\"><strong>Bias direction.</strong> " +
            escapeHtml(BIAS_NOTE) + "</p>" +
            "<p class=\"region-defs-doc\">" +
            "Full convention + references: " +
            "<a href=\"#\" data-region-defs-docs=\"true\">" +
            "docs/protocols/region-labels.md</a>" +
            "</p>";
        panel.innerHTML = html;
    }

    /**
     * Build the markup for one definition row.
     */
    function rowFor(def, isInUse) {
        var badge = isInUse
            ? "<span class=\"region-defs-badge in-use\">in this structure</span>"
            : "<span class=\"region-defs-badge available\">default available</span>";
        return (
            "<details class=\"region-defs-row\"" +
            (isInUse ? " open" : "") + ">" +
            "<summary>" +
            "<code class=\"region-defs-key\">" + escapeHtml(def.key) + "</code>" +
            " " + badge +
            " <span class=\"region-defs-short\">" +
            escapeHtml(def.short) + "</span>" +
            "</summary>" +
            "<dl class=\"region-defs-body\">" +
            "<dt>Atoms</dt><dd>" + escapeHtml(def.atoms) + "</dd>" +
            "<dt>Common practice</dt><dd>" + escapeHtml(def.practice) + "</dd>" +
            "<dt>Reference</dt><dd>" + escapeHtml(def.citation) + "</dd>" +
            "</dl>" +
            "</details>"
        );
    }

    /**
     * True iff a label matches the *-electrode convention.
     * Mirrors molbuilder.config.transport.is_electrode_label.
     */
    function isElectrodeLabel(label) {
        if (typeof label !== "string") return false;
        var lo = label.toLowerCase();
        return (
            lo === "electrode" ||
            lo.endsWith("-electrode") ||
            lo.endsWith("_electrode")
        );
    }

    function escapeHtml(s) {
        if (s == null) return "";
        return String(s)
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&#39;");
    }

    /**
     * Wire the ⓘ button: click toggles the popover, aria-expanded
     * stays in sync.  Caller passes a getter for the current set of
     * region labels so the popover content stays fresh when the
     * structure changes.
     *
     * @param {function():Set<string>} getPresentLabels
     */
    function init(getPresentLabels) {
        var btn = document.getElementById("selection-target-info-btn");
        var panel = document.getElementById("selection-target-info-panel");
        if (!btn || !panel) return;   // panel not in this template
        btn.addEventListener("click", function () {
            var nowHidden = !panel.hidden;
            if (!nowHidden) {
                // About to open — refresh the content.
                var labels = (typeof getPresentLabels === "function")
                    ? getPresentLabels() : new Set();
                renderPopover(panel, labels);
            }
            panel.hidden = nowHidden;
            btn.setAttribute("aria-expanded", nowHidden ? "false" : "true");
        });
        // Close on Esc.
        document.addEventListener("keydown", function (e) {
            if (e.key === "Escape" && !panel.hidden) {
                panel.hidden = true;
                btn.setAttribute("aria-expanded", "false");
            }
        });
    }

    // Public surface.
    window.molbuilder = window.molbuilder || {};
    window.molbuilder.regionLabelDefinitions = {
        CANONICAL_DEFINITIONS: CANONICAL_DEFINITIONS,
        renderPopover: renderPopover,
        isElectrodeLabel: isElectrodeLabel,
        init: init
    };
})();
