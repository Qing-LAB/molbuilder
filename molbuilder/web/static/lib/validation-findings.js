/* Validation findings — THE presentation channel for scientific findings.
 *
 * MODULE: lib/validation-findings (classic IIFE; publishes
 * ``molbuilder.validationFindings``).  Takes the ``issues[]`` array any
 * validating endpoint returns and puts every entry on screen.
 *
 * USED BY: the structure-optimization tab (SIESTA + PySCF panels), the
 * transport tab, the spectra tab/inspector.  Nothing else renders a finding.
 *
 * WHY THIS EXISTS (contract R2, docs/science/validation.md § 4.1).  There were
 * FOUR implementations of this one job — one per tab plus the per-card panels
 * form-schema.js creates — and they had drifted into three different row
 * vocabularies, two empty-state behaviours, two bucket types and two orderings.
 * The drift was not cosmetic; each copy had lost findings:
 *
 *   * ALL THREE dropped an issue whose ``workflow_group`` named a card the form
 *     schema had not rendered: they iterated the card PANELS and wrote only
 *     buckets that matched one, so a finding tagged with an unrendered role
 *     appeared nowhere at all.
 *   * spectra additionally dropped any issue whose severity was not exactly
 *     "error" / "warn" / "info" from its residual list, and re-ordered the rest.
 *   * transport returned early when the residual markup was missing, which also
 *     skipped clearing the per-card panels, so stale findings survived a
 *     re-Generate; its buckets were a bare object, so a group literally named
 *     "constructor" threw.
 *   * spectra had no null guard on its panel and threw AFTER clearing cards.
 *
 * So the contract's rules are the module's behaviour, with no knob to opt out:
 *
 *   R3 nothing is dropped.  An unroutable finding (no group, or a group with no
 *      rendered card) goes to the residual panel.  ``render`` returns the
 *      counts it actually wrote, and ``total`` always equals ``issues.length``.
 *   R4 severity means one thing.  One row vocabulary
 *      (``li.issue-item[data-severity]``, styled once in form-components.css);
 *      an unrecognised or missing severity renders as ``info`` — never dropped.
 *      Server order is preserved (validate() emits geometry, then config, then
 *      engine checks — a documented deterministic order).
 *
 * Surface:
 *
 *   const summary = molbuilder.validationFindings.render(issues, {
 *       panel:     <element>,   // the row parent AND visibility owner
 *       formScope: <element>,   // optional: searched for .card-issues panels
 *       emptyText: "No issues yet.",  // optional: shown instead of hiding
 *   });
 *   // summary = {total, residual, byGroup: {<role>: n}, counts: {error, warn, info}}
 *
 *   molbuilder.validationFindings.clear({panel, formScope});
 *
 * ``panel`` and ``formScope`` are ELEMENTS, not ids: the spectra inspector
 * mounts inside a subtree and resolves its own ids against that root, so this
 * module must never reach for ``document`` itself.
 */
(function (root) {
    "use strict";

    var SEVERITIES = ["error", "warn", "info"];

    function _severityOf(issue) {
        // R4: coerce, never drop.  An unknown severity is still a finding.
        var s = issue && issue.severity;
        return (SEVERITIES.indexOf(s) !== -1) ? s : "info";
    }

    // One row shape for every surface (form-components.css owns the styling).
    function _row(doc, issue) {
        var li = doc.createElement("li");
        li.className = "issue-item";
        li.setAttribute("data-severity", _severityOf(issue));
        var msg = doc.createElement("span");
        msg.className = "issue-msg";
        msg.textContent = (issue && issue.message) || "";
        li.appendChild(msg);
        // ``where`` is the STABLE identifier (contract R1) -- shown so a user
        // can quote it, and never parsed from the prose message.
        if (issue && issue.where) {
            var where = doc.createElement("span");
            where.className = "issue-where";
            where.textContent = issue.where;
            li.appendChild(where);
        }
        return li;
    }

    function _cardPanels(formScope) {
        var out = {};
        if (!formScope || !formScope.querySelectorAll) return out;
        var found = formScope.querySelectorAll(".card-issues[data-workflow-group]");
        for (var i = 0; i < found.length; i++) {
            out[String(found[i].getAttribute("data-workflow-group"))] = found[i];
        }
        return out;
    }

    function _emptyPanel(el) {
        if (!el) return;
        while (el.firstChild) el.removeChild(el.firstChild);
    }

    function clear(opts) {
        opts = opts || {};
        var panels = _cardPanels(opts.formScope);
        // Clear the CARDS FIRST and unconditionally: transport's copy returned
        // early on missing residual markup and left stale card findings up.
        Object.keys(panels).forEach(function (role) {
            _emptyPanel(panels[role]);
            panels[role].hidden = true;
        });
        if (opts.panel) {
            _emptyPanel(opts.panel);
            opts.panel.hidden = true;
        }
    }

    function render(issues, opts) {
        opts = opts || {};
        var panel = opts.panel || null;
        var list = (issues || []).filter(Boolean);
        var doc = (panel && panel.ownerDocument) || root.document;
        var summary = {
            total: list.length, residual: 0, byGroup: {},
            counts: { error: 0, warn: 0, info: 0 },
        };

        clear({ panel: panel, formScope: opts.formScope });

        var panels = _cardPanels(opts.formScope);
        if (!list.length) {
            if (panel && opts.emptyText) {
                var empty = doc.createElement("li");
                empty.className = "issues-empty";
                empty.textContent = opts.emptyText;
                panel.appendChild(empty);
                panel.hidden = false;
            }
            return summary;
        }

        // ONE pass over the FINDINGS (not over the panels): that inversion is
        // the bug fix -- iterating panels is what silently dropped a finding
        // whose group had no rendered card.
        list.forEach(function (issue) {
            summary.counts[_severityOf(issue)] += 1;
            var group = issue && issue.workflow_group;
            var target = (group && panels[String(group)]) || null;
            if (target) {
                target.appendChild(_row(doc, issue));
                target.hidden = false;
                summary.byGroup[String(group)] =
                    (summary.byGroup[String(group)] || 0) + 1;
                return;
            }
            if (!panel) return;      // nowhere to put it; counted, not silent
            panel.appendChild(_row(doc, issue));
            summary.residual += 1;
        });
        if (panel) {
            if (summary.residual) {
                panel.hidden = false;
            } else if (opts.emptyText) {
                // Every finding landed on a card: say so rather than leaving a
                // blank region (the behaviour the spectra panel wanted).
                var note = doc.createElement("li");
                note.className = "issues-empty";
                note.textContent = opts.emptyText;
                panel.appendChild(note);
                panel.hidden = false;
            } else {
                panel.hidden = true;
            }
        }
        return summary;
    }

    root.molbuilder = root.molbuilder || {};
    root.molbuilder.validationFindings = { render: render, clear: clear };
})(typeof window !== "undefined" ? window : this);
