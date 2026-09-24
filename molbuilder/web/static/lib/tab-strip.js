/* lib/tab-strip.js — the engine strip: ONE switcher for every tab that offers
 * two engine forms.
 *
 * Contract: docs/web/ui-contract.md § 1 (a widget on more than one page has
 *           one shared owner — the sheet is form-components.css, the
 *           behaviour is this file), docs/web/spectra.md § 5, and
 *           docs/web/overview.md § 1 (a capability used on more than one tab
 *           is a module with one door).
 * Owns:     which button is active and which panel is shown.  Nothing else —
 *           what a tab does on a change is the tab's (``onChange``).
 * Called by: /structure-optimization and /spectrum-calculation at mount.
 * Markup it reads: a strip element holding ``.tab-btn[data-tab][aria-controls]``
 *           buttons; each ``aria-controls`` names the id of the panel it shows.
 *
 * NEVER: read a tab's state, decide a default, or reach outside the panels
 *        the buttons name.
 *
 * Public API (window.molbuilder.tabStrip):
 *   mount(stripEl, {onChange(name, meta)}) -> {select(name, meta), active(), names()}
 *     select(name, meta) shows the named panel and marks its button; meta is
 *     handed to onChange unchanged (a click passes {byUser: true}); returns
 *     false when no button carries that name.
 */
(function (root) {
    "use strict";

    function mountTabStrip(stripEl, opts) {
        opts = opts || {};
        var doc = stripEl.ownerDocument || document;
        var buttons = Array.prototype.slice.call(
            stripEl.querySelectorAll(".tab-btn"));

        function panelOf(btn) {
            var id = btn.getAttribute("aria-controls");
            return id ? doc.getElementById(id) : null;
        }
        function active() {
            for (var i = 0; i < buttons.length; i++) {
                if (buttons[i].classList.contains("active")) {
                    return buttons[i].dataset.tab || null;
                }
            }
            return null;
        }
        function select(name, meta) {
            var found = false;
            buttons.forEach(function (b) {
                var on = b.dataset.tab === name;
                if (on) found = true;
                b.classList.toggle("active", on);
                b.setAttribute("aria-selected", on ? "true" : "false");
                var p = panelOf(b);
                if (p) p.hidden = !on;
            });
            if (found && typeof opts.onChange === "function") {
                opts.onChange(name, meta || {});
            }
            return found;
        }
        buttons.forEach(function (b) {
            b.addEventListener("click", function () {
                select(b.dataset.tab, { byUser: true });
            });
        });
        return {
            select: select,
            active: active,
            names: function () {
                return buttons.map(function (b) { return b.dataset.tab; });
            },
        };
    }

    root.molbuilder = root.molbuilder || {};
    root.molbuilder.tabStrip = { mount: mountTabStrip };
})(typeof window !== "undefined" ? window : this);
