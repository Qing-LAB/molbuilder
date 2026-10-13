/* lib/tab-strip.js — THE tab strip: one implementation of the ARIA tabs
 * pattern for every strip of tabs in the app.
 *
 * Contract: docs/web/ui-contract.md § 4.1 (a tab strip follows the ARIA tabs
 *           pattern — one tab stop, the arrow keys move the selection, each
 *           tab and its panel paired by id) and § 1 (a widget on more than one
 *           page has one owner: this file its behaviour; each strip's sheet
 *           its look, keyed on `[aria-selected="true"]`).
 * Owns:     which tab is selected, which panel is shown, the keyboard, and
 *           the ids that pair them.  Nothing else — what a page does on a
 *           change is the page's (``onChange``).
 * Called by: the engine strips (Structure optimization, Spectrum, Transport),
 *           Transport's rung strip, Modify's source and operation strips,
 *           the spectrum's mode tabs, the transport report's tabs, Task
 *           setup's group tabs, the sidebar's run-history view.
 * Markup it reads: a strip holding ``[role="tab"]`` elements, each named by
 *           ``data-tab`` and, when it shows a panel, ``aria-controls`` naming
 *           the panel's id.  A tab with no panel only says which is chosen
 *           (``onChange`` acts on it).
 *
 * NEVER: read a page's state, decide a default, or reach outside the panels
 *        the tabs name.
 *
 * Public API (window.molbuilder.tabStrip):
 *   mount(stripEl, {onChange(name, meta), panelOf(tab), listeners}) -> {select(name, meta), active(), names()}
 *     panelOf, for a strip built before it is in the page: the panel a tab
 *     shows, which ``aria-controls`` cannot find by id there.
 *     listeners, for a strip inside a module that is disposed: its owner's
 *     listener scope (``inspectorLifecycle.listeners()``), which the strip
 *     registers through, so the owner's dispose removes them too.
 *     select(name, meta) selects the named tab and shows its panel; meta is
 *     handed to onChange unchanged (a click or a key passes {byUser: true});
 *     returns false when no tab carries that name.
 */
(function (root) {
    "use strict";

    var _seq = 0;

    function mountTabStrip(stripEl, opts) {
        opts = opts || {};
        var doc = stripEl.ownerDocument || document;
        var tabs = Array.prototype.slice.call(
            stripEl.querySelectorAll('[role="tab"]'));
        var strip = ++_seq;
        function on(target, event, handler) {
            if (opts.listeners) opts.listeners.on(target, event, handler);
            else target.addEventListener(event, handler);
        }

        function panelOf(tab) {
            if (typeof opts.panelOf === "function") return opts.panelOf(tab);
            var id = tab.getAttribute("aria-controls");
            return id ? doc.getElementById(id) : null;
        }
        // EACH TAB AND ITS PANEL PAIRED BY ID: the panel is labelled by its
        // tab, so a screen reader names the panel it lands in.
        tabs.forEach(function (t, i) {
            if (!t.id) t.id = "tab-strip-" + strip + "-" + i;
            var p = panelOf(t);
            if (p) {
                p.setAttribute("role", "tabpanel");
                p.setAttribute("aria-labelledby", t.id);
            }
        });

        function active() {
            for (var i = 0; i < tabs.length; i++) {
                if (tabs[i].getAttribute("aria-selected") === "true") {
                    return tabs[i].dataset.tab || null;
                }
            }
            return null;
        }
        function select(name, meta) {
            var found = tabs.some(function (t) { return t.dataset.tab === name; });
            if (!found) return false;
            tabs.forEach(function (t) {
                var on = t.dataset.tab === name;
                t.setAttribute("aria-selected", on ? "true" : "false");
                // ONE TAB STOP: the selected tab; the arrows reach the rest.
                t.tabIndex = on ? 0 : -1;
                var p = panelOf(t);
                if (p) p.hidden = !on;
            });
            if (typeof opts.onChange === "function") {
                opts.onChange(name, meta || {});
            }
            return true;
        }
        // THE TABS A KEY CAN REACH: shown and enabled.
        function reachable() {
            return tabs.filter(function (t) { return !t.hidden && !t.disabled; });
        }
        tabs.forEach(function (t) {
            on(t, "click", function () {
                select(t.dataset.tab, { byUser: true });
            });
            on(t, "keydown", function (ev) {
                var on = reachable();
                var i = on.indexOf(t);
                if (i < 0 || !on.length) return;
                var to = ev.key === "ArrowRight" ? on[(i + 1) % on.length]
                    : ev.key === "ArrowLeft" ? on[(i - 1 + on.length) % on.length]
                    : ev.key === "Home" ? on[0]
                    : ev.key === "End" ? on[on.length - 1] : null;
                if (!to) return;
                ev.preventDefault();
                select(to.dataset.tab, { byUser: true });
                to.focus();
            });
        });
        // THE STATE THE MARKUP ARRIVED WITH, made whole: the selected tab
        // (else the first reachable) is the one tab stop, its panel shown.
        var first = tabs.filter(function (t) {
            return t.getAttribute("aria-selected") === "true";
        })[0] || reachable()[0];
        if (first) {
            tabs.forEach(function (t) {
                var on = t === first;
                t.setAttribute("aria-selected", on ? "true" : "false");
                t.tabIndex = on ? 0 : -1;
                var p = panelOf(t);
                if (p) p.hidden = !on;
            });
        }
        return {
            select: select,
            active: active,
            names: function () {
                return tabs.map(function (t) { return t.dataset.tab; });
            },
        };
    }

    root.molbuilder = root.molbuilder || {};
    root.molbuilder.tabStrip = { mount: mountTabStrip };
})(typeof window !== "undefined" ? window : this);
