/* lib/one-stop-list.js — THE list of choices: one tab stop, the arrow keys.
 *
 * Contract: docs/web/ui-contract.md § 4.1 (every action a click performs has
 *           a keyboard path; a list of choices is one tab stop, the arrows
 *           and Home / End moving between its items, Enter doing what a
 *           click does) and § 1 (a widget on more than one page has one
 *           owner).
 * Owns:     which item holds the tab stop, the keys that move it, Enter /
 *           Space on an item, and the keys' place across a redraw.  Nothing
 *           else -- what choosing an item does is the page's own click
 *           handler, run by Enter as by a click, so a mouse and a keyboard
 *           reach one action.  An item is one the page shows: hidden, or not
 *           drawn (a filter's ``display: none``), it is passed over.  A
 *           control inside an item -- a row's menu button, a state's Restore
 *           -- is a stop only while its item holds the list's: Tab from the
 *           item reaches it, and the list costs one stop.
 * Called by: the projects sidebar's rows, the spectrum's modes table, the run
 *           history's states.
 * Markup it reads: a list element holding its items (``opts.items``, a
 *           selector); the page gives them their roles and their state.
 *
 * NEVER: read a page's state, choose an item, or act on a key pressed on a
 *        control inside an item (a row's own menu button keeps its keys).
 *
 * Public API (window.molbuilder.oneStopList):
 *   mount(listEl, {items, current, commit, listeners}) -> {settle(), redraw(draw)}
 *     items    the selector of the list's items;
 *     current  the selector of the item the stop rests on after a redraw --
 *              the chosen one -- else the first;
 *     commit   for a list where choosing twice commits (the sidebar's file):
 *              Enter on the current item runs it, instead of a click.
 *     listeners, for a list inside a module that is disposed: its owner's
 *              listener scope (``inspectorLifecycle.listeners()``), as
 *              `lib/tab-strip.js` takes it.
 *     settle() after the page redraws the items, or shows or hides some:
 *              the stop on the item the keys are on, else the current
 *              item, else the first.  One listener on the list serves every
 *              redraw.
 *     redraw(draw) runs the page's ``draw``, which replaces the items, and
 *              keeps the keys where they were: had they an item, they are
 *              on the item now carrying its id, else on the stop -- a list
 *              drawn again under them (a run followed, a folder opened)
 *              leaves them on the page's body only when it holds no item.
 */
(function (root) {
    "use strict";

    function mountOneStopList(listEl, opts) {
        opts = opts || {};
        var itemSel = opts.items;
        function on(target, event, handler) {
            if (opts.listeners) opts.listeners.on(target, event, handler);
            else target.addEventListener(event, handler);
        }

        // SHOWN: not hidden, and not put out of view itself -- a row a
        // filter hides is ``display: none`` of its own; a row inside a shut
        // panel is not, and keeps the list's stop for when it is shown.
        function shown(el) {
            if (el.hidden) return false;
            var view = listEl.ownerDocument.defaultView;
            return !view || view.getComputedStyle(el).display !== "none";
        }
        function items() {
            return Array.prototype.slice.call(listEl.querySelectorAll(itemSel))
                .filter(shown);
        }
        var CONTROLS = "a[href], button, input, select, textarea, [tabindex]";
        function stopOn(item) {
            Array.prototype.slice.call(listEl.querySelectorAll(itemSel))
                .forEach(function (el) {
                    var stop = el === item;
                    el.tabIndex = stop ? 0 : -1;
                    Array.prototype.slice.call(el.querySelectorAll(CONTROLS))
                        .forEach(function (c) { c.tabIndex = stop ? 0 : -1; });
                });
        }
        function stopItem() {
            var all = items();
            var cur = opts.current ? listEl.querySelector(opts.current) : null;
            return cur && all.indexOf(cur) !== -1 ? cur : all[0] || null;
        }
        // ...THE STOP FOLLOWS FOCUS: an item the keys are on keeps it.
        function settle() {
            var had = listEl.ownerDocument.activeElement;
            var item = had && listEl.contains(had) && had.closest
                ? had.closest(itemSel) : null;
            stopOn(item && shown(item) ? item : stopItem());
        }
        function redraw(draw) {
            var doc = listEl.ownerDocument;
            var had = doc.activeElement;
            var item = had && listEl.contains(had) && had.closest
                ? had.closest(itemSel) : null;
            var key = item ? item.id : "";
            draw();
            settle();
            if (!item) return;
            var again = key ? doc.getElementById(key) : null;
            var to = again && listEl.contains(again) && shown(again)
                ? again : stopItem();
            if (to) {
                stopOn(to);
                to.focus({ preventScroll: true });
            }
        }
        // AN ITEM OF THIS LIST, the key's own target: a key pressed on a
        // control inside an item is that control's.
        function itemOf(target) {
            var el = target && target.closest ? target.closest(itemSel) : null;
            return el && el === target && listEl.contains(el) ? el : null;
        }

        on(listEl, "keydown", function (ev) {
            var item = itemOf(ev.target);
            if (!item) return;
            var all = items();
            var i = all.indexOf(item);
            var to = ev.key === "ArrowDown" ? all[i + 1]
                : ev.key === "ArrowUp" ? all[i - 1]
                : ev.key === "Home" ? all[0]
                : ev.key === "End" ? all[all.length - 1] : null;
            if (to) {
                ev.preventDefault();
                stopOn(to);
                to.focus();
            } else if (ev.key === "Enter" || ev.key === " ") {
                ev.preventDefault();
                if (typeof opts.commit === "function" && opts.current
                        && item.matches(opts.current)) {
                    opts.commit(item);
                } else {
                    item.click();
                }
            }
        });
        // THE STOP FOLLOWS FOCUS: an item clicked, or reached by the arrows,
        // is where Tab comes back to.
        on(listEl, "focusin", function (ev) {
            var item = ev.target && ev.target.closest
                ? ev.target.closest(itemSel) : null;
            if (item && listEl.contains(item)) stopOn(item);
        });
        settle();
        return { settle: settle, redraw: redraw };
    }

    root.molbuilder = root.molbuilder || {};
    root.molbuilder.oneStopList = { mount: mountOneStopList };
})(typeof window !== "undefined" ? window : this);
