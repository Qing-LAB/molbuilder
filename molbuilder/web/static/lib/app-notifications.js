/* App-wide system-notification bar (notifications.md).
 *
 * ONE consistent surface for system-level messages that matter regardless of
 * tab -- debuting with the non-blocking/error-explicit store failures, a save
 * or a read (`notifications.md` § 3; the workspace side is `workspace.md`
 * § 5's `onPersistError` -- "never silent").  A STACK of notifications the user clears
 * INDIVIDUALLY (× per row); dedup by id; errors persist until dismissed.
 *
 * Public: window.molbuilder.notify = { show, clear, clearAll, list }.
 * Auto-source: listens for the ``molbuilder:persist-error`` DOM event.
 */
(function (root) {
    "use strict";

    var LEVELS = { error: 1, warn: 1, info: 1 };
    var _items = [];          // [{ id, level, message, count, el }]
    var _seq = 0;

    function _host() { return root.document && root.document.getElementById("app-notifications"); }

    function _runtime() {
        return root.molbuilder && root.molbuilder.runtime;
    }

    // Build (or rebuild) the DOM for one item.
    function _rowEl(item) {
        var doc = root.document;
        var li = doc.createElement("div");
        li.className = "app-notification app-notification--" + item.level;
        // Errors are assertive; warn/info polite (a11y, notifications.md §5).
        li.setAttribute("role", item.level === "error" ? "alert" : "status");
        if (item.level !== "error") li.setAttribute("aria-live", "polite");

        var body = doc.createElement("span");
        body.className = "app-notification__msg";
        // THE SERVER'S WORDS, DRAWN (`lib/dom.js`, ui-contract.md § 12.1).
        body.appendChild(root.molbuilder.dom.codeWords(
            item.message + (item.count > 1 ? "  (×" + item.count + ")" : "")));
        li.appendChild(body);

        // ONE LINK, where the message names what to do (notifications.md § 1).
        if (item.action) {
            var a = doc.createElement("a");
            a.className = "app-notification__action";
            a.href = item.action.href;
            a.textContent = item.action.label;
            if (item.action.newTab) {
                a.target = "_blank";
                a.rel = "noopener";
            }
            li.appendChild(a);
        }

        var x = doc.createElement("button");
        x.type = "button";
        x.className = "app-notification__dismiss";
        x.setAttribute("aria-label", "Dismiss notification");
        x.textContent = "×";
        x.addEventListener("click", function () { clear(item.id); });
        // Esc dismisses the focused notification (§5).
        li.addEventListener("keydown", function (e) {
            if (e.key === "Escape") { clear(item.id); }
        });
        li.appendChild(x);
        return li;
    }

    function _render() {
        var host = _host();
        if (!host) return;
        host.textContent = "";
        if (!_items.length) { host.hidden = true; return; }
        host.hidden = false;
        // Newest on top.
        _items.slice().reverse().forEach(function (item) {
            item.el = _rowEl(item);
            host.appendChild(item.el);
        });
        // "Clear all" affordance when 2+ are present.
        if (_items.length >= 2) {
            var clearAllBtn = root.document.createElement("button");
            clearAllBtn.type = "button";
            clearAllBtn.className = "app-notification__clear-all";
            clearAllBtn.textContent = "Clear all (" + _items.length + ")";
            clearAllBtn.addEventListener("click", clearAll);
            host.appendChild(clearAllBtn);
        }
    }

    // show({level, message, id?, detail?, action?}) -> {dismiss}.  Dedup by id: a
    // repeated id UPDATES its row (message + a ×N counter) instead of stacking.
    function show(opts) {
        opts = opts || {};
        var level = LEVELS[opts.level] ? opts.level : "info";
        var message = String(opts.message == null ? "" : opts.message);
        var id = opts.id != null ? String(opts.id) : ("_n" + (++_seq));
        var action = (opts.action && opts.action.label && opts.action.href)
            ? { label: String(opts.action.label), href: String(opts.action.href),
                newTab: !!opts.action.newTab }
            : null;

        var existing = null;
        for (var i = 0; i < _items.length; i++) {
            if (_items[i].id === id) { existing = _items[i]; break; }
        }
        if (existing) {
            existing.level = level;
            existing.message = message;
            existing.action = action;
            existing.count += 1;
        } else {
            _items.push({ id: id, level: level, message: message, count: 1,
                          action: action });
        }
        _render();
        return { dismiss: function () { clear(id); } };
    }

    function clear(id) {
        id = String(id);
        var next = [];
        for (var i = 0; i < _items.length; i++) {
            if (_items[i].id !== id) next.push(_items[i]);
        }
        if (next.length !== _items.length) { _items = next; _render(); }
    }

    function clearAll() {
        if (_items.length) { _items = []; _render(); }
    }

    function list() {
        return _items.map(function (it) {
            return { id: it.id, level: it.level, message: it.message, count: it.count };
        });
    }

    // ---- Source: workspace store failures (`notifications.md` § 3) ---- //
    // A burst of failures updates ONE row (stable id) instead of stacking.
    // A failed READ is a row of its own: the save warning is lowered by the
    // next save that lands, and that save does not put back what the read
    // could not.
    function _onPersistError(detail) {
        if (detail && detail.op === "read-state") {
            show({
                id:      "read-error",
                level:   "error",
                message: "What this page saved earlier could not be read back ("
                       + detail.error + "), so it was not put back.",
                detail:  detail,
            });
            return;
        }
        // ...THE READER'S SENTENCE, as the read row says it: the server's own
        // words, or what kind of failure it was (ui-contract.md § 12).
        show({
            id:      "persist-error",
            level:   "error",
            message: "Couldn't save state to disk ("
                   + ((detail && detail.error) || "no reason given")
                   + "). Your edits are kept in this page, but Retract and "
                   + "a reopened tab may not bring back everything.",
            detail:  detail,
        });
    }

    var api = { show: show, clear: clear, clearAll: clearAll, list: list };
    root.molbuilder = root.molbuilder || {};
    root.molbuilder.notify = api;
    if (_runtime() && typeof _runtime().register === "function") {
        _runtime().register("notify", api);
    }

    // Wire the persist-error DOM event (dispatched by the workspace dispatcher).
    if (root.addEventListener) {
        // Saving recovered -- take the warning down.
        root.addEventListener("molbuilder:persist-ok", function () {
            clear("persist-error");
        });
        root.addEventListener("molbuilder:persist-error", function (e) {
            _onPersistError(e && e.detail);
        });
    }
})(typeof window !== "undefined" ? window : this);
