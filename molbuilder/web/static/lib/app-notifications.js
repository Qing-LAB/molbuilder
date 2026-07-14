/* App-wide system-notification bar (notifications.md).
 *
 * ONE consistent surface for system-level messages that matter regardless of
 * tab -- debuting with the non-blocking/error-explicit persist failures
 * (workspace-contract.md §4.7).  A STACK of notifications the user clears
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
        body.textContent = item.message + (item.count > 1 ? "  (×" + item.count + ")" : "");
        li.appendChild(body);

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

    // show({level, message, id?, detail?}) -> {dismiss}.  Dedup by id: a
    // repeated id UPDATES its row (message + a ×N counter) instead of stacking.
    function show(opts) {
        opts = opts || {};
        var level = LEVELS[opts.level] ? opts.level : "info";
        var message = String(opts.message == null ? "" : opts.message);
        var id = opts.id != null ? String(opts.id) : ("_n" + (++_seq));

        var existing = null;
        for (var i = 0; i < _items.length; i++) {
            if (_items[i].id === id) { existing = _items[i]; break; }
        }
        if (existing) {
            existing.level = level;
            existing.message = message;
            existing.count += 1;
        } else {
            _items.push({ id: id, level: level, message: message, count: 1 });
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

    // ---- Source: workspace persist failures (workspace-contract.md §4.7) ---- //
    // A burst of failures updates ONE row (stable id) instead of stacking.
    function _onPersistError(detail) {
        var where = (detail && detail.op) ? detail.op : "write";
        var code = (detail && detail.status) ? " · HTTP " + detail.status
                 : (detail && detail.error) ? " · " + detail.error : "";
        show({
            id:      "persist-error",
            level:   "error",
            message: "Couldn't save state to disk (" + where + code
                   + "). Your edits are kept in memory, but retract / "
                   + "crash-recovery history may be incomplete.",
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
        root.addEventListener("molbuilder:persist-error", function (e) {
            _onPersistError(e && e.detail);
        });
    }
})(typeof window !== "undefined" ? window : this);
