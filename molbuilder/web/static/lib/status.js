/**
 * status.js — the ONE status-line writer.
 *
 * `.status` and its severities live in `page-shell.css`, declared once, so an
 * error looks the same on every tab (`ui-contract.md` § 4, § 5).
 *
 * The severities are the four
 * `page-shell.css` declares; an unknown one is a mistake worth hearing about
 * rather than a class that quietly does nothing.
 *
 * Exports (on window.molbuilder.status):
 *   set(target, msg, kind)  → wrote it?  target is an element or an id
 *   writer(target)          → a bound (msg, kind) for a fixed slot
 */
(function () {
    "use strict";

    var root = (typeof globalThis !== "undefined") ? globalThis
            : (typeof window !== "undefined") ? window : this;

    //: The severities `page-shell.css` declares.  `null` is the neutral line.
    var KINDS = ["ok", "warn", "error", "muted"];

    function _resolve(target) {
        if (!target) return null;
        if (typeof target === "string") {
            return root.document.getElementById(target);
        }
        return target;                       // already an element
    }

    /**
     * Write a message into a status line.  Returns false when there was no
     * slot to write into -- callers that care can say so; the point is that
     * reporting a failure never becomes a failure of its own.
     */
    function set(target, msg, kind) {
        var el = _resolve(target);
        if (!el) {
            // A status slot
            // that does not exist is a bug in the page, and the message it
            // was carrying is usually the report of another bug.
            if (root.console && root.console.warn) {
                root.console.warn("[status] no slot "
                    + (typeof target === "string" ? "#" + target : "(element)")
                    + " for: " + msg);
            }
            return false;
        }
        if (kind && KINDS.indexOf(kind) === -1) {
            if (root.console && root.console.warn) {
                root.console.warn("[status] unknown severity " + kind
                    + " (known: " + KINDS.join(", ") + ") for: " + msg);
            }
            kind = null;
        }
        /* THE SEVERITY IS REPLACED, NOTHING ELSE: every class the line
         * carries for its own layout stays.  And a line whose text and severity are
         * unchanged is not written again: a status line is a live region,
         * and a poll that rewrote the same words re-announced them to a
         * screen reader on every tick. */
        var text = msg == null ? "" : String(msg);
        var classes = String(el.className || "").split(/\s+/).filter(
            function (c) { return c && KINDS.indexOf(c) === -1; });
        if (classes.indexOf("status") === -1) classes.unshift("status");
        if (kind) classes.push(kind);
        var cls = classes.join(" ");
        if (el.textContent === text && el.className === cls) return true;
        el.textContent = text;
        el.className = cls;
        return true;
    }

    /** A `(msg, kind)` bound to one slot. */
    function writer(target) {
        return function (msg, kind) { return set(target, msg, kind); };
    }

    var api = { set: set, writer: writer, KINDS: KINDS };
    if (typeof module !== "undefined" && module.exports) {
        module.exports = api;
    }
    root.molbuilder = root.molbuilder || {};
    root.molbuilder.status = api;
})();
