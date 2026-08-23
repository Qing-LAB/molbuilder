/**
 * status.js — the ONE status-line writer.
 *
 * The CSS side has been a properly shared component for a long time:
 * `.status` and its severities live in `page-shell.css`, declared once, so an
 * error looks the same on every tab (`ui-contract.md` § 4, § 5).  The WRITER
 * was hand-rolled fifteen times.
 *
 * That is not a cosmetic duplication.  The copies disagreed about things that
 * matter:
 *
 *   * a missing slot.  Most copies returned silently; one logged a warning,
 *     and that one is there because a silent return once turned a MolView
 *     mount failure completely invisible -- the catch handler died on a
 *     phantom id and the user saw nothing at all.
 *   * the severity vocabulary.  Some passed any `kind` straight into the
 *     class; others allowed exactly `ok|warn|error` and silently DROPPED
 *     anything else, so `setStatus(msg, "muted")` rendered as neutral on one
 *     tab and as nothing on another.
 *
 * One writer, so those cannot differ again.  The severities are the four
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
            // LOUD, because the silent version hid a real one.  A status slot
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
        el.textContent = msg == null ? "" : String(msg);
        el.className = "status" + (kind ? " " + kind : "");
        return true;
    }

    /** A `(msg, kind)` bound to one slot — the shape most callers had. */
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
