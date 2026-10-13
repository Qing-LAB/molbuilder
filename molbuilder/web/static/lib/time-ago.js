/* time-ago.js — how long ago, said one way on every page
 * (`ui-contract.md` § 4).
 *
 * `timeAgo(epochSeconds)`: the age of a moment, in the unit a person reads
 * it in -- each unit until one and a half of the next (90 s, 90 min, 36 h),
 * "just now" below that, and past 30 days the date itself, which reads better
 * than a count of days; "" when there is no moment to age.  A moment ahead
 * of this clock (another machine's) is "just now".
 *
 * Exports (on window.molbuilder.timeAgo): the function itself.
 */
(function (root) {
    "use strict";

    const MINUTE = 60, HOUR = 60 * MINUTE, DAY = 24 * HOUR, MONTH = 30 * DAY;

    function timeAgo(epochSeconds) {
        if (!(epochSeconds > 0)) return "";
        const d = Math.max(0, Date.now() / 1000 - epochSeconds);
        if (d < 1.5 * MINUTE) return "just now";
        if (d < 1.5 * HOUR) return Math.round(d / MINUTE) + "m ago";
        if (d < 1.5 * DAY) return Math.round(d / HOUR) + "h ago";
        if (d < MONTH) return Math.round(d / DAY) + "d ago";
        return new Date(epochSeconds * 1000).toLocaleDateString();
    }

    root.molbuilder = root.molbuilder || {};
    root.molbuilder.timeAgo = timeAgo;
})(typeof window !== "undefined" ? window : this);
