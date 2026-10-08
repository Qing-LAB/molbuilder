/**
 * state-chip.js — the ONE state chip: a stage's or a run's state word, in
 * its tone.
 *
 * The words are the doors' (`jobset/runstatus.py`, `jobset/ready.py`;
 * `running-a-job.md` § 4.2, `job-system.md`, *The task*): every page that
 * shows where a stage or a run stands -- the Results ladder and Run panel,
 * the transport report, the bench summary, Task setup's Prep -- draws it
 * here, so a word reads the same everywhere.  Its look is `state-chip.css`.
 *
 * Exports (on window.molbuilder):
 *   stateChip(state)  → a <span> holding the word, toned
 *   stateTone(state)  → "ok" | "busy" | "warn" | "bad" | "idle"
 */
(function () {
    "use strict";

    var root = (typeof globalThis !== "undefined") ? globalThis
            : (typeof window !== "undefined") ? window : this;

    /* A word missing here still renders -- as itself, in the neutral tone
     * -- because inventing a severity for a state we do not know is worse
     * than showing the state. */
    var TONE = {
        finished:   "ok",
        running:    "busy",
        queued:     "busy",
        pending:    "busy",
        prepared:   "busy",
        ready:      "idle",
        waiting:    "idle",
        missing:    "bad",
        unknown:    "warn",
        unreadable: "bad",
        failed:     "bad",
    };

    function stateTone(state) {
        return TONE[state] || "idle";
    }

    function stateChip(state) {
        var s = root.document.createElement("span");
        s.className = "state-chip is-" + stateTone(state);
        s.textContent = String(state);
        return s;
    }

    root.molbuilder = root.molbuilder || {};
    root.molbuilder.stateChip = stateChip;
    root.molbuilder.stateTone = stateTone;
})();
