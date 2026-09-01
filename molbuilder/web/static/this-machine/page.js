/* This machine — notification channels and this server's listener.
 *
 * `this-machine.md` is the contract.  Two rules govern everything below:
 *
 *   1. THIS PAGE WRITES SECRETS AND NEVER READS THEM BACK.  Nothing here
 *      fetches a stored key or an unmasked webhook address, because the API
 *      does not offer one -- the rule is enforced at the door
 *      (`notify_setup.py::_row`), not by this file remembering to be careful.
 *      The single exception is a key the server has just MINTED, which is
 *      shown once because it cannot reach the cluster otherwise.
 *
 *   2. THE SERVER'S ANSWER IS THE STATE.  Every mutation repaints from the
 *      `channels` array the response carries, rather than patching the DOM
 *      from what was typed.  A page that believes its own optimistic edit is
 *      a page that shows a channel it failed to save.
 */
(function () {
    "use strict";

    var status = (window.molbuilder && window.molbuilder.status) || null;
    function say(slot, msg, kind) {
        if (status) { status.set(slot, msg, kind); return; }
        var el = document.getElementById(slot);      // pragma: no cover
        if (el) { el.textContent = msg; }
    }
    function $(id) { return document.getElementById(id); }

    var CHANNELS = "/api/notify/channels";
    var LISTENER = "/api/notify/listener";

    /* Remove is destructive, so it asks -- by becoming its own confirmation
     * rather than opening a modal.  One click arms, the next removes, and
     * anything else disarms; a mis-click costs a second, not a channel. */
    var armed = null;

    function ago(seconds) {
        if (!seconds) return "";
        var d = Math.max(0, Date.now() / 1000 - seconds);
        if (d < 90) return "just now";
        if (d < 5400) return Math.round(d / 60) + "m ago";
        if (d < 129600) return Math.round(d / 3600) + "h ago";
        return Math.round(d / 86400) + "d ago";
    }

    /* ---------- channels ---------- */

    function rowFor(c) {
        var row = document.createElement("div");
        row.className = "tm-row";
        row.setAttribute("role", "listitem");

        var name = document.createElement("div");
        name.className = "tm-row-name";
        name.textContent = c.name;
        var kind = document.createElement("div");
        kind.className = "tm-row-kind";
        kind.textContent = c.kind;
        name.appendChild(kind);

        var where = document.createElement("div");
        where.className = "tm-row-where";
        /* MASKED IS NOT AN ERROR, so it is not styled as one -- but it has to
         * be legible as masked, or a person compares a truncated address
         * against the real one and concludes it is wrong. */
        where.textContent = c.where + "  (masked)";
        where.title = "Addresses are never shown in full: for Slack and "
            + "Discord the address IS the credential. Test the channel to "
            + "prove it works.";

        var test = document.createElement("div");
        test.className = "tm-row-test";
        if (c.tested_ok === true) {
            test.textContent = "reached " + ago(c.tested_at);
            test.setAttribute("data-state", "ok");
        } else if (c.tested_ok === false) {
            test.textContent = "failed " + ago(c.tested_at);
            test.setAttribute("data-state", "bad");
        } else {
            test.textContent = "never tested";
        }

        var acts = document.createElement("div");
        acts.className = "tm-row-actions";
        var t = document.createElement("button");
        t.type = "button";
        t.textContent = "Test";
        t.addEventListener("click", function () { testChannel(c.name); });
        var d = document.createElement("button");
        d.type = "button";
        d.className = "danger";
        d.textContent = armed === c.name ? "Really remove" : "Remove";
        d.addEventListener("click", function () { removeChannel(c.name); });
        acts.appendChild(t);
        acts.appendChild(d);

        row.appendChild(name);
        row.appendChild(where);
        row.appendChild(test);
        row.appendChild(acts);
        return row;
    }

    function paintChannels(d) {
        var list = $("tm-list");
        if (list) {
            list.textContent = "";
            (d.channels || []).forEach(function (c) {
                list.appendChild(rowFor(c));
            });
        }
        /* A BROKEN FILE AND NO FILE both mean nothing is sent and look
         * identical from outside.  Saying which is most of what this page is
         * worth. */
        if (d.problem) {
            say("tm-file", "There is a file at " + d.path + " but it cannot be "
                + "used: " + d.problem + " — nothing is being sent.",
                "error");
        } else if (!(d.channels || []).length) {
            say("tm-file", "No channels are set up, so nothing is sent. "
                + "They live in " + d.path + ".", "muted");
        } else {
            say("tm-file", (d.channels.length === 1 ? "One channel"
                : d.channels.length + " channels") + " in " + d.path
                + (d.mode && d.mode !== "0o600"
                   ? "  — warning: that file is " + d.mode
                     + ", not 0600" : ""),
                d.mode && d.mode !== "0o600" ? "warn" : "ok");
        }
    }

    async function loadChannels() {
        try {
            var r = await fetch(CHANNELS);
            paintChannels(await r.json());
        } catch (e) {
            say("tm-file", "Could not ask this server: " + e, "error");
        }
    }

    async function saveChannel(ev) {
        if (ev) ev.preventDefault();
        var name = ($("tm-name") || {}).value || "";
        var url = ($("tm-url") || {}).value || "";
        var key = ($("tm-key") || {}).value || "";
        var listener = $("tm-kind-listener") && $("tm-kind-listener").checked;
        name = name.trim();
        if (!name) { say("tm-form-note", "a name is required", "error"); return; }
        // ANY OTHER ACTION CANCELS A PENDING REMOVE.  An arm is a momentary
        // intent; leaving it standing across a save means a later single
        // click on that row deletes a channel the person had stopped
        // thinking about.
        armed = null;
        say("tm-form-note", "saving…", "muted");
        try {
            var r = await fetch(CHANNELS + "/" + encodeURIComponent(name), {
                method: "PUT",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    url: url.trim(),
                    key: listener ? key.trim() : "",
                }),
            });
            var d = await r.json();
            if (!d.ok) {
                say("tm-form-note", d.error || "could not save", "error");
                return;
            }
            /* CLEARED ON SUCCESS: the key field is write-only, and a secret
             * left sitting in the DOM is how it ends up in a screenshot. */
            if ($("tm-key")) $("tm-key").value = "";
            paintChannels(d);
            say("tm-form-note", "Saved " + name
                + ". Test it — that is the only way to know it works.",
                "ok");
        } catch (e) {
            say("tm-form-note", String(e), "error");
        }
    }

    async function testChannel(name) {
        armed = null;
        say("tm-form-note", "sending one report to " + name + "…", "muted");
        try {
            var d = await (await fetch(
                CHANNELS + "/" + encodeURIComponent(name) + "/test",
                { method: "POST" })).json();
            if (d.channels) paintChannels(d);
            if (d.ok) { say("tm-form-note", "It arrived.", "ok"); return; }
            /* The listener refuses every way identically, so the hint names
             * all of the possibilities rather than guessing between them. */
            say("tm-form-note",
                d.error || d.hint || ("it answered " + d.status), "error");
        } catch (e) { say("tm-form-note", String(e), "error"); }
    }

    async function removeChannel(name) {
        if (armed !== name) {
            armed = name;
            say("tm-form-note", "Click again to remove " + name
                + " — nothing will be sent there afterwards.", "warn");
            loadChannels();
            return;
        }
        armed = null;
        try {
            var d = await (await fetch(
                CHANNELS + "/" + encodeURIComponent(name),
                { method: "DELETE" })).json();
            paintChannels(d);
            say("tm-form-note", "Removed " + name + ".", "ok");
        } catch (e) { say("tm-form-note", String(e), "error"); }
    }

    /* ---------- the listener ---------- */

    function paintListener(d) {
        var users = $("tm-users");
        if (users) {
            users.textContent = "";
            (d.users || []).forEach(function (u) {
                var chip = document.createElement("span");
                chip.className = "tm-user";
                chip.textContent = u;
                users.appendChild(chip);
            });
        }
        if (!d.configured) {
            say("tm-listener-state", "This server is not receiving reports. "
                + "Issue a key below and it will — the key file is the "
                + "switch, so nothing else needs configuring.", "muted");
        } else if (!d.live) {
            /* CONFIGURED BUT NOT LIVE is a real state with a real cause: the
             * route is registered at startup from the key file, so the first
             * key ever issued does not open it until then.  A person watching
             * a 404 deserves to be told, rather than left to guess between
             * the four things a 404 can mean. */
            say("tm-listener-state", "A key file is in place (route "
                + d.route + ") but the route is not registered in the running "
                + "server. It is read at startup — restart to open it.",
                "warn");
        } else {
            say("tm-listener-state", "Receiving reports at /api/" + d.route
                + ", for " + (d.users || []).length + " user"
                + ((d.users || []).length === 1 ? "" : "s") + ".", "ok");
        }
    }

    async function loadListener() {
        try {
            paintListener(await (await fetch(LISTENER)).json());
        } catch (e) {
            say("tm-listener-state", "Could not ask this server: " + e,
                "error");
        }
    }

    async function issueKey(ev) {
        if (ev) ev.preventDefault();
        var user = (($("tm-user") || {}).value || "").trim();
        if (!user) {
            say("tm-listener-note", "a user id is required", "error");
            return;
        }
        var replace = $("tm-replace") && $("tm-replace").checked;
        say("tm-listener-note", "issuing…", "muted");
        try {
            var d = await (await fetch(
                LISTENER + "/keys/" + encodeURIComponent(user),
                { method: "POST",
                  headers: { "Content-Type": "application/json" },
                  body: JSON.stringify({ replace: !!replace }) })).json();
            if (!d.ok) {
                say("tm-listener-note", d.error || "could not issue", "error");
                return;
            }
            showIssued(d);
            say("tm-listener-note", d.joined
                ? "Issued, on the route already in that file — everybody "
                  + "already set up keeps working."
                : "Issued, and the route segment was generated.", "ok");
            loadListener();
        } catch (e) { say("tm-listener-note", String(e), "error"); }
    }

    /* THE ONE MOMENT A SECRET IS ON SCREEN.  It is not fetched; it is the
     * answer to the act that created it, and nothing can bring it back. */
    function showIssued(d) {
        var box = $("tm-issued");
        var body = $("tm-issued-body");
        if (!box || !body) return;
        body.textContent =
            "# On the machine that runs the jobs, as the channel of your\n"
            + "# choice inside \"$cfg/notify\":\n"
            + JSON.stringify(
                { channels: { molbuilder: { url: d.url, key: d.key } } },
                null, 2);
        box.hidden = false;
    }

    /* ---------- wiring ---------- */

    function syncKindField() {
        var listener = $("tm-kind-listener") && $("tm-kind-listener").checked;
        var field = $("tm-key-field");
        if (field) field.hidden = !listener;
        var url = $("tm-url");
        if (url) {
            url.placeholder = listener
                ? "https://your-server:8888/api/<segment>"
                : "https://hooks.slack.com/services/…";
        }
    }

    function init() {
        ["tm-kind-webhook", "tm-kind-listener"].forEach(function (id) {
            var el = $(id);
            if (el) el.addEventListener("change", syncKindField);
        });
        syncKindField();
        var form = $("tm-form");
        if (form) form.addEventListener("submit", saveChannel);
        var kf = $("tm-key-form");
        if (kf) kf.addEventListener("submit", issueKey);
        var hide = $("tm-issued-hide");
        if (hide) hide.addEventListener("click", function () {
            var box = $("tm-issued");
            var body = $("tm-issued-body");
            if (body) body.textContent = "";       // out of the DOM, not just
            if (box) box.hidden = true;            // out of sight
        });
        loadChannels();
        loadListener();
    }

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", init);
    } else {
        init();
    }
})();
