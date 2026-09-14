/* The JupyterNB tab -- four states, and the person decides between them.
 *
 * Contract: docs/web/jupyter.md.  What this file is careful about:
 *
 *   * NOTHING IS STARTED BY LOOKING.  Opening the tab probes and reports;
 *     starting a notebook runs code as the account serving this page, so it
 *     is a thing a person asks for (`jupyter.md` 4, access-control.md 6).
 *   * THE ENV IS OPT-IN.  "not installed" is an ordinary state with an
 *     install command as its answer -- not a Start button that would fail.
 *   * WHERE A NOTEBOOK LANDS is the folder the sidebar has selected.  The
 *     SERVER is rooted at the projects tree and cannot be re-rooted without
 *     a restart (which kills every kernel), so the selection decides where
 *     Lab OPENS, not what it can reach.
 *   * ONE SENTENCE, ONE ROW.  The page has no heading and no lede; this
 *     message is the whole of what the tab says, so each state's text has to
 *     carry what you can do from here, not just what is true.
 */
const $ = (id) => document.getElementById(id);

const state   = $("nb-state");
const actions = $("nb-actions");
const cmd     = $("nb-cmd");
const wrap    = $("nb-frame-wrap");
const frame   = $("nb-frame");

/** Poll while something is starting; stopped as soon as it answers. */
let pollTimer = null;
/** How many polls a start gets before the tab stops waiting and says so.
 *
 *  THE START ENDPOINT CANNOT REPORT AN ASYNCHRONOUS FAILURE.  It answers 202
 *  the moment the signal is DELIVERED to the supervisor, and the supervisor's
 *  handler is a no-op when it holds no notebook argv -- which is the ordinary
 *  state of a supervisor that predates this feature, because a supervisor
 *  SURVIVES a code reload by design (`jupyter.md` § 3.4).  A shepherd that
 *  starts and exits at once (broken env) reads the same.  Without a bound the
 *  tab just redrew the Start button with no message: click, nothing, click,
 *  nothing, and the only evidence in a log the person is not reading. */
const START_POLLS = 12;
let startPollsLeft = 0;
/** What the frame was pointed at, so a selection change can offer to move. */
let framedDir = null;

/** Set the one message.  Strings become TEXT NODES and never markup: the
 *  folder name in the running state is chosen by the person, so a folder
 *  called `<img onerror=...>` would run through innerHTML -- caught by the
 *  XSS audit on 2026-09-14, which is what that audit is for. */
function say(...parts) {
  state.className = "status nb-msg";
  state.replaceChildren(...parts.map((p) =>
    typeof p === "string" ? document.createTextNode(p) : p));
}

/** The same message, in the shell's `status error` colour -- the ONE home for
 *  that severity is page-shell.css, and this page uses it rather than picking
 *  a red of its own (`ui-contract.md` 5). */
function sayError(...parts) {
  say(...parts);
  state.className = "status nb-msg error";
}

/** A `<code>` span for the message -- the only element it ever contains. */
const code = (text) =>
  Object.assign(document.createElement("code"), { textContent: text });

function button(label, onClick, opts = {}) {
  const b = document.createElement("button");
  b.type = "button";
  b.textContent = label;
  if (opts.primary) b.className = "primary";
  b.addEventListener("click", onClick);
  return b;
}

/** The selected folder, relative to the projects root -- Lab's own path shape.
 *
 *  Three answers, and the third is the one that matters:
 *    * a path   -- that folder
 *    * ""       -- the projects root itself
 *    * null     -- **NOT KNOWN YET**.
 *
 *  `getCurrentDir` reads sessionStorage and answers immediately, but
 *  `getProjectsRoot` returns "" until the sidebar has resolved
 *  `/api/files/roots` -- an async bootstrap that is usually still in flight
 *  when this page first renders.  Folding that into "" meant the frame opened
 *  at the ROOT on almost every load, and Lab's Launcher creates a notebook in
 *  the folder the frame is showing: every new notebook landed in `projects/`
 *  however carefully the person had picked a folder first (reported
 *  2026-09-14, `projects/Untitled.ipynb` was the evidence).  Waiting one poll
 *  is the whole fix; guessing the root is what was wrong.
 */
function selectedRelative() {
  const p = (window.molbuilder && window.molbuilder.projects) || null;
  if (!p || typeof p.getProjectsRoot !== "function") return "";
  const root = p.getProjectsRoot() || "";
  if (!root) return null;                    // bootstrap still in flight
  const dir = (p.getCurrentDir && p.getCurrentDir()) || "";
  if (!dir || !dir.startsWith(root)) return "";
  return dir.slice(root.length).replace(/^\/+/, "");
}

function labUrl(st, rel) {
  // `/lab/tree/<path>` is JupyterLab's own "open here" URL, and the token is
  // how Jupyter authenticates the browser.  Both are Jupyter's shapes; this
  // composes them and invents nothing.
  //
  // THE HOST IS THIS PAGE'S, NOT THE SERVER'S BIND ADDRESS.  The status
  // payload carries an absolute url built from what the notebook BOUND to
  // (127.0.0.1 here), and `127.0.0.1` in a browser means the BROWSER's own
  // machine -- so through a tunnel, or from any other host, the frame asked
  // its own laptop for port 6007 and got "refused to connect" (measured
  // 2026-09-14).  The notebook is always on this same host, one port up, so
  // the only honest base is the one this page was reached at.
  //
  // `reset` -- START CLEAN, EVERY TIME.  Lab restores a saved WORKSPACE (which
  // documents were open, which side panel was showing) on load, and that
  // restore fought everything this tab decides: it reopened the file browser
  // panel over the single-document layout, and it argued with the folder the
  // projects sidebar had selected.  The notebook FILE is the state worth
  // keeping and it is on disk; the panel layout is not (decided with the user
  // 2026-09-14).  `?reset` is Jupyter's own flag for this.
  const base = `${location.protocol}//${location.hostname}:${st.port}`;
  const path = rel ? `/tree/${rel.split("/").map(encodeURIComponent).join("/")}` : "";
  const query = st.token
    ? `?token=${encodeURIComponent(st.token)}&reset`
    : "?reset";
  return `${base}/lab${path}${query}`;
}

async function post(path) {
  const r = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
  });
  let body = {};
  try { body = await r.json(); } catch (_) { /* a refusal may carry no JSON */ }
  return { ok: r.ok && body.ok !== false, status: r.status, body };
}

/** Run `fn` once the projects root is known.  Idempotent: a second call
 *  before the root lands replaces nothing and subscribes nothing twice --
 *  `onChange`-style subscribers throw if registered twice. */
let rootWaiter = null;
function whenRootKnown(fn) {
  const p = (window.molbuilder && window.molbuilder.projects) || null;
  if (!p || typeof p.onProjectsRootResolved !== "function") return false;
  if (rootWaiter) return true;
  rootWaiter = p.onProjectsRootResolved(() => {
    if (rootWaiter) { rootWaiter(); rootWaiter = null; }
    fn();
  });
  return true;
}

/** How long to wait for the sidebar before giving up and opening at the root.
 *
 *  THE DOOR CAN NEVER FIRE.  `onProjectsRootResolved` publishes only after
 *  `/api/files/roots` SUCCEEDS, so a failed bootstrap leaves it silent -- and
 *  the first version of this wait had no terminal state at all: no message
 *  change, no button, no retry, and a selection change could not recover it
 *  either, because that handler is gated on the frame being visible.  A page
 *  reload was the only way out.  Opening at the projects root is a worse
 *  default than the selected folder and a far better one than a dead tab. */
const ROOT_WAIT_MS = 8000;
let rootGaveUp = false;

function stopPolling() {
  if (pollTimer) { clearTimeout(pollTimer); pollTimer = null; }
}

function pollSoon(ms) {
  stopPolling();
  pollTimer = setTimeout(refresh, ms);
}

function render(st) {
  actions.replaceChildren();
  cmd.hidden = true;
  // THE START WORKED, SO STOP COUNTING.  `startPollsLeft` was only ever
  // decremented on the nothing-is-running path, and never cleared when a
  // start succeeded -- so it sat at whatever was left over, and the next time
  // the notebook legitimately went away (a Stop the person clicked, or
  // Jupyter's own idle shutdown) the tab spent twelve seconds showing a stale
  // message and then printed a red "none started" error about a start that
  // had worked, with no button and no way out but a reload.
  if (st.running) startPollsLeft = 0;

  // (1) The env is not installed.  An opt-in env, so this is ordinary.
  if (!st.env_installed) {
    say(`JupyterLab is optional and its env (${st.env_name}) is not `
        + `installed. Install it, then reload this tab:`);
    cmd.textContent = st.install_command;
    cmd.hidden = false;
    wrap.hidden = true;
    stopPolling();
    return;
  }

  // (4) Up and answering -- frame it.
  if (st.running && st.answering) {
    // NO TOKEN, NO FRAME.  The token is what authenticates this browser to a
    // live kernel, and the server withholds it from a caller who may not
    // control the notebook (`blueprints/jupyter.py`).  Framing Lab anyway
    // would put Jupyter's own login page inside the tab, which reads as
    // molbuilder being broken rather than as a refusal.
    if (!st.token) {
      wrap.hidden = true;
      stopPolling();
      say("A notebook server is running on this machine, but using it means "
          + "running code as the account serving this page — so it is an "
          + "admin action. Ask whoever administers this server.");
      return;
    }
    const rel = selectedRelative();
    if (rel === null && !rootGaveUp) {
      // The sidebar has not resolved its root yet.  Framing now would open
      // Lab at the projects root and put every new notebook there.
      //
      // WAIT ON THE DOOR, NOT A TIMER.  `onProjectsRootResolved` fires the
      // moment the root lands (and immediately if it already has), so there
      // is nothing to poll for.  A 200 ms `pollSoon` stood here until
      // 2026-09-14 and never stopped: the sidebar publishes nothing when
      // `/api/files/roots` FAILS, so a failed bootstrap left this tab asking
      // the status endpoint five times a second forever -- and each of those
      // costs the server two blocking round-trips to Jupyter.
      say("Opening JupyterLab…");
      wrap.hidden = true;
      stopPolling();
      if (!whenRootKnown(refresh)) {
        // No door on this page at all -- proceed at the root rather than wait
        // for something that cannot happen.
        rootGaveUp = true;
        refresh();
        return;
      }
      setTimeout(() => {
        if (rootGaveUp) return;
        rootGaveUp = true;
        refresh();
      }, ROOT_WAIT_MS);
      return;
    }
    // Gave up waiting: `null` now means the projects root itself.
    const where = rel === null ? "" : rel;
    if (frame.src === "" || framedDir === null) {
      framedDir = where;
      frame.src = labUrl(st, where);
    }
    wrap.hidden = false;
    // WHAT IS OPEN, AND WHERE -- Jupyter's own answer (`open_notebooks`),
    // not a guess.  The tab used to claim "new notebooks are saved in
    // <the folder molbuilder selected>", which is true for exactly as long
    // as it takes to click a folder in Lab's own file browser: with several
    // notebooks open at once (multi-document mode) that claim is usually
    // wrong.  So the row states the fact it can know -- the full path of
    // every notebook Lab has open -- and names the control that decides the
    // next one instead of pretending to be it.
    const open = Array.isArray(st.open) ? st.open : [];
    const said = ["The kernel is the notebook env's own python — numpy, "
                  + "scipy, pandas and matplotlib. "];
    if (open.length) {
      said.push(open.length === 1 ? "Open: " : `Open (${open.length}): `);
      open.forEach((nb, i) => {
        if (i) said.push(", ");
        said.push(code(nb.path || "?"));
      });
      said.push(". ");
    }
    said.push("A new notebook is saved in the folder Lab's own file browser "
              + "is showing.");
    say(...said);
    const rel2 = selectedRelative();
    if (rel2 !== null && rel2 !== framedDir) {
      actions.appendChild(button(
        `Open ${rel2 || "projects/"}`, () => {
          framedDir = rel2;
          frame.src = labUrl(st, rel2);
          refresh();
        }, { primary: true }));
    }
    if (st.may_control) {
      actions.appendChild(button("Stop JupyterLab", async () => {
        say("Stopping…");
        const res = await post("/api/jupyter/stop");
        if (!res.ok) {
          // REFUSED, so change nothing.  Blanking the frame here tore Lab
          // down -- layout, open documents and any unsaved editor state --
          // and the next poll then re-framed it from scratch, for a stop
          // that never happened.  Start checked its answer; this did not.
          sayError(`Could not stop it: ${res.body.error || res.body.message ||
                   ("HTTP " + res.status)}`);
          return;
        }
        frame.src = "about:blank";
        framedDir = null;
        pollSoon(600);
      }));
    }
    stopPolling();
    return;
  }

  // (3) The process is up but not serving yet -- starting, or wedged.
  if (st.running && !st.answering) {
    say("Starting JupyterLab… it is up but not answering yet.");
    wrap.hidden = true;
    pollSoon(1500);
    return;
  }

  // Asked for a start, and nothing came up.  Say so once, with the log,
  // rather than redrawing the button as though nothing had been asked.
  if (startPollsLeft > 0) {
    startPollsLeft -= 1;
    if (startPollsLeft === 0) {
      stopPolling();
      wrap.hidden = true;
      sayError("Asked for a notebook and none started. The supervisor may "
               + "predate this feature — it survives a code reload, so "
               + "`molbuilder serve stop` then `serve start` gives it one. "
               + "The notebook log says which.");
      return;
    }
    pollSoon(1200);
    return;
  }

  // (2) Installed, nothing running.  Ask.
  stopPolling();
  wrap.hidden = true;
  if (!st.supervised) {
    // NOT `molbuilder jupyter start` -- that verb signals the supervisor, so
    // it is the one command guaranteed to fail in exactly this state, and the
    // CLI's own docstring says so.  `serve start` is the answer.
    say("JupyterLab is not running, and this molbuilder has no supervisor to "
        + "hold one — it was started with `serve foreground` or "
        + "`--no-supervise`. Restart it with `molbuilder serve start` and the "
        + "button appears here.");
    return;
  }
  if (!st.may_control) {
    say("JupyterLab is not running. Starting one runs code on this machine, "
        + "so it is an admin action — ask whoever administers this "
        + "server, or start it there with `molbuilder jupyter start`.");
    return;
  }
  say("Start JupyterLab to write notebooks under the project folder you pick "
      + "on the left. Its kernel is the notebook env's own python — numpy, "
      + "scipy, pandas and matplotlib. It runs as the account serving this "
      + "page and stops when molbuilder stops.");
  actions.appendChild(button("Start JupyterLab", async () => {
    say("Starting…");
    actions.replaceChildren();
    const res = await post("/api/jupyter/start");
    if (!res.ok) {
      sayError(`Could not start it: ${res.body.error || res.body.message ||
           ("HTTP " + res.status)}`);
      return;
    }
    startPollsLeft = START_POLLS;
    pollSoon(1200);
  }, { primary: true }));
}

async function refresh() {
  try {
    const r = await fetch("/api/jupyter/status");
    const st = await r.json();
    if (!st.ok) throw new Error("status refused");
    render(st);
  } catch (err) {
    stopPolling();
    sayError("Could not ask this server about the notebook: "
             + (err.message || err));
  }
}

// A selection change does NOT reload the frame -- that would tear down the
// Lab UI and anything unsaved in it.  It re-renders, which offers the move.
if (window.molbuilder && window.molbuilder.projects
    && typeof window.molbuilder.projects.onChange === "function") {
  window.molbuilder.projects.onChange(() => { if (!wrap.hidden) refresh(); });
}

refresh();
