/* The JupyterNB tab -- four states, and the person decides between them.
 *
 * Contract: docs/web/jupyter.md.  What this file is careful about:
 *
 *   * NOTHING IS STARTED BY LOOKING.  Opening the tab probes and reports;
 *     starting a notebook runs code as the account serving this page, so it
 *     is a thing a person asks for (`jupyter.md` 4, access-control.md 6).
 *   * THE ENV IS OPT-IN.  "not installed" is an ordinary state with an
 *     install command as its answer -- not a Start button that would fail.
 *   * WHERE A NOTEBOOK LANDS is decided by LAB's own file browser, not by
 *     molbuilder.  The projects sidebar -- mounted here but not shown -- only
 *     decides which folder Lab OPENS at, once, when the frame is first
 *     pointed.  The server is rooted at the projects tree and cannot be
 *     re-rooted without a restart, so that root is what it can reach.
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
/** How long "up but not answering" is given before it is called wedged.
 *  `null` = not in that state; armed on entry, cleared on any other state. */
const WEDGE_POLLS = 20;          // 20 x 1.5s = 30s
let wedgePolls = null;
/** WHAT IS IN THE FRAME, by identity -- port, token and folder.
 *
 *  Keying the "do I need to re-point?" question on `framedDir` alone was a
 *  hole: a notebook that went away by any route the TAB did not perform (its
 *  own idle shutdown, `molbuilder jupyter stop`, a Stop in another browser
 *  tab) left `framedDir` set and `frame.src` non-empty.  The next Start got a
 *  new shepherd with a NEW TOKEN, render reached the framed branch, and the
 *  guard said there was nothing to do -- so the tab re-revealed the old
 *  document, still pointing at a dead server with a dead token, under a
 *  message saying all was well.  Only a page reload escaped it. */
let framedKey = null;

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
  // `reset` -- START CLEAN, EVERY TIME.  Lab restores a saved WORKSPACE on
  // load -- which documents were open, which side panel was showing -- and
  // that restore argues with the one thing this tab decides: the folder Lab
  // opens at.  The notebook FILE is the state worth keeping and it is on
  // disk; a panel layout is not (decided with the user 2026-09-14).
  // `?reset` is Jupyter's own flag for it.
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
  // NOTHING RUNNING MEANS NOTHING TO SHOW.  Whatever is in the frame now
  // belongs to a server that is gone; blank it here, once, rather than
  // leaving every later branch to remember.
  if (!st.running) {
    if (frame.src && frame.src !== "about:blank") frame.src = "about:blank";
    framedKey = null;
  }
  if (!(st.running && !st.answering)) wedgePolls = null;
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
    // THE KEY IS THE SERVER'S IDENTITY, AND ONLY THAT.
    //
    // It carried the folder too, and that swallowed a second decision: the
    // folder is in the key, so a selection change made the key differ and
    // the frame was RE-POINTED -- tearing Lab down with anything unsaved in
    // it, which is exactly what the Stop handler below refuses to do and
    // what the subscription at the foot of this file says must not happen.
    // Port and token are what "is the frame pointing at a live server?"
    // means; the folder is a separate, person-driven question.
    const key = `${st.port}|${st.token}`;
    if (framedKey !== key) {
      framedKey = key;
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
    // (An "Open <folder>" button stood here, offering to move Lab when the
    // sidebar's selection changed.  The sidebar is not SHOWN on this tab any
    // more -- Lab carries its own file browser, and a second tree that does
    // not decide where a notebook saves is a control that lies -- so nothing
    // on this page can change the selection and the button could never
    // appear.  Deleted with `framedDir` and the selection subscription.)
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
        framedKey = null;
        pollSoon(600);
      }));
    }
    // A SLOW HEARTBEAT, not silence.  This called `stopPolling()`, so once
    // framed the tab asked nothing ever again -- and Jupyter's own idle
    // shutdown (an hour, `jupyter.py`) would take the server out from under
    // a frame still claiming all was well.  Thirty seconds is cheap (two
    // probes) and is what lets the row and the frame both notice.
    pollSoon(30000);
    return;
  }

  // (3) The process is up but not serving yet -- starting, or wedged.
  if (st.running && !st.answering) {
    // BOUNDED, because this branch's own comment calls the state "starting,
    // OR WEDGED".  It polled forever, and each poll costs the server two
    // blocking probes of Jupyter -- so a wedged notebook meant an eternal
    // "Starting…" at 0.67 requests a second, which is the cost this file
    // refuses elsewhere.
    // ITS OWN COUNTER, not the Start button's.  Keying this on
    // `startPollsLeft` would bound it only for someone who had just clicked
    // Start -- a tab OPENED onto an already-wedged notebook has that counter
    // at zero and would poll forever, which is the case this bound exists
    // for.  `wedgePolls` is armed on entry to the state and cleared on the
    // way out (top of `render`).
    if (wedgePolls === null) wedgePolls = WEDGE_POLLS;
    wedgePolls -= 1;
    if (wedgePolls <= 0) {
      stopPolling();
      wrap.hidden = true;
      sayError("The notebook process is up but never started answering. "
               + "It is wedged; stop it and look at the notebook log.");
      if (st.may_control) {
        actions.appendChild(button("Stop JupyterLab", async () => {
          say("Stopping…");
          await post("/api/jupyter/stop");
          pollSoon(600);
        }));
      }
      return;
    }
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
      // THE SERVE LOG, NOT THE NOTEBOOK LOG.  Every way `_start_jupyter`
      // can fail -- no argv (a supervisor that predates the feature), the
      // log could not be opened, the spawn raised -- writes to the SERVE
      // log, because the notebook log is the thing that could not be
      // started.  This named the notebook log for the very cause it names
      // in the sentence before.
      sayError("Asked for a notebook and none started. The supervisor may "
               + "predate this feature — it survives a code reload, so "
               + "`molbuilder serve stop` then `serve start` gives it one. "
               + "`molbuilder serve status` names the server log, which says "
               + "which.");
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
  say("Start JupyterLab to write notebooks under your projects tree. Its "
      + "kernel is the notebook env's own python — numpy, scipy, pandas and "
      + "matplotlib. It runs as the account serving this page and stops when "
      + "molbuilder stops.");
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
    // A HICCUP MUST NOT BE TERMINAL.  This stopped polling, printed an error
    // and left `actions` as it found it -- which on the start path is EMPTY,
    // because the click cleared it.  One dropped fetch mid-start therefore
    // left no message that helps, no button, no retry and no poll: reload
    // only, the same dead end the projects-root wait was fixed for.
    stopPolling();
    actions.replaceChildren();
    sayError("Could not ask this server about the notebook: "
             + (err.message || err));
    actions.appendChild(button("Try again", () => {
      say("Checking…");
      refresh();
    }, { primary: true }));
  }
}

// NO SELECTION SUBSCRIPTION.  One stood here to re-render when the projects
// sidebar's selection moved -- but the sidebar is mounted and NOT SHOWN on
// this tab (`jupyternb.html`), so nothing here can move it, and the only
// thing the re-render offered was a button that could never appear.  The
// selection is read ONCE, when the frame is first pointed.

refresh();
