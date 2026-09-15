/* The JupyterNB tab -- ten states, in one table.
 *
 * Contract: docs/web/jupyter.md § 4.  What this file is careful about:
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
 *
 * THE STATES ARE A TABLE, AND WAITING HAS ONE RULE (`plan.md` § 5n, J6/J7).
 * Until 2026-09-15 this file had five independent waiting mechanisms, each
 * added after its own incident: a poll timer, a `startPollsLeft` counter at
 * 1200 ms, a `wedgePolls` counter at 1500 ms, a `rootGaveUp` flag on a raw
 * `setTimeout`, and three bare `pollSoon(...)` calls.  Each
 * had its own clearing rule scattered through `render` -- one cleared in a
 * branch, one at the top of the function, and `rootGaveUp` never cleared at
 * all.  They are now ONE rule: a state is entered, and `STATES` says how
 * long it may last and what it becomes when that runs out.  The document
 * listed four states while this function had thirteen `return`s; the table
 * below and `jupyter.md` § 4 are now the same list.
 *
 * AND EVERY STATE EITHER POLLS OR OFFERS A BUTTON.  Four of them did
 * neither until 2026-09-15, so a notebook started from a terminal was
 * invisible to an open tab, and a budget that ran out was a dead end a page
 * reload was the only way out of (`plan.md` § 5n.8).
 */
const $ = (id) => document.getElementById(id);

const state   = $("nb-state");
const actions = $("nb-actions");
const cmd     = $("nb-cmd");
const wrap    = $("nb-frame-wrap");
const frame   = $("nb-frame");

/** Set the one message.  Strings become TEXT NODES and never markup: a path
 *  in the running state is chosen by the person, so a folder called
 *  `<img onerror=...>` would run through innerHTML -- caught by the XSS audit
 *  on 2026-09-14, which is what that audit is for. */
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

/** WHAT IS IN THE FRAME, by identity -- port and token, and only those.
 *
 *  It carried the selected folder too, and that swallowed a second decision:
 *  the folder was in the key, so a selection change made the key differ and
 *  the frame was RE-POINTED -- tearing Lab down with anything unsaved in it.
 *  Port and token are what "is the frame pointing at a live server?" means.
 *
 *  Keying on the folder ALONE was a third bug: a notebook that went away by
 *  any route the tab did not perform (its own idle shutdown, `molbuilder
 *  jupyter stop`, a Stop in another browser tab) left the key set and
 *  `frame.src` non-empty, so the next Start -- a new shepherd with a NEW
 *  TOKEN -- found the guard satisfied and re-revealed a dead document under
 *  a message saying all was well. */
let framedKey = null;

/** Set when the person clicks Start, cleared the moment something is running.
 *  THE START ENDPOINT CANNOT REPORT AN ASYNCHRONOUS FAILURE: it answers 202
 *  when the signal is DELIVERED, and the supervisor's handler is a no-op if
 *  it holds no notebook argv -- the ordinary state of a supervisor that
 *  predates this feature, because a supervisor SURVIVES a code reload by
 *  design (`jupyter.md` § 3.4).  A shepherd that starts and exits at once (a
 *  broken env) reads the same.  Without this the tab just redrew the Start
 *  button: click, nothing, click, nothing. */
let startAsked = false;

/** Set when THIS BROWSER cannot reach the notebook's port, whatever the
 *  server says about it.  Read by state 4b's `when`. */
let frameUnreachable = false;
/** The `port|token` the reachability probe has already answered for, so it
 *  runs once per server and not once per poll. */
let probedKey = null;

/** CAN THIS BROWSER REACH THE NOTEBOOK AT ALL?
 *
 *  **The server's own probe cannot answer this** (`plan.md` § 5n.8).
 *  `answering()` runs on the machine molbuilder is on; the FRAME runs in
 *  somebody's browser, and the two are not the same host through an ssh
 *  tunnel or behind a reverse proxy.  The tab builds the frame URL from
 *  `location.hostname` and `st.port` -- this page's host, one port up --
 *  and through `ssh -L 8000:server:8000` that is the LAPTOP's 8001, which
 *  nothing is listening on.  The iframe then showed the browser's
 *  connection-refused page inside a tab whose message said all was well,
 *  and NO state covered it: there is no load or error detection on an
 *  iframe that can be relied on cross-origin.
 *
 *  So the page asks the network directly.  A `no-cors` fetch cannot read
 *  the response -- that is the point -- but it distinguishes the two cases
 *  that matter: a reachable port RESOLVES (opaquely, CORS or not), and a
 *  refused connection or an unresolvable host REJECTS.  One request per
 *  server, not per poll.
 */
async function probeFrameReachable(base, key) {
  if (probedKey === key) return;
  probedKey = key;
  try {
    await fetch(`${base}/lab`, { mode: "no-cors", cache: "no-store" });
    frameUnreachable = false;
  } catch (_) {
    frameUnreachable = true;
    currentState = null;               // 4b is a different state; fresh clock
    refresh();
  }
}

/** Set when the projects root could not be resolved in time.  Once true,
 *  `opening` never matches again and `framed` opens at the projects root --
 *  a worse default than the selected folder and a far better one than a tab
 *  that waits forever. */
let rootIsLost = false;

/** Is there a projects-root door on this page at all?  In `opening`'s `when`
 *  rather than inside it: with no door there is nothing to wait FOR, so the
 *  honest answer is that this is not the waiting state -- `framed` takes it
 *  and opens at the root.  Handling it inside `enter` meant a state that had
 *  entered and immediately had to undo itself. */
function hasRootDoor() {
  const p = (window.molbuilder && window.molbuilder.projects) || null;
  return !!(p && typeof p.onProjectsRootResolved === "function");
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
 *  2026-09-14, `projects/Untitled.ipynb` was the evidence).  Waiting is the
 *  whole fix; guessing the root is what was wrong.
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

/** THE ONE BASE the frame and the reachability probe both use -- this
 *  page's host, one port up.  Two spellings would let the probe answer for
 *  an address the frame does not visit. */
const frameBase = (st) =>
  `${location.protocol}//${location.hostname}:${st.port}`;


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
  // CLEAN ON A NEW SERVER, PERSISTENT WHILE ONE RUNS (user, 2026-09-15).
  //
  // `?reset` used to be on EVERY load.  It was aimed at a real problem -- a
  // restored workspace argues with the folder this tab selects -- but at the
  // wrong moment: every LOAD, when what was meant was every START.  Leaving
  // /jupyternb destroys the iframe, so a tab switch is a load, and switching
  // away and back therefore threw away the notebook you had open while its
  // kernel was still running.
  //
  // The reset now happens once, server-side, when the notebook server
  // starts (`jupyter.prepare_lab_home`).  So there is nothing to reset here
  // and exactly one thing to decide: does Lab have a workspace to restore?
  //
  //   * NO  -- first framing since the server started, so carry the folder
  //            the projects sidebar selected and let Lab open there.
  //   * YES -- somebody has been working in it.  Carry NO tree path, so
  //            nothing competes with the restore.
  //
  // Asking the server (`st.workspace_saved`) rather than remembering in the
  // browser is what keeps the two halves from disagreeing: the wipe and the
  // flag read the same directory.  It also means the tree path and a restore
  // are never sent together, so Lab's own precedence between them -- which
  // molbuilder has not measured -- cannot decide anything here.
  const base = frameBase(st);
  const path = (rel && !st.workspace_saved)
    ? `/tree/${rel.split("/").map(encodeURIComponent).join("/")}`
    : "";
  const query = st.token ? `?token=${encodeURIComponent(st.token)}` : "";
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
 *  `onChange`-style subscribers throw if registered twice.  Returns false
 *  when the page has no such door at all. */
let rootWaiter = null;
function whenRootKnown(fn) {
  // NO DOOR CHECK HERE.  `hasRootDoor()` is that question's one home and
  // `opening.when` already asks it, so the copy this function carried --
  // with a `false` return nobody read -- could not fire (found in review
  // 2026-09-15, the residue of moving the check out of `enter`).
  if (rootWaiter) return;
  rootWaiter = window.molbuilder.projects.onProjectsRootResolved(() => {
    if (rootWaiter) { rootWaiter(); rootWaiter = null; }
    fn();
  });
}

/* ---- Waiting: one rule ---------------------------------------------------
 *
 * A state may declare `budgetMs`, and when the tab has been in it for that
 * long without leaving, it renders `expired` instead.  MILLISECONDS, not a
 * poll count, because the three budgets this replaced did not all wake the
 * same way: two polled (at two different intervals) and one waited on an
 * event.  Time is what they actually meant, and expressing it as time lets
 * each state keep the wake-up that suits it -- a poll where there is nothing
 * to subscribe to, the projects-root door where there is.
 *
 * ENTERING resets the clock, and that is the ONLY reset.  Three scattered
 * ones are gone with it: `startPollsLeft` was cleared inside a branch and
 * left stale everywhere else, so a notebook going away later spent twelve
 * seconds showing a stale message and then printed a red error about a start
 * that had worked; `wedgePolls` was cleared at the top of `render`; and
 * `rootGaveUp` was never cleared, so one slow bootstrap disabled the wait for
 * the life of the page. */
let currentState = null;
let enteredAt = 0;
let pollTimer = null;

function stopPolling() {
  if (pollTimer) { clearTimeout(pollTimer); pollTimer = null; }
}

function pollSoon(ms) {
  stopPolling();
  pollTimer = setTimeout(refresh, ms);
}

/* ---- The states ----------------------------------------------------------
 *
 * FIRST MATCH WINS, so the order is the contract; `jupyter.md` § 4 lists them
 * in this order.  Each row:
 *
 *   when(st)   -- is this the state?  `st` is the status payload.
 *   enter(st)  -- draw it.  May append buttons; must not poll.
 *   frame      -- is the iframe revealed?  (absent = no)
 *   pollMs     -- ask again after this long.  (absent = nothing to wait for)
 *   budgetMs   -- how long this state may last before `expired` takes over.
 *   expired(st)-- draw the give-up state.  Same freedom as `enter`.
 */
const STATES = [
  {
    // (1) The env is not installed.  An opt-in env, so this is ordinary.
    name: "env-missing",
    when: (st) => !st.env_installed,
    enter: (st) => {
      say(`JupyterLab is optional and its env (${st.env_name}) is not `
          + `installed. Install it, then reload this tab:`);
      cmd.textContent = st.install_command;
      cmd.hidden = false;
    },
    // SLOWLY, because this is the one poll that costs a subprocess: when the
    // env is absent the status endpoint re-asks the manager rather than
    // trusting a snapshot bound at process start.  Sixty seconds is enough
    // that "install it, then reload this tab" stops being the only way, and
    // rare enough that an idle tab is not running `conda env list` at people.
    pollMs: 60000,
  },
  {
    // (2) Running, but this caller was not given the token.
    //
    // NO TOKEN, NO FRAME.  The token authenticates this browser to a live
    // kernel, and the server withholds it from a caller who may not control
    // the notebook (`blueprints/jupyter.py`).  Framing Lab anyway would put
    // Jupyter's own login page inside the tab, which reads as molbuilder
    // being broken rather than as a refusal.
    name: "no-token",
    when: (st) => st.running && st.answering && !st.token,
    enter: () => say(
      "A notebook server is running on this machine, but using it means "
      + "running code as the account serving this page — so it is an admin "
      + "action. Ask whoever administers this server."),
    pollMs: 30000,
  },
  {
    // (3) Answering, but the sidebar has not resolved its root yet.
    //
    // WAIT ON THE DOOR, NOT A TIMER.  `onProjectsRootResolved` fires the
    // moment the root lands (and immediately if it already has), so there is
    // nothing to poll for -- a 200 ms `pollSoon` stood here until 2026-09-14
    // and never stopped, because the sidebar publishes nothing when
    // `/api/files/roots` FAILS, and each of those polls cost the server two
    // blocking round-trips to Jupyter.
    //
    // THE DOOR CAN NEVER FIRE, which is why this state has a budget at all:
    // a failed bootstrap leaves it silent, and the first version of this wait
    // had no terminal state -- no message change, no button, no retry, and a
    // page reload the only way out.  Opening at the projects root is a worse
    // default than the selected folder and a far better one than a dead tab.
    name: "opening",
    // `!st.workspace_saved` BECAUSE THE FOLDER IS ONLY USED WHEN THERE IS
    // NOTHING TO RESTORE (`labUrl`).  Without this term the tab hid the
    // frame for the full 8 s budget waiting for a value it was about to
    // throw away -- and it did so on the headline case J13 was written
    // for: switch away, come back, the sidebar's bootstrap is slow or has
    // failed, and the notebook you left open is behind "Opening
    // JupyterLab…" (found in review 2026-09-15).
    when: (st) => st.running && st.answering && !rootIsLost
                  && !st.workspace_saved
                  && hasRootDoor() && selectedRelative() === null,
    enter: () => {
      say("Opening JupyterLab…");
      whenRootKnown(refresh);
    },
    budgetMs: 8000,
    // Re-render with the payload in hand rather than re-ask the server: the
    // only thing that changed is a flag of ours, and `opening` cannot match
    // now that it is set.
    expired: (st) => { rootIsLost = true; render(st); },
  },
  {
    // (4a) Answering on the SERVER, unreachable from THIS BROWSER.
    //
    // The one state that is about the viewer's network rather than the
    // server's.  It exists because the frame URL is `this page's host, one
    // port up` -- true on the machine, false through a tunnel that forwards
    // only molbuilder's port, and false behind a reverse proxy that
    // terminates TLS and proxies one port.  Before 2026-09-15 every one of
    // those landed in state 4 with a message saying all was well
    // (`plan.md` § 5n.8, and `jupyter.md` § 2.1a).
    name: "frame-unreachable",
    when: (st) => st.running && st.answering && frameUnreachable,
    enter: (st) => {
      const base = frameBase(st);
      say("JupyterLab is running on the server, but this browser cannot "
          + "reach it. The notebook has its own port — the kernel talks to "
          + "it directly, so molbuilder cannot carry that traffic for you. "
          + "Make ",
          code(base),
          " reachable from here: forward that port too if you are "
          + "tunnelling (it is molbuilder's port plus one), or proxy it "
          + "beside molbuilder's own.");
      actions.appendChild(recheckButton("Try the frame again"));
    },
  },
  {
    // (4b) Up and answering -- frame it.
    name: "framed",
    when: (st) => st.running && st.answering,
    frame: true,
    // A SLOW HEARTBEAT, not silence.  This stopped polling once framed, so
    // the tab asked nothing ever again -- and Jupyter's own idle shutdown
    // (an hour) would take the server out from under a frame still claiming
    // all was well.  Thirty seconds is two probes, and is what lets the row
    // and the frame both notice.
    pollMs: 30000,
    enter: (st) => {
      // `selectedRelative()` is null only while the root is unknown, and
      // state (3) owns that -- except when it gave up, where null now means
      // the projects root itself.
      const rel = selectedRelative();
      const key = `${st.port}|${st.token}`;
      if (framedKey !== key) {
        framedKey = key;
        frame.src = labUrl(st, rel === null ? "" : rel);
      }
      // ASK THE NETWORK, ONCE PER SERVER.  Not awaited: the frame is
      // already pointed and a reachable notebook must not wait on a probe.
      // If it fails, `frameUnreachable` flips and state 4a takes over.
      probeFrameReachable(frameBase(st), key);
      // WHAT IS OPEN, AND WHERE -- Jupyter's own answer (`open_notebooks`),
      // not a guess.  The tab used to claim "new notebooks are saved in <the
      // folder molbuilder selected>", which is true for exactly as long as
      // it takes to click a folder in Lab's own file browser: with several
      // notebooks open at once (multi-document mode) that claim is usually
      // wrong.  So the row states the fact it can know -- the full path of
      // every notebook Lab has open -- and names the control that decides
      // the next one instead of pretending to be it.
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
      said.push("A new notebook is saved in the folder Lab's own file "
                + "browser is showing.");
      say(...said);
      if (st.may_control) actions.appendChild(stopButton());
    },
  },
  {
    // (5) The process is up but not serving yet -- starting, or wedged.
    //
    // ITS OWN BUDGET, not the Start button's.  Keying this on the start
    // attempt would bound it only for someone who had just clicked Start --
    // a tab OPENED onto an already-wedged notebook has made no attempt and
    // would poll forever, which is the case this bound exists for.  Each
    // poll costs the server two blocking probes of Jupyter.
    name: "starting",
    when: (st) => st.running && !st.answering,
    enter: () => say("Starting JupyterLab… it is up but not answering yet."),
    pollMs: 1500,
    budgetMs: 30000,
    expired: (st) => {
      sayError("The notebook process is up but never started answering. "
               + "It is wedged, or a cold first start is still importing. "
               + "Check again, or stop it and read the notebook log.");
      actions.appendChild(recheckButton());
      if (st.may_control) actions.appendChild(stopButton());
    },
  },
  {
    // (6) Asked for a start, and nothing has come up yet.
    name: "asked",
    when: (st) => startAsked && !st.running,
    enter: () => say("Starting…"),
    pollMs: 1200,
    budgetMs: 15000,
    expired: (st) => {
      startAsked = false;
      // A PORT CLASH IS THE ONE CAUSE THE SERVER CAN NAME, so say it instead
      // of a list of things to go and check.  The notebook port is
      // `serve + 1`, so two molbuilders on adjacent ports collide by
      // construction (`jupyter.port_clash`).
      if (st.port_clash) {
        sayError("Asked for a notebook and none started: " + st.port_clash);
        return;
      }
      // TWO LOGS, AND THEY HOLD DIFFERENT FAILURES.  Everything
      // `_start_jupyter` cannot do -- no argv (a supervisor predating this
      // feature), the log could not be opened, the spawn raised -- is in the
      // SERVE log, because the notebook log is the thing that could not be
      // started.  Everything jupyter-server itself refuses is in the
      // NOTEBOOK log.  Naming only one sent people to the wrong file.
      sayError("Asked for a notebook and none started. The supervisor may "
               + "predate this feature — it survives a code reload, so "
               + "`molbuilder serve stop` then `serve start` gives it one. "
               + "`molbuilder serve status` names the server log; "
               + "`molbuilder jupyter status` names the notebook log, which "
               + "is where jupyter's own refusals are written.");
      actions.appendChild(recheckButton());
    },
  },
  {
    // (7) Nothing running, and nothing that could hold one.
    //
    // NOT `molbuilder jupyter start` -- that verb signals the supervisor, so
    // it is the one command guaranteed to fail in exactly this state, and
    // the CLI's own docstring says so.  `serve start` is the answer.
    name: "unsupervised",
    when: (st) => !st.supervised,
    enter: () => say(
      "JupyterLab is not running, and this molbuilder has no supervisor to "
      + "hold one — it was started with `serve foreground` or "
      + "`--no-supervise`. Restart it with `molbuilder serve start` and the "
      + "button appears here."),
    pollMs: 30000,
  },
  {
    // (8) Nothing running, and this caller may not start one.
    name: "no-control",
    when: (st) => !st.may_control,
    enter: () => say(
      "JupyterLab is not running. Starting one runs code on this machine, "
      + "so it is an admin action — ask whoever administers this server, or "
      + "start it there with `molbuilder jupyter start`."),
    pollMs: 30000,
  },
  {
    // (9) Installed, nothing running, and the person may ask.
    name: "idle",
    when: () => true,
    // A HEARTBEAT, for the reason `framed` got one: four of the ten states
    // declared neither a poll nor a budget, so a notebook started from a
    // TERMINAL left the Start button sitting over a running, answering
    // server until somebody reloaded the page (`plan.md` § 5n.8).  Cheap
    // here -- with nothing running, `status` makes no HTTP call at all.
    pollMs: 15000,
    enter: () => {
      say("Start JupyterLab to write notebooks under your projects tree. Its "
          + "kernel is the notebook env's own python — numpy, scipy, pandas "
          + "and matplotlib. It runs as the account serving this page and "
          + "stops when molbuilder stops.");
      actions.appendChild(button("Start JupyterLab", async () => {
        say("Starting…");
        actions.replaceChildren();
        const res = await post("/api/jupyter/start");
        if (!res.ok) {
          // AND A WAY FORWARD.  The click emptied the action row and
          // `render` had already stopped polling, so this used to leave a
          // red sentence, no button and no timer.
          sayError(`Could not start it: ${res.body.error || res.body.message
                   || ("HTTP " + res.status)}`);
          actions.appendChild(recheckButton());
          return;
        }
        startAsked = true;
        // Leave `asked` a clean entry, so its budget starts now, and let
        // the TABLE say how soon to look -- `pollSoon(1200)` stood here and
        // was a second spelling of `asked`'s own `pollMs`, which is exactly
        // the drift the table was built to end (found in review
        // 2026-09-15).
        currentState = null;
        refresh();
      }, { primary: true }));
    },
  },
];

/** (10) The status endpoint could not be asked.  Not in `STATES` because it
 *  is reached without a payload -- there is nothing to match `when` against.
 *  A HICCUP MUST NOT BE TERMINAL: this used to stop polling, print an error
 *  and leave `actions` as it found it, which on the start path is EMPTY
 *  because the click cleared it.  One dropped fetch mid-start therefore left
 *  no message that helps, no button, no retry and no poll: reload only. */
function unreachable(err) {
  stopPolling();
  currentState = "unreachable";
  actions.replaceChildren();
  cmd.hidden = true;
  wrap.hidden = true;
  sayError("Could not ask this server about the notebook: "
           + (err.message || err));
  actions.appendChild(button("Try again", () => {
    say("Checking…");
    refresh();
  }, { primary: true }));
}

/** A way out of any state that has stopped polling.
 *
 *  EVERY GIVE-UP MUST OFFER ONE (`plan.md` § 5n.8).  `render` returns right
 *  after `expired`, before it schedules anything, so a budget that runs out
 *  used to leave a red sentence and no timer -- and `asked`'s give-up left an
 *  EMPTY action row, because the Start click had cleared it, so after a
 *  failed start there was not even a Start button to press again.  A page
 *  reload was the only way out of four states, and nothing on screen said
 *  so.  `unreachable()` already had this button for exactly this reason; the
 *  other paths did not. */
function recheckButton(label = "Check again") {
  return button(label, () => {
    say("Checking…");
    currentState = null;        // a fresh entry, so budgets start over
    frameUnreachable = false;
    refresh();
  }, { primary: true });
}


function stopButton() {
  return button("Stop JupyterLab", async () => {
    say("Stopping…");
    const res = await post("/api/jupyter/stop");
    if (!res.ok) {
      // REFUSED, so change nothing.  Blanking the frame here tore Lab down
      // -- layout, open documents and any unsaved editor state -- and the
      // next poll then re-framed it from scratch, for a stop that never
      // happened.  Start checked its answer; this did not.
      sayError(`Could not stop it: ${res.body.error || res.body.message
               || ("HTTP " + res.status)}`);
      actions.appendChild(recheckButton());
      return;
    }
    frame.src = "about:blank";
    framedKey = null;
    currentState = null;
    refresh();          // the table decides what comes next, not a number here
  });
}

/** First match wins.  `idle` ends the table with `when: () => true`, so this
 *  always answers -- the table is total by construction, not by a fallback
 *  branch nothing can reach. */
const pick = (st) => STATES.find((s) => s.when(st));

function render(st) {
  actions.replaceChildren();
  cmd.hidden = true;
  stopPolling();
  // NOTHING RUNNING MEANS NOTHING TO SHOW.  Whatever is in the frame now
  // belongs to a server that is gone; blank it here, once, rather than
  // leaving every later branch to remember.
  if (!st.running) {
    if (frame.src && frame.src !== "about:blank") frame.src = "about:blank";
    framedKey = null;
  } else {
    // THE START WORKED, SO STOP COUNTING -- the one place this is cleared.
    startAsked = false;
  }

  const s = pick(st);
  const now = Date.now();
  if (s.name !== currentState) { currentState = s.name; enteredAt = now; }
  const spent = now - enteredAt;

  wrap.hidden = !s.frame;
  if (s.budgetMs && spent >= s.budgetMs) {
    s.expired(st);
    return;                       // a give-up state waits for a person
  }
  s.enter(st);
  if (s.pollMs !== undefined) pollSoon(s.pollMs);
  else if (s.budgetMs) pollSoon(s.budgetMs - spent);
}

async function refresh() {
  try {
    const r = await fetch("/api/jupyter/status");
    const st = await r.json();
    if (!st.ok) throw new Error("status refused");
    render(st);
  } catch (err) {
    unreachable(err);
  }
}

// NO SELECTION SUBSCRIPTION.  One stood here to re-render when the projects
// sidebar's selection moved -- but the sidebar is mounted and NOT SHOWN on
// this tab (`jupyternb.html`), so nothing here can move it, and the only
// thing the re-render offered was a button that could never appear.  The
// selection is read ONCE, when the frame is first pointed.

refresh();
