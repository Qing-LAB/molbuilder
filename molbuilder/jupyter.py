"""The live-notebook process: start it, stop it, and leave nothing behind.

Contract: `docs/web/jupyter.md`.  The shape in one line -- **Flask cannot carry
the kernel**, so the tab is an iframe pointed at a Jupyter server molbuilder
starts and stops, and the WebSocket goes browser -> Jupyter directly.

THE LIFECYCLE IS THE HARD HALF.  Three layers, because no single one covers
every way a parent can die -- and **this module owns the first two.**  The
third is `serve_daemon`'s, deliberately: it is layer 1 and may import nothing
of the application, so it reconciles with its OWN helpers rather than calling
anything here (`serve_daemon.stop_by_pidfile`).  This docstring claimed all
three until 2026-09-14.

1. **`PR_SET_PDEATHSIG`** -- the only mechanism that survives `kill -9` of the
   parent, which runs no handler.  The shepherd asks the kernel to signal IT
   when its parent goes.
2. **A process group** -- so a stop takes the SERVER, not just the shepherd:
   the tree is shepherd -> the manager's `run` -> `jupyter-server`, and
   signalling one of them is not signalling the others.

   **THE GROUP DOES NOT REACH THE KERNELS, and it does not need to.**
   `jupyter_client` launches every kernel with `start_new_session=True`
   (`launcher.py`), so each kernel is its own session and group leader and
   `killpg` never touches it -- measured 2026-09-14, kernel pgid 2152481
   against the shepherd's 2141249.  Jupyter owns that half and owns it
   twice: a graceful SIGTERM makes the server shut its kernels down itself,
   and if the server dies with no handler at all, `ipykernel`'s parent
   poller sees `JPY_PARENT_PID` vanish and the kernel exits on its own.
   Measured the same day by `kill -9` on the server: the kernel was gone in
   about a second, with nothing orphaned.

   So what layer 2 guarantees is that **the server dies**, and the server
   dying is what collects the kernels.  This block claimed the group reached
   the kernels directly; a `/proc`-walking reaper was drafted to make that
   true and then dropped, because it would have been a third copy of a
   backstop Jupyter already provides and passes.
3. **Reconciliation at startup** -- a machine crash, and a survivor that was
   re-parented before the signal landed, are outside layers 1 and 2.  The next
   `supervise()` reads the pidfile and, only if the pid is alive and really
   is a shepherd of ours, stops it.  **`serve start` ONLY** -- this said
   "AND `serve foreground`" until 2026-09-15 and that was false: `supervise`
   has one call site, `cli.cmd_serve_start`, and `serve foreground` goes to
   `cli._supervise_forever`, which reconciles nothing.  A notebook that
   outlived a crash is collected by the next `serve start`, and a
   `serve foreground` walks past it.
   Same rule `serve_daemon` states: a stale file whose pid was recycled is
   REPORTED stale, never signalled.  **That code is in `serve_daemon`, not
   here** -- see above.

**PARENTED TO THE SUPERVISOR, not to the server child** (`jupyter.md` § 3.4 --
the user's decision against two alternatives, reaffirmed 2026-09-14).  The
supervisor respawns the server child on `RELOAD_EXIT_CODE`, so a Jupyter
parented there would be killed by every unrelated code reload, destroying
notebook state for a change that had nothing to do with it.  One level up it
survives a reload and dies with the daemon.

**WHY A SHEPHERD PROCESS AT ALL**, rather than the supervisor running
``<mgr> run -n molbuilder-jupyternb jupyter lab`` itself:

* the supervisor **imports nothing of the application** -- that is the whole
  reason a child which fails to import leaves the parent alive -- so it cannot
  use `envs.builds.dispatch_into_env`, the one door into an env.  The shepherd
  is an ordinary molbuilder process and can.
* the supervisor must be able to stop the notebook and every kernel under it
  with one signal.  It starts the shepherd in its own SESSION
  (``start_new_session=True``), which makes the shepherd a process-group
  leader; the manager's ``run`` and ``jupyter-server`` inherit that group.
  So the shepherd stops them by signalling its OWN group, and no part of
  this needs the door to hand back a pid.  (The KERNELS are not in that
  group -- see layer 2 above; Jupyter collects those itself.)

So the supervisor only ever launches an argv it was handed, and this module is
what that argv runs.
"""
from __future__ import annotations

import ctypes
import os
import secrets
import signal
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

#: What the kernel is asked to send the shepherd when its parent dies.
#: Linux-only, which matches this project's stated platform.
_PR_SET_PDEATHSIG = 1

#: How long a stop waits for the group to go before it stops asking politely.
#: A kernel mid-cell gets a moment to unwind; a wedged one does not get to
#: hold a GPU indefinitely.  It sits inside the supervisor's own 10 s wait for
#: the shepherd (`serve_daemon._stop_jupyter`), so the polite phase always
#: finishes first there.  It must also stay strictly BELOW
#: `serve_daemon.stop_by_pidfile`'s own grace, which reconciliation uses: the
#: two were both 5.0, so the reconciliation poll timed out at the same instant
#: the shepherd would have exited and reported "(forced)" for every perfectly
#: polite stop.  **Used by `_stop_own_group`** -- it was a constant nothing
#: read until 2026-09-14, while the code slept a hardcoded 1.0 beside it.
_STOP_GRACE_S = 4.0

#: The schema this module's data file must stamp itself with.  Gated by
#: `persist.check_schema`, the one enforcement point for the convention --
#: the same posture `warm-files.toml` has.
SCHEMA = "molbuilder/jupyter@1"


def jupyter_port(serve_port: int) -> int:
    """The port the notebook listens on for a molbuilder serving on
    ``serve_port``.

    **Derived, not configured.**  One number to remember, and the pair moves
    together when a second molbuilder runs on another port -- the case a fixed
    default would break.
    """
    return serve_port + 1


def serve_port_of(notebook_port: int) -> int:
    """The serve port a notebook port belongs to -- `jupyter_port` inverted.

    One home for the pairing, in both directions.  The CSP grant in
    `notebook_argv` needs to name molbuilder's own port and had its own
    `port - 1` until 2026-09-14; a security header is the last place two
    copies of one derivation should be allowed to disagree.
    """
    return notebook_port - 1


def port_clash(serve_port: int) -> Optional[str]:
    """Is this molbuilder's notebook port already claimed?  A sentence, or
    ``None``.

    **TWO MOLBUILDERS ON ADJACENT PORTS COLLIDE BY CONSTRUCTION.**
    `jupyter_port` is ``p + 1``, so a molbuilder on 6006 wants 6007 for its
    notebook while a molbuilder on 6007 wants 6007 for its WEB server.
    Whichever starts second loses, and which one that is decides the symptom:
    the notebook refuses to start, or `serve start` cannot bind.

    `--ServerApp.port_retries=0` already makes that LOUD rather than silent --
    measured 2026-09-15 against a live notebook on 6007: *"ERROR: the Jupyter
    server could not be started because port 6007 is not available"*, exit 1.
    Without it jupyter-server picks a random free port out of fifty while the
    runtime file, the `frame-ancestors` grant and `answering()` all keep
    naming ``p + 1``, and the tab reports "wedged" over a healthy notebook.

    So the gap this closes is not the crash; it is that **nothing said so in
    advance when it was knowable**.  Two questions, in order of how much they
    can explain:

    1. is the notebook port another molbuilder's SERVE port?  Attributable
       from the pidfiles alone, and it names the other server.
    2. is anything at all listening there?  A bind test -- generic, and the
       answer is only *"something"*.

    **NOT a change to the derivation** (`plan.md` § 5n, J15).  ``p + 1`` is
    one number to remember and the pairing has one home in both directions;
    ``p + 1000`` moves the collision without removing it, and letting Jupyter
    choose freely makes the runtime file authoritative for a port that four
    other places derive.
    """
    import socket

    from .config_dir import ports_with_pidfile
    from .serve_daemon import pid_state as serve_pid_state, read_pid as serve_pid
    nb = jupyter_port(serve_port)

    # DIRECTION 2 FIRST, because it is the one that stops `serve start` dead.
    # Is OUR OWN web port some other molbuilder's NOTEBOOK port?  Then this
    # server cannot bind at all, and the failure lands in the serve log after
    # the terminal is gone.  The docstring named both directions from the
    # start and the code asked only the first (found in review 2026-09-15).
    for other in ports_with_pidfile("jupyter"):
        if (jupyter_port(other) == serve_port
                and pid_state(read_pid(other)) == "ours"):
            return (f"port {serve_port} is already the NOTEBOOK port of the "
                    f"molbuilder serving on {other} ({other} + 1).  This "
                    f"server cannot bind it while that notebook runs.  "
                    f"Choose a port that is not adjacent to another "
                    f"molbuilder.")

    # OUR OWN notebook already holding it is not a clash, it is the notebook.
    if pid_state(read_pid(serve_port)) == "ours":
        return None

    # DIRECTION 1: is our notebook's port another molbuilder's WEB port?
    # Asked directly rather than by scanning the runtime directory -- the
    # old loop could only ever fire on `other == nb`, so the glob answered a
    # question `read_pid(nb)` answers on its own (a missing pidfile reads as
    # "dead", which is exactly what the scan was looking for).
    if serve_pid_state(serve_pid(nb)) == "ours":
        return (f"port {nb} is the WEB port of another molbuilder, and it is "
                f"the port this one's notebook needs ({serve_port} + 1).  "
                f"The notebook will refuse to start while that server runs.  "
                f"Serve this molbuilder on a port that is not adjacent to "
                f"another.")
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind(("127.0.0.1", nb))
    except OSError:
        return (f"something is already listening on port {nb}, which is the "
                f"port this molbuilder's notebook needs ({serve_port} + 1).  "
                f"The notebook will refuse to start until that port is free.")
    return None


def shepherd_argv(serve_port: int, *, host: str,
                  cert: Optional[str] = None,
                  key: Optional[str] = None) -> List[str]:
    """The command line the SUPERVISOR launches, spelled once.

    Built by the `serve` verb and handed to `supervise`, exactly as the server
    child's own argv is -- so the supervisor holds a list of strings and never
    imports what runs at the other end.
    """
    argv = [sys.executable, "-m", "molbuilder", "jupyter", "_shepherd",
            "--serve-port", str(serve_port), "--host", host]
    if cert:
        argv += ["--cert", cert]
    if key:
        argv += ["--key", key]
    return argv


# WHERE THE FILES ARE: `config_dir`, asked directly.
#
# `pid_path`, `log_path` and `runtime_path` stood here as one-line forwards
# to `jupyter_pidfile` / `jupyter_log` / `jupyter_runtime` until 2026-09-15
# (`plan.md` § 5n, J17).  That is the pattern `serve_daemon` deleted as K-D3
# -- its own import comment says so: *"Four one-line pass-throughs … stood
# between this module and those doors until 2026-09-13."*
#
# They were worse than plain forwards, because they RENAMED: the notebook
# log was `jupyter_log` where `serve_daemon` opened it and `log_path` where
# `cli` printed it, so grepping for either name found half the callers.  Two
# of the three had no caller outside this module at all.


# --------------------------------------------------------------------- #
#  Reading the state -- verify before you signal                         #
# --------------------------------------------------------------------- #

def read_pid(serve_port: int) -> Optional[int]:
    """The shepherd's pid for ``serve_port``, or ``None``.

    **`serve_daemon`'s reader, at this module's address** -- the same
    delegation `pid_state` below already makes, and for the same reason: two
    copies of "what does this pidfile say" is how one of them comes to
    treat a corrupt file differently from the other (J16).
    """
    from .config_dir import jupyter_pidfile
    from .serve_daemon import read_pidfile
    return read_pidfile(jupyter_pidfile(serve_port))


#: What a shepherd's command line carries, beside ``molbuilder`` -- the word
#: `serve_daemon.pid_state` checks for before it signals anything.
_MARKER = b"_shepherd"


def pid_state(pid: Optional[int]) -> str:
    """``"ours"`` | ``"foreign"`` | ``"dead"`` -- what that pid actually is.

    **`serve_daemon`'s, with this module's marker.**  That module owns
    "verify a pid before signalling it" for the supervisor's own pidfile, and
    a second copy here is how one of the two comes to be laxer than the other.
    The reason is its reason: the kernel would refuse a cross-user signal
    anyway (EPERM), but *"that pid is not a notebook of yours"* beats
    *"permission denied"*, and a recycled pid must never be signalled.
    """
    from .serve_daemon import pid_state as _verify
    return _verify(pid, marker=_MARKER)


def read_runtime(serve_port: int) -> Dict[str, object]:
    """``{"url": ..., "base": ..., "token": ...}``, or ``{}``.

    *(It advertised ``port`` -- which nothing read, `status` computes it from
    `jupyter_port` -- and omitted ``base``, which `open_notebooks` does read.
    The docstring named the dead key and hid the live one, found in review
    2026-09-15; the key is gone with it.)*

    The token is a CREDENTIAL -- it authenticates a browser to a live kernel --
    so this is read server-side and handed to the page, never published.
    """
    import json

    from .config_dir import jupyter_runtime
    try:
        doc = json.loads(
            jupyter_runtime(serve_port).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return doc if isinstance(doc, dict) else {}


def answering(serve_port: int, *, timeout: float = 1.5) -> bool:
    """Is something listening on the notebook port and talking HTTP?

    The SECOND question (`deployment.md` § 1.0b's rule): a process that is up
    and not answering is exactly the failure a single word would call healthy.
    TLS is not verified here -- molbuilder gave Jupyter the certificate, and
    what is being asked is *"is it answering"*, not *"is it trusted"*.
    """
    import urllib.error
    import urllib.request

    from .serve_daemon import unverified_ctx
    runtime = read_runtime(serve_port)
    url = str(runtime.get("url") or "")
    if not url:
        return False
    try:
        with urllib.request.urlopen(url, timeout=timeout,
                                    context=unverified_ctx()):
            return True
    except urllib.error.HTTPError:
        # It answered -- with a refusal, because no token was presented.
        # That is a server that is up, which is the question.
        return True
    except Exception:  # noqa: BLE001 - not answering is the answer
        return False


def open_notebooks(serve_port: int, *,
                   timeout: float = 1.5) -> List[Dict[str, object]]:
    """Every notebook the running Jupyter has open, and the PATH it will save
    to -- relative to the projects root, which is Jupyter's own root_dir.

    **Asked of the server, not inferred from the tab.**  Which folder a new
    notebook lands in is decided inside Lab (its file browser's directory),
    somewhere molbuilder cannot see and should not guess at.  Jupyter answers
    it exactly, in `GET /api/sessions`, so the tab can state where each open
    notebook lives instead of describing where it *ought* to be.

    Empty on any failure: this decorates the control row and must never be the
    reason it cannot render.
    """
    import json
    import urllib.request

    from .serve_daemon import unverified_ctx
    runtime = read_runtime(serve_port)
    base  = str(runtime.get("base") or "")
    token = str(runtime.get("token") or "")
    if not base or not token:
        return []
    req = urllib.request.Request(
        f"{base}/api/sessions",
        headers={"Authorization": f"token {token}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout,
                                    context=unverified_ctx()) as r:
            sessions = json.loads(r.read().decode("utf-8"))
    except Exception:  # noqa: BLE001 - a decoration, never a failure
        return []
    out: List[Dict[str, object]] = []
    for s in sessions if isinstance(sessions, list) else []:
        if not isinstance(s, dict) or s.get("type") != "notebook":
            continue
        kernel = s.get("kernel") if isinstance(s.get("kernel"), dict) else {}
        out.append({
            "path":  str(s.get("path") or ""),
            "state": str((kernel or {}).get("execution_state") or ""),
        })
    return sorted(out, key=lambda n: str(n["path"]))


def status(serve_port: int, *, include_private: bool = True
           ) -> Dict[str, object]:
    """**Two questions, answered separately**, for `deployment.md` § 1.0b's
    reason: the wedge worth catching is up-and-not-answering.

    `include_private=False` withholds the two things a caller who may not
    CONTROL the notebook has no business with: the token, which authenticates
    a browser to a live kernel, and the paths of the notebooks currently open,
    which are somebody's project tree.  Withholding them here rather than
    deleting them afterwards is the difference between a credential that was
    never produced and one whose safety depends on a later statement in a
    function that will grow.
    """
    pid = read_pid(serve_port)
    state = pid_state(pid)
    running = state == "ours"
    runtime = read_runtime(serve_port) if running else {}
    up = answering(serve_port) if running else False
    return {
        "running": running,
        "pid": pid if running else None,
        "pid_state": state,
        "port": jupyter_port(serve_port),
        "answering": up,
        "url": str(runtime.get("url") or ""),
        "token": (str(runtime.get("token") or "")
                  if include_private else ""),
        # Only when there is something to ask: one HTTP call, and asking a
        # server that is not answering is a timeout on every poll.
        "open": (open_notebooks(serve_port)
                 if (up and include_private) else []),
        # IS THERE A LAYOUT TO RESTORE?  What lets the tab point the frame
        # correctly after a switch away and back -- see `workspace_saved`.
        # Not private: it is one boolean about this machine's own Lab, and
        # the tab needs it in exactly the state where it may frame.
        "workspace_saved": (workspace_saved(serve_port)
                            if running else False),
    }


def _forget(serve_port: int) -> None:
    from .config_dir import jupyter_pidfile, jupyter_runtime
    for p in (jupyter_pidfile(serve_port),
              jupyter_runtime(serve_port)):
        try:
            p.unlink()
        except OSError:
            pass

def _set_pdeathsig() -> bool:
    """Ask the kernel to send SIGTERM when this process's parent dies.

    Layer 1, and the only one that survives `kill -9` of the parent -- a
    SIGKILLed process runs no handler, so nothing it "would have done on the
    way out" happens.  Best-effort: a platform without `prctl` keeps layers 2
    and 3, and `False` says which machine this is.
    """
    try:
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        return libc.prctl(_PR_SET_PDEATHSIG, signal.SIGTERM, 0, 0, 0) == 0
    except Exception:  # noqa: BLE001 - not Linux, or no libc by that name
        return False


def _stop_own_group() -> None:
    """Signal this process's whole group: the shepherd, the manager's ``run``,
    and ``jupyter-server``.

    **NOT the kernels**, and that is not a gap.  `jupyter_client` starts every
    kernel with ``start_new_session=True``, so each is its own group leader
    and no ``killpg`` of ours reaches it.  What this guarantees is that the
    SERVER dies; Jupyter then collects its own kernels two ways -- the
    server's SIGTERM handler shuts them down gracefully, and if the server is
    SIGKILLed instead, `ipykernel`'s parent poller sees the parent go and each
    kernel exits by itself.  Both measured 2026-09-14; a `kill -9` on the
    server left no kernel behind after about a second.

    SIGTERM is ignored in this process first, or the signal we are about to
    send would re-enter this handler.
    """
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    try:
        os.killpg(os.getpgrp(), signal.SIGTERM)
    except OSError:
        pass
    time.sleep(_STOP_GRACE_S)
    try:
        os.killpg(os.getpgrp(), signal.SIGKILL)
    except OSError:
        pass


# --------------------------------------------------------------------- #
#  The settings, as DATA -- `data/jupyter.toml` and the one reader        #
# --------------------------------------------------------------------- #

class JupyterRulesError(Exception):
    """`data/jupyter.toml` is missing, malformed, or names nothing usable."""


#: The sections the file may carry, closed by contract.  A typo in a settings
#: file otherwise disables a setting in silence, which is the failure this
#: whole table exists to end -- so an unknown section is refused BY NAMING
#: THE ONES THAT EXIST, `warmfiles`' own refusal style.
_SECTIONS = ("schema", "server", "lab")
_LAB_SECTIONS = ("home", "config", "overrides")


@dataclass(frozen=True)
class JupyterRules:
    """What `data/jupyter.toml` declares, parsed once.

    ``server`` and ``lab_config``/``lab_home`` are keyed by the traitlets
    OPTION each answers (``ServerApp.port_retries``,
    ``LabApp.workspaces_dir``), because that key is what reaches the command
    line -- so adding a setting is a row here and nothing in the code.
    """
    server: Dict[str, object]
    lab_home: Dict[str, str]
    lab_config: Dict[str, str]
    overrides: Dict[str, Dict[str, object]]
    path: str


def rules_path() -> Path:
    """Where the table lives.  Shipped by `pyproject.toml`'s ``data/*.toml``."""
    return Path(__file__).resolve().parent / "data" / "jupyter.toml"


def load_rules() -> JupyterRules:
    """Read and validate the settings table.  THE ONE READER.

    The admission rule the file states is not enforceable from here -- *"is
    this value knowable before the process starts"* is a question about the
    value, not its shape -- so what this checks is the shape: the stamp, the
    sections, and that every row is an option with a scalar value.  The rule
    itself is kept by review, and by the fact that a runtime value has
    nowhere in the file to come from.
    """
    import tomllib

    from .persist import check_schema
    path = rules_path()
    try:
        with open(path, "rb") as fh:
            raw = tomllib.load(fh)
    except OSError as exc:
        raise JupyterRulesError(
            f"cannot read the notebook settings table at {path}: {exc}. "
            f"It ships with molbuilder (`pyproject.toml`, `data/*.toml`); an "
            f"installed copy missing it was built by an older setuptools."
        ) from exc
    except tomllib.TOMLDecodeError as exc:
        raise JupyterRulesError(f"{path}: not valid TOML -- {exc}") from exc

    check_schema(str(raw.get("schema", "")), SCHEMA, label=str(path))
    unknown = [k for k in raw if k not in _SECTIONS]
    if unknown:
        raise JupyterRulesError(
            f"{path}: unknown section(s) {', '.join(sorted(unknown))} -- this "
            f"file carries {', '.join(_SECTIONS)} and nothing else.")
    lab = raw.get("lab") or {}
    if not isinstance(lab, dict):
        raise JupyterRulesError(f"{path}: [lab] must be a table.")
    unknown = [k for k in lab if k not in _LAB_SECTIONS]
    if unknown:
        raise JupyterRulesError(
            f"{path}: unknown [lab.{unknown[0]}] -- [lab] carries "
            f"{', '.join('lab.' + s for s in _LAB_SECTIONS)} and nothing else.")

    def _flat(section: str, body: object) -> Dict[str, object]:
        if body is None:
            return {}
        if not isinstance(body, dict):
            raise JupyterRulesError(f"{path}: [{section}] must be a table.")
        for key, val in body.items():
            if "." not in key:
                raise JupyterRulesError(
                    f"{path}: [{section}] key {key!r} is not a traitlets "
                    f"option -- every row here is `Class.trait`, because the "
                    f"key IS what reaches the command line.")
            if isinstance(val, (dict, list)):
                raise JupyterRulesError(
                    f"{path}: [{section}] {key} holds a {type(val).__name__}; "
                    f"a command-line option carries one scalar.")
        return dict(body)

    overrides = lab.get("overrides") or {}
    if not isinstance(overrides, dict) or not all(
            isinstance(v, dict) for v in overrides.values()):
        raise JupyterRulesError(
            f"{path}: [lab.overrides] is one table per PLUGIN ID, each "
            f"holding that plugin's settings.")
    return JupyterRules(
        server=_flat("server", raw.get("server")),
        lab_home=_flat("lab.home", lab.get("home")),
        lab_config=_flat("lab.config", lab.get("config")),
        overrides={str(k): dict(v) for k, v in overrides.items()},
        path=str(path),
    )


def option_argv(settings: Dict[str, object]) -> List[str]:
    """``{trait: value}`` -> ``--trait=value``, sorted.  THE ONE EMITTER.

    Every Jupyter setting molbuilder states goes out through here, whether it
    was authored in `data/jupyter.toml` or computed from a runtime fact -- so
    there is one spelling of an option on the command line and one place a
    row can go missing from.  Sorted because the argv is compared in tests and
    read in logs, and a dict's order is not a fact about the settings.

    Booleans are spelled the way traitlets parses them (`True` / `False`),
    which is Python's own `repr` and not TOML's lower case.
    """
    out: List[str] = []
    for key in sorted(settings):
        val = settings[key]
        out.append(f"--{key}={val!r}" if isinstance(val, bool)
                   else f"--{key}={val}")
    return out


#: The `[lab.home]` option whose directory holds Lab's saved WORKSPACE --
#: which documents were open, which panel was showing.  Named once, because
#: two things key off it: the wipe at every start, and the flag that tells
#: the tab whether there is anything to restore.
_WORKSPACE_OPT = "LabApp.workspaces_dir"


def _workspace_dir(serve_port: int) -> Optional[Path]:
    """Where Lab saves THIS server's workspace, or ``None`` if none declared.

    **PER SERVER, under a Lab home that is otherwise shared** -- found in
    review 2026-09-15, hours after J13 shipped, and it re-created the very
    bug J13 was written for.

    `config_dir.jupyter_lab_home()` is deliberately not port-keyed, and its
    reason was sound while the directory held only DEFAULTS: *"the defaults
    do not differ between servers."*  J13 then put per-server SESSION state
    -- the saved workspace, emptied at every notebook start -- into that
    same shared directory, and the reason stopped covering it.  Two
    molbuilders and it fails in both directions: starting B's notebook wipes
    A's layout, so A's next tab switch loses the notebook whose kernel is
    still running (J13, verbatim); and B's saved workspace makes A's first
    framing report `workspace_saved` true, so A opens at the projects root
    instead of the selected folder -- the `projects/Untitled.ipynb` failure
    state 3 exists to prevent.  Both Labs also wrote the same
    `default.jupyterlab-workspace`, so A could restore B's layout.

    Only the workspace moves.  `settings/` and `user-settings/` stay shared,
    because the original reasoning is still exactly right for them: the
    defaults do not differ between servers, and a person's own change inside
    Lab should follow them to whichever molbuilder they open next.
    """
    from .config_dir import jupyter_lab_home
    name = load_rules().lab_home.get(_WORKSPACE_OPT)
    return (jupyter_lab_home() / name / str(serve_port)) if name else None


def workspace_saved(serve_port: int) -> bool:
    """Has the framed Lab saved a workspace since THIS server started?

    **This is what makes a tab switch keep your notebook open** without any
    browser state at all (`plan.md` § 5n, J13, the user's own ask).  Leaving
    `/jupyternb` destroys the iframe, so coming back always re-points it --
    that is a page navigation and cannot be avoided in the tab.  What decides
    whether Lab comes back where you left it is its WORKSPACE, and the tab
    needs to know one thing to point correctly: is there one?

    * **No** -- this is the first framing since the server started, so the
      URL carries the folder the projects sidebar has selected and Lab opens
      there.
    * **Yes** -- somebody has been working in it, so the URL carries no tree
      path and Lab restores what was open.

    Asking the SERVER rather than remembering in the browser is the whole
    trick: the wipe at `prepare_lab_home` and this flag are the same fact
    read from the same directory, so they cannot disagree, and nothing has to
    survive a page load.
    """
    d = _workspace_dir(serve_port)
    try:
        return d is not None and any(d.iterdir())
    except OSError:
        return False


def _forget_workspace(home: Path, rules: "JupyterRules",
                      serve_port: int) -> None:
    """Empty Lab's workspace directory.  Called at every shepherd start.

    **CLEAN ON A NEW SERVER, PERSISTENT WHILE ONE RUNS** -- the user's rule,
    2026-09-15: *"we would like to have this persistent when switching tab,
    but do not need this when server get shutdown and restarted."*

    This SUPERSEDES the decision taken with the user on 2026-09-14, which put
    Jupyter's `?reset` on every page load (`jupyter.md` § 4.1).  That reset
    was aimed at a real problem -- a restored workspace argued with the
    folder the projects sidebar had selected -- but it was applied at the
    wrong moment: every LOAD, when what was meant was every START.  A tab
    switch is a load, so switching away and back threw away the notebook you
    had open while its kernel was still running.

    Doing it here instead of in the browser is what makes the two halves
    impossible to disagree: one directory, emptied by the process that owns
    it, and `workspace_saved()` reading the same directory back.

    Only the CONTENTS go, and only under the Lab home -- `user-settings/` is
    a person's own and is never touched.
    """
    name = rules.lab_home.get(_WORKSPACE_OPT)
    if not name:
        return
    # THIS SERVER'S workspace only -- see `_workspace_dir`.  Emptying the
    # shared `workspaces/` would take every other molbuilder's layout with
    # it, which is the bug this whole mechanism exists to fix.
    ws = home / name / str(serve_port)
    # A DELETE, SO SAY EXACTLY WHAT THIS GUARD DOES.  `name` comes from a
    # data file; this refuses a name that ESCAPES the Lab home -- a
    # separator, a `..`, an absolute path.
    #
    # **RESOLVED, because the unresolved form let the one spelling this
    # comment named walk straight through** (found in review 2026-09-15,
    # hours after it was written).  `home / ".."` is a real directory whose
    # `.parent` IS `home`, so `ws.parent != home` was False and the loop
    # below would have emptied the Lab home's PARENT -- the whole state
    # directory: logs, reports, run/.  `../../../escape` was refused and
    # `..` was not, and the comment claimed both.
    #
    # It still does NOT catch a wrong but well-formed name: point the row at
    # `user-settings` and that is what gets emptied.  What catches THAT is
    # the test asserting `user-settings/` survives a start, and it is the
    # only thing that does.
    if not ws.is_dir() or ws.resolve().parent.parent != home.resolve():
        return
    import shutil as _shutil
    for child in ws.iterdir():
        try:
            if child.is_dir() and not child.is_symlink():
                _shutil.rmtree(child)
            else:
                child.unlink()
        except OSError:
            pass            # a layout we could not remove is not a reason
                            # to refuse to start a notebook


def prepare_lab_home(serve_port: int) -> Dict[str, str]:
    """Prepare the framed Lab's own directories and config; return every
    generated path keyed by the command-line option it answers.

    **WHAT IS HERE IS `data/jupyter.toml`'s.**  The directory names, the
    config file's name and every override are rows in that table; this
    function turns them into paths under `config_dir.jupyter_lab_home()` and
    writes the two generated files.  Adding a directory is a row there and
    nothing here.

    Separate from ``~/.jupyter`` for one reason: **the Lab in this frame is
    molbuilder's, not the one the person runs themselves.**  Sharing them let
    each write over the other -- Lab saves a user setting the first time it
    resolves one, and a user setting beats an override, so the framed Lab
    adopted whatever the standalone Lab had written and ignored every default
    here (measured 2026-09-14: the theme stayed light).

    The overrides file and the server config are rewritten at every start, so
    the answer to *"why does my framed Lab look like this"* is always the
    table and never a stale file.  The other directories are only created;
    what Lab saves into them is the person's.
    """
    import json
    import shutil
    from .config_dir import ensure_private_dir, jupyter_lab_home
    rules = load_rules()
    home = ensure_private_dir(jupyter_lab_home())
    _forget_workspace(home, rules, serve_port)
    out = {}
    for opt, name in rules.lab_home.items():
        # THE WORKSPACE IS THIS SERVER'S; everything else is shared
        # (`_workspace_dir` says why, and what it cost to find out).
        d = home / name / str(serve_port) if opt == _WORKSPACE_OPT \
            else home / name
        out[opt] = str(ensure_private_dir(d))

    app_settings = rules.lab_home.get("LabApp.app_settings_dir")
    if app_settings is None:
        raise JupyterRulesError(
            f"{rules.path}: [lab.home] declares no "
            f"`LabApp.app_settings_dir`, and that is where the overrides go.")
    (home / app_settings / "overrides.json").write_text(
        json.dumps(rules.overrides, indent=2) + "\n", encoding="utf-8")

    # THE CONFIG IS A FILE IN THE PACKAGE, COPIED -- never a string literal
    # here.  `data/jupyter_server_config.py`'s own docstring says why it is a
    # file under `data/` rather than a module inlined with `inspect.getsource`
    # (it must subclass jupyter_server's `AsyncCheckpoints`, which the HOST
    # env does not have).  An ABSOLUTE `config_file` is then loaded INSTEAD of
    # searching the config path, so the framed server does not read a personal
    # `~/.jupyter/jupyter_server_config.py` either.
    for opt, name in rules.lab_config.items():
        src = rules_path().parent / name
        if not src.is_file():
            raise JupyterRulesError(
                f"{rules.path}: [lab.config] names {name!r}, which is not in "
                f"{src.parent}.  It ships with molbuilder (`pyproject.toml`, "
                f"`data/*.py`).")
        dst = home / name
        shutil.copyfile(src, dst)
        out[opt] = str(dst)
    return out


def notebook_argv(conda: str, env_name: str, *, host: str, port: int,
                  root_dir: str,
                  cert: Optional[str], key: Optional[str],
                  lab_dirs: Optional[Dict[str, str]] = None) -> List[str]:
    """The ``jupyter lab`` command line, as the door will carry it.

    Every setting here is Jupyter's own (`jupyter.md` § 4): molbuilder states
    what it needs and re-implements none of it.  **One map, one emitter** --
    the authored rows from `data/jupyter.toml` and the computed rows below go
    out through `option_argv` together, because they are the same kind of
    fact and were three different mechanisms until 2026-09-15.

    The computed half is here because each row needs something only this
    process knows, which is the file's own admission rule read backwards:

    **The token is not emitted here** *(2026-09-20)*.  It rides
    ``JUPYTER_TOKEN`` in the child's environment (`start_notebook`), because
    ``/proc/<pid>/cmdline`` is world-readable and ``/proc/<pid>/environ`` is
    not.  jupyter-server reads that variable through its own ``token``
    default, so the flag has to be ABSENT rather than duplicated -- an
    explicit trait would win over the default and the environment would be
    ignored.
    * ``tornado_settings`` -- the ``frame-ancestors`` grant.  Jupyter refuses
      to be framed by default and the tab is an iframe (§ 2).
    * ``ip`` / ``port`` / ``root_dir`` -- the bind address, the derived port,
      the projects tree.
    * ``certfile`` / ``keyfile`` -- only when `serve` itself has TLS.
    * ``lab_dirs`` -- what `prepare_lab_home` generated, keyed by the option
      each answers, so a new generated path is a row in the table and
      nothing here.
    """
    import json
    from .envs.builds import conda_run_argv
    # WHO MAY FRAME THIS.  Jupyter refuses framing by default (§ 2), and the
    # parent is molbuilder's page -- whose ORIGIN is whatever the person
    # reached it at, which this process cannot know: it knows the address it
    # BOUND to, and a browser may arrive through a tunnel or a name.  So the
    # grant is bounded by PORT rather than by host -- molbuilder's own serve
    # port, both schemes.  An attacker would have to be serving on that exact
    # port of some host the victim visits, and the notebook still refuses
    # every request without the token.  Naming a single host here instead
    # broke the frame for every browser that was not on this machine.
    # THE INVERSE OF `jupyter_port`, asked of it rather than re-derived.  A
    # second `port - 1` here would let the two drift the day the derivation
    # changes, and the drift would land in a SECURITY header -- the grant
    # would silently name a port that is not molbuilder's.
    serve_port = serve_port_of(port)
    frame_ancestors = (f"frame-ancestors 'self' "
                       f"http://*:{serve_port} https://*:{serve_port}")
    settings: Dict[str, object] = dict(load_rules().server)
    settings.update({
        "ServerApp.ip": host,
        "ServerApp.port": port,
        "ServerApp.root_dir": root_dir,
        "ServerApp.tornado_settings": json.dumps(
            {"headers": {"Content-Security-Policy": frame_ancestors}}),
    })
    settings.update(lab_dirs or {})
    if cert and key:
        settings["ServerApp.certfile"] = cert
        settings["ServerApp.keyfile"] = key
    inner = ["jupyter", "lab", "--no-browser", *option_argv(settings)]
    return list(conda_run_argv(conda, env_name, *inner))


def run_shepherd(serve_port: int, *, host: str,
                 cert: Optional[str] = None,
                 key: Optional[str] = None) -> int:
    """Be the notebook's parent: start it through the door, and take the
    whole group down when told.

    Runs in the foreground of its own process; the SUPERVISOR is what puts it
    in the background, and what it is parented to.
    """
    from .diagnostics import get_capabilities
    from .envs.builds import dispatch_into_env
    from .envs.recipes import effective_name, recipe_by_name
    from .persist import write_json
    from .config_dir import (PRIVATE_FILE_MODE, ensure_private_dir,
                             jupyter_pidfile, jupyter_runtime)
    from .projects import projects_root

    _set_pdeathsig()
    if os.getppid() == 1:
        # The parent died between the fork and the prctl above -- the window
        # layer 1 cannot cover from inside.  Nothing to shepherd.
        return 1

    caps = get_capabilities()
    if caps.conda_binary is None:
        sys.stderr.write(
            "molbuilder jupyter: no env manager found, so the notebook env "
            "cannot be entered.  Record one as `envs.manager` in "
            "molbuilder.json, or activate one before starting the server.\n")
        return 2
    recipe = recipe_by_name("molbuilder-jupyternb")
    env_name = effective_name(recipe, caps)
    if not caps.env_available(env_name):
        # THE REMEDY HAS ONE HOME (`envs.hints`), and this hand-copied it.
        # That module's own docstring records the incident it was created by:
        # `recipes.py` copied `_cli._fix_cmd`'s output, the copy drifted to a
        # recipe name `recipe_by_name` does not accept, and the remedy printed
        # was itself a usage error.  This copy had the same two seeds -- a
        # hardcoded recipe name beside an `{env_name}` config may have
        # renamed.  `hints` is stdlib-only and floor 1, so there was never a
        # layering reason not to call it.
        from .envs.hints import fix_cmd
        sys.stderr.write(
            f"molbuilder jupyter: env `{env_name}` is not installed.  It is "
            f"opt-in: `{fix_cmd('install', recipe.name, '--yes')}`.\n")
        return 2

    port = jupyter_port(serve_port)
    token = secrets.token_urlsafe(32)
    # EVERYTHING THAT CAN RAISE HAPPENS BEFORE THE PIDFILE EXISTS.
    # `prepare_lab_home` creates three directories and writes two files; a
    # read-only or full state dir raised out of `run_shepherd` with the
    # pidfile and the 0600 runtime file already on disk, because it sat
    # between the write and the `try`.  The next `serve start` then reported
    # a stale pidfile for a notebook that had never run.
    lab_dirs = prepare_lab_home(serve_port)
    # AND THE REST OF IT.  `projects_root()` reads `molbuilder.json` and
    # `notebook_argv` reads `data/jupyter.toml` (through `load_rules`, whose
    # own error message anticipates a wheel that shipped without it) -- both
    # can raise, and both sat BELOW the pidfile write until 2026-09-15 while
    # the comment above claimed everything that can raise happens first.
    # Hoisting `prepare_lab_home` had fixed the measured case and left the
    # invariant false (found in review 2026-09-15).
    argv = notebook_argv(caps.conda_binary, env_name, host=host, port=port,
                         root_dir=str(projects_root()),
                         cert=cert, key=key, lab_dirs=lab_dirs)
    scheme = "https" if (cert and key) else "http"
    # BRACKET AN IPv6 LITERAL.  `--host ::1` produced `http://::1:8001/lab`,
    # which `urlopen` cannot parse -- so `answering()` reported down forever
    # and the tab stuck on "Starting...".  (The browser was unaffected: the
    # tab rebuilds the base from `location.hostname`.)
    _hostpart = f"[{host}]" if ":" in host else host
    ensure_private_dir(jupyter_pidfile(serve_port).parent, tighten=True)
    jupyter_pidfile(serve_port).write_text(f"{os.getpid()}\n")
    # 0600: the token authenticates a browser to a live kernel.
    write_json(jupyter_runtime(serve_port),
               {"url": f"{scheme}://{_hostpart}:{port}/lab",
                "base": f"{scheme}://{_hostpart}:{port}",
                "token": token},
               mode=PRIVATE_FILE_MODE)

    def _on_stop(*_):
        # REMOVE OUR FILES BEFORE WE DIE.  `_stop_own_group` ends by
        # SIGKILLing our own process group -- which contains us -- so nothing
        # after it runs and `run_shepherd`'s `finally` never fires on this
        # path.  That is the ONLY path `jupyter stop`, `serve stop` and
        # supervisor exit take, so every clean stop used to leave the pidfile
        # and the 0600 token file behind, and the next `serve start` then
        # reported a stale pidfile for a notebook that had stopped cleanly
        # (measured 2026-09-14).
        _forget(serve_port)
        _stop_own_group()

    signal.signal(signal.SIGTERM, _on_stop)
    signal.signal(signal.SIGINT, _on_stop)

    # NO `JUPYTER_PATH`.  The kernel lives in the SAME prefix the server runs
    # in, so jupyter finds it through `sys.prefix` with no search path at all
    # (verified 2026-09-14: `jupyter kernelspec list` with the variable unset
    # resolves `python3` in this env).  A `kernel_search_path()` stood here
    # until then, from the design where every env offered itself as a kernel;
    # once that was removed it contributed ZERO kernels and one cross-env
    # leak -- the host env's `share/jupyter` went on the path FIRST, handing
    # the framed Lab a `jupyterlab-plotly` labextension built against a
    # `plotly` this env does not have.
    env = dict(os.environ)
    # The token travels here rather than on the command line: `environ` is
    # owner-only, `cmdline` is not.  Verified end to end 2026-09-20 -- the
    # runtime file's token authenticates against the running server, and the
    # argv carries no token at all.
    env["JUPYTER_TOKEN"] = token
    try:
        # NO PIPE AND NO SINK.  This process's stdout and stderr ARE the
        # notebook log -- the supervisor opened it and handed it over -- so
        # the child inheriting them writes straight into that file.
        #
        # It used to be streamed through `sink=sys.stderr`, which copied
        # every line through Python into the same file it was already bound
        # for, and kept a copy of all of it in memory for the life of the
        # server (`run_streaming` accumulates, for the tail-on-failure a
        # BUILD wants; this return value is discarded).  A server is not a
        # build: it runs for days and Jupyter logs every request.
        #
        # The cost is stated at `dispatch_into_env`: no pipe means no
        # automatic fallback for a broken `mamba run`.  Here that is the
        # right trade -- the manager's error goes into the notebook log like
        # everything else, and the tab already says a notebook was asked for
        # and none started.
        rc, _out = dispatch_into_env(
            argv, caps.env_prefix(env_name), env=env, inherit_stdio=True)
    finally:
        _forget(serve_port)
    return rc if rc is not None else 1
