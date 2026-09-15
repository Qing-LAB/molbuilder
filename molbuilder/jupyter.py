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
   `supervise()` (so `serve start` AND `serve foreground`) reads the pidfile
   and, only if the pid is alive and really is a shepherd of ours, stops it.
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

#: Jupyter's own idle reaping, in seconds.  **Its settings, not a timer of
#: ours** (`jupyter.md` § 4): a hand-rolled one would be a second opinion
#: about when a notebook is idle, and Jupyter is the half that knows.
_CULL_IDLE_KERNEL_S = 30 * 60
_SHUTDOWN_NO_ACTIVITY_S = 60 * 60


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


# --------------------------------------------------------------------- #
#  Where the files are -- through config_dir's doors, like every other    #
# --------------------------------------------------------------------- #

def pid_path(serve_port: int) -> Path:
    from .config_dir import jupyter_pidfile
    return jupyter_pidfile(serve_port)


def log_path(serve_port: int) -> Path:
    from .config_dir import jupyter_log
    return jupyter_log(serve_port)


def runtime_path(serve_port: int) -> Path:
    from .config_dir import jupyter_runtime
    return jupyter_runtime(serve_port)


# --------------------------------------------------------------------- #
#  Reading the state -- verify before you signal                         #
# --------------------------------------------------------------------- #

def read_pid(serve_port: int) -> Optional[int]:
    try:
        return int(pid_path(serve_port).read_text().strip())
    except (OSError, ValueError):
        return None


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
    """``{"url": ..., "token": ..., "port": ...}``, or ``{}``.

    The token is a CREDENTIAL -- it authenticates a browser to a live kernel --
    so this is read server-side and handed to the page, never published.
    """
    import json
    try:
        doc = json.loads(runtime_path(serve_port).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return doc if isinstance(doc, dict) else {}


def _unverified_ctx():
    """The TLS context for talking to OUR OWN notebook.

    Verification is off on purpose and the reason is one sentence:
    molbuilder handed Jupyter that certificate, and the question being asked
    is *"is it answering"*, not *"is it trusted"*.  One home, because this is
    a security knob and it was built twice in this file -- once in
    `answering`, once in `open_notebooks` -- with nothing binding the copies.
    """
    import ssl
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


def answering(serve_port: int, *, timeout: float = 1.5) -> bool:
    """Is something listening on the notebook port and talking HTTP?

    The SECOND question (`deployment.md` § 1.0b's rule): a process that is up
    and not answering is exactly the failure a single word would call healthy.
    TLS is not verified here -- molbuilder gave Jupyter the certificate, and
    what is being asked is *"is it answering"*, not *"is it trusted"*.
    """
    import urllib.error
    import urllib.request
    runtime = read_runtime(serve_port)
    url = str(runtime.get("url") or "")
    if not url:
        return False
    try:
        with urllib.request.urlopen(url, timeout=timeout,
                                    context=_unverified_ctx()):
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
                                    context=_unverified_ctx()) as r:
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
    }


def _forget(serve_port: int) -> None:
    for p in (pid_path(serve_port), runtime_path(serve_port)):
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


#: WHAT A FRAMED LAB STARTS AS.  Lab's defaults are right for a Lab in a
#: window and wrong for one inside a tab of another application.  Both plugin
#: ids and both value shapes are Jupyter's own, read out of
#: ``<env>/share/jupyter/lab/schemas/`` rather than guessed -- which is why
#: `fetchNews` is a quoted STRING enum ("true" / "false" / "none").
#:
#: These are DEFAULTS.  Lab reads them from `app_settings_dir`, and a person's
#: own change inside Lab is written to the USER settings directory, which
#: still wins -- so this sets where the tab STARTS, not what anyone is allowed.
#:
#: **This table and `_SERVER_CONFIG` below were deleted by accident on
#: 2026-09-14** -- a helper that removed three dead functions took everything
#: between two of them, and `prepare_lab_home` then raised `NameError` on
#: every notebook start.  Nothing caught it: no test reaches this path, and
#: the shepherd's traceback goes to the notebook log.  Restored verbatim from
#: the files the working code had already generated.
_LAB_OVERRIDES: Dict[str, Dict[str, object]] = {
    "@jupyterlab/application-extension:shell": {
        # MULTI-DOCUMENT MODE -- Lab's document TAB BAR, so several notebooks
        # are open at once.  Single-document mode was tried first, to be rid
        # of Lab's file browser; it takes the tab bar with it, and reading two
        # notebooks side by side is worth more than losing the panel is
        # (decided with the user 2026-09-14).
        "startMode": "multiple",
    },
    "@jupyterlab/notebook-extension:tracker": {
        # CLOSING A NOTEBOOK SHUTS ITS KERNEL DOWN.  Lab keeps it running by
        # default so you can reopen with your variables; inside a tab of
        # another application a kernel nobody can see is memory -- and on a
        # GPU box a device -- held for no one.  A page RELOAD is not a close,
        # so reopening the same notebook still reconnects.  Asked for by the
        # user 2026-09-14.
        "kernelShutdown": True,
    },
    "@jupyterlab/apputils-extension:themes": {
        # Lab renders light by default, inside an application that is dark
        # everywhere else: the frame read as a different program pasted into
        # the page.
        "theme": "JupyterLab Dark",
        # Lab leaves its scrollbars light-on-light otherwise -- the one part
        # of the frame that still flashed white against this palette.
        "theme-scrollbars": True,
    },
    "@jupyterlab/apputils-extension:notification": {
        # Jupyter asks every viewer whether it may fetch its news feed, in a
        # popup over the frame.  A tab inside molbuilder is not where that is
        # answered, and the answer would be a network call from a machine
        # that may have no route out.
        "fetchNews": "false",
        "checkForUpdates": False,
    },
}


#: A jupyter-server config file, written beside the Lab settings and handed to
#: the server as `ServerApp.config_file`.  It exists for ONE thing.
#:
#: **Jupyter writes a `.ipynb_checkpoints/` directory beside every notebook it
#: saves.**  In a projects tree that is a directory in every folder somebody
#: has opened a notebook in -- swept up by result scans, carried along by
#: every copy to a cluster, and holding a stale duplicate of work nobody asked
#: it to keep.
#:
#: Jupyter has no switch for it.  `FileCheckpoints.checkpoint_dir` only
#: RENAMES the directory, and pointing it at one shared absolute path is worse
#: than the problem: the checkpoint file is named after the notebook alone, so
#: two `Untitled.ipynb` in different folders collide and a restore hands back
#: the wrong file.  What the contents manager does take is a
#: `checkpoints_class`, and jupyter-server ships no no-op one -- so molbuilder
#: writes it.  A config file is executed Python, which is why the class can
#: live here instead of on `PYTHONPATH`.
#:
#: `restore_checkpoint` REFUSES rather than quietly doing nothing: with
#: `list_checkpoints` empty Lab offers nothing to restore, and a path that
#: could still be reached must never silently discard an edit.
_SERVER_CONFIG = '# Generated by molbuilder at every notebook start -- edits here are lost.\n# Why this file exists: molbuilder.jupyter._SERVER_CONFIG.\nfrom datetime import datetime, timezone\n\nfrom jupyter_server.services.contents.checkpoints import AsyncCheckpoints\n\n\nclass NoCheckpoints(AsyncCheckpoints):\n    """Answer the checkpoint API without writing anything to disk."""\n\n    async def create_checkpoint(self, contents_mgr, path):\n        return {"id": "no-checkpoint",\n                "last_modified": datetime.now(timezone.utc)}\n\n    async def list_checkpoints(self, path):\n        return []\n\n    async def rename_checkpoint(self, checkpoint_id, old_path, new_path):\n        return None\n\n    async def delete_checkpoint(self, checkpoint_id, path):\n        return None\n\n    async def restore_checkpoint(self, contents_mgr, checkpoint_id, path):\n        from tornado.web import HTTPError\n        raise HTTPError(\n            400,\n            "molbuilder runs this notebook server with checkpoints disabled, "\n            "so there is nothing to restore.",\n        )\n\n\n# Both sections: `AsyncContentsManager` redeclares the trait that\n# `ContentsManager` defines, and the running manager inherits from both.\nc.ContentsManager.checkpoints_class = NoCheckpoints        # noqa: F821\nc.AsyncContentsManager.checkpoints_class = NoCheckpoints   # noqa: F821\n'


def prepare_lab_home() -> Dict[str, str]:
    """Prepare the framed Lab's own directories and config; return every
    generated path keyed by the command-line option it answers.

    Three directories and one config file, all separate from ``~/.jupyter``
    for one reason: **the Lab in this frame is molbuilder's, not the one the
    person runs themselves.**
    Sharing them let each write over the other -- Lab saves a user setting the
    first time it resolves one, and a user setting beats an override, so the
    framed Lab adopted whatever the standalone Lab had written and ignored
    every default here (measured 2026-09-14: the theme stayed light).

    The overrides file is rewritten at every start, so the answer to "why does
    my framed Lab look like this" is always the table above and never a stale
    file.  The other two are only created; what Lab saves into them is the
    person's.
    """
    import json
    from .config_dir import ensure_private_dir, jupyter_lab_home
    home = ensure_private_dir(jupyter_lab_home())
    dirs = {
        "LabApp.app_settings_dir":  ensure_private_dir(home / "settings"),
        "LabApp.user_settings_dir": ensure_private_dir(home / "user-settings"),
        "LabApp.workspaces_dir":    ensure_private_dir(home / "workspaces"),
    }
    (dirs["LabApp.app_settings_dir"] / "overrides.json").write_text(
        json.dumps(_LAB_OVERRIDES, indent=2) + "\n", encoding="utf-8")
    cfg = home / "jupyter_server_config.py"
    cfg.write_text(_SERVER_CONFIG, encoding="utf-8")
    out = {k: str(v) for k, v in dirs.items()}
    # An ABSOLUTE `config_file` is loaded INSTEAD of searching the config path
    # (`jupyter_core.application.load_config_file`), so the framed server does
    # not read a personal `~/.jupyter/jupyter_server_config.py` either -- the
    # same isolation the settings home gets, for the same reason.
    out["ServerApp.config_file"] = str(cfg)
    return out


def notebook_argv(conda: str, env_name: str, *, host: str, port: int,
                  token: str, root_dir: str,
                  cert: Optional[str], key: Optional[str],
                  lab_dirs: Optional[Dict[str, str]] = None) -> List[str]:
    """The ``jupyter lab`` command line, as the door will carry it.

    Every setting here is Jupyter's own (`jupyter.md` § 4): molbuilder states
    what it needs and does not re-implement any of it.

    * ``--ServerApp.token`` -- generated by us so the tab can build a URL that
      works without a person copying anything.  It is a CREDENTIAL and lives
      in a 0600 runtime file.
    * ``tornado_settings`` carries ``frame-ancestors`` -- Jupyter refuses to
      be framed by default, and the tab is an iframe (§ 2).
    * ``cull_idle_timeout`` / ``shutdown_no_activity_timeout`` -- Jupyter's
      own idle reaping rather than a timer of ours.
    * ``lab_dirs`` -- everything `prepare_lab_home` generated, keyed by the
      option each answers (the key carries its own ``LabApp.`` /
      ``ServerApp.`` prefix).  Passed straight through, so another generated
      path is a line there and nothing here.
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
    settings = {"headers": {"Content-Security-Policy": frame_ancestors}}
    inner = [
        "jupyter", "lab",
        "--no-browser",
        f"--ServerApp.ip={host}",
        f"--ServerApp.port={port}",
        # A TAKEN PORT IS A REFUSAL, NOT A SILENT MOVE.  jupyter-server
        # defaults `port_retries` to 50 and then picks a RANDOM free port in
        # that range -- so if `serve_port + 1` were busy, Jupyter would start
        # happily somewhere nobody recorded, while the runtime file, the
        # `frame-src` CSP, the `frame-ancestors` grant and `answering()` all
        # kept naming the port it was asked for.  The tab would say "wedged"
        # over a perfectly healthy notebook.  Zero makes the clash an error
        # molbuilder can see and report.
        "--ServerApp.port_retries=0",
        f"--ServerApp.token={token}",
        f"--ServerApp.root_dir={root_dir}",
        f"--ServerApp.tornado_settings={json.dumps(settings)}",
        f"--MappingKernelManager.cull_idle_timeout={_CULL_IDLE_KERNEL_S}",
        "--MappingKernelManager.cull_connected=False",
        f"--ServerApp.shutdown_no_activity_timeout="
        f"{_SHUTDOWN_NO_ACTIVITY_S}",
    ]
    for _opt, _path in sorted((lab_dirs or {}).items()):
        inner.append(f"--{_opt}={_path}")
    if cert and key:
        inner += [f"--ServerApp.certfile={cert}",
                  f"--ServerApp.keyfile={key}"]
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
    from .config_dir import PRIVATE_FILE_MODE, ensure_private_dir
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
    lab_dirs = prepare_lab_home()
    scheme = "https" if (cert and key) else "http"
    # BRACKET AN IPv6 LITERAL.  `--host ::1` produced `http://::1:8001/lab`,
    # which `urlopen` cannot parse -- so `answering()` reported down forever
    # and the tab stuck on "Starting...".  (The browser was unaffected: the
    # tab rebuilds the base from `location.hostname`.)
    _hostpart = f"[{host}]" if ":" in host else host
    ensure_private_dir(pid_path(serve_port).parent, tighten=True)
    pid_path(serve_port).write_text(f"{os.getpid()}\n")
    # 0600: the token authenticates a browser to a live kernel.
    write_json(runtime_path(serve_port),
               {"url": f"{scheme}://{_hostpart}:{port}/lab",
                "base": f"{scheme}://{_hostpart}:{port}",
                "token": token, "port": port},
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
    argv = notebook_argv(caps.conda_binary, env_name, host=host, port=port,
                         token=token, root_dir=str(projects_root()),
                         cert=cert, key=key, lab_dirs=lab_dirs)
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
