"""Where molbuilder keeps its own per-user files — stated once.

``$XDG_CONFIG_HOME/molbuilder``, else ``~/.config/molbuilder``.  Three
modules computed that same two-line rule independently:

* ``runtime_config._machine_config_file`` -> ``molbuilder.json``
* ``scheduler/record.machine_scope_path`` -> ``environment.json`` and the
  ``environments/`` beside it
* ``auth_setup.default_secret_dir`` -> ``secret_key``

They agreed, and two of them said so in prose -- one docstring reads
*"Mirrors auth_setup.default_secret_dir's convention"*, the other
*"mirrored rather than imported"*.  **A comment is not a mechanism**, and
`configuration.md` M-4 already made this exact call one level down: it gave
``environment.json`` one home for its FILENAME, which "was a string literal
in three modules".  The directory that filename sits in never got the same
treatment.  This is it.

*(``auth_setup.default_secret_dir`` is named above as it was.  It survived
this module as a one-line ``return config_dir()`` -- the duplication gone,
the second NAME still there -- along with ``auth_setup.secret_key_path`` and
``google_client_secret_path``, pass-throughs to the two doors below.  All
three were deleted 2026-09-13; callers ask here.)*

**L1: pure stdlib, no molbuilder deps -- any layer may use it.**  That line
is copied deliberately from ``persist.py``, which is the precedent: the same
shape (one rule, several callers, one of them ``scheduler/record.py``) and
the same resolution.  It is what lets ``record.py`` import this without
giving up the stdlib-only property it claims -- the property is *depends
only on stdlib*, not *imports nothing from molbuilder*, which is why
``record.py`` can already do ``from ..persist import write_json``.

**Why no ``paths.state`` setting to override it** (user decision,
2026-08-23).  ``XDG_CONFIG_HOME`` already moves this directory, and that is
the documented answer to the case that motivates moving it at all --
``auth_setup``'s own docstring: *"a user with ``$XDG_CONFIG_HOME=/scratch/
$USER`` keeps secrets off the NFS-mounted $HOME on HPC nodes."*  A config
key would be a second way to say one thing, and the ordering is *delete >
one home > parameter > abstraction*: one function is one home, a key is a
parameter.  It would also be circular for the first caller, which uses this
to FIND ``molbuilder.json``.  If a need ever appears to split them -- the
config in a repo, the state on scratch -- this function is where the
override hangs, and nothing here has to move first.
"""
from __future__ import annotations

import os
import stat
from pathlib import Path
from typing import List

__all__ = [
    # The directories.  A format owner asks for one of these and joins its
    # own filename; nobody else joins at all.
    "config_dir", "state_dir", "runtime_dir", "logs_dir", "reports_dir",
    # The files with no format to own them -- spelled here and nowhere else.
    "session_key", "google_client_secret", "secrets_dir",
    "relative_home",
    "serve_pidfile", "serve_log", "serve_stacks_log",
    "ports_with_pidfile",
    "jupyter_pidfile", "jupyter_log", "jupyter_runtime",
    "jupyter_lab_home",
    "CONFIG_DIR_ENV", "DIRNAME",
    # Making one of those directories, privately.
    "PRIVATE_DIR_MODE", "PRIVATE_FILE_MODE", "ensure_private_dir",
]

#: A directory that holds credentials, or sits around files that do: a listable
#: directory names a file even when the file itself is shut
#: (`configuration.md` § 2.1b).
PRIVATE_DIR_MODE = 0o700

#: A file carrying a credential: owner reads and writes, nobody else.
#: (Three pairs of names carried these two numbers until 2026-09-13 --
#: `CONFIG_*_MODE` in `runtime_config`, `REPORT_*_MODE` in the notify
#: blueprint -- so the writer of `molbuilder.json` used one pair and the
#: audit checked the same file with another.  K-D2.)
PRIVATE_FILE_MODE = 0o600

#: The directory name under the XDG config root.  One string, because it is
#: the half of the path that is not the XDG convention.
DIRNAME = "molbuilder"

#: Name this and it IS the root, exactly as given
#: (`archive/2026-09-01-config-access-plan.md` § 3.1).
#:
#: Spelled like ``MOLBUILDER_DATA_DIR`` and ``MOLBUILDER_PROJECTS``, which are
#: already the convention for "the program's own <thing> directory".
CONFIG_DIR_ENV = "MOLBUILDER_CONFIG_DIR"


def config_dir() -> Path:
    """Where this installation's own configuration lives.

    ``$MOLBUILDER_CONFIG_DIR`` if set, else ``$XDG_CONFIG_HOME/molbuilder``,
    else ``~/.config/molbuilder``.

    Not required to exist -- every caller either writes it on demand or treats
    an absent file as *unset*.  It IS created up front on a first install:
    ``molbuilder envs init-config``, which ``envs bootstrap`` runs at the end,
    makes the directory and seeds ``molbuilder.json`` (`configuration.md`
    § 2.1c).  That changes nothing here; an absent directory is still legal and
    still means *nothing was said*.  Read at CALL time rather
    than captured at import, so a test (or an operator) that moves the root
    moves every one of the callers above together.

    **The override is used EXACTLY AS GIVEN** -- no ``molbuilder`` component is
    appended.  ``XDG_CONFIG_HOME`` names a root shared by every application, so
    ours must add its own name under it; ``MOLBUILDER_CONFIG_DIR`` names OUR
    directory, and appending to it would put the files somewhere the person did
    not ask for.  The two variables answer different questions and are treated
    differently on purpose.

    **It is an override, not a search step.**  Set it and that is the root,
    entire: nothing falls back past it, and a file in one of the other two
    places is not consulted.  A fallback here would recreate exactly the
    shadowing that `configuration.md` § 2.1a exists to warn about -- one
    setting, two files, one of them silently winning.
    """
    override = os.environ.get(CONFIG_DIR_ENV)
    if override:
        return Path(override)
    xdg = os.environ.get("XDG_CONFIG_HOME")
    return (Path(xdg) if xdg else Path.home() / ".config") / DIRNAME


def state_dir() -> Path:
    """Where operational state lives -- logs and reports.

    ``$XDG_STATE_HOME/molbuilder``, else ``~/.local/state/molbuilder``
    (`archive/2026-09-01-config-access-plan.md` § 3.2).

    ``XDG_STATE_HOME`` entered the Base Directory spec in 0.8 for state that
    persists across restarts but is not portable or important enough for
    ``$XDG_DATA_HOME`` -- and the spec names LOGS first, which is what this
    holds.  ``~/.var/log`` and ``~/.local/log`` are not conventions:
    ``~/.var/app/`` is flatpak's, and the latter is not in the spec at all.

    **Separate from the config root on purpose.**  Configuration is edited and
    backed up; logs grow and are deleted.  A person who wants both in one
    place sets ``XDG_STATE_HOME``; there is no config key for it and cannot
    be: this module is the bootstrap that finds ``molbuilder.json``, so it
    must answer before any config is read (``paths.logs`` is refused by
    `runtime_config._read_paths` for exactly that reason -- this docstring
    said the opposite until 2026-09-13).
    """
    xdg = os.environ.get("XDG_STATE_HOME")
    return (Path(xdg) if xdg else Path.home() / ".local" / "state") / DIRNAME


def runtime_dir() -> Path:
    """Where pidfiles and sockets live.

    ``$XDG_RUNTIME_DIR/molbuilder`` when the variable is set, else
    ``state_dir()/run``.

    ``XDG_RUNTIME_DIR`` is the spec's directory for exactly this -- owner-only,
    and **cleared when the session ends**, which is right for a pidfile and
    wrong for anything meant to outlive a logout.  It is not always set (cron,
    a detached ssh, some containers), and the fallback is deliberately the
    STATE directory rather than a temp dir: a supervisor's pidfile that
    vanished under it would leave a running server nothing can find.
    """
    xdg = os.environ.get("XDG_RUNTIME_DIR")
    if xdg:
        return Path(xdg) / DIRNAME
    return state_dir() / "run"


# ══ THE FILES ═══════════════════════════════════════════════════════════════
#
# A CALLER NAMES THE FILE IT WANTS AND GETS A PATH.  It never names a
# directory and it never joins (user, 2026-08-31: *"users ... should go through
# API rather than go through directly for some variables ... they don't need to
# handcraft anything or derive anything"*).
#
# The filenames live here and nowhere else.  They were spread across seven
# modules -- `runtime_config`, `auth_setup`, `scheduler/record`, `monitor`,
# `cli`, `serve_daemon` -- each joining its own onto a directory.  Each join
# was small and correct; together they were seven modules that had to agree
# about a spelling with nothing making them.  `configuration.md` M-4 recorded
# exactly this for ONE file -- *"a string literal in three modules"* -- fixed
# that one, and did not generalise the rule, so the next six grew back.
#
# The environment variables are read above, to DERIVE these answers.  No
# caller sees them.

#: THE DIVISION, and A11 draws it: **the module that owns a FORMAT owns its
#: NAME**, and this module owns the DIRECTORY.  So a file with a format owner
#: keeps its name there and that owner exposes the path function --
#: `runtime_config.machine_config_path`, `scheduler/record.machine_scope_path`,
#: `monitor.default_notify_path`.  Each asks here for the directory and joins
#: once, in the one module entitled to spell it.
#:
#: What lives HERE is the files with no format to own: opaque secrets, a
#: pidfile, a log.  Nobody else may spell these.
#:
#: WHERE they sit changed on 2026-09-20 (user): every secret now lives in
#: `secrets/`, not beside `molbuilder.json`.  The property that matters is
#: unchanged -- ONE fixed home per file, resolved by ONE function, which is
#: what stops a reader and a writer meaning different files (2.1e).  That
#: property was never about WHICH directory, only about there being exactly
#: one; a directory named `secrets` that did not hold the secrets misled
#: every reader who opened it.
#:
#: (Pulling `environment.json` and `notify` in here was tried and reverted the
#: same day -- it took a name away from its format owner, which is the rule
#: A11 exists to hold, and
#: `test_config_dir_has_one_home.py::TestNoModuleNamesOneOfThoseFilesItself`
#: said so -- it asserts each filename below appears in exactly the one
#: module entitled to spell it, so a registry here fails on the spelling
#: alone.  `configuration.md` § 2.3 records why a `retrieve_secret(name)`
#: door is the same proposal and meets the same test.)
SESSION_KEY_FILENAME = "secret_key"
GOOGLE_CLIENT_SECRET_FILENAME = "google_client_secret"

#: The directory every credential lives in -- the ones with a fixed home
#: (`secret_key`, `notify`, `notify_keys`) and the ones `molbuilder.json`
#: names by path (a TLS key, a provider's client secret).
SECRETS_DIRNAME = "secrets"


def session_key() -> Path:
    """The Flask session-signing key -- ``<config dir>/secrets/secret_key``.

    One home and one name, which is the whole point.  HISTORICALLY it was
    written to one path and read from another (``~/.molbuilder/secret.key``)
    -- two directories and two spellings -- so running ``auth-setup`` produced
    a key the server never read and reported success (`configuration.md`
    § 2.1e).  Both of those paths are dead; ask this function.

    It moved under `secrets/` on 2026-09-20 with every other credential.  That
    is a change of directory, not of the rule: still one home, still resolved
    here, still not nameable in `molbuilder.json`.
    """
    return secrets_dir() / SESSION_KEY_FILENAME


def google_client_secret() -> Path:
    """The DEFAULT home for Google's OAuth client secret.

    A default, not a fixed home: a provider entry may set
    ``auth.providers[].client_secret_file`` and `oauth.py` reads whatever the
    config names.  It moved into `secrets/` with the three fixed-home files
    on 2026-09-20 so that every credential molbuilder writes lands in one
    directory, whether or not the config could have named it.
    """
    return secrets_dir() / GOOGLE_CLIENT_SECRET_FILENAME


def read_session_key() -> "bytes | None":
    """The session key's BYTES, or ``None`` when it has not been made yet.

    **Ask for the SECRET, not for the path to it** *(user, 2026-09-20: "we
    should avoid user access the file directly, the api should return the
    KEY/SECRET")*.  Every caller used to do ``session_key().read_bytes()``,
    which is a second place that knows a credential is a file, how it is
    encoded, and what an unreadable one means.  `monitor` already did this
    correctly for its two -- `load_channels` and `read_notify_keys` hand back
    values -- and these two were the ones still handing out a path.

    ``None`` rather than an exception for "absent", because absent is an
    ordinary state: the server makes the key on first run (`web/auth.py`).
    A file that EXISTS but cannot be read is a different thing and raises.

    This is not a `retrieve_secret("name")` registry, which
    `configuration.md` § 2.3 refuses and which was reverted inside a day on
    2026-08-31: a name-keyed table has to re-spell filenames their owners
    own.  One named function per secret, on the module that owns it.
    """
    p = session_key()
    if not p.exists():
        return None
    return p.read_bytes()


def relative_home(resolve) -> str:
    """Where a file sits RELATIVE to the config directory -- ``secrets/notify``.

    **For the text molbuilder shows a person.**  `notify-token` prints a shell
    recipe that runs on a CLUSTER, so it cannot use an absolute local path: it
    builds `$cfg` from the same three branches `config_dir` does and joins a
    name.  It joined ``notify``, and when the credentials moved into `secrets/`
    the printed recipe went on telling people to write a webhook where nothing
    reads it -- silently, because a notifier swallows every failure by design.
    The AST guard could not catch that: it matches the literal ``"notify"`` and
    the string there was ``"$cfg/notify"``.  Deriving the tail is what closes
    it.

    **It lives here, not in `placement`** *(moved 2026-09-20)*.  It never
    touches that module's table -- it is `config_dir` arithmetic over a
    resolver the caller supplies -- and `placement` in this codebase means JOB
    placement nearly everywhere else (`scheduler/place.Placement`,
    `calcdirs.Placement`, `placement.domain` through `jobset/`), so
    `from .placement import relative_home` in `cli.py` read like scheduler
    code.  Here it sits beside the function that defines what it is relative
    TO, and it ships with this module to a compute node.

    **A path outside the config directory RAISES**, deliberately.  The first
    version returned the absolute path instead, which rendered as
    ``$cfg//run/user/1000/molbuilder/jupyter-8888.json`` -- a broken recipe
    from a function whose name promises a relative one.  Nothing can use that,
    so a resolver that is not under the config directory is a call-site
    mistake and says so.
    """
    return Path(resolve()).relative_to(Path(config_dir())).as_posix()


def secrets_dir() -> Path:
    """Every credential molbuilder keeps, in one directory.

    Two kinds live here and they differ in who names them, not in where they
    sit *(2026-09-20)*:

    * **fixed home** -- `secret_key`, `notify`, `notify_keys`.  Resolved by one
      function each and NOT nameable in `molbuilder.json`, which refuses
      `secret_key_file` / `notify_keys_file` outright.  That is what keeps a
      reader and a writer from meaning different files (2.1e).
    * **operator-named** -- a TLS key, a provider's `client_secret_file`.  The
      config names these by path, so this is their suggested home and the name
      is yours.

    The DIRECTORY has one owner, which is why this function exists --
    `envs init-config` used to join ``root / "secrets"`` itself, and a
    directory nobody owns is one the placement audit cannot check (A11).
    """
    return config_dir() / SECRETS_DIRNAME


def logs_dir() -> Path:
    """molbuilder's own operational output -- diagnostics, deleted when fixed."""
    return state_dir() / "logs"


def reports_dir() -> Path:
    """Per-run measurements -- kept, grepped a year later, NOT diagnostics.

    **``reports/``, not ``logs/``, and the distinction is the point** (user,
    2026-08-27: *this is a different kind of log, not of the status of
    molbuilder but collection of computation results*).  ``logs/`` holds
    molbuilder's own operational output -- read when something is wrong,
    deleted when it is fixed.  These are measurements from calculations:
    energies, iteration counts, when a relaxation step landed -- kept, grepped
    a year later, plotted.  Filing them under ``logs/`` invited exactly one
    mistake: treating them as disposable.  One file per user, JSON Lines, so
    ``jq`` and ``pandas`` read it with no parser of ours in the middle.
    """
    return state_dir() / "reports"


def serve_pidfile(port: int) -> Path:
    """The supervisor's pidfile -- the address ``stop``/``restart`` act on."""
    return runtime_dir() / f"serve-{port}.pid"


def ports_with_pidfile(prefix: str = "serve") -> List[int]:
    """Every port that has a pidfile -- `serve_pidfile` INVERTED.

    One home for the ``<prefix>-<port>.pid`` shape, in both directions, for
    the reason `jupyter.serve_port_of` exists: a second place that knows how
    to take the name apart drifts the day the name changes.

    **This is what lets `serve status` answer without being told a port.**
    It defaulted to 8000, so a server on 8888 was reported *"not running"* --
    confidently wrong, when the port was on disk the whole time
    (`plan.md` § 5n, J14).

    It lists only what has a FILE.  A `serve foreground` writes none, so it
    cannot appear here, and a caller that means *"what is running"* has to
    say so rather than let an empty list imply it.  Whether each pid is alive
    and really ours is `serve_daemon.pid_state`'s question, not this one's.
    """
    out: List[int] = []
    for p in runtime_dir().glob(f"{prefix}-*.pid"):
        try:
            out.append(int(p.stem.split("-", 1)[1]))
        except (IndexError, ValueError):
            continue        # not one of ours; the name is the only claim
    return sorted(out)


def serve_log(port: int) -> Path:
    """Everything the server prints."""
    return logs_dir() / f"serve-{port}.log"


def serve_stacks_log(port: int) -> Path:
    """Thread stacks, appended on ``SIGUSR1`` and before any forced child kill."""
    return logs_dir() / f"serve-{port}.stacks.log"


# --------------------------------------------------------------------- #
#  The notebook server (docs/web/jupyter.md)                             #
# --------------------------------------------------------------------- #
#
# KEYED BY THE SERVE PORT, not by Jupyter's own.  What a person stops is
# "the notebook belonging to the molbuilder I am running", and they know
# which molbuilder that is by the port they opened in a browser.  Jupyter's
# port is derived from it (`jupyter.jupyter_port`), so keying these files by
# the derived number would ask somebody to do the arithmetic before they
# could find a log.

def jupyter_pidfile(serve_port: int) -> Path:
    """The SHEPHERD's pid -- the address `jupyter stop` acts on.

    Beside the serve pidfile and for the same reason: it is an address, not a
    secret, and it lives where a per-user runtime file belongs.
    """
    return runtime_dir() / f"jupyter-{serve_port}.pid"


def jupyter_log(serve_port: int) -> Path:
    """Everything the notebook server prints."""
    return logs_dir() / f"jupyter-{serve_port}.log"


def jupyter_runtime(serve_port: int) -> Path:
    """Where the notebook is and the token that reaches it.

    **A CREDENTIAL**: the token is what authenticates a browser to a live
    kernel, which is arbitrary code execution as this account.  Written
    through `persist.write_json(mode=PRIVATE_FILE_MODE)` and read by the tab's
    own blueprint -- never by the browser directly.
    """
    return runtime_dir() / f"jupyter-{serve_port}.json"


def jupyter_lab_home() -> Path:
    """The settings home of the FRAMED JupyterLab -- molbuilder's, not yours.

    **Three directories and one file**, every one of them named in
    ``data/jupyter.toml``'s ``[lab.home]`` and ``[lab.config]`` rather than
    here, and each handed to Lab on the command line:

      * ``settings/overrides.json`` -- the DEFAULTS a framed Lab starts with
        (`jupyter.md` 4.1), rewritten at every start.
      * ``user-settings/`` -- where Lab saves what a person changes inside
        the frame.  **Never touched** by anything molbuilder does.
      * ``workspaces/<serve port>/`` -- which documents were open.
        **Emptied at every notebook-server start** and kept across every
        page load, which is what lets a tab switch come back to your
        notebook while a restart gives you a clean one (`jupyter.md` 4.1).
        **The one thing under here that IS port-keyed** -- see below.
      * ``jupyter_server_config.py`` -- copied from
        ``data/jupyter_server_config.py``; it is what suppresses
        ``.ipynb_checkpoints`` (`jupyter.md` 4.2).

    *(This said "two directories" while the code made three and wrote a
    config file beside them -- the door's own contract behind its caller,
    `plan.md` 5n J11.)*

    **Separate from ``~/.jupyter`` on purpose.**  Lab writes a user setting the
    first time it resolves one, and a user setting BEATS an override -- so a
    framed Lab sharing the person's own settings home adopted whatever their
    standalone Lab had written and ignored every default here (measured
    2026-09-14: the theme stayed light on the first reload and never changed
    again).  Keeping the two apart also means molbuilder's defaults never
    appear in the Lab they run themselves.

    **Not port-keyed, with one exception that proves the rule.**  The
    defaults do not differ between servers, and neither does a person's own
    change to one -- so ``settings/`` and ``user-settings/`` are shared, and
    a setting changed in one molbuilder's framed Lab follows them to the
    next.  ``workspaces/`` is SESSION state, which does differ, so it is
    keyed by serve port one level down (`jupyter._workspace_dir`).

    That exception was missed when the workspace arrived on 2026-09-15:
    this sentence justified sharing a directory that had just stopped
    holding only defaults, and the shared layout re-created the bug the
    workspace was added to fix -- one server's start emptied another's.
    """
    return state_dir() / "jupyter-lab"


def ensure_private_dir(d: Path, *, mode: int = PRIVATE_DIR_MODE,
                       tighten: bool = False) -> Path:
    """``mkdir -p`` at ``mode`` -- the ONE creator for a directory in this tree.

    It lives here, in the module that owns the directories, because the
    SUPERVISOR needs it too: `serve_daemon` is L1 and imports nothing of the
    application it restarts, so a creator one layer up would have left it with a
    private copy -- which is what it had, and the copy is why two creators
    disagreed about the case below.  Pure stdlib, which is also what lets this
    module keep travelling beside a job.

    The umask can only REMOVE bits from a requested mode, never add them, so
    ``0700`` is a ceiling rather than a suggestion.  (Verified, because the
    opposite is the intuitive reading and it is wrong.)

    **``tighten`` is off by default, and that is a decision rather than an
    oversight.** ``mode=`` covers only what this call CREATES; a directory that
    is already there keeps whatever it had.  Re-moding it is right for a
    directory this program owns and is about to write a credential into -- the
    supervisor's log directory, measured at ``0775`` around a log carrying a
    provider's ``client_secret``.  It is wrong for the config root: on a
    cluster ``XDG_CONFIG_HOME=/scratch/$USER`` is how a person keeps tokens off
    an NFS ``$HOME``, and silently re-moding what they set up is the program
    deciding for them.  Seeding seeds; `envs doctor` reports what arrived loose
    (`placement.findings`).

    Best-effort on the tightening: a directory somebody else owns is not ours to
    re-mode, and refusing to log would be the worse outcome.
    """
    d = Path(d)
    # EVERY MISSING ANCESTOR AT ``mode`` TOO.  `Path.mkdir(parents=True,
    # mode=)` gives the mode to the leaf only and creates missing parents at
    # the umask -- so the first thing made under a fresh config root (say
    # `environments/` by `jobset probe --write`) left the ROOT itself 0755
    # around every secret written into it later, and `envs doctor` then
    # reported a directory this program had made.  Measured 2026-09-14.
    for parent in reversed(d.parents):
        if not parent.exists():
            parent.mkdir(mode=mode, exist_ok=True)
    d.mkdir(exist_ok=True, mode=mode)
    if tighten:
        try:
            if stat.S_IMODE(d.stat().st_mode) != mode:
                os.chmod(d, mode)
        except OSError:
            pass
    return d
