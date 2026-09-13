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

__all__ = [
    # The directories.  A format owner asks for one of these and joins its
    # own filename; nobody else joins at all.
    "config_dir", "state_dir", "runtime_dir", "logs_dir", "reports_dir",
    # The files with no format to own them -- spelled here and nowhere else.
    "session_key", "google_client_secret", "secrets_dir",
    "serve_pidfile", "serve_log", "serve_stacks_log",
    "CONFIG_DIR_ENV", "DIRNAME",
    # Making one of those directories, privately.
    "PRIVATE_DIR_MODE", "CREDENTIAL_FILE_MODE", "ensure_private_dir",
]

#: A directory that holds credentials, or sits around files that do: a listable
#: directory names a file even when the file itself is shut
#: (`configuration.md` § 2.1b).
PRIVATE_DIR_MODE = 0o700

#: A file carrying a credential: owner reads and writes, nobody else.
CREDENTIAL_FILE_MODE = 0o600

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
    backed up; logs grow and are deleted.  A person may still put both in one
    place -- ``molbuilder.json``'s ``paths`` block names this directory, and
    :func:`molbuilder.runtime_config.logs_dir` is where that override is
    applied.  It cannot be applied HERE: this module is the bootstrap that
    finds ``molbuilder.json``, so it must answer before any config is read.
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

#: The directory for files `molbuilder.json` names by path.
SECRETS_DIRNAME = "secrets"


def session_key() -> Path:
    """The Flask session-signing key.

    One home and one name.  It was written as ``<config dir>/secret_key`` and
    read as ``~/.molbuilder/secret.key`` -- two directories and two spellings
    -- so running ``auth-setup`` produced a key the server never read and
    reported success (`configuration.md` § 2.1e).
    """
    return config_dir() / SESSION_KEY_FILENAME


def google_client_secret() -> Path:
    """The Google OAuth client secret."""
    return config_dir() / GOOGLE_CLIENT_SECRET_FILENAME


def secrets_dir() -> Path:
    """Files ``molbuilder.json`` names by PATH -- a TLS key, a client secret.

    The suggested home rather than a required one: the config names those files,
    so the name is the operator's.  The DIRECTORY still has one owner, which is
    why this function exists -- `envs init-config` used to join ``root /
    "secrets"`` itself, the only place in the tree naming that directory, and a
    directory nobody owns is a directory the audit below cannot check (A11).
    """
    return config_dir() / SECRETS_DIRNAME


def logs_dir() -> Path:
    """molbuilder's own operational output -- diagnostics, deleted when fixed."""
    return state_dir() / "logs"


def reports_dir() -> Path:
    """Per-run measurements -- kept, grepped a year later, NOT diagnostics.

    Beside ``logs/`` and deliberately not inside it: filing measurements under
    a name that reads as *disposable* invited exactly that mistake once.
    """
    return state_dir() / "reports"


def serve_pidfile(port: int) -> Path:
    """The supervisor's pidfile -- the address ``stop``/``restart`` act on."""
    return runtime_dir() / f"serve-{port}.pid"


def serve_log(port: int) -> Path:
    """Everything the server prints."""
    return logs_dir() / f"serve-{port}.log"


def serve_stacks_log(port: int) -> Path:
    """Thread stacks, appended on ``SIGUSR1`` and before any forced child kill."""
    return logs_dir() / f"serve-{port}.stacks.log"


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
    d.mkdir(parents=True, exist_ok=True, mode=mode)
    if tighten:
        try:
            if stat.S_IMODE(d.stat().st_mode) != mode:
                os.chmod(d, mode)
        except OSError:
            pass
    return d
