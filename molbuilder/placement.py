"""Where every configured file sits, what mode it must have, and who checks.

`configuration.md` § 3.1 draws the tree and § 2.1b states the mode rule.  This
module is the same facts as a table code can act on, and it exists because prose
cannot be executed: three of that page's own statements were measured false on
2026-09-12 -- the config root was said to be `0700` and nothing created it that
way, the serve log was said to be `0600` and `serve status` made it `0664`, and a
row claimed this page owned "the mode and the durability of every file listed
here" while several had no stated mode and were created with none.

**The table is the authority and § 3.1 cites it**, not the other way round.  A
test comparing prose to code would break on formatting and still prove nothing
about coverage; a checker that silently omits a file is the defect to prevent,
and the way to prevent it is to have one list that both the creators and the
audit read.

**Two jobs, kept apart, because they belong to different verbs.**  `envs
init-config` SEEDS -- it makes what is missing and leaves what is there, since
seeding must not police a directory the operator set up deliberately.  `envs
doctor` AUDITS -- it reports what arrived loose.  That division was already
written in `initconfig._ensure_root`'s docstring (*"this seeds, it does not
police -- `envs doctor` is where a permissions audit would belong"*) and had
neither a table nor an audit to be true of.

A directory or file ARRIVES loose in ways no writer controls: copied from
another machine, restored from a backup, unpacked from an archive that dropped
its modes, or made by a command that forgot.  That is why the check is on the
way in rather than only at creation.
"""
from __future__ import annotations

import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Tuple

from .config_dir import (PRIVATE_FILE_MODE, PRIVATE_DIR_MODE, config_dir,
                         google_client_secret, jupyter_lab_home, logs_dir,
                         reports_dir, runtime_dir, secrets_dir, session_key,
                         state_dir)

__all__ = ["Place", "places", "findings"]


@dataclass(frozen=True)
class Place:
    """One entry of the tree, as a fact that can be checked.

    ``resolve`` is the OWNER's resolver -- never a path spelled here, which is
    A11 and is why this table imports each owner rather than joining anything.
    ``pattern`` covers a family whose names carry a port (the serve logs); the
    audit then checks whatever exists rather than having to be told which ports
    were ever used.  ``mode=None`` means *deliberately not policed*: the file is
    not a credential and nothing in the tree states a mode for it.
    """
    what: str
    resolve: Callable[[], Path]
    is_dir: bool
    mode: Optional[int]
    holds_credential: bool
    why: str
    pattern: Optional[str] = None


def places() -> Tuple[Place, ...]:
    """THE TABLE -- every entry `configuration.md` § 3.1 draws.

    The owners are imported here rather than at module scope: `runtime_config`,
    `monitor` and `scheduler.record` all import `config_dir`, so a module-level
    import of them would close a cycle.  They are each asked for their own
    path, which is the point of the table.
    """
    from .monitor import default_notify_path, notify_keys_path
    from .runtime_config import machine_config_path
    from .scheduler.record import (FILENAME as ENVIRONMENT_FILENAME,
                                   environments_dir, machine_scope_path)

    return (
        Place("the config directory", config_dir, True, PRIVATE_DIR_MODE, True,
              "it holds the session key, the OAuth client secret and the "
              "notify keys, and a listable directory names a file even when "
              "the file itself is shut"),
        Place("molbuilder.json", lambda: machine_config_path(), False,
              PRIVATE_FILE_MODE, True,
              "it carries tls.key's path and the auth.providers block"),
        # The NAME from its owner, not a literal: this row re-spelled
        # `environment.json` until 2026-09-13 -- caught by running the retired
        # A11 speller check against the tree, three days after its retirement.
        Place(ENVIRONMENT_FILENAME, machine_scope_path, False, None, False,
              "what the machine IS -- cores, GPUs, queues.  No credential, and "
              "nothing in the tree states a mode for it"),
        Place("the session key", session_key, False, PRIVATE_FILE_MODE, True,
              "anyone who can read it can forge a session"),
        Place("the Google client secret", google_client_secret, False,
              PRIVATE_FILE_MODE, True, "a provider credential"),
        Place("the notify channels file", default_notify_path, False,
              PRIVATE_FILE_MODE, True,
              "for Slack and Discord the URL IS the credential, so this file "
              "is private even when it looks like it holds no key"),
        Place("the notify signing keys", notify_keys_path, False,
              PRIVATE_FILE_MODE, True, "they sign run reports as you"),
        Place("environments/", environments_dir, True, PRIVATE_DIR_MODE, False,
              "it sits inside the config root and inherits its discipline"),
        Place("secrets/", secrets_dir, True, PRIVATE_DIR_MODE, True,
              "every file in it is named by molbuilder.json BY PATH -- a TLS "
              "key, a provider's client secret"),
        Place("the state directory", state_dir, True, None, False,
              "logs and measurements; the modes that matter are below it"),
        Place("logs/", logs_dir, True, PRIVATE_DIR_MODE, True,
              "the serve log carries a provider's client_secret, routed there "
              "deliberately to keep it out of a user-visible response"),
        Place("a serve log", logs_dir, False, PRIVATE_FILE_MODE, True,
              "measured 2026-09-12 carrying a client_secret", "serve-*.log"),
        Place("reports/", reports_dir, True, None, False,
              "per-run measurements, kept and grepped; a scientific report, "
              "with nothing in it to guard"),
        Place("the runtime directory", runtime_dir, True, PRIVATE_DIR_MODE,
              False,
              "ours, and when $XDG_RUNTIME_DIR is absent it falls back inside "
              "the state root, where the default umask is not good enough"),
        Place("a serve pidfile", runtime_dir, False, None, False,
              "an address, not a secret", "serve-*.pid"),
        # THE NOTEBOOK'S FOUR, absent from this table until 2026-09-15 while
        # the newest credential in the tree lived in two of them
        # (`plan.md` § 5n.8).  This module's own docstring says why that
        # mattered: the audit exists because a file "ARRIVES loose in ways no
        # writer controls" -- restored from a backup, copied off another
        # machine, or left by a molbuilder older than the 2026-09-14 fix,
        # which is exactly the measured 0664 case.  The writers are careful
        # now; nothing was WATCHING.
        Place("a notebook log", logs_dir, False, PRIVATE_FILE_MODE, True,
              "jupyter-server prints its own URL with the token in it at "
              "every start -- measured 2026-09-14 at 0664 with 14 "
              "occurrences of `token=`, and the token reaches a live kernel",
              "jupyter-*.log"),
        Place("a notebook pidfile", runtime_dir, False, None, False,
              "an address, not a secret -- the shepherd's, beside the "
              "serve pidfile", "jupyter-*.pid"),
        Place("a notebook runtime file", runtime_dir, False,
              PRIVATE_FILE_MODE, True,
              "it holds the notebook's TOKEN, which authenticates a browser "
              "to a live kernel -- arbitrary code execution as this account",
              "jupyter-*.json"),
        Place("the framed Lab's home", jupyter_lab_home, True,
              PRIVATE_DIR_MODE, True,
              "molbuilder's own Lab settings and per-server workspaces; it "
              "sits in the state root, where the default umask is not good "
              "enough"),
    )


def findings() -> List[str]:
    """What in the tree exists and arrived looser than it should be.

    One sentence each, naming the exact `chmod`.  Only what EXISTS is reported:
    an absent file is not a finding -- most of this tree is optional, and a
    workstation with no HTTPS and no sign-in has an empty `secrets/` and no
    `notify` at all.

    A warning, never a refusal (`configuration.md` § 2.1a's reasoning): the fix
    is one command and the person is told it, where refusing would lock them
    out of their own installation over a condition they can repair.
    """
    out: List[str] = []
    for place in places():
        if place.mode is None:
            continue
        try:
            base = place.resolve()
        except Exception:          # a resolver that needs config we cannot read
            continue
        targets: List[Path] = []
        if place.pattern:
            try:
                targets = sorted(p for p in Path(base).glob(place.pattern)
                                 if p.is_file())
            except OSError:
                targets = []
        elif Path(base).exists():
            targets = [Path(base)]
        for target in targets:
            try:
                actual = stat.S_IMODE(target.stat().st_mode)
            except OSError:
                continue
            if actual == place.mode:
                continue
            if not (actual & ~place.mode):
                continue           # tighter than required is not a finding
            out.append(
                f"{place.what} is mode {actual:04o}, and {place.mode:04o} is "
                f"what it should be -- {place.why}.\n"
                f"  Fix it with: chmod {place.mode:04o} {target}")
    return out


def machine_config_warnings() -> List[str]:
    """Everything wrong with where and how the config ARRIVED, one line each.

    The shadow (a ``./molbuilder.json`` nobody reads -- phrased by
    `runtime_config.machine_config_shadow`) followed by every mode finding
    in the tree (`findings`).  Every command that reads the config prints
    these -- `serve`, the jobset verbs, `auth-setup` -- so adding a check
    here reaches all of them.

    It lived in `runtime_config` until 2026-09-13 and imported this module
    lazily to get the findings: the one place `runtime_config` reached UP,
    for a sum that is an audit (K-Y2).  Here it imports downward only.
    """
    from .runtime_config import machine_config_shadow
    shadow = machine_config_shadow()
    return ([shadow] if shadow else []) + findings()
