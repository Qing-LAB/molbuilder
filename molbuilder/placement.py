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

from .config_dir import (PRIVATE_FILE_MODE, PRIVATE_DIR_MODE, SECRETS_DIRNAME,
                         config_dir,
                         google_client_secret, jupyter_lab_home, logs_dir,
                         reports_dir, runtime_dir, secrets_dir, session_key,
                         state_dir)

__all__ = ["Place", "places", "findings", "misplaced",
           "machine_config_warnings"]


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
    why: str
    pattern: Optional[str] = None

    #: **Is this a credential molbuilder KEEPS** -- one that survives the
    #: session and must therefore live in `config_dir.secrets_dir()`?
    #: `misplaced` enforces exactly that, and nothing else reads this.
    #:
    #: THIS WAS THREE FIELDS FOR A FEW HOURS ON 2026-09-20 and they were
    #: compensating for each other.  `holds_credential` was declared here and
    #: **read by nothing, ever** -- `git log -S` finds no commit that used it
    #: -- yet `configuration.md` was written claiming it and this field "need
    #: different rules, so they are different fields".  That was a
    #: justification invented for a split that existed only in prose.  A third
    #: field, `outside_secrets_because`, then carried an exemption for the
    #: notebook token, whose own reason text said *"it is not a credential
    #: molbuilder KEEPS"* -- i.e. the predicate was wrong, not the row.
    #:
    #: Defining this as KEPT rather than as *purpose* makes all three
    #: collapse: the notebook token is simply `False` and says why in `why`,
    #: and the checker needs no exemption branch.  A log that merely CONTAINS
    #: a credential (a serve log, a notebook log) is `False` too and is
    #: policed by `mode`, which is the field that always did that job.
    credential_store: bool = False


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
        Place("the config directory", config_dir, True, PRIVATE_DIR_MODE,
              "it holds `secrets/` and molbuilder.json, and a listable "
              "directory names a file even when the file itself is shut"),
        Place("molbuilder.json", lambda: machine_config_path(), False,
              PRIVATE_FILE_MODE,
              "it carries tls.key's path and the auth.providers block"),
        # The NAME from its owner, not a literal: this row re-spelled
        # `environment.json` until 2026-09-13 -- caught by running the retired
        # A11 speller check against the tree, three days after its retirement.
        Place(ENVIRONMENT_FILENAME, machine_scope_path, False, None,
              "what the machine IS -- cores, GPUs, queues.  No credential, and "
              "nothing in the tree states a mode for it"),
        Place("the session key", session_key, False, PRIVATE_FILE_MODE,
              "anyone who can read it can forge a session",
              credential_store=True),
        Place("the Google client secret", google_client_secret, False,
              PRIVATE_FILE_MODE, "a provider credential",
              credential_store=True),
        Place("the notify channels file", default_notify_path, False,
              PRIVATE_FILE_MODE,
              "for Slack and Discord the URL IS the credential, so this file "
              "is private even when it looks like it holds no key",
              credential_store=True),
        Place("the notify signing keys", notify_keys_path, False,
              PRIVATE_FILE_MODE, "they sign run reports as you",
              credential_store=True),
        Place("environments/", environments_dir, True, PRIVATE_DIR_MODE,
              "it sits inside the config root and inherits its discipline"),
        Place("secrets/", secrets_dir, True, PRIVATE_DIR_MODE,
              "EVERY credential molbuilder keeps is in it -- the four with a "
              "fixed home.  A file molbuilder.json merely NAMES is the "
              "operator's and the system's; `secrets/` is offered as a home "
              "for it, never policed as one"),
        # EVERY FILE IN `secrets/`, whatever the operator called it.  The
        # README tells them to put a TLS key here and nothing checked it: a
        # key at 0644 was invisible until 2026-09-20.  `pattern` already
        # existed for the port-carrying serve logs; the easy, high-value use
        # of it was the one missing.  It catches molbuilder's own slip too --
        # `initconfig` wrote this directory's README at 0664.
        Place("a file in secrets/", secrets_dir, False, PRIVATE_FILE_MODE,
              "everything in this directory is a credential or names one",
              "*"),
        Place("the state directory", state_dir, True, None,
              "logs and measurements; the modes that matter are below it"),
        Place("logs/", logs_dir, True, PRIVATE_DIR_MODE,
              "the serve log carries a provider's client_secret, routed there "
              "deliberately to keep it out of a user-visible response"),
        Place("a serve log", logs_dir, False, PRIVATE_FILE_MODE,
              "measured 2026-09-12 carrying a client_secret", "serve-*.log"),
        Place("reports/", reports_dir, True, None,
              "per-run measurements, kept and grepped; a scientific report, "
              "with nothing in it to guard"),
        Place("the runtime directory", runtime_dir, True, PRIVATE_DIR_MODE,
              "ours, and when $XDG_RUNTIME_DIR is absent it falls back inside "
              "the state root, where the default umask is not good enough"),
        Place("a serve pidfile", runtime_dir, False, None,
              "an address, not a secret", "serve-*.pid"),
        # THE NOTEBOOK'S FOUR, absent from this table until 2026-09-15 while
        # the newest credential in the tree lived in two of them
        # (`plan.md` § 5n.8).  This module's own docstring says why that
        # mattered: the audit exists because a file "ARRIVES loose in ways no
        # writer controls" -- restored from a backup, copied off another
        # machine, or left by a molbuilder older than the 2026-09-14 fix,
        # which is exactly the measured 0664 case.  The writers are careful
        # now; nothing was WATCHING.
        Place("a notebook log", logs_dir, False, PRIVATE_FILE_MODE,
              "jupyter-server prints its own URL with the token in it at "
              "every start -- measured 2026-09-14 at 0664 with 14 "
              "occurrences of `token=`, and the token reaches a live kernel",
              "jupyter-*.log"),
        Place("a notebook pidfile", runtime_dir, False, None,
              "an address, not a secret -- the shepherd's, beside the "
              "serve pidfile", "jupyter-*.pid"),
        # NOT a credential_store, and the distinction is the whole definition:
        # this token is not KEPT.  It is regenerated at every start, deleted
        # at every clean stop, and meaningless without the live process it
        # authenticates to -- one fact with the pidfile beside it, and
        # splitting them would let a token outlive its pidfile.  It is still
        # 0600, which is the rule that protects it.  (Caveat: `runtime_dir`
        # falls back inside the state root when $XDG_RUNTIME_DIR is unset --
        # cron, detached ssh, containers -- so "erased at logout" is not
        # guaranteed; the harm is bounded because a token to a dead server
        # authenticates nothing.)
        Place("a notebook runtime file", runtime_dir, False,
              PRIVATE_FILE_MODE,
              "it holds the notebook's TOKEN, which authenticates a browser "
              "to a live kernel -- arbitrary code execution as this account",
              "jupyter-*.json"),
        Place("the framed Lab's home", jupyter_lab_home, True,
              PRIVATE_DIR_MODE,
              "molbuilder's own Lab settings and per-server workspaces; it "
              "sits in the state root, where the default umask is not good "
              "enough"),
    )


def misplaced() -> List[str]:
    """Credential stores that do not live in `secrets/`.

    **The location rule, made executable.**  `configuration.md` § 3.1 states
    that every credential molbuilder keeps is in `secrets/`; until 2026-09-20
    that was prose only, and prose cannot be executed -- which is this
    module's whole reason for existing.  Measured that day: `session_key()`
    could be pointed back at the config root and 222 tests still passed,
    because every test and every audit row asks the SAME resolver and so moves
    with it.  Nothing compared the answer against the rule.

    **EVERY COMPARISON IS ON A RESOLVED PATH**, and that is not a detail.  The
    first version of this function compared the paths as spelled, and so let
    through the very regression it was written for, one component longer:

        secrets/../secret_key   ->  really the config root, reported CLEAN

    `Path.parents` does not collapse ``..`` and does not follow symlinks, so a
    credential could also be a link out of the tree -- to a world-readable
    file, or off the encrypted volume entirely -- and this said nothing while
    `findings` printed a `chmod` that would fix the target and leave the
    credential where it was.  `resolve()` on both sides closes both.

    **A resolver that raises is a FINDING, not a pass.**  It read `except
    Exception: continue`, i.e. *"cannot tell, therefore fine"* -- the shape
    `access-control.md` § 8 rule 1 argues against everywhere else in this
    codebase.  It matters in practice: `monitor._secrets_dir()` raises
    `ModuleNotFoundError` by design when the shipped companion is absent.

    Unlike the mode check this does not need the file to exist: a resolver
    pointing outside `secrets/` is wrong whether or not anything has been
    written there yet, and catching it before first use is the point.
    """
    out: List[str] = []

    # THE ANCHOR ITSELF IS CHECKED FIRST.  Everything below is measured
    # against `secrets_dir()`, so if that moved, every credential could follow
    # it out of the config directory and this function would report nothing --
    # it would be comparing them against the wrong place and finding agreement.
    try:
        root = Path(config_dir()).resolve()
        home = Path(secrets_dir()).resolve()
    except Exception as exc:
        return [f"the secrets directory could not be resolved ({exc}), so "
                f"where credentials belong cannot be established.  Nothing "
                f"below was checked."]
    if not (root == home.parent and home.name == SECRETS_DIRNAME):
        out.append(
            f"the secrets directory resolves to {home}, which is not "
            f"{root / SECRETS_DIRNAME} -- credentials would follow it out of "
            f"the config directory and every check below would agree with "
            f"it.\n  This is a CODE defect: `config_dir.secrets_dir` no "
            f"longer answers with a child of `config_dir` "
            f"(docs/configuration.md 3.1).")

    for place in places():
        if not place.credential_store:
            continue
        try:
            target = Path(place.resolve()).resolve()
        except Exception as exc:
            out.append(
                f"{place.what}: its resolver raised {type(exc).__name__} "
                f"({exc}), so whether it is in {home} cannot be established.\n"
                f"  Treated as a finding rather than a pass: a check that "
                f"cannot tell must not report all clear.")
            continue
        if home == target or home in target.parents:
            continue
        out.append(
            f"{place.what} resolves to {target}, which is outside "
            f"{home} -- every credential molbuilder keeps belongs there "
            f"({place.why}).\n"
            f"  This is a CODE defect, not a permissions one: the resolver "
            f"that owns this file points out of the secrets directory "
            f"(docs/configuration.md 3.1).")
    return out


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
    out: List[str] = list(misplaced())
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

    **ONE LIST, NOT THREE** *(proposed and declined 2026-09-20)*.  Printing the
    three kinds under their own headers -- shadow, misplaced, mode -- restates
    what each message already carries: a mode finding ends with ``Fix it with:
    chmod ...``, a misplaced one says ``This is a CODE defect, not a
    permissions one``.  The split would cost four call sites their one-call
    shape to tell a reader what the line in front of them already says.
    """
    from .runtime_config import machine_config_shadow
    shadow = machine_config_shadow()
    return ([shadow] if shadow else []) + misplaced() + findings()
