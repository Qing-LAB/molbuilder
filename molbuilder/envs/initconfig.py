"""Seed the per-user config directory, once, at install time.

**The problem this closes is named in the contract itself.**
`execution/running-a-job.md` § 5.2, on the activation:

    has **no default**: a target whose record carries none is refused at
    prep ... On a fresh install that would bite a workstation first.

Nothing created the config directory or this machine's record, so on a fresh
machine every prep refused a target whose record could not say how to enter an
environment -- and the record could not say it, because nobody had been asked.

**Install time is when the answer is known and nowhere else is.**  The
installer has just located the env manager and asked the person to confirm it;
that is the one moment in the program's life where "how does this machine enter
a conda env" is both known and being discussed.  Every later surface can only
report that nobody ever said.  The answer goes into ``molbuilder.json``'s
``env_init`` -- where a machine molbuilder is installed on declares how a shell
enters an environment there -- and the probe copies it into this machine's
record, which is what every prep reads (`configuration.md` § 4).

**Asked, not sniffed.**  ``activation`` is DECLARED, never detected --
``detect_conda_activation`` was deleted 2026-08-13 (V22) for having zero
callers, and `running-a-job.md` § 5 is why.  This module does not restore it:
the value arrives as an ARGUMENT, from a person answering a prompt (or from
``--yes`` taking the recommendation, printed).  A declaration made by the
operator at install time is what the rule asks for; what it forbids is a
generator guessing at emit time.

**Never overwrites what a PERSON wrote.**  Every step reports ``created``,
``kept`` or ``rewritten``, and nothing a person wrote is replaced --
`configuration.md` M-6, *"the probe asks before it overwrites"*, generalised
to the whole directory.  Two things are written into, and say so: the two
READMEs (below), and a ``molbuilder.json`` that declares no
``env_init.activation``, which gains the one asked (that section alone,
2026-10-02's R17; `configuration.md` § 2.1c, "Two things it rewrites").
Re-running is a no-op that prints what is already there, which is what makes it
safe to call from ``bootstrap`` unconditionally.

**The two READMEs are the exception** *(user, 2026-09-20: "always overwrite")*.
They are MOLBUILDER'S OWN TEXT, not the operator's, and the rule above was
protecting the wrong thing: when the credentials moved into ``secrets/`` the
seeded README went on telling every pre-existing install that they sit beside
``molbuilder.json``, and it is the file an operator opens precisely when they
do not already know. A wrong map is worse than no map. They report
``rewritten`` when the text on disk differs and ``kept`` when it does not, so
re-running is still a no-op -- the CONTENT is compared, which is what keeps
that true while making a stale one impossible.

**No new writer.**  The record goes through ``resolve_environment`` ->
``diagnostics.local_facts`` -> ``record.probe_queues`` -> ``write_environment``,
the same steps in the same order ``jobset probe --write`` takes, so the two
cannot disagree about what this machine is.  ``local_facts`` and the queue
probe were inline in that command until this module became their second
caller -- the queues not until 2026-10-02, so a cluster was seeded with none
(R4).  This module contributes the DIRECTORY, the CONFIG and the ANSWER it
asked for; the record it delegates.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from ..config_dir import (PRIVATE_FILE_MODE, config_dir, ensure_private_dir,
                          secrets_dir)

__all__ = [
    "Step", "conda_hook", "seed_document", "seeding_blockers",
    "ensure_dirs", "seed_machine_config", "seed_environment_record",
    "init_config",
]


@dataclass(frozen=True)
class Step:
    """One thing the seeding either made or found already there.

    ``action`` is ``"created"``, ``"kept"`` or ``"rewritten"``.  The third
    is for the two READMEs this module authors, and for a ``molbuilder.json``
    given the ``env_init`` it lacked: the module never deletes, and never
    replaces anything a person wrote (see the note at the top).  ``note`` carries what a person
    needs to act on: which activation form was written, or why a preamble was
    not.
    """
    path: Path
    action: str
    note: str = ""

    @property
    def created(self) -> bool:
        return self.action == "created"

    @property
    def rewritten(self) -> bool:
        return self.action == "rewritten"


def conda_hook(conda_binary: Optional[str]) -> Optional[str]:
    """The shell line that makes ``conda activate`` work in a NON-interactive
    shell, or ``None`` when this installation has no such hook.

    ``conda activate`` is a shell FUNCTION.  It exists in an interactive shell
    because the profile sourced ``etc/profile.d/conda.sh``; a wrapper script
    runs under ``bash job.sh`` with no profile, where the function is simply
    not defined and the job dies on the activation line.  Sourcing the hook is
    the preamble that fixes it, and the installer is holding the path to the
    binary it needs to derive it.

    Returns ``None`` for micromamba, which ships no ``conda.sh`` at all, and
    whenever the hook file is not actually on disk.  A preamble that names
    a file that does not exist is worse than no preamble: it fails later, on
    the cluster, inside a job.  **Checked, not assumed.**

    **The root comes from the MANAGER, not from its binary's path**
    (`installation.md` M2).  This used to be
    ``Path(conda_binary).resolve().parent.parent`` -- which is the install root
    for ``<root>/bin/conda`` and for ``<root>/condabin/conda``, is ``/usr`` for a
    distro-packaged ``/usr/bin/conda``, and says nothing at all when the thing on
    PATH is the shell wrapper a cluster module provides.  `<mgr> info --json`
    answers with ``root_prefix`` outright, and the value this function returns is
    WRITTEN INTO THE USER'S ``molbuilder.json`` -- so a guess here does not fail
    here: it fails in a job, on a cluster, weeks later.
    """
    if not conda_binary:
        return None
    root = _manager_root(conda_binary)
    if root is None:
        return None
    hook = root / "etc" / "profile.d" / "conda.sh"
    if not hook.is_file():
        return None
    return f"source {hook}"


def _manager_root(conda_binary: str) -> Optional[Path]:
    """Ask the manager where it is installed.  ``None`` if it will not say.

    No fallback to a path derivation, on purpose: the caller's answer is written
    into a config file, and *"I could not determine this"* is a better thing to
    write than a plausible guess -- `conda_hook` then returns ``None`` and
    `init-config` seeds no preamble, which is a state the seeded file explains.
    """
    from ..diagnostics import manager_info
    root = manager_info(conda_binary).get("root_prefix")
    return Path(root) if root else None


def seed_document(activation: str, preamble: Optional[str] = None,
                  projects: Optional[Path] = None) -> "dict":
    """The contents of a freshly seeded ``molbuilder.json``.

    **Your preferences, and how a shell enters an environment on this
    machine** (`configuration.md` § 4): ``env_init`` holds the answer just
    asked for, and every other section a person may fill is present and
    empty, each with a comment saying what it is for.  Neither a machine's
    measured facts -- its queues, its cores -- nor a job's values, which a job
    states itself.

    The ``_``-prefixed keys are comments.  `running-a-job.md` § 5 makes them
    legal by name -- *"a key starting with ``_`` is a comment ... and is
    ignored by design"* -- and they are the whole of the guidance this file
    carries: each one names the COMMAND and the DOCUMENT that own its subject
    rather than restating either.  `configuration.md` line 42 is the reason:
    *"A second copy is a copy that drifts."*
    """
    doc = {
        "_README": [
            "molbuilder's server-wide configuration for THIS machine:",
            "your PREFERENCES, and how a shell enters an environment here",
            "(`env_init`).  Seeded by `molbuilder envs init-config` (which",
            "`bootstrap` runs).",
            "",
            "HOW TO READ THIS FILE.  Every section is OPTIONAL and starts",
            "empty; fill in only what you need.  A key starting with `_` is a",
            "comment and is ignored.  An UNKNOWN key is REFUSED, so a typo is",
            "named rather than silently doing nothing.",
            "",
            "WHAT IS NOT HERE, and where it is:",
            "  a MACHINE's facts -- cores, GPUs, scheduler, the queues you",
            "      can reach -- are its RECORD, environment.json beside this",
            "      file, written ON that machine by `molbuilder jobset probe",
            "      --write`, which also copies `env_init` into it;",
            "  a JOB's values -- queue, wall, memory, ranks, cores per rank,",
            "      GPUs -- are stated by the job: its task.json, or the",
            "      prep/launch flags.  Prep refuses one stated nowhere.",
            "",
            "Secrets are SEPARATE 0600 files; this file carries their PATHS,",
            "never their bytes.  See secrets/README beside this file.",
            "",
            "Every key this file may hold, and what it is for:",
            "docs/configuration.md section 4.",
        ],
        "_launch": [
            "How `jobset launch` sends a job when no --mode is given:",
            "\"direct\" runs it here with bash, \"submit\" hands it to the",
            "scheduler.  Unset, launch asks for --mode.",
            "-> docs/execution/running-a-job.md section 5.4",
        ],
        "launch": {},
        "_env_init": [
            "How a shell on THIS machine enters a conda environment -- asked",
            "at install, which is why it is filled in below.  `activation` is",
            "\"conda activate\" or \"source activate\" and has NO default.",
            "`preamble` is shell run BEFORE it: the `module load` lines on a",
            "cluster, or sourcing conda's hook on a workstation.",
            "`molbuilder jobset probe --write` copies both into every record",
            "it writes; prep reads them from the target's record.",
            "-> docs/execution/running-a-job.md section 5.2",
        ],
        "env_init": {"activation": activation},
        # THE USER READS THIS BLOCK, so it says what to do and not what we
        # learned.  It invited `logs`, `run` and `reports` until 2026-09-12 --
        # keys retired on 2026-08-31 and REFUSED since, so a person with a
        # quota'd $HOME who followed the advice in their own seeded config made
        # every later read of it raise.  The file molbuilder generates was
        # telling them how to brick it.  `configuration.md` 2.1d owns the rule
        # (the `serve` supervisor is L1 and the config reader L2, so a
        # config-derived answer was unreachable); `runtime_config._read_paths`
        # states it at the reader.  Three places said it, one was right.
        "_paths": [
            "Where molbuilder keeps things that are not its own code.",
            "`projects` is the project tree.  Every surface resolves it",
            "through one door, so setting it here moves the tree for all of",
            "them at once -- sidebar, CLI verbs, workspace store,",
            "pseudopotentials.  Where it resolves from is PRINTED by every",
            "jobset verb, by `envs doctor` and by `serve`, so you never have",
            "to infer it.",
            "`projects` is the ONLY key here; a relative value resolves",
            "against the molbuilder root.  To put logs, pidfiles and",
            "reports somewhere other than ~/.local/state (a small $HOME,",
            "a large scratch), set XDG_STATE_HOME and XDG_RUNTIME_DIR --",
            "they are not keys in this file (configuration.md 2.1d).",
        ],
        "paths": {},
        "_auth": [
            "SIGN-IN -- the one section you should NOT hand-write.",
            "    molbuilder auth-setup        (CAS or Google; --help for more)",
            "writes this block and preserves everything else in this file.  It",
            "is ABSENT here on purpose: an empty `auth` is refused, because",
            "the block must name at least one provider.  For GitHub /",
            "Microsoft / ORCID see ops/deployment.md and",
            "ops/access-control.md.",
        ],
        "_tls": [
            "HTTPS for the server -- paths to the cert and key, never bytes:",
            "    \"tls\": {\"cert\": \"...fullchain.pem\",",
            "             \"key\":  \"...privkey.pem\"}",
            "secrets/ beside this file is the suggested home for the key.",
            "-> docs/ops/deployment.md section 5",
        ],
        "tls": {},
        "_admin": ["Who may use the operator-only web actions: restarting",
                   "the server, the rate limiter's block list.",
                   "-> docs/ops/access-control.md"],
        "admin": {},
        "_rate_limit": ["Request throttling for the web surface; its keys "
                        "and defaults:",
                        "-> docs/ops/deployment.md section 4"],
        "rate_limit": {},
        "_envs": [
            "Which conda environment each backend runs in, when it is not",
            "the default name, and `manager`: the env-manager binary by",
            "absolute path when the one on PATH is not the one to use.",
            "-> docs/ops/installation.md",
        ],
        "envs": {},
        "_checkpoint": ["Which run files are too large to save with a folder.",
                        "-> docs/execution/checkpointing.md"],
        "checkpoint": {},
        "_retired": [
            "REFUSED by name if you port an older file here, each with what",
            "to do instead: `scheduler`, `script_generation` and `execution`",
            "(retired 2026-10-02 -- `script_generation` is now `env_init`,",
            "`execution` is now `launch`),",
            "`notify_keys_file`, `notify_route`, `secret_key_file`, and",
            "top-level `cert`/`key`.",
        ],
    }
    if preamble:
        doc["env_init"]["preamble"] = preamble
    if projects is not None:
        doc["paths"] = {"projects": str(projects)}
    return doc


def seeding_blockers() -> List[str]:
    """Why seeding would fail, answerable BEFORE anything slow runs.

    **Asked first because it is asked last.**  `envs bootstrap` seeds the config
    directory at the END, after up to forty minutes of conda installs, and a
    failure there is deliberately not fatal -- built envs are worth keeping.  The
    cost of that ordering is that a read-only or unreachable config root is
    discovered once the expensive work is already done, and the person is told
    to fix it and run `init-config` themselves.  Every reason in this list is a
    fact about the filesystem that was equally true before the first install, so
    bootstrap now asks at the start and says so while the user can still decide.

    Returns human-readable lines, each naming the path and what to do; empty
    means nothing here will stop the seeding.  **Warnings, not refusals** -- the
    envs are still worth installing, and `config_dir` is movable
    (`MOLBUILDER_CONFIG_DIR`), which is usually the right fix on a cluster where
    $HOME is read-only or over quota.

    This checks only what this module OWNS: whether the directory it writes can
    be written.  Whether the two install-time questions can be ASKED is the
    CLI's to know -- it owns the prompting -- and it is the other half of the
    same late surprise.
    """
    root = config_dir()
    blockers: List[str] = []
    if root.exists() and not root.is_dir():
        blockers.append(
            f"{root} exists and is NOT a directory, so the config directory "
            f"cannot be created there.  Move that file aside, or point "
            f"MOLBUILDER_CONFIG_DIR somewhere else.")
        return blockers
    if root.is_dir():
        if not os.access(root, os.W_OK):
            blockers.append(
                f"{root} is not writable, so molbuilder.json cannot be "
                f"seeded there.  Fix its permissions, or set "
                f"MOLBUILDER_CONFIG_DIR to a directory you own.")
        return blockers
    # Not there yet: the question is whether we could MAKE it, which is a
    # property of the nearest ancestor that does exist.
    for parent in root.parents:
        if parent.exists():
            if not os.access(parent, os.W_OK):
                blockers.append(
                    f"{root} does not exist and cannot be created: "
                    f"{parent} is not writable.  Set MOLBUILDER_CONFIG_DIR to "
                    f"a directory you own (on HPC, scratch is the usual "
                    f"answer -- it also keeps secrets off an NFS $HOME).")
            break
    return blockers


def _ensure_root() -> bool:
    """Make the config directory if it is not there.  ``True`` if it made it.

    Through `config_dir.ensure_private_dir`, which is the ONE creator for a
    directory in this tree and states the mode once (2026-09-13).  It does not
    tighten a directory that is already there, which is this function's own
    rule: seeding seeds, `envs doctor` reports what arrived loose.

    Private, and shared by both doors, because ``seed_machine_config`` is
    reachable on its own -- a public function that works only if you happened
    to call a different one first is a trap, and it caught its own test.
    """
    root = config_dir()
    if root.is_dir():
        return False
    ensure_private_dir(root)
    return True


def ensure_dirs() -> List[Step]:
    """The config directory and the ``environments/`` beside it.

    ``0700`` on the config directory: it holds the session key, the OAuth
    client secret and the notify keys, and `configuration.md` § 2.1b already
    requires ``0600`` on those files -- a world-readable directory around them
    is the same mistake one level up.  An EXISTING directory's mode is left
    alone; this seeds, it does not police (``envs doctor`` is where a
    permissions audit would belong).

    ``environments/`` is created empty and that is not pointless: it is the
    answer to *"where do I put the record my colleague sent me from the
    cluster"*, and an empty directory answers it where an absent one does not.
    """
    steps: List[Step] = []
    root = config_dir()
    made = _ensure_root()
    steps.append(Step(root, "created", "mode 0700 -- it holds secrets")
                 if made else Step(root, "kept"))

    from ..scheduler import environments_dir
    envs = environments_dir()
    if envs.is_dir():
        steps.append(Step(envs, "kept"))
    else:
        # 0700 like its parent.  It sits inside a directory that holds the
        # session key and the OAuth client secret, and a looser mode on a
        # child of that is the same mistake one level down.  (It was created
        # at the umask default until 2026-09-12, so 0775 on most systems.)
        ensure_private_dir(envs)
        steps.append(Step(envs, "created",
                          "mode 0700; drop a colleague's "
                          "`probe --name <cluster>` record here"))
    steps.extend(_readme(envs, _ENVIRONMENTS_README,
                         "the probe runs ON THE TARGET; this is where its "
                         "record is copied TO"))
    steps.extend(_seed_secrets_dir())
    return steps


#: What ``environments/README`` says.  A FILE, for the same reason the secrets
#: one is: the person who needs it is looking at the directory, and a
#: conda-only install has no checkout to read docs from.
_ENVIRONMENTS_README = 'molbuilder — environments\n=========================\n\nOne file per MACHINE YOU PREPARE FOR BUT ARE NOT ON: `<name>.json`, where the\nname is yours and becomes `--target <name>`.\n\nTHIS MACHINE\'S OWN RECORD IS NOT IN HERE.  It is `../environment.json`, one\nlevel up.  The two are different scopes, not copies — `record_scopes()` walks\ncalculation → target → machine and takes the first match.\n\nTHE PROBE RUNS ON THE TARGET, NOT HERE\n    This is the part that catches people.  A record describes cores, GPUs, the\n    scheduler and the queues you can actually reach; none of that is knowable\n    from your laptop, and a probe run here would faithfully measure YOUR box\n    and label it with the cluster\'s name.  So:\n\n    1. ON THE TARGET -- a login node is fine, the scheduler is read from\n       `sinfo` rather than from being on a compute node:\n\n           molbuilder jobset probe --write --name sol\n           # -> wrote ~/.config/molbuilder/environments/sol.json\n\n       `--write` shows what it measured and asks before overwriting an\n       existing record, difference by difference.  Silence keeps the record;\n       `--yes` takes every probed value.  The record carries a copy of that\n       machine\'s `env_init` -- how a shell enters an environment there, from\n       ITS molbuilder.json.\n\n    2. COPY IT HERE:\n\n           scp cluster:~/.config/molbuilder/environments/sol.json \\\n               ~/.config/molbuilder/environments/\n\n    3. CONFIRM IT LANDED AND PARSES:\n\n           molbuilder jobset machines\n\n       This prints every record, its path and when it was measured.  It is the\n       only step that answers "did the copy work?" -- a record that is present\n       but corrupt is LISTED AND MARKED, never skipped, because a silently\n       dropped record looks exactly like one that was never copied.\n\n    Then `prep --target sol` sizes for Sol from anywhere.\n\nWHAT DOES NOT BELONG HERE\n    A record written from scratch.  These are records of measurement; a\n    copied `env_init` that is wrong for its machine is the one thing edited\n    here by hand.  A fact the probe cannot see is `--set` on the probe, run on\n    that machine (the record says you asserted it); what you WANT is a\n    preference in `molbuilder.json`.\n\n    An empty directory is the normal state if you only ever run locally.\n\nReference: docs/execution/preparing-for-another-machine.md § 1a (these commands,\nverified end to end), docs/configuration.md § 5 (the record\'s schema and the\nfact-vs-preference rule).\n'


def _readme(directory: Path, text: str, note: str) -> List[Step]:
    """Write ``<directory>/README``, or rewrite it when its text is out of
    date.  One writer for both.

    **Overwritten when it is out of date** *(user, 2026-09-20)*.  It used to be
    left alone like everything else here, on the reason that a person may have
    added notes -- but this is molbuilder's own text, and on 2026-09-20 the
    credentials moved into `secrets/` while every install seeded before that
    kept a README saying they live one level up.  That file is what an operator
    opens when they do NOT already know where their credentials are, so a stale
    one sends them to the wrong place with full confidence.  The write is
    conditional on the CONTENT differing, not on a timestamp, so a second run
    is still a no-op and still reports ``kept``.

    **Written 0600 when it lands in `secrets/`** *(2026-09-20)*.  A plain
    `write_text` takes the umask, which here meant 0664 -- inside a directory
    whose own README says every file molbuilder keeps there is 0600, and
    which the placement audit checks (its row for this README).  So `envs
    init-config` seeded a file that `envs doctor` would immediately report.
    It is only a README, but a seeder that trips its own audit teaches an
    operator to ignore the audit, which is the expensive part.
    `environments/README`
    keeps the umask: that directory holds machine records, not credentials.
    """
    path = directory / "README"
    try:
        current = path.read_text(encoding="utf-8")
    except OSError:
        current = None
    if current == text:
        return [Step(path, "kept")]
    path.write_text(text, encoding="utf-8")
    if directory == secrets_dir():
        path.chmod(PRIVATE_FILE_MODE)
    if current is None:
        return [Step(path, "created", note)]
    return [Step(path, "rewritten",
                 "it described the old layout -- replaced with the current one")]


#: What ``secrets/README`` says.  It is a FILE rather than a docstring because
#: the person who needs it is looking at the directory, not at the source --
#: and because a conda-only install has no checkout to read docs from.
_SECRETS_README = 'molbuilder — secrets\n====================\n\nThis directory holds EVERY credential molbuilder keeps.\nIt is created mode 0700 (owner only), and every file molbuilder keeps here\nis 0600.  A file you name in molbuilder.json and keep here -- a TLS key -- is\nyours and the system\'s: molbuilder reads it and says nothing of its mode.\nNothing here is ever printed: `config_provenance` logs only the sections\nflagged safe, and any section holding a secret — or a path to one — is\nexcluded by construction.\n\nWHY THE MODES MATTER\n    0700 on this directory, 0600 on each file molbuilder keeps.  `molbuilder.json` is itself\n    checked for 0600 and WARNS (never refuses — refusing would lock you out\n    of your own server).  A world-readable directory around 0600 files is the\n    same mistake one level up, so this one is tight from the start.\n\n        ls -l ~/.config/molbuilder ~/.config/molbuilder/secrets\n        chmod 700 ~/.config/molbuilder/secrets\n        molbuilder envs doctor     # names each of molbuilder\'s own files\n                                   # that arrived looser, with its chmod\n\nWHAT BELONGS HERE — everything sensitive\n    Two kinds, differing in WHO NAMES them, not in where they sit:\n\n  1. FIXED HOME.  molbuilder resolves these itself and molbuilder.json\n     CANNOT name them — one function each, so a reader and a writer can\n     never mean different files:\n\n      secret_key           the Flask session key.  Created on FIRST SERVER\n                           RUN at 0600.  Do not make it by hand; deleting it\n                           logs everyone out and a new one appears.\n                           (`secret_key_file` in config is REFUSED.)\n      notify               the run-report CHANNELS file — see below.\n      notify_keys          the operator\'s run-report signing keys.\n                           (`notify_keys_file` in config is REFUSED.)\n      <kind>_client_secret an OAuth provider\'s client secret, one per kind:\n                           google_client_secret (`molbuilder auth-setup`\n                           writes it), github_client_secret,\n                           microsoft_client_secret, orcid_client_secret.\n                           (`client_secret_file` in config is REFUSED.)\n\n  2. OPERATOR-NAMED.  Referenced by a PATH in molbuilder.json, so the name\n     is yours and this is their suggested home:\n\n      the TLS private key (and cert, if not system-managed)\n          "tls": {"cert": "~/.config/molbuilder/secrets/fullchain.pem",\n                  "key":  "~/.config/molbuilder/secrets/privkey.pem"}\n\nHOW ANYTHING REACHES A FILE IN HERE\n    Through the one function that owns it, never by building the path:\n\n      config_dir.secrets_dir()            this directory\n      config_dir.session_key()            secret_key\n      config_dir.client_secret(kind)      <kind>_client_secret\n      monitor.default_notify_path()       notify\n      monitor.notify_keys_path()          notify_keys\n\n    Review keeps each name in its one home: a module that joined one of\n    these names into a path itself would be a second answer to where the\n    file lives.  `tests/test_config_dir_has_one_home.py` fails if any door\n    stops moving with MOLBUILDER_CONFIG_DIR.  Set that variable and the\n    whole directory relocates — which is how you keep credentials off an\n    NFS-mounted $HOME on a login node.\n\n    THE FOUR ABOVE MOVED HERE ON 2026-09-20 (user).  They used to sit beside\n    molbuilder.json, one level up, on the rule "this directory is for what\n    the config NAMES".  A directory called `secrets` that did not hold the\n    secrets misled everyone who opened it.  What mattered was never WHICH\n    directory — only that each file has exactly one home and one resolver,\n    and that is unchanged.\n\nRUN-REPORT CHANNELS — the `notify` file in this directory\n    Shape: {"channels": {"<name>": {"url": ..., "kind"?: ..., "key"?: ...}}}\n    `kind` is "molbuilder" | "slack" | "discord"; omitted, it is read off the\n    URL\'s host.  Absent file means no notifier and the run proceeds exactly as\n    with the feature off; a malformed file says so in the monitor log and\n    carries on, and one bad channel does not cost the others.\n\n    THE EASY WAY is the web UI (Settings → run reports), which writes this\n    file at 0600 for you.  `molbuilder notify-token` is for the OTHER side:\n    it writes notify_keys and PRINTS the JSON to paste on the machine that\n    runs the job.  The shapes below are for reading and for testing.\n\n    EXAMPLES.  The URLs are <ANGLE-BRACKET> placeholders rather than\n    realistic fakes, deliberately: a fake that LOOKS real IS a credential as\n    far as a secret scanner is concerned -- GitHub\'s push protection rejected\n    an earlier draft of THIS FILE for carrying a "Slack Incoming Webhook\n    URL".  Which is the clearest demonstration of the point below: for Slack\n    and Discord the URL *is* the secret.  Substitute the bracketed parts:\n\n      {\n        "channels": {\n          "local": {\n            "kind": "molbuilder",\n            "url":  "http://127.0.0.1:8765/hook",\n            "key":  "mock-local-key-not-a-real-secret"\n          },\n          "team-slack": {\n            "kind": "slack",\n            "url":  "https://hooks.slack.com/services/<TEAM-ID>/<CHANNEL-ID>/<TOKEN>"\n          },\n          "team-discord": {\n            "kind": "discord",\n            "url":  "https://discord.com/api/webhooks/<CHANNEL-ID>/<TOKEN>"\n          }\n        }\n      }\n\n    NOTE THE TWO SHAPES, because they differ in where the credential is:\n      * Slack and Discord put the credential IN THE URL.  A third party hands\n        you nothing but a URL, so the URL *is* the secret — which is why this\n        file is 0600 even when it looks like it holds no key.\n      * a molbuilder listener takes a plain url plus a `key` that SIGNS the\n        body and NEVER TRAVELS.  That is the shape to prefer when you control\n        the receiver: a leaked URL cannot be used to post as you.\n\n    A LOCAL LISTENER is the way to try this without a third party.  Point a\n    "molbuilder" channel at 127.0.0.1 and run anything that answers — the\n    signature is computed over the body with `key`, so a listener that does\n    not check it will still receive the report and show you the shape.\n\nIF YOU BACK THIS UP\n    Back up the whole config directory, not just this folder, and treat the\n    copy with the same care.  `secret_key` is the one file whose loss is\n    harmless (sessions end, a new key appears).  A leaked webhook URL, OAuth\n    client secret or TLS key is not.\n\nReference: docs/configuration.md § 2.1b (the mode rule), § 2.3 (how a secret\nis written -- atomically, so an interrupted write cannot destroy the one it\nreplaces), § 3.1 (the whole directory tree), § 2.1e (the session\nkey\'s one home), docs/execution/run-reports.md § 4.1b (channel kinds),\ndocs/ops/access-control.md (what sign-in exposes).\n'


def _seed_secrets_dir() -> List[Step]:
    """``secrets/`` and the README that says how to treat it.

    **EVERY credential molbuilder keeps is in here** *(user, 2026-09-20)*, in
    two kinds that differ in who NAMES the file, not in where it sits:

    * **fixed home** -- ``secret_key``, ``notify``, ``notify_keys`` and each
      OAuth kind's ``<kind>_client_secret``.  Each resolves through one
      function, and ``molbuilder.json`` cannot name any of them:
      ``secret_key_file``, ``notify_keys_file`` and ``client_secret_file`` are
      REFUSED rather than ignored, which is what stops a reader and a writer
      meaning different files (§ 2.1e, § 3.1).
    * **operator-named** -- the cert files: a TLS key or certificate.  The
      config names these by path, so this is their suggested home.

    THIS DOCSTRING DESCRIBED THE OPPOSITE UNTIL 2026-09-20, and so did the note
    this function prints: that the directory was only for what config names,
    and that three secrets *"CANNOT live here"*.  They moved in; the text did
    not follow, and `envs init-config` went on telling operators the reverse of
    what the code does.  The README it writes had already been corrected, so
    one function handed out two contradictory descriptions of one directory.

    ``secrets/`` is no longer empty on a working installation -- the session key
    appears on first server run.
    """
    steps: List[Step] = []
    _ensure_root()
    d = secrets_dir()
    if d.is_dir():
        steps.append(Step(d, "kept"))
    else:
        ensure_private_dir(d)
        steps.append(Step(d, "created",
                          "mode 0700 -- 0600 on everything you put in it"))
    steps.extend(_readme(d, _SECRETS_README,
                         "the mode rule, the two kinds of credential in here, "
                         "the function each is reached through, and mock "
                         "channel examples"))
    return steps


def seed_machine_config(activation: str, preamble: Optional[str] = None,
                        projects: Optional[Path] = None) -> Step:
    """Write ``molbuilder.json`` if it is absent; otherwise report it kept --
    or, when it declares no ``env_init.activation``, give it the one asked,
    that section alone through the one writer (R17).

    Nothing else in an existing file is touched: it is a person's.  The ``note`` on
    a kept file says whether it still READS -- a file carrying a section
    retired since it was written is refused by every reader, and this is the
    run of the installer most likely to be the first to say so -- and whether
    it declares the activation the probe copies.
    """
    from ..runtime_config import machine_config_path, write_config_scope

    path = machine_config_path()
    if path.exists():
        note = _config_note()
        if "declares no env_init.activation" not in note:
            return Step(path, "kept", note)
        # ASKED, SO WRITTEN: a file with no `env_init` gets the answer just
        # given -- that section alone, through the one writer; nothing the
        # person wrote is touched.
        write_config_scope(
            {"_env_init": seed_document(activation)["_env_init"],
             "env_init": seed_document(activation, preamble)["env_init"]})
        return Step(path, "rewritten",
                    f'added env_init.activation = "{activation}"'
                    + _preamble_note(activation, preamble))
    _ensure_root()
    # THE ONE WRITER of this file (configuration.md § 2.3): its own path from
    # its own resolver, validated with the server's validator before a byte
    # lands, 0600 from the first byte.  Until 2026-09-14 this joined the
    # filename itself and wrote through `write_json` -- a second writer whose
    # seed was never validated (review C-Y1).
    write_config_scope(seed_document(activation, preamble, projects))
    return Step(path, "created", f'env_init.activation = "{activation}"'
                + _preamble_note(activation, preamble))


def _config_note() -> str:
    """What an already-present config is: readable or refused and why, and
    the activation it declares."""
    try:
        from ..runtime_config import get_env_init
        said = get_env_init().get("activation")
    except Exception as exc:            # the refusal says what to do
        return f"left as it is -- but it does not read: {exc}"
    if said:
        return f'left as it is (env_init.activation = "{said}")'
    return ("left as it is -- but it declares no env_init.activation, so the "
            "probe has nothing to copy and every prep for this machine "
            "refuses (running-a-job.md 5.2)")


def seed_environment_record() -> Step:
    """This machine's own ``environment.json``, via the probe's own doors --
    carrying the copy of ``env_init`` the probe makes.

    Delegated to ``resolve_environment`` + ``local_facts`` + ``probe_queues``
    + ``write_environment`` -- what ``jobset probe --write`` calls -- so there
    is one prober and one writer, and re-probing later cannot disagree with
    what was seeded here.  Ordered AFTER the config: the record copies its
    ``env_init``, so a record written first would carry nothing.  An existing
    record is kept -- re-probing is `jobset probe`'s, which asks before it
    overwrites.
    """
    import getpass
    from datetime import datetime, timezone

    from ..diagnostics import local_facts
    from ..runtime_config import get_env_init
    from ..scheduler import (machine_scope_path, read_environment,
                             resolve_environment, write_environment)
    from ..scheduler.record import probe_command, probe_queues
    # Its own directory: a public function that worked only after another one
    # had made it is a trap (`_ensure_root`).
    _ensure_root()
    path = machine_scope_path()
    if path.exists():
        before = read_environment(path)
        said = ((before.env_init if before is not None else None)
                or {}).get("activation")
        return Step(path, "kept", (
            f'activation "{said}"; re-probe with `{probe_command()}` when '
            f'the machine changes' if said else
            f"carries NO activation -- `{probe_command()}` copies env_init "
            f"into it"))
    # THE SAME STEPS ``jobset probe --write`` TAKES, in the same order: the
    # node, the three facts that travel (`local_facts`), the queues.  The
    # stamp too -- `detected_at` was null in every record seeded here.
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    env, _note = local_facts(resolve_environment(now_iso=now), get_env_init())
    queue_notes, _summary = probe_queues(env, getpass.getuser())
    write_environment(env, path)
    sg = getattr(env, "env_init", None) or {}
    note = "probed this machine"
    if env.domains:
        note += f"; {len(env.domains)} queue(s)"
    # WHAT THE QUEUE PROBE COULD NOT MEASURE is said here as `jobset probe`
    # says it -- a measurement that quietly did not happen is the defect
    # those notes exist for (W54 review: this dropped them).
    for n in queue_notes:
        note += f"; {n.rstrip('.')}"
    if sg.get("activation"):
        note += f'; carries activation "{sg["activation"]}"'
    else:
        note += "; carries NO activation -- prep for this machine will refuse"
    return Step(path, "created", note)


def _preamble_note(activation: str, preamble: Optional[str]) -> str:
    """What the recorded preamble means for the activation beside it -- said
    at install, where it can still be fixed, rather than by a job failing on
    its activation line."""
    if preamble:
        return f"; preamble {preamble!r}"
    if activation == "conda activate":
        # ``conda activate`` is a shell function that a non-interactive shell
        # has never defined, so no preamble here means a wrapper that will
        # fail inside the job.
        return ("; no preamble -- `conda activate` needs conda's hook "
                "sourced in a non-interactive shell, and none was found")
    # ``source activate`` is a script on PATH.  It needs no hook, so the
    # absence is the correct state and must not be reported as a miss --
    # which is what it said until 2026-09-08, on every machine that HAD a hook
    # and simply did not need it.
    return "; no preamble (source activate needs none)"


def init_config(activation: str,
                preamble: Optional[str] = None,
                probe: bool = True,
                projects: Optional[Path] = None) -> List[Step]:
    """Seed the whole directory.  Idempotent; returns what it did.

    ``activation`` and ``preamble`` go into ``molbuilder.json``'s
    ``env_init``, and the record the probe writes carries a copy
    (`configuration.md` § 4).  ``probe=False`` writes no record: `jobset probe
    --write` makes it later, copying the same answer.

    ``projects`` writes ``paths.projects``.  ``None`` leaves the section empty
    and the default applies -- which is the right answer on a workstation and
    the wrong one on a cluster home with a quota, so the CLI ASKS rather than
    assuming either (user, 2026-09-12).  Declared, never detected: the same
    rule ``activation`` follows, for the same reason.
    """
    # THE BLOCKERS FIRST.  `seeding_blockers()` holds the exact sentence for an
    # unwritable config root -- and `bootstrap` prints `init-config` as the
    # remedy -- while this function answered the same condition with a raw
    # `PermissionError` traceback from `mkdir`.  That is the *"a remedy the
    # program prints that it then refuses to run"* class this migration exists
    # to close, arrived at from the inside (D1).
    blockers = seeding_blockers()
    if blockers:
        raise RuntimeError("\n".join(blockers))

    steps = list(ensure_dirs())
    steps.append(seed_machine_config(activation, preamble, projects))
    if probe:
        steps.append(seed_environment_record())
    return steps
