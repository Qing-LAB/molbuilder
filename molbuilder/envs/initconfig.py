"""Seed the per-user config directory, once, at install time.

**The problem this closes is named in the contract itself.**
`execution/running-a-job.md` § 5.2, on ``script_generation.activation``:

    has **no default** -- if it is unset in every scope, rendering **any**
    wrapper refuses with an operator message pointing here ... On a fresh
    install that is the *"the ``.fdf`` saved but no ``.run.sh`` appeared"*
    symptom, and it bites a workstation first.

Nothing created ``molbuilder.json``, so on a fresh machine the chain ran:
no config file -> ``activation`` unset -> ``jobset probe --write`` recorded no
``script_generation`` (it reads the value from the config, `jobset/_cli.py`
§ "HOW THIS MACHINE ENTERS ITS ENVIRONMENT TRAVELS WITH THE RECORD") -> every
wrapper refused, and ``prep`` refused a target whose record could not say how
to enter an environment.  Four surfaces reporting one absent file.

**Install time is when the answer is known and nowhere else is.**  The
installer has just located the env manager and asked the person to confirm it;
that is the one moment in the program's life where "how does this machine enter
a conda env" is both known and being discussed.  Every later surface can only
report that nobody ever said.

**Asked, not sniffed.**  ``activation`` is DECLARED, never detected --
``detect_conda_activation`` was deleted 2026-08-13 (V22) for having zero
callers, and `running-a-job.md` § 5 is why.  This module does not restore it:
the value arrives as an ARGUMENT, from a person answering a prompt (or from
``--yes`` taking the recommendation, printed).  A declaration made by the
operator at install time is what the rule asks for; what it forbids is a
generator guessing at emit time.

**Never overwrites.**  Every step reports ``created`` or ``kept``, and a file
that exists is never touched -- `configuration.md` M-6, *"the probe asks before
it overwrites"*, generalised to the whole directory.  Re-running is a no-op
that prints what is already there, which is what makes it safe to call from
``bootstrap`` unconditionally.

**No new writer.**  The record goes through ``resolve_environment`` ->
``diagnostics.local_facts`` -> ``write_environment``, the same three steps in
the same order ``jobset probe --write`` takes, so the two cannot disagree about
what this machine is.  ``local_facts`` was inline in that command until this
module became its second caller; skipping it -- which the first draft here did
-- writes a record carrying no activation, which is the refusal this module
exists to remove, reproduced one file over.  This module contributes the
DIRECTORY and the CONFIG; the record it delegates.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from ..config_dir import config_dir

__all__ = [
    "Step", "conda_hook", "seed_document",
    "ensure_dirs", "seed_machine_config", "seed_environment_record",
    "init_config",
]


@dataclass(frozen=True)
class Step:
    """One thing the seeding either made or found already there.

    ``action`` is ``"created"`` or ``"kept"`` -- there is no third value,
    because the module never modifies and never deletes.  ``note`` carries
    what a person needs to act on: which activation form was written, or why
    a preamble was not.
    """
    path: Path
    action: str
    note: str = ""

    @property
    def created(self) -> bool:
        return self.action == "created"


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
    whenever the derived file is not actually on disk.  A preamble that names
    a file that does not exist is worse than no preamble: it fails later, on
    the cluster, inside a job.  **Checked, not assumed.**
    """
    if not conda_binary:
        return None
    root = Path(conda_binary).resolve().parent.parent
    hook = root / "etc" / "profile.d" / "conda.sh"
    if not hook.is_file():
        return None
    return f"source {hook}"


def seed_document(activation: str,
                  preamble: Optional[str] = None,
                  projects: Optional[Path] = None) -> "dict":
    """The contents of a freshly seeded ``molbuilder.json``.

    **Minimal on purpose.**  Only ``script_generation`` -- the one section with
    no default, whose absence stops the program.  Notably NOT ``scheduler``:
    partition and QOS are site facts, a wrong guess emits jobs that are
    rejected by the queue, and the door that learns them for real is
    ``jobset probe``, which writes them into ``environment.json`` where they
    belong (`configuration.md` M-1, *"the split is fact vs preference"*).

    The ``_``-prefixed keys are comments.  `running-a-job.md` § 5 makes them
    legal by name -- *"a key starting with ``_`` is a comment ... and is
    ignored by design"* -- and they are the whole of the guidance this file
    carries: each one names the COMMAND and the DOCUMENT that own its subject
    rather than restating either.  `configuration.md` line 42 is the reason:
    *"A second copy is a copy that drifts."*
    """
    doc = {
        "_README": [
            "molbuilder's server-wide configuration for THIS machine.",
            "Seeded by `molbuilder envs init-config` (which `bootstrap` runs).",
            "",
            "HOW TO READ THIS FILE.  Every section is OPTIONAL and starts",
            "empty; fill in only what you need.  A key starting with `_` is a",
            "comment and is ignored.  An UNKNOWN top-level key is REFUSED, so",
            "a typo is named rather than silently doing nothing.",
            "",
            "WHO FILLS WHAT.",
            "  YOU        preferences: execution, paths, admin, rate_limit,",
            "             tls, envs, checkpoint, scheduler.directives",
            "  A COMMAND  `molbuilder auth-setup` writes the `auth` block",
            "  NOT HERE   machine FACTS -- cores, GPUs, scheduler kind, the",
            "             queues you can actually reach -- are PROBED into",
            "             environment.json beside this file by",
            "             `molbuilder jobset probe --write`.  Never",
            "             hand-write them: configuration.md M-1 is the rule",
            "             (fact vs preference).",
            "",
            "Secrets are SEPARATE 0600 files; this file carries their PATHS,",
            "never their bytes.  See secrets/README beside this file.",
            "",
            "Full reference: docs/configuration.md section 4.",
        ],
        "_script_generation": [
            "How a generated wrapper enters its conda env.  THE ONE REQUIRED",
            "VALUE.  `activation` is \"conda activate\" or \"source activate\"",
            "and has NO default -- unset, EVERY wrapper refuses to render.  It",
            "was asked at install time, which is why it is filled in below.",
            "`preamble` is shell run BEFORE activation: the `module load`",
            "lines on a cluster, or sourcing conda's hook on a workstation.",
            "-> docs/execution/running-a-job.md section 5.2",
        ],
        "script_generation": {"activation": activation},
        "_execution": [
            "Defaults every calculation on this machine inherits -- the ask,",
            "before any per-job override.  Also settable per project in a",
            "`.molbuilder.json`, which wins.",
        ],
        "execution": {},
        "_scheduler": [
            "WHAT YOU WANT from the scheduler -- not what it IS.",
            "",
            "The facts (scheduler kind, node topology, and the",
            "(partition, qos) domains you can actually reach) are PROBED:",
            "    molbuilder jobset probe --write",
            "writes them to environment.json beside this file.  For a cluster",
            "you prep FOR but are not ON, add --name <cluster> and it lands",
            "in environments/<cluster>.json.",
            "",
            "What stays yours here is WHICH of the probed domains to use:",
            "    \"scheduler\": {\"directives\": {\"partition\": \"...\",",
            "                                  \"qos\": \"...\"}}",
            "`scheduler.routing` is retired and REFUSED -- it was the",
            "declarative form of what is now probed.",
            "-> docs/execution/scheduler.md",
        ],
        "scheduler": {},
        "_paths": [
            "Where molbuilder keeps things that are not its own code.",
            "`projects` is the project tree.  Every surface resolves it",
            "through one door, so setting it here moves the tree for all of",
            "them at once -- sidebar, CLI verbs, workspace store,",
            "pseudopotentials.  Where it resolves from is PRINTED by every",
            "jobset verb, by `envs doctor` and by `serve`, so you never have",
            "to infer it.",
            "`logs`, `run`, `reports` override the XDG state directories, so a",
            "small $HOME can keep its secrets while logs go to scratch.",
            "A relative value resolves against the molbuilder root.",
        ],
        "paths": {},
        "_auth": [
            "SIGN-IN -- the one section you should NOT hand-write.",
            "    molbuilder auth-setup        (CAS or Google; --help for more)",
            "writes this block and preserves everything else in this file.  It",
            "is ABSENT above on purpose: an empty `auth` is refused, because",
            "the block must name at least one provider.  For GitHub /",
            "Microsoft / ORCID see docs/ops/examples/molbuilder.json.example",
            "and ops/access-control.md.",
        ],
        "_tls": [
            "HTTPS for the server -- paths to the cert and key, never bytes:",
            "    \"tls\": {\"cert\": \"...fullchain.pem\",",
            "             \"key\":  \"...privkey.pem\"}",
            "secrets/ beside this file is the suggested home for the key.",
            "Top-level \"cert\"/\"key\" are retired and REFUSED by name.",
            "-> docs/ops/deployment.md section 5",
        ],
        "tls": {},
        "_admin": ["Who may administer this installation.",
                   "-> docs/ops/access-control.md"],
        "admin": {},
        "_rate_limit": ["Request throttling for the web surface.",
                        "-> docs/ops/access-control.md"],
        "rate_limit": {},
        "_envs": [
            "This machine's conda setup, when it is not the default:",
            "`manager` pins the env-manager binary by absolute path when it is",
            "not on PATH, and the per-backend env NAMES if yours differ.",
            "-> docs/ops/installation.md",
        ],
        "envs": {},
        "_checkpoint": ["Run-checkpoint behaviour.",
                        "-> docs/execution/checkpointing.md"],
        "checkpoint": {},
        "_retired": [
            "REFUSED by name if you port an older file here, each with what",
            "to do instead: `notify_keys_file`, `notify_route` (retired",
            "2026-08-31), `secret_key_file`, `scheduler.routing`, and",
            "top-level `cert`/`key`.",
        ],
    }
    if preamble:
        doc["script_generation"]["preamble"] = preamble
    if projects is not None:
        doc["paths"] = {"projects": str(projects)}
    return doc


def _ensure_root() -> bool:
    """Make the config directory if it is not there.  ``True`` if it made it.

    ``mode=`` on mkdir needs no chmod after it: the umask can only REMOVE bits
    from a requested mode, never add them, so 0700 is a ceiling rather than a
    suggestion.  (Verified, because the opposite is the intuitive reading and
    it is wrong.)

    Private, and shared by both doors, because ``seed_machine_config`` is
    reachable on its own -- a public function that works only if you happened
    to call a different one first is a trap, and it caught its own test.
    """
    root = config_dir()
    if root.is_dir():
        return False
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
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
        envs.mkdir(parents=True, exist_ok=True, mode=0o700)
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
_ENVIRONMENTS_README = 'molbuilder — environments\n=========================\n\nOne file per MACHINE YOU PREPARE FOR BUT ARE NOT ON: `<name>.json`, where the\nname is yours and becomes `--target <name>`.\n\nTHIS MACHINE\'S OWN RECORD IS NOT IN HERE.  It is `../environment.json`, one\nlevel up.  The two are different scopes, not copies — `record_scopes()` walks\ncalculation → target → machine and takes the first match.\n\nTHE PROBE RUNS ON THE TARGET, NOT HERE\n    This is the part that catches people.  A record describes cores, GPUs, the\n    scheduler and the queues you can actually reach; none of that is knowable\n    from your laptop, and a probe run here would faithfully measure YOUR box\n    and label it with the cluster\'s name.  So:\n\n    1. ON THE TARGET -- a login node is fine, the scheduler is read from\n       `sinfo` rather than from being on a compute node:\n\n           molbuilder jobset probe --write --name sol\n           # -> wrote ~/.config/molbuilder/environments/sol.json\n\n       `--write` shows what it measured and asks before overwriting an\n       existing record, difference by difference.  Silence keeps the record;\n       `--yes` takes every probed value.\n\n    2. COPY IT HERE:\n\n           scp cluster:~/.config/molbuilder/environments/sol.json \\\n               ~/.config/molbuilder/environments/\n\n    3. CONFIRM IT LANDED AND PARSES:\n\n           molbuilder jobset machines\n\n       This prints every record, its path and when it was measured.  It is the\n       only step that answers "did the copy work?" -- a record that is present\n       but corrupt is LISTED AND MARKED, never skipped, because a silently\n       dropped record looks exactly like one that was never copied.\n\n    Then `prep --target sol` sizes for Sol from anywhere.\n\nIF MOLBUILDER CANNOT BE INSTALLED ON THE TARGET\n    Declare what the probe would have measured, rather than guessing later:\n\n        molbuilder jobset probe --write --name sol \\\n            --scheduler slurm --set gpus_per_node=4 --set gpu_type=a100\n\n    A declared fact beats detection and the record stores `source: flag`, so a\n    reader can always see which numbers were measured and which were asserted.\n\nWHAT DOES NOT BELONG HERE\n    Nothing you write by hand.  These are records of measurement; if you find\n    yourself editing one, what you want is `--set` on the probe (which records\n    that you asserted it) or a preference in `molbuilder.json` (which is where\n    what-you-WANT lives, as opposed to what-the-machine-IS).\n\n    An empty directory is the normal state if you only ever run locally.\n\nReference: docs/execution/preparing-for-another-machine.md § 1a (these commands,\nverified end to end), docs/configuration.md § 5 (the record\'s schema and the\nfact-vs-preference rule).\n'


def _readme(directory: Path, text: str, note: str) -> List[Step]:
    """Write ``<directory>/README`` if absent.  One writer for both.

    Never overwrites -- a person may have added their own notes to it, and
    this module's whole discipline is that an existing file is reported and
    left alone.
    """
    path = directory / "README"
    if path.exists():
        return [Step(path, "kept")]
    path.write_text(text, encoding="utf-8")
    return [Step(path, "created", note)]


#: What ``secrets/README`` says.  It is a FILE rather than a docstring because
#: the person who needs it is looking at the directory, not at the source --
#: and because a conda-only install has no checkout to read docs from.
_SECRETS_README = 'molbuilder — secrets\n====================\n\nThis directory is for secret FILES that `molbuilder.json` points at by PATH.\nIt is created mode 0700 (owner only), and everything in it should be 0600.\nNothing here is ever printed: `config_provenance` logs only the sections\nflagged safe, and any section holding a secret — or a path to one — is\nexcluded by construction.\n\nWHY THE MODES MATTER\n    0700 on this directory, 0600 on each file.  `molbuilder.json` is itself\n    checked for 0600 and WARNS (never refuses — refusing would lock you out\n    of your own server).  A world-readable directory around 0600 files is the\n    same mistake one level up, so this one is tight from the start.\n\n        ls -l ~/.config/molbuilder ~/.config/molbuilder/secrets\n        chmod 700 ~/.config/molbuilder/secrets\n        chmod 600 ~/.config/molbuilder/secrets/*\n\nWHAT BELONGS HERE\n    Files referenced by a PATH in molbuilder.json.  The path is yours to\n    choose, so this is the suggested home rather than a required one:\n\n      the TLS private key (and cert, if not system-managed)\n          "tls": {"cert": "~/.config/molbuilder/secrets/fullchain.pem",\n                  "key":  "~/.config/molbuilder/secrets/privkey.pem"}\n\n      an OAuth provider\'s client secret, for providers whose wizard takes a\n      file path (CAS, GitHub, Microsoft, ORCID) — written by\n      `molbuilder auth-setup`, which records the path it used\n\nFOUR FILES THAT CANNOT LIVE HERE\n    Each has ONE fixed home directly in the config directory, resolved by a\n    single function so the reader and the writer cannot disagree about which\n    file they mean.  molbuilder.json cannot name them:\n\n      ../secret_key            the session key.  Created on FIRST SERVER RUN\n                               at 0600.  Do not make it by hand; deleting it\n                               logs everyone out and a new one appears.\n                               (`secret_key_file` in config is REFUSED.)\n      ../google_client_secret  Google\'s OAuth client secret, written by\n                               `molbuilder auth-setup`.\n      ../notify                the run-report CHANNELS file — see below.\n      ../notify_keys           the operator\'s run-report signing keys.\n                               (`notify_keys_file` in config is REFUSED.)\n\n    So this directory may be empty on a working installation.  That is\n    normal: a workstation with no HTTPS and no sign-in needs nothing here.\n\nRUN-REPORT CHANNELS — `../notify`, NOT a file in here\n    Shape: {"channels": {"<name>": {"url": ..., "kind"?: ..., "key"?: ...}}}\n    `kind` is "molbuilder" | "slack" | "discord"; omitted, it is read off the\n    URL\'s host.  Absent file means no notifier and the run proceeds exactly as\n    with the feature off; a malformed file says so in the monitor log and\n    carries on, and one bad channel does not cost the others.\n\n    THE EASY WAY is the web UI (Settings → run reports) or\n    `molbuilder notify-token`; both write this file at 0600 for you.  The\n    shapes below are for reading and for testing.\n\n    EXAMPLES.  The URLs are <ANGLE-BRACKET> placeholders rather than\n    realistic fakes, deliberately: a fake that LOOKS real IS a credential as\n    far as a secret scanner is concerned -- GitHub\'s push protection rejected\n    an earlier draft of THIS FILE for carrying a "Slack Incoming Webhook\n    URL".  Which is the clearest demonstration of the point below: for Slack\n    and Discord the URL *is* the secret.  Substitute the bracketed parts:\n\n      {\n        "channels": {\n          "local": {\n            "kind": "molbuilder",\n            "url":  "http://127.0.0.1:8765/hook",\n            "key":  "mock-local-key-not-a-real-secret"\n          },\n          "team-slack": {\n            "kind": "slack",\n            "url":  "https://hooks.slack.com/services/<TEAM-ID>/<CHANNEL-ID>/<TOKEN>"\n          },\n          "team-discord": {\n            "kind": "discord",\n            "url":  "https://discord.com/api/webhooks/<CHANNEL-ID>/<TOKEN>"\n          }\n        }\n      }\n\n    NOTE THE TWO SHAPES, because they differ in where the credential is:\n      * Slack and Discord put the credential IN THE URL.  A third party hands\n        you nothing but a URL, so the URL *is* the secret — which is why this\n        file is 0600 even when it looks like it holds no key.\n      * a molbuilder listener takes a plain url plus a `key` that SIGNS the\n        body and NEVER TRAVELS.  That is the shape to prefer when you control\n        the receiver: a leaked URL cannot be used to post as you.\n\n    A LOCAL LISTENER is the way to try this without a third party.  Point a\n    "molbuilder" channel at 127.0.0.1 and run anything that answers — the\n    signature is computed over the body with `key`, so a listener that does\n    not check it will still receive the report and show you the shape.\n\nIF YOU BACK THIS UP\n    Back up the whole config directory, not just this folder, and treat the\n    copy with the same care.  `secret_key` is the one file whose loss is\n    harmless (sessions end, a new key appears).  A leaked webhook URL, OAuth\n    client secret or TLS key is not.\n\nReference: docs/configuration.md § 2.1b (the mode rule), § 2.3 (how a secret\nis written -- atomically, so an interrupted write cannot destroy the one it\nreplaces), § 3.1 (the whole directory tree), § 2.1e (the session\nkey\'s one home), docs/execution/run-reports.md § 4.1b (channel kinds),\ndocs/ops/access-control.md (what sign-in exposes).\n'


def _seed_secrets_dir() -> List[Step]:
    """``secrets/`` and the README that says how to treat it.

    The directory is a SUGGESTED home, not a required one: what goes in it are
    files ``molbuilder.json`` names by PATH (the TLS key, a provider's
    client-secret file), and a path is the operator's to choose.  What makes it
    worth creating anyway is the README -- the mode rule, and the two files
    that CANNOT live here.

    ``secret_key`` and ``google_client_secret`` each resolve through one
    function in `config_dir` and sit directly in the config directory;
    `configuration.md` § 2.1e is explicit that the config cannot name the
    session key.  A README telling a person to put them here would be actively
    wrong, so it tells them the opposite.

    An EMPTY ``secrets/`` is the normal state on a workstation with no HTTPS
    and no sign-in.  The README says so, because an empty directory otherwise
    reads as something half-done.
    """
    steps: List[Step] = []
    root = config_dir()
    _ensure_root()
    d = root / "secrets"
    if d.is_dir():
        steps.append(Step(d, "kept"))
    else:
        d.mkdir(parents=True, exist_ok=True, mode=0o700)
        steps.append(Step(d, "created",
                          "mode 0700 -- 0600 on everything you put in it"))
    steps.extend(_readme(d, _SECRETS_README,
                         "the mode rule, mock channel examples, and the four "
                         "secrets that cannot live here"))
    return steps


def seed_machine_config(activation: str,
                        preamble: Optional[str] = None,
                        projects: Optional[Path] = None) -> Step:
    """Write ``molbuilder.json`` if it is absent; otherwise report it kept.

    An existing file is never merged into.  Merging would mean this module
    forming an opinion about a file a person has been editing -- and the one
    thing worth adding, ``activation``, is exactly the thing they may have
    deliberately left for a project-scope ``.molbuilder.json`` to supply
    (§ 5.1: project wins).  The ``note`` says whether the kept file resolves
    an activation, so the operator learns the one fact that matters without
    this module touching anything.
    """
    from ..persist import write_json
    from ..runtime_config import CONFIG_FILENAME

    path = config_dir() / CONFIG_FILENAME
    if path.exists():
        return Step(path, "kept", _activation_note())
    _ensure_root()
    # THE MODE IS CLAIMED BEFORE THE CONTENT LANDS.  `persist.write_bytes`
    # PRESERVES an existing target's mode and gives a NEW one 0644, so
    # writing first and chmod-ing after leaves the file briefly readable by
    # everyone with its contents already in it.  `auth_setup` engineered
    # against exactly that window -- *"create with mode bits at open() time
    # so there's no world-readable window"* -- and this is the same file.
    # Touching it 0600 first makes the writer's own preserve-the-mode branch
    # do the work.
    path.touch(mode=0o600)
    write_json(path, seed_document(activation, preamble, projects))
    note = f'script_generation.activation = "{activation}"'
    if preamble:
        note += f"; preamble = {preamble!r}"
    elif activation == "conda activate":
        # Worth saying only for THIS form.  ``conda activate`` is a shell
        # function that a non-interactive shell has never defined, so no
        # preamble here means a wrapper that will fail inside the job.
        note += ("; no preamble -- `conda activate` needs conda's hook "
                 "sourced in a non-interactive shell, and none was found")
    else:
        # ``source activate`` is a script on PATH.  It needs no hook, so the
        # absence is the correct state and must not be reported as a miss --
        # which is what it said until 2026-09-08, on every machine that HAD a
        # hook and simply did not need it.
        note += "; no preamble (source activate needs none)"
    return Step(path, "created", note)


def _activation_note() -> str:
    """What an already-present config resolves ``activation`` to.

    Read through the same door every wrapper uses, so this reports what the
    generator will actually see rather than what the file appears to say.
    """
    try:
        from ..runtime_config import get_script_generation
        current = get_script_generation(project_dir=None).get("activation")
    except Exception as exc:            # a broken config is its own error
        return f"left as it is (could not be read: {exc})"
    if current:
        return f'left as it is (activation = "{current}")'
    return ("left as it is -- but script_generation.activation is UNSET, so "
            "every wrapper will refuse to render (running-a-job.md 5.2)")


def seed_environment_record() -> Step:
    """This machine's own ``environment.json``, via the probe's own doors.

    Delegated to ``resolve_environment`` + ``write_environment`` -- what
    ``jobset probe --write`` calls -- so there is one prober and one writer,
    and re-probing later cannot disagree with what was seeded here.

    Ordered AFTER the config on purpose: the record carries
    ``script_generation`` copied out of ``molbuilder.json``, so a record
    written first would carry nothing and reproduce the very refusal this
    module exists to remove.
    """
    from ..diagnostics import local_facts
    from ..scheduler import (machine_scope_path, resolve_environment,
                             write_environment)
    path = machine_scope_path()
    if path.exists():
        return Step(path, "kept", "re-probe with `molbuilder jobset probe "
                                  "--write` when the machine changes")
    # THE SAME TWO STEPS ``jobset probe --write`` TAKES, in the same order:
    # resolve, then attach the three facts that travel (`local_facts`).  A
    # record written without the second step carries no activation, which is
    # precisely the refusal this module exists to remove -- so seeding it that
    # way would have shipped the bug in a new place.
    env, _note = local_facts(resolve_environment())
    write_environment(env, path)
    sg = getattr(env, "script_generation", None) or {}
    note = "probed this machine"
    if sg.get("activation"):
        note += f'; carries activation "{sg["activation"]}"'
    else:
        note += "; carries NO activation -- prep for this machine will refuse"
    return Step(path, "created", note)


def init_config(activation: str,
                preamble: Optional[str] = None,
                probe: bool = True,
                projects: Optional[Path] = None) -> List[Step]:
    """Seed the whole directory.  Idempotent; returns what it did.

    ``probe=False`` skips the record for the case where the machine being
    installed on is not the machine that will run anything -- a build host, a
    container image baked once and copied.  The config still gets seeded,
    because ``activation`` is a property of how the image enters conda and
    travels with it.

    ``projects`` writes ``paths.projects``.  ``None`` leaves the section empty
    and the default applies -- which is the right answer on a workstation and
    the wrong one on a cluster home with a quota, so the CLI ASKS rather than
    assuming either (user, 2026-09-12).  Declared, never detected: the same
    rule ``activation`` follows, for the same reason.
    """
    steps = list(ensure_dirs())
    steps.append(seed_machine_config(activation, preamble, projects))
    if probe:
        steps.append(seed_environment_record())
    return steps
