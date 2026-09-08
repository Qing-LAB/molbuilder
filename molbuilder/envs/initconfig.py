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
                  preamble: Optional[str] = None) -> "dict":
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
        "_comment_top":
            "molbuilder's server-wide configuration, seeded by "
            "`molbuilder envs init-config`.  Every section is optional and "
            "unknown top-level keys are REFUSED (a `_` prefix marks a "
            "comment).  See docs execution/running-a-job.md section 5 for "
            "the full list of sections.",
        "_comment_script_generation":
            "How a generated wrapper enters its conda env.  `activation` is "
            "\"conda activate\" or \"source activate\" and has NO default -- "
            "unset, EVERY wrapper refuses to render.  `preamble` is arbitrary "
            "shell run before activation (the `module load` lines on a "
            "cluster, or sourcing conda's hook on a workstation).  "
            "running-a-job.md section 5.2.",
        "_comment_signin":
            "Sign-in, TLS, the admin list and the rate limiter are sections "
            "of this same file.  `molbuilder auth-setup` writes the `auth` "
            "block for you (CAS or Google) and preserves everything else "
            "here -- run `molbuilder auth-setup --help`.  For the OTHER "
            "providers (GitHub, Microsoft, ORCID), TLS and the admin list, "
            "the fully worked template with every key explained is "
            "docs/ops/examples/molbuilder.json.example in the molbuilder "
            "repo; ops/access-control.md and ops/deployment.md section 5 are "
            "the reference.  Secrets are separate 0600 files in this "
            "directory -- this file carries their PATHS, never their bytes.",
        "_comment_scheduler":
            "Deliberately absent.  Partition, QOS and node topology are FACTS "
            "about a machine, not preferences: `molbuilder jobset probe "
            "--write` measures them into environment.json beside this file, "
            "and `--name <cluster>` writes environments/<cluster>.json for a "
            "machine you prep FOR but are not ON.  configuration.md M-1.",
        "script_generation": {"activation": activation},
    }
    if preamble:
        doc["script_generation"] = {"preamble": preamble,
                                    "activation": activation}
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
        envs.mkdir(parents=True, exist_ok=True)
        steps.append(Step(envs, "created",
                          "drop a colleague's `probe --name <cluster>` "
                          "record here"))
    return steps


def seed_machine_config(activation: str,
                        preamble: Optional[str] = None) -> Step:
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
    write_json(path, seed_document(activation, preamble))
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
                probe: bool = True) -> List[Step]:
    """Seed the whole directory.  Idempotent; returns what it did.

    ``probe=False`` skips the record for the case where the machine being
    installed on is not the machine that will run anything -- a build host, a
    container image baked once and copied.  The config still gets seeded,
    because ``activation`` is a property of how the image enters conda and
    travels with it.
    """
    steps = list(ensure_dirs())
    steps.append(seed_machine_config(activation, preamble))
    if probe:
        steps.append(seed_environment_record())
    return steps
