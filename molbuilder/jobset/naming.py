"""THE NAMES OF A CALCULATION'S STAGES, RUNS AND FOLDERS -- floor 3
(`execution/architecture.md` § 2.1, § 3.2).

A stage's number and folder (:func:`stage_home`, the one door), a plan's job
folders (:func:`job_dir_names`), which stage a job is (:func:`stage_refs`),
the names of its files (:func:`run_names`), and a folder that holds files
and never a run, opened and said to be one (:func:`open_container`): what
the plan asks to name what it writes, and every floor above it to find it.
Read off the description, the plan and the disk -- never stored.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

from .. import calcdirs
from ..identity import StageRef, parse_token
from ..paths import (bench_container as _paths_bench_container,
                     trial_label as _paths_trial_label,
                     trial_name as _paths_trial_name)
from ..runfiles import parse as _rf_parse
from .model import KIND_SWEEP, JobSet

if TYPE_CHECKING:
    from ..paths import Shape
    from ..runfiles import RunNames


@dataclass(frozen=True)
class StageHome:
    """Where a stage lives, and its number -- `execution/architecture.md`
    § 3's ``StageHome``, answered by :func:`stage_home` alone (§ 3.2)."""
    name: str
    seq: int
    #: ``<NN>_<name>`` -- the stage's directory in the hierarchy, its decks'
    #: token in either shape.
    token: str
    #: The stage's folder, or ``None`` when no calculation folder was given.
    dir: Optional[Path]
    #: THE LABEL THE RUNG'S FILES ARE NAMED ON -- the calculation's, or a
    #: transport lead's own stem (`identity.rung_label`).
    label: str = ""

    def names(self, shape: str) -> "RunNames":
        """The names of this stage's files in ``shape`` (`runfiles.RunNames`)
        -- THE ONE COMPOSER every reader of a stage's runs asks, so a lead's
        stem is spelt nowhere else."""
        from ..runfiles import RunNames
        return RunNames.of(self.label, self.token, shape)


def ladder_homes(base, task) -> Tuple[StageHome, ...]:
    """Every described stage's home, numbered by the ONE rule
    (`project-layout.md` § 4.2; W38 F4, W55 B8):

    * a stage that has files keeps the number they carry -- its directory in
      the hierarchy, its files' token in the flat shape (`paths.stages_in`);
    * a stage that has none takes its place in the description when no
      folder holds that number, else the next number after every one in use.

    So before anything is produced the numbers are the description's order,
    reordering is free, and once a stage has files its number never moves --
    a stage removed after its prep leaves its number taken and every later
    stage where it was; a stage added after production takes the next one.
    ``base`` ``None`` -- a description with no folder -- numbers by place."""
    from ..identity import stage_key, stage_token
    from ..identity import rung_label
    from ..paths import Shape, stages_in
    shape = Shape.named(task.shape)
    folder = Path(base) if base is not None else None
    on_disk = (stages_in(folder, shape, task.label)
               if folder is not None else [])
    held: Dict[str, int] = {}
    for seq, name in on_disk:
        # Two folders of one name (one left by a removal, one added since):
        # the newer number is the stage's -- numbers only grow.
        k = stage_key(name)
        held[k] = max(held.get(k, 0), seq)
    in_use = {seq for seq, _ in on_disk}
    homes = []
    for place, st in enumerate(task.stages, start=1):
        seq = held.get(stage_key(st.name))
        if seq is None:
            seq = place if place not in in_use else max(in_use) + 1
        in_use.add(seq)
        token = stage_token(seq, st.name)
        homes.append(StageHome(
            name=st.name, seq=seq, token=token,
            dir=(folder / shape.stage_dir(token)
                 if folder is not None else None),
            label=rung_label(task.label, st.name)))
    return tuple(homes)


def described_refs(base, task) -> List[StageRef]:
    """The description's stages, as refs (`identity.StageRef`) numbered by
    the one door (:func:`ladder_homes`), so ``#N`` names the folder
    ``NN_…`` everywhere -- the stages a verb that takes one offers."""
    return [StageRef(h.seq, h.name) for h in ladder_homes(base, task)]


def stage_home(base, task, stage: Optional[str]) -> StageHome:
    """This stage's number, token and folder -- THE ONE DOOR every reader
    asks (`execution/architecture.md` § 3.2; W55 B8): :func:`ladder_homes`'
    answer for it.

    An unknown stage is refused by name: an empty token would silently drop
    the stage from every artifact name (`job-contracts.md` § 6.3)."""
    from .errors import PrepError
    if not stage:
        # EVERY LADDER HAS A STAGE TO NAME (`engines/stages.md` § 6.5).
        raise PrepError("which stage? Every description's ladder has one to "
                        "name -- none was given.")
    from ..identity import stage_key
    for home in ladder_homes(base, task):
        if stage_key(home.name) == stage_key(stage):
            return home
    raise PrepError(f"stage {stage!r} is not in this description's "
                    f"ladder: {', '.join(s.name for s in task.stages)}.")


def trial_dir(shape, stage_token: str, job_name: str) -> str:
    """The path from the bundle to ONE trial's directory — **the rule**.

    ``<container>/bench-<point>``, where the container is the stage's bench
    folder in hierarchical and the flat one otherwise
    (:func:`bench_container`).

    `prep` cannot call :func:`job_dir_names` instead — it is *building* the
    JobSet in the loop that needs the directory, so there is nothing to ask
    yet.  That is what makes a shared RULE the fix rather than a shared
    lookup.
    """
    return f"{_paths_bench_container(shape, stage_token)}/{_paths_trial_name(job_name)}"


class NamingError(ValueError):
    """A job of a plan no file of its can be named for -- a deck named on
    no stage, or not on its label -- the naming authority's refusal
    (:func:`job_dir_names`, :func:`run_names`), said by every door."""


def job_dir_names(jobset: JobSet, shape: "Shape") -> Dict[str, str]:
    """``{job name: directory name}`` for a whole JobSet — the naming authority.

    One question, not two kinds (`generator.md` § 5): *does this job have a
    stage, a point, or both?*

    | the deck says | the set says | directory |
    |---|---|---|
    | a stage token, job named for the stage | — | ``<NN>_<name>`` — the rung itself |
    | a stage token, job named by coordinate | — | a trial, in the stage's bench CONTAINER (:func:`bench_container`): ``<NN>_<name>/bench/bench-<point>`` hierarchical, ``bench_<NN>_<name>/bench-<point>`` flat |
    | no token | — | **refused**: not a job prep wrote |

    **A job whose deck carries no stage token is refused**: every
    description has a stage, and every deck prep writes carries its token.
    The split is read off each deck's own name.

    **The seq is read back off the deck, not counted here.** ``job.script`` is
    ``<label>_<NN>_<name>.fdf`` (decision 27), so the token the directory is
    named for is the one the deck already carries — which is what makes
    ``<NN>_<name>/<label>_<NN>_<name>.fdf`` a self-check rather than a
    repetition (§ 4.1). Counting positions here instead would reintroduce
    exactly what `engines/stages.md` R5 forbids: a number that shifts when the
    ladder changes, silently handing one stage's directory to another.

    ``shape`` decides where a **stage** sits: hierarchical gives each one a
    directory, flat is depth 1 and they all sit in the bundle root
    (:class:`~molbuilder.paths.Shape`).  A described trial nests
    under its stage's directory, so the shape reaches it through the stage.
    It is required: the description's, `Shape.named(task.shape)` (A16).
    """
    sh = shape
    refs = stage_refs(jobset)
    out: Dict[str, str] = {}
    for j in jobset.jobs:
        if refs[j.name].token:
            # A rung of the ladder: the stage directory itself.
            out[j.name] = sh.stage_dir(refs[j.name].token)
            continue
        trial_token = _trial_stage_token(jobset, j)
        if trial_token:
            # A trial NESTS inside the stage's bench CONTAINER
            # (job-contracts.md § 6.3's Directories table, the cross-layer
            # authority: "benchmark | bench/ inside the stage").  The
            # container is what gives the stage's bench state ONE home --
            # its trials, its own job-set.json, its verdict -- so two
            # stages' benchmarks can never collide.
            out[j.name] = trial_dir(sh, trial_token, j.name)
            continue
        raise NamingError(
            f"job {j.name!r} ({j.script}): its deck names no stage -- not a "
            f"job prep wrote, so it has no folder to name.  A calculation "
            f"prepared before 2.17.0 names a lead's deck on its own label: "
            f"`molbuilder jobset migrate` in the calculation's folder renames "
            f"its files to the lead's stem.")
    return out


def _trial_stage_token(jobset: JobSet, job) -> Optional[str]:
    """The ``<NN>_<stage>`` a TRIAL's deck carries, or ``None``.

    A trial's script is ``<label>-<point>_<NN>_<stage>.ext`` — its own § 6.3
    label (the calculation's, qualified by the coordinate) plus the stage
    token.  Reading the name back with that full label is what keeps a
    stage name containing ``_`` unambiguous, exactly as for a rung
    (`runfiles.parse`, the one grammar).
    """
    got = _rf_parse(os.path.basename(job.script),
                    _paths_trial_label(jobset.name, job.name))
    return got.stage if got is not None else None


def stage_refs(jobset: JobSet) -> Dict[str, StageRef]:
    """``{job name: StageRef}`` for **every** job — *which stage is this?*

    This is the after-produce half of the resolver (§ 8f) and **the only place
    the two kinds are told apart**. ``seq`` is recovered from each deck's own
    token, which is where `project-layout.md` § 4.1 says it lives: *"read off
    the directory name and stored nowhere else"*. Nothing here counts
    positions, so a stage removed after its prep leaves a gap rather than
    renumbering.

    **Total on purpose.** Every job gets a ref; one with no assigned ordinal
    gets ``seq=None`` rather than being left out of the mapping, so each
    caller reads one and never tests membership.

    ``seq=None`` is still never a guess: a sweep point has no order at all, and
    a ladder job whose deck carries no token has an ordinal nobody assigned
    (§ 4.2's number is assigned once and never invented).

    The ref carries the **job's** name, not the token's. They are the same
    string for anything a producer built — ``siesta/stages.py`` names each job
    for its stage — and where they could differ it is the job name that
    dependency edges, ``--stage-resources`` keys and the CLI all point at, so
    resolving to the other one would hand back a name this JobSet does not have.
    """
    # NO kind branch: the parse is anchored on the rung's label (a lead's
    # own stem, `identity.rung_label`), so a TRIAL's script (whose
    # label is the coordinate-qualified one) never matches and gets
    # seq=None.
    from ..identity import rung_label
    out: Dict[str, StageRef] = {}
    for j in jobset.jobs:
        got = _rf_parse(os.path.basename(j.script),
                        rung_label(jobset.name, j.name))
        parsed = parse_token(got.stage) if got and got.stage else None
        out[j.name] = StageRef(parsed[0] if parsed else None, j.name)
    return out


def run_names(jobset: JobSet, job, shape: "Shape") -> "RunNames":
    """THE NAMES OF ``job``'S FILES -- its stage's, in the calculation's
    shape (`runfiles.RunNames`, `job-contracts.md` § 2.2a): a benchmark
    trial's on its own label (`paths.trial_label`) and in its own folder, a
    ladder stage's on the calculation's -- a transport lead's on the lead's
    own stem (`identity.rung_label`; a transport calculation has no
    sweep) -- sharing the calculation's folder in the flat shape.  The
    stage is the one its deck's name carries, read back through the
    grammar.  ONE answer, asked by prep, launch and status alike: a run's
    record lies in its folder, and its name is the names'."""
    from ..paths import trial_label
    from ..runfiles import RunNames, parse
    from ..identity import rung_label
    trial = jobset.kind == KIND_SWEEP
    label = (trial_label(jobset.name, job.name) if trial
             else rung_label(jobset.name, job.name))
    rec = parse(Path(job.script).name, label)
    if rec is None or rec.stage is None:
        raise NamingError(
            f"job {job.name!r}: its deck {job.script!r} is not named on "
            f"{label!r} with a stage (`runfiles.compose`)")
    return RunNames.of(label, rec.stage, shape.name, trial=trial)


def open_container(base, folder, plan) -> Path:
    """Make ``folder`` -- one that holds files and never a run: a
    submission's ``launch/``, the calculation's ``pseudos/``, a point's
    folder below its stage, which prep renders the point's deck into -- and
    say it is a container, with every directory above it (§ 1.4a).  Returns
    ``folder``."""
    base, folder = Path(base), Path(folder)
    plan.folder(folder)
    stamp_containers(base, [folder, *folder.parents], plan)
    return folder


def stamp_containers(base: Path, dirs, plan) -> None:
    """Each of ``dirs`` below the calculation root, stamped a container --
    outermost first."""
    for c in reversed(list(dirs)):
        if c == base or base not in c.parents:
            continue
        plan.text(*calcdirs.record(c, role=calcdirs.CONTAINER, root=base))


__all__ = ["NamingError", "StageHome", "described_refs", "job_dir_names",
           "ladder_homes", "open_container", "run_names", "stage_home",
           "stage_refs", "stamp_containers", "trial_dir"]
