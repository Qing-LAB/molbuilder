"""Materialize engine — turn a :class:`JobSet` into on-disk per-job
directories (docs/execution/job-system.md; naming: job-contracts § 6.3).

Filesystem ONLY: it knows nothing about schedulers or engines.  For each
job it creates the directory :func:`job_dir_names` assigns (a stage's
``<NN>_<name>/``, a trial's ``<NN>_<name>/bench/bench-<point>/``) and
copies in, as real files, the static ``shared`` package plus the job's own
``script`` (`project-layout.md` § 1.0: a run directory holds everything it
runs from).
"""

from __future__ import annotations

from .model import KIND_SWEEP

from typing import TYPE_CHECKING
if TYPE_CHECKING:                      # annotations only
    from ..paths import Shape
    from ..runfiles import RunNames

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .. import calcdirs, runrecord
from ..pseudos import PSEUDO_DIRNAME
from ..identity import StageRef, parse_token, resolve_stage_ref
from ..runfiles import FIRST_ATTEMPT, parse as _rf_parse
from .model import JobSet, warm_carry
from ..paths import (attempt_dir, attempt_index, attempts_in,
                     trial_label as _paths_trial_label,
                     trial_name as _paths_trial_name,
                     bench_containers_in as _bench_containers_in,
                     bench_container as _paths_bench_container)

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
    #: transport lead's own stem (`transport.stages.rung_label`).
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
    from ..transport.stages import rung_label
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


def trial_work_dir(container, shape, names: "RunNames") -> Path:
    """Where a trial's files GO, given the directory it lives in.

    :func:`trial_dir` (via :func:`job_dir_names`) answers *where does this
    trial live*; this answers *where does prep put its deck and package*,
    and the two differ the moment a trial keeps attempts
    (`project-layout.md` § 1.5a):

    * **hierarchical** — the attempt, ``bench-<point>/run-<n>``;
    * **flat** — the container itself, because flat has no directory layer
      to open.

    **It takes the container rather than recomputing it**, so it cannot
    diverge from :func:`job_dir_names`.

    `resolve_attempt` is the rule, not restated: reuse the last attempt
    until it has been launched, then open the next -- a benchmark's sweep is
    prepared once (`job-system.md` § 5.0), so its prep finds ``run-0``.

    **The read-side twin is :func:`run_dir`** — *where does this trial
    actually run*, which needs no shape because by then the directory is
    there to be found.  Both resolve to the newest attempt, and that is the
    invariant: if they ever disagree, prep writes where nothing reads.
    """
    d = Path(container)
    if shape is None or not shape.keeps_attempts_as_directories:
        return d
    # NOTHING IS MADE HERE: a container not there yet holds no attempt
    # (`paths.attempts_in`), and the plan makes the folder when it is
    # written.
    attempt, _fresh = resolve_attempt(d, names)
    return attempt


def sweep_set_dirs(bundle) -> "List[Path]":
    """Every folder a SWEEP's plan can lie in, in this bundle -- the bench
    containers the layout declares (`paths.bench_containers_in`), in order.

    The search counterpart of :func:`bench_container`, and it lives beside it
    for that reason: the namer says where a sweep's state GOES and this says
    where to look for it, so a layout change moves one file instead of two
    that must be kept in step by hand.  Whether a folder holds a plan, and
    whether it is a sweep's, is the plan's door's answer (`JobSet.of`,
    `architecture.md` A16); nothing is read here.
    """
    base = Path(bundle)
    return [base / rel for rel, _tok in _bench_containers_in(base)]


def bench_stage_of(base, where) -> "Optional[str]":
    """The stage a sweep measures, by its name -- read off where the sweep
    lives: the bench container that holds ``where`` (the container itself,
    or a trial in it), through the layout's search half
    (`paths.bench_containers_in`), so both layouts answer.  A refusal or a
    status names the stage's own verbs with it, never a ``<stage>``.
    ``None`` for a sweep in no stage's container -- a hand-built one."""
    from ..identity import command_stage
    try:
        rel = Path(where).resolve().relative_to(Path(base).resolve())
    except (ValueError, OSError):
        return None
    for name, token in _bench_containers_in(base):
        if rel == Path(name) or Path(name) in rel.parents:
            return command_stage(token)
    return None


def bench_owner(folder) -> "Optional[Tuple[Path, str]]":
    """``(calculation folder, stage name)`` when ``folder`` IS a stage's
    bench container -- the declared one of the calculation one level up
    (flat's ``bench_<NN>_<stage>``) or two (the hierarchy's
    ``<NN>_<stage>/bench``), through the layout's search half -- else
    ``None``.  A sweep's own job-set names its trials from the calculation,
    so a reader standing in its container reads them from there."""
    from ..identity import command_stage
    # THE CALCULATION ROOT through its one door (`calcdirs.root_of`): the
    # container is stamped, so the root answers from any depth.
    f = Path(folder).resolve()
    calc = calcdirs.root_of(f)
    if calc is not None:
        rel = f.relative_to(calc.resolve())
        for name, token in _bench_containers_in(calc):
            if Path(name) == rel:
                return calc, command_stage(token)
    return None


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
        raise ValueError(
            f"job {j.name!r} ({j.script}): its deck names no stage -- not a "
            f"job prep wrote, so it has no folder to name.  A calculation "
            f"prepared before 2.17.0 names a lead's deck on its own label: "
            f"`molbuilder jobset migrate --bundle <the calculation>` renames "
            f"its files to the lead's stem (job-system.md R-P2).")
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
    # own stem, `transport.stages.rung_label`), so a TRIAL's script (whose
    # label is the coordinate-qualified one) never matches and gets
    # seq=None.
    from ..transport.stages import rung_label
    out: Dict[str, StageRef] = {}
    for j in jobset.jobs:
        got = _rf_parse(os.path.basename(j.script),
                        rung_label(jobset.name, j.name))
        parsed = parse_token(got.stage) if got and got.stage else None
        out[j.name] = StageRef(parsed[0] if parsed else None, j.name)
    return out


def materialize(jobset: JobSet, base_dir, plan=None, *,
                shape, only=None) -> List[Path]:
    """Create each job's directory under ``base_dir`` with its copies --
    the jobs named in ``only`` (a set of job names), or every job.

    Returns the list of created job directories (in JobSet order).  Idempotent:
    a copy already there is kept, never written twice.  Raises
    ``ValueError`` if the JobSet is structurally invalid (so a bad carry /
    duplicate name can't produce a broken tree).

    ``plan`` (`jobset.planned.Plan`) receives the folders and copies instead
    of the disk, and is read for the files it already holds -- `prep`
    decides everything before it writes (`job-system.md` § 5.0); without one
    they are written now.  ``shape`` is the layout as the caller read it
    (`Shape.named(task.shape)`, the one door; A16).
    """
    errors = jobset.validate()
    if errors:
        raise ValueError(
            "cannot materialize an invalid JobSet:\n  - "
            + "\n  - ".join(errors))
    from .planned import Plan
    own = plan is None
    plan = Plan() if own else plan
    base = Path(base_dir)
    created: List[Path] = []
    sh = shape
    dirs = job_dir_names(jobset, sh)
    for job in jobset.jobs:
        if only is not None and job.name not in only:
            continue
        # A TRIAL KEEPS ATTEMPTS EXACTLY AS A STAGE DOES, and the shape
        # decides (`project-layout.md` § 1.5a).  `trial_work_dir` is the
        # one answer to *where do this trial's files go*, and `prep` asks
        # the same one.
        d = base / dirs[job.name]
        if jobset.kind == KIND_SWEEP:
            # A TRIAL'S FOLDER IS A RUN's, opened by the one opener.
            d = open_run(base, trial_work_dir(d, sh,
                                              run_names(jobset, job, sh)),
                         plan)
        else:
            plan.folder(d)
        created.append(d)
        if d.resolve() == base.resolve():
            # FLAT: depth 1 (`project-layout.md` § 1) -- the job runs in the
            # bundle root, where every file it needs ALREADY SITS, and flat's
            # warm files are ONE SHARED SET there (§ 1) -- except the
            # pseudopotentials: they live in `pseudos/`, and SIESTA
            # opens `<element>.psml` in the directory it runs from and has no
            # search path.
            for _ps in plan.glob(base / PSEUDO_DIRNAME, "*.psml"):
                _dst = d / _ps.name
                if not plan.is_file(_dst):
                    plan.copy(_ps, _dst)
            continue
        # The static package arrives as REAL COPIES (user, 2026-08-24;
        # `project-layout.md` § 1.0: the run directory "holds everything",
        # and a symlink holds nothing).  The deck is not in this list: it is
        # born in the directory, so there is no root copy to reach for.
        for fname in list(jobset.shared):
            src = base / fname
            dst = d / os.path.basename(fname)
            if not plan.is_file(src):
                continue          # prep's own missing-input gates report it
            if not plan.is_file(dst):
                plan.copy(src, dst)
        # NOTHING ELSE IS LINKED IN: what a stage continues from is a real
        # file COPIED by `prepare_attempt` from the attempt you name
        # (project-layout.md 1.6).
    if own:
        plan.carry_out()
    return created


# --------------------------------------------------------------------- #
#  Attempts — one directory per try at a stage (project-layout.md § 1.6)  #
# --------------------------------------------------------------------- #


def run_names(jobset: JobSet, job, shape: "Shape") -> "RunNames":
    """THE NAMES OF ``job``'S FILES -- its stage's, in the calculation's
    shape (`runfiles.RunNames`, `job-contracts.md` § 2.2a): a benchmark
    trial's on its own label (`paths.trial_label`) and in its own folder, a
    ladder stage's on the calculation's -- a transport lead's on the lead's
    own stem (`transport.stages.rung_label`; a transport calculation has no
    sweep) -- sharing the calculation's folder in the flat shape.  The
    stage is the one its deck's name carries, read back through the
    grammar.  ONE answer, asked by prep, launch and status alike: a run's
    record lies in its folder, and its name is the names'."""
    from ..paths import trial_label
    from ..runfiles import RunNames, parse
    from ..transport.stages import rung_label
    trial = jobset.kind == KIND_SWEEP
    label = (trial_label(jobset.name, job.name) if trial
             else rung_label(jobset.name, job.name))
    rec = parse(Path(job.script).name, label)
    if rec is None or rec.stage is None:
        raise ValueError(
            f"job {job.name!r}: its deck {job.script!r} is not named on "
            f"{label!r} with a stage (`runfiles.compose`)")
    return RunNames.of(label, rec.stage, shape.name, trial=trial)


def latest_attempt(stage_dir: Path) -> Optional[Path]:
    """The newest attempt under ``stage_dir``, or ``None`` if there are none.

    **Where a stage's state actually is.** `project-layout.md` § 1.5 is flat
    about it — *"Where a run happens: inside the attempt directory"* — and
    *"everything the run writes"* is *"created in place"*, because the wrapper
    is invoked there. So anything asking *what happened to this stage?* asks
    here first, and only falls back to the container for a flat run, which
    § 1.5 says is untouched and *"is a run"* in its own right.

    This is a layout question, so it is answered in the layout layer rather
    than by each observer working out where to look.
    """
    ns = attempts_in(stage_dir)
    return attempt_dir(stage_dir, ns[-1]) if ns else None


def run_dir(container: Path) -> Path:
    """**The directory a stage or trial actually uses** — its newest attempt
    when the shape keeps them, the container itself when it does not.

    THE OTHER HALF OF :func:`latest_attempt`, which answers *is there an
    attempt* and is right to return ``None`` for flat.  Almost every caller
    wants *where do I look*, and this answers it in the layout layer.

    Keep using `latest_attempt` where ``None`` is the ANSWER (has this been
    prepared at all?); use this where a path is wanted.

    **The write-side twin is :func:`trial_work_dir`** — *where does prep PUT
    this trial's files*, which takes the shape because it runs before the
    directory exists and may have to open one.  The two agree by
    construction and must keep agreeing: prep uses (or opens) the newest
    attempt, and this finds the newest.  If either ever picks a different
    one, prep writes where nothing reads.
    """
    return latest_attempt(container) or Path(container)


def resolve_attempt(stage_dir: Path, names: "RunNames") -> Tuple[Path, bool]:
    """The attempt directory to prepare into, and whether it is a fresh one.

    The last attempt is REUSED when it has not been launched -- the one prep
    opened, which `launch` then runs -- and a new one is opened only when the
    last has: what a launch of a stage again opens (`job-system.md` § 5.4).
    That makes the numbering mean something: every ``run-<n>`` on disk but
    the newest was actually started.  ``names`` are the stage's
    (:func:`run_names`), which name its launch record.
    """
    existing = attempts_in(stage_dir)
    if existing:
        last = attempt_dir(stage_dir, existing[-1])
        try:
            launched = runrecord.launch_record(last, names) is not None
        except runrecord.LaunchRecordError as e:
            from .errors import PrepError
            raise PrepError(str(e)) from e
        if not launched:
            return last, False
        return attempt_dir(stage_dir, existing[-1] + 1), True
    return attempt_dir(stage_dir, FIRST_ATTEMPT), True


def ever_launched(jobset: JobSet, base_dir, job, shape) -> bool:
    """Whether ``job``'s stage has been launched -- any run of it
    (`project-layout.md` § 1.6.1): where its runs are folders, every
    ``run-<n>`` but the newest was, and the newest when its launch record
    says so (:func:`resolve_attempt`'s rule); where its runs share a
    folder, its newest launched run's record.  ``shape`` is the
    description's (`Shape.named(task.shape)`, the one door; A16).  A record
    that does not read raises, as the one door does
    (`runrecord.launch_record`)."""
    base = Path(base_dir)
    sh = shape
    names = run_names(jobset, job, sh)
    container = base / job_dir_names(jobset, sh)[job.name]
    if not sh.keeps_attempts_as_directories:
        return runrecord.launch_record(container, names) is not None
    ns = attempts_in(container)
    return len(ns) > 1 or (bool(ns) and runrecord.launch_record(
        attempt_dir(container, ns[-1]), names) is not None)


@dataclass(frozen=True)
class Attempt:
    """One try at a stage: the directory, and what was put in it -- the
    stage's own files (``brought``) and what it continues from, one
    hand-over (``hand``, ``None`` when it takes nothing).  ``fresh`` is
    False when an unlaunched attempt was **reused** rather than opened
    (:func:`resolve_attempt`)."""
    stage:          str
    dir:            Path
    fresh:          bool
    brought:        List[str]
    hand:           Optional["HandOver"] = None

    @property
    def copied(self) -> List[str]:
        """The files it took from the run it continues from."""
        return self.hand.names_taken if self.hand is not None else []


#: Why ``--from`` and ``--cold`` mean nothing on the flat layout -- said by
#: `continuation` before prep writes anything, and by `prepare_attempt`
#: to a caller that asks it anyway.
FLAT_HAS_NO_ATTEMPTS = (
    "this calculation's shape is 'flat', which has no attempt directories "
    "to open: each run carries the number launch gives it "
    "(<label>_<NN>_<name>-run<N>.out) and every stage reads the files the "
    "stage before it left in the one folder (project-layout.md § 1) -- so "
    "there is no run to name with --from, and none to skip with --cold.")


def open_run(base, run_dir, plan) -> Path:
    """THE ONE OPENER OF A RUN FOLDER (`project-layout.md` § 1.6.2): a
    stage's attempt (:func:`prepare_attempt`, a bias point's included) and a
    benchmark trial's folder, in either shape.  It makes the folder and every
    directory above it down from the calculation root, and says what each is
    (§ 1.4a, invariant 6b): each above a container, the run a run.  The root
    says itself, through its description, and gets no record.

    ``plan`` receives the folders and the stamps, as :func:`materialize`'s
    copies.  Returns ``run_dir``."""
    base, run_dir = Path(base), Path(run_dir)
    plan.folder(run_dir)
    if run_dir.resolve() == base.resolve():
        return run_dir
    _stamp_containers(base, run_dir.parents, plan)
    plan.text(*calcdirs.record(run_dir, role=calcdirs.RUN, root=base))
    return run_dir


def open_container(base, folder, plan) -> Path:
    """Make ``folder`` -- one that holds files and never a run: a
    submission's ``launch/``, the calculation's ``pseudos/`` -- and say it
    is a container, with every directory above it (§ 1.4a).  Returns
    ``folder``."""
    base, folder = Path(base), Path(folder)
    plan.folder(folder)
    _stamp_containers(base, [folder, *folder.parents], plan)
    return folder


def open_scratch(base, folder, plan) -> Path:
    """Make a run's scratch folder -- a device run's ``tsgf/``, the engine's
    scratch of the run above it (`engines/transport.md` § 2a.11) -- and say
    what it is (`calcdirs.SCRATCH`, invariant 6b): the run door and the
    Results sidebar read the role, and the checkpoint keeps the marker, so a
    restore brings the folder back empty.  Returns ``folder``."""
    base, folder = Path(base), Path(folder)
    plan.folder(folder)
    plan.text(*calcdirs.record(folder, role=calcdirs.SCRATCH, root=base))
    return folder


def _stamp_containers(base: Path, dirs, plan) -> None:
    """Each of ``dirs`` below the calculation root, stamped a container --
    outermost first."""
    for c in reversed(list(dirs)):
        if c == base or base not in c.parents:
            continue
        plan.text(*calcdirs.record(c, role=calcdirs.CONTAINER, root=base))


def bring_files(jobset: JobSet, job, base, src_dir, run, names, plan
                ) -> List[str]:
    """Copy into ``run`` what it runs from -- the deck, its wrappers, the
    shared package and the bundles that travel beside the deck -- from
    ``src_dir`` (where prep rendered them: the stage's folder, or a swept
    point's), else the calculation's root.  COPIED, never linked
    (`project-layout.md` § 1.0: the run directory "holds everything"; a
    synced-back bundle's links would dangle).  Returns what was brought.

    ONE COPY STEP for every run folder: a stage's attempt
    (:func:`prepare_attempt`) and a swept run's point
    (:func:`open_sweep_run`)."""
    base, src_dir, run = Path(base), Path(src_dir), Path(run)
    brought: List[str] = []

    def _bring(fname: str) -> None:
        bn = os.path.basename(fname)
        dst = run / bn
        for src in (src_dir / bn, base / fname, base / bn):
            if plan.is_file(src) and src.resolve() != dst.resolve():
                # REFRESHED every time: a REUSED unlaunched run must see the
                # prep's deck, not an earlier one's.
                plan.copy(src, dst)
                brought.append(bn)
                return

    for fname in [job.script] + list(jobset.shared):
        _bring(fname)
    # THE BUNDLES THAT TRAVEL BESIDE THE DECK -- the monitor's, the job's
    # finish, a PySCF script's code -- from the one list the wrapper's
    # writer reads too (`runwrap.bundles_for`): one file each, so none can
    # be half-shipped.  And makov_payne_correction.py: the post-run script a
    # CHARGED deck's own header instructs the user to run "after SIESTA
    # finishes" -- HERE, beside the .out.
    from ..runwrap import bundles_for
    from ..runfiles import MAKOV_PAYNE_SCRIPT
    for extra in (*(name for name, _build in bundles_for(job.script,
                                                        job.finish)),
                  MAKOV_PAYNE_SCRIPT):
        _bring(extra)
    for wrapper in (names.name(".run.sh"), names.name(".sbatch")):
        _bring(wrapper)
    return brought


@dataclass(frozen=True)
class SweepRun:
    """A swept stage's run (`engines/transport.md` § 2a.11): its folder, its
    points in walk order -- ``[(run-<n>/<point>, Point)]``, each frame's
    voltages in turn (`transport.stages.rung_points`) -- and whether it is a
    fresh one (``False``: the unlaunched run prep opened, reused)."""
    stage:  str
    dir:    Path
    points: List[Tuple[Path, object]]
    fresh:  bool


def open_sweep_run(jobset: JobSet, base_dir, stage_name: str, task, *,
                   plan, shape, frames: Optional[int] = None
                   ) -> SweepRun:
    """Open a swept stage's run -- ``run-<n>/`` -- and in it each point's
    folder, holding copies of that point's prepared files (its deck, run
    script and the shared package; :func:`bring_files`), and the run's
    scratch folder where the rung keeps one.  ``points`` is the listing
    order (frames outer); the walk runs them round by round
    (`engines/transport.md` § 2a.11).  The run is the
    stage's newest when it has not been launched -- reused, and emptied
    whole first (:func:`empty_run`) -- else the next (a launch again,
    `job-system.md` § 5.4, rule 3).  Each point
    folder is a run folder (`open_run`); the run above them is their
    container.  ``frames`` is the junction's count when the caller holds it
    (prep, composing it in the same plan); else the record's.  What the
    points take from upstream is the caller's (launch's gather), and what
    each starts from the walk's."""
    from ..transport.stages import point_folders
    base = Path(base_dir)
    sh = shape
    job = jobset.job(stage_name)
    stage_dir = base / job_dir_names(jobset, sh)[job.name]
    rn = run_names(jobset, job, sh)
    run, fresh = resolve_attempt(stage_dir, rn)
    sources = dict((pt, d) for d, pt in point_folders(base, task, stage_name,
                                                      frames=frames))
    points = [(run / pt.rel, pt) for pt in sources]
    if not fresh:
        empty_run(run, plan)
    for pdir, pt in points:
        open_run(base, pdir, plan)
        bring_files(jobset, job, base, sources[pt], pdir, rn, plan)
    # THE RUN'S SCRATCH FOLDER, opened with the run (`engines/transport.md`
    # § 2a.11): the rung declares it (`Job.scratch_dir`).
    if job.scratch_dir:
        open_scratch(base, run / job.scratch_dir, plan)
    return SweepRun(stage=stage_name, dir=run, points=points, fresh=fresh)


def empty_run(run, plan) -> List[str]:
    """A run never launched, emptied whole -- every file in it and every
    folder below it -- for the prep or the launch that fills it again
    (`job-system.md` § 5.0, 2a: *its first run emptied of anything put
    there before*; a run a refused send opened, § 6.0 step 4).  A link is
    removed, never followed.  Returns the paths taken out, from ``run``."""
    run = Path(run)
    out: List[str] = []
    for here, dirs, files in os.walk(run, topdown=False):
        here = Path(here)
        for name in sorted(files):
            plan.remove(here / name)
            out.append(str((here / name).relative_to(run)))
        for name in sorted(dirs):
            d = here / name
            if d.is_symlink():
                plan.remove(d)
            else:
                plan.remove_folder(d)
            out.append(str(d.relative_to(run)) + "/")
    return out


def empty_old_points(stage_dir, points, plan, *, keep=()) -> List[str]:
    """A swept rung prepared again: the folders prep rendered below its
    stage for points the description no longer holds, taken out whole --
    its files rendered again (`job-system.md` § 5.0, 2a).  ``points`` are
    the points prepared now (``Point.rel``); a run folder (``run-<n>``) and
    the folders ``keep`` names -- the stage's benchmark container, the walk
    a launch writes -- are never touched.  Returns the folders taken out,
    from ``stage_dir``."""
    stage_dir = Path(stage_dir)
    wanted = {str(Path(r)) for r in points}
    above = {str(parent) for r in wanted for parent in Path(r).parents
             if str(parent) != "."}
    out: List[str] = []

    def gone(folder: Path) -> None:
        for here, dirs, files in os.walk(folder, topdown=False):
            for name in sorted(files):
                plan.remove(Path(here) / name)
            for name in sorted(dirs):
                d = Path(here) / name
                if d.is_symlink():
                    plan.remove(d)
                else:
                    plan.remove_folder(d)
        plan.remove_folder(folder)
        out.append(str(folder.relative_to(stage_dir)) + "/")

    def walk(folder: Path) -> None:
        for d in sorted(p for p in folder.iterdir()
                        if p.is_dir() and not p.is_symlink()):
            rel = str(d.relative_to(stage_dir))
            if folder == stage_dir and (attempt_index(d.name) is not None
                                        or d.name in keep):
                continue
            if rel in wanted:
                continue
            if rel in above:
                walk(d)
            else:
                gone(d)

    if stage_dir.is_dir():
        walk(stage_dir)
    return out


#: A hand-over's two records (`job-system.md` § 5.4): the one run a run
#: continues from (`.continued-from`), and a transport rung's inputs, each
#: from its run (`.gathered-from`).
CONTINUED, GATHERED = "continued", "gathered"


@dataclass(frozen=True)
class HandOver:
    """WHAT ONE RUN TAKES FROM OTHER RUNS, AND THE RECORD THAT SAYS SO --
    every hand-over's one shape (`job-system.md` § 5.4), whoever decided it:
    the continuation door, the gather door, a launch again keeping what its
    last run was given, the points' walk.

    ``target`` is the run that takes -- an attempt, a point of a run of
    points, or a flat calculation's folder -- named by its stage's
    ``names`` as run ``run``; ``files`` each file it takes, with the run it
    comes from; ``record`` which record says so: :data:`CONTINUED` -- the
    one run it continues from, ``source`` (a taken-over point's is the run
    that COMPUTED it, one hop, whatever folder its files are copied from)
    -- or :data:`GATHERED`, each file with its run.  ``walk``: copied by the
    walk just before the run starts (a point's start, which the point
    before it may finish in the same walk), not with the plan."""
    target: Path
    names: "RunNames"
    run: int
    record: str
    files: Tuple[Tuple["runrecord.RunRef", str], ...] = ()
    source: Optional["runrecord.RunRef"] = None
    walk: bool = False

    def __post_init__(self):
        if self.record not in (CONTINUED, GATHERED):
            raise ValueError(f"a hand-over's record is {CONTINUED!r} or "
                             f"{GATHERED!r}, not {self.record!r}")
        if (self.record == CONTINUED) != (self.source is not None):
            raise ValueError("a continued hand-over names the run it "
                             "continues from; a gathered one names a run "
                             "per file")

    @property
    def names_taken(self) -> List[str]:
        """The files it takes, by name, in order."""
        return [name for _src, name in self.files]

    def written_as_ended(self, base) -> Dict[str, str]:
        """THE FILES IT TAKES THAT WERE WRITTEN AS THEIR RUN ENDED
        (:func:`written_as_ended`, `running-a-job.md` § 3.6), each from the
        folder it is taken from."""
        out: Dict[str, str] = {}
        by_folder: Dict[Path, List[str]] = {}
        for src, name in self.files:
            by_folder.setdefault(Path(src.path(base)), []).append(name)
        for folder, names in by_folder.items():
            out.update(written_as_ended(folder, self.names.label, names))
        return out

    def walk_copies(self, base, cwd) -> List[Tuple[str, str]]:
        """What the walk copies just before the run starts: ``(file, folder)``
        pairs, each from ``cwd``, the folder the walk runs in."""
        dst = os.path.relpath(self.target, cwd)
        return [(os.path.relpath(src.path(base) / name, cwd), dst)
                for src, name in self.files]


def written_as_ended(folder, label: str, files, *,
                     run: Optional[str] = None) -> Dict[str, str]:
    """THE FILES WRITTEN AS THEIR RUN ENDED (`running-a-job.md` § 3.6):
    ``{name: why}`` for each of ``files`` in ``folder`` whose write time is
    at or after the second the run's monitor closed -- the monitor log's
    closing record (`parse/instruments/monitor.py`, ``ended_at``), read
    through the registry -- a file a scheduler's kill may have caught
    mid-write.  The log is the newest of ``label``'s in the folder, or
    ``run``'s own where a stage's runs share the folder (its name,
    ``H2_01_coarse-run1``).  Empty where no monitor closed, or every file
    was written before.  A fact shown, never a veto."""
    from datetime import datetime
    from .. import runfiles as _rf
    from ..parse import detect
    folder = Path(folder)
    if run is not None:
        rec = _rf.parse_run_name(run, label)
        logs = (_rf.find(folder, label, role=".monitor.log", stage=rec.stage,
                         run=rec.run) if rec is not None else [])
    else:
        logs = sorted(_rf.find(folder, label, role=".monitor.log"),
                      key=lambda pr: (pr[1].run is None, pr[1].run or 0))
    if not logs:
        return {}
    try:
        at = detect(str(logs[-1][0])).parse(logs[-1][0]).metrics.get("ended_at")
        end_s = datetime.fromisoformat(at).timestamp() if at else None
    except Exception:                       # noqa: BLE001 -- an unreadable
        return {}                           # log says nothing
    if end_s is None:
        return {}
    out: Dict[str, str] = {}
    for name in files:
        try:
            wrote = (folder / name).stat().st_mtime
        except OSError:
            continue
        if wrote >= int(end_s):
            w = datetime.fromtimestamp(wrote).astimezone()
            out[name] = (f"written as the job ended ({w:%H:%M:%S}; the job "
                         f"ended {at[11:19]}), it may be incomplete; a start "
                         f"from it may need more iterations or fail; "
                         f"`--cold` starts without it")
    return out


def hand_over(base, h: HandOver, plan=None) -> List[str]:
    """CARRY ONE HAND-OVER OUT -- the one function every hand-over goes
    through (`job-system.md` § 5.4): its files copied into its run -- as
    real files, never linked, since the engine writes to those very names
    (`project-layout.md` § 1.6.5) -- unless the walk copies them
    (``h.walk``), and its record written beside the run
    (`runrecord.write_continued_from`, `write_gathered_from`).  Into
    ``plan`` (`jobset.planned.Plan`); with none, now -- a flat run's
    record, written with its launch record.  Returns the files copied."""
    from .planned import Plan
    own = plan is None
    plan = Plan() if own else plan
    copied: List[str] = []
    if not h.walk:
        for src, name in h.files:
            plan.copy(src.path(base) / name, Path(h.target) / name)
            copied.append(name)
    if h.record == CONTINUED:
        runrecord.write_continued_from(h.target, h.source, names=h.names,
                                       run=h.run, plan=plan)
    else:
        runrecord.write_gathered_from(h.target, list(h.files), plan=plan)
    if own:
        plan.carry_out()
    return copied


def prepare_attempt(jobset: JobSet, base_dir, stage_name: str, *,
                    continue_from: Optional[str] = None,
                    cold: bool = False,
                    carry: Optional[List[str]] = None,
                    named: bool = True, plan=None,
                    shape=None) -> "Attempt":
    """Set ONE stage up to run, and report what was done.

    ``plan`` receives the attempt's folder and everything put in it, and is
    read for the stage's files it already holds -- `prep` opens the attempt
    as part of its plan, written after the save (`job-system.md` § 5.0),
    and `launch` fills it as part of its own, written at the send; a caller
    with none has it written now.  ``shape`` is the layout as the caller read
    it (`Shape.named(task.shape)`, the one door; A16).

    ``named`` says who chose ``continue_from``: the person, by ``--from``, or
    the launch, by default (`job-system.md` § 5.4) -- so a refusal quotes
    what was typed, never a ``--from`` nobody typed.

    The five steps § 1.6 names: **resolve** the next ``run-<n>``, **create**
    it, **copy** the deck / monitor / shared package in, **copy** whatever this
    run continues from, and **report** — the report being the point, since
    preparing is still design and the split from starting is what gives you
    somewhere to look before committing cluster time.

    ``continue_from`` is a bundle-relative attempt directory —
    ``"01_coarse/run-0"``. **Which run is never a guess** (§ 1.6):
    continuing from ``run-0`` and from ``run-2`` are different scientific
    choices, so the callers pass one they can name -- a first launch, by
    default, the stage before it's newest attempt, which must have
    finished, or the one a person named (`continuation.continuation_answer`,
    `job-system.md` § 5.4); a launch again, the SAME stage's latest
    (user, 2026-08-21).  ``cold=True`` means start
    clean, and with a directory per attempt that is simply *skip the copy* —
    there is nothing to move aside, because a fresh attempt is empty unless
    something is put in it.  A REUSED attempt -- never launched -- is
    emptied whole first (:func:`empty_run`), whoever fills it: prep
    preparing it again, or the launch that sends it.

    ``carry`` names the files to copy; it defaults to :func:`warm_carry` for
    **this pair** — the stage being prepared and the stage that produced the
    attempt named by ``continue_from``. They are **copied, never linked** — the
    engine writes to those very filenames, and writing through a link would
    destroy the result you started from.

    ``stage_name`` goes through the ONE resolver, so it takes a name, a number
    or a token — the same three spellings every other surface takes, and the
    same refusal when it matches none of them.
    """
    base = Path(base_dir)
    sh = shape
    if not sh.keeps_attempts_as_directories:
        raise ValueError(FLAT_HAS_NO_ATTEMPTS)
    dir_of = job_dir_names(jobset, sh)
    refs = stage_refs(jobset)
    stage_name = resolve_stage_ref([refs[j.name] for j in jobset.jobs],
                                   stage_name).name
    job = jobset.job(stage_name)

    # WHAT IT CONTINUES FROM, CHECKED BEFORE ANYTHING IS WRITTEN (W52), so a
    # refusal leaves no attempt opened or stripped.
    names: List[str] = (
        continuation_files(jobset, base, stage_name, continue_from,
                           named=named, carry=carry, shape=sh)
        if continue_from and not cold else [])

    from .planned import Plan
    own = plan is None
    plan = Plan() if own else plan
    stage_dir = base / dir_of[stage_name]
    plan.folder(stage_dir)
    rn = run_names(jobset, job, sh)
    attempt, is_new = resolve_attempt(stage_dir, rn)
    # AN ATTEMPT NEVER LAUNCHED, REUSED -- prepared again, or filled at the
    # launch after a refused send: what was put into it before is taken
    # out first, so it starts from what this one takes.  A changed mind
    # that left an `.XV` lying there would change nothing -- the engine
    # finds it and warm-starts anyway, the *"present but not honoured"*
    # failure wearing its other face, and silent.
    if not is_new:
        empty_run(attempt, plan)
    # THE ONE OPENER (`open_run`): the attempt and every container down to
    # it, each saying what it is, which is not recoverable later: a bench trial's
    # directory is structurally identical to a stage's own, and § 1.4 calls
    # one a run and the other a container.
    open_run(base, attempt, plan)
    # THE RUN'S SCRATCH FOLDER, opened with the run (`engines/transport.md`
    # § 2a.11): the rung declares it (`Job.scratch_dir`).
    if job.scratch_dir:
        open_scratch(base, attempt / job.scratch_dir, plan)

    brought = bring_files(jobset, job, base, stage_dir, attempt, rn, plan)

    # WHAT IT CONTINUES FROM, ONE HAND-OVER (`job-system.md` § 5.4): the
    # files the run holds of what the stage takes, and the record of the
    # run they came from, beside the run.
    hand = None
    if continue_from and not cold:
        source = runrecord.RunRef.read(continue_from, shared=False)
        took = tuple((source, name) for name in names
                     if (source.path(base) / name).is_file())
        if took:
            hand = HandOver(attempt, rn, FIRST_ATTEMPT, CONTINUED,
                            files=took, source=source)
            hand_over(base, hand, plan)
    if own:
        plan.carry_out()

    return Attempt(
        stage=stage_name,
        dir=attempt,
        fresh=is_new,
        brought=brought,
        hand=hand,
    )


def continuation_files(jobset: JobSet, base_dir, stage_name: str,
                       continue_from: str, *, named: bool,
                       carry: Optional[List[str]] = None,
                       shape=None) -> List[str]:
    """The files ``stage_name`` would carry from ``continue_from`` --
    CHECKED, never copied: that attempt exists, the stage declares warm
    files for this pair (:func:`warm_carry`), and the attempt holds at least
    one of them.  Raises ``ValueError`` otherwise, worded for who chose the
    run: the person, by ``--from`` (``named``), or the door that chose it --
    a first launch's default, or a launch again's own latest run
    (`job-system.md` § 5.4).

    ONE CHECK, asked by :func:`prepare_attempt` before it writes anything
    and by `launch` before it shows what it will send -- so a run that
    cannot be continued is refused before the question, not after the yes.
    ``shape`` is the layout as the caller read it (`Shape.named(task.shape)`,
    A16).
    """
    base = Path(base_dir)
    said = (f"--from {continue_from!r}" if named else
            f"the run it continues from ({continue_from})")
    src = base / continue_from
    if not src.is_dir():
        raise ValueError(
            f"{said}: no such attempt under "
            f"{base}. Name an attempt directory that has already run, "
            f"e.g. '01_coarse/run-0'.")
    job = jobset.job(stage_name)
    if job is None:
        raise ValueError(f"{stage_name!r} is not prepared -- it has no job "
                         f"to carry its files.")
    # The pair, resolved here and nowhere else -- `--from` is what names
    # the source, so this is the first moment both stages are known.
    dir_of = job_dir_names(jobset, shape)
    names = (carry if carry is not None
             else warm_carry(job, _source_job(jobset, dir_of, continue_from)))
    if not names:
        raise ValueError(
            f"{said}: {stage_name!r} declares no "
            f"warm-restart files, so there is nothing to continue.\n"
            f"  A stage whose description says `restart: clean` carries "
            f"none of the group -- its deck omits MD.UseSaveXV / "
            f"DM.UseSaveDM / MD.UseSaveCG, so files copied in would sit "
            f"there unread (run-identity.md § 4, *present but not "
            f"honoured*).  A stage continues when its `restart` is "
            f"`continue` in task.json at its prep; or start it cold.")
    if not any((src / name).is_file() for name in names):
        raise ValueError(
            f"{said}: that attempt holds none "
            f"of the files this stage would continue from "
            f"({', '.join(names)}). Did it run?")
    return list(names)


def _source_job(jobset: JobSet, dir_of: Dict[str, str], continue_from):
    """Which job produced the attempt named by ``--from``, or ``None``.

    ``continue_from`` is bundle-relative and always ``<stage dir>/run-<n>``
    (`job-system.md` § 5.3), so the stage is the attempt's PARENT read back
    through the SAME naming authority that wrote it. Nothing is parsed out of
    the name: :func:`job_dir_names` is asked, and a parent that matches no job
    simply has no answer — which :func:`warm_carry` then treats as *unverified*
    rather than guessing.
    """
    if not continue_from:
        return None
    parent = str(Path(continue_from).parent)
    for j in jobset.jobs:
        if dir_of.get(j.name) == parent:
            return j
    return None


__all__ = [
    "StageHome", "stage_home", "ladder_homes", "described_refs", "Attempt",
    "CONTINUED", "GATHERED", "HandOver", "hand_over",
    "bench_owner", "bench_stage_of", "trial_dir",
           "trial_work_dir",
           "materialize", "job_dir_names", "stage_refs",
           "latest_attempt", "run_dir",
           "resolve_attempt", "ever_launched", "empty_run",
           "empty_old_points", "prepare_attempt", "open_sweep_run",
           "open_run", "open_container",
           ]
