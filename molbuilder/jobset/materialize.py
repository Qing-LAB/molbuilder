"""Materialize engine — turn a :class:`JobSet` into on-disk per-job
directories (docs/execution/job-system.md; naming: job-contracts § 6.3).

Filesystem ONLY: it knows nothing about schedulers or engines.  For each
job it creates the directory :func:`job_dir_names` assigns (a stage's
``<NN>_<name>/``, a trial's ``<NN>_<name>/bench/bench-<point>/``, the
bundle root for a stageless calculation, ``bench-<name>/`` for hand-built
sets) and copies in, as real files, the static ``shared`` package plus
the job's own ``script`` (`project-layout.md` § 1.0: a run directory holds
everything it runs from).

*(R8, 2026-08-12: this header still described Carry symlinks laid into a
producer's directory and "the submit engine's dependency ordering" — both
deleted 2026-08-10 with stage chaining (a carry is a COPY prep makes at
`--from`, and nothing orders anything), and the `_mb_point` helper it
called its ancestor is long gone.  A front door describing a deleted
design misleads at the file's most-read lines.)*
"""

from __future__ import annotations

from typing import TYPE_CHECKING
if TYPE_CHECKING:                      # annotations only
    from ..paths import Shape

import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .. import calcdirs, runrecord
from ..pseudos import PSEUDO_DIRNAME
from ..identity import StageRef, parse_stage_token, resolve_stage_ref
from .model import JobSet, warm_carry
from ..paths import (attempt_dir, attempts_in,
                     trial_name as _paths_trial_name,
                     trials_in as _paths_trials_in,
                     bench_containers_in as _bench_containers_in,
                     bench_container as _paths_bench_container)

@dataclass(frozen=True)
class StageHome:
    """Where a stage lives, and its number -- `execution/architecture.md`
    § 3's ``StageHome``, answered by :func:`stage_home` alone (§ 3.2)."""
    name: str
    seq: int
    #: ``<NN>_<name>`` -- the stage's directory in the hierarchy, its decks'
    #: token in either shape; ``""`` when no stage was named.
    token: str
    #: The stage's folder, or ``None`` when no calculation folder was given.
    dir: Optional[Path]


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
                 if folder is not None else None)))
    return tuple(homes)


def described_refs(base, task) -> List[StageRef]:
    """The description's stages that run, as refs (`identity.StageRef`)
    numbered by the one door (:func:`ladder_homes`), so ``#N`` names the
    folder ``NN_…`` everywhere -- the stages a verb that takes one offers.
    It was ``commands.enabled_refs`` until 2026-10-03, numbered by place."""
    return [StageRef(h.seq, h.name)
            for h, s in zip(ladder_homes(base, task), task.stages)
            if getattr(s, "enabled", True) is not False]


def stage_home(base, task, stage: Optional[str]) -> StageHome:
    """This stage's number, token and folder -- THE ONE DOOR every reader
    asks (`execution/architecture.md` § 3.2; W55 B8): :func:`ladder_homes`'
    answer for it.  It was ``prep.token_for`` until 2026-10-03, numbering by
    the stage's place in the description, so a stage removed after its prep
    renumbered every stage after it (W38 F4).

    An unknown stage is refused by name: an empty token would silently drop
    the stage from every artifact name (`job-contracts.md` § 6.3)."""
    from .errors import PrepError
    if not stage:
        # Asked without naming a rung; every ladder has one.
        return StageHome(name="", seq=0, token="",
                         dir=Path(base) if base is not None else None)
    from ..identity import stage_key
    for home in ladder_homes(base, task):
        if stage_key(home.name) == stage_key(stage):
            return home
    raise PrepError(f"stage {stage!r} is not in this description's "
                    f"ladder: {', '.join(s.name for s in task.stages)}.")


#: One attempt at running a stage.  ``project-layout.md`` § 1.5: immutable once
#: it has run, so a re-run is a NEW directory rather than an overwrite.



def trial_dir(shape, stage_token: Optional[str], job_name: str) -> str:
    """The path from the bundle to ONE trial's directory — **the rule**.

    ``<container>/bench-<point>``, where the container is the stage's bench
    folder in hierarchical and the flat one otherwise
    (:func:`bench_container`).

    **This exists because the rule was written twice.**
    :func:`job_dir_names` composed it for a whole JobSet, and
    `prep.prep_calculation` composed it again from the same two facts —
    with a comment saying so and calling it safe: *"the same one
    `job_dir_names` will answer for this job, computed from the same two
    facts (token + trial-ness), so the deck is born where the launch will
    look for it."*

    They agreed, and a second computation that must be kept in step by hand
    only ever agrees until something moves.  What moved was the attempt
    layer (`project-layout.md` § 1.5a): one side learned about `run-<n>`
    and the other did not, so the deck landed in the container while the
    shared package landed in the attempt.

    `prep` cannot call :func:`job_dir_names` instead — it is *building* the
    JobSet in the loop that needs the directory, so there is nothing to ask
    yet.  That is what makes a shared RULE the fix rather than a shared
    lookup.
    """
    return f"{_paths_bench_container(shape, stage_token)}/{_paths_trial_name(job_name)}"


def trials_in(container) -> "List[Path]":
    """The trial directories in a bench container — :func:`trial_dir`'s search.

    `project-layout.md` § 4.5: for every name it composes, the framework owns
    the search.  A caller wanting the trials globbed ``bench-*`` itself, which
    is this module's prefix spelled somewhere else.
    """
    c = Path(container)
    # The PREFIX is not spelled here: `paths.trial_point` is the reader for
    # `paths.trial_name`, and asking it is what keeps the pair in step.  This
    # returns PATHS where `paths.trials_in` returns the points -- two callers,
    # one rule (N4, 2026-09-09).
    return [c / _paths_trial_name(pt) for pt in _paths_trials_in(c)]


def trial_work_dir(container, shape) -> Path:
    """Where a trial's files GO, given the directory it lives in.

    :func:`trial_dir` (via :func:`job_dir_names`) answers *where does this
    trial live*; this answers *where does prep put its deck and package*,
    and the two differ the moment a trial keeps attempts
    (`project-layout.md` § 1.5a):

    * **hierarchical** — the attempt, ``bench-<point>/run-<n>``;
    * **flat** — the container itself, because flat separates attempts by
      the wrapper's filename index and has no directory layer to open.

    **It takes the container rather than recomputing it.**  A first version
    took ``(base, shape, stage_token, job_name)`` and rebuilt the path — and
    derived the token by a different route than `job_dir_names` does, so a
    flat grouped sweep put its record somewhere the reader did not look.
    That is the very divergence `trial_dir` was extracted to end, so this
    asks its caller for the answer instead of computing a second one.

    `resolve_attempt` is the rule, not restated: reuse the last attempt
    until it has been launched, then open the next.  Preparing twice before
    launching refreshes ``run-0`` rather than leaking ``run-1``.

    **The read-side twin is :func:`run_dir`** — *where does this trial
    actually run*, which needs no shape because by then the directory is
    there to be found.  Both resolve to the newest attempt, and that is the
    invariant: if they ever disagree, prep writes where nothing reads.
    """
    d = Path(container)
    if shape is None or not shape.keeps_attempts_as_directories:
        return d
    d.mkdir(parents=True, exist_ok=True)
    attempt, _fresh = resolve_attempt(d)
    return attempt


def shape_of(jobset: JobSet, base_dir) -> Optional["Shape"]:
    """The layout this bundle uses, read from its description.

    **The one place a surface asks.** `engines/stages.md` § 6.7 puts the shape
    in `task.json` and says *"`prep` **reads** it; it does not decide it"* —
    so this reads it, and every layer below takes the answer as an argument
    rather than going looking for it a second time.

    ``None`` only when there is no ``task.json`` to read — bundles produced
    before 2026-08-10, hand-built JobSets in the tests, and the OLD bench
    bundle format (which folds away at plan step 6 u5).
    :func:`job_dir_names` reads ``None`` as the hierarchy, which is what they
    all are. That fallback is transitional and dies with the last such
    bundle; it is **not** an inference from data, which § 6.7 forbids, but
    the absence of a file that is now always written.

    *(This branched on ``kind != "ladder"`` until 2026-08-12 — "a benchmark
    bundle carries no description and needs none" — which `generator.md` § 5
    said would stop being true under the fold, and did: a described sweep is
    a ParameterSet inside a described calculation, shaped like anything
    else.)*
    """
    from ..task import FILENAME, read_task
    from ..paths import Shape
    desc = Path(base_dir) / FILENAME
    if not desc.is_file():
        return None
    return Shape.named(read_task(desc).shape)


def sweep_set_paths(bundle) -> "List[Path]":
    """Every place a SWEEP's ``job-set.json`` can be in this bundle.

    The search counterpart of :func:`bench_container`, and it lives beside it
    for that reason: the namer says where a sweep's state GOES and this says
    where to look for it, so a layout change moves one file instead of two
    that must be kept in step by hand.

    **N4, 2026-09-09: it now asks, and this docstring used to say why it
    could not.**  It read: *"It cannot simply call `bench_container` -- that
    takes a shape and a token, and the caller this exists for is an error path
    with no `job-set.json` to read them from.  That asymmetry -- one door to
    COMPOSE a path, none to FIND one -- is what the paths framework is for;
    when it lands, this is one of its callers and the patterns below move into
    it."*  It landed: `paths.bench_containers_in` is `bench_container`'s search
    half, and it takes ``shape=None`` for exactly this caller.

    So the two globs are gone.  They were also WIDER than the rule -- ``*/`` at
    depth 1 matches any directory, not just a declared container -- and the
    caller had to read every hit to find out whether it was a sweep at all.
    Asking narrows the answer to the containers the layout declares.

    Returns the paths that exist, unread: whether a set is a sweep is in the
    file, and reading it is the caller's business.
    """
    from .model import FILENAME as _JS
    base = Path(bundle)
    return sorted(p for p in (base / rel / _JS
                              for rel, _tok in _bench_containers_in(base))
                  if p.is_file())


def bench_stage_of(base, where) -> "Optional[str]":
    """The stage a sweep measures, by its name -- read off where the sweep
    lives: the bench container that holds ``where`` (the container itself,
    or a trial in it), through the layout's search half
    (`paths.bench_containers_in`), so both layouts answer.  A refusal or a
    status names the stage's own verbs with it, never a ``<stage>``.
    ``None`` for a sweep in no stage's container -- a hand-built one.

    *(It read the folder's first part as a stage token until 2026-10-01 --
    the hierarchy's ``01_coarse/bench`` -- and so named no flat sweep's
    stage: flat qualifies the container's own name, ``bench_01_coarse``;
    the W52 review.)*"""
    from ..identity import command_stage
    try:
        rel = Path(where).resolve().relative_to(Path(base).resolve())
    except (ValueError, OSError):
        return None
    for name, token in _bench_containers_in(base):
        if token and (rel == Path(name) or Path(name) in rel.parents):
            return command_stage(token)
    return None


def bench_owner(folder) -> "Optional[Tuple[Path, str]]":
    """``(calculation folder, stage name)`` when ``folder`` IS a stage's
    bench container -- the declared one of the calculation one level up
    (flat's ``bench_<NN>_<stage>``) or two (the hierarchy's
    ``<NN>_<stage>/bench``), through the layout's search half -- else
    ``None``.  A sweep's own job-set names its trials from the calculation,
    so a reader standing in its container reads them from there (W52:
    `status` in a bench folder read every trial as never prepped, and named
    the folder as the calculation)."""
    from ..identity import command_stage
    from ..task import FILENAME as _TASK
    f = Path(folder).resolve()
    for calc in (f.parent, f.parent.parent):
        if not (calc / _TASK).is_file():
            continue
        rel = f.relative_to(calc)
        for name, token in _bench_containers_in(calc):
            if token and Path(name) == rel:
                return calc, command_stage(token)
    return None


def job_dir_names(jobset: JobSet, shape: "Shape" = None) -> Dict[str, str]:
    """``{job name: directory name}`` for a whole JobSet — the naming authority.

    One question, not two kinds (`generator.md` § 5): *does this job have a
    stage, a point, or both?*

    | the deck says | the set says | directory |
    |---|---|---|
    | a stage token, job named for the stage | — | ``<NN>_<name>`` — the rung itself |
    | a stage token, job named by coordinate | — | a trial, in the stage's bench CONTAINER (:func:`bench_container`): ``<NN>_<name>/bench/bench-<point>`` hierarchical, ``bench_<NN>_<name>/bench-<point>`` flat |
    | no token | ``kind="ladder"``, job named AS the set | ``.`` — the bundle root |
    | no token | ``kind="sweep"`` | ``bench/bench-<name>`` — the trial, in the bare container where its sweep's record already sits (until 2026-08-13 these fell to the root, final review A-2) |
    | no token | ``kind="ladder"``, job named its own way | ``bench-<name>`` at the root — told apart by name alone |

    **No DESCRIPTION reaches the three tokenless rows any more.**  They were
    written for `engines/stages.md` § 6.5's stage-LESS calculation, which
    that section retired on 2026-08-16: every description now carries at
    least one stage, one stage is named and tokened like any other, so every
    described deck carries a token and takes one of the first two rows.
    What still arrives here tokenless is a HAND-BUILT :class:`JobSet` — one
    assembled in code with no description behind it — and the rows stay
    because the naming authority must answer for those too.  They are no
    longer a statement about what a calculation can be.

    Until 2026-08-10 every kind got the trial prefix, so a staged run's
    directories came out ``point-coarse/`` (`worked-example.md` gap 6); until
    2026-08-12 the split was a branch on ``JobSet.kind`` and trials could not
    nest at all.  Now it is read off each deck's own name.

    **The seq is read back off the deck, not counted here.** ``job.script`` is
    ``<label>_<NN>_<name>.fdf`` (decision 27), so the token the directory is
    named for is the one the deck already carries — which is what makes
    ``<NN>_<name>/<label>_<NN>_<name>.fdf`` a self-check rather than a
    repetition (§ 4.1). Counting positions here instead would reintroduce
    exactly what `engines/stages.md` R5 forbids: a number that shifts when the
    ladder changes, silently handing one stage's directory to another.

    **A tokenless job is the one place ``kind`` is consulted, and that is
    not the branching the paragraph below forbids** (R1, 2026-08-12).  For
    a TOKENED job the deck already answers, and asking ``kind`` a second
    time is how directory and deck disagree.  For a tokenless job the deck
    says nothing — the table's question has the answer *neither* — and
    ``kind`` is the only data left: a stageless described RUN is the
    calculation itself (its deck, wrapper and attempts live at the root),
    while a hand-built SWEEP's points are siblings told apart by name.
    Until R1 both fell to ``bench-<name>``, which put a stageless
    calculation's RUN in a directory named for a benchmark, made its
    attempt unreachable, and broke `engines/stages.md` § 6.5's
    single-parameter-set form
    end-to-end.  Inventing a seq for either would still be guessing at the
    one number that must never be guessed.

    **The conventions meet in one expression**, because :func:`stage_refs`
    already answered which applies: a job with an ordinal has a token and
    is named by it; a job without one has no token and falls to the
    kind-split above.

    ``shape`` decides where a **stage** sits: hierarchical gives each one a
    directory, flat is depth 1 and they all sit in the bundle root
    (:class:`~molbuilder.paths.Shape`).  A described trial nests
    under its stage's directory, so the shape reaches it through the stage;
    only the tokenless fallback ignores it.

    ``None`` means *hierarchical*, and it now means only one thing: **a ladder
    with no description to read**. Every surface resolves the shape through
    :func:`shape_of` and passes it down, so the default is reached by
    hand-built JobSets (the tests) and by bundles produced before `task.json`
    was written — both of which are hierarchical, because that is the only
    shape a JobSet was emitted for.

    It is not an inference from data, which § 6.7 forbids; it is the absence of
    a file that is now always written. **This paragraph said "no producer emits
    a flat ladder yet" until 2026-08-10, and that stopped being true in the
    commit that made flat emit one** — the kind of sentence that survives the
    change it describes because nothing executes it.
    """
    from ..paths import Shape
    sh = shape or Shape.named("hierarchical")
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
            # stages' benchmarks can never collide.  Until 2026-08-12 the
            # trials sat directly in the stage; until 2026-08-13 this line
            # spelled the flat container ``bench/`` itself, unqualified --
            # exactly the two-flat-stages collision 2026-08-12 plan A5
            # closed on the record side (final review A-1):
            # bench_container is now the one spelling for both sides.
            out[j.name] = trial_dir(sh, trial_token, j.name)
            continue
        # Tokenless: the deck says nothing, so the SET is the only data
        # left (see the docstring's R1 paragraph -- no DESCRIPTION
        # reaches these rows any more; what still arrives tokenless is
        # a HAND-BUILT JobSet).  Such a ladder runs its own-named jobs
        # as siblings at the root, and such a sweep's points live in the
        # bare ``bench/`` container beside their own record (A-2,
        # 2026-08-13).
        if jobset.kind == "ladder":
            out[j.name] = ("." if j.name == jobset.name
                           else _paths_trial_name(j.name))
        else:
            # THE SAME RULE with no stage token, so it asks for it too --
            # a third spelling of `<container>/bench-<point>` is a third
            # thing to keep in step.
            out[j.name] = trial_dir(sh, "", j.name)
    return out


def _trial_stage_token(jobset: JobSet, job) -> Optional[str]:
    """The ``<NN>_<stage>`` a TRIAL's deck carries, or ``None``.

    A trial's script is ``<label>-<point>_<NN>_<stage>.ext`` — its own § 6.3
    label (the calculation's, qualified by the coordinate) plus the stage
    token.  Anchoring the parse on that full label is what keeps a stage
    name containing ``_`` unambiguous, exactly as for a rung
    (`identity.parse_stage_token`).
    """
    from ..identity import stage_token
    parsed = parse_stage_token(os.path.basename(job.script),
                               f"{jobset.name}-{job.name}")
    return stage_token(*parsed) if parsed else None


def stage_refs(jobset: JobSet) -> Dict[str, StageRef]:
    """``{job name: StageRef}`` for **every** job — *which stage is this?*

    This is the after-produce half of the resolver (§ 8f) and **the only place
    the two kinds are told apart**. ``seq`` is recovered from each deck's own
    token, which is where `project-layout.md` § 4.1 says it lives: *"read off
    the directory name and stored nowhere else"*. Nothing here counts
    positions, so a disabled stage leaves a gap rather than renumbering.

    **Total on purpose.** Every job gets a ref; one with no assigned ordinal
    gets ``seq=None`` rather than being left out of the mapping. Omission was
    the shape until 2026-08-10, and it pushed the same question — *what if
    there is no ordinal?* — out to four callers, who answered it four different
    ways: ``bench-<name>`` here, the row number in ``plan``, ``None`` in
    ``runstatus``, and a whole second lookup-and-refusal branch in the CLI.
    Two of those four printed a **position** where a reader reads an ordinal.
    A total answer is what lets each caller read one and never test membership.

    ``seq=None`` is still never a guess: a sweep point has no order at all, and
    a ladder job whose deck carries no token has an ordinal nobody assigned
    (§ 4.2's number is assigned once and never invented).

    The ref carries the **job's** name, not the token's. They are the same
    string for anything a producer built — ``siesta/stages.py`` names each job
    for its stage — and where they could differ it is the job name that
    dependency edges, ``--stage-resources`` keys and the CLI all point at, so
    resolving to the other one would hand back a name this JobSet does not have.
    """
    # NO kind branch (2026-08-12): the parse is anchored on the jobset's
    # label, so a TRIAL's script (whose label is the coordinate-qualified
    # one) never matches and gets seq=None -- the same answer the old
    # ``if ladder`` guard produced, read off the deck instead of a field.
    out: Dict[str, StageRef] = {}
    for j in jobset.jobs:
        parsed = parse_stage_token(os.path.basename(j.script), jobset.name)
        out[j.name] = StageRef(parsed[0] if parsed else None, j.name)
    return out


def materialize(jobset: JobSet, base_dir) -> List[Path]:
    """Create each job's directory under ``base_dir`` with its copies.

    Returns the list of created job directories (in JobSet order).  Idempotent:
    re-running refreshes the copies without duplicating anything.  Raises
    ``ValueError`` if the JobSet is structurally invalid (so a bad carry /
    duplicate name can't produce a broken tree).
    """
    errors = jobset.validate()
    if errors:
        raise ValueError(
            "cannot materialize an invalid JobSet:\n  - "
            + "\n  - ".join(errors))
    base = Path(base_dir)
    created: List[Path] = []
    sh = shape_of(jobset, base_dir)
    dirs = job_dir_names(jobset, sh)
    for job in jobset.jobs:
        # A TRIAL KEEPS ATTEMPTS EXACTLY AS A STAGE DOES, and the shape
        # decides (`project-layout.md` § 1.5a).  `trial_work_dir` is the
        # one answer to *where do this trial's files go*, and `prep` asks
        # the same one -- when only this side knew, the package moved into
        # `run-0` and the deck stayed in the container.
        d = base / dirs[job.name]
        if jobset.kind == "sweep":
            d = trial_work_dir(d, sh)
        d.mkdir(parents=True, exist_ok=True)
        created.append(d)
        if d.resolve() == base.resolve():
            # FLAT: depth 1 (`project-layout.md` § 1) -- the job runs in the
            # bundle root, where every file it needs ALREADY SITS.  There is
            # nothing to link, and linking would DESTROY: `relink` unlinks the
            # existing entry first, and ``../<name>`` points outside the
            # bundle.  Without this guard a flat prep replaced its own decks,
            # wrappers and monitor with dangling symlinks to the parent
            # directory -- found by M5 pass 1, 2026-08-10.
            #
            # The carry is skipped for the same reason and a second one: flat's
            # warm files are ONE SHARED SET at the root (§ 1), so the next
            # stage finds them lying there; there is no producer directory to
            # reach into.
            #
            # EXCEPT THE PSEUDOPOTENTIALS, and the exception is this guard's
            # own premise going stale (fixed 2026-09-11).  "Every file it
            # needs already sits" held until `engines._pseudo_dir` began ADOPTING
            # root `<El>.psml` into `pseudos/` (2026-08-28, 08656f2c) -- which
            # in flat is the run directory being emptied of the one input
            # SIESTA cannot look for anywhere else ("it opens
            # `<element>.psml` in the directory it runs from and has no search
            # path").  `project-layout.md` § 2241 says each run directory
            # receives its own copies and `_pseudo_dir`'s own docstring says
            # the run directories are untouched by it; in flat both stopped
            # being true, so every flat SIESTA prep since that date rendered a
            # deck that dies in `initatom` with "Pseudopotential file not
            # found".  Hierarchical never reached this branch and never broke.
            for _ps in sorted((base / PSEUDO_DIRNAME).glob("*.psml")):
                _dst = d / _ps.name
                if not _dst.exists():
                    shutil.copy2(_ps, _dst)
            continue
        # The static package arrives as REAL COPIES (user, 2026-08-24;
        # `project-layout.md` § 1.0: the run directory "holds everything",
        # and a symlink holds nothing).  These were relative symlinks to
        # root copies, which is how a ten-trial sweep came to keep its 50
        # rendered files at the bundle root with directories full of
        # pointers.  The deck is NOT in this list any more: it is born in
        # the directory (`prep_calculation` / step 1's adoption), so
        # there is no root copy to reach for.
        import shutil as _sh
        for fname in list(jobset.shared):
            src = base / fname
            dst = d / os.path.basename(fname)
            if not src.is_file():
                continue          # prep's own missing-input gates report it
            if dst.is_symlink():
                dst.unlink()      # a pre-2026-08-24 bundle's link, replaced
            if not dst.is_file():
                _sh.copy2(src, dst)
        # NOTHING ELSE IS LINKED IN.  A second loop here laid the `Carry`
        # symlinks -- into a producer's directory, before the producer had
        # run, so they dangled by design.  Deleted 2026-08-10 with `Carry`
        # itself: what a stage continues from is a real file COPIED by
        # `prepare_attempt` from the attempt you name (project-layout.md 1.6).
    return created


# --------------------------------------------------------------------- #
#  Attempts — one directory per try at a stage (project-layout.md § 1.6)  #
# --------------------------------------------------------------------- #


def launch_record_at(kind: str, job, container: Path,
                     attempt: Optional[Path]) -> Tuple[Path, Optional[str]]:
    """``(where, basename)`` -- where a job's launch is recorded
    (`project-layout.md` § 1.6.3), for :func:`runrecord.launch_record_path`,
    :func:`runrecord.launch_record`, :func:`runrecord.write_launch` and every reader: the
    attempt's ``run.json`` when there is an attempt; a sweep trial's own, at
    the trial's top; a flat stage's ``<basename>.run.json`` in the
    calculation's directory, which every stage shares -- ``basename`` its
    deck's stem.  ONE answer: the writer and three readers each spelled it
    until 2026-10-01, and the one that did not take the flat case relaunched
    a flat stage still in the queue (W52)."""
    if attempt is not None:
        return Path(attempt), None
    if kind == "sweep":
        return Path(container), None
    return Path(container), Path(job.script).stem


def latest_attempt(stage_dir: Path) -> Optional[Path]:
    """The newest attempt under ``stage_dir``, or ``None`` if there are none.

    **Where a stage's state actually is.** `project-layout.md` § 1.5 is flat
    about it — *"Where a run happens: inside the attempt directory"* — and
    *"everything the run writes"* is *"created in place"*, because the wrapper
    is invoked there. So anything asking *what happened to this stage?* asks
    here first, and only falls back to the container for a flat run, which
    § 1.5 says is untouched and *"is a run"* in its own right.

    This is a layout question, so it is answered in the layout layer rather
    than by each observer working out where to look. ``runstatus`` globbed the
    container until 2026-08-10 and therefore reported a finished hierarchical
    stage as *"prepped, not launched"* — forever.
    """
    ns = attempts_in(stage_dir)
    return attempt_dir(stage_dir, ns[-1]) if ns else None


def run_dir(container: Path) -> Path:
    """**The directory a stage or trial actually uses** — its newest attempt
    when the shape keeps them, the container itself when it does not.

    THE OTHER HALF OF :func:`latest_attempt`, which answers *is there an
    attempt* and is right to return ``None`` for flat.  Almost every caller
    wants *where do I look*, and each was writing the fallback itself:
    ``latest_attempt(d) or d`` in two places, ``attempt or d`` in a third,
    ``att if att is not None else container`` in a fourth.  Five spellings
    of one rule, in four files.

    That is the shape the 2026-08-30 failure came in.  When § 1.5a gave
    sweep trials attempts (2026-08-27), the observers that spelled the rule
    were migrated and the two places in `submit` that had quietly composed a
    CONTAINER path instead were not: the grouped bench then ``cd``ed a level
    above its wrapper (every trial rc=127, Sol job 62372574) and wrote
    ``run.json`` where nothing read it (every re-launch re-submitted).

    `latest_attempt` already says the principle -- *"this is a layout
    question, so it is answered in the layout layer rather than by each
    observer working out where to look"*.  It was answering half of it.

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


def stage_stdout(attempt_dir: Path, label: str, token: Optional[str],
                 engine: str) -> Optional[Path]:
    """The newest of THIS stage's engine outputs in *attempt_dir*, or ``None``.

    The stdout roles are the catalogue's (`runfiles.stdout_roles`) and the
    name is read by the grammar (`runfiles.find`, stage and run counter
    included), so a flat bundle -- one directory holding every stage's
    output -- answers with this stage's file and never a neighbour's.  Two
    readers wanted this: `prep`, for the geometry a `relax` stage left, and
    `summarize`, for the reference step of a force-constant run; the second
    globbed ``*.out`` until 2026-09-24 and on a flat bundle would have read
    the other stage's run.
    """
    from ..runfiles import find as _rf_find
    from ..runfiles import stdout_roles
    hits = [p for role in stdout_roles(engine)
            for p, _rec in _rf_find(Path(attempt_dir), label, role=role,
                                   stage=(token or None))]
    return hits[-1] if hits else None


def resolve_attempt(stage_dir: Path) -> Tuple[Path, bool]:
    """The attempt directory to prepare into, and whether it is a fresh one.

    § 1.6: *"Preparing again is safe until the run has been launched.
    Otherwise splitting the two steps leaks directories — prepare, change your
    mind, prepare again, and an empty ``run-3`` sits there forever."*

    So the last attempt is REUSED when it has not been launched, and a new one
    is opened only when the last has. That also makes the numbering mean
    something: every ``run-<n>`` on disk was actually started.
    """
    existing = attempts_in(stage_dir)
    if existing:
        last = attempt_dir(stage_dir, existing[-1])
        try:
            launched = runrecord.launch_record(last) is not None
        except runrecord.LaunchRecordError as e:
            from .errors import PrepError
            raise PrepError(str(e)) from e
        if not launched:
            return last, False
        return attempt_dir(stage_dir, existing[-1] + 1), True
    return Path(stage_dir) / "run-0", True


@dataclass(frozen=True)
class Attempt:
    """One try at a stage: the directory, and what was put in it.

    **§ 9.4's fourth value object, and the author's own smell.**
    :func:`prepare_attempt` returned a ``Dict[str, object]`` when it landed on
    2026-08-10 — *"a bag the CLI unpacks by string key"* — so every surface
    spelled ``rep["continued_from"]`` and a typo was a ``KeyError`` at best and
    a silent ``None`` at worst. The dict was noticed while being written and
    shipped anyway, which is the argument for naming the habit rather than the
    instance.

    ``fresh`` is False when an unlaunched attempt was **reused** rather than
    opened — § 1.6's *"preparing again is safe until the run has been
    launched"*, which is what keeps a changed mind from leaking empty
    directories. ``continued_from`` is **None** when this run starts from the
    structure, and that is a different claim from *"continued from nothing"*
    (`checkpointing.md` S3), which is why `run.json` omits the key entirely
    rather than writing null.
    """
    stage:          str
    dir:            Path
    fresh:          bool
    brought:        List[str]
    copied:         List[str]
    continued_from: Optional[str]
    cold:           bool


#: Why ``--from`` and ``--cold`` mean nothing on the flat layout -- said by
#: `continuation` before prep writes anything, and by `prepare_attempt`
#: to a caller that asks it anyway.
FLAT_HAS_NO_ATTEMPTS = (
    "this calculation's shape is 'flat', which has no attempt directories "
    "to open: runs are told apart by the wrapper's output index "
    "(<label>_<NN>_<name>-run<N>.out) and every stage reads the files the "
    "stage before it left in the one folder (project-layout.md § 1) -- so "
    "there is no run to name with --from, and a stage starts clean by its "
    "run card's `restart: clean`, not by --cold.")


def prepare_attempt(jobset: JobSet, base_dir, stage_name: str, *,
                    continue_from: Optional[str] = None,
                    cold: bool = False,
                    carry: Optional[List[str]] = None,
                    container: Optional[Path] = None,
                    named: bool = True) -> "Attempt":
    """Set ONE stage up to run, and report what was done.

    ``named`` says who chose ``continue_from``: the person, by ``--from``, or
    `prep`, by default (`job-system.md` § 5.4) -- so a refusal
    quotes what was typed, never a ``--from`` nobody typed.

    The five steps § 1.6 names: **resolve** the next ``run-<n>``, **create**
    it, **copy** the deck / monitor / shared package in, **copy** whatever this
    run continues from, and **report** — the report being the point, since
    preparing is still design and the split from starting is what gives you
    somewhere to look before committing cluster time.

    ``continue_from`` is a bundle-relative attempt directory —
    ``"01_coarse/run-0"``. **Which run is never a guess** (§ 1.6):
    continuing from ``run-0`` and from ``run-2`` are different scientific
    choices, so the callers pass one they can name -- `prep`, by default,
    the stage before it's newest attempt, which must have concluded, or the
    one a person named (`continuation.continuation_answer`,
    `job-system.md` § 5.4); the
    submission door, re-submitting a launched stage, the SAME stage's latest
    (user, 2026-08-21).  ``cold=True`` means start
    clean, and with a directory per attempt that is simply *skip the copy* —
    there is nothing to move aside, because a fresh attempt is empty unless
    something is put in it.  Re-preparing a REUSED attempt, either statement
    first takes away what an earlier one carried in; saying neither leaves
    the attempt's carry as it is.

    ``carry`` names the files to copy; it defaults to :func:`warm_carry` for
    **this pair** — the stage being prepared and the stage that produced the
    attempt named by ``continue_from``. They are **copied, never linked** — the
    engine writes to those very filenames, and writing through a link would
    destroy the result you started from.

    ``stage_name`` goes through the ONE resolver, so it takes a name, a number
    or a token — the same three spellings every other surface takes, and the
    same refusal when it matches none of them. It spelled its own lookup and
    its own refusal until 2026-08-10, listing *"coarse, medium, tight"* with no
    order at the one moment you are choosing which stage to run. That is the
    gap decision 28 names, and a second listing format is how it comes back.
    """
    base = Path(base_dir)
    sh = shape_of(jobset, base_dir)
    if sh is not None and not sh.keeps_attempts_as_directories:
        raise ValueError(FLAT_HAS_NO_ATTEMPTS)
    dir_of = job_dir_names(jobset, sh)
    refs = stage_refs(jobset)
    stage_name = resolve_stage_ref([refs[j.name] for j in jobset.jobs],
                                   stage_name).name
    job = next(j for j in jobset.jobs if j.name == stage_name)

    # WHAT IT CONTINUES FROM, CHECKED BEFORE ANYTHING IS WRITTEN (W52).  The
    # attempt was opened and filled, and an earlier carry undone, before
    # these refusals until 2026-10-01 -- so a mistyped --from stripped an
    # attempt prepared a moment ago, and a re-launch that could not continue
    # left a fresh attempt behind it.
    names: List[str] = (
        continuation_files(jobset, base, stage_name, continue_from,
                           named=named, carry=carry)
        if continue_from and not cold else [])

    # ``container`` overrides WHERE the run-<n> opens -- the transport
    # composite's bias scan keeps one attempt ladder PER POINT
    # (``04_device/v0.2/run-<n>``; archive/2026-09-01-transport-design.md § 4.3, layout
    # ruled 2026-08-29), and the point's directory already holds its own
    # deck + wrapper, so everything below reads it exactly like the
    # stage's own directory.  Default: the job's own, as ever.
    stage_dir = (Path(container) if container is not None
                 else base / dir_of[stage_name])
    stage_dir.mkdir(parents=True, exist_ok=True)
    attempt, is_new = resolve_attempt(stage_dir)
    attempt.mkdir(parents=True, exist_ok=True)

    # WHAT EACH OF THESE DIRECTORIES IS, said by the code that just made them
    # (`project-layout.md` § 1.4a, invariant 6b).  This is the one place both
    # kinds are created, so it is the one place that knows which is which --
    # and knowing is not recoverable later: a bench trial's directory is
    # structurally identical to a stage's own, and § 1.4 calls one a run and
    # the other a container.
    #
    # EVERY container down the chain, not just the leaf: a bias scan passes
    # `container=<...>/v0.2`, whose parent `04_device/` is then created by
    # `parents=True` and would be the one directory in the tree that never
    # answered.
    for _c in reversed(stage_dir.parents):
        if _c == base or base not in _c.parents:
            continue
        calcdirs.write(_c, role=calcdirs.CONTAINER, root=base)
    calcdirs.write(stage_dir, role=calcdirs.CONTAINER, root=base)
    calcdirs.write(attempt, role=calcdirs.RUN, root=base)

    # Inputs: the deck, wrappers and shared package, COPIED in -- real
    # files, per L2 (roadmap 7.10; `project-layout.md` § 1.0: the run
    # directory "holds everything").  These were relative symlinks up to
    # the bundle root, laid with a computed prefix; since 2026-08-24 the
    # rendered files are BORN in the stage directory, so the stage dir is
    # the source and the root is only a legacy fallback (a bundle prepped
    # before the layout repair).  Identical bytes for every attempt argued
    # for links once; a synced-back bundle whose links dangled on the
    # other machine is the argument that outranks it.
    import shutil as _sh
    brought: List[str] = []

    def _bring(fname: str) -> None:
        bn = os.path.basename(fname)
        dst = attempt / bn
        for src in (stage_dir / bn, base / fname, base / bn):
            if src.is_file() and src.resolve() != dst.resolve():
                # REFRESHED every time, exactly as the old relink was
                # (unlink + relay): a REUSED unlaunched attempt must see
                # the re-prep's deck, not the first prep's -- skip-if-
                # exists here kept a stale ELPA-2STAGE deck under a
                # re-prep whose pin said otherwise (caught by
                # test_a_declared_pin_reaches_the_run_deck..., 2026-08-24).
                if dst.is_symlink() or dst.exists():
                    dst.unlink()
                _sh.copy2(src, dst)
                brought.append(bn)
                return

    for fname in [job.script] + list(jobset.shared):
        _bring(fname)
    # The monitor and what it imports, as ONE file (`runwrap.MONITOR_BUNDLE`,
    # built from `MONITOR_COMPANIONS`): one file cannot be half-shipped, which
    # is how `config_dir.py` once travelled with bench trials and not with run
    # attempts, killing every run's monitor at import.
    # makov_payne_correction.py: the post-run script a CHARGED deck's own
    # header instructs the user to run "after SIESTA finishes" -- HERE,
    # beside the .out.
    # The FINISH, when the job has one (`Job.finish`): the bundle its wrapper
    # runs after the engine, beside the deck -- `engines/vibration.md` § 5.5.
    from ..runwrap import MONITOR_BUNDLE
    for extra in (MONITOR_BUNDLE, "makov_payne_correction.py",
                  *((job.finish,) if job.finish else ())):
        _bring(extra)
    stem = Path(job.script).stem
    for wrapper in (f"{stem}.run.sh", f"{stem}.sbatch"):
        _bring(wrapper)

    # Re-preparing an attempt that was already carried into: UNDO the previous
    # carry first.  § 1.6 makes re-prep *"changing your mind about the setup"*,
    # and a mind changed from ``--from A`` to ``--cold`` that leaves A's ``.XV``
    # lying in the directory has changed nothing -- the engine finds it and
    # warm-starts anyway.  That is the *"present but not honoured"* failure
    # wearing its other face, and it is silent.  Only files the marker says we
    # carried in are removed, and never a symlink, so nothing a user put here
    # by hand is touched.
    #
    # ONLY WHEN THE CALLER SAYS WHAT IT NOW CONTINUES FROM -- a run, or
    # ``cold``.  A caller that says neither changes nothing: prep's five
    # steps opened the attempt that way and undid a carry a moment before its
    # own refusal, which left an attempt prepared a minute ago stripped of
    # what it was to start from (W52).
    marker = runrecord.continued_from_marker(attempt)
    if not is_new and marker.is_file() and (cold or continue_from):
        # The WHOLE declared set, not the pair-filtered one: the previous prep
        # may have named a different source and so copied a conditional file
        # this one would not, and a mind changed from `--from A` to `--cold`
        # that leaves A's `.CG` behind has changed nothing.
        for w in job.warm:
            f = attempt / w.name
            if f.is_file() and not f.is_symlink():
                f.unlink()
        marker.unlink()

    copied: List[str] = []
    if continue_from and not cold:
        src = base / continue_from
        for name in names:
            f = src / name
            if f.is_file():
                shutil.copy2(f, attempt / name)
                copied.append(name)

    # Leave the provenance where ``launch`` can find it: prep is what knows
    # which attempt this one continues from, and submit writes run.json.  A
    # marker file beats threading the value through a launch argument that
    # every caller would have to remember to pass.
    if copied:
        marker.write_text(str(continue_from) + "\n", encoding="utf-8")

    return Attempt(
        stage=stage_name,
        dir=attempt,
        fresh=is_new,
        brought=brought,
        copied=copied,
        continued_from=(None if (cold or not continue_from)
                        else str(continue_from)),
        cold=bool(cold),
    )


def continuation_files(jobset: JobSet, base_dir, stage_name: str,
                       continue_from: str, *, named: bool,
                       carry: Optional[List[str]] = None) -> List[str]:
    """The files ``stage_name`` would carry from ``continue_from`` --
    CHECKED, never copied: that attempt exists, the stage declares warm
    files for this pair (:func:`warm_carry`), and the attempt holds at least
    one of them.  Raises ``ValueError`` otherwise, worded for who chose the
    run: the person, by ``--from`` (``named``), or the door that chose it --
    `prep`'s default, `launch` re-launching a stage (`job-system.md` § 5.4).

    ONE CHECK, asked by :func:`prepare_attempt` before it writes anything
    and by `launch` before it shows what it will send -- so a run that
    cannot be continued is refused before the question, not after the yes.
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
    job = next(j for j in jobset.jobs if j.name == stage_name)
    # The pair, resolved here and nowhere else -- `--from` is what names
    # the source, so this is the first moment both stages are known.
    dir_of = job_dir_names(jobset, shape_of(jobset, base))
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

    The parent, not the first path component (A-3, 2026-08-13): a STAGELESS
    calculation's stage dir is ``.`` — its attempts sit at the root, so
    ``--from run-0`` has ``run-0`` as its head and ``.`` as its parent.
    Matching on the head could never equal ``.``, so continuing a stageless
    calculation from its own attempt read as *unverified* and silently
    withheld every conditional carry (``.CG``) — prep still reported
    success.  ``Path("run-0").parent`` is ``"."``, exactly the naming
    authority's answer for the `engines/stages.md` § 6.5 root job.
    """
    if not continue_from:
        return None
    parent = str(Path(continue_from).parent)
    for j in jobset.jobs:
        if dir_of.get(j.name) == parent:
            return j
    return None


__all__ = [
    "StageHome", "stage_home", "ladder_homes", "described_refs", "Attempt", "bench_owner", "bench_stage_of", "trial_dir",
           "trial_work_dir",
           "trials_in",
           "materialize", "job_dir_names", "stage_refs",
           "latest_attempt", "run_dir",
           "resolve_attempt",
           "prepare_attempt",
           ]
