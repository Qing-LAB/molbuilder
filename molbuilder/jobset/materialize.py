"""Materialize engine — turn a :class:`JobSet` into on-disk per-job
directories (docs/execution/job-system.md; naming: job-contracts § 6.3).

Filesystem ONLY: it knows nothing about schedulers or engines.  For each
job it creates the directory :func:`job_dir_names` assigns (a stage's
``<NN>_<name>/``, a trial's ``<NN>_<name>/bench/bench-<point>/``, the
bundle root for a stageless calculation, ``bench-<name>/`` for hand-built
sets) and lays relative symlinks for the static ``shared`` package plus
the job's own ``script``.

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
    from .shape import Shape

import json
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ..identity import StageRef, parse_stage_token, resolve_stage_ref
from .model import JobSet, warm_carry

#: One attempt at running a stage.  ``project-layout.md`` § 1.5: immutable once
#: it has run, so a re-run is a NEW directory rather than an overwrite.
ATTEMPT_RE = re.compile(r"^run-(\d+)$")

#: Written by ``launch`` into the attempt, AFTER the launch succeeds
#: (``project-layout.md`` § 1.6).  Its presence is the only honest answer to
#: *has this been launched?* -- a queued job has produced nothing yet, so
#: "no output" and "not started" are indistinguishable from the directory alone.
RUN_LAUNCH_SCHEMA = "molbuilder/run-launch@1"
RUN_LAUNCH_FILE = "run.json"


def job_dir_name(job_name: str) -> str:
    """The on-disk directory for a **trial** — ``bench-<point>``.

    `job-contracts.md` § 6.3 is the authority: ``bench-`` plus the coordinate
    as ONE qualifier (``bench-G1K4C6``).  This wrote ``point-…`` until the
    fold (C6, 2026-08-12) — the name the docs had already retired.

    A ladder's stage directory is NOT this: see :func:`job_dir_names`, which is
    what every caller should use, because the answer depends on the job SET
    (the deck each job carries) rather than on a name alone.
    """
    return f"bench-{job_name}"


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
    return f"{bench_container(shape, stage_token)}/{job_dir_name(job_name)}"


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
    from .shape import Shape
    desc = Path(base_dir) / FILENAME
    if not desc.is_file():
        return None
    return Shape.named(read_task(desc).shape)


def bench_container(shape: "Shape", token: str = "") -> str:
    """Where a stage's bench state lives — its trials, the sweep's own
    ``job-set.json``, its verdict — relative to the bundle root.

    ``<NN>_<stage>/bench`` in the hierarchy; ``bench_<NN>_<stage>`` at the
    root of a FLAT calculation; bare ``bench`` for a stageless one.
    `job-contracts.md` § 6.3: *"benchmark | bench/ inside the stage it
    measures"* — in flat there IS no stage directory to sit inside, so the
    token qualifies the container's own name instead (2026-08-12 plan A5:
    unqualified, two flat stages' benchmarks shared one root ``bench/``
    and each prep overwrote the other's job-set, plan and verdict).  The
    underscore join keeps it apart from a TRIAL's dash-joined
    ``bench-<point>``, which lives INSIDE a container.

    **This is the ONE spelling of that rule.**  `prep` lays the sweep's
    record down with it and :func:`job_dir_names` places the trials with
    it — which is what makes "record here, trials there" impossible to
    reintroduce on one side only.  Until 2026-08-13 the rule lived twice
    (here as an inline ``if``, in `prep` as ``_bench_container``) and the
    two disagreed in BOTH non-hierarchical layouts: flat trials fell into
    an unqualified shared ``bench/`` while the record sat in
    ``bench_<NN>_<stage>/``, and a stageless sweep's trials sat at the
    ROOT while its record sat in ``bench/`` — so `launch` launched trials
    in directories the underway-ask never looked at (final review A-1/A-2).
    """
    sd = shape.stage_dir(token) if token else "."
    if sd == ".":
        return f"bench_{token}" if token else "bench"
    return f"{sd}/bench"


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
    (:class:`~molbuilder.jobset.shape.Shape`).  A described trial nests
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
    from .shape import Shape
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
                           else job_dir_name(j.name))
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
    """Create each job's directory under ``base_dir`` with its symlinks.

    Returns the list of created job directories (in JobSet order).  Idempotent:
    re-running refreshes the symlinks without duplicating anything.  Raises
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


def attempts(stage_dir: Path) -> List[int]:
    """The attempt numbers present under ``stage_dir``, ascending."""
    if not Path(stage_dir).is_dir():
        return []
    out = []
    for d in Path(stage_dir).iterdir():
        m = ATTEMPT_RE.match(d.name)
        if m and d.is_dir():
            out.append(int(m.group(1)))
    return sorted(out)


def was_launched(attempt_dir: Path) -> bool:
    """Whether ``launch`` has launched this attempt — i.e. ``run.json`` exists.

    This is the whole reason that file exists. Without it, preparing a stage
    twice could rewrite the setup underneath a job already sitting in a queue,
    because a queued job has written nothing and looks exactly like one that
    was never started (§ 1.6).
    """
    return (Path(attempt_dir) / RUN_LAUNCH_FILE).is_file()


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
    ns = attempts(stage_dir)
    return (Path(stage_dir) / f"run-{ns[-1]}") if ns else None


def resolve_attempt(stage_dir: Path) -> Tuple[Path, bool]:
    """The attempt directory to prepare into, and whether it is a fresh one.

    § 1.6: *"Preparing again is safe until the run has been launched.
    Otherwise splitting the two steps leaks directories — prepare, change your
    mind, prepare again, and an empty ``run-3`` sits there forever."*

    So the last attempt is REUSED when it has not been launched, and a new one
    is opened only when the last has. That also makes the numbering mean
    something: every ``run-<n>`` on disk was actually started.
    """
    existing = attempts(stage_dir)
    if existing:
        last = Path(stage_dir) / f"run-{existing[-1]}"
        if not was_launched(last):
            return last, False
        return Path(stage_dir) / f"run-{existing[-1] + 1}", True
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
    linked:         List[str]
    copied:         List[str]
    continued_from: Optional[str]
    cold:           bool


def prepare_attempt(jobset: JobSet, base_dir, stage_name: str, *,
                    continue_from: Optional[str] = None,
                    cold: bool = False,
                    carry: Optional[List[str]] = None) -> "Attempt":
    """Set ONE stage up to run, and report what was done.

    The five steps § 1.6 names: **resolve** the next ``run-<n>``, **create**
    it, **link** the deck / monitor / shared package in, **copy** whatever this
    run continues from, and **report** — the report being the point, since
    preparing is still design and the split from starting is what gives you
    somewhere to look before committing cluster time.

    ``continue_from`` is a bundle-relative attempt directory —
    ``"01_coarse/run-0"``. **Which run you continue from is something you say,
    not something molbuilder guesses** (§ 1.6): continuing from ``run-0`` and
    from ``run-2`` are different scientific choices.  *(One amendment,
    user 2026-08-21: the submission door calls this with the SAME stage's
    LATEST attempt by default when re-submitting a launched stage — the
    one source that is never a guess; every other source stays yours to
    name.)* ``cold=True`` means start
    clean, and with a directory per attempt that is simply *skip the copy* —
    there is nothing to move aside, because a fresh attempt is empty unless
    something is put in it.

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
        raise ValueError(
            "this calculation's shape is 'flat', which has no attempt "
            "directories to open: attempts are told apart by the wrapper's "
            "output index (<label>_<NN>_<name>-run<N>.out), the warm files "
            "are one shared set, and continuing is free -- the next stage "
            "finds them lying there (project-layout.md § 1).  Prepare the "
            "wrappers and submit; there is nothing to name with --from.")
    dir_of = job_dir_names(jobset, sh)
    refs = stage_refs(jobset)
    stage_name = resolve_stage_ref([refs[j.name] for j in jobset.jobs],
                                   stage_name).name
    job = next(j for j in jobset.jobs if j.name == stage_name)

    stage_dir = base / dir_of[stage_name]
    stage_dir.mkdir(parents=True, exist_ok=True)
    attempt, is_new = resolve_attempt(stage_dir)
    attempt.mkdir(parents=True, exist_ok=True)

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
    linked: List[str] = []

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
                linked.append(bn)
                return

    for fname in [job.script] + list(jobset.shared):
        _bring(fname)
    # mb_monitor.py: the load monitor.  makov_payne_correction.py: the
    # post-run script a CHARGED deck's own header instructs the user to
    # run "after SIESTA finishes" -- i.e. HERE, beside the .out.
    for extra in ("mb_monitor.py", "makov_payne_correction.py"):
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
    marker = attempt / ".continued-from"
    if not is_new and marker.is_file():
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
        if not src.is_dir():
            raise ValueError(
                f"--from {continue_from!r}: no such attempt under "
                f"{base}. Name an attempt directory that has already run, "
                f"e.g. '01_coarse/run-0'.")
        # The pair, resolved here and nowhere else -- `--from` is what names
        # the source, so this is the first moment both stages are known.
        names = (carry if carry is not None
                 else warm_carry(job, _source_job(jobset, dir_of,
                                                  continue_from)))
        if not names:
            raise ValueError(
                f"--from {continue_from!r}: {stage_name!r} declares no "
                f"warm-restart files, so there is nothing to continue.\n"
                f"  A stage whose description says `restart: clean` carries "
                f"none of the group -- its deck omits MD.UseSaveXV / "
                f"DM.UseSaveDM / MD.UseSaveCG, so files copied in would sit "
                f"there unread (run-identity.md § 4, *present but not "
                f"honoured*).  Set this stage's `restart` to `continue` in "
                f"task.json and produce again, or drop --from.")
        for name in names:
            f = src / name
            if f.is_file():
                shutil.copy2(f, attempt / name)
                copied.append(name)
        if not copied:
            raise ValueError(
                f"--from {continue_from!r}: that attempt holds none "
                f"of the files this stage would continue from "
                f"({', '.join(names)}). Did it run?")

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
        linked=linked,
        copied=copied,
        continued_from=(None if (cold or not continue_from)
                        else str(continue_from)),
        cold=bool(cold),
    )


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


def write_run_launch(attempt_dir: Path, *, mode: str, command: List[str],
                     job_id: Optional[str] = None,
                     continued_from: Optional[str] = None,
                     launched_at: Optional[str] = None,
                     placed_on: Optional[dict] = None) -> Path:
    """Record a launch into the attempt — ``molbuilder/run-launch@1``.

    Written **after** the launch succeeds, so a failed launch leaves the
    attempt exactly as prepare left it and is still safe to prepare again
    (§ 1.6). ``continued_from`` is the run's provenance — *this geometry came
    from ``01_coarse/run-0``* — which is worth recording whether or not
    anything reads it back.

    ``placed_on`` is WHERE IT RAN: the domain name, its partition and qos, and
    its ``node_type`` (2026-08-23, `execution/submission.md` § 5). The
    placement was already in this file, buried inside the ``sbatch`` argv as
    ``-p``/``-q`` — so reading it back meant parsing a command line, which is
    the re-derivation A4 exists to remove. Naming it is what lets a later
    reader ask *was this measured on the kind of node the run will use?*
    without inventing an answer.

    Absent when there was no placement to record — a direct run, or a machine
    with no queue at all. **Absent means the question cannot be answered**,
    which a reader must not mistake for *yes*.
    """
    from datetime import datetime, timezone
    p = Path(attempt_dir) / RUN_LAUNCH_FILE
    body = {
        "schema": RUN_LAUNCH_SCHEMA,
        "mode": mode,
        "command": list(command),
        "job_id": job_id,
        "launched_at": launched_at or datetime.now(timezone.utc)
                                              .strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    # ABSENT, not null, when this run started from the structure.
    # ``checkpointing.md`` S3 words its test that way -- *"names a directory
    # that exists or is absent"* -- and the two are different to a reader that
    # tests for the key rather than for its truthiness.
    if continued_from:
        body["continued_from"] = str(continued_from)
    # Same absent-not-null rule: a direct run has no placement, and a reader
    # testing for the key learns that rather than reading a null as "nowhere".
    if placed_on:
        body["placed_on"] = dict(placed_on)
    p.write_text(json.dumps(body, indent=2) + "\n", encoding="utf-8")
    return p


__all__ = ["Attempt", "trial_dir", "trial_work_dir",
           "materialize", "job_dir_name", "job_dir_names", "stage_refs",
           "attempts", "was_launched", "latest_attempt", "resolve_attempt",
           "prepare_attempt",
           "write_run_launch", "RUN_LAUNCH_SCHEMA", "RUN_LAUNCH_FILE"]
