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

#: Written by ``submit`` into the attempt, AFTER the launch succeeds
#: (``project-layout.md`` § 1.6).  Its presence is the only honest answer to
#: *has this been launched?* -- a queued job has produced nothing yet, so
#: "no output" and "not started" are indistinguishable from the directory alone.
RUN_LAUNCH_SCHEMA = "molbuilder/run-launch@1"
RUN_LAUNCH_FILE = "run.json"


def relink(link_dir: Path, target: str, link_name: str) -> None:
    """Create ``link_dir/link_name`` -> ``target`` (a relative path),
    replacing any existing entry (``ln -sfn`` semantics).  Dangling
    targets are allowed (carry-forward before the producer runs, and the
    rendered wrappers the prep step links in afterwards)."""
    link_path = link_dir / link_name
    if link_path.is_symlink() or link_path.exists():
        link_path.unlink()
    os.symlink(target, link_path)


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


def job_dir_names(jobset: JobSet, shape: "Shape" = None) -> Dict[str, str]:
    """``{job name: directory name}`` for a whole JobSet — the naming authority.

    One question, not two kinds (`generator.md` § 5): *does this job have a
    stage, a point, or both?*

    | the deck says | the set says | directory |
    |---|---|---|
    | a stage token, job named for the stage | — | ``<NN>_<name>`` — the rung itself |
    | a stage token, job named by coordinate | — | ``<NN>_<name>/bench/bench-<point>`` — a trial, in the stage's ``bench/`` container |
    | no token | ``kind="ladder"``, job named AS the set | ``.`` — the bundle root: a STAGELESS calculation (`engines/stages.md` § 6.5) IS its own one rung, and runs where its deck already sits.  In § 6.5's form the job and the set share the calculation's one label (`run-identity.md` § 2) — an identity that survives submit's only-narrowing, where a length test flipped the answer mid-flight (found by this row's first cut, R1) |
    | no token | anything else | ``bench-<name>`` at the root — hand-built sets (sweeps, and tokenless ladders whose jobs are their own names), told apart by name alone |

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
    attempt unreachable, and broke § 6.5's single-parameter-set form
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
            # A trial NESTS inside a ``bench/`` CONTAINER inside the stage
            # it measures -- <NN>_<stage>/bench/bench-<point>/
            # (job-contracts.md § 6.3's Directories table, the cross-layer
            # authority: "benchmark | bench/ inside the stage").  The
            # container is what gives the stage's bench state ONE home --
            # its trials, its own job-set.json, its verdict -- so two
            # stages' benchmarks can never collide.  Until 2026-08-12 the
            # trials sat directly in the stage (generator.md § 5's earlier
            # wording, since amended): the authority row won.
            sd = sh.stage_dir(trial_token)
            container = "bench" if sd == "." else f"{sd}/bench"
            out[j.name] = f"{container}/{job_dir_name(j.name)}"
            continue
        # Tokenless: the deck says nothing, so the SET is the only data
        # left (see the docstring's R1 paragraph).  A stageless ladder IS
        # its own one rung and runs at the root; a hand-built sweep's
        # points are siblings told apart by name.
        out[j.name] = ("." if jobset.kind == "ladder"
                       and j.name == jobset.name
                       else job_dir_name(j.name))
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
    dirs = job_dir_names(jobset, shape_of(jobset, base_dir))
    for job in jobset.jobs:
        d = base / dirs[job.name]
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
        # static package + this job's own input script: same bytes, in the
        # bundle root.  The prefix is COMPUTED, not "..": a nested trial dir
        # (<NN>_<stage>/bench-<point>/) is depth 2, and a hardcoded one-level
        # hop would dangle -- the same destruction class the flat guard
        # above records (M5, 2026-08-10).
        up = os.path.relpath(str(base), str(d))
        for fname in list(jobset.shared) + [job.script]:
            relink(d, os.path.join(up, fname), os.path.basename(fname))
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
    """Whether ``submit`` has launched this attempt — i.e. ``run.json`` exists.

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
    from ``run-2`` are different scientific choices. ``cold=True`` means start
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

    # Inputs: the deck and the shared package, linked UP to the bundle root.
    # Identical bytes for every attempt, so a link is right here -- it is the
    # warm state below that must be a copy.  The prefix is COMPUTED, not
    # "../..": a stageless calculation's stage dir IS the bundle root (its
    # attempt sits at depth 1) and a bench trial's attempt sits at depth 3,
    # so a hardcoded two-level hop dangles at both -- the same destruction
    # class materialize() records above (M5, 2026-08-10).
    up = os.path.relpath(str(base), str(attempt))
    linked: List[str] = []
    for fname in [job.script] + list(jobset.shared):
        relink(attempt, os.path.join(up, fname),
               os.path.basename(fname))
        linked.append(os.path.basename(fname))
    # mb_monitor.py: the load monitor.  makov_payne_correction.py: the
    # post-run script a CHARGED deck's own header instructs the user to
    # run "after SIESTA finishes" -- i.e. HERE, beside the .out, so the
    # link is what keeps that instruction honest in the hierarchical
    # shape (E6 integration, 2026-08-12; prep writes it at the root).
    for extra in ("mb_monitor.py", "makov_payne_correction.py"):
        if (base / extra).exists():
            relink(attempt, os.path.join(up, extra), extra)
            linked.append(extra)
    stem = Path(job.script).stem
    for wrapper in (f"{stem}.run.sh", f"{stem}.sbatch"):
        if (base / wrapper).exists():
            relink(attempt, os.path.join(up, wrapper), wrapper)
            linked.append(wrapper)

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

    # Leave the provenance where ``submit`` can find it: prep is what knows
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
    (`job-system.md` § 5.3), so the stage is its head component read back
    through the SAME naming authority that wrote it. Nothing is parsed out of
    the name: :func:`job_dir_names` is asked, and a head that matches no job
    simply has no answer — which :func:`warm_carry` then treats as *unverified*
    rather than guessing.
    """
    if not continue_from:
        return None
    head = Path(continue_from).parts[0] if Path(continue_from).parts else ""
    for j in jobset.jobs:
        if dir_of.get(j.name) == head:
            return j
    return None


def write_run_launch(attempt_dir: Path, *, mode: str, command: List[str],
                     job_id: Optional[str] = None,
                     continued_from: Optional[str] = None,
                     launched_at: Optional[str] = None) -> Path:
    """Record a launch into the attempt — ``molbuilder/run-launch@1``.

    Written **after** the launch succeeds, so a failed submission leaves the
    attempt exactly as prepare left it and is still safe to prepare again
    (§ 1.6). The last field is the run's provenance — *this geometry came from
    ``01_coarse/run-0``* — which is worth recording whether or not anything
    reads it back.
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
    p.write_text(json.dumps(body, indent=2) + "\n", encoding="utf-8")
    return p


__all__ = ["Attempt",
           "materialize", "job_dir_name", "job_dir_names", "stage_refs",
           "relink",
           "attempts", "was_launched", "latest_attempt", "resolve_attempt",
           "prepare_attempt",
           "write_run_launch", "RUN_LAUNCH_SCHEMA", "RUN_LAUNCH_FILE"]
