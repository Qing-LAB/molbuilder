"""Materialize engine — turn a :class:`JobSet` into on-disk per-job
directories (docs/execution/job-system.md).

Filesystem ONLY: it knows nothing about schedulers or engines.  For each
job it creates ``point-<name>/`` and lays relative symlinks for

  * the static ``shared`` package + the job's own ``script`` (identical
    bytes for the job — pseudos, geometry, monitor, the per-job input),
  * each ``carry`` file from the producing job's directory.

This is the generalization of the benchmark's ``_mb_point`` helper.

Carry symlinks point at concrete filenames in the producer's dir (e.g.
``../point-stage1/job.XV``).  At materialize time (prep) the producer has
not run yet, so these are intentionally **dangling** symlinks — they
resolve once the producer writes the file, and the submit engine's
dependency ordering guarantees the consumer starts only after that.
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
    """The on-disk directory for a **sweep** job — the benchmark's
    ``point-<...>`` convention.

    A ladder's stage directory is NOT this: see :func:`job_dir_names`, which is
    what every caller should use, because the answer depends on the job SET
    (its kind, and the deck each job carries) rather than on a name alone.
    """
    return f"point-{job_name}"


def shape_of(jobset: JobSet, base_dir) -> Optional["Shape"]:
    """The layout this bundle uses, read from its description.

    **The one place a surface asks.** `engines/stages.md` § 6.7 puts the shape
    in `task.json` and says *"`prep` **reads** it; it does not decide it"* —
    so this reads it, and every layer below takes the answer as an argument
    rather than going looking for it a second time.

    ``None`` where the question does not arise: a **sweep** is laid out
    ``point-<name>`` in either shape, which is why a benchmark bundle carries
    no description and needs none. ``None`` also when a ladder has no
    ``task.json`` — bundles produced before 2026-08-10, and the hand-built
    JobSets in the tests — and :func:`job_dir_names` reads that as the
    hierarchy, which is what they all are. That fallback is transitional and
    dies with the last such bundle; it is **not** an inference from data, which
    § 6.7 forbids, but the absence of a file that is now always written.
    """
    if jobset.kind != "ladder":
        return None
    from ..task import FILENAME, read_task
    from .shape import Shape
    desc = Path(base_dir) / FILENAME
    if not desc.is_file():
        return None
    return Shape.named(read_task(desc).shape)


def job_dir_names(jobset: JobSet, shape: "Shape" = None) -> Dict[str, str]:
    """``{job name: directory name}`` for a whole JobSet — the naming authority.

    Two kinds, two conventions, and `project-layout.md` § 4.1 is explicit about
    which is which:

    | kind | directory |
    |---|---|
    | ``sweep`` (the benchmark) | ``point-<name>`` |
    | ``ladder`` (a stage ladder) | ``<seq>_<name>`` — ``01_coarse``, ``02_tight`` |

    Until 2026-08-10 every kind got ``point-<name>``, so a staged run's
    directories came out ``point-coarse/`` (`worked-example.md` gap 6). The
    contract's fix is *"branch on ``JobSet.kind``"* — and that branch lives in
    :func:`stage_refs`, once, rather than here as well.

    **The seq is read back off the deck, not counted here.** ``job.script`` is
    ``<label>_<NN>_<name>.fdf`` (decision 27), so the token the directory is
    named for is the one the deck already carries — which is what makes
    ``<NN>_<name>/<label>_<NN>_<name>.fdf`` a self-check rather than a
    repetition (§ 4.1). Counting positions here instead would reintroduce
    exactly what `engines/stages.md` R5 forbids: a number that shifts when the
    ladder changes, silently handing one stage's directory to another.

    A ladder job whose script carries no token falls back to ``point-<name>``.
    That is not a staged-producer JobSet — hand-written, or older than decision
    27 — and inventing a seq for it would be guessing at the one number that
    must never be guessed.

    **The two conventions meet in one expression**, because :func:`stage_refs`
    already answered which applies: a job with an ordinal has a token and is
    named by it; a job without one has no token and falls back. Branching on
    ``jobset.kind`` a second time here is how the directory and the deck get to
    disagree about what a job is.

    ``shape`` decides where a **ladder's** stages sit: hierarchical gives each
    one a directory, flat is depth 1 and they all sit in the bundle root
    (:class:`~molbuilder.jobset.shape.Shape`). It is **not consulted for a
    sweep** — ``point-<name>`` is the benchmark's own convention and is the
    same in either layout, which is why a bench bundle needs no description to
    be laid out.

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
    return {j.name: (sh.stage_dir(refs[j.name].token)
                     if refs[j.name].token else job_dir_name(j.name))
            for j in jobset.jobs}


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
    ways: ``point-<name>`` here, the row number in ``plan``, ``None`` in
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
    ladder = jobset.kind == "ladder"
    out: Dict[str, StageRef] = {}
    for j in jobset.jobs:
        parsed = (parse_stage_token(os.path.basename(j.script), jobset.name)
                  if ladder else None)
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
        # static package + this job's own input script: same bytes, one
        # level up in the bundle root.
        for fname in list(jobset.shared) + [job.script]:
            relink(d, os.path.join("..", fname), os.path.basename(fname))
        # runtime-produced carry-forward from the producing job's dir.
        for c in job.carry:
            target = os.path.join("..", dirs[c.from_job], c.pattern)
            relink(d, target, os.path.basename(c.pattern))
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

    # Inputs: the deck and the shared package, linked UP to the stage dir and
    # the bundle root.  Identical bytes for every attempt, so a link is right
    # here -- it is the warm state below that must be a copy.
    linked: List[str] = []
    for fname in [job.script] + list(jobset.shared):
        relink(attempt, os.path.join("..", "..", fname),
               os.path.basename(fname))
        linked.append(os.path.basename(fname))
    for extra in ("mb_monitor.py",):
        if (base / extra).exists():
            relink(attempt, os.path.join("..", "..", extra), extra)
            linked.append(extra)
    stem = Path(job.script).stem
    for wrapper in (f"{stem}.run.sh", f"{stem}.sbatch"):
        if (base / wrapper).exists():
            relink(attempt, os.path.join("..", "..", wrapper), wrapper)
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
