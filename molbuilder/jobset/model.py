"""``job-set@1`` data model — the declarative description of a set of
related jobs that share a package (docs/execution/job-system.md).

Pure dataclasses + JSON (de)serialization + structural validation.  NO
filesystem, NO scheduler, NO engine knowledge — those live in the
materialize / submit engines and the producers respectively.  Keeping
this layer pure is what lets the bench sweep and the SIESTA stage ladder
share one execution core without either knowing about the other.

Shared information is modeled in exactly two sanctioned channels:
  * ``JobSet.shared``  — static package files, identical for every job
    (pseudopotentials, geometry, monitor); symlinked into each job dir.
  * ``WarmFile``       — *what this job would take from a run it continues*,
    with the source left OPEN, because `--from` names it at prep
    (`project-layout.md` § 1.6).
Nothing reaches across jobs outside these two.

There used to be a third, ``Carry``: *take file X from job Y*.  It named the
producer, so it was only expressible once the producer was known at produce
time — which is a **chain**.  Deleted 2026-08-10 (user) along with
``depends_on``, ``dep_kind`` and the wrapper's ``carry_deref``: **a JobSet has
no edges**.  Whether a later stage should pick up an earlier one cannot be
settled without reviewing the earlier one's result, so no field is allowed to
settle it (`job-system.md` § 2, decision 6).
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

# Matches the molbuilder/<name>@<major> convention used by
# scheduler/record.py and bench/result.py (the same check_schema gate --
# name AND major since U9).
SCHEMA = "molbuilder/job-set@1"

#: The file a JobSet is written to.  **A11: one spelling per name molbuilder
#: writes** -- this was a bare literal in `prep` (three times), in `_cli` (as a
#: private `_JOBSET_FILE` nobody could import) and in `checkpoint`'s bundle
#: descriptors, so the name of the file this module defines was spelled
#: everywhere except here.  The pattern is `task.FILENAME` and
#: `environment.FILENAME`: the module that owns the format owns its name.
FILENAME = "job-set.json"

_KINDS = ("sweep", "ladder")


@dataclass
class Resources:
    """A per-job scheduler ask.  Every field is optional; ``None`` means
    "inherit the job-level default / per-job estimate" (assistant, not
    nanny — no surprise resource choices).

    Field names match the EXCHANGE vocabulary used by the other persisted
    artifacts (bench-manifest, scheduler config) so the system speaks one
    language on files: ``mpi_np`` / ``cpus_per_task`` / ``time`` / ``mem`` /
    ``exclusive`` (NOT ``omp`` / ``walltime``).  ``domain`` is a
    probed domain name (`configuration.md` § 5 -- it was
    ``scheduler.routing`` until 2026-08-17) the submit engine
    resolves to ``-p``/``-q``; ``gres`` is a raw SLURM gres string (e.g.
    ``"gpu:a100:1"``) or None.

    **One field here becomes no scheduler flag at all, and that is not an
    oversight.**  ``continue_retries`` is the warm-retry budget baked into
    the run wrapper at install time (running-a-job.md § 3.5) — it never
    reaches an ``sbatch`` line.  It rides this class because this is the
    road every *"field the deck never carries"* already rides
    (engines/stages.md § 5, the row that groups it with ``mpi_np`` and
    ``omp_threads``); the alternative was a second, hand-maintained road
    from a job to its wrapper.  Decided 2026-08-07, and written here as
    well as in job-contracts.md § 6.2 because a field sitting in a class
    called *a per-job scheduler ask* is otherwise an invitation to render
    it into a directive.  **Do not emit it as one.**
    """
    domain:        Optional[str]   = None
    time:          Optional[str]   = None    # SLURM -t (D-HH:MM:SS); == scheduler defaults.time
    exclusive:     Optional[bool]  = None
    mem:           Optional[str]   = None    # SLURM --mem (e.g. "120G", "0")
    gres:          Optional[str]   = None    # SLURM --gres (e.g. "gpu:a100:1")
    #: Whether this run uses a GPU -- the ANSWER, carried rather than
    #: re-derived.  `read_by = ["wrapper"]` on the catalogue item says the
    #: wrapper depends on it (`engines/template.md` § 6.1), and until
    #: 2026-08-23 the wrapper satisfied that by grepping the rendered deck
    #: for `Diag.ELPA.GPU` at four sites -- a layer re-deriving what another
    #: layer already held, and doing it by matching a SIESTA keyword, so a
    #: PySCF GPU run could not route at all.  ``None`` means the caller did
    #: not say, and the wrapper falls back to reading the artifact it was
    #: handed -- which is a different thing from re-deriving: a wrapper
    #: written for a deck someone points at has nothing else to ask.
    use_gpu:       Optional[bool]  = None
    mpi_np:        Optional[int]   = None    # SLURM -n (MPI ranks)
    cpus_per_task: Optional[int]   = None    # SLURM -c (OMP cores/rank); == SiestaConfig.omp_threads
    continue_retries: Optional[int] = None   # NOT a SLURM flag -- baked into the wrapper
    max_memory_mb:    Optional[int] = None   # NOT a SLURM flag either -- `ulimit -v`
    #  ^ added 2026-08-11.  It is a MACHINE fact (how much memory one rank may
    #  take on this node), so it belongs to the allocation -- and until it lived
    #  here it reached the wrapper from `cli.py` and `web/blueprints/build.py`
    #  but NOT from `jobset/prep.py`, so a staged run silently dropped a cap the
    #  user had set.  Three call sites building one wrapper from loose keyword
    #  arguments, and one forgot a field; carried on the allocation, it cannot
    #  be forgotten by one of them (generator.md § 5).

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: Optional[Dict[str, Any]]) -> "Resources":
        d = d or {}
        known = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class WarmFile:
    """One file this job continues **from**, and what makes it safe to take.

    **Not an edge.**  The deleted ``Carry`` said *take X from job Y* and was
    only answerable once you had decided who Y is.  Stages do not chain
    (`project-layout.md` § 1.6), so at produce time nobody knows: the run a
    stage continues from is named at `prep`, by a person who has just looked
    at it, and it may be any finished attempt -- ``02_medium/run-1``,
    ``01_coarse/run-0``, or an earlier attempt of this same stage (§ 2.3.4's
    *"a redo is the same instruction"*).  So what a job can state in advance is
    **its own half**: which files it would take, and the condition on each.

    ``requires_same`` names a key both jobs must agree on for this file to
    mean anything -- looked up in :attr:`Job.traits`, compared as opaque
    strings.  ``.CG`` is the whole reason it exists: *"a CG state is
    meaningless to a Broyden stage, so blindly carrying it would corrupt the
    restart"* (`job-system.md` § 4.1).  ``None`` is unconditional.

    **Why the framework never spells the rule.**  `run-identity.md` § 4 rule 1
    puts the warm group in *the engine's* contract -- *"a new engine that
    cannot fill this in is a new engine whose restart behaviour nobody has
    thought about yet"* -- and rule 4 records what the alternative cost:
    *"the set used to be three suffixes written into the producer, which meant
    a TranSIESTA ladder could not express its `.TSHS` dependency without
    changing molbuilder's code."*  This class is the declaration those rules
    ask for; `jobset` compares two strings and knows nothing else.
    """
    name:          str
    requires_same: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"name": self.name}
        # ABSENT, not null, when the file is unconditional -- the same reading
        # `checkpointing.md` S3 asks for elsewhere: a key that is missing and a
        # key that is null are different claims to anything testing for it.
        if self.requires_same:
            d["requires_same"] = self.requires_same
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "WarmFile":
        return cls(name=d["name"], requires_same=d.get("requires_same"))


@dataclass
class Job:
    """One unit of work.  ``name`` is unique within the set and becomes the
    job directory (``bench-<name>/``) and the SLURM ``-J`` name.  ``script``
    is the per-job input filename (e.g. the rendered ``.fdf``).

    **A Job names no other Job.**  There is no ``depends_on``, no ``dep_kind``
    and no ``carry`` -- all three were deleted 2026-08-10 (user).  What a job
    declares instead is ``warm``: *what would this job take from a run it
    continues, whichever run that turns out to be* -- which is what `prep`
    reads, because `--from` names the source and a produce-time edge cannot
    (see :class:`WarmFile`).  ``traits`` are the opaque per-job values a
    ``WarmFile.requires_same`` is compared against; SIESTA puts its optimizer
    there.
    """
    name:       str
    script:     str
    resources:  Resources       = field(default_factory=Resources)
    warm:       List[WarmFile]  = field(default_factory=list)
    traits:     Dict[str, str]  = field(default_factory=dict)
    #: A TRIAL's sweep coordinate, as data (`job-contracts.md` § 6.3: the
    #: name is an identifier, never a parser target -- what varied travels
    #: here).  ``{}`` for a rung or a whole calculation.  `summarize` reads
    #: it to table value coordinates and to carry the winner's value pins
    #: into ``run-config.toml`` (`generator.md` § 4.3a).
    point:      Dict[str, Any]  = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "name": self.name,
            "script": self.script,
            "resources": self.resources.to_dict(),
            "warm": [w.to_dict() for w in self.warm],
            "traits": dict(self.traits),
        }
        # ABSENT, not empty, when this is no trial -- same reading as
        # WarmFile.requires_same above.
        if self.point:
            d["point"] = dict(self.point)
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Job":
        return cls(
            name=d["name"],
            script=d["script"],
            resources=Resources.from_dict(d.get("resources")),
            warm=[WarmFile.from_dict(w) for w in (d.get("warm") or [])],
            traits=dict(d.get("traits") or {}),
            point=dict(d.get("point") or {}),
        )


#: The trait a warm file may be conditioned on, spelled ONCE.
#:
#: It is a string in three places that must agree -- each engine's ``_traits``
#: producer and every ``requires_same`` row of a ``warm-files.toml`` -- and a
#: typo on any of them reads as *"the optimizers disagree"*, which withholds
#: the file silently rather than failing.  The rules files are data and say it
#: for themselves; this is the one Python spelling, and it lives beside
#: :func:`warm_carry` because that is the only place the comparison happens.
#:
#: Declared twice until 2026-08-18 -- once per engine, each with a comment
#: explaining that it had to match the other.  A comment is not a mechanism.
OPTIMIZER_TRAIT = "optimizer"


def warm_carry(job: "Job", source: Optional["Job"]) -> List[str]:
    """What ``job`` takes from ``source`` — the pair's answer, not an edge's.

    `project-layout.md` § 2.3.4 states the rule as three rows, and only the
    third needs two stages:

    | `.XV` | the relaxed coordinates | **always** |
    | `.DM` | the converged density | when the description says to reuse it |
    | `.CG` | the optimiser's own history | **only if both stages use the same algorithm** |

    A producer states rows one and two on the job itself — they are properties
    of the destination alone — and row three as a :class:`WarmFile` condition.
    **This function is the only place the third is evaluated**, because it is
    the only place both stages are known: `--from` names the source at `prep`,
    and it need not be the ladder's next-door neighbour — continuing `tight`
    from `01_coarse` skips `medium` entirely, and comparing `tight` against
    `medium` would then answer a question nobody asked.

    That was the shape until 2026-08-10: the set came off ``Job.carry``, whose
    ``from_job`` is the **immediate predecessor**, fixed at produce time. So a
    ladder that skipped a rung carried `.CG` on the strength of a comparison
    with a stage that never ran — § 9's diagnosis in its second form, *a caller
    reads a field computed for a different question.*

    ``source is None`` (an attempt this JobSet cannot place — a hand-made path,
    or one naming a stage that is not here) drops **every** conditional file.
    Unverified is not the same as satisfied, and the direction of the mistake
    is not symmetric: a `.CG` wrongly withheld costs some optimizer steps; a
    `.CG` wrongly carried *"would corrupt the restart"* (`job-system.md` § 4.1)
    and the run still reports success.

    It lives here, beside the declaration it reads, rather than in the engine
    that lays the files down: it touches no filesystem and names no suffix,
    which is exactly the line this module holds.
    """
    out: List[str] = []
    for w in job.warm:
        if w.requires_same:
            if source is None:
                continue
            mine = job.traits.get(w.requires_same)
            theirs = source.traits.get(w.requires_same)
            # `mine is None` cannot happen through `validate()`, which refuses
            # a condition on a trait the job lacks; the guard is here because
            # `None == None` would otherwise read as agreement between two
            # jobs that have each said nothing.
            if mine is None or mine != theirs:
                continue
        out.append(w.name)
    return out


@dataclass
class JobSet:
    """A set of related jobs sharing a static package.  ``kind`` is
    ``"sweep"`` (the benchmark's independent points) or ``"ladder"`` (the
    SIESTA stage relaxation).  **Both are sets of independent jobs**; the kind
    says how the directories are named and whether a PERSON should take them in
    order, never whether one job waits for another -- neither does.  ``shared``
    are package files symlinked into every job directory."""
    name:   str
    engine: str
    kind:   str
    shared: List[str]   = field(default_factory=list)
    jobs:   List[Job]   = field(default_factory=list)

    # ----- persistence (job-set@1) ----------------------------------- #

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema": SCHEMA,
            "name": self.name,
            "engine": self.engine,
            "kind": self.kind,
            "shared": list(self.shared),
            "jobs": [j.to_dict() for j in self.jobs],
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "JobSet":
        # Major-version check via the shared helper (persist.py), same rule
        # scheduler/record.py + bench/result.py use.
        from ..persist import check_schema
        check_schema(str(d.get("schema") or ""), SCHEMA, label="job-set")
        return cls(
            name=d["name"],
            engine=d["engine"],
            kind=d["kind"],
            shared=list(d.get("shared") or []),
            jobs=[Job.from_dict(j) for j in (d.get("jobs") or [])],
        )

    def write(self, path) -> Path:
        """Persist to ``job-set.json`` -- the bundle's plan, carried
        host->target (the exchange vocabulary, job-contracts § 6.2; its
    original home data-vocabulary.md is archived).  Pretty JSON so a human can
        read/diff the plan in the bundle."""
        from ..persist import write_json
        return write_json(path, self.to_dict())

    @classmethod
    def load(cls, path) -> "JobSet":
        """Read a ``job-set.json`` back into a JobSet (schema checked by name
        and major -- persist.check_schema
        via ``from_dict``)."""
        from ..persist import read_json
        return cls.from_dict(read_json(path))

    # ----- structural validation ------------------------------------- #

    def validate(self) -> List[str]:
        """Return human-readable structural errors (empty == OK).  Checks
        the invariants the engines can't recover from -- exactly the same
        discipline the description's own stage validation applies.  (It used
        to name a per-engine stage validator alongside it; those went with the
        engine-config stage lists, and the ladder-level checks they made live
        in ``task.py``, where a description's ladder is read and refused.)

          * non-empty; ``kind`` known;
          * unique job names (the dir + ``-J`` collide otherwise);
          * every ``warm.requires_same`` names a trait this job HAS.

        The acyclic/ordered checks went with the edges on 2026-08-10: with no
        job naming another, there is no graph left to be cyclic.
        """
        errors: List[str] = []
        if self.kind not in _KINDS:
            errors.append(f"kind = {self.kind!r}: must be one of {_KINDS}")
        if not self.jobs:
            errors.append("jobs: empty; a JobSet needs at least one job")
            return errors
        seen: set = set()
        for i, j in enumerate(self.jobs):
            prefix = f"jobs[{i}]({j.name})"
            if j.name in seen:
                errors.append(
                    f"{prefix}.name: duplicate; job dirs / -J names collide")
            for w in j.warm:
                # A condition on a trait this job does not declare can never be
                # satisfied, so the file would simply never arrive -- and the
                # WRONG kind of silence, because "fail safe" here means the
                # stage starts cold while everything reports success.  The
                # comparison is fail-safe on purpose (:func:`warm_carry`); the
                # DECLARATION being unsatisfiable is a producer bug, and the
                # producer is who this message is for.
                if w.requires_same and w.requires_same not in j.traits:
                    errors.append(
                        f"{prefix}.warm {w.name!r} requires the same "
                        f"{w.requires_same!r}, which this job does not "
                        f"declare in traits ({sorted(j.traits) or 'none'}): "
                        f"the condition could never be met, so the file would "
                        f"never be carried")
            seen.add(j.name)
        return errors


__all__ = ["Resources", "WarmFile", "Job", "JobSet", "SCHEMA"]
