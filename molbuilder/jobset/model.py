"""``job-set@1`` data model — the declarative description of a set of
related jobs that share a package (docs/execution/job-system.md).

Pure dataclasses + JSON (de)serialization + structural validation.  NO
filesystem, NO scheduler, NO engine knowledge — those live in the
materialize / submit engines and the producers respectively.  Keeping
this layer pure is what lets the bench sweep and the SIESTA stage ladder
share one execution core without either knowing about the other.

Shared information is modeled in exactly two sanctioned channels:
  * ``JobSet.shared``  — static package files, identical for every job
    (pseudopotentials, geometry); copied into each job dir as real files.
  * ``WarmFile``       — *what this job would take from a run it continues*,
    with the source left OPEN, because `--from` names it at prep
    (`project-layout.md` § 1.6).
Nothing reaches across jobs outside these two.

**A JobSet has no edges**: whether a later stage should pick up an earlier
one cannot be settled without reviewing the earlier one's result, so no field
is allowed to settle it (`job-system.md` § 2, decision 6).
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Matches the molbuilder/<name>@<major> convention used by
# scheduler/record.py and bench/result.py (the same check_schema gate --
# name AND major).
SCHEMA = "molbuilder/job-set@1"

#: The file a JobSet is written to.  **A11: one spelling per name molbuilder
#: writes** -- one home, the catalogue's (`runfiles.WRITTEN`,
#: `job-contracts.md` § 2.2: every file molbuilder writes is a row there, and
#: a fixed name's owner takes it from the row).
from ..runfiles import JOBSET_FILE as FILENAME  # noqa: E402

#: The two kinds, each a name with ONE home.  `KIND_SWEEP` is exported
#: because a reader outside this module has to ask *is this a benchmark
#: sweep* -- `parse.sidecars.job_set` does, to decide whether the Results
#: tab may offer the file.  A second spelling of `"sweep"` out there is a
#: second opinion about what a sweep is.
KIND_SWEEP = "sweep"
KIND_LADDER = "ladder"

_KINDS = (KIND_SWEEP, KIND_LADDER)


#: An engine config's spelling of a `Resources` field -- the exchange
#: vocabulary (`job-contracts.md` § 6.2): PySCF's ``threads`` is SIESTA's
#: ``omp_threads``, the cores one process runs on; ``gpu_count`` is the count
#: in ``gres``; a name missing here is the field's own.  Read by the run
#: card's assembly, the placement's check and `resolve`'s hand-over to the
#: deck writer -- one map, the three of them.
AS_RESOURCE = {
    "omp_threads": "cpus_per_task",
    "threads": "cpus_per_task",
    "gpu_count": "gres",
    "max_memory_mb": "max_memory_mb",
}

@dataclass
class Resources:
    """A per-job scheduler ask.  Every field is optional, and ``None``
    means **unstated** — which nothing fills in: prep and launch refuse a
    launch value stated nowhere (`architecture.md` § 5.2; user, 2026-10-02:
    *"explicit job config is the only way allowed"*).

    Field names match the EXCHANGE vocabulary used by the other persisted
    artifacts (bench-manifest, scheduler config) so the system speaks one
    language on files: ``mpi_np`` / ``cpus_per_task`` / ``time`` / ``mem`` /
    ``exclusive`` (NOT ``omp`` / ``walltime``).  ``domain`` is a
    probed domain name (`configuration.md` § 5) the submit engine
    resolves to ``-p``/``-q``; ``gres`` is SLURM's GPU ask, a count
    (``"gpu:1"``; no card -- `scheduler.md` R2a) or None.

    **One field here becomes no scheduler flag at all, and that is not an
    oversight.**  ``continue_retries`` is the warm-retry budget baked into
    the run wrapper at install time (running-a-job.md § 3.5) — it never
    reaches an ``sbatch`` line.  It rides this class because this is the
    road every *"field the deck never carries"* already rides
    (engines/stages.md § 5, the row that groups it with ``mpi_np`` and
    ``omp_threads``); the alternative was a second, hand-maintained road
    from a job to its wrapper.  Written here as
    well as in job-contracts.md § 6.2 because a field sitting in a class
    called *a per-job scheduler ask* is otherwise an invitation to render
    it into a directive.  **Do not emit it as one.**
    """
    domain:        Optional[str]   = None
    time:          Optional[str]   = None    # SLURM -t (D-HH:MM:SS)
    exclusive:     Optional[bool]  = None
    mem:           Optional[str]   = None    # SLURM --mem (e.g. "120G", "0")
    gres:          Optional[str]   = None    # SLURM --gres, a count: "gpu:1"
    #: Whether a GPU ask carries ``--gres-flags=enforce-binding`` -- the
    #: job's cores on the socket its GPUs sit on (`execution/gpu.md` G9).
    #: ``None`` is the rule, and the rule is that it does; ``False`` arrives
    #: from the description's ``allocation.gpu_binding``, which turns it off
    #: for the calculation, its benchmark and its runs alike.
    gpu_binding:   Optional[bool]  = None
    #: Whether this run uses a GPU -- the ANSWER, carried rather than
    #: re-derived.  `read_by = ["wrapper"]` on the catalogue item says the
    #: wrapper depends on it (`engines/template.md` § 6.1).  `resolve` sets it on every job
    #: from the job's own values; :func:`gpu_request` reads it, with the
    #: count, for every reader.
    use_gpu:       Optional[bool] = field(default=None, metadata={"axis": "rider"})
    mpi_np:        Optional[int]   = None    # SLURM -n (MPI ranks)
    cpus_per_task: Optional[int]   = None    # SLURM -c (OMP cores/rank); == SiestaConfig.omp_threads
    # NOT a SLURM flag -- baked into the wrapper.
    continue_retries: Optional[int] = field(default=None, metadata={"axis": "rider"})
    #: WHEN this calculation should say something -- also not a SLURM flag,
    #: and here for the reason the paragraph above gives: this is the road a
    #: field takes from a job to its wrapper, and the alternative is a second
    #: hand-maintained one.  Set from `task.json`'s `notify` block at prep
    #: (`archive/2026-09-01-bench-and-junction-plan.md` § 2.9).
    #:
    #: WHERE to send it is deliberately absent: the destination and its
    #: credential are the user's own file on the machine that runs the job,
    #: because a description travels and a token must not travel with it.
    notify_on_scf:      Optional[bool] = field(default=None, metadata={"axis": "rider"})
    notify_every_hours: Optional[float] = field(default=None, metadata={"axis": "rider"})
    #: WHICH channels, by NAME -- the only part of "where" that may ride
    #: here, and it rides for the same reason the two above do.  A name is a
    #: label the person chose on the machine that runs the job; the address
    #: and the key it resolves to stay in that machine's own file.
    #:
    #: ``None`` is unset: no flag is rendered and the monitor sends
    #: nothing.  ``("*",)`` is every channel that machine has, and an empty
    #: tuple is none at all (`run-reports.md` 3.0).
    notify_channels: Optional[Tuple[str, ...]] = field(default=None, metadata={"axis": "rider"})

    #: WHAT each report carries beyond the name (`stages.md` § 6.9).  `None`
    #: is every field the monitor could determine; `()` is the summary line
    #: alone.  Baked into the monitor's command line at `prep`, like the
    #: cadence and the channel names -- so a running job's format cannot
    #: change under it because `task.json` was edited while it queued.
    notify_report: Optional[Tuple[str, ...]] = None
    #: WHICH binary the wrapper launches; ``None`` = the engine's own
    #: (``siesta`` for a ``.fdf``).  Also not a SLURM flag -- the same
    #: job-to-wrapper road as ``continue_retries``.  Set by the transport
    #: composite's transmission stage (``tbtrans`` post-processes the
    #: device run, archive/2026-09-01-transport-design.md § 4.2): the deck cannot say it,
    #: because the transmission deck IS the device deck -- the same text,
    #: read by a different program.
    program:          Optional[str] = None
    max_memory_mb:    Optional[int] = None   # NOT a SLURM flag either -- `ulimit -v`
    #  ^ a MACHINE fact (how much memory one rank may take on this node), so
    #  it belongs to the allocation: carried on it, no call site building a
    #  wrapper can forget it (generator.md § 5).

    def __post_init__(self) -> None:
        """``time`` and ``mem`` hold the RECORD's spelling, always, and a
        sequence field holds its own type.

        The two fields are documented above as SLURM's own -- ``-t`` takes
        ``D-HH:MM:SS``, ``--mem`` takes ``80G`` -- while a person says
        ``4h`` and ``80GB``.  Normalising HERE rather than at each caller is
        the point: this class is reached by the CLI's ``--time``/``--mem``,
        by ``execution``, by `prep`'s fold of
        ``task.json``'s allocation, and by `from_dict` over a job-set file
        somebody edited -- four roads, and the fix that patched one of them
        would leave three.  A type that enforces its own invariant cannot
        be reached down a road that forgot.

        *This is the whole of the 2026-08-24 failure: `prep` copied the
        browser's ``"4h"`` into this field, `sbatch` was handed ``-t 4h``,
        and SLURM refused the tool's own written value.*
        """
        from ..scheduler.quantities import (canonical_gres, canonical_mem,
                                            canonical_time)
        if self.time:
            self.time = canonical_time(self.time)
        if self.mem:
            self.mem = canonical_mem(self.mem)
        # AND A GPU ASK, which reaches `sbatch --gres` as it stands here: a
        # count, `gpu:N` (`scheduler.md` R2a).
        if self.gres:
            self.gres = canonical_gres(self.gres)
        # A TUPLE OUT, A TUPLE BACK.  `to_dict` is `asdict`, so a job-set
        # file stores the names as a JSON array and `from_dict` hands them
        # back as a LIST -- and a list never equals the tuple it was written
        # from.
        #
        # Normalising HERE for the reason the paragraph above gives -- four
        # roads reach this class, and a fix at one of them leaves three.
        if self.notify_channels is not None:
            self.notify_channels = tuple(self.notify_channels)
        if self.notify_report is not None:
            self.notify_report = tuple(self.notify_report)

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: Optional[Dict[str, Any]]) -> "Resources":
        """A key this class does not know is REFUSED, not dropped.

        A hand-edited ``job-set.json`` saying ``"memory": "256G"`` or
        ``"walltime": "4h"`` -- both plausible, neither a field name -- would
        otherwise lose the ask with no complaint.

        `task.py`'s allocation reader has always refused an unknown key and
        named the known ones (`_check_keys`).  Two readers of one concept
        disagreeing about strictness is how a person learns that stating a
        thing does not mean it is read.
        """
        d = d or {}
        known = {f.name for f in dataclasses.fields(cls)}
        unknown = [k for k in d if k not in known]
        if unknown:
            raise ValueError(
                f"resources: unknown key(s) "
                + ", ".join(repr(k) for k in sorted(unknown))
                + f" (known keys: {', '.join(sorted(known))})")
        return cls(**{k: v for k, v in d.items() if k in known})


class GpuRequestError(ValueError):
    """A GPU request that cannot be sent: a run on the GPU stating no count,
    or a count stated for a run that does not use the GPU
    (`execution/gpu.md` G5)."""


#: `gpu.md` G5, both halves -- each says where the two facts are stated.
_GPU_NO_COUNT = (
    "this run uses a GPU (`use_gpu` -- its run card, else the template) and "
    "states no GPU count.  Write it on the run card -- \"execution\": "
    "{\"gpu_count\": N} in task.json, the calculation's or this stage's -- "
    "or say it on the prep: --gpus N (docs/execution/gpu.md G5).")
_GPU_NOT_USED = (
    "this run asks for {count} GPU(s) and does not use the GPU (`use_gpu` is "
    "off -- its run card, else the template): its deck runs on the CPU, so "
    "the GPUs would be held and never used.  Set `use_gpu` on the run card "
    "to run it on the GPU, or remove the count -- `gpu_count` on the run "
    "card, or --gpus on the prep (docs/execution/gpu.md G5).")


@dataclass(frozen=True)
class GpuRequest:
    """Whether a run uses a GPU, and how many it asks for -- one answer, and
    a consistent one by construction (`execution/gpu.md` § 1.1, G5).

    The two facts are stated in different places -- ``use_gpu`` on the run
    card or in the template, the count on the run card or as ``--gpus`` --
    so they can disagree, and each disagreement is refused here: a run on
    the GPU with no count, and a count for a run on the CPU, which would
    hold devices its deck never uses.
    """
    uses: bool = False
    count: Optional[int] = None

    def __post_init__(self) -> None:
        if self.uses and self.count is None:
            raise GpuRequestError(_GPU_NO_COUNT)
        if self.count is not None and not self.uses:
            raise GpuRequestError(_GPU_NOT_USED.format(count=self.count))

    @property
    def gres(self) -> Optional[str]:
        """What ``--gres`` carries: ``gpu:<count>`` -- a count, never a card
        (`scheduler.md` R2a) -- or ``None`` for a run on the CPU."""
        return f"gpu:{self.count}" if self.uses else None


def gpu_request(resources) -> GpuRequest:
    """**Does this job use a GPU, and how many does it ask for** -- the one
    door every reader asks (`execution/architecture.md` § 3.2): the header,
    the run script, launch and its queue table, a benchmark's trials and
    their report, the Task setup card.

    Read off the job's ``Resources``: ``use_gpu`` as `resolve` carries it
    from the job's own values -- what its deck renders -- and the count in
    ``gres``.  An unstated ``use_gpu`` is the item's default, no GPU
    (`gpu.md` § 1.1).  Raises :class:`GpuRequestError` when the two
    disagree; prep asks, of the job its stage resolves to, before anything
    is written (`prep._resolve_stage`), so a job it wrote never does.
    """
    from ..scheduler.quantities import parse_gres_flag
    gres = getattr(resources, "gres", None)
    return GpuRequest(
        uses=bool(getattr(resources, "use_gpu", None)),
        count=None if gres in (None, "") else parse_gres_flag(gres))


@dataclass
class WarmFile:
    """One file this job continues **from**, and what makes it safe to take.

    **Not an edge.**  An edge -- *take X from job Y* -- is only answerable
    once you have decided who Y is.  Stages do not chain
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

    **A Job names no other Job as its source or its dependency** -- its
    ``group`` names the stages it shares one queue wait with, never a file.
    What a job declares instead is ``warm``: *what would this job take from a run it
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
    #: here).  ``{}`` for a rung.  `summarize` reads
    #: it to table value coordinates and to carry the winner's value pins
    #: into the benchmark's report (`generator.md` § 4.3a).
    point:      Dict[str, Any]  = field(default_factory=dict)
    #: THE BUNDLE THAT FINISHES THIS JOB, run by its wrapper after the engine
    #: exits cleanly, when the engine alone leaves no result: a SIESTA
    #: force-constant run leaves force constants, and ``mb_vibration.pyz``
    #: derives the modes from them in the same job (`engines/vibration.md`
    #: § 5.5).  The deck's own statement, copied by `prep` from its spec
    #: (`DeckSpec.finish`); ``None`` for a job whose engine writes its result.
    finish:     Optional[str]   = None
    #: WHETHER A RE-RUN OF THIS JOB CONTINUES from what the last one left --
    #: the rung's kind's one section-level fact in its warm-files
    #: (`job-contracts.md` § 4.2a), read by `prep` and baked into the
    #: wrapper, which says a retry of a run that cannot resume repeats it
    #: (`running-a-job.md` § 3.5).  False for a SIESTA force-constant rung
    #: and a PySCF vibration.
    resumes:    bool            = True
    #: WHERE IT WAS ADMITTED: a run prepared for a queue -- the queue its
    #: request was admitted on at prep (``domain``, ``partition``, ``qos``,
    #: bound on the target's record) and where each value came from
    #: (``from``: ``flag``, ``run card`` or ``description``, per field;
    #: `job-system.md` § 6.0) -- what its header renders and launch sends to.
    #: ``None`` with no queue: a trial, a machine with no scheduler.
    placement:  Optional[Dict[str, Any]] = None
    #: THE GROUP IT WAS PREPARED IN: the stages that share one job, named
    #: together at prep, in that order -- this one among them
    #: (`project-layout.md` § 1.6.6); ``None`` for a stage prepared alone.
    group:      Optional[List[str]] = None

    @property
    def relaunch_continues(self) -> bool:
        """Whether this stage, launched again warm, continues from its own
        latest run -- its kind resumes (:attr:`resumes`) and it takes
        something from a run (:attr:`warm`; a stage set ``restart: clean``
        takes nothing).  One that does not runs again from its deck alone,
        as a cold launch does (`job-system.md` § 5.4, *A stage launched
        again*)."""
        return self.resumes and bool(self.warm)

    def to_dict(self) -> Dict[str, Any]:
        # EVERY KEY, EVERY JOB (`job-contracts.md` § 6.1): `point` empty for
        # a job that is no trial, `finish`, `placement` and `group` null
        # where there is none, `resumes` true or false.
        return {
            "name": self.name,
            "script": self.script,
            "resources": self.resources.to_dict(),
            "warm": [w.to_dict() for w in self.warm],
            "traits": dict(self.traits),
            "point": dict(self.point),
            "finish": self.finish,
            "resumes": bool(self.resumes),
            "placement": ({k: (dict(v) if isinstance(v, dict) else v)
                           for k, v in self.placement.items()}
                          if self.placement is not None else None),
            "group": (list(self.group) if self.group is not None else None),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Job":
        # A sweep holds six of these; a refusal that does not say WHICH one
        # is a refusal you have to bisect a file to act on.
        missing = [k for k in JOB_KEYS if k not in d]
        if missing:
            raise WrittenBefore(
                f"job {d.get('name', '?')!r} has no "
                f"{', '.join(repr(k) for k in missing)}")
        try:
            res = Resources.from_dict(d["resources"])
        except ValueError as exc:
            raise ValueError(f"job {d['name']!r}: {exc}") from None
        return cls(
            name=d["name"],
            script=d["script"],
            resources=res,
            warm=[WarmFile.from_dict(w) for w in d["warm"]],
            traits=dict(d["traits"]),
            point=dict(d["point"]),
            finish=(str(d["finish"]) if d["finish"] is not None else None),
            resumes=bool(d["resumes"]),
            placement=(dict(d["placement"]) if d["placement"] is not None
                       else None),
            group=([str(n) for n in d["group"]] if d["group"] is not None
                   else None),
        )


#: The trait a warm file may be conditioned on, spelled ONCE.
#:
#: It is a string in three places that must agree -- each engine's ``_traits``
#: producer and every ``requires_same`` row of a ``warm-files.toml`` -- and a
#: typo on any of them reads as *"the optimizers disagree"*, which withholds
#: the file silently rather than failing.  The rules files are data and say it
#: for themselves; this is the one Python spelling, and it lives beside
#: :func:`warm_carry` because that is the only place the comparison happens.
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


#: Every key a job in `job-set.json` carries, always (`job-contracts.md`
#: § 6.1) -- the one list the writer's dict and the reader's check share.
JOB_KEYS = ("name", "script", "resources", "warm", "traits", "point",
            "finish", "resumes", "placement", "group")


class WrittenBefore(ValueError):
    """A `job-set.json` written before every job key was (2026-10-06): read
    by no fallback -- `molbuilder jobset migrate` rewrites it, every value
    kept (`JobSet.load` names the calculation)."""


@dataclass
class JobSet:
    """A set of related jobs sharing a static package.  ``kind`` is
    ``"sweep"`` (the benchmark's independent points) or ``"ladder"`` (the
    SIESTA stage relaxation).  **Both are sets of independent jobs**; the kind
    says how the directories are named and whether a PERSON should take them in
    order, never whether one job waits for another -- neither does.  ``shared``
    are package files copied into every job directory."""
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
            shared=list(d["shared"]),
            jobs=[Job.from_dict(j) for j in d["jobs"]],
        )

    def write(self, path, *, plan=None) -> Path:
        """Persist to ``job-set.json`` -- the bundle's plan, carried
        host->target (the exchange vocabulary, `job-contracts.md` § 6.2;
        the persisted-artifacts registry is that document's § 6).  Pretty JSON so a
        human can read/diff the plan in the bundle.  ``plan``
        (`jobset.planned.Plan`) receives it instead of the disk."""
        from ..persist import json_text, write_json
        if plan is not None:
            return plan.text(path, json_text(self.to_dict()))
        return write_json(path, self.to_dict())

    @classmethod
    def load(cls, path) -> "JobSet":
        """Read a ``job-set.json`` back into a JobSet (schema checked by name
        and major -- persist.check_schema
        via ``from_dict``).  One written before every job key was is refused
        naming the command that rewrites it, its calculation by its folder --
        found where the layout puts it: this folder, or one or two above
        (a flat stage's bench folder, a hierarchical one's)."""
        from ..persist import read_json
        from ..runfiles import TASK_FILE
        from .commands import command
        path = Path(path)
        try:
            return cls.from_dict(read_json(path))
        except WrittenBefore as exc:
            calc = next((d for d in (path.parent, *list(path.parents)[1:3])
                         if (d / TASK_FILE).is_file()), None)
            way = (f"`{command('migrate', base=calc)}` rewrites it, every "
                   f"value kept and the old file beside the new"
                   if calc is not None else
                   "it lies in no described calculation")
            raise WrittenBefore(
                f"{path}: {exc} -- written before every job key was "
                f"(2026-10-06); {way}.") from None

    # ----- structural validation ------------------------------------- #

    def validate(self) -> List[str]:
        """Return human-readable structural errors (empty == OK).  Checks
        the invariants the engines can't recover from -- exactly the same
        discipline the description's own stage validation applies.

          * non-empty; ``kind`` known;
          * unique job names (the dir + ``-J`` collide otherwise);
          * every ``warm.requires_same`` names a trait this job HAS.

        No acyclic/ordered check: with no job naming another, there is no
        graph to be cyclic.
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
