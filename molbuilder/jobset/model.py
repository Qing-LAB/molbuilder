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
  * ``Carry``          — a runtime-produced file (one job's output) fed
    to a dependent job; symlinked after the producer runs.
Nothing reaches across jobs outside these two.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

# Matches the molbuilder/<name>@<major> convention used by
# bench/environment.py and bench/result.py (same major-version check).
SCHEMA = "molbuilder/job-set@1"

_DEP_KINDS = ("afterok", "afterany")
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
    ``scheduler.routing`` name (slurm-integration.md § 4.3) the submit engine
    resolves to ``-p``/``-q``; ``gres`` is a raw SLURM gres string (e.g.
    ``"gpu:a100:1"``) or None.
    """
    domain:        Optional[str]   = None
    time:          Optional[str]   = None    # SLURM -t (D-HH:MM:SS); == scheduler defaults.time
    exclusive:     Optional[bool]  = None
    mem:           Optional[str]   = None    # SLURM --mem (e.g. "120G", "0")
    gres:          Optional[str]   = None    # SLURM --gres (e.g. "gpu:a100:1")
    mpi_np:        Optional[int]   = None    # SLURM -n (MPI ranks)
    cpus_per_task: Optional[int]   = None    # SLURM -c (OMP cores/rank); == SiestaConfig.omp_threads

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: Optional[Dict[str, Any]]) -> "Resources":
        d = d or {}
        known = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class Carry:
    """One runtime-produced file carried forward: ``pattern`` (a concrete
    filename, e.g. ``"job.XV"``) taken from job ``from_job``'s directory and
    symlinked into the consuming job's directory.  Concrete, not a glob: the
    symlink is laid at materialize time (before the producer runs) and
    resolves once the file appears (docs/execution/job-system.md D1)."""
    pattern:  str
    from_job: str

    def to_dict(self) -> Dict[str, Any]:
        return {"pattern": self.pattern, "from_job": self.from_job}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Carry":
        return cls(pattern=d["pattern"], from_job=d["from_job"])


@dataclass
class Job:
    """One unit of work.  ``name`` is unique within the set and becomes the
    job directory (``point-<name>/``) and the SLURM ``-J`` name.  ``script``
    is the per-job input filename (e.g. the rendered ``.fdf``).  ``depends_on``
    names the producer job this one waits for (None = independent);
    ``dep_kind`` is the SLURM dependency kind; ``carry`` lists the
    restart files pulled from the producer (§ 5, § 8 D1)."""
    name:       str
    script:     str
    resources:  Resources       = field(default_factory=Resources)
    depends_on: Optional[str]   = None
    dep_kind:   str             = "afterok"
    carry:      List[Carry]     = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "script": self.script,
            "resources": self.resources.to_dict(),
            "depends_on": self.depends_on,
            "dep_kind": self.dep_kind,
            "carry": [c.to_dict() for c in self.carry],
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Job":
        return cls(
            name=d["name"],
            script=d["script"],
            resources=Resources.from_dict(d.get("resources")),
            depends_on=d.get("depends_on"),
            dep_kind=d.get("dep_kind", "afterok"),
            carry=[Carry.from_dict(c) for c in (d.get("carry") or [])],
        )


@dataclass
class JobSet:
    """A set of related jobs sharing a static package.  ``kind`` is
    ``"sweep"`` (independent jobs, e.g. the benchmark) or ``"ladder"``
    (a dependency chain, e.g. the SIESTA stage relaxation).  ``shared``
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
        # bench/environment + bench/result use.
        from ..persist import check_schema_major
        check_schema_major(str(d.get("schema") or ""), SCHEMA, label="job-set")
        return cls(
            name=d["name"],
            engine=d["engine"],
            kind=d["kind"],
            shared=list(d.get("shared") or []),
            jobs=[Job.from_dict(j) for j in (d.get("jobs") or [])],
        )

    def write(self, path) -> Path:
        """Persist to ``job-set.json`` -- the bundle's plan, carried
        host->target (data-vocabulary.md § 1).  Pretty JSON so a human can
        read/diff the plan in the bundle."""
        from ..persist import write_json
        return write_json(path, self.to_dict())

    @classmethod
    def load(cls, path) -> "JobSet":
        """Read a ``job-set.json`` back into a JobSet (major-version checked
        via ``from_dict``)."""
        from ..persist import read_json
        return cls.from_dict(read_json(path))

    # ----- structural validation ------------------------------------- #

    def validate(self) -> List[str]:
        """Return human-readable structural errors (empty == OK).  Checks
        the invariants the engines can't recover from -- exactly the same
        discipline as ``validate_stages`` / ``validate_siesta_stages``:

          * non-empty; ``kind`` known;
          * unique job names (the dir + ``-J`` collide otherwise);
          * ``dep_kind`` known;
          * ``depends_on`` references a PRIOR job (acyclic, ordered);
          * every ``carry.from_job`` references a prior job.
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
            if j.dep_kind not in _DEP_KINDS:
                errors.append(
                    f"{prefix}.dep_kind = {j.dep_kind!r}: must be one of "
                    f"{_DEP_KINDS}")
            if j.depends_on is not None and j.depends_on not in seen:
                errors.append(
                    f"{prefix}.depends_on = {j.depends_on!r}: must reference "
                    f"a PRIOR job (forward / unknown / self reference)")
            for c in j.carry:
                if c.from_job not in seen:
                    errors.append(
                        f"{prefix}.carry from {c.from_job!r}: must reference "
                        f"a prior job")
            seen.add(j.name)
        return errors


__all__ = ["Resources", "Carry", "Job", "JobSet", "SCHEMA"]
