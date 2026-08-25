"""The scheduler subsystem — what a machine offers, and what a job may ask.

The contract is ``docs/execution/scheduler.md``.  In one line: whether a
request fits a queue (**admission**), which queue it is placed in
(**placement**), and the directives that placement produces (**emission**)
belong together, because when they did not, a ``#SBATCH`` header could name a
15-minute queue while the ``sbatch`` command line asked for 38 minutes.

Phase 1 (2026-08-23) moved the two pieces that already worked:

  * :mod:`molbuilder.scheduler.record` — ``Machine``/``Environment``,
    ``Domain``, ``Topology``, ``Site``; reading and writing records; the
    scopes a record may live in; named targets.  **It also RUNS the
    detection commands** — ``_run``, ``detect_scheduler``,
    ``detect_topology``, ``detect_site`` — which is the one thing this list
    used to attribute to `probe`.  Was ``molbuilder/environment.py``.
  * :mod:`molbuilder.scheduler.probe` — **pure text parsing**: ``sinfo`` /
    ``scontrol`` / ``qos`` output → ``Partition`` / ``Domain`` data, testable
    on captured text.  It runs no subprocess at all; its own docstring has
    always said so, while this list said it did *"detection from sinfo /
    scontrol / lscpu"* until 2026-08-24 — so a reader asking *"where do we
    run sinfo?"* was sent to the wrong file by the package's own map.  Was
    ``molbuilder/scheduler_probe.py``.

Phase 2 (2026-08-23) split the CHECK out of the record:

  * :mod:`molbuilder.scheduler.admit` — can this domain take this request, and
    if not, why not.  It shared a module with the record until then, which is
    how the comparison drifted away from the fields it compares.

They move so that admission, placement and emission have somewhere to be:
leaving the data model in a general-purpose ``environment.py`` is what let the
CHECK drift away from the record it checks (contract § 5).

**Both modules are stdlib-only, and this file must stay that way.**  A record
travels with a bundle and is read on the target inside a backend environment
that has no molbuilder installed, so importing anything heavier here would
break reading a record on the machine that runs the job.

The re-exports below are the subsystem's public surface — what the rest of
molbuilder is meant to use.  Anything not listed is internal to its module.
"""
from __future__ import annotations

# The two quantities a job asks for -- a wall and an amount of memory --
# and every dialect each is written in.  One object, one module: they were
# split across this package and `jobset/` until 2026-08-24, which is how a
# human-dialect value came to be read by the record reader.
from .quantities import (  # noqa: F401
    parse_walltime, parse_duration, parse_memory,
    slurm_time, slurm_mem, canonical_time, canonical_mem, human_wall,
)
from .record import (  # noqa: F401
    SCHEMA, FILENAME,
    Topology, Site, Domain, Device, Environment,
    detect_scheduler, detect_topology, detect_site,
    resolve_environment,
    machine_scope_path, environments_dir, named_environments,
    record_scopes,
    read_environment, write_environment, machine_for,
    UnknownTarget, AmbiguousTarget,
    known_machines, choice_required,
    topology_field_types,
)
from .admit import (  # noqa: F401
    Request, admits, parse_mem_gb, domain_ceiling_s, domain_serves_gpu,
)

__all__ = [
    "SCHEMA", "FILENAME",
    "Topology", "Site", "Domain", "Device", "Environment",
    "detect_scheduler", "detect_topology", "detect_site",
    "resolve_environment",
    "machine_scope_path", "environments_dir", "named_environments",
    "record_scopes",
    "read_environment", "write_environment", "machine_for",
    "UnknownTarget", "AmbiguousTarget",
    "known_machines", "choice_required",
    "Request", "admits", "parse_mem_gb",
    "domain_ceiling_s", "domain_serves_gpu",
    # M-1's typed `--set` door.  It was public and USED but absent from
    # the old module's __all__ -- packaging surfaced the gap, because a
    # package re-exports a list where a module exported a namespace.
    "topology_field_types",
    # The quantities a job asks for, and how each is written.
    "parse_walltime", "parse_duration", "parse_memory",
    "slurm_time", "slurm_mem", "canonical_time", "canonical_mem",
    "human_wall",
]
