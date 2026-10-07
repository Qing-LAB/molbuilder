"""The scheduler subsystem — what a machine offers, and what a job may ask.

The contract is ``docs/execution/scheduler.md``.  In one line: whether a
request fits a queue (**admission**), which queue it is placed in
(**placement**), and the directives that placement produces (**emission**)
belong together.

  * :mod:`molbuilder.scheduler.record` — ``Environment``,
    ``Domain``, ``Topology``, ``Site``; reading and writing records; the
    scopes a record may live in; named targets.  **It also RUNS the
    detection commands** — ``_run``, ``detect_scheduler``,
    ``detect_topology``, ``detect_site``, ``probe_queues``.
  * :mod:`molbuilder.scheduler.probe` — **pure text parsing**: ``sinfo`` /
    ``scontrol`` / ``qos`` output → ``Partition`` / ``Domain`` data, testable
    on captured text.  It runs no subprocess at all.
  * :mod:`molbuilder.scheduler.admit` — can this domain take this request, and
    if not, why not.

**Nothing here is shipped beside a job.**  The record is JSON and reads
anywhere; what travels with a run is `runwrap.MONITOR_COMPANIONS`
(`record.py`'s header).

The re-exports below are the subsystem's public surface — what the rest of
molbuilder is meant to use.  Anything not listed is internal to its module.
"""
from __future__ import annotations

# The two quantities a job asks for -- a wall and an amount of memory --
# and every dialect each is written in.  One object, one module.
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
    named_environment_path, calculation_record,
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
    "named_environment_path", "calculation_record",
    "record_scopes",
    "read_environment", "write_environment", "machine_for",
    "UnknownTarget", "AmbiguousTarget",
    "known_machines", "choice_required",
    "Request", "admits", "parse_mem_gb",
    "domain_ceiling_s", "domain_serves_gpu",
    # M-1's typed `--set` door.
    "topology_field_types",
    # The quantities a job asks for, and how each is written.
    "parse_walltime", "parse_duration", "parse_memory",
    "slurm_time", "slurm_mem", "canonical_time", "canonical_mem",
    "human_wall",
]
