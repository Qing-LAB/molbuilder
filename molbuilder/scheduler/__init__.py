"""The scheduler subsystem — what a machine offers, and what a job may ask.

The contract is ``docs/execution/scheduler.md``.  In one line: whether a
request fits a queue (**admission**), which queue it is placed in
(**placement**), and the directives that placement produces (**emission**)
belong together, because when they did not, a ``#SBATCH`` header could name a
15-minute queue while the ``sbatch`` command line asked for 38 minutes.

Phase 1 (2026-08-23) moved the two pieces that already worked:

  * :mod:`molbuilder.scheduler.record` — ``Machine``/``Environment``,
    ``Domain``, ``Topology``, ``Site``; reading and writing records; the
    scopes a record may live in; named targets.  Was ``molbuilder/environment
    .py``.
  * :mod:`molbuilder.scheduler.probe` — detection from ``sinfo`` /
    ``scontrol`` / ``lscpu``.  Was ``molbuilder/scheduler_probe.py``.

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

from .record import (  # noqa: F401
    SCHEMA, FILENAME,
    Topology, Site, Domain, Environment,
    detect_scheduler, detect_topology, detect_site,
    resolve_environment,
    machine_scope_path, environments_dir, named_environments,
    record_scopes,
    read_environment, write_environment, machine_for,
    UnknownTarget, AmbiguousTarget,
    known_machines, choice_required,
    domain_admits, domain_ceiling_s, domain_serves_gpu,
)

__all__ = [
    "SCHEMA", "FILENAME",
    "Topology", "Site", "Domain", "Environment",
    "detect_scheduler", "detect_topology", "detect_site",
    "resolve_environment",
    "machine_scope_path", "environments_dir", "named_environments",
    "record_scopes",
    "read_environment", "write_environment", "machine_for",
    "UnknownTarget", "AmbiguousTarget",
    "known_machines", "choice_required",
    "domain_admits", "domain_ceiling_s", "domain_serves_gpu",
]
